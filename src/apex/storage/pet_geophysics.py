"""Convert PET's geophysical simulator sections into ``[geophysics]``.

PET's AVO simulator (SimulatorWrap ``flow_rock.flow_avo`` @ ``d29dc57``) reads two blocks
of ``[simulator]`` as ``[key, value]`` pairs: ``pem`` (the petro-elastic model, the
vintages, the overburden pressure) and ``avo`` (angles, wavelet, shale, time axis,
reflectivity, thinning, region of interest). Every key goes through one mapping table,
:data:`PEM_KEYS` and :data:`AVO_KEYS`; a key outside it fails loudly. (A TOML table is
accepted too, though SimulatorWrap itself iterates the pairs.)

PET's AVO datum at a vintage is one vector: the cube ``(NY, NX, n_time, n_angle)`` (J
before I) under a mask of active columns, ``step_x``/``step_y`` thinning and a region of
interest, in C order (``get_avo_result``). The importer rebuilds that mask from the grid
file, the deck and the ``avo`` keys, so each value lands on its column, sample and angle;
the columns PET keeps, in its order (I fastest, then J), are exactly APEX's traces with the
region written as a file (``region = { file }``). The data become APEX's per-angle trace
fields ``<datatype>_<angle>`` (``form = "traces"``). A baseline makes them differences
(``four_d = "differences"``), as PET's are (SimulatorWrap cannot run without one:
``pet_parity.md`` G10), from it (``fixed``) or from the vintage before (``rolling``).

Refused, with the reason (rather than imported into something else):

- AVO data PET compresses (``compress_data``), for the same phase;
- several angles, unless the caller vouches for them: SimulatorWrap gives every angle the
  last angle's traces (``pet_parity.md``, G7), so its own multi-angle data are not data of
  the angles they name (noise added to a synthetic truth makes them differ, still wrongly);
- a region of interest that follows each member's signal (``roi_use_signal``), compaction
  (``PORV_RC``), and ``no_flow`` inputs (``param_file``, ``grid``).

PET's gravity and seafloor displacement (``flow_grav``, ``flow_seafloor_disp``) read the
``grav`` and ``sea_disp`` blocks (:data:`GRAV_KEYS`, :data:`DISP_KEYS`), and ``pem``'s
phases and ``press_conv``. Their data at a vintage are one value per station of
``measurement_locations``, which APEX's regular stations reproduce over ``grid_file``'s
active cells, so each value is APEX's station of the same index: the data become
``kind = "gravity"`` and ``"seafloor_displacement"`` with the reference's settings (its
gravitational constant, the baseline's pore volume, Geertsma truncated as it truncates,
its free surface at sea level). Refused: a ``rolling`` baseline (a rolling difference of
these is the difference of two fixed ones), compressed data, and the
displacement's ``fast`` Geertsma mode (``pet_parity.md`` G15).
"""

import datetime as dt
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.config import avo_datatype
from apex.storage.pet_format import PetField, PetScalar, field_variance

#: PET's ``pem`` keys (``flow_rock._getpeminfo``): what each becomes.
PEM_KEYS: dict[str, str] = {
    "model": "[geophysics] pem: standardrp -> standard, softsandrp -> soft_sand",
    "vintage": "[[geophysics.data]] vintages (days after startdate, as ISO dates)",
    "overburden": "[geophysics] overburden_pressure = { file, key = 'obvalues', unit }",
    "press_conv": "the overburden file's unit (the deck's pressure unit over press_conv)",
    "baseline": "four_d = 'differences', baseline (days after startdate, as an ISO date)",
    "phases": "not read (APEX reads the phases from the run's INTEHEAD)",
    "dimension": "the grid's size, checked against grid_file (without [simulator] dimension)",
    "ntg": "not read (neither PEM uses NTG)",
    "depth": "not read (PET's AVO does not use it)",
    "actnum": "not read (grid_file's ACTNUM fixes the traces)",
    "percentile": "not read (SimulatorWrap's aquifer filter hard-codes its 90th percentile)",
    "clusters": "not read (PET's AVO does not use it)",
    "compaction": "refused: PORV_RC compaction is not ported",
    "param_file": "refused: PET's no_flow mode",
    "grid": "refused: PET's no_flow mode",
}

#: PET's ``avo`` keys (``flow_avo._get_avo_info``'s ``config_para_list``).
AVO_KEYS: dict[str, str] = {
    "angle": "angles",
    "frequency": "wavelet = { frequency, length }",
    "wave_len": "wavelet = { frequency, length }",
    "vp_shale": "overburden = { vp, vs, density }",
    "vs_shale": "overburden = { vp, vs, density }",
    "den_shale": "overburden = { vp, vs, density }",
    "t_min": "times = { start, end, step }",
    "t_max": "times = { start, end, step }",
    "t_sampling": "times = { start, end, step }",
    "pp_func": "reflectivity",
    "step_x": "step_y (PET's step_x thins J)",
    "step_y": "step_x (PET's step_y thins I)",
    "roi_mode": "region = { file }: PET's region, rebuilt",
    "use_ellipse": "region = { file }: PET's region, rebuilt",
    "roi_margin_m": "region = { file }: PET's region, rebuilt",
    "x_range": "region = { file }: PET's region, rebuilt",
    "y_range": "region = { file }: PET's region, rebuilt",
    "roi_mask_file": "region = { file }: PET's region, rebuilt",
    "roi_use_signal": "refused when it applies: a region that follows each member's signal",
    "dz": "not read (PET's traces use the run's DZ and ZCORN, as APEX's do)",
    "tops": "not read (as dz)",
    "baseline_mode": "baseline_mode: fixed, or rolling (each vintage minus the one before)",
    "roi_signal_percentile": "not read (with roi_use_signal only)",
}

#: SimulatorWrap's petro-elastic classes (``model = "<module> <class>"``).
_MODELS = {
    ("standardrp", "elasticproperties"): "standard",
    ("softsandrp", "elasticproperties"): "soft_sand",
}

#: PyLops' ``pp_func`` names: APEX's reflectivity, and whether the reference can run it
#: (G1: only ``approx_zoeppritz_pp`` takes the arrays SimulatorWrap passes).
_REFLECTIVITY: dict[str, tuple[dict[str, Any], bool]] = {
    "approx_zoeppritz_pp": ({"name": "approx_zoeppritz_pp"}, True),
    "zoeppritz_pp": ({"name": "zoeppritz"}, False),
    "akirichards": ({"name": "aki_richards", "options": {"form": "linear"}}, False),
    "fatti": ({"name": "fatti"}, False),
}

#: Pa per unit of each deck unit system's pressure.
_DECK_PRESSURE = {"METRIC": 1e5, "FIELD": 6894.757293168361, "LAB": 101325.0, "PVT-M": 101325.0}

#: The overburden file units APEX takes, in Pa.
_FILE_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "psi": 6894.757293168361}

#: SimulatorWrap's region modes (``_get_avo_info``); ``auto`` is its ellipse.
_ROI_MODES = ("none", "off", "all", "ellipse", "rectangle", "file", "auto")

#: Where the importer writes PET's region of interest.
REGION_FILE = "avo_region.npy"

#: The keys of SimulatorWrap's stations (``measurement_locations``), common to both kinds.
_STATION_KEYS: dict[str, str] = {
    "baseline": "options.baseline (days after startdate, as an ISO date)",
    "vintage": "vintages (days after startdate, as ISO dates)",
    "padding": "stations.padding (SimulatorWrap's default 1500)",
    "meas_spacing": "stations.spacing (SimulatorWrap's default 1500)",
    "water_depth": "stations.depth (SimulatorWrap's default 300)",
    "seabed": "stations.seabed_file (its x, y and depth, written as an .npy)",
    "baseline_mode": "fixed (rolling is refused: the difference of two fixed ones)",
}

#: PET's ``grav`` keys (``flow_grav._get_grav_info``'s ``config_para_list``).
GRAV_KEYS: dict[str, str] = {
    **_STATION_KEYS,
    "well_coord": "stations.refine = [x, y, 5000] (SimulatorWrap's radius)",
    "method": "not read (PET's gravity does not use it)",
    "model": "not read (PET's gravity does not use it)",
    "poisson": "not read (PET's gravity does not use it)",
    "compressibility": "not read (PET's gravity does not use it)",
    "z_base": "not read (PET's gravity does not use it)",
}

#: PET's ``sea_disp`` keys (``flow_seafloor_disp._get_disp_info``'s ``config_para_list``).
DISP_KEYS: dict[str, str] = {
    **_STATION_KEYS,
    "method": "volume_change: 'pressure' with volume = 'pore'; anything else 'pore_volume'",
    "model": "kernel geertsma, whatever the model (SimulatorWrap computes Geertsma: G16)",
    "poisson": "the kernel's poisson (above 0.5 taken as 0.5, as SimulatorWrap does)",
    "compressibility": "compressibility in 1/Pa (PET's is per the pressure unit it passes)",
    "z_base": "not read (SimulatorWrap's rigid basement is never reached: G16)",
    "receiver_trunc_factor": "geertsma truncation_factor = min(factor, 3) (G14)",
    "geertsma_mode": "integral; fast is refused (G15)",
}

#: Each geophysical section: APEX's kind, its keys, and the substring SimulatorWrap's
#: ``extract_data`` finds its data types by.
_SECTIONS: dict[str, tuple[str, dict[str, str], str]] = {
    "avo": ("avo", AVO_KEYS, "avo"),
    "grav": ("gravity", GRAV_KEYS, "grav"),
    "sea_disp": ("seafloor_displacement", DISP_KEYS, "sea_disp"),
}

#: The reference's gravitational constant (``calc_grav``'s 6.67e-3 for µGal).
_REFERENCE_G = 6.67e-11
#: ``measurement_locations``'s radius of finer stations around ``well_coord`` (m).
_REFINE_RADIUS = 5000.0


@dataclass
class GeophysicsImport:
    """What :func:`convert_geophysics` made of PET's sections."""

    section: dict[str, Any] | None = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    datatypes: set[str] = field(default_factory=set)
    ignored: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check_sections(sim: Mapping[str, Any], source: Path) -> bool:
    """Return whether ``[simulator]`` has PET's geophysical sections.

    Called before the data are read, so that a case with sections APEX does not import
    is refused for them, not for its data.

    Raises
    ------
    ConfigError
        For a ``pem`` without ``avo``, ``grav`` or ``sea_disp``, or one of those without
        ``pem``.
    """
    present = [name for name in _SECTIONS if name in sim]
    if "pem" not in sim and not present:
        return False
    if not present:
        raise ConfigError(
            f"{source}: a pem section without avo, grav or sea_disp is PET's flow_rock data "
            "(bulk impedance, ...), which APEX does not port"
        )
    if "pem" not in sim:
        # SimulatorWrap builds its model from pem, and gravity and displacement read its
        # phases and press_conv.
        raise ConfigError(f"{source}: PET's {', '.join(present)} section(s) need a pem section")
    return True


def start_date(sim: Mapping[str, Any], source: Path) -> dt.date | None:
    """Return ``[simulator] startdate`` (SimulatorWrap's ``day/month/year``), if given.

    Raises
    ------
    ConfigError
        If it is not ``day/month/year``.
    """
    if "startdate" not in sim:
        return None
    try:
        day, month, year = (int(p) for p in str(sim["startdate"]).split("/"))
        return dt.date(year, month, day)
    except ValueError as error:
        raise ConfigError(
            f"{source}: startdate {sim['startdate']!r} is not day/month/year"
        ) from error


def convert_geophysics(
    sim: Mapping[str, Any],
    entries: Sequence[PetScalar | PetField],
    *,
    var_csv: Path,
    base: Path,
    output_dir: Path,
    deck_text: str,
    grid_file: Path | None,
    compressed_types: Sequence[str],
    angles_checked: bool,
    source: Path,
) -> GeophysicsImport:
    """Map PET's ``pem`` and ``avo`` blocks and AVO data to ``[geophysics]``.

    Parameters
    ----------
    sim
        PET's ``[simulator]`` table.
    entries
        PET's observation cells (:func:`~apex.storage.pet_format.read_pet_data`), their
        times ISO dates where PET's were days after ``startdate``.
    var_csv, base
        PET's variance table, and the folder its relative files are resolved against.
    output_dir
        The APEX case: the grid, overburden, region and field files are written there.
    deck_text
        The deck template: its unit system, and the completions PET's region surrounds.
    grid_file
        An EGRID of the deck (``apex pet-import --grid-file``): its active columns are
        PET's, and its pillars place the region.
    compressed_types
        The data types of PET's ``compress_data``.
    angles_checked
        Whether multi-angle observations are known to hold each angle's own traces.
    source
        The PET configuration, for messages.

    Raises
    ------
    ConfigError
        For a key outside the mapping table, a refused feature (module docstring), or
        observations that do not fit PET's mask and time axis.
    """
    out = GeophysicsImport()
    if not check_sections(sim, source):
        return out
    pem = _pairs(sim["pem"], "pem", source)
    tables = {name: _pairs(sim[name], name, source) for name in _SECTIONS if name in sim}
    for name, table, known in (("pem", pem, PEM_KEYS),
                               *((n, t, _SECTIONS[n][1]) for n, t in tables.items())):  # fmt: skip
        unknown = sorted(set(table) - set(known))
        if unknown:
            raise ConfigError(
                f"{source}: [simulator] {name} has keys SimulatorWrap does not read: "
                f"{', '.join(unknown)} (known: {', '.join(known)})"
            )
    names = list(dict.fromkeys(e.datatype for e in entries))
    claimed = {n: {d for d in names if _SECTIONS[n][2] in d} for n in tables}
    for name in tables:
        others = set().union(*(c for n, c in claimed.items() if n != name))
        both = sorted(claimed[name] & others)
        if both:
            raise ConfigError(
                f"{source}: the data types {both} name two geophysical sections "
                "(SimulatorWrap's extract_data fills them from each)"
            )
        if not claimed[name]:
            raise ConfigError(
                f"{source}: a {name} section but no {name} data (data types containing "
                f"{_SECTIONS[name][2]!r}, as SimulatorWrap reads them)"
            )
        compressed = sorted(claimed[name] & set(compressed_types))
        if compressed:
            raise ConfigError(
                f"{source}: PET compresses the {name} data {compressed} with wavelets; the "
                "importer does not convert that yet: leave them out of "
                "compress_data and give them absolute variances"
            )
    _refuse(pem, source)
    start = start_date(sim, source)
    if start is None:
        raise ConfigError(f"{source}: geophysical vintages are days after [simulator] startdate")
    if grid_file is None:
        raise ConfigError(
            f"{source}: geophysical data need an EGRID of the deck, whose active cells fix "
            "PET's traces and stations: give --grid-file (e.g. the truth's or a base-case "
            "run's EGRID)"
        )
    if not grid_file.is_file():
        raise ConfigError(f"grid file not found: {grid_file}")
    shutil.copyfile(grid_file, output_dir / grid_file.name)

    section: dict[str, Any] = {}
    data: list[dict[str, Any]] = []
    if "avo" in tables:
        avo_section, avo_data = _avo(
            sim, pem, tables["avo"], entries, claimed["avo"], start, output_dir / grid_file.name,
            var_csv=var_csv, base=base, output_dir=output_dir, deck_text=deck_text,
            angles_checked=angles_checked, out=out, source=source,
        )  # fmt: skip
        section.update(avo_section)
        data += avo_data
    else:
        unused = sorted(
            k
            for k in pem
            if not PEM_KEYS[k].startswith("not read")
            and not (k == "press_conv" and _pressure_based(tables.get("sea_disp")))
        )
        if unused:
            out.notes.append(
                f"PET's pem {', '.join(unused)} serve its AVO only: without AVO data they are "
                "not used"
            )
    for name in ("grav", "sea_disp"):
        if name in tables:
            data += _station_data(
                name, tables[name], pem, entries, claimed[name], start,
                var_csv=var_csv, base=base, output_dir=output_dir, deck_text=deck_text,
                grid_path=output_dir / grid_file.name,
                length=_grid_length(output_dir / grid_file.name, deck_text, out), out=out,
                source=source,
            )  # fmt: skip
    out.datatypes = set().union(*claimed.values())
    out.section = {**section, "grid_file": grid_file.name, "data": data}
    if "max_field_values" in section:  # tomli_w writes it in the order given
        out.section["max_field_values"] = out.section.pop("max_field_values")
    out.ignored = sorted(
        f"simulator.{name}.{key}"
        for name, table, known in (("pem", pem, PEM_KEYS),
                                   *((n, t, _SECTIONS[n][1]) for n, t in tables.items()))
        for key in table
        if known[key].startswith("not read")
    )  # fmt: skip
    if "phases" in pem:
        out.notes.append(
            f"PET's pem phases {pem['phases']!r} choose which saturations and masses it "
            "derives; APEX reads the run's phases from its restart files (INTEHEAD)"
        )
    return out


def _avo(
    sim: Mapping[str, Any],
    pem: Mapping[str, Any],
    avo: Mapping[str, Any],
    entries: Sequence[PetScalar | PetField],
    datatypes: set[str],
    start: dt.date,
    grid_path: Path,
    *,
    var_csv: Path,
    base: Path,
    output_dir: Path,
    deck_text: str,
    angles_checked: bool,
    out: GeophysicsImport,
    source: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the AVO part of ``[geophysics]`` (pem, overburden) and its data."""
    options, (step_x, step_y) = _avo_options(avo, out, source)
    vintages = _days(pem, "pem", "vintage", start, source)
    zero_vintage = False
    if "baseline" not in pem and "baseline_mode" in avo:
        out.notes.append(
            f"avo baseline_mode {avo['baseline_mode']!r} without a pem baseline: SimulatorWrap "
            "ignores it (its data are then each vintage's)"
        )
    if "baseline" in pem:
        (baseline,) = _days(pem, "pem", "baseline", start, source)
        mode = str(avo.get("baseline_mode", "fixed")).lower()
        if mode not in ("fixed", "rolling"):
            raise ConfigError(f"{source}: avo baseline_mode {mode!r} is neither fixed nor rolling")
        options.update(four_d="differences", baseline=baseline, baseline_mode=mode)
        early = [v for v in vintages if v < baseline]
        if early:
            raise ConfigError(f"{source}: pem vintages {early} are before the baseline {baseline}")
        listed = [v for v in vintages if v != baseline]
        if mode == "rolling" and any(b <= a for a, b in pairwise(listed)):
            raise ConfigError(
                f"{source}: pem vintages {listed}: SimulatorWrap's rolling differences take "
                "the vintages in the listed order; list them in increasing order"
            )
        zero_vintage = baseline in vintages
        if zero_vintage:
            vintages.remove(baseline)
            out.notes.append(
                f"PET's vintage at the baseline {baseline} is a difference from itself, zero "
                "by construction: it is left out, with its data"
            )
    angles = [float(a) for a in options["angles"]]
    if len(angles) > 1 and not angles_checked:
        raise ConfigError(
            f"{source}: SimulatorWrap gives every angle the last angle's traces "
            "(pet_parity.md G7), so its multi-angle data are not data of the angles they "
            "name. If these observations hold each angle's own traces (field data, or data "
            "made without SimulatorWrap), import them with --avo-angles-checked"
        )
    active, centres = _read_grid(grid_path)
    nx, ny = active.shape
    _check_dimension(sim, pem, (nx, ny), source)

    region = _region(avo, active, centres, deck_text, base, out, source)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    kept = active & (i % step_x == 0) & (j % step_y == 0)
    if region is not None:
        kept &= region
        if np.any(active & ~region):  # else the region keeps every active column anyway
            np.save(output_dir / REGION_FILE, region)
            options["region"] = {"file": REGION_FILE}
            out.notes.append(f"PET's region of interest is written as {REGION_FILE}")
    n_trace = int(kept.sum())
    if n_trace == 0:
        raise ConfigError(f"{source}: PET's mask keeps no trace")
    times = np.arange(options["times"]["start"], options["times"]["end"], options["times"]["step"])
    shape = (n_trace, times.size, len(angles))

    data: list[dict[str, Any]] = []
    observed = 0  # the observed fields' count, for max_field_values
    for datatype in sorted(datatypes):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", datatype):
            raise ConfigError(f"{source}: AVO data type {datatype!r} is not a valid APEX name")
        dated: list[str] = []
        for entry in (e for e in entries if e.datatype == datatype):
            where = f"{source}: {datatype}@{entry.time}"
            if not isinstance(entry, PetField):
                raise ConfigError(f"{where}: AVO data must be an .npz vector")
            if zero_vintage and entry.time == options.get("baseline"):
                continue  # PET's zero difference at the baseline (noted above)
            if entry.time not in vintages:
                raise ConfigError(
                    f"{where}: PET fills AVO data only at the pem vintages "
                    f"({', '.join(sorted(vintages))})"
                )
            if entry.values.size != math.prod(shape):
                raise ConfigError(
                    f"{where}: {entry.values.size} values, but PET's mask keeps {n_trace} "
                    f"traces of {times.size} samples at {len(angles)} angle(s) = "
                    f"{math.prod(shape)} (grid_file's active columns, step_x/step_y and the "
                    "region of interest)"
                )
            kind, variance = field_variance(entry, var_csv)
            if kind != "abs":
                raise ConfigError(
                    f"{where}: AVO data need an absolute variance (['abs', v], one per "
                    "value); APEX has one variance per field"
                )
            # PET's C order over (J, I, time, angle) keeps the columns J-major, I fastest:
            # APEX's trace order (Eclipse order of the kept columns).
            cube = entry.values.reshape(shape)
            if len(angles) > 1 and all(
                np.array_equal(cube[..., 0], cube[..., a]) for a in range(1, len(angles))
            ):
                raise ConfigError(f"{where}: every angle holds the same traces (G7)")
            for a, angle in enumerate(angles):
                name = avo_datatype(datatype, angle)
                target = output_dir / "fields" / f"{name}_{entry.time}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez(target, values=cube[:, :, a].ravel())  # i = trace * n_time + sample
                out.fields.append(
                    {
                        "time": entry.time,
                        "datatype": name,
                        "file": str(target.relative_to(output_dir)),
                        "variance": float(variance),
                    }
                )
            dated.append(entry.time)
        if len(set(dated)) != len(dated):
            raise ConfigError(f"{source}: {datatype} has two cells at one vintage")
        observed += len(dated)
        unused = sorted(set(vintages) - set(dated))
        if options.get("baseline_mode") == "rolling":
            # Every PET vintage stays: SimulatorWrap's rolling reference moves on at each
            # one, data or not, so each datum is its vintage minus the one listed before.
            kept_vintages = list(vintages)
            if unused:
                out.notes.append(
                    f"{datatype}: PET's vintages {unused} have no data but are kept: the "
                    "rolling differences take each vintage minus the one before it"
                )
        else:
            kept_vintages = sorted(dated)
            if unused:
                out.notes.append(
                    f"{datatype}: PET's vintages {unused} have no data and are left out"
                )
        data.append(
            {"kind": "avo", "datatype": datatype, "vintages": kept_vintages, "options": options}
        )

    section: dict[str, Any] = {
        "pem": _pem_section(pem, out, source),
        "overburden_pressure": _overburden(pem, deck_text, base, output_dir, out, source),
    }
    # The observed fields, as the composition counts them (a data-less rolling vintage is
    # computed in passing, never forecast as data).
    n_values = observed * math.prod(shape)
    if n_values > 1_000_000:
        # Above the default, composition refuses untransformed fields; the case is still
        # PET's, so the limit is raised to fit it, and the cost is stated.
        section["max_field_values"] = n_values
        out.notes.append(
            f"the AVO traces are {n_values:,} values per member, so max_field_values is set "
            "to that; they fill checkpoints and memory (8 bytes per value and member): a "
            "wavelet transform per AVO data type would shrink them"
        )
    return section, data


def _pressure_based(table: Mapping[str, Any] | None) -> bool:
    """Return whether a ``sea_disp`` table takes the pressures (and so ``press_conv``)."""
    return table is not None and str(table.get("method", "")).lower() == "pressure"


def _station_data(
    name: str,
    table: Mapping[str, Any],
    pem: Mapping[str, Any],
    entries: Sequence[PetScalar | PetField],
    datatypes: set[str],
    start: dt.date,
    *,
    var_csv: Path,
    base: Path,
    output_dir: Path,
    deck_text: str,
    grid_path: Path,
    length: float,
    out: GeophysicsImport,
    source: Path,
) -> list[dict[str, Any]]:
    """Return the ``[[geophysics.data]]`` of PET's ``grav`` or ``sea_disp`` data.

    ``length`` is m per unit of the grid's coordinates, in which SimulatorWrap places its
    stations (``meas_spacing``, ``padding``, ``water_depth``, the seabed table and
    ``well_coord`` are taken in them).
    """
    kind = _SECTIONS[name][0]
    mode = str(table.get("baseline_mode", "fixed")).lower()
    if mode != "fixed":
        raise ConfigError(
            f"{source}: {name} baseline_mode {mode!r}: a rolling difference of {name} is the "
            "difference of two fixed ones, so only fixed is imported"
        )
    if "baseline" not in table:
        # SimulatorWrap then has no baseline state, and fails computing the differences.
        raise ConfigError(f"{source}: {name} needs baseline: its data are differences from it")
    (baseline,) = _days(table, name, "baseline", start, source)
    vintages = _days(table, name, "vintage", start, source)
    early = [v for v in vintages if v < baseline]
    if early:
        raise ConfigError(f"{source}: {name} vintages {early} are before the baseline {baseline}")
    zero_vintage = baseline in vintages
    if zero_vintage:
        vintages.remove(baseline)
        out.notes.append(
            f"PET's {name} vintage at the baseline {baseline} is a difference from itself, zero "
            "by construction: it is left out, with its data"
        )
    options: dict[str, Any] = {
        "baseline": baseline,
    }
    stations = _stations(name, table, base, output_dir, grid_path, length, out, source)
    np.save(output_dir / f"{name}_stations.npy", stations)
    options["stations"] = {"file": f"{name}_stations.npy"}
    if name == "grav":
        options["gravitational_constant"] = _REFERENCE_G
        out.notes.append(
            f"{name}: SimulatorWrap's gravitational constant, 6.67e-11, is kept (APEX's "
            "default is CODATA's 6.67430e-11, 0.06 % more)"
        )
        _phases(pem, deck_text, out, source)
        units = _deck_units(deck_text, out)
        if units != "METRIC":
            out.notes.append(
                f"grav on a {units} deck: SimulatorWrap takes RPORV, the densities and the "
                "grid's lengths as if SI; APEX converts them, so its Δg are not PET's"
            )
    else:
        options.update(_displacement(table, pem, deck_text, out, source))
    out.notes.append(
        f"{name}: every active cell counts; SimulatorWrap drops the cells above RPORV's 90th "
        "percentile in a member it judges to hold an aquifer (filter_rporv), which is not "
        "ported. The number of stations is checked against the data by apex validate"
    )
    data: list[dict[str, Any]] = []
    for datatype in sorted(datatypes):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", datatype):
            raise ConfigError(f"{source}: {name} data type {datatype!r} is not a valid APEX name")
        dated: list[str] = []
        for entry in (e for e in entries if e.datatype == datatype):
            where = f"{source}: {datatype}@{entry.time}"
            if zero_vintage and entry.time == baseline:
                continue  # PET's zero difference at the baseline (noted above)
            if not isinstance(entry, PetField):
                raise ConfigError(f"{where}: {name} data must be an .npz vector")
            if entry.time not in vintages:
                raise ConfigError(
                    f"{where}: PET fills {name} data only at its vintages "
                    f"({', '.join(sorted(vintages))})"
                )
            if entry.values.size != stations.shape[0]:
                raise ConfigError(
                    f"{where}: {entry.values.size} values, but SimulatorWrap's stations over "
                    f"this grid are {stations.shape[0]} (meas_spacing, padding, well_coord)"
                )
            kind_, variance = field_variance(entry, var_csv)
            if kind_ != "abs":
                raise ConfigError(
                    f"{where}: {name} data need an absolute variance (['abs', v], one per "
                    "value); APEX has one variance per field"
                )
            target = output_dir / "fields" / f"{datatype}_{entry.time}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez(target, values=np.asarray(entry.values, dtype=np.float64).ravel())
            out.fields.append(
                {
                    "time": entry.time,
                    "datatype": datatype,
                    "file": str(target.relative_to(output_dir)),
                    "variance": float(variance),
                }
            )
            dated.append(entry.time)
        if len(set(dated)) != len(dated):
            raise ConfigError(f"{source}: {datatype} has two cells at one vintage")
        if not dated:
            raise ConfigError(f"{source}: {datatype} has no data at {name}'s vintages")
        data.append(
            {
                "kind": kind,
                "datatype": datatype,
                "vintages": sorted(dated),
                "options": dict(options),
            }
        )
        unused = sorted(set(vintages) - set(dated))
        if unused:
            out.notes.append(f"{datatype}: PET's vintages {unused} have no data and are left out")
    return data


def _stations(
    name: str,
    table: Mapping[str, Any],
    base: Path,
    output_dir: Path,
    grid_path: Path,
    length: float,
    out: GeophysicsImport,
    source: Path,
) -> NDArray[np.float64]:
    """Return SimulatorWrap's stations ``(n, 3)`` in m, as its ``measurement_locations``.

    Computed as it computes them, in the grid's raw coordinates (before MAPAXES, its own
    length unit; ``meas_spacing``, ``padding``, ``water_depth``, the seabed table and
    ``well_coord`` in that unit), then scaled by ``length`` m.
    """
    spacing = float(table.get("meas_spacing", 1500.0))
    padding = float(table.get("padding", 1500.0))
    if not spacing > 0 or padding < 0:
        raise ConfigError(f"{source}: {name} meas_spacing must be positive, padding at least 0")
    defaulted = [k for k in ("meas_spacing", "padding") if k not in table]
    depth: float | dict[str, NDArray[Any]]
    seabed = table.get("seabed")
    if isinstance(seabed, str):  # SimulatorWrap reads a seabed only when it is a file name
        origin = base / seabed
        try:
            values = np.loadtxt(origin, comments="#", ndmin=2)
        except (OSError, ValueError) as error:
            raise ConfigError(f"{source}: cannot read {name} seabed {origin}: {error}") from error
        if values.shape[1] != 5 or values.shape[0] == 0:
            raise ConfigError(
                f"{source}: {name} seabed {origin.name} needs rows of 5 values: SimulatorWrap "
                "reads x, y, z, column, row"
            )
        depth = {"x": values[:, 0], "y": values[:, 1], "z": values[:, 2]}
    else:
        depth = table.get("water_depth", 300)
        if not isinstance(depth, int | float):
            raise ConfigError(f"{source}: {name} water_depth must be a number")
        if "water_depth" not in table:
            defaulted.append("water_depth")
    well = None
    if name == "grav" and table.get("well_coord") is not None:
        well = table["well_coord"]
        if not (isinstance(well, list) and len(well) == 2):
            raise ConfigError(f"{source}: grav well_coord must be [x, y] (the grid's unit)")
        well = [float(well[0]), float(well[1])]
    coord, zcorn, actnum = _reference_grid(grid_path)
    x, y = _reference_centres(coord, zcorn, actnum)
    stations = _reference_stations(x, y, depth, padding, spacing, well) * length
    if stations.shape[0] == 0:
        raise ConfigError(f"{source}: {name}: SimulatorWrap's stations over this grid are none")
    if defaulted:
        out.notes.append(
            f"{name}: {', '.join(defaulted)} not given, so SimulatorWrap's defaults are used "
            "(meas_spacing and padding 1500, water_depth 300, in the grid's length unit)"
        )
    out.notes.append(
        f"{name}: SimulatorWrap's {stations.shape[0]} stations are written as "
        f"{name}_stations.npy (in m, the grid's own frame), as it computes them"
    )
    return stations


def _reference_grid(path: Path) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.bool_]]:
    """Return an EGRID's COORD, ZCORN and ACTNUM as PET's ``misc.ecl`` holds them.

    ``(NJ+1, NI+1, 2, 3)``, ``(NK, 2, NJ, 2, NI, 2)`` and ``(NK, NJ, NI)``, in the file's
    own dtype (float32 for REAL) and unit, before MAPAXES: SimulatorWrap's grid.

    Raises
    ------
    ConfigError
        If the file cannot be read as a corner-point grid.
    """
    from resdata.resfile import ResdataFile  # optional dependency (the reservoir extra)

    try:
        egrid = ResdataFile(str(path))
        head = np.array(egrid["GRIDHEAD"][0].numpy_view(), copy=True)
        # numpy_view does not own its buffer: each copy is made while egrid lives.
        coord = np.array(egrid["COORD"][0].numpy_view(), copy=True)
        zcorn = np.array(egrid["ZCORN"][0].numpy_view(), copy=True)
        actnum = np.array(egrid["ACTNUM"][0].numpy_view(), copy=True) if "ACTNUM" in egrid else None
    except (OSError, KeyError, ValueError) as error:
        raise ConfigError(f"cannot read the grid file {path}: {error}") from error
    ni, nj, nk = (int(v) for v in head[1:4])
    active = np.ones((nk, nj, ni), dtype=bool) if actnum is None else actnum.astype(bool)
    return (
        coord.reshape(nj + 1, ni + 1, 2, 3),
        zcorn.reshape(nk, 2, nj, 2, ni, 2),
        active.reshape(nk, nj, ni),
    )


def _reference_centres(
    coord: NDArray[Any], zcorn: NDArray[Any], actnum: NDArray[np.bool_]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the active cells' x and y as SimulatorWrap's ``find_cell_centre`` does.

    Its code (``flow_rock.py`` L41-78, d29dc57), arithmetic and dtypes as written: the
    cell's mean depth along its four pillars' mean, in the grid's raw coordinates.
    """
    c, a, b = np.where(actnum)
    xb = 0.25 * (
        coord[a, b, 0, 0]
        + coord[a, b + 1, 0, 0]
        + coord[a + 1, b, 0, 0]
        + coord[a + 1, b + 1, 0, 0]
    )
    yb = 0.25 * (
        coord[a, b, 0, 1]
        + coord[a, b + 1, 0, 1]
        + coord[a + 1, b, 0, 1]
        + coord[a + 1, b + 1, 0, 1]
    )
    zb = 0.25 * (
        coord[a, b, 0, 2]
        + coord[a, b + 1, 0, 2]
        + coord[a + 1, b, 0, 2]
        + coord[a + 1, b + 1, 0, 2]
    )
    xt = 0.25 * (
        coord[a, b, 1, 0]
        + coord[a, b + 1, 1, 0]
        + coord[a + 1, b, 1, 0]
        + coord[a + 1, b + 1, 1, 0]
    )
    yt = 0.25 * (
        coord[a, b, 1, 1]
        + coord[a, b + 1, 1, 1]
        + coord[a + 1, b, 1, 1]
        + coord[a + 1, b + 1, 1, 1]
    )
    zt = 0.25 * (
        coord[a, b, 1, 2]
        + coord[a, b + 1, 1, 2]
        + coord[a + 1, b, 1, 2]
        + coord[a + 1, b + 1, 1, 2]
    )
    z = (
        zcorn[c, 0, a, 0, b, 0]
        + zcorn[c, 0, a, 1, b, 0]
        + zcorn[c, 0, a, 0, b, 1]
        + zcorn[c, 0, a, 1, b, 1]
        + zcorn[c, 1, a, 0, b, 0]
        + zcorn[c, 1, a, 1, b, 0]
        + zcorn[c, 1, a, 0, b, 1]
        + zcorn[c, 1, a, 1, b, 1]
    ) / 8
    denom = zt - zb
    valid = denom > 0
    x = np.copy(xb).astype(float)
    y = np.copy(yb).astype(float)
    if np.any(valid):
        frac = np.empty_like(z, dtype=float)
        frac[valid] = (z[valid] - zb[valid]) / denom[valid]
        x[valid] = xb[valid] + (xt[valid] - xb[valid]) * frac[valid]
        y[valid] = yb[valid] + (yt[valid] - yb[valid]) * frac[valid]
    return x, y  # fmt: skip


def _reference_stations(
    x: NDArray[Any],
    y: NDArray[Any],
    water_depth: float | Mapping[str, NDArray[Any]],
    pad: float,
    dxy: float,
    well_coord: Sequence[float] | None = None,
    r0: float = _REFINE_RADIUS,
) -> NDArray[np.float64]:
    """Return SimulatorWrap's stations ``(n, 3)``, as its ``measurement_locations`` does.

    Its code (``flow_rock.py`` L92-175, d29dc57), in the grid's raw unit: APEX's own
    ``regular_stations`` is the same algorithm, but in m, where the ceilings and the
    refinement's rounding can come out otherwise; so the import writes these.
    """
    from scipy.interpolate import griddata

    x_min, x_max = np.min(x) - pad, np.max(x) + pad
    y_min, y_max = np.min(y) - pad, np.max(y) + pad
    nx = int(np.ceil((x_max - x_min) / dxy))
    ny = int(np.ceil((y_max - y_min) / dxy))
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    if well_coord is not None:
        xc, yc = well_coord[0], well_coord[1]
        dxy_fine = round(dxy / 2)
        fx_min, fx_max = max(x_min, xc - r0), min(x_max, xc + r0)
        fy_min, fy_max = max(y_min, yc - r0), min(y_max, yc + r0)
        pts_coarse = np.column_stack((gx.ravel(), gy.ravel()))
        if fx_max > fx_min and fy_max > fy_min:
            nfx = int(np.ceil((fx_max - fx_min) / dxy_fine)) + 1
            nfy = int(np.ceil((fy_max - fy_min) / dxy_fine)) + 1
            xf, yf = np.meshgrid(np.linspace(fx_min, fx_max, nfx), np.linspace(fy_min, fy_max, nfy))
            pts_fine = np.column_stack((xf.ravel(), yf.ravel()))
            inside = (pts_fine[:, 0] - xc) ** 2 + (pts_fine[:, 1] - yc) ** 2 <= r0**2
            outside = (pts_coarse[:, 0] - xc) ** 2 + (pts_coarse[:, 1] - yc) ** 2 > r0**2
            tol = min(dxy, dxy_fine) * 1e-3
            digits = max(0, int(-np.floor(np.log10(tol))))
            both = np.round(np.vstack((pts_coarse[outside], pts_fine[inside])), digits)
            unique = np.unique(both, axis=0)
            gx, gy = unique[:, 0], unique[:, 1]
    px, py = gx.flatten(), gy.flatten()
    if isinstance(water_depth, Mapping):
        pz = griddata(
            (water_depth["x"], water_depth["y"]), np.abs(water_depth["z"]), (px, py),
            method="nearest",
        )  # fmt: skip
    else:
        pz = np.ones_like(px) * water_depth
    out: NDArray[np.float64] = np.column_stack((px, py, pz)).astype(np.float64)
    return out


def _displacement(
    table: Mapping[str, Any],
    pem: Mapping[str, Any],
    deck_text: str,
    out: GeophysicsImport,
    source: Path,
) -> dict[str, Any]:
    """Return a displacement datum's options other than its baseline and stations."""
    missing = [
        k for k in ("method", "model", "poisson", "compressibility", "z_base") if k not in table
    ]
    if missing:
        raise ConfigError(
            f"{source}: sea_disp needs {', '.join(missing)} (SimulatorWrap reads them)"
        )
    model = str(table["model"]).lower()
    poisson = float(table["poisson"])
    if poisson > 0.5:
        out.notes.append(f"sea_disp poisson {poisson:g} is taken as 0.5, as SimulatorWrap does")
        poisson = 0.5
    if model != "geertsma":
        # map_z_response lowercases the model, which _build_disp_kernel then compares with
        # 'van_Opstal': no model ever reaches the rigid basement.
        out.notes.append(
            f"sea_disp model {table['model']!r}: SimulatorWrap computes Geertsma whatever the "
            "model (pet_parity.md G16), so its data are Geertsma's and are imported as such; "
            "z_base is not used"
        )
    if str(table.get("geertsma_mode", "integral")).lower() != "integral":
        raise ConfigError(
            f"{source}: sea_disp geertsma_mode {table['geertsma_mode']!r}: the fast mode is "
            "not Geertsma's solution (pet_parity.md G15) and is not imported"
        )
    factor = float(table.get("receiver_trunc_factor", 3.0))
    if not factor > 0:
        raise ConfigError(f"{source}: sea_disp receiver_trunc_factor must be positive")
    if factor > 3:
        out.notes.append(
            f"sea_disp receiver_trunc_factor {factor:g}: SimulatorWrap zeroes beyond 3 "
            "depths whatever the factor (pet_parity.md G14), so 3 is imported"
        )
    kernel = {
        "name": "geertsma",
        "options": {"poisson": poisson, "truncation_factor": min(factor, 3.0),
                    "free_surface": "datum"},
    }  # fmt: skip
    out.notes.append(
        "sea_disp: Geertsma is imported as SimulatorWrap computes it, truncated and with "
        "the free surface at sea level (pet_parity.md G12, G13); APEX's defaults are "
        "untruncated, with each station on the free surface"
    )
    units = _deck_units(deck_text, out)
    if units != "METRIC":
        out.notes.append(
            f"sea_disp on a {units} deck: SimulatorWrap takes RPORV and the grid's lengths as "
            "if in m³ and m; APEX converts them, so its displacements are not PET's"
        )
    options: dict[str, Any] = {"kernel": kernel}
    if str(table["method"]).lower() == "pressure":
        conversion = float(pem.get("press_conv", 1.0))
        if not conversion > 0:
            raise ConfigError(f"{source}: pem press_conv must be positive, got {conversion:g}")
        # SimulatorWrap's ΔP is the deck's times press_conv, and compressibility is per that
        # unit: per Pa it is c * press_conv / (Pa per deck unit).
        per_pa = float(table["compressibility"]) * conversion / _DECK_PRESSURE[units]
        if not per_pa > 0:
            raise ConfigError(f"{source}: sea_disp compressibility must be positive")
        options.update(volume_change="pressure", volume="pore", compressibility=per_pa)
    else:
        if str(table["method"]).lower() != "porv":
            out.notes.append(
                f"sea_disp method {table['method']!r}: SimulatorWrap takes any method but "
                "'pressure' as the change of RPORV, as imported"
            )
        options["volume_change"] = "pore_volume"
    return options


def _phases(pem: Mapping[str, Any], deck_text: str, out: GeophysicsImport, source: Path) -> None:
    """Refuse pem phases PET's gravity cannot take; note how its two-phase ones differ.

    ``calc_mass`` takes ``OIL WAT GAS``, ``OIL GAS`` or ``WAT GAS`` (any other set leaves
    its mass change undefined, and ``calc_grav`` fails); with two, the liquid is 1 minus
    the gas saturation, so a third phase of the deck is counted at that liquid's density.
    """
    raw = pem.get("phases", "")
    # SimulatorWrap splits a string, and each item of a list, on spaces.
    items = raw if isinstance(raw, list) else [raw]
    named = frozenset(word.upper() for item in items for word in str(item).split())
    combinations = (
        frozenset({"OIL", "WAT", "GAS"}), frozenset({"OIL", "GAS"}), frozenset({"WAT", "GAS"})
    )  # fmt: skip
    if named not in combinations:
        raise ConfigError(
            f"{source}: pem phases {raw!r}: SimulatorWrap's gravity takes 'OIL WAT GAS', "
            "'OIL GAS' or 'WAT GAS' (any other leaves its mass change undefined)"
        )
    deck = {
        short
        for word, short in (("OIL", "OIL"), ("WATER", "WAT"), ("GAS", "GAS"))
        if re.search(rf"^\s*{word}\s*$", deck_text, re.MULTILINE)
    }
    extra = sorted(named - deck)
    if deck and extra:
        out.notes.append(
            f"grav: PET's pem phases {raw!r} name {', '.join(extra)}, which the deck's RUNSPEC "
            "lacks: SimulatorWrap reads their densities from restarts that do not hold them"
        )
    left_out = sorted(deck - named)
    if left_out:
        liquid = sorted(named - {"GAS"})[0]
        out.notes.append(
            f"grav: PET's pem phases {raw!r} leave out the deck's {', '.join(left_out)}, whose "
            f"pore space SimulatorWrap's gravity fills with {liquid} (1 minus the gas "
            "saturation); APEX counts every phase of the run with its own density"
        )


def _pairs(value: object, name: str, source: Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k).lower(): v for k, v in value.items()}
    if isinstance(value, list) and all(isinstance(p, list) and len(p) == 2 for p in value):
        return {str(p[0]).lower(): p[1] for p in value}
    raise ConfigError(f"{source}: [simulator] {name} must be [key, value] pairs (or a table)")


def _pem_section(pem: Mapping[str, Any], out: GeophysicsImport, source: Path) -> dict[str, Any]:
    """Return ``[geophysics] pem``: soft sand keeps the reference's CO₂ correlation."""
    name = _model(pem, source)
    if name != "soft_sand":
        return {"name": name}
    out.notes.append(
        "softsandrp's CO₂ is Batzle & Wang's gas correlation, kept (gas_model = "
        "'batzle_wang', extrapolate_gas_correlation = true) to reproduce PET; APEX's default "
        "for CO₂ is Span & Wagner's equation of state, 2.6-6 times softer at storage "
        "conditions"
    )
    return {
        "name": name,
        "options": {"gas_model": "batzle_wang", "extrapolate_gas_correlation": True},
    }


def _refuse(pem: Mapping[str, Any], source: Path) -> None:
    for key in ("compaction", "param_file", "grid"):
        if key in pem:
            raise ConfigError(f"{source}: pem {key}: {PEM_KEYS[key]}")


def _model(pem: Mapping[str, Any], source: Path) -> str:
    if "model" not in pem:
        raise ConfigError(f"{source}: pem needs model")
    parts = tuple(str(pem["model"]).split())
    if len(parts) != 2 or parts not in _MODELS:
        raise ConfigError(
            f"{source}: pem model {pem['model']!r} is not ported (supported: "
            f"{', '.join(' '.join(m) for m in _MODELS)})"
        )
    return _MODELS[(parts[0], parts[1])]


def _avo_options(
    avo: Mapping[str, Any], out: GeophysicsImport, source: Path
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Return the ``[geophysics.data.options]`` table and APEX's ``(step_x, step_y)``."""
    missing = [
        key
        for key in ("angle", "frequency", "wave_len", "vp_shale", "vs_shale", "den_shale",
                    "t_min", "t_max", "t_sampling", "pp_func")
        if key not in avo
    ]  # fmt: skip
    if missing:
        raise ConfigError(f"{source}: avo needs {', '.join(missing)} (SimulatorWrap reads them)")
    angle = avo["angle"]
    pp_func = str(avo["pp_func"])
    if pp_func not in _REFLECTIVITY:
        raise ConfigError(
            f"{source}: avo pp_func {pp_func!r} is not one of PyLops' {', '.join(_REFLECTIVITY)}"
        )
    reflectivity, runs = _REFLECTIVITY[pp_func]
    if not runs:
        out.notes.append(
            f"avo pp_func {pp_func!r} fails in SimulatorWrap (pet_parity.md G1); imported as "
            f"APEX's {reflectivity['name']!r}"
        )
    # PET thins its cube's first axis, J, by step_x, and I by step_y (flow_avo.get_avo_result).
    pet_x, pet_y = int(avo.get("step_x", 1)), int(avo.get("step_y", 1))
    if pet_x < 1 or pet_y < 1:
        raise ConfigError(f"{source}: avo step_x and step_y must be at least 1")
    if (pet_x, pet_y) != (1, 1):
        out.notes.append(
            f"PET's step_x = {pet_x} thins J and step_y = {pet_y} thins I; APEX's step_x "
            f"thins I, so the case has step_x = {pet_y}, step_y = {pet_x}"
        )
    options: dict[str, Any] = {
        "angles": [float(a) for a in (angle if isinstance(angle, list) else [angle])],
        "reflectivity": reflectivity,
        "wavelet": {"frequency": float(avo["frequency"]), "length": float(avo["wave_len"])},
        "overburden": {
            "vp": float(avo["vp_shale"]),
            "vs": float(avo["vs_shale"]),
            "density": float(avo["den_shale"]),
        },
        "times": {
            "start": float(avo["t_min"]),
            "end": float(avo["t_max"]),
            "step": float(avo["t_sampling"]),
        },
        "form": "traces",
        "sampling": "nearest",
        "step_x": pet_y,
        "step_y": pet_x,
    }
    out.notes.append(
        "AVO traces are sampled as SimulatorWrap does (sampling = 'nearest', a staircase); "
        "APEX's default, 'interfaces', samples the exact interface times"
    )
    return options, (pet_y, pet_x)


def _check_dimension(
    sim: Mapping[str, Any], pem: Mapping[str, Any], shape: tuple[int, int], source: Path
) -> None:
    # SimulatorWrap sizes its cube by the dimensions it was given (``dimension`` of
    # [simulator] when it has three values, else of pem, for dz/tops files that are not
    # .npz; pem's two values are NX and NZ, with NY = 1): other than the run's grid, its
    # masks do not align.
    where, raw = "[simulator]", sim.get("dimension")
    if not (isinstance(raw, list) and len(raw) >= 3):
        where, raw = "pem", pem.get("dimension")
    if raw is None:
        return
    values = [int(float(v)) for v in (raw if isinstance(raw, list) else [raw])]
    dimension = [values[0], 1] if len(values) == 2 else values[:2]
    if dimension != list(shape):
        raise ConfigError(
            f"{source}: {where} dimension {values} (NX, NY {dimension}) is not grid_file's "
            f"{list(shape)}: "
            "SimulatorWrap could not align its AVO cube with the grid; check the grid file"
        )


def _days(table: Mapping[str, Any], name: str, key: str, start: dt.date, source: Path) -> list[str]:
    """Return a key's ISO dates (PET's days after ``startdate``), in the order given."""
    if key not in table:
        raise ConfigError(f"{source}: {name} needs {key} (days after startdate)")
    raw = table[key] if isinstance(table[key], list) else [table[key]]
    out: list[str] = []
    for value in raw:
        days = float(value)
        if not days.is_integer() or days < 0:
            raise ConfigError(f"{source}: {name} {key} {value!r} must be whole days, at least 0")
        out.append((start + dt.timedelta(days=int(days))).isoformat())
    if len(set(out)) != len(out):
        raise ConfigError(f"{source}: {name} {key} repeats a day")
    return out


def _read_grid(path: Path) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Return ``(n_x, n_y)`` active columns and ``(n_x, n_y, 2)`` column centres of an EGRID.

    The centres are SimulatorWrap's (``_get_grid_coordinates``): the mean of the column's
    four pillars' tops, in the grid's own length unit (before MAPAXES), as its region's
    margin is, and in float32, in its order of summation, as PET's ``misc.ecl`` reads
    COORD: a column on the region's boundary then falls on the same side. (This repeats
    the simulation layer's grid reading, which storage may not import.)
    """
    try:
        from resdata.grid import Grid  # optional dependency (the reservoir extra)
    except ImportError as error:
        raise ConfigError(
            "reading the grid file needs the reservoir extra (resdata): "
            "uv sync --inexact --extra reservoir"
        ) from error
    try:
        grid = Grid(str(path))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the grid file {path}: {error}") from error
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)
    index = grid.export_index(active_only=False)  # every cell, global (F) order
    actnum = np.asarray(index["active"].to_numpy() >= 0).reshape((nx, ny, nz), order="F")
    # numpy_view does not own its buffer: the keyword is held until the copy is made (a
    # temporary would be freed as soon as numpy_view returns).
    keyword = grid.export_coord()
    coord = np.array(keyword.numpy_view(), dtype=np.float32, copy=True)
    del keyword
    # PET's COORD is (NJ+1, NI+1, 2, 3): its centre of (J, I) sums (j, i), (j+1, i),
    # (j, i+1), (j+1, i+1).
    tops = coord.reshape(ny + 1, nx + 1, 6)[:, :, :2]  # (ny+1, nx+1, 2), float32
    pet = np.float32(0.25) * (tops[:-1, :-1] + tops[1:, :-1] + tops[:-1, 1:] + tops[1:, 1:])
    centres = pet.transpose(1, 0, 2).astype(np.float64)  # (nx, ny, 2)
    active: NDArray[np.bool_] = actnum.any(axis=2)
    return active, centres


def _region(
    avo: Mapping[str, Any],
    active: NDArray[np.bool_],
    centres: NDArray[np.float64],
    deck_text: str,
    base: Path,
    out: GeophysicsImport,
    source: Path,
) -> NDArray[np.bool_] | None:
    """Return PET's region of interest as ``(n_x, n_y)``, or ``None`` when it keeps all.

    As ``get_avo_result``'s ``_build_roi_mask``: seeds at the completions' columns that
    the rendered deck itself names (PET does not follow includes), or, with fewer than two,
    also every active column; ``x_range``/``y_range`` in their place; an ellipse or
    rectangle widened by ``roi_margin_m`` (in the grid's length unit); or a file.
    """
    # SimulatorWrap reads its flags with bool(): any non-empty string is true.
    for key in ("use_ellipse", "roi_use_signal"):
        if isinstance(avo.get(key), str):
            out.notes.append(
                f"avo {key} = {avo[key]!r}: SimulatorWrap reads it with bool(), so it is "
                f"{bool(avo[key])}"
            )
    if "roi_mode" in avo:
        mode = str(avo["roi_mode"]).lower()
    else:
        mode = "ellipse" if bool(avo.get("use_ellipse", True)) else "none"
    if mode not in _ROI_MODES:
        raise ConfigError(f"{source}: avo roi_mode {mode!r} is not SimulatorWrap's")
    if mode in ("none", "off", "all"):
        return None
    if bool(avo.get("roi_use_signal", False)):
        raise ConfigError(
            f"{source}: avo roi_use_signal: a region of interest that follows each member's "
            "signal is not imported"
        )
    margin = float(avo.get("roi_margin_m", 500.0))
    if margin < 0:  # SimulatorWrap's _get_avo_info refuses it
        raise ConfigError(f"{source}: avo roi_margin_m must be at least 0, got {margin:g}")
    if mode == "file":
        if "roi_mask_file" not in avo:
            return None  # SimulatorWrap then keeps its all-true region
        return _region_file(base / str(avo["roi_mask_file"]), active.shape, source)
    x, y = centres[..., 0], centres[..., 1]
    ranges = [_given_range(avo.get(key)) for key in ("x_range", "y_range")]
    if ranges[0] is None or ranges[1] is None:  # the completions place it
        sx, sy = _seeds(centres, active, deck_text, out, source)
        ranges = [
            r or (float(v.min()), float(v.max())) for r, v in zip(ranges, (sx, sy), strict=True)
        ]
    (x_min, x_max), (y_min, y_max) = ranges  # type: ignore[misc]
    if mode == "rectangle":
        rect: NDArray[np.bool_] = (
            (x >= x_min - margin) & (x <= x_max + margin)
            & (y >= y_min - margin) & (y <= y_max + margin)
        )  # fmt: skip
        return rect
    rx = max(1.0, 0.5 * (x_max - x_min) + margin)
    ry = max(1.0, 0.5 * (y_max - y_min) + margin)
    dx = (x - 0.5 * (x_min + x_max)) / rx
    dy = (y - 0.5 * (y_min + y_max)) / ry
    ellipse: NDArray[np.bool_] = dx * dx + dy * dy <= 1.0
    return ellipse


def _given_range(value: object) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return min(low, high), max(low, high)
    return None


def _seeds(
    centres: NDArray[np.float64],
    active: NDArray[np.bool_],
    deck_text: str,
    out: GeophysicsImport,
    source: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return SimulatorWrap's seeds' x and y (``_build_roi_mask``).

    Its centres are ``(NY, NX)``, J before I. It decides from their median differences
    whether x runs along their second axis (``use_ji``, as on 3dBox); if so a completion
    ``(I, J)`` is seeded at its own column, else at the transposed one, ``(J, I)``, which
    is reproduced. With fewer than two completions (in the deck's own lines: includes are
    not read) every active column is a seed too.
    """
    x, y = centres[..., 0].T, centres[..., 1].T  # (NY, NX), as SimulatorWrap holds them

    def median_step(values: NDArray[np.float64], axis: int) -> float:
        d = np.diff(values, axis=axis)
        d = d[np.isfinite(d)]
        return float(np.median(np.abs(d))) if d.size else 0.0

    use_ji = (median_step(x, 1) + median_step(y, 0)) > (median_step(x, 0) + median_step(y, 1))
    n_rows, n_cols = x.shape  # SimulatorWrap's "nx" and "ny": NY and NX
    seeds: list[tuple[int, int]] = []  # (row, column) of its (NY, NX) arrays
    for i, j in _completions(deck_text, source):
        if use_ji and 0 <= j < n_rows and 0 <= i < n_cols:
            seeds.append((j, i))
        elif not use_ji and 0 <= i < n_rows and 0 <= j < n_cols:
            seeds.append((i, j))
    if not use_ji and seeds:
        out.notes.append(
            "the grid's x runs along J, so SimulatorWrap seeds each completion (I, J) at the "
            "column (J, I) (pet_parity.md G11); reproduced"
        )
    if len(seeds) < 2:
        rows, cols = np.nonzero(active.T)
        seeds.extend(zip(rows.tolist(), cols.tolist(), strict=True))
        out.notes.append(
            "PET's region surrounds the completions its rendered deck names (not those of "
            "included files); with fewer than two it surrounds every active column"
        )
    r = np.array([s[0] for s in seeds], dtype=np.intp)
    c = np.array([s[1] for s in seeds], dtype=np.intp)
    return x[r, c], y[r, c]


def _region_file(path: Path, shape: tuple[int, int], source: Path) -> NDArray[np.bool_]:
    """Return SimulatorWrap's ``roi_mask_file`` as ``(n_x, n_y)``.

    Its array (``mask``, else the first) is ``(NY, NX)``, or ``(NX, NY)`` and transposed
    (``_normalize_roi_xy_to_grid``); a square one is taken as ``(NY, NX)``.
    """
    if not path.is_file():
        raise ConfigError(f"{source}: avo roi_mask_file not found: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{source}: cannot read avo roi_mask_file {path}: {error}") from error
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        # SimulatorWrap opens it as an .npz (``with np.load(...)``, ``.files``).
        raise ConfigError(f"{source}: avo roi_mask_file must be an .npz, as PET reads it")
    with loaded as data:
        if not data.files:
            raise ConfigError(f"{source}: avo roi_mask_file {path} holds no array")
        values = np.asarray(data["mask" if "mask" in data.files else data.files[0]])
    nx, ny = shape
    if values.shape == (ny, nx):
        out: NDArray[np.bool_] = values.astype(bool).T
        return out
    if values.shape == (nx, ny):
        return values.astype(bool)
    raise ConfigError(f"{path}: shape {values.shape} is not the grid's (NY, NX) = {(ny, nx)}")


def _completions(deck_text: str, source: Path) -> list[tuple[int, int]]:
    """Return the 0-based ``(I, J)`` of each COMPDAT line, read as SimulatorWrap reads them.

    ``_load_completion_ij`` reads the rendered deck line by line: in a COMPDAT block (to a
    line that is only ``/``), each line of three or more tokens gives ``int(tok[1]) - 1``,
    ``int(tok[2]) - 1``, unless those are not integers; includes are not followed.

    Raises
    ------
    ConfigError
        If a line of a COMPDAT block is Mako (its rendered text is unknown here).
    """
    out: list[tuple[int, int]] = []
    in_compdat = False
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        if line.upper().startswith("COMPDAT"):
            in_compdat = True
            continue
        if line == "/":
            in_compdat = False
            continue
        if not in_compdat:
            continue
        if "${" in line or line.startswith(("%", "<%")):
            raise ConfigError(
                f"{source}: the deck's COMPDAT holds Mako ({line!r}), so the completions "
                "PET's region surrounds are not known; set avo roi_mode, or give the region "
                "as a file in the APEX case"
            )
        tokens = line.replace("/", " ").split()
        if len(tokens) < 3:
            continue
        try:
            out.append((int(tokens[1]) - 1, int(tokens[2]) - 1))
        except ValueError:
            continue
    return out


def _deck_units(deck_text: str, out: GeophysicsImport) -> str:
    for line in deck_text.splitlines():
        word = line.split("--", 1)[0].strip().upper()
        if word in _DECK_PRESSURE:
            return word
    note = (
        "the deck template names no unit system (FIELD, METRIC, LAB, PVT-M); METRIC is "
        "assumed for PET's pressures and volumes"
    )
    if note not in out.notes:  # each section asks
        out.notes.append(note)
    return "METRIC"


#: The EGRID ``GRIDUNIT`` lengths, and a deck unit system's, in m.
_GRID_LENGTHS = {"METRES": 1.0, "FEET": 0.3048, "CM": 0.01}
_DECK_LENGTHS = {"METRIC": 1.0, "FIELD": 0.3048, "LAB": 0.01, "PVT-M": 1.0}


def _grid_length(path: Path, deck_text: str, out: GeophysicsImport) -> float:
    """Return m per unit of an EGRID's coordinates: its ``GRIDUNIT``, else the deck's."""
    from resdata.resfile import ResdataFile  # optional dependency (the reservoir extra)

    try:
        egrid = ResdataFile(str(path))
        unit = str(egrid["GRIDUNIT"][0][0]).strip().upper() if "GRIDUNIT" in egrid else ""
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the grid file {path}: {error}") from error
    if unit in _GRID_LENGTHS:
        return _GRID_LENGTHS[unit]
    return _DECK_LENGTHS[_deck_units(deck_text, out)]


def _overburden(
    pem: Mapping[str, Any],
    deck_text: str,
    base: Path,
    output_dir: Path,
    out: GeophysicsImport,
    source: Path,
) -> dict[str, Any]:
    """Return ``overburden_pressure``: PET's file, in the unit its pressures are in.

    SimulatorWrap passes the deck's pressures times ``press_conv`` (default 1) to the PEM,
    which compares them with the overburden file's values; so the file is in that unit.
    """
    if "overburden" not in pem:
        raise ConfigError(
            f"{source}: pem needs overburden (an .npz with obvalues per active cell); or set "
            "[geophysics] overburden_pressure = { gradient = ... } in the APEX case"
        )
    origin = base / str(pem["overburden"])
    if not origin.is_file():
        raise ConfigError(f"overburden file not found: {origin}")
    with np.load(origin) as data:
        if "obvalues" not in data.files:
            raise ConfigError(f"{origin}: PET's standardrp reads the array obvalues")
    target = output_dir / f"overburden{origin.suffix}"
    shutil.copyfile(origin, target)
    units = _deck_units(deck_text, out)
    conversion = float(pem.get("press_conv", 1.0))
    if not conversion > 0:
        raise ConfigError(f"{source}: pem press_conv must be positive, got {conversion:g}")
    pa = _DECK_PRESSURE[units] / conversion
    unit = next((u for u, f in _FILE_UNITS.items() if math.isclose(f, pa, rel_tol=1e-9)), None)
    if unit is None:
        raise ConfigError(
            f"{source}: pem press_conv {conversion:g} on a {units} deck gives pressures of "
            f"{pa:g} Pa per unit, which is none of {', '.join(_FILE_UNITS)}"
        )
    if unit != "MPa":
        out.notes.append(
            f"PET passes the PEM pressures in {unit} ({units} deck, press_conv "
            f"{conversion:g}), where its fits take MPa (pet_parity.md G9); APEX converts the "
            "units, so its elastic properties are those of the true pressures"
        )
    return {"file": target.name, "key": "obvalues", "unit": unit}
