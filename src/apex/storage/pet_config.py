"""Convert a PET history-matching configuration into a runnable APEX case.

Supported: ES (``scheme = "es"``, ``approx`` analysis; ``full`` is the same in one step),
ES-MDA (``scheme = "esmda"`` or ``daalg = ["esmda", ...]``, ``approx``
analysis), LM-EnRML (``scheme = "lmenrml"`` or ``"co_lm_enrml"``) and GN-EnRML (``"gnenrml"``
or ``"gn_enrml"``) with the ``approx``, ``full``, ``subspace``, ``subspace2`` or (GN-EnRML)
``margis`` analysis, and
GIES/RLM-MAC (``daalg = ["gies", "rlmmac"]``, PET-fork). The prior is
either a stored ensemble (``importstate``/``importstaticvar``, or ``--prior``) or PET's
variogram description, converted to the ``gaussian_field`` generator. Field data
(``.npz`` cells) become ``[[observations.fields]]`` and the ``compress`` block becomes
``wavelet`` data transforms (:mod:`apex.storage.pet_compress`), and the ``localization``
block (``autoadaloc`` rank-r, ``distance_loc``) the ``adaptive_projected`` or
``distance_entries`` localization (:mod:`apex.storage.pet_localization`). PET's
geophysical simulators (``pem`` with ``avo``, ``grav`` or ``sea_disp`` in ``[simulator]``)
become ``[geophysics]``: its AVO cubes per-angle trace fields, its gravity and displacement
data station fields (:mod:`apex.storage.pet_geophysics`). Features APEX
does not support (local analysis, empirical data covariance, data screening) are refused,
not silently dropped. Keys that only affect PET's own output (``savefolder``, ...) are ignored
and reported.

The deck template is PET's ``<runfile>.mako``. PET renders it in ``<case>/En_<i>/``, so
``INCLUDE '../<path>'`` means ``<case>/<path>``. Those files are copied into the APEX case
and the includes rewritten to ``'${case_dir}/<path>'``, so the converted case is
self-contained (and include lines stay short).
"""

import datetime as dt
import math
import re
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.storage.pet_compress import Compression, convert_compress
from apex.storage.pet_format import (
    PetField,
    PetScalar,
    field_variance,
    read_pet_data,
    scalar_observations,
    write_observations,
)
from apex.storage.pet_geophysics import check_sections, convert_geophysics, start_date
from apex.storage.pet_localization import convert_localization, localization_name

#: PET data-assimilation keys that change the analysis and are not supported yet.
UNSUPPORTED_DATAASSIM = (
    "localanalysis",
    # PET's emp_cov also changes the data input (datavar holds an error ensemble, and the
    # perturbations are that ensemble): not mappable; APEX's ApproxUpdate
    # empirical_data_error is only its analysis branch (docs/migration/pet_parity.md).
    "emp_cov",
    "screendata",
    "scale_data",
    "actnum",
    "remove_outliers",
)

#: PET keys that only affect PET's own files or bookkeeping.
IGNORED_KEYS = (
    "savefolder",
    "savedata",
    "obsname",
    "truedataindex",
    "assimindex",
    "restart",
    "restartsave",
    "restart_file",
    "reporttype",
    "reportpoint",
    "datatype",
)


@dataclass
class ImportResult:
    """What :func:`import_pet_case` wrote and what it ignored."""

    config: Path
    ignored: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def import_pet_case(
    pet_config: Path,
    output_dir: Path,
    *,
    prior_file: Path | None = None,
    simulator: str = "opm_flow",
    seed: int | None = None,
    name: str | None = None,
    pet_geometry: bool = True,
    truth_file: Path | None = None,
    grid_file: Path | None = None,
    avo_angles_checked: bool = False,
) -> ImportResult:
    """Write an APEX case (``config.toml``, observations, deck, prior) for ``pet_config``.

    Parameters
    ----------
    pet_config
        PET TOML configuration; relative paths in it are resolved against its directory.
    output_dir
        Directory for the APEX case; created if needed.
    prior_file
        Prior ensemble ``.npz`` to use instead of the config's ``importstate``; a relative
        path is relative to the working directory, not to ``pet_config``'s folder.
    simulator
        APEX simulator plugin for the runfile (default ``opm_flow``).
    seed
        Run seed; defaults to the config's ``ensemble.seed``, else 0.
    name
        Run name; defaults to ``output_dir``'s name.
    truth_file
        The true model (``.npz``, one array per state variable) of a synthetic case; it is
        copied into the case, which becomes ``kind = "synthetic"``. Without it the case is
        ``kind = "real"``: PET configurations do not name a true model.
    grid_file
        An EGRID of the deck, for AVO data: its active columns fix the traces. It is
        copied into the case.
    avo_angles_checked
        Import multi-angle AVO data, which the caller knows hold each angle's own traces
        (SimulatorWrap's do not: ``pet_parity.md`` G7).
    pet_geometry
        Reproduce the correlation geometry PET's sampler actually produces (default; exact
        on square grids, see ``docs/migration/pet_parity.md``), or, if false, the geometry
        PET's ``range``/``aniso``/``angle`` describe.

    Raises
    ------
    ConfigError
        If the configuration uses a scheme or feature APEX cannot run.
    """
    try:
        raw = tomllib.loads(pet_config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read PET configuration {pet_config}: {error}") from error
    base = pet_config.resolve().parent
    ensemble, dataassim = (dict(raw.get(k, {})) for k in ("ensemble", "dataassim"))
    # PET reads the simulator's section as [fwdsim], else [simulator] (read_config).
    sim_section = "fwdsim" if raw.get("fwdsim") else "simulator"
    sim = dict(raw.get(sim_section) or {})
    output_dir.mkdir(parents=True, exist_ok=True)
    ignored = sorted(
        f"{section}.{key}"
        for section, table in (("dataassim", dataassim), (sim_section, sim))
        for key in table
        if key in IGNORED_KEYS
    )

    unsupported = [key for key in UNSUPPORTED_DATAASSIM if key in dataassim]
    if unsupported:
        raise ConfigError(f"{pet_config}: APEX does not support {', '.join(unsupported)} yet")

    method = _method(dataassim, pet_config)
    notes: list[str] = []
    if method.get("strategy", {}).get("name") == "hybrid":
        method = _multilevel_method(method, ensemble, sim, pet_config, notes)
    state_variables, prior = _state_and_prior(
        ensemble, base, output_dir, prior_file, pet_config, notes, pet_geometry
    )
    if method.get("strategy", {}).get("name") == "full":
        scaling = _pet_state_scaling(ensemble, state_variables, notes)
        if scaling is not None:
            np.savez(output_dir / "state_scaling.npz", scaling=scaling)
            method["strategy"]["options"]["state_scaling_file"] = "state_scaling.npz"
    avo = check_sections(sim, pet_config)
    var_csv = base / dataassim["datavar"]
    deck = _deck_template(base, sim, output_dir, pet_config)
    entries, labels = _dated(
        read_pet_data(base / dataassim["data"], var_csv, base),
        _start(sim, avo, pet_config, notes),
        deck.read_text(),
        notes,
        pet_config,
    )
    compress = _table(dataassim["compress"]) if "compress" in dataassim else None
    geophysics = convert_geophysics(
        sim,
        entries,
        var_csv=var_csv,
        base=base,
        output_dir=output_dir,
        deck_text=deck.read_text(),
        grid_file=grid_file,
        compressed_types=_names(compress.get("compress_data", [])) if compress else [],
        angles_checked=avo_angles_checked,
        source=pet_config,
    )
    notes.extend(geophysics.notes)
    ignored = sorted(
        [*ignored, *(f"{sim_section}.{k.split('.', 1)[1]}" for k in geophysics.ignored)]
    )
    entries = [e for e in entries if e.datatype not in geophysics.datatypes]
    compression = (
        convert_compress(compress, entries, base, output_dir)
        if compress is not None
        else Compression()
    )
    notes.extend(compression.notes)
    observations = _write_observations(entries, compression.compressed, output_dir, notes, var_csv)
    observations.setdefault("fields", []).extend(geophysics.fields)
    if not observations["fields"]:
        del observations["fields"]
    if "localization" in dataassim:
        if geophysics.datatypes and _is_distance_loc(dataassim["localization"]):
            raise ConfigError(
                f"{pet_config}: distance_loc tapers whole fields around a cell, which "
                "cannot localize geophysical traces or stations; use autoadaloc"
            )
        if method["name"] == "gies":
            raise ConfigError(
                f"{pet_config}: GIES's localization block is PET-fork's, which the "
                "importer does not read; set [method] localization in the APEX case by hand"
            )
        method["localization"] = convert_localization(
            dataassim["localization"],
            strategy=str(method.get("strategy", {}).get("name")),
            datatypes=list(dict.fromkeys(e.datatype for e in entries)),  # AVO's left out
            times=list(dict.fromkeys(e.time for e in entries)),
            time_labels=labels,
            variables=[str(v["name"]) for v in state_variables],
            base=base,
            output_dir=output_dir,
            notes=notes,
        )

    simulator_options: dict[str, Any] = {"deck_template_file": deck.name, "keep_files": False}
    if "sim_limit" in sim:
        simulator_options["timeout_s"] = float(sim["sim_limit"])
    config: dict[str, Any] = {
        "run": {
            "name": name or output_dir.name,
            "seed": int(seed if seed is not None else ensemble.get("seed", 0)),
        },
        "state": {"variables": state_variables},
        "prior": prior,
        "observations": observations,
        "case": _case(truth_file, output_dir),
        "simulator": {"name": simulator, "options": simulator_options},
        "execution": {"name": "process", "options": {"workers": int(sim.get("parallel", 1))}},
        "method": method,
    }
    if compression.transforms:
        config["data_transforms"] = compression.transforms
    if geophysics.section is not None:
        config["geophysics"] = geophysics.section
    path = output_dir / "config.toml"
    header = f"# Converted from PET configuration {pet_config} by `apex pet-import`.\n"
    path.write_text(header + tomli_w.dumps(config))
    return ImportResult(path, ignored, notes)


def _start(sim: Mapping[str, Any], avo: bool, source: Path, notes: list[str]) -> dt.date | None:
    """Return ``startdate``; outside AVO cases one APEX cannot read is left with a note."""
    try:
        return start_date(sim, source)
    except ConfigError:
        if avo:
            raise
        notes.append(
            f"[simulator] startdate {sim['startdate']!r} is not day/month/year, so PET's "
            "report labels are kept as they are"
        )
        return None


#: A deck's START record: ``START`` alone on its line, then ``1 'JAN' 2022 /``.
_START = re.compile(r"^\s*START\s*$\s*^\s*(\d+)\s+'?([A-Za-z]{3})'?\s+(\d{4})", re.MULTILINE)
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _dated(
    entries: list[PetScalar | PetField],
    start: dt.date | None,
    deck_text: str,
    notes: list[str],
    source: Path,
) -> tuple[list[PetScalar | PetField], dict[str, str]]:
    """Return the entries with PET's day labels as ISO dates, and each label's date.

    PET's report labels are often days after ``[simulator] startdate`` (its AVO data must
    be: SimulatorWrap compares them with the vintages' days), while APEX's deck simulators
    report at ISO dates. Labels are converted only when there is a start date and every
    one is a whole number of days; the deck's own START must then be that date.

    Raises
    ------
    ConfigError
        If the deck's START is another date: the dates would match no report time.
    """
    if start is None or not entries:
        return entries, {}
    days: dict[str, float] = {}
    for label in dict.fromkeys(e.time for e in entries):
        try:
            value = float(label)
        except ValueError:
            return entries, {}
        if not (math.isfinite(value) and value.is_integer() and value >= 0):
            return entries, {}
        days[label] = value
    # Comments may stand between START and its date (Eclipse skips them).
    match = _START.search("\n".join(line.split("--", 1)[0] for line in deck_text.splitlines()))
    if match is not None and match.group(2).upper().replace("JLY", "JUL") in _MONTHS:
        month = _MONTHS.index(match.group(2).upper().replace("JLY", "JUL")) + 1
        try:
            deck_start: dt.date | None = dt.date(int(match.group(3)), month, int(match.group(1)))
        except ValueError:
            deck_start = None  # the simulator will refuse it
        if deck_start is not None and deck_start != start:
            raise ConfigError(
                f"{source}: PET's report labels are days after startdate {start.isoformat()}, "
                f"but the deck starts on {deck_start.isoformat()}"
            )
    labels = {k: (start + dt.timedelta(days=int(v))).isoformat() for k, v in days.items()}
    notes.append(
        f"PET's report labels are days after its startdate {start.isoformat()}; the case "
        "uses their dates"
    )
    return [replace(e, time=labels[e.time]) for e in entries], labels


def _names(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def _is_distance_loc(block: object) -> bool:
    if not isinstance(block, Mapping):
        return False  # convert_localization refuses it
    return localization_name({str(k).lower(): v for k, v in block.items()}) == "distance_loc"


def _case(truth_file: Path | None, output_dir: Path) -> dict[str, Any]:
    if truth_file is None:
        return {"kind": "real"}
    if not truth_file.is_file():
        raise ConfigError(f"true-model file not found: {truth_file}")
    shutil.copyfile(truth_file, output_dir / "truth.npz")
    return {"kind": "synthetic", "truth_file": "truth.npz"}


def _write_observations(
    entries: list[PetScalar | PetField],
    compressed: set[tuple[str, str]],
    output_dir: Path,
    notes: list[str],
    var_csv: Path,
) -> dict[str, Any]:
    """Write scalar data to ``observations.csv`` and fields to ``fields/``; return the section.

    A field's ``['abs', v]`` variance applies to every value, as in PET. Compressed fields
    get the wavelet noise estimate instead and their variance cell is not read (as in PET);
    a relative variance of an uncompressed field (one per value) is not supported.
    """
    section: dict[str, Any] = {}
    scalars = [e for e in entries if isinstance(e, PetScalar)]
    if scalars:
        try:
            write_observations(scalar_observations(scalars), output_dir / "observations.csv")
        except ValueError as error:
            raise ConfigError(f"PET data: {error}") from error
        section["file"] = "observations.csv"
    fields = []
    placeholder = False
    for entry in entries:
        if not isinstance(entry, PetField):
            continue
        if (entry.time, entry.datatype) in compressed:
            variance, placeholder = 1.0, True
        elif (cell := field_variance(entry, var_csv))[0] == "abs":
            variance = cell[1]
        else:
            raise ConfigError(
                f"PET data: {entry.datatype}@{entry.time} is an uncompressed field with a "
                "relative variance, which APEX does not support (one variance per field)"
            )
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{entry.datatype}_{entry.time}") + ".npz"
        target = output_dir / "fields" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, values=entry.values)
        fields.append(
            {
                "time": entry.time,
                "datatype": entry.datatype,
                "file": str(target.relative_to(output_dir)),
                "variance": float(variance),
            }
        )
    if placeholder:
        notes.append(
            "compressed fields get variance = 1.0 in the config; the wavelet transform "
            "replaces it with its noise estimate, as PET does"
        )
    if fields:
        names = [f["file"] for f in fields]
        if len(set(names)) != len(names):
            raise ConfigError("PET data: two fields map to the same file name")
        section["fields"] = fields
    return section


#: PET's EnRML schemes: APEX method, the ported analyses (the first is the default), and
#: the iteration keys of its control (PET key, APEX schedule key).
_LM_KEYS = (
    ("lambda", "initial"),
    ("lambda_factor", "factor"),
    ("lambda_min", "minimum"),
    ("lambda_max", "maximum"),
)
_GN_KEYS = (("gamma", "initial"), ("gamma_max", "maximum"), ("gamma_factor", "factor"))
_ENRML_SCHEMES: dict[str, tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "lmenrml": ("lm_enrml", ("approx", "full", "subspace", "subspace2"), _LM_KEYS),
    "co_lm_enrml": ("lm_enrml", ("approx",), _LM_KEYS),
    "gnenrml": ("gn_enrml", ("approx", "full", "subspace", "subspace2", "margis"), _GN_KEYS),
    "gn_enrml": ("gn_enrml", ("subspace",), _GN_KEYS),
}


def _method(dataassim: Mapping[str, Any], source: Path) -> dict[str, Any]:
    scheme, flavour = _scheme(dataassim)
    energy = _fraction(dataassim.get("energy"))
    if scheme == "esmda":
        analysis = str(dataassim.get("analysis", "approx"))
        if flavour == "esmda_hybrid":
            analysis = "hybrid"
        if analysis not in ("approx", "hybrid") or flavour not in (
            None,
            "esmda",
            "esmda_approx",
            "esmda_hybrid",
        ):
            raise ConfigError(
                f"{source}: only the approx and hybrid (multilevel) ES-MDA analyses are ported "
                f"(got {analysis!r})"
            )
        mda = _table(dataassim.get("mda", {}))
        steps = int(mda["tot_assim_steps"]) if "tot_assim_steps" in mda else None
        inflation = mda.get("inflation_param")
        if inflation is None:
            if steps is None:
                raise ConfigError(f"{source}: [dataassim.mda] needs tot_assim_steps")
            inflation = [float(steps)] * steps
        elif not isinstance(inflation, list):
            inflation = [float(inflation)] * (steps or 1)
        strategy_options = {"energy": energy if energy is not None else 0.98}
        if analysis == "hybrid":  # completed by _multilevel_method (levels from [ensemble])
            return {
                "name": "multilevel_esmda",
                "options": {"inflation": [float(a) for a in inflation]},
                "strategy": {"name": "hybrid", "options": strategy_options},
            }
        return {
            "name": "esmda",
            "options": {"inflation": [float(a) for a in inflation]},
            "strategy": {"name": "approx", "options": strategy_options},
        }
    if scheme == "es":
        analysis = str(dataassim.get("analysis", "approx"))
        # PET's ES (an EnKF with one data group) maps "full" onto approx: one step, the
        # prior-increment term is never reached.
        if analysis not in ("approx", "full"):
            raise ConfigError(f"{source}: only the approx ES analysis is ported (got {analysis!r})")
        return {
            "name": "es",
            "strategy": {
                "name": "approx",
                "options": {"energy": energy if energy is not None else 0.98},
            },
        }
    if scheme == "gies":
        if flavour not in (None, "rlmmac"):
            raise ConfigError(
                f"{source}: only GIES with the rlmmac update is ported (got {flavour!r})"
            )
        iteration = _table(dataassim.get("iteration", {}))
        # converge_on_rejection keeps its default, false (the user, 2026-10-04): imported
        # cases stop only on an improving trial, not on PET-fork's worse one.
        schedule = {
            apex_key: float(iteration[pet_key])
            for pet_key, apex_key in (
                ("lambda", "initial"),
                ("lambda_factor", "factor"),
                ("lambda_min", "minimum"),
                ("data_misfit_tol", "data_mismatch_tolerance"),
            )
            if pet_key in iteration
        }
        # PET-fork divides λ by lambda_factor after an accepted step; APEX multiplies by
        # `reduction` (0.9 by default since 2026-10-13). Imported cases keep PET-fork's
        # rule (the user, 2026-10-14): 1/factor, which at its default factor of 2 is 0.5,
        # the same operation bit for bit, as for any power of two; another factor's
        # reciprocal rounds, so λ can differ from PET-fork's in its last bit.
        factor = schedule.get("factor", 2.0)
        if not factor > 1.0:
            raise ConfigError(
                f"{source}: GIES's lambda_factor must be greater than 1 (got {factor:g}): "
                "it multiplies the damping after a rejected step and divides it after an "
                "accepted one"
            )
        schedule["reduction"] = 1.0 / factor
        return {
            "name": "gies",
            "strategy": {
                "name": "rlmmac",
                "options": {"energy": energy if energy is not None else 0.95},
            },
            "schedule": schedule,
            "max_iterations": int(iteration.get("max_iter", 10)),
        }
    if scheme in _ENRML_SCHEMES:
        name, analyses, keys = _ENRML_SCHEMES[scheme]
        # converge_on_rejection keeps its default, false, as for GIES above (the user);
        # so does lm's confirm_convergence, true (finding 34, 2026-10-19: PET's rule stops
        # a heavily damped run at its prior).
        # PET's legacy names pin the analysis (co_lm_enrml: approx; gn_enrml: subspace).
        analysis = str(dataassim.get("analysis", analyses[0]))
        if analysis not in analyses:
            raise ConfigError(
                f"{source}: scheme {scheme!r} is ported with the "
                f"{' and '.join(analyses)} analyses only (got {analysis!r})"
            )
        iteration = _table(dataassim.get("iteration", {}))
        lm_schedule: dict[str, Any] = {
            apex_key: iteration[pet_key]
            if pet_key in ("lambda", "gamma") and iteration[pet_key] == "auto"
            else float(iteration[pet_key])
            for pet_key, apex_key in (*keys, ("data_misfit_tol", "data_mismatch_tolerance"))
            if pet_key in iteration
        }
        # PET's IterativeEnRML reads the truncation energy from the iteration block only
        # (a top-level dataassim energy does not reach its approx update).
        lm_energy = _fraction(iteration.get("energy"))
        return {
            "name": name,
            "options": {"max_inner_iterations": int(iteration.get("max_inner_iter", 10))},
            "strategy": {
                "name": analysis,
                # The transform analyses use no SVD (no energy).
                "options": {}
                if analysis in ("subspace2", "margis")
                else {"energy": lm_energy if lm_energy is not None else 0.95},
            },
            "schedule": lm_schedule,
            "max_iterations": int(iteration.get("max_iter", 10)),
        }
    raise ConfigError(
        f"{source}: scheme {scheme!r} is not ported to APEX (supported: es, esmda, gies, "
        f"{', '.join(_ENRML_SCHEMES)})"
    )


def _multilevel_method(
    method: dict[str, Any],
    ensemble: Mapping[str, Any],
    sim: Mapping[str, Any],
    source: Path,
    notes: list[str],
) -> dict[str, Any]:
    """Add the fidelity levels of PET's ``esmda_hybrid``.

    ``[ensemble] multilevel`` gives ``levels``, ``en_size`` and ``ml_weights`` (or
    ``cov_wgt``); SimulatorWrap's ``trunc_level`` (per level) and ``upscale`` pairs
    (``state``, ``dim``, ``maxtrunc``, ``maxdiff``, ``wells``, ``radius``, ``us_type``) give
    each level's grid coarsening; a level with ``trunc_level`` 0 runs the full grid.
    """
    if "multilevel" not in ensemble:
        raise ConfigError(f"{source}: the hybrid analysis needs [ensemble] multilevel")
    multilevel = _table(ensemble["multilevel"])
    n_levels = int(multilevel["levels"])
    sizes = [int(n) for n in multilevel["en_size"]]
    # PET: cov_wgt, when given, replaces ml_weights (extract_multilevel_info).
    weights = [float(w) for w in multilevel.get("cov_wgt", multilevel.get("ml_weights", []))]
    if len(sizes) != n_levels or len(weights) != n_levels:
        raise ConfigError(
            f"{source}: multilevel needs en_size and ml_weights for {n_levels} levels"
        )
    trunc = [float(t) for t in sim.get("trunc_level", [0.0] * n_levels)]
    upscale = {str(pair[0]).lower(): pair[1:] for pair in sim.get("upscale", [])}
    levels: list[dict[str, Any]] = []
    for level in range(n_levels):
        entry: dict[str, Any] = {"members": sizes[level], "weight": weights[level]}
        if upscale and level < len(trunc) and trunc[level] > 0:
            entry["upscale"] = {
                "method": "haar" if str(upscale.get("us_type", ["unif"])[0]).lower() == "haar"
                else "uniform",
                "dim": [int(n) for n in upscale["state"][1]],
                "state": str(upscale["state"][0]),
                "max_trunc": trunc[level],
                "max_diff": float(upscale.get("maxdiff", [0.0])[0]),
                "wells": [[int(v) for v in w] for w in upscale.get("wells", [])],
                "radius": [int(r) for r in upscale.get("radius", [[]])[0]],
            }  # fmt: skip
        levels.append(entry)
    method["options"]["levels"] = levels
    if any("upscale" in level for level in levels):
        notes.append(
            "multilevel levels coarsen the grid with Eclipse COARSEN: run with "
            "--simulator eclipse (OPM Flow has no COARSEN; give coarse levels their own "
            "deck_template_file instead)"
        )
    return method


def _scheme(dataassim: Mapping[str, Any]) -> tuple[str, str | None]:
    if "scheme" in dataassim:
        return str(dataassim["scheme"]).lower(), None
    daalg = dataassim.get("daalg")
    if isinstance(daalg, list) and daalg:
        return str(daalg[0]).lower(), str(daalg[1]).lower() if len(daalg) > 1 else None
    raise ConfigError("PET [dataassim] needs `scheme` (or legacy `daalg`)")


def _table(value: object) -> dict[str, Any]:
    # PET accepts both TOML tables and the legacy list of [key, value] pairs.
    if isinstance(value, dict):
        return {str(k).lower(): v for k, v in value.items()}
    if isinstance(value, list):
        return {str(pair[0]).lower(): pair[1] for pair in value}
    raise ConfigError(f"expected a table or [key, value] pairs, got {value!r}")


def _fraction(energy: object) -> float | None:
    if energy is None:
        return None
    value = float(energy)  # type: ignore[arg-type]
    return value / 100.0 if value > 1 else value


def _state_and_prior(
    ensemble: Mapping[str, Any],
    base: Path,
    output_dir: Path,
    prior_file: Path | None,
    source: Path,
    notes: list[str],
    pet_geometry: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map PET's state and prior to APEX state variables and a ``[prior]`` section.

    A stored ensemble (``--prior`` or ``importstate``) becomes the ``file`` generator;
    otherwise each ``prior_<name>`` with a variogram becomes ``gaussian_field``.
    """
    state = ensemble.get("state")
    names = [state] if isinstance(state, str) else list(state or [])
    if not names:
        raise ConfigError(f"{source}: [ensemble] needs `state`")
    stored = ensemble.get("importstate") or ensemble.get("importstaticvar")
    variables: list[dict[str, Any]] = []
    stored_path: Path | None = None
    if prior_file is not None:
        # The caller's path (``--prior``) is the caller's: relative to the working directory,
        # as every command-line path is; only the PET configuration's own paths are relative
        # to its folder (the release tester, 2026-10-19).
        stored_path = prior_file.expanduser().absolute()
    elif stored is not None:
        stored_path = Path(stored) if Path(stored).is_absolute() else base / stored
    if stored_path is not None:
        if not stored_path.is_file():
            raise ConfigError(f"prior ensemble not found: {stored_path}")
        target = output_dir / "prior_ensemble.npz"
        if stored_path.resolve() != target.resolve():
            shutil.copy2(stored_path, target)
        with np.load(target) as data:
            for name in names:
                if name not in data or data[name].ndim != 2:
                    raise ConfigError(f"{stored_path}: needs a (size, n_ensemble) array {name!r}")
                variables.append(_variable(name, int(data[name].shape[0]), ensemble, source))
        prior: dict[str, Any] = {
            "generator": {"name": "file", "options": {"ensemble_file": target.name}}
        }
        if "ne" in ensemble:
            prior["ensemble_size"] = int(ensemble["ne"])
        return variables, prior

    if "ne" not in ensemble:
        raise ConfigError(f"{source}: [ensemble] needs `ne` to sample a prior")
    generators: dict[str, Any] = {}
    for name in names:
        info = {str(k).lower(): v for k, v in dict(ensemble.get(f"prior_{name}", {})).items()}
        if "vario" not in info or "grid" not in info:
            raise ConfigError(
                f"{source}: prior_{name} needs `vario` and `grid` (or give a stored ensemble "
                "with --prior)"
            )
        generator, size = _gaussian_field(name, info, base, output_dir, source, notes, pet_geometry)
        variables.append(_variable(name, size, ensemble, source))
        generators[name] = generator
    return variables, {"ensemble_size": int(ensemble["ne"]), "variables": generators}


#: PET variogram names to APEX models (PET's ``exp`` is exp(-3 u^1.9), not exponential).
_PET_VARIOGRAMS = {"sph": ("spherical", 1.0), "cub": ("cubic", 1.0), "exp": ("stable", 1.9)}


def _gaussian_field(
    name: str,
    info: Mapping[str, Any],
    base: Path,
    output_dir: Path,
    source: Path,
    notes: list[str],
    pet_geometry: bool,
) -> tuple[dict[str, Any], int]:
    grid = [int(n) for n in info["grid"]]
    nx, ny, nz = [*grid, 1][:3]
    where = f"{source}: prior_{name}"

    def layers(*keys: str, default: object = None) -> list[Any]:
        # PET accepts both spellings (``var``/``variance``, ``range``/``corr_length``) and
        # copies a single entry to every layer; anything else must have one per layer.
        value = next((info[k] for k in keys if k in info), default)
        if value is None:
            raise ConfigError(f"{where} needs `{keys[0]}`")
        values = list(value) if isinstance(value, list) else [value]
        if len(values) == 1:
            return values * nz
        if len(values) != nz:
            raise ConfigError(f"{where} `{keys[0]}` has {len(values)} entries; give 1 or {nz}")
        return values

    varios = [str(v).lower() for v in layers("vario")]
    unknown = sorted(set(varios) - set(_PET_VARIOGRAMS))
    if unknown:
        raise ConfigError(f"{where} vario {unknown} is not supported")
    ranges = [float(r) for r in layers("range", "corr_length")]
    aniso = [float(a) for a in layers("aniso", default=1.0)]
    angles = [float(a) for a in layers("angle", default=0.0)]
    options: dict[str, Any] = {
        "grid": [nx, ny, nz],
        "variance": [float(v) for v in layers("var", "variance", default=1.0)],
        "variogram": [_PET_VARIOGRAMS[v][0] for v in varios],
        "exponent": [_PET_VARIOGRAMS[v][1] for v in varios],
    }
    try:
        geometry, geometry_notes = pet_field_geometry(nx, ny, ranges, aniso, angles, pet_geometry)
    except ConfigError as error:
        raise ConfigError(f"{where}: {error}") from error
    options.update(geometry)
    notes.extend(f"prior_{name}: {note}" for note in geometry_notes)
    mean = info.get("mean", 0.0)
    if isinstance(mean, str):
        mean_path = base / mean
        if not mean_path.is_file():
            raise ConfigError(f"prior mean file not found: {mean_path}")
        target = output_dir / f"{name}_prior_mean{mean_path.suffix}"
        shutil.copy2(mean_path, target)
        options["mean_file"] = target.name
    else:
        values = [float(m) for m in (mean if isinstance(mean, list) else [mean])]
        if len(values) == 1:
            options["mean"] = values[0]
        elif len(values) == nx * ny * nz:
            options["mean"] = values
        elif len(values) == nz:  # one per layer
            options["mean"] = [m for m in values for _ in range(nx * ny)]
        else:
            raise ConfigError(
                f"{where} mean has {len(values)} entries; give 1, one per layer ({nz}) or one "
                f"per cell ({nx * ny * nz})"
            )
    return {"name": "gaussian_field", "options": options}, nx * ny * nz


def pet_field_geometry(
    nx: int,
    ny: int,
    ranges: list[float],
    aniso: list[float],
    angles: list[float],
    pet_geometry: bool = True,
) -> tuple[dict[str, list[float]], list[str]]:
    """``gaussian_field`` ranges and azimuths for PET's per-layer ``range``/``aniso``/``angle``.

    Returns the ``major_range``, ``minor_range`` and ``azimuth`` options and notes on how
    PET's sampler was interpreted.

    Raises
    ------
    ConfigError
        If PET's sampler is degenerate (``angle`` of 90 degrees).
    """
    notes: list[str] = []
    if pet_geometry and nx == ny:
        # What PET samples (geostat gen_cov2d, parity case prior_field_aniso): the rotation
        # matrix is multiplied element-wise, so the angle only divides every range by
        # |cos(angle)|; cells are enumerated y-fastest, which on a square grid transposes
        # the field, so PET's major axis (range * aniso) runs along j (azimuth 90).
        scales = [abs(math.cos(math.radians(a))) for a in angles]
        if min(scales) < 1e-12:
            raise ConfigError(f"angle {angles}: PET's sampler is degenerate at 90 degrees")
        if any(a != 0 for a in angles):
            notes.append(
                f"PET's angle {angles} does not rotate its field (the rotation is applied "
                "element-wise); reproduced as ranges divided by |cos(angle)|. Use "
                "--intended-geometry for a rotated field."
            )
        if any(a != 1 for a in aniso):
            notes.append(
                "PET enumerates cells y-fastest, so its major axis (aniso) runs along j; "
                "reproduced with azimuth 90. Use --intended-geometry for a major axis along i."
            )
        return {
            "minor_range": [r / c for r, c in zip(ranges, scales, strict=True)],
            "major_range": [r * a / c for r, a, c in zip(ranges, aniso, scales, strict=True)],
            "azimuth": [90.0 if a != 1 else 0.0 for a in aniso],
        }, notes
    # The intended geometry: range along the minor axis, aniso = major / minor, the major
    # axis along i rotated by `angle` degrees clockwise.
    if pet_geometry:
        notes.append(
            f"on a non-square grid ({nx} x {ny}) PET's y-fastest cell order scrambles its "
            "field, which no stationary covariance reproduces; APEX samples the intended "
            "geometry in Eclipse cell order instead."
        )
    return {
        "minor_range": ranges,
        "major_range": [r * a for r, a in zip(ranges, aniso, strict=True)],
        "azimuth": [-a for a in angles],
    }, notes


def _variable(name: str, size: int, ensemble: Mapping[str, Any], source: Path) -> dict[str, Any]:
    variable: dict[str, Any] = {"name": name, "size": size}
    limits = dict(ensemble.get(f"prior_{name}", {})).get("limits")
    if limits is None:
        return variable
    if isinstance(limits, dict):
        low, high = limits.get("lower"), limits.get("upper")
    elif (
        isinstance(limits, list)
        and len(limits) == 2
        and not any(isinstance(v, (list, dict)) for v in limits)
    ):
        low, high = limits
    else:
        raise ConfigError(
            f"{source}: prior_{name} limits must be one [lower, upper] pair; APEX bounds a "
            f"variable as a whole, not per layer (got {limits!r})"
        )
    if low is not None:
        variable["lower"] = float(low)
    if high is not None:
        variable["upper"] = float(high)
    return variable


def _deck_template(base: Path, sim: Mapping[str, Any], output_dir: Path, source: Path) -> Path:
    runfile = sim.get("runfile")
    if runfile is None:
        raise ConfigError(f"{source}: [simulator] needs `runfile`")
    template = base / f"{runfile}.mako"
    if not template.is_file():
        raise ConfigError(f"deck template not found: {template}")
    text = template.read_text()
    # PET renders decks in <case>/En_<i>/, so '../x' is <case>/x: copy x into the new case.
    for relative in sorted(set(re.findall(r"'\.\./([^']+)'", text))):
        origin = base / relative
        if not origin.is_file():
            raise ConfigError(f"{template}: included file not found: {origin}")
        (output_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, output_dir / relative)
    deck = output_dir / "deck.mako"
    deck.write_text(text.replace("'../", "'${case_dir}/"))
    return deck


def _pet_state_scaling(
    ensemble: Mapping[str, Any], variables: Sequence[Mapping[str, Any]], notes: list[str]
) -> NDArray[np.float64] | None:
    """Return PET's ``state_scaling`` (``calc_scaling``) from the configured prior variances.

    The full analysis depends on the ratios between the variables' scalings, so an imported
    case gets PET's, not APEX's prior-ensemble default. Per-layer variances give each
    layer's cells its value (layers slowest, as the state is stored); PET's ``np.tile``
    cycles through the layers cell by cell instead, which looks like a PET bug
    (docs/migration/pet_parity.md). ``None``, with a note, when a variance is not given.
    """
    rows = []
    for variable in variables:
        name, size = str(variable["name"]), int(variable["size"])
        info = {str(k).lower(): v for k, v in dict(ensemble.get(f"prior_{name}", {})).items()}
        value = info.get("var", info.get("variance"))
        if value is None:
            notes.append(
                f"prior_{name} gives no variance: the full analysis scales it by the prior "
                "ensemble's standard deviation (PET needs the configured variance)"
            )
            return None
        values = np.sqrt(np.asarray(value if isinstance(value, list) else [value], dtype=float))
        if values.size == 1:
            rows.append(np.full(size, values[0]))
        elif size % values.size == 0:
            rows.append(np.repeat(values, size // values.size))
            notes.append(
                f"prior_{name}: per-layer variances scale each layer's cells (PET tiles them "
                "cell by cell, which looks like a PET bug)"
            )
        else:
            raise ConfigError(f"prior_{name}: {values.size} variances do not divide {size} cells")
    return np.concatenate(rows)
