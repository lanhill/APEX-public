"""An APEX configuration as a self-contained ERT case.

``export_case(config, out)`` writes:

- ``case.ert``: one ``FIELD`` per gridded state variable (prior from ``INIT_FILES``,
  ``FORWARD_INIT:False``, the variable's bounds as ``MIN``/``MAX``), a ``SUMMARY`` line
  per simulated key, ``OBS_CONFIG``, ``FORWARD_MODEL APEX_RENDER_DECK`` then ``FLOW``,
  ``QUEUE_SYSTEM LOCAL`` with ``MAX_RUNNING`` from the workers, ``RANDOM_SEED`` from the
  run's seed, and, unless ``ert_defaults``, no outlier rejection (``ENKF_ALPHA``), the
  approx strategy's truncation energy (``ENKF_TRUNCATION``), and a ``STD_CUTOFF``
  that drops only constant responses, for like-for-like comparisons with APEX;
- ``prior/<FIELD>_<IENS>.grdecl``: APEX's prior, drawn with the run's own generators and
  stream, in Eclipse cell order;
- ``grid/<ECLBASE>.GRID``: the grid's dimensions (``DIMENS``), which is all ERT reads from a
  grid that is not an EGRID (enough for ``FIELD``; distance localization in ERT would need
  an EGRID, a later phase);
- ``observations/obs.txt`` and ``observations/observations.csv``: ERT's bulk ``SUMMARY``
  form (``keyword``, ``value``, ``error`` = sqrt(variance), ``date``, ``well``), the
  history-matching data only;
- ``observations/held_out.csv``: a ``[validation]`` split's ``cv`` and ``test`` data with
  their roles. They are simulated (their keys are in ``SUMMARY``) but not assimilated: ERT
  assimilates every observation of ``OBS_CONFIG``. The workflows ``workflows/apex_cv`` and
  ``apex_cv_select``, hooked at ``POST_SIMULATION`` and ``POST_EXPERIMENT``, score them with
  APEX's CV criterion and apply the configuration's stopping rule (``APEX_CV``, :mod:`.cv`);
- ``apex_export.json``: what ``APEX_RENDER_DECK`` and the import need (deck template, its
  values, the fields' state variables, the roles).

Phase A1 exported OPM Flow cases of grid fields. adds:

- Eclipse decks: ERT's ``ECLIPSE100``/``ECLIPSE300`` step, the version from ``eclrun_args``,
  ``MAX_RUNNING`` at most the licences;
- scalar state variables: one ``SURFACE APEXSCAL`` (1 x n, ``prior/APEXSCAL_<i>.irap``),
  since ERT's ``GEN_KW`` takes no prior files (since ERT 25); ``APEX_RENDER_DECK`` reads it
  back;
- parameterizations: the state (the latent fields) is exported, and ``APEX_RENDER_DECK``
  maps it to the properties before rendering;
- data transforms: ``GEN_DATA APEX_TRANSFORMED`` with a ``GENERAL_OBSERVATION`` of the
  transformed data, written by the step ``APEX_TRANSFORM`` from the simulated summary.

Cases with field data (no deck simulator produces them), transforms with a
``[validation]`` split and model-error models are refused, with the reason.
"""

import csv
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apex import __version__
from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.services import composition
from apex.simulation.grdecl import write_grdecl

#: ERT field names have at most 8 characters.
MAX_FIELD_NAME = 8
#: The SURFACE that carries the scalar state variables.
SCALARS = "APEXSCAL"
#: The GEN_DATA of the transformed data, written by the step APEX_TRANSFORM.
TRANSFORMED = "APEX_TRANSFORMED"
#: Outlier rejection off: an observation is rejected beyond this many standard deviations.
NO_OUTLIER_REJECTION = 1e6
#: ERT needs a positive STD_CUTOFF; this one drops only responses with no spread at all.
MIN_STD_CUTOFF = 1e-30


@dataclass(frozen=True)
class ExportResult:
    """What :func:`export_case` wrote."""

    case: Path
    realizations: int
    fields: dict[str, str]  # ERT field name -> APEX state variable
    assimilated: int
    held_out: int
    #: How to run the case in ERT as the configuration's own method: the ``apex ert run``
    #: arguments (``esmda``: its inflation as ``--weights``; ``es``: one smoother step).
    run_arguments: tuple[str, ...] = ("--mode", "es_mda")


def run_arguments(config: object) -> tuple[str, ...]:
    """Return ``apex ert run``'s mode and weights for the configuration's method.

    ES-MDA's inflation factors are ERT's ``--weights`` (the release tester was told
    ``4,4,4,4`` for a case of ``[5, 5, 5, 5, 5]``); ES is ERT's ``ensemble_smoother``. Other
    methods have no ERT equivalent: ERT's ES-MDA with its own default weights. ERT rescales
    the weights so that ``sum(1 / w) = 1``; APEX's ES-MDA refuses inflation that does not
    already sum so, so the two agree.
    """
    from apex.domain.config import HistoryMatchConfig

    assert isinstance(config, HistoryMatchConfig)
    if config.method.name == "es":
        return ("--mode", "ensemble_smoother")
    inflation = config.method.options.get("inflation")
    if config.method.name == "esmda" and isinstance(inflation, list | tuple) and inflation:
        return ("--mode", "es_mda", "--weights", ",".join(f"{float(a):.17g}" for a in inflation))
    return ("--mode", "es_mda")


def export_case(
    config_path: Path, out_dir: Path, *, ert_defaults: bool = False, overwrite: bool = False
) -> ExportResult:
    """Write the ERT case of an APEX configuration to ``out_dir`` (see the module docstring).

    Raises
    ------
    ConfigError
        If the configuration is not one phase A1 exports, or ``out_dir`` exists and is not
        empty without ``overwrite``.
    """
    config = composition.load_config(config_path)
    _check_exportable(config)
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise ConfigError(f"{out_dir} is not empty; pass overwrite to replace the case there")
    run = composition.build_run(config)
    workflow = run.workflow
    layout = workflow.layout
    prior = layout.split(composition.draw_prior(run))
    n_members = next(iter(prior.values())).shape[-1]

    gridded = [v for v in layout.variables if v.grid is not None]
    scalars = [v for v in layout.variables if v.grid is None]
    fields = _field_names([v.name for v in gridded])
    for variable in gridded:
        name = fields[variable.name]
        for member in range(n_members):
            write_grdecl(
                out_dir / "prior" / f"{name}_{member}.grdecl", name, prior[variable.name][:, member]
            )
    if scalars:
        # ERT's GEN_KW takes no INIT_FILES (since ERT 25): the scalar variables travel as one
        # SURFACE (1 x n), whose per-realization files are APEX's prior exactly.
        stacked = np.vstack([prior[v.name] for v in scalars])
        for member in range(n_members):
            write_scalars(out_dir / "prior" / f"{SCALARS}_{member}.irap", stacked[:, member])
    grid = tuple(int(n) for n in gridded[0].grid or ()) if gridded else (1, 1, 1)
    eclbase = _eclbase(config.run.name)
    _write_dimens(out_dir / "grid" / f"{eclbase}.GRID", grid)

    observations = workflow.analysis_observations
    roles = workflow.role_rows()
    assimilated = roles["history_matching"]
    held_out = {role: rows for role, rows in roles.items() if role != "history_matching"}
    keys = observations.keys
    transformed = bool(workflow.transforms)
    # The simulator's keys: with data transforms, the raw ones the transform
    # step reads; ERT assimilates the transformed data.
    simulated_keys = workflow.raw_observations.keys
    if transformed:
        (out_dir / "observations").mkdir(parents=True, exist_ok=True)
        (out_dir / "observations" / "transformed.txt").write_text(
            "".join(
                f"{float(v)!r} {math.sqrt(float(e))!r}\n"
                for v, e in zip(observations.values, observations.variances, strict=True)
            )
        )
        (out_dir / "observations" / "obs.txt").write_text(
            f"GENERAL_OBSERVATION {TRANSFORMED} {{ DATA = {TRANSFORMED}; RESTART = 0; "
            "OBS_FILE = transformed.txt; };\n"
        )
    else:
        _write_observations(
            out_dir / "observations" / "observations.csv",
            [keys[i] for i in assimilated],
            observations.values[assimilated],
            observations.variances[assimilated],
        )
        (out_dir / "observations" / "obs.txt").write_text(
            "SUMMARY { VALUES = observations.csv; };\n"
        )
    if held_out:
        _write_held_out(out_dir / "observations" / "held_out.csv", observations, held_out)

    simulator = config.simulator.options
    template = Path(str(simulator["deck_template_file"])).resolve()
    manifest = {
        "written_by": f"apex {__version__} (apex ert export)",
        "config": str(config_path.resolve()),
        "deck_template_file": str(template),
        # The folder its INCLUDEs name (the simulator's case_dir; default the template's).
        "case_dir": str(Path(str(simulator.get("case_dir") or template.parent)).resolve()),
        "template_values": dict(simulator.get("template_values") or {}),
        "eclbase": eclbase,
        "fields": {
            fields[v.name]: {"variable": v.name, "file": f"{fields[v.name].lower()}.grdecl"}
            for v in gridded
        },
        "scalars": {
            "file": f"{SCALARS.lower()}.irap",
            "variables": [[v.name, v.size] for v in scalars],
        }
        if scalars
        else None,
        "parameterized": bool(config.parameterizations),
        "transformed": transformed,
        "roles": {role: len(rows) for role, rows in roles.items()},
    }
    (out_dir / "apex_export.json").write_text(json.dumps(manifest, indent=2) + "\n")

    workers = int(config.execution.options.get("workers", 1))
    simulator_name = config.simulator.name
    if simulator_name == "eclipse":
        licenses = simulator.get("licenses")
        if licenses is not None:
            # The site's licences cap ERT's local queue as they cap APEX's executor.
            workers = min(workers, int(licenses))
    lower, upper = layout.bounds()
    sizes = [v.size for v in layout.variables]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    offsets = dict(zip([v.name for v in layout.variables], starts.tolist(), strict=True))
    lines = [
        f"-- Written by apex {__version__} (apex ert export) from",
        f"-- {config_path.resolve()}",
        f"-- Run, e.g.: ert {' '.join(run_arguments(config)[1:])} case.ert  (ERT 26.x)",
        "-- ERT rescales --weights so that the sum of 1/w is 1, as APEX's ES-MDA requires.",
        "-- ERT runs APEX_RENDER_DECK, so its environment needs APEX with the ert extra",
        "-- (in APEX's folder: uv sync --locked --extra ert).",
        "",
        f"NUM_REALIZATIONS {n_members}",
        f"MIN_REALIZATIONS {math.ceil(100 * config.failures.min_success_fraction)}%",
        "RUNPATH runs/realization-<IENS>/iter-<ITER>",
        "ENSPATH storage",
        f"ECLBASE {eclbase}",
        f"GRID grid/{eclbase}.GRID",
        f"RANDOM_SEED {config.run.seed}",
        "",
        "-- Parameters: APEX's prior (INIT_FILES), written to each run path for the deck.",
    ]
    for variable in gridded:
        name = fields[variable.name]
        offset = offsets[variable.name]
        bounds = ""
        if np.isfinite(lower[offset]):
            bounds += f" MIN:{float(lower[offset])!r}"
        if np.isfinite(upper[offset]):
            bounds += f" MAX:{float(upper[offset])!r}"
        lines.append(
            f"FIELD {name} PARAMETER {name.lower()}.grdecl "
            f"INIT_FILES:prior/{name}_<IENS>.grdecl FORWARD_INIT:False{bounds}"
        )
    if scalars:
        lines.append(
            f"SURFACE {SCALARS} OUTPUT_FILE:{SCALARS.lower()}.irap "
            f"INIT_FILES:prior/{SCALARS}_%d.irap BASE_SURFACE:prior/{SCALARS}_0.irap "
            "FORWARD_INIT:False"
        )
    simulated = sorted({k.datatype for k in simulated_keys})
    lines += ["", "-- Responses: every simulated key (held-out data included, not assimilated)."]
    lines += [f"SUMMARY {datatype}" for datatype in simulated]
    lines += [
        "",
        "OBS_CONFIG observations/obs.txt",
        "",
        "-- APEX's Mako deck, rendered in each run path from the fields above.",
        "FORWARD_MODEL APEX_RENDER_DECK(<EXPORT>=<CONFIG_PATH>/apex_export.json, "
        "<OUTPUT>=<ECLBASE>.DATA)",
        _simulator_step(config),
        *(
            [
                f"GEN_DATA {TRANSFORMED} RESULT_FILE:{TRANSFORMED.lower()}_%d.txt REPORT_STEPS:0",
                "-- The data transforms of APEX, applied to the simulated summary.",
                "FORWARD_MODEL APEX_TRANSFORM(<EXPORT>=<CONFIG_PATH>/apex_export.json, "
                f"<OUTPUT>={TRANSFORMED.lower()}_0.txt)",
            ]
            if transformed
            else []
        ),
        "",
        "QUEUE_SYSTEM LOCAL",
        f"QUEUE_OPTION LOCAL MAX_RUNNING {workers}",
    ]
    if not ert_defaults:
        lines += [
            "",
            "-- As APEX: no outlier rejection, and only exactly constant responses dropped",
            "-- (ERT's defaults reject data beyond 3 sigma and drop spreads below 1e-6).",
            f"ENKF_ALPHA {NO_OUTLIER_REJECTION:g}",
            f"STD_CUTOFF {MIN_STD_CUTOFF:g}",
        ]
        energy = _approx_energy(config)
        if energy is not None:
            lines += [
                "-- As APEX's approx update: truncate the SVD at the same singular-value",
                "-- fraction (ERT's rule is APEX's and PET's: cumulative sum of s, first",
                "-- reaching it); ERT's default is the exact inversion (no truncation).",
                f"ANALYSIS_SET_VAR STD_ENKF ENKF_TRUNCATION {energy!r}",
            ]
    if held_out:
        # APEX's CV criterion inside ERT (phase A2): score each simulated iterate's held-out
        # data, and pick the posterior by the configuration's stopping rule at the end.
        export = (out_dir / "apex_export.json").resolve()
        rule = config.validation.stopping_rule() if config.validation is not None else None
        name = rule.name if rule is not None else "select"
        patience = int(rule.options.get("patience", 2)) if rule is not None else 2
        workflows = out_dir / "workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "apex_cv").write_text(f"APEX_CV {export}\n")
        (workflows / "apex_cv_select").write_text(f"APEX_CV_SELECT {export} {name} {patience}\n")
        lines += [
            "",
            "-- APEX's cross-validation criterion on the held-out data (apex_cv.json).",
            "LOAD_WORKFLOW workflows/apex_cv APEX_CV_SCORE",
            "LOAD_WORKFLOW workflows/apex_cv_select APEX_CV_PICK",
            "HOOK_WORKFLOW APEX_CV_SCORE POST_SIMULATION",
            "HOOK_WORKFLOW APEX_CV_PICK POST_EXPERIMENT",
        ]
    case = out_dir / "case.ert"
    case.write_text("\n".join(lines) + "\n")
    return ExportResult(
        case=case,
        realizations=n_members,
        fields={fields[v.name]: v.name for v in gridded},
        assimilated=len(assimilated),
        held_out=int(sum(len(r) for r in held_out.values())),
        run_arguments=run_arguments(config),
    )


def _approx_energy(config: object) -> float | None:
    """Return the truncation energy of an ``approx`` strategy, else ``None``."""
    from apex.domain.config import HistoryMatchConfig

    assert isinstance(config, HistoryMatchConfig)
    strategy = config.method.strategy
    if strategy is None or strategy.name != "approx":
        return None
    return float(strategy.options.get("energy", 0.98))


def _check_exportable(config: object) -> None:
    from apex.domain.config import HistoryMatchConfig

    assert isinstance(config, HistoryMatchConfig)
    problems = []
    if config.simulator.name not in ("opm_flow", "eclipse"):
        problems.append(
            f"the simulator is {config.simulator.name!r} (export runs the deck simulators "
            "opm_flow and eclipse)"
        )
    elif config.simulator.name == "eclipse" and eclipse_version(config) is None:
        problems.append(
            'ERT\'s ECLIPSE step needs the version: give eclrun_args = ["--version=2023.1"]'
        )
    if config.observations.fields:
        # The deck simulators read summary vectors only (simulation/deck.py read_summary):
        # such a case cannot run on them, in APEX or in ERT.
        problems.append(
            "it has field data ([[observations.fields]]), which the deck simulators do not produce"
        )
    if config.data_transforms and config.validation is not None:
        problems.append(
            "it has [[data_transforms]] and a [validation] split: held-out rows have no place "
            "in the transformed data ERT assimilates"
        )
    if config.model_error is not None:
        problems.append(
            "it has a model-error model ([model_error]): ERT's forward model would "
            "run the simulator alone, without the residuals"
        )
    if config.geophysics is not None:
        problems.append(
            "it has geophysical data ([geophysics]): ERT's forward model would run "
            "the simulator alone, without the geophysical forward model"
        )
    grids = {tuple(v.grid) for v in config.state.variables if v.grid is not None}
    if len(grids) > 1:
        problems.append("the grid fields have different grids")
    if problems:
        raise ConfigError("ERT export cannot export this case: " + "; ".join(problems))


def eclipse_version(config: object) -> str | None:
    """Return the Eclipse version of an ``eclipse`` case's ``eclrun_args``, if given."""
    from apex.domain.config import HistoryMatchConfig

    assert isinstance(config, HistoryMatchConfig)
    arguments = [str(a) for a in config.simulator.options.get("eclrun_args", ())]
    for i, argument in enumerate(arguments):
        if argument.startswith("--version="):
            return argument.split("=", 1)[1]
        if argument == "--version" and i + 1 < len(arguments):
            return arguments[i + 1]
    return None


def _simulator_step(config: object) -> str:
    """Return the case's simulator step: ``FLOW``, or ERT's ``ECLIPSE100``/``ECLIPSE300``."""
    from apex.domain.config import HistoryMatchConfig

    assert isinstance(config, HistoryMatchConfig)
    if config.simulator.name == "opm_flow":
        return "FORWARD_MODEL FLOW"
    step = "ECLIPSE300" if config.simulator.options.get("version") == "e300" else "ECLIPSE100"
    return f"FORWARD_MODEL {step}(<VERSION>={eclipse_version(config)}, <NUM_CPU>=1)"


def write_scalars(path: Path, values: np.ndarray) -> None:
    """Write scalar state values as a 1 x n binary Irap surface.

    Binary, as ERT prefers (ASCII surfaces are deprecated since ERT 25): exact to the float32
    ERT stores, where ASCII keeps about six significant digits.
    """
    from surfio import IrapHeader, IrapSurface

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(values.size)
    header = IrapHeader(
        ncol=n, nrow=1, xori=0.0, yori=0.0, xinc=1.0, yinc=1.0, rot=0.0, xrot=0.0, yrot=0.0
    )
    surface = IrapSurface(header=header, values=np.asarray(values, dtype=np.float32).reshape(n, 1))
    surface.to_binary_file(path)


def read_scalars(path: Path) -> np.ndarray:
    """Read the values ``(n,)`` of a surface written by :func:`write_scalars` (or ASCII)."""
    from surfio import IrapSurface

    try:
        surface = IrapSurface.from_binary_file(path)
    except (OSError, ValueError, RuntimeError):
        surface = IrapSurface.from_ascii_file(path)  # an ERT that writes ASCII
    return np.asarray(surface.values, dtype=np.float64).ravel()


def _field_names(variables: list[str]) -> dict[str, str]:
    """Return an ERT field name per state variable: upper case, at most 8 characters."""
    names: dict[str, str] = {}
    for variable in variables:
        base = re.sub(r"[^A-Za-z0-9]", "", variable).upper()[:MAX_FIELD_NAME] or "FIELD"
        name, n = base, 1
        while name in names.values():
            suffix = str(n)
            name, n = base[: MAX_FIELD_NAME - len(suffix)] + suffix, n + 1
        names[variable] = name
    return names


def _eclbase(run_name: str) -> str:
    """Return an ERT ``ECLBASE``: upper case letters, digits and underscores."""
    return re.sub(r"[^A-Za-z0-9_]", "_", run_name).upper() or "CASE"


def _write_dimens(path: Path, grid: tuple[int, ...]) -> None:
    """Write an unformatted Eclipse file with one keyword, ``DIMENS`` ``(nx, ny, nz)``.

    Fortran sequential records, big-endian: a header (8-character name, count, ``INTE``)
    and the data, each framed by its length.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"DIMENS  " + struct.pack(">i", len(grid)) + b"INTE"
    data = struct.pack(f">{len(grid)}i", *grid)
    with path.open("wb") as handle:
        for record in (header, data):
            handle.write(struct.pack(">i", len(record)) + record + struct.pack(">i", len(record)))


def _split_key(key: DataKey) -> tuple[str, str]:
    """Return ERT's ``keyword`` and ``well`` of an APEX summary key ``VECTOR:WELL``."""
    keyword, _, well = key.datatype.partition(":")
    return keyword, well


def _write_observations(
    path: Path, keys: list[DataKey], values: np.ndarray, variances: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["keyword", "value", "error", "date", "well"])
        for key, value, variance in zip(keys, values, variances, strict=True):
            keyword, well = _split_key(key)
            writer.writerow(
                [keyword, repr(float(value)), repr(math.sqrt(float(variance))), key.time, well]
            )


def _write_held_out(path: Path, observations: object, roles: dict[str, np.ndarray]) -> None:
    from apex.domain.observations import ObservationSet

    assert isinstance(observations, ObservationSet)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "keyword", "value", "error", "date", "well"])
        for role, rows in sorted(roles.items()):
            for i in rows:
                key = observations.keys[i]
                keyword, well = _split_key(key)
                writer.writerow([
                    role, keyword, repr(float(observations.values[i])),
                    repr(math.sqrt(float(observations.variances[i]))), key.time, well,
                ])  # fmt: skip
