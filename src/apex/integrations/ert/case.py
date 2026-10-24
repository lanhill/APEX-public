"""APEX history matching of an ERT case (``APEX_HISTORY_MATCH``).

An ERT user keeps their case (``case.ert``: fields, observations, forward model) and adds
an ``apex.toml`` with APEX's method and, optionally, its validation split, metrics, output
and prior. APEX runs its own loop; each forecast is one ``ert ensemble_experiment`` of the
case's forward model on APEX's parameters (the ``ert`` simulator and executor), so APEX's
methods, localizations, CV and checkpoints apply unchanged and resume is bit-identical.

From the case (ERT's parsed configuration, :func:`read_ert_case`):

- each ``FIELD`` is a state variable (lower-case name, ``MIN``/``MAX`` as bounds, its grid);
  its prior is the realizations' ``INIT_FILES`` unless ``apex.toml`` has a ``[prior]``;
- each observation is a datum (``variance = error**2``, ERT's effective error): summary
  observations as declared, and general (``GEN_DATA``), RFT, breakthrough and seismic ones
  from ERT's own tables (:mod:`.observations`), whose responses are then read from each
  batch's storage;
- ``NUM_REALIZATIONS`` is the ensemble size, ``MAX_RUNNING`` the workers and ``RANDOM_SEED``
  the seed (unless ``apex.toml`` sets ``[run] seed``).

:func:`write_apex_case` writes the inputs and the configuration into
``<case>/apex_runs/<name>.inputs/``; the run goes to ``<case>/apex_runs/<name>/``.

Several methods: ``apex.toml`` may list ``[methods.<label>]`` (each a complete
``[method]``, auxiliary techniques included) instead of one ``[method]``. Each runs, with the
same seed and prior, into ``<case>/apex_runs/<run>/<label>/`` (inputs beside it, in
``<label>.inputs/``) and is written back as the experiment ``apex:<run>:<label>``
(:func:`history_match_all`).
"""

import csv
import logging
import re
import sys
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from numpy.typing import NDArray

import apex.integrations.ert.forward_init  # noqa: F401  (registers the ert_forward_init prior)
from apex.core.errors import ConfigError
from apex.core.progress import ProgressReporter
from apex.domain.results import HistoryMatchResult
from apex.integrations.ert.observations import SUMMARY, case_observations
from apex.simulation.grdecl import read_grdecl

#: Sections ``apex.toml`` may not have: they come from the ERT case.
FROM_THE_CASE = ("state", "simulator", "observations")
#: The program that reads a batch's responses of every observation type.
OBSERVATIONS_PROGRAM = "apex.integrations.ert.observations"
#: The forward-model step that writes what ERT would write of the parameters.
PARAMETER_STEP = "APEX_WRITE_PARAMETERS"
#: The joint prior generator of FORWARD_INIT:True parameters (:mod:`.forward_init`).
FORWARD_INIT_PRIOR = "ert_forward_init"
#: Where the runs go, next to the case.
RUNS = "apex_runs"
#: A method entry's label: one folder name (the rule for study names).
LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ErtField:
    """One ``FIELD`` of an ERT case (cells in Eclipse order, ``nx * ny * nz``)."""

    name: str
    grid: tuple[int, int, int]
    output_file: str
    init_files: str | None
    lower: float | None = None
    upper: float | None = None
    #: ERT's ``file_format`` of the field's files (``grdecl``, ``bgrdecl``, ``roff``, ...).
    file_format: str = "grdecl"


@dataclass(frozen=True)
class ErtParameter:
    """A parameter the batch's ``APEX_WRITE_PARAMETERS`` step writes.

    ``GEN_KW`` groups (latent standard-normal values, one per key in the case's order),
    surfaces (``ncol * nrow`` values) and fields in ROFF or binary GRDECL or with transforms
    (flat, Eclipse order, in ERT's stored space: after its ``INIT_TRANSFORM``).
    """

    name: str
    kind: str
    size: int
    grid: tuple[int, ...] | None = None
    lower: float | None = None
    upper: float | None = None
    #: The prior ``(size, n_realizations)``, from INIT_FILES or ERT's sampler; ``None`` for
    #: ``FORWARD_INIT:True``, whose prior the forward model makes.
    prior: NDArray[np.float64] | None = None
    forward_init: bool = False


@dataclass(frozen=True)
class ErtObservation:
    """One observation: ``key`` (``VECTOR:WELL``, or a response key), time, value, error.

    ``date`` is a summary's date, or for other types (``kind``) ERT's index string of the
    observation (``"0, 42"``: report step and index of a ``GEN_DATA`` response).
    """

    key: str
    date: str
    value: float
    error: float
    kind: str = "summary"


@dataclass(frozen=True)
class ErtCaseInfo:
    """What APEX takes from an ERT case."""

    case_file: Path
    realizations: int
    eclbase: str
    fields: list[ErtField]
    observations: list[ErtObservation]
    max_running: int = 1
    seed: int | None = None
    skipped: list[str] = field(default_factory=list)
    #: The case's ERT storage (``ENSPATH``), where a run is written back.
    ens_path: Path | None = None
    #: Where a batch's responses are read: ``summary`` files, or (other observation types)
    #: the batch's ERT storage.
    responses: str = "summary"
    #: Parameters written by the batch's parameter step (GEN_KW, SURFACE, other FIELDs).
    parameters: list[ErtParameter] = field(default_factory=list)


def read_ert_case(case_file: Path) -> ErtCaseInfo:
    """Read an ERT case with ERT's own parser (and its site plugins, APEX's included).

    Raises
    ------
    ConfigError
        If ERT refuses the case, or it has parameters or observations APEX cannot take.
    """
    from ert.config import Field, GenKwConfig, SurfaceConfig

    from apex.integrations.ert.config import parse_case

    case_file = case_file.resolve()
    try:
        config = parse_case(case_file)
    except Exception as error:  # ERT's configuration errors have no common base
        raise ConfigError(f"ERT cannot read {case_file}: {error}") from error
    realizations = int(config.runpath_config.num_realizations)
    if config.analysis_config.design_matrix is not None:
        raise ConfigError(f"{case_file}: a DESIGN_MATRIX is not taken by APEX_HISTORY_MATCH")
    fields: list[ErtField] = []
    parameters: list[ErtParameter] = []
    groups: dict[str, list[Any]] = {}
    for name, parameter in config.ensemble_config.parameter_configs.items():
        if isinstance(parameter, GenKwConfig):
            groups.setdefault(str(parameter.group_name), []).append(parameter)
        elif isinstance(parameter, Field) and _direct(parameter):
            fields.append(
                ErtField(
                    name=name,
                    grid=(int(parameter.nx), int(parameter.ny), int(parameter.nz)),
                    output_file=str(parameter.output_file),
                    init_files=str(parameter.forward_init_file) or None,
                    lower=parameter.truncation_min,
                    upper=parameter.truncation_max,
                    file_format=str(parameter.file_format.value),
                )
            )
        elif isinstance(parameter, (Field, SurfaceConfig)):
            parameters.append(_step_parameter(case_file, parameter, realizations))
        else:
            raise ConfigError(
                f"{case_file}: parameter {name} is a {type(parameter).__name__}; APEX takes "
                "FIELD, SURFACE and GEN_KW parameters"
            )
    for group, keys in groups.items():
        prior = None
        if config.random_seed is not None:
            # ERT's own prior: each key's standard-normal draw from RANDOM_SEED.
            draws = [
                np.asarray(k.sample_values(str(config.random_seed), list(range(realizations)),
                                           realizations), dtype=np.float64)
                for k in keys
            ]  # fmt: skip
            prior = np.ascontiguousarray(np.stack(draws))
        parameters.append(ErtParameter(group, "gen_kw", len(keys), prior=prior))
    observations = [
        # Summary observations keep their declared values (ERT's tables are float32).
        ErtObservation(
            key=str(declaration.key),
            date=str(declaration.date),
            value=float(declaration.value),
            error=float(declaration.error),
        )
        for declaration in config.observation_declarations
        if declaration.type == "summary_observation"
    ]
    if any(d.type != "summary_observation" for d in config.observation_declarations):
        # General (GEN_DATA), RFT, breakthrough and seismic observations: ERT's own tables
        # and index strings; their responses are read from each batch's storage.
        observations.extend(
            ErtObservation(key.datatype, key.time, value, error, kind=kind)
            for kind, key, value, error in case_observations(config)
            if kind != SUMMARY
        )
    return ErtCaseInfo(
        case_file=case_file,
        realizations=int(config.runpath_config.num_realizations),
        eclbase=str(config.runpath_config.summary_file_base_name),
        fields=fields,
        observations=observations,
        max_running=int(config.queue_config.max_running or 1),
        seed=None if config.random_seed is None else int(config.random_seed),
        ens_path=Path(config.ens_path).resolve(),
        responses="summary" if all(o.kind == SUMMARY for o in observations) else "storage",
        parameters=parameters,
    )


def _direct(field: Any) -> bool:  # noqa: ANN401 (ERT's Field)
    """Whether the ert simulator writes a FIELD itself: ASCII GRDECL without transforms.

    Such fields reach the forward model at full precision; the others go through ERT's own
    writer in the batch's parameter step.
    """
    return (
        str(field.file_format.value) == "grdecl"
        and field.input_transformation is None
        and field.output_transformation is None
        and not field.forward_init
    )


def _step_parameter(case_file: Path, parameter: Any, realizations: int) -> ErtParameter:  # noqa: ANN401
    """Describe a SURFACE or FIELD the parameter step writes, with its prior (ERT's readers)."""
    from ert.config import Field
    from ert.config.field import field_transform
    from ert.field_utils import read_field
    from surfio import IrapSurface

    if isinstance(parameter, Field):
        shape = (int(parameter.nx), int(parameter.ny), int(parameter.nz))
        grid: tuple[int, ...] | None = shape
        size = shape[0] * shape[1] * shape[2]
        # MIN/MAX bound the state only when they bound it as written: with an output
        # transform, ERT truncates the transformed values (as the step does).
        bounded = parameter.output_transformation is None
        lower = parameter.truncation_min if bounded else None
        upper = parameter.truncation_max if bounded else None
        kind = "field"
    else:
        grid, size, lower, upper, kind = (
            None,
            int(parameter.ncol * parameter.nrow),
            None,
            None,
            "surface",
        )
    if parameter.forward_init:
        return ErtParameter(parameter.name, kind, size, grid, lower, upper, None, True)
    columns = []
    for i in range(realizations):
        name = parameter.forward_init_file.replace("<IENS>", str(i)).replace("%d", str(i))
        path = Path(name) if Path(name).is_absolute() else case_file.parent / name
        try:
            if kind == "field":
                assert grid is not None
                values = read_field(path, parameter.name, grid)
                # As ERT stores it: after INIT_TRANSFORM; flat in Eclipse order (i fastest).
                stored = np.asarray(field_transform(values, parameter.input_transformation))
                columns.append(stored.astype(np.float64).ravel(order="F"))
            else:
                surface = (
                    IrapSurface.from_binary_file(path)
                    if parameter.file_format == "binary"
                    else IrapSurface.from_ascii_file(path)
                )
                columns.append(np.asarray(surface.values, dtype=np.float64).ravel())
        except (OSError, ValueError) as error:
            raise ConfigError(f"{parameter.name}, realization {i}: {error}") from error
    prior = np.ascontiguousarray(np.stack(columns, axis=-1))
    if prior.shape[0] != size:
        raise ConfigError(f"{parameter.name}: {prior.shape[0]} values, expected {size}")
    return ErtParameter(parameter.name, kind, size, grid, lower, upper, prior)


@dataclass(frozen=True)
class PreparedCase:
    """An ERT case written as an APEX run: its configuration and run directory."""

    config: Path
    run_dir: Path


@dataclass(frozen=True)
class MethodRun:
    """One method of ``apex.toml``, as a run.

    Attributes
    ----------
    label
        The ``[methods.<label>]`` entry, or ``None`` for a single ``[method]``.
    choices
        ``apex.toml``'s choices with this entry as ``[method]``.
    name
        The run's name (``<run>:<label>`` for an entry; ``apex:<name>`` in ERT's storage).
    run_dir
        Where the run goes.
    inputs
        Where its inputs go.
    """

    label: str | None
    choices: dict[str, Any]
    name: str
    run_dir: Path
    inputs: Path


def method_runs(
    info: ErtCaseInfo, apex_file: Path, only: Sequence[str] | None = None
) -> list[MethodRun]:
    """Return the runs ``apex.toml`` asks for: its ``[method]``, or each ``[methods.<label>]``.

    Parameters
    ----------
    info
        The case.
    apex_file
        ``apex.toml``.
    only
        Labels to run (``--methods``); all by default. Only with ``[methods]``.

    Raises
    ------
    ConfigError
        If ``apex.toml`` cannot be read, has both or neither of ``[method]`` and
        ``[methods]``, an entry is not a table or has a bad label, or ``only`` names an
        unknown entry.
    """
    choices = _read_choices(apex_file)
    runs = info.case_file.parent / RUNS
    run = dict(choices.get("run", {}))
    entries = choices.get("methods")
    if entries is None:
        if only:
            raise ConfigError(f"{apex_file}: --methods needs [methods.<label>] entries")
        name = _run_name(info, run, choices)
        return [MethodRun(None, choices, name, runs / name, runs / f"{name}.inputs")]
    if "method" in choices:
        raise ConfigError(f"{apex_file}: give either [method] or [methods.<label>], not both")
    if not isinstance(entries, dict) or not entries:
        raise ConfigError(f"{apex_file}: [methods] needs at least one [methods.<label>] table")
    unknown = sorted(set(only or ()) - set(entries))
    if unknown:
        raise ConfigError(f"{apex_file}: no [methods.<label>] for {unknown}; has {sorted(entries)}")
    base = str(run.get("name") or info.case_file.stem)
    result = []
    for label, entry in entries.items():
        if not LABEL.match(label) or label in (".", ".."):
            raise ConfigError(f"{apex_file}: method label {label!r} is not one folder name")
        if not isinstance(entry, dict):
            raise ConfigError(f"{apex_file}: [methods.{label}] must be a table")
        if only and label not in only:
            continue
        own = {k: v for k, v in choices.items() if k != "methods"}
        own["method"] = entry
        own["run"] = {**run, "name": f"{base}:{label}"}
        result.append(
            MethodRun(
                label, own, f"{base}:{label}", runs / base / label, runs / base / f"{label}.inputs"
            )
        )
    return result


def _read_choices(apex_file: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(apex_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read {apex_file}: {error}") from error


def write_apex_case(
    info: ErtCaseInfo, apex_file: Path, method: MethodRun | None = None
) -> PreparedCase:
    """Write the APEX inputs and configuration of ``info`` with ``apex.toml``'s choices.

    Rewriting is allowed (a resumed run reads the same inputs again): the inputs are
    derived from the case and ``apex.toml`` alone.

    Parameters
    ----------
    info
        The case.
    apex_file
        ``apex.toml``.
    method
        One of :func:`method_runs`; default: ``apex.toml``'s single ``[method]``.

    Raises
    ------
    ConfigError
        If ``apex.toml`` sets what the case defines, or the prior or data cannot be read.
    """
    apex_file = apex_file.resolve()
    unsupported = [f"{f.name} ({f.file_format})" for f in info.fields if f.file_format != "grdecl"]
    if unsupported:
        # The ert simulator writes ASCII GRDECL: under a ROFF or binary name, the forward
        # model would read a file in the wrong format.
        raise ConfigError(
            f"{info.case_file}: APEX writes FIELDs as ASCII GRDECL; not {', '.join(unsupported)}"
            " (use a .grdecl output file and INIT_FILES)"
        )
    if method is None:
        runs = method_runs(info, apex_file)
        if runs[0].label is not None:
            raise ConfigError(f"{apex_file}: with [methods], prepare one method at a time")
        method = runs[0]
    choices: dict[str, Any] = dict(method.choices)
    # With [[parameterizations]], the case's FIELDs are the properties the
    # forward model reads, and the state they come from, with its prior, is apex.toml's.
    parameterized = bool(choices.get("parameterizations"))
    from_case = [s for s in FROM_THE_CASE if not (parameterized and s == "state")]
    given = [s for s in from_case if s in choices]
    if parameterized and ("state" not in choices or "prior" not in choices):
        raise ConfigError(
            f"{apex_file}: with [[parameterizations]], give the state they map ([state]) and "
            "its prior ([prior]); the case's FIELDs are their properties"
        )
    if given:
        raise ConfigError(
            f"{apex_file}: [{'], ['.join(given)}] come from the ERT case; remove them"
        )
    execution = dict(choices.pop("execution", {}))
    if execution.get("name", "ert") != "ert":
        raise ConfigError(f"{apex_file}: an ERT case runs on the 'ert' executor")
    run = dict(choices.pop("run", {}))
    name = method.name
    if "seed" not in run:
        if info.seed is None:
            raise ConfigError(f"{apex_file}: set [run] seed (the case has no RANDOM_SEED)")
        run["seed"] = info.seed
    inputs = method.inputs
    inputs.mkdir(parents=True, exist_ok=True)
    variables = {f.name: f.name.lower() for f in info.fields}

    state = choices.pop("state", None)
    prior = choices.pop("prior", None)
    ert_options = dict(choices.pop("ert", {}))
    unknown = sorted(set(ert_options) - {"forward_init_steps"})
    if unknown:
        raise ConfigError(f"{apex_file}: [ert] takes forward_init_steps only, not {unknown}")
    forward_init = [p.name for p in info.parameters if p.forward_init]
    steps = ert_options.get("forward_init_steps")
    if forward_init and (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1):
        # Needed with any prior: the steps that make them must not run in the batches,
        # where they would overwrite APEX's values.
        raise ConfigError(
            f"{apex_file}: {', '.join(forward_init)} have FORWARD_INIT:True; say how many of "
            "the case's first FORWARD_MODEL steps make them: [ert] forward_init_steps = N"
        )
    if prior is None:
        prior = _prior(info, inputs, ert_options, apex_file)
    prior.setdefault("ensemble_size", info.realizations)
    _write_observations(inputs / "observations.csv", info.observations)

    config = {
        "run": {**run, "name": name, "output_dir": str(method.run_dir)},
        "state": state or {
            "variables": [
                {
                    "name": variables[f.name], "size": int(np.prod(f.grid)), "grid": list(f.grid),
                    **({"lower": f.lower} if f.lower is not None else {}),
                    **({"upper": f.upper} if f.upper is not None else {}),
                }
                for f in info.fields
            ] + [
                {
                    "name": p.name.lower(), "size": p.size,
                    **({"grid": list(p.grid)} if p.grid is not None else {}),
                    **({"lower": p.lower} if p.lower is not None else {}),
                    **({"upper": p.upper} if p.upper is not None else {}),
                }
                for p in info.parameters
            ]
        },
        "prior": prior,
        "observations": {"file": "observations.csv"},
        "case": choices.pop("case", {"kind": "real"}),
        "simulator": {
            "name": "ert",
            "options": {
                "case_file": str(info.case_file),
                "eclbase": info.eclbase,
                "responses": info.responses,
                **(
                    {"results_program": [sys.executable, "-m", OBSERVATIONS_PROGRAM]}
                    if info.responses == "storage"
                    else {}
                ),
                "fields": {
                    variables[f.name]: {"keyword": f.name, "file": f.output_file}
                    for f in info.fields
                },
                **(
                    {
                        "step_parameters": {p.name.lower(): p.name for p in info.parameters},
                        "parameter_step": f"{PARAMETER_STEP}(<CASE>={info.case_file})",
                        **(
                            {"skip_forward_steps": int(ert_options["forward_init_steps"])}
                            if any(p.forward_init for p in info.parameters)
                            else {}
                        ),
                    }
                    if info.parameters
                    else {}
                ),
            },
        },
        "execution": {
            "name": "ert",
            "options": {"workers": info.max_running, **execution.get("options", {})},
        },
        **choices,
    }  # fmt: skip
    path = inputs / "config.toml"
    path.write_text(
        f"# Written by APEX_HISTORY_MATCH from {info.case_file} and {apex_file}.\n"
        + tomli_w.dumps(_without_none(config))
    )
    return PreparedCase(path, method.run_dir)


def _prior(
    info: ErtCaseInfo, inputs: Path, ert_options: dict[str, Any], apex_file: Path
) -> dict[str, Any]:
    """Return the case's own prior: INIT_FILES and ERT's sampler, and FORWARD_INIT's."""
    arrays: dict[str, Any] = dict(_init_files(info))
    generated = []
    for parameter in info.parameters:
        if parameter.forward_init:
            generated.append(parameter.name.lower())
        elif parameter.prior is None:
            raise ConfigError(
                f"{apex_file}: GEN_KW {parameter.name} needs a seed for ERT's sampler: set "
                "RANDOM_SEED in the case, or give apex.toml a [prior]"
            )
        else:
            arrays[parameter.name.lower()] = parameter.prior
    prior: dict[str, Any] = {}
    if arrays:
        np.savez(inputs / "prior.npz", **arrays)
        prior["generator"] = {"name": "file", "options": {"ensemble_file": "prior.npz"}}
    if generated:
        steps = int(ert_options["forward_init_steps"])  # checked by write_apex_case
        prior["groups"] = {
            "forward_init": {
                "name": FORWARD_INIT_PRIOR,
                "variables": generated,
                "options": {"case_file": str(info.case_file), "steps": steps,
                            "work_dir": str(inputs / "forward_init")},
            }
        }  # fmt: skip
    return prior


def run_directory(info: ErtCaseInfo, apex_file: Path) -> Path:
    """Return where the run of ``info`` with ``apex_file`` goes (``<case>/apex_runs/<name>``).

    Raises
    ------
    ConfigError
        If ``apex.toml`` has ``[methods]``: each has its own (:func:`method_runs`).
    """
    runs = method_runs(info, apex_file)
    if runs[0].label is not None:
        raise ConfigError(f"{apex_file}: with [methods], each method has its own run directory")
    return runs[0].run_dir


def _run_name(info: ErtCaseInfo, run: Mapping[str, Any], choices: Mapping[str, Any]) -> str:
    method = choices.get("method", {}).get("name", "apex")
    return str(run.get("name") or f"{info.case_file.stem}_{method}")


def _init_files(info: ErtCaseInfo) -> dict[str, NDArray[np.float64]]:
    """Return each field's prior ``(n_cells, n_realizations)`` from its ``INIT_FILES``."""
    prior = {}
    for f in info.fields:
        if not f.init_files or not re.search(r"<IENS>|%d", f.init_files):
            raise ConfigError(
                f"FIELD {f.name} has no per-realization INIT_FILES (<IENS> or %d): give "
                "apex.toml a [prior]"
            )
        columns = []
        for i in range(info.realizations):
            file = f.init_files.replace("<IENS>", str(i)).replace("%d", str(i))
            path = Path(file) if Path(file).is_absolute() else info.case_file.parent / file
            try:
                columns.append(read_grdecl(path, f.name))
            except (OSError, ValueError) as error:
                raise ConfigError(f"FIELD {f.name}, realization {i}: {error}") from error
            if columns[-1].size != np.prod(f.grid):
                raise ConfigError(
                    f"{path} has {columns[-1].size} values; FIELD {f.name} has "
                    f"{int(np.prod(f.grid))} cells"
                )
        prior[f.name.lower()] = np.ascontiguousarray(np.stack(columns, axis=-1))
    return prior


def _write_observations(path: Path, observations: list[ErtObservation]) -> None:
    """Write APEX's long-format CSV (``time,datatype,value,variance``)."""
    if not observations:
        raise ConfigError("the ERT case has no observations to history-match")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "datatype", "value", "variance"])
        for o in observations:
            # ERT standardises dates to ISO date-times; APEX's report times are dates.
            date = o.date.removesuffix("T00:00:00") if o.kind == SUMMARY else o.date
            writer.writerow([date, o.key, repr(o.value), repr(o.error**2)])


def _without_none(value: Any) -> Any:  # noqa: ANN401 (TOML values)
    """Drop ``None`` (TOML has no null)."""
    if isinstance(value, Mapping):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


class MethodsFailed(ConfigError):
    """Some of ``apex.toml``'s methods failed; the others ran (and were written back)."""

    def __init__(
        self, failed: Mapping[str, str], results: Mapping[str, HistoryMatchResult]
    ) -> None:
        self.failed = dict(failed)
        self.results = dict(results)
        reasons = "; ".join(f"{label}: {reason}" for label, reason in self.failed.items())
        super().__init__(
            f"methods failed ({len(self.failed)} of {len(failed) + len(results)}): {reasons}"
        )


def history_match(
    case_file: Path,
    apex_file: Path | None = None,
    *,
    max_steps: int | None = None,
    progress: ProgressReporter | None = None,
    write_back: bool = True,
    storage: Any = None,  # noqa: ANN401 (an open ERT storage)
) -> HistoryMatchResult:
    """Run (or continue) APEX's history matching of an ERT case with one ``[method]``.

    ``apex_file`` defaults to ``apex.toml`` next to the case. A run directory that already
    has a checkpointed step is resumed with its stored configuration (bit-identical to an
    uninterrupted run); otherwise the inputs are written and the run starts.

    With ``write_back`` (the default), the run's iterates are then added to the case's ERT
    storage (:mod:`.writeback`): to ``storage`` if given (ERT's own, inside ``ert
    workflow``), else to the case's ``ENSPATH``, opened for writing.

    Raises
    ------
    ConfigError
        If the case or ``apex.toml`` cannot be used, or ``apex.toml`` lists ``[methods]``
        (use :func:`history_match_all`).
    """
    results = history_match_all(
        case_file, apex_file, max_steps=max_steps, progress=progress, write_back=write_back,
        storage=storage, single=True,
    )  # fmt: skip
    return next(iter(results.values()))


def history_match_all(
    case_file: Path,
    apex_file: Path | None = None,
    *,
    methods: Sequence[str] | None = None,
    max_steps: int | None = None,
    progress: ProgressReporter | None = None,
    write_back: bool = True,
    storage: Any = None,  # noqa: ANN401 (an open ERT storage)
    single: bool = False,
) -> dict[str | None, HistoryMatchResult]:
    """Run (or continue) every method of ``apex.toml`` on an ERT case, in order.

    Each method is a run of its own (see :func:`history_match` for one), written back as
    ``apex:<name>``. A method that fails does not stop the others.

    Parameters
    ----------
    methods
        Labels of ``[methods]`` to run (all by default).
    single
        Refuse ``[methods]`` (for :func:`history_match`).

    Returns
    -------
    dict
        Each method's result, by label (``None`` for a single ``[method]``).

    Raises
    ------
    MethodsFailed
        After the others ran, if a method failed; it names them and holds the results.
    ConfigError
        If the case or ``apex.toml`` cannot be used.
    """
    from apex.services.composition import load_config, resume, start_config

    case_file = case_file.resolve()
    apex_file = (case_file.parent / "apex.toml" if apex_file is None else apex_file).resolve()
    info = read_ert_case(case_file)
    runs = method_runs(info, apex_file, methods)
    if single and runs[0].label is not None:
        raise ConfigError(
            f"{apex_file}: has [methods]; run them all with history_match_all "
            "(apex ert history-match, or the APEX_HISTORY_MATCH workflow)"
        )
    results: dict[str | None, HistoryMatchResult] = {}
    failed: dict[str, str] = {}
    for run in runs:
        try:
            if (run.run_dir / "state.json").exists():
                # A checkpointed run goes on with the inputs it started from.
                result = resume(run.run_dir, max_steps, progress)
            else:
                prepared = write_apex_case(info, apex_file, run)
                result = start_config(load_config(prepared.config), max_steps, progress)
            if write_back:
                _write_back(info, run.run_dir, storage)
        # One method's failure must not stop the others; it is reported at the end.
        except Exception as error:
            if run.label is None:
                raise
            logger.exception(
                "method_failed", extra={"method": run.label, "run_dir": str(run.run_dir)}
            )
            failed[run.label] = f"{type(error).__name__}: {error}"
            continue
        results[run.label] = result
    if failed:
        raise MethodsFailed(failed, {k: v for k, v in results.items() if k is not None})
    return results


def _write_back(info: ErtCaseInfo, run_dir: Path, storage: Any) -> None:  # noqa: ANN401
    from apex.integrations.ert.writeback import write_back

    if storage is not None:
        write_back(storage, info.case_file, run_dir)
        return
    assert info.ens_path is not None  # read_ert_case sets it
    with open_for_writing(Path(info.ens_path), info.case_file) as opened:
        write_back(opened, info.case_file, run_dir)


@contextmanager
def open_for_writing(path: Path, case_file: Path | None = None) -> Iterator[Any]:
    """Open an ERT storage to write in, migrated to the installed ERT's version first.

    ERT opens an older storage for writing without migrating it, and what is then written
    is in the new format under the old version's index, which a later migration damages.
    ERT's own command line, GUI and run models migrate first; so does APEX. ``case_file``
    is parsed, only for a migration, since ERT's migrations from very old storages read
    the case's configuration.
    """
    from ert.storage import local_storage_set_ert_config, open_storage
    from ert.storage.local_storage import LocalStorage

    from apex.integrations.ert.config import parse_case

    if LocalStorage.check_migration_needed(path):
        logger.info("migrating ERT storage", extra={"path": str(path)})
        if case_file is not None:
            local_storage_set_ert_config(parse_case(case_file))
        LocalStorage.perform_migration(path)
    with open_storage(path, mode="w") as opened:
        yield opened
