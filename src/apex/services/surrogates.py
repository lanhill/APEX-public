"""Surrogate models' use cases: loading a model, building one, checking one.

**Loading** (§2): the plugin named in the manifest is created from the registry
``surrogate_models`` through its ``from_arrays`` (pure: the plugin's own manifest and arrays
in, the fitted model out). The manifest's options are checked against the plugin's declared
options as a configuration's are, and the model must answer exactly the keys the
manifest records.

**Building** (§7, ``apex surrogate build``): a build's configuration names a
history-matching case; its simulator stack, executor and parameterizations are composed
here (:func:`surrogate_case`), the design drawn from the build's stream ``design``
(:func:`draw_design`), the full runs made and the model fitted, bounded and gated by
``workflows/surrogate.py``, and the files written in the build's directory
(``storage/surrogate_directory.py``). A build resumes where it stopped; one whose plugin,
options or gate changed refits on the runs already made.

**Checking** (§7, ``apex surrogate check``): an existing model gated on new full runs, from
a case's prior or a run's posterior; the model file never changes, the verdict goes to
``checks/<id>/``.

**History matching** (§5, phase 3): :func:`surrogate_forecasting` composes what a run with
``[surrogate]`` holds (``workflows/surrogate_forecasts.py``) after the refusals
(:func:`refusals`, :func:`refuse_model`): the model, its local executor, the checks'
tolerances and what a refit needs. Workers load the model by reference
(:func:`fitted_model`, once per process).
"""

import datetime as dt
import functools
import hashlib
import json
import logging
import math
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from apex.algorithms.surrogates.designs import (
    BUILTIN_PLANS,
    ensemble_design,
    lhs_design,
    repeated_columns,
    sampled_design,
)
from apex.algorithms.surrogates.domain import check_members, check_request, refuse_outside
from apex.algorithms.surrogates.gate import gate
from apex.core.errors import (
    ApexError,
    ConfigError,
    PluginError,
    UnknownPluginError,
    WorkflowError,
    validation_problems,
)
from apex.core.plugins import executors, priors, surrogate_designs, surrogate_models
from apex.core.progress import ProgressReporter
from apex.core.simulation import Executor, Simulator
from apex.core.surrogates import (
    AdaptiveDesign,
    BuiltinDesign,
    CoarseCorrectedSurrogate,
    SurrogateModel,
)
from apex.domain.config import HistoryMatchConfig, PluginSection, SurrogateRunSection
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import (
    BuildCost,
    CoarseSection,
    DomainSummary,
    FitRecord,
    GateResult,
    GateTolerances,
    SnapshotProvenance,
    SnapshotReference,
    SnapshotSet,
    SurrogateBuildPlan,
    SurrogateBuildSummary,
    SurrogateCheckSummary,
    SurrogateConfig,
    SurrogateManifest,
    SurrogateSeries,
    TrainingDomain,
    key_document,
    time_position,
)
from apex.services import composition
from apex.services.identity import code_version, tree_hash
from apex.services.plugins import check_options
from apex.simulation.surrogate import SurrogateSimulator
from apex.storage import provenance
from apex.storage.inputs import read_observation_section
from apex.storage.locks import exclusive, serialized
from apex.storage.run_directory import SURROGATE, RunDirectory
from apex.storage.run_results import RunResults
from apex.storage.surrogate_directory import (
    BUILD_FINGERPRINT,
    CHECK,
    CHECK_FINGERPRINT,
    GATE,
    MODEL,
    SNAPSHOTS,
    SUMMARY,
    StoredDesign,
    SurrogateDirectory,
    load_snapshots,
)
from apex.storage.surrogate_files import (
    cost_path,
    read_cost,
    read_surrogate,
    write_cost,
    write_surrogate,
)
from apex.workflows.parameterization import PropertyMap, parameterized
from apex.workflows.surrogate import MembersOutcome, check_success, fit_and_gate, run_members
from apex.workflows.surrogate_forecasts import ActiveModel, SurrogateForecasting

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSurrogate:
    """A fitted surrogate ready to predict, with what its files say about it."""

    path: Path
    model: SurrogateModel
    manifest: SurrogateManifest
    domain: TrainingDomain
    identity: str


def load_surrogate(path: Path, *, identity: str | None = None) -> LoadedSurrogate:
    """Load the surrogate whose manifest is ``path``.

    Parameters
    ----------
    path
        The manifest (``.json``); its arrays are beside it.
    identity
        The identity expected (e.g. a run's recorded one); ``None`` accepts any.

    Raises
    ------
    ConfigError
        If the files are refused (``storage.surrogate_files.read_surrogate``), the options are
        not the plugin's, or the model does not answer the manifest's keys.
    UnknownPluginError
        If no plugin of that name is registered.
    PluginError
        If the plugin cannot be loaded from arrays.
    """
    stored = read_surrogate(path, identity=identity)
    manifest = stored.manifest
    factory = surrogate_models.factory(manifest.plugin)
    problems = check_options("surrogate_models", manifest.plugin, manifest.options)
    if problems:
        raise ConfigError(
            f"the surrogate {Path(path).name} names options its plugin {manifest.plugin!r} "
            "does not take: " + "; ".join(problems)
        )
    from_arrays = getattr(factory, "from_arrays", None)
    if not callable(from_arrays):
        raise PluginError(f"the surrogate plugin {manifest.plugin!r} has no from_arrays")
    model = from_arrays(manifest.model, stored.arrays)
    if not isinstance(model, SurrogateModel):
        raise PluginError(
            f"the surrogate plugin {manifest.plugin!r} did not give a surrogate model"
        )
    if tuple(model.keys) != manifest.data_keys():
        raise ConfigError(
            f"the surrogate {Path(path).name}: its plugin answers other keys than its manifest "
            "records"
        )
    return LoadedSurrogate(
        path=Path(path),
        model=model,
        manifest=manifest,
        domain=manifest.training_domain(),
        identity=stored.identity,
    )


# =================================================================== the workflow (§7)

#: The build's random streams, spawned from ``[surrogate] seed`` in this order.
BUILD_STREAMS = ("design", "surrogate", "design_fit")

#: The build's subfolder of the coarse stack's runs.
COARSE = "coarse"

#: Simulators whose runs a surrogate cannot stand in for ("Refused").
REFUSED_SIMULATORS = {
    "intersect_visage": "its coupling reads each step's restart files",
    "ert": "ERT runs a case's members in batches; build from the native case (apex ert convert)",
}


def build_streams(seed: int) -> dict[str, np.random.Generator]:
    """Return the build's streams, one per name of :data:`BUILD_STREAMS`, from ``seed``."""
    children = np.random.SeedSequence(seed).spawn(len(BUILD_STREAMS))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(BUILD_STREAMS, children, strict=True)
    }


def load_surrogate_config(path: Path) -> SurrogateConfig:
    """Read and validate a build's configuration, resolving its relative paths.

    Raises
    ------
    ConfigError
        If the file cannot be read or does not match the schema.
    """
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the surrogate configuration {path}: {error}") from error
    try:
        config = SurrogateConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(
            f"invalid surrogate configuration {path}:\n{validation_problems(error)}"
        ) from error
    return resolve_surrogate_config(config, Path(path).resolve().parent)


def resolve_surrogate_config(config: SurrogateConfig, base: Path) -> SurrogateConfig:
    """Return ``config`` with its paths resolved against ``base`` and its output folder set."""

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    section = config.surrogate
    design = config.design
    execution = config.execution
    if execution is not None:
        execution = execution.model_copy(update={"options": {
            key: str(resolve(Path(value)))
            if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
            else value
            for key, value in execution.options.items()
        }})  # fmt: skip
    # The plugin's options naming files are resolved as every plugin's are (the convention:
    # options named *_file, *_dir or *_path, against the configuration's folder).
    # Not through symbolic links: a model records them relative to itself, as written.
    options = {
        key: os.path.abspath(base / value) if _is_path_option(key, value) else value
        for key, value in section.options.items()
    }
    return config.model_copy(update={
        "surrogate": section.model_copy(update={
            "output_dir": resolve(section.output_dir or Path("runs") / section.name),
            "options": options}),
        "case": config.case.model_copy(update={"config_file": resolve(config.case.config_file)}),
        "design": design.model_copy(update={
            "ensemble_file": design.ensemble_file and resolve(design.ensemble_file),
            "run_dir": None if design.run_dir is None else resolve(design.run_dir),
            # A plan's options naming files are resolved as every plugin's.
            "options": {
                key: os.path.abspath(base / str(value)) if _is_path_option(key, value) else value
                for key, value in design.options.items()
            },
        }),
        "execution": execution,
        "coarse": None if config.coarse is None else config.coarse.model_copy(update={
            "options": {
                key: os.path.abspath(base / str(value)) if _is_path_option(key, value) else value
                for key, value in config.coarse.options.items()
            },
            "parameterizations": [
                p.model_copy(update={"options": {
                    key: os.path.abspath(base / str(value)) if _is_path_option(key, value)
                    else value
                    for key, value in p.options.items()
                }})
                for p in config.coarse.parameterizations
            ],
        }),
    })  # fmt: skip


def _is_path_option(key: str, value: object) -> bool:
    return key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)


def options_relative_to(options: Mapping[str, object], folder: Path) -> dict[str, object]:
    """Return a plugin's options with those naming files relative to ``folder``.

    A model's manifest records them so (a model's folder, ``folder``), so that its identity
    does not depend on where the build ran: a build and its case files moved together keep
    the model's identity, and a refit finds them again (:func:`options_resolved_in`).
    """
    return {
        key: os.path.relpath(os.path.abspath(str(value)), os.path.abspath(folder))
        if _is_path_option(key, value)
        else value
        for key, value in options.items()
    }


def options_resolved_in(options: Mapping[str, object], folder: Path) -> dict[str, object]:
    """Return a manifest's options with those naming files resolved against ``folder``.

    The inverse of :func:`options_relative_to`; absolute paths (models written before
    2026-11-02) are kept as they are. Both work on the paths as written, never following
    symbolic links, so that a link moved with the model still points where it did.
    """
    return {
        key: os.path.abspath(folder / str(value)) if _is_path_option(key, value) else value
        for key, value in options.items()
    }


def _digest(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _with_contents(value: object) -> object:
    """Return a JSON value with every existing file or folder it names replaced by a digest.

    A file by its SHA-256, a folder by its input files' (``services.identity.tree_hash``:
    outputs and hidden files left out). Strings that name nothing on disk stay as they are.
    """
    if isinstance(value, dict):
        return {k: _with_contents(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_with_contents(v) for v in value]
    if isinstance(value, str) and ("/" in value or value.endswith((".npz", ".npy", ".csv"))):
        path = Path(value)
        if path.is_file():
            return {"file_sha256": provenance.file_hash(path)}
        if path.is_dir():
            # A folder (a deck's case_dir, say) by the content of its input files, as run
            # identity hashes a model folder: a copied case is the same case.
            return {"dir_sha256": tree_hash(path)}
    return value


def stack_provenance(case: HistoryMatchConfig, simulator: Simulator) -> SnapshotProvenance:
    """Return what names a case's simulator stack, as a model records it.

    ``options_digest``: the simulator's options, with the content of every file they name;
    ``deck_digest``: the deck template's folder (its includes), as run identity hashes it;
    ``case_digest``: what else maps the simulator's inputs to its answers
    (``[geophysics]``, ``[model_error]``). Paths do not count, contents do: every file and
    folder an option names (one with ``/`` in it, or a data file's suffix) is hashed by its
    content, so a copied case is the same case. A path given without a separator, or a file
    a deck reads through a path written inside it (not an option), is not followed: such
    a change is seen only through the deck template's folder digest.
    """
    options = json.loads(case.simulator.model_dump_json())["options"]
    template = case.simulator.options.get("deck_template_file")
    version = getattr(simulator, "version", None)
    document = json.loads(case.model_dump_json(include={"geophysics", "model_error"}))
    return SnapshotProvenance(
        simulator=case.simulator.name,
        simulator_version=None if not callable(version) else _version(version),
        options_digest=_digest(_with_contents(options)),
        deck_digest=None if template is None else tree_hash(Path(str(template)).parent),
        case_digest=_digest(_with_contents(document)),
        apex_commit=code_version()["commit"],
    )


def _version(version: Callable[[], object]) -> str | None:
    try:
        value = version()
    except (ApexError, OSError):
        return None
    return None if value is None else str(value)


@dataclass(frozen=True)
class SurrogateCase:
    """A case as a build runs it: its stack, its keys and where its runs go.

    Attributes
    ----------
    config
        The case's history-matching configuration (resolved).
    layout
        Its state layout: designs are drawn in state space.
    properties
        Its parameterizations: what maps a member's state to the surrogate's inputs.
    input_layout
        The surrogate's inputs (the properties the simulator reads).
    keys
        The keys the surrogate answers.
    variances
        ``(n_keys,)``, their observation errors' variances, for a gate against them;
        ``None`` for relative errors.
    simulator
        The stack, given the state (its parameterizations applied).
    executor
        Where the full runs go.
    provenance
        The stack's provenance (without the design's plan and seed).
    """

    config: HistoryMatchConfig
    layout: StateLayout
    properties: PropertyMap
    input_layout: StateLayout
    keys: tuple[DataKey, ...]
    variances: NDArray[np.float64] | None
    simulator: Simulator
    executor: Executor
    provenance: SnapshotProvenance
    coarse_simulator: Simulator | None = None
    coarse_config: HistoryMatchConfig | None = None


def _refuse(case: HistoryMatchConfig) -> None:
    if case.model_error is not None:
        raise ConfigError(
            "a surrogate cannot stand in for a stack with [model_error] (the "
            "model-error model reads each run's case files)"
        )
    name = case.simulator.name
    if name in REFUSED_SIMULATORS:
        raise ConfigError(f"a surrogate cannot stand in for {name!r}: {REFUSED_SIMULATORS[name]}")
    if case.execution.name == "ert":
        raise ConfigError("the 'ert' executor runs ERT's batches; use another [execution]")


def _input_layout(layout: StateLayout, properties: PropertyMap) -> StateLayout:
    """Return the layout of the surrogate's inputs: the state, or the properties."""
    if properties.identity:
        return layout
    one = {v.name: np.zeros(v.size) for v in layout.variables}
    sizes = {name: int(np.asarray(value).size) for name, value in properties.member(one).items()}
    return StateLayout(
        [StateVariable(name, sizes[name], grid=properties.grids.get(name))
         for name in properties.names]
    )  # fmt: skip


def case_keys(
    case: HistoryMatchConfig, outputs: object
) -> tuple[tuple[DataKey, ...], NDArray[np.float64] | None]:
    """Return a build's keys and, for the case's observations, their variances.

    Raises
    ------
    ConfigError
        If the case transforms field data that would be gated, or a key's time is neither
        a step index nor an ISO date.
    """
    if outputs == "observations":
        observations = read_observation_section(case.observations)
        keys = observations.keys
        variances: NDArray[np.float64] | None = np.asarray(observations.variances, np.float64)
        if case.data_transforms and any(key.index is not None for key in keys):
            # The gate would compare field data in raw space, where their
            # errors are not the method's; refused until that is tested.
            raise ConfigError(
                "the case transforms its field data ([[data_transforms]]): a surrogate of "
                "them is gated in the raw space, which is not the method's; list scalar "
                "outputs ([case] outputs = { vectors, times })"
            )
    else:
        assert isinstance(outputs, SurrogateSeries)
        keys = tuple(DataKey(t, v) for v in outputs.vectors for t in outputs.times)
        variances = None
    check_key_times(keys)
    return tuple(keys), variances


def check_key_times(keys: Sequence[DataKey]) -> None:
    """Refuse keys whose times the domain cannot bound.

    Raises
    ------
    ConfigError
        If a time is neither a step index nor an ISO date.
    """
    bad = []
    for key in keys:
        try:
            time_position(key.time)
        except ValueError:
            bad.append(key.time)
    if bad:
        raise ConfigError(
            "a surrogate's keys need times that are step indices or ISO dates (its domain "
            f"bounds them in time), got {sorted(set(bad))[:5]}"
        )


def surrogate_case(
    case: HistoryMatchConfig,
    keys: Sequence[DataKey],
    variances: NDArray[np.float64] | None,
    *,
    execution: PluginSection | None,
    output_dir: Path,
    coarse: CoarseSection | None = None,
) -> SurrogateCase:
    """Compose the stack a build (or a check) runs, from the case's configuration.

    ``keys`` are what the full runs answer (:func:`case_keys`), ``variances`` their
    observation errors' for a gate against them (``None``: relative errors).

    Raises
    ------
    ConfigError
        If the case is refused, a plugin is unknown or refuses its options, or
        the keys cannot be bounded in time.
    """
    _refuse(case)
    if execution is not None:
        case = case.model_copy(update={"execution": execution})
    case = composition.executor_defaults(
        case.model_copy(update={"run": case.run.model_copy(update={"output_dir": output_dir})})
    )
    layout = composition.build_layout(case)
    try:
        stack = composition.forecast_stack(case, layout)
        properties = PropertyMap(composition.build_parameterizations(case), layout)
        executor = executors.create(case.execution.name, **case.execution.options)
        simulator = parameterized(stack.simulator, properties)
        composition.check_executor(executor, simulator)
        composition.prepare_remote(executor, simulator, case)
    except (TypeError, ValueError, PluginError) as error:
        raise ConfigError(f"invalid case for a surrogate: {error}") from error
    provenance_ = stack_provenance(case, stack.simulator)
    coarse_simulator, coarse_config = None, None
    if coarse is not None:
        built = coarse_stack(case, coarse, layout, executor)
        coarse_simulator, coarse_config = built.simulator, built.config
        provenance_ = provenance_.model_copy(update={"coarse": built.provenance})
    return SurrogateCase(
        config=case,
        layout=layout,
        properties=properties,
        input_layout=_input_layout(layout, properties),
        keys=tuple(keys),
        variances=variances,
        simulator=simulator,
        executor=executor,
        provenance=provenance_,
        coarse_simulator=coarse_simulator,
        coarse_config=coarse_config,
    )


@dataclass(frozen=True)
class CoarseStack:
    """A coarse stack as a case runs it.

    Attributes
    ----------
    simulator
        Takes a member's state: the case's parameterizations, the coarse ones, then the
        coarse simulator.
    config
        The case with the coarse simulator, options and parameterizations.
    provenance
        What names it (:func:`stack_provenance`).
    """

    simulator: Simulator
    config: HistoryMatchConfig
    provenance: SnapshotProvenance


def coarse_stack(
    case: HistoryMatchConfig, coarse: CoarseSection, layout: StateLayout, executor: object
) -> CoarseStack:
    """Compose a case's coarse stack.

    The case's simulator with the coarse options, its parameterizations followed by the
    coarse ones; the same layout, keys and executor.

    Raises
    ------
    ConfigError
        If the executor runs members remotely, the case has ``[geophysics]``, the coarse
        deck would inherit the case's ``case_dir``, or the stack is refused or invalid.
    """
    if getattr(executor, "remote", False):
        raise ConfigError(
            "a coarse stack is a second stack: remote members carry one; use the "
            "executor's files mode or local workers"
        )
    if case.geophysics is not None:
        raise ConfigError(
            "a coarse stack with [geophysics]: its grid and masks are the fine deck's; a "
            "coarse stack forecasts the wells' data only"
        )
    if "case_dir" in case.simulator.options and "case_dir" not in coarse.options:
        raise ConfigError(
            "the coarse stack: the case's simulator has a case_dir (its decks' includes); "
            "give the coarse deck's own in the coarse options, or its includes are the fine "
            "case's"
        )
    config = case.model_copy(update={
        "simulator": case.simulator.model_copy(update={
            "name": coarse.simulator or case.simulator.name,
            "options": {**case.simulator.options, **coarse.options},
        }),
        "parameterizations": [*case.parameterizations, *coarse.parameterizations],
    })  # fmt: skip
    _refuse(config)
    try:
        stack = composition.forecast_stack(config, layout)
        properties = PropertyMap(composition.build_parameterizations(config), layout)
        simulator = parameterized(stack.simulator, properties)
        if executor is not None:
            composition.check_executor(executor, simulator)
    except (TypeError, ValueError, PluginError) as error:
        raise ConfigError(f"invalid coarse stack: {error}") from error
    return CoarseStack(simulator, config, stack_provenance(config, stack.simulator))


def coarse_relative_to(coarse: CoarseSection, folder: Path) -> CoarseSection:
    """Return a coarse stack with its file options relative to ``folder`` (a manifest's)."""
    return coarse.model_copy(update={
        "options": options_relative_to(coarse.options, folder),
        "parameterizations": [
            p.model_copy(update={"options": options_relative_to(p.options, folder)})
            for p in coarse.parameterizations
        ],
    })  # fmt: skip


def coarse_resolved_in(coarse: CoarseSection, folder: Path) -> CoarseSection:
    """Return a manifest's coarse stack with its file options resolved against ``folder``."""
    return coarse.model_copy(update={
        "options": options_resolved_in(coarse.options, folder),
        "parameterizations": [
            p.model_copy(update={"options": options_resolved_in(p.options, folder)})
            for p in coarse.parameterizations
        ],
    })  # fmt: skip


def _case_config(path: Path) -> HistoryMatchConfig:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the case configuration {path}: {error}") from error
    if "optimizer" in document:
        raise ConfigError(
            f"{path} is an optimization; surrogates of optimizations come with "
            "phase 4: give a history-matching configuration"
        )
    return composition.load_config(path)


# ----------------------------------------------------------------------------- designs


def _ensemble_file(path: Path, layout: StateLayout) -> dict[str, NDArray[np.float64]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            blocks = {}
            for variable in layout.variables:
                if variable.name not in data.files:
                    raise ConfigError(f"{path} has no array {variable.name!r}")
                block = np.asarray(data[variable.name], dtype=np.float64)
                if block.ndim == 1 and variable.size == 1:
                    block = block[None, :]
                if block.ndim != 2 or block.shape[0] != variable.size:
                    raise ConfigError(
                        f"{path}: {variable.name!r} must be ({variable.size}, n), got {block.shape}"
                    )
                blocks[variable.name] = block
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the design's ensemble {path}: {error}") from error
    return blocks


def _run_columns(
    run_dir: Path, iterations: object, case: SurrogateCase
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.bool_]]:
    """Return a run's saved iterates pooled: states ``{var: (size, n)}``, outputs ``(n_keys, n)``.

    Also ``(n,)``, which members' stored forecasts are finite at every key: the others are
    the design's failed runs (recorded, never silently dropped).

    Raises
    ------
    ConfigError
        If the run is of another stack, or its stored forecasts are not the keys' raw values.
    """
    run_config = composition.snapshot_config(RunDirectory(run_dir))
    _refuse(run_config)
    mismatches = case.provenance.stack_mismatches(
        stack_provenance(run_config, composition.forecast_stack(run_config, checks=False).simulator)
    )
    if mismatches:
        raise ConfigError(
            f"{run_dir} ran another simulator stack than the case ({', '.join(mismatches)} "
            "differ): its forecasts are not the case's"
        )
    workflow = composition.build_run(run_config, surrogate=False).workflow
    analysis = workflow.analysis_observations
    index = {key: row for row, key in enumerate(analysis.keys)}
    missing = [k for k in case.keys if k not in index]
    if missing:
        raise ConfigError(
            f"{run_dir} did not forecast {len(missing)} of the surrogate's keys, e.g. {missing[0]}"
        )
    if workflow.transforms and any(k.index is not None for k in case.keys):
        raise ConfigError(f"{run_dir} stored its field forecasts transformed, not raw")
    rows = [index[k] for k in case.keys]
    results = RunResults(run_dir, case.layout)
    chosen = results.iterations if iterations == "all" else list(iterations)  # type: ignore[call-overload]
    states: dict[str, list[NDArray[np.float64]]] = {v.name: [] for v in case.layout.variables}
    outputs = []
    for iteration in chosen:
        try:
            iterate = results.iterate(int(iteration))
        except FileNotFoundError as error:
            raise ConfigError(f"{run_dir}: iteration {iteration}: {error}") from error
        predicted = composition.all_roles(iterate, workflow.role_rows(), analysis.n_data)[rows]
        # Only the simulator's forecasts are snapshots: a surrogate run's own
        # forecasts would train a surrogate on a surrogate.
        simulated = np.array([s != SURROGATE for s in iterate.forecast_source], dtype=bool)
        for name in states:
            states[name].append(np.asarray(iterate.state[name])[:, simulated])
        outputs.append(predicted[:, simulated])
    if not outputs or not sum(o.shape[1] for o in outputs):
        raise ConfigError(
            f"{run_dir} has no saved iterate forecast by the simulator (a surrogate's "
            "forecasts are not snapshots)"
        )
    pooled = np.ascontiguousarray(np.concatenate(outputs, axis=1))
    return (
        {k: np.ascontiguousarray(np.concatenate(v, axis=1)) for k, v in states.items()},
        pooled,
        np.all(np.isfinite(pooled), axis=0),
    )


def _stored_generators(case: HistoryMatchConfig, layout: StateLayout) -> list[str]:
    """Return the prior generators of the case that ignore their random stream (``stored``)."""
    names = {case.prior.generator_for(v.name).name for v in layout.variables}
    names |= {section.name for section in case.prior.groups.values()}
    found = []
    for name in sorted(names):
        try:
            factory = priors.factory(name)
        except PluginError:
            continue  # unknown: refused when the prior is drawn
        if getattr(factory, "stored", False):
            found.append(name)
    return found


def _prior_draw(
    case: SurrogateCase, plan: str
) -> Callable[[int, np.random.Generator], dict[str, NDArray[np.float64]]]:
    """Return ``draw(n, rng)``: ``n`` members of the case's prior, in state space.

    Raises
    ------
    ConfigError
        If a stored prior gives the same columns whatever the stream (held-out runs would
        repeat training runs).
    """
    layout = case.layout
    stored = _stored_generators(case.config, layout)
    if stored:
        raise ConfigError(
            f"plan {plan!r} draws from the prior, but {', '.join(stored)} gives stored "
            "columns whatever the random stream: the held-out runs would repeat the "
            "training runs; use plan = 'file' with its ensemble_file"
        )

    def draw(n: int, generator: np.random.Generator) -> dict[str, NDArray[np.float64]]:
        drawn = composition.draw_prior_ensemble(case.config, layout, n, generator)
        return dict(layout.split(drawn))

    return draw


def design_plan(config: SurrogateConfig) -> BuiltinDesign | AdaptiveDesign:
    """Return ``[design] plan`` from the registry ``surrogate_designs``, with its options.

    Raises
    ------
    ConfigError
        If the plan is unknown, refuses its options, or is neither core's nor adaptive.
    """
    name = config.design.plan
    try:
        factory = surrogate_models.factory(config.surrogate.model)
    except UnknownPluginError:
        factory = None  # refused with its own message where the surrogate is created
    corrects = callable(getattr(factory, "predict_corrected", None))
    if corrects and config.coarse is None:
        raise ConfigError(
            f"the surrogate {config.surrogate.model!r} corrects a coarse simulation: it needs "
            "the runs' coarse outputs, from a [coarse] stack"
        )
    if factory is not None and not corrects and config.coarse is not None:
        raise ConfigError(
            f"[coarse] doubles the runs, but the surrogate {config.surrogate.model!r} does not "
            "correct a coarse simulation: remove [coarse] or use a coarse-corrected plugin"
        )
    try:
        plan = surrogate_designs.create(name, **config.design.options)
    except UnknownPluginError as error:
        raise ConfigError(
            f"unknown design plan {name!r}: one of {', '.join(surrogate_designs.names())}"
        ) from error
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[design] plan = {name!r}: {error}") from error
    if isinstance(plan, BuiltinDesign):
        if config.coarse is not None and name == "run":
            raise ConfigError("plan 'run' has no coarse runs: [coarse] needs a drawn design")
        return plan
    if config.coarse is not None:
        raise ConfigError(
            f"the adaptive plan {name!r} with a [coarse] stack is not supported yet: its fits "
            "between batches would need the coarse runs too"
        )
    if not (isinstance(plan, AdaptiveDesign) and getattr(plan, "adaptive", False)):
        raise ConfigError(f"the design plan {name!r} is neither core's nor an adaptive plan")
    try:
        factory = surrogate_models.factory(config.surrogate.model)
    except UnknownPluginError:
        return plan  # refused with its own message where the surrogate is created
    missing = [c for c in getattr(plan, "uses", ()) if not callable(getattr(factory, c, None))]
    if missing:
        raise ConfigError(
            f"the design plan {name!r} needs a surrogate with {', '.join(missing)}; "
            f"{config.surrogate.model!r} has none"
        )
    return plan


def draw_design(
    config: SurrogateConfig, case: SurrogateCase, rng: np.random.Generator
) -> StoredDesign:
    """Draw the design ``[design]`` describes, from the stream ``design``.

    Member ids: for drawn designs, the held-out runs are ``0..H-1`` and the training runs
    ``H..H+T-1``; for ``file`` and ``run``, the ensemble's columns.

    Raises
    ------
    ConfigError
        If the design cannot be drawn (unbounded box, a stored prior, too few columns).
    """
    section = config.design
    n_train, n_held = section.training, section.held_out
    layout = case.layout
    try:
        if section.plan == "prior":
            design = sampled_design(_prior_draw(case, "prior"), n_train, n_held, rng)
        elif section.plan == "lhs":
            box = {}
            for variable in layout.variables:
                lower, upper = section.box.get(variable.name, (variable.lower, variable.upper))
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    raise ConfigError(
                        f"plan 'lhs' needs a bounded box: state variable {variable.name!r} has "
                        f"bounds [{lower}, {upper}]; give [design.box] {variable.name} = [lo, hi]"
                    )
                box[variable.name] = (np.full(variable.size, lower), np.full(variable.size, upper))
            design = lhs_design(box, n_train, n_held, rng)
        elif section.plan == "file":
            assert section.ensemble_file is not None
            ensemble = _ensemble_file(section.ensemble_file, layout)
            design = ensemble_design("file", ensemble, n_held, rng, n_train)
        else:
            assert section.run_dir is not None
            ensemble, outputs, finite = _run_columns(section.run_dir, section.iterations, case)
            # The design chooses among the members whose forecasts are finite; ids are the
            # pooled columns, so the others are recorded as failed by theirs.
            usable = np.flatnonzero(finite)
            design = ensemble_design(
                "run", {k: v[:, usable] for k, v in ensemble.items()}, n_held, rng, n_train
            )
            assert design.training_columns is not None
            assert design.held_out_columns is not None
            held = [int(usable[c]) for c in design.held_out_columns]
            train = [int(usable[c]) for c in design.training_columns]
            return StoredDesign(
                plan="run",
                held_out_ids=tuple(held),
                training_ids=tuple(train),
                held_out=design.held_out,
                training=design.training,
                held_out_outputs=np.ascontiguousarray(outputs[:, held]),
                training_outputs=np.ascontiguousarray(outputs[:, train]),
                failed_ids=tuple(int(i) for i in np.flatnonzero(~finite)),
            )
    except ValueError as error:
        raise ConfigError(f"the design cannot be drawn: {error}") from error
    if design.held_out_columns is not None and design.training_columns is not None:
        held_ids, train_ids = design.held_out_columns, design.training_columns
    else:
        held_ids = tuple(range(n_held))
        train_ids = tuple(range(n_held, n_held + n_train))
    return StoredDesign(
        plan=section.plan,
        held_out_ids=held_ids,
        training_ids=train_ids,
        held_out=design.held_out,
        training=design.training,
    )


# ------------------------------------------------------------------------------ building


#: The case's sections that fix its runs (the method, the observations' values, the run's
#: name and executor do not).
_RUN_SECTIONS = ("state", "prior", "parameterizations", "simulator", "geophysics", "model_error")


def fingerprint_parts(config: SurrogateConfig, case: SurrogateCase) -> dict[str, str]:
    """Return what fixes a build's runs, part by part (a digest or a value each).

    The case's sections that shape its runs (``case.<section>``, files by content), the
    stack's provenance (``stack.<field>``: simulator, its version, its options' and deck's
    digests), the keys, the design and the seed. The plugin, its options, the gate and the
    domain's extent are left out: changing them refits on the runs already made (the runs are
    the cost). A resume refused names the parts that differ.
    """
    document = json.loads(case.config.model_dump_json(include=set(_RUN_SECTIONS)))
    parts = {f"case.{name}": _digest(_with_contents(document.get(name))) for name in _RUN_SECTIONS}
    stack = case.provenance.model_dump(
        include={"simulator", "simulator_version", "options_digest", "deck_digest", "case_digest"}
    )
    parts |= {f"stack.{name}": str(value) for name, value in sorted(stack.items())}
    parts["keys"] = _digest([key_document(k) for k in case.keys])
    design = json.loads(config.design.model_dump_json())
    if not design.get("options"):
        # Absent before the plans' options existed: builds keep their digest.
        design.pop("options", None)
    parts["design"] = _digest(design)
    parts["seed"] = str(config.surrogate.seed)
    if config.coarse is not None and case.provenance.coarse is not None:
        # The coarse stack's runs are runs too: what shapes them (absent without one).
        coarse = case.provenance.coarse.model_dump(
            include={"simulator", "simulator_version", "options_digest", "deck_digest",
                     "case_digest"}
        )  # fmt: skip
        parts["coarse"] = _digest({
            "section": _with_contents(json.loads(config.coarse.model_dump_json())),
            "stack": coarse,
        })  # fmt: skip
    if config.design.plan not in BUILTIN_PLANS:
        # An adaptive plan's runs are chosen by the surrogate (the user,
        # 2026-11-02): its plugin and options are part of what the runs are.
        parts["surrogate"] = _digest({
            "model": config.surrogate.model,
            "options": _with_contents(config.surrogate.options),
        })  # fmt: skip
    return parts


def _fingerprint(parts: Mapping[str, str]) -> str:
    return _digest(dict(parts))


def _fit_fingerprint(config: SurrogateConfig) -> str:
    # Files the options name count by their content (an edited case file refits); options
    # naming no file are unchanged, so existing builds keep their fingerprints.
    return _digest({
        "model": config.surrogate.model, "options": _with_contents(config.surrogate.options),
        "extent": config.surrogate.extent, "gate": json.loads(config.gate.model_dump_json()),
    })  # fmt: skip


@dataclass(frozen=True)
class BuildResult:
    """A build's outcome: its directory and summary."""

    path: Path
    summary: SurrogateBuildSummary

    @property
    def complete(self) -> bool:
        """Whether every run was made and the model written."""
        return self.summary.status == "complete"


def plan_build(config: SurrogateConfig) -> SurrogateBuildPlan:
    """Return what a build will cost (``apex validate``); nothing runs.

    Raises
    ------
    ConfigError
        If the configuration, its case or its plugin is invalid.
    """
    problems = check_options("surrogate_models", config.surrogate.model, config.surrogate.options)
    if problems:
        raise ConfigError("[surrogate]: " + "; ".join(problems))
    design_plan(config)  # an unknown plan, its options, the surrogate it needs: before any run
    assert config.surrogate.output_dir is not None
    case = _build_case(config)
    warnings = []
    if config.design.plan == "lhs" and case.layout.n_state > 50:
        warnings.append(
            f"a Latin hypercube over {case.layout.n_state} state entries spreads few runs over "
            "many coordinates; a field's design is better drawn from its prior (plan 'prior')"
        )
    if config.design.plan == "run":
        assert config.design.run_dir is not None
        _, _, finite = _run_columns(config.design.run_dir, config.design.iterations, case)
        available = int(np.count_nonzero(finite))
        if available < config.design.training + config.design.held_out:
            raise ConfigError(
                f"{config.design.run_dir} holds {available} members with finite forecasts, "
                "fewer than the "
                f"{config.design.training + config.design.held_out} the design asks for"
            )
    runs = 0 if config.design.plan == "run" else config.design.training + config.design.held_out
    if config.coarse is not None:
        runs *= 2  # every run also on the coarse stack
    return SurrogateBuildPlan(
        name=config.surrogate.name,
        model=config.surrogate.model,
        plan=config.design.plan,
        training=config.design.training,
        held_out=config.design.held_out,
        simulator_runs=runs,
        keys=len(case.keys),
        datatypes=sorted({k.datatype for k in case.keys}),
        measure="relative" if case.variances is None else "sigma_ratio",
        simulator=case.config.simulator.name,
        executor=case.config.execution.name,
        inputs=case.input_layout.n_state,
        output_dir=str(config.surrogate.output_dir),
        warnings=warnings,
    )


def build(
    path: Path, *, progress: ProgressReporter | None = None, max_runs: int | None = None
) -> BuildResult:
    """Start or resume the build ``path`` describes (``apex surrogate build CONFIG``)."""
    return build_config(load_surrogate_config(path), progress=progress, max_runs=max_runs)


def resume_build(
    path: Path, *, progress: ProgressReporter | None = None, max_runs: int | None = None
) -> BuildResult:
    """Continue the build in the directory ``path`` from its saved configuration.

    Raises
    ------
    ConfigError
        If ``path`` holds no surrogate build.
    """
    directory = SurrogateDirectory(path)
    if not SurrogateDirectory.holds_build(path):
        raise ConfigError(f"{path} holds no surrogate build")
    config = SurrogateConfig.model_validate_json((directory.path / "config.json").read_text())
    return build_config(config, progress=progress, max_runs=max_runs)


def build_config(
    config: SurrogateConfig,
    *,
    progress: ProgressReporter | None = None,
    max_runs: int | None = None,
) -> BuildResult:
    """Start or resume a build from a resolved configuration (one process per directory).

    Raises
    ------
    ConfigError
        If the configuration, its case or its plugin is invalid, or the directory holds a
        build of other runs.
    WorkflowError
        If too many runs fail.
    """
    assert config.surrogate.output_dir is not None
    with exclusive(config.surrogate.output_dir, "surrogate build"):
        return _build(config, progress, max_runs)


def _build_case(config: SurrogateConfig) -> SurrogateCase:
    case_config = _case_config(config.case.config_file)
    keys, variances = case_keys(case_config, config.case.outputs)
    assert config.surrogate.output_dir is not None
    return surrogate_case(
        case_config, keys, variances, execution=config.execution,
        output_dir=config.surrogate.output_dir, coarse=config.coarse,
    )  # fmt: skip


def _members(
    ids: Sequence[int], blocks: Mapping[str, NDArray[np.float64]]
) -> list[tuple[int, dict[str, NDArray[np.float64]]]]:
    return [(int(m), {k: v[:, j] for k, v in blocks.items()}) for j, m in enumerate(ids)]


def _snapshot_set(
    case: SurrogateCase,
    ids: Sequence[int],
    blocks: Mapping[str, NDArray[np.float64]],
    outputs: Mapping[int, NDArray[np.float64]],
    provenance_: SnapshotProvenance,
    failed: Sequence[int],
    coarse: Mapping[int, NDArray[np.float64]] | None = None,
) -> SnapshotSet:
    """Return the successful members of ``ids`` as snapshots of the surrogate's inputs.

    With ``coarse`` (a coarse stack's values by member), only members with both, and the
    coarse values as the set's ``coarse_outputs``.
    """
    columns = [
        j for j, m in enumerate(ids)
        if int(m) in outputs and (coarse is None or int(m) in coarse)
    ]  # fmt: skip
    state = {k: np.ascontiguousarray(v[:, columns]) for k, v in blocks.items()}
    inputs = case.properties.ensemble(state)
    return SnapshotSet(
        layout=case.input_layout,
        inputs={k: np.ascontiguousarray(v) for k, v in inputs.items()},
        keys=case.keys,
        outputs=np.stack([outputs[int(ids[j])] for j in columns], axis=-1).reshape(
            len(case.keys), len(columns)
        ),
        run_ids=tuple(int(ids[j]) for j in columns),
        provenance=provenance_,
        failed=tuple(int(m) for m in failed),
        coarse_outputs=None
        if coarse is None
        else np.stack([coarse[int(ids[j])] for j in columns], axis=-1).reshape(
            len(case.keys), len(columns)
        ),
    )


def _initial_adaptive_design(
    config: SurrogateConfig, case: SurrogateCase, plan: AdaptiveDesign, rng: np.random.Generator
) -> StoredDesign:
    """Return an adaptive plan's first design: the held-out runs, then the initial ones."""
    n_train, n_held = config.design.training, config.design.held_out
    n_initial = int(plan.initial_runs(n_train))
    if not 2 <= n_initial <= n_train:
        raise ConfigError(
            f"the design plan {config.design.plan!r} starts with {n_initial} runs: it needs "
            f"2 to training = {n_train}"
        )
    try:
        drawn = sampled_design(_prior_draw(case, config.design.plan), n_initial, n_held, rng)
    except ValueError as error:
        raise ConfigError(f"the design cannot be drawn: {error}") from error
    return StoredDesign(
        plan=config.design.plan,
        held_out_ids=tuple(range(n_held)),
        training_ids=tuple(range(n_held, n_held + n_initial)),
        held_out=drawn.held_out,
        training=drawn.training,
    )


def _adaptive_design(
    design: StoredDesign,
    initial: int,
    pool: Mapping[str, NDArray[np.float64]],
    batches: Sequence[Sequence[int]],
) -> StoredDesign:
    """Return the design of the initial training runs and the pool's chosen columns.

    Member ids: the held-out runs', then the training runs' in order, contiguous.
    """
    chosen = [i for batch in batches for i in batch]
    start = design.training_ids[0]
    return StoredDesign(
        plan=design.plan,
        held_out_ids=design.held_out_ids,
        training_ids=tuple(range(start, start + initial + len(chosen))),
        held_out=design.held_out,
        training={
            k: np.ascontiguousarray(np.concatenate(
                [np.asarray(design.training[k])[:, :initial], pool[k][:, chosen]], axis=1
            ))
            for k in design.training
        },
    )  # fmt: skip


def _run_adaptive(
    config: SurrogateConfig,
    case: SurrogateCase,
    directory: SurrogateDirectory,
    plan: AdaptiveDesign,
    design: StoredDesign,
    plan_provenance: SnapshotProvenance,
    *,
    keep: str,
    max_runs: int | None,
    progress: ProgressReporter | None,
) -> tuple[StoredDesign, MembersOutcome]:
    """Run an adaptive design's batches until ``training`` runs are made.

    Each round runs the members not finished, fits the surrogate on the training runs so far
    (stream ``design_fit``, its state checkpointed per batch) and asks the plan for the next
    batch from the pool (prior draws from the ``design`` stream's state after the first
    design, so a resume redraws the same pool). The batches chosen are in ``state.json``;
    a resume reruns no choice already made. Returns the design and the last outcome (not
    complete when ``max_runs`` stopped it).
    """
    n_train = config.design.training
    saved = dict(directory.read_state()["adaptive"])
    pool_rng = np.random.default_rng()
    pool_rng.bit_generator.state = saved["pool_stream"]
    pool = _prior_draw(case, config.design.plan)(int(plan.candidate_runs()), pool_rng)
    pool_inputs = case.properties.ensemble(pool)
    n_pool = next(iter(pool.values())).shape[1]
    batches = [[int(i) for i in batch] for batch in saved.get("batches", [])]
    taken = {i for batch in batches for i in batch}
    # The state is the record: the training runs are the initial ones and the pool's columns
    # of the batches chosen (a design saved behind its state is rebuilt from it).
    design = _adaptive_design(design, int(saved["initial"]), pool, batches)
    directory.save_design(design, case.layout)
    budget = max_runs
    while True:
        members = _members(design.held_out_ids, design.held_out) + _members(
            design.training_ids, design.training
        )
        before = len(directory.journal())
        outcome = run_members(
            directory, case.simulator, case.executor, members, case.keys, label="design",
            keep_simulations=keep, max_runs=budget, progress=progress,
        )  # fmt: skip
        if budget is not None:
            budget = max(budget - (len(directory.journal()) - before), 0)
        if not outcome.complete or len(design.training_ids) >= n_train:
            return design, outcome
        for label, ids in (("held-out", design.held_out_ids), ("training", design.training_ids)):
            check_success(label, len(ids), sum(int(m) in outcome.failed for m in ids),
                          config.design.min_success_fraction)  # fmt: skip
        failed_ids = sorted(outcome.failed)
        training = _snapshot_set(case, design.training_ids, design.training, dict(outcome.values),
                                 plan_provenance, failed_ids)  # fmt: skip
        fit_rng = np.random.default_rng()
        fit_rng.bit_generator.state = saved["fit_stream"]
        section = config.surrogate
        try:
            model = surrogate_models.create(section.model, **section.options)
            model.fit(training, fit_rng)
        except (TypeError, ValueError, ArithmeticError, LookupError, AttributeError) as error:
            raise ConfigError(
                f"the surrogate {section.model!r} cannot be fitted between the design's "
                f"batches: {type(error).__name__}: {error}"
            ) from error
        remaining = [i for i in range(n_pool) if i not in taken]
        count = min(int(plan.batch_runs()), n_train - len(design.training_ids), len(remaining))
        if count < 1:
            raise ConfigError(
                f"the design plan {config.design.plan!r}'s pool is used up before {n_train} "
                "training runs: raise its candidates"
            )
        candidates = {k: np.ascontiguousarray(v[:, remaining]) for k, v in pool_inputs.items()}
        try:
            picks = [int(p) for p in plan.choose(
                model, training.inputs, candidates, count, keys=case.keys,
                variances=case.variances,
            )]  # fmt: skip
        except (TypeError, ValueError) as error:
            raise ConfigError(f"the design plan {config.design.plan!r}: {error}") from error
        if (
            len(picks) != count
            or len(set(picks)) != count
            or not all(0 <= p < len(remaining) for p in picks)
        ):
            raise PluginError(
                f"the design plan {config.design.plan!r} chose {picks}, not {count} distinct "
                f"candidates of {len(remaining)}"
            )
        chosen = [remaining[p] for p in picks]
        taken.update(chosen)
        batches.append(chosen)
        saved["batches"] = batches
        saved["fit_stream"] = fit_rng.bit_generator.state
        directory.write_state(adaptive=saved)  # the record first, then the design from it
        design = _adaptive_design(design, int(saved["initial"]), pool, batches)
        directory.save_design(design, case.layout)
        logger.info(
            "design batch chosen",
            extra={"event": "surrogate_design_batch", "batch": len(saved["batches"]),
                   "runs": count, "training": len(design.training_ids)},
        )  # fmt: skip


def _build(
    config: SurrogateConfig, progress: ProgressReporter | None, max_runs: int | None
) -> BuildResult:
    section = config.surrogate
    assert section.output_dir is not None
    case = _build_case(config)
    directory = SurrogateDirectory(section.output_dir)
    parts = fingerprint_parts(config, case)
    directory.open(
        kind=BUILD_FINGERPRINT,
        config_text=config.model_dump_json(indent=2),
        fingerprint=_fingerprint(parts),
        parts=parts,
    )
    composition.configure_logging(directory.path)
    directory.write_json(
        "provenance.json",
        provenance.collect(
            {"case_config": config.case.config_file, **_simulator_files(case.config)},
            simulator={"name": case.provenance.simulator,
                       "version": case.provenance.simulator_version},
        ),
    )  # fmt: skip
    state = directory.read_state()
    fit_key = _fit_fingerprint(config)
    model_files = (directory.model, directory.model.with_suffix(".npz"))
    if state.get("complete") and state.get("fit_fingerprint") == fit_key:
        saved = directory.read_json(SUMMARY)
        # A complete build whose model files are gone (deleted, say, to drop a model a
        # changed format cannot read) is fitted again on the runs it made.
        if saved is not None and all(f.is_file() for f in model_files):
            return BuildResult(directory.path, SurrogateBuildSummary.model_validate(saved))
        if saved is not None:
            logger.info(
                "surrogate model files missing; refitting on the runs made",
                extra={"event": "surrogate_refit_missing_model"},
            )
    plan = design_plan(config)
    adaptive = not isinstance(plan, BuiltinDesign)
    streams = build_streams(section.seed)
    design = directory.load_design()
    if design is not None and adaptive and "adaptive" not in directory.read_state():
        design = None  # saved, but its state was never written: draw it again (same draws)
    if design is None:
        if isinstance(plan, BuiltinDesign):
            design = draw_design(config, case, streams["design"])
        else:
            design = _initial_adaptive_design(config, case, plan, streams["design"])
            # The state first: it is the record (a design saved without it is drawn again,
            # identically, from the seed). The pool is drawn from here on, redrawn identically
            # on resume; the fits between batches draw from their own stream.
            directory.write_state(adaptive={
                "initial": len(design.training_ids),
                "pool_stream": streams["design"].bit_generator.state,
                "fit_stream": streams["design_fit"].bit_generator.state, "batches": [],
            })  # fmt: skip
        directory.save_design(design, case.layout)
        directory.write_state(design_stream=streams["design"].bit_generator.state)
        logger.info(
            "design drawn",
            extra={"event": "surrogate_design", "plan": design.plan,
                   "training": len(design.training_ids), "held_out": len(design.held_out_ids)},
        )  # fmt: skip
    plan_provenance = case.provenance.model_copy(update={
        "plan": design.plan, "seed": section.seed,
        "source": None if config.design.run_dir is None else str(config.design.run_dir),
    })  # fmt: skip
    keep = config.output.keep_simulations
    coarse_values: dict[int, NDArray[np.float64]] | None = None
    if design.plan == "run":
        assert design.held_out_outputs is not None
        assert design.training_outputs is not None
        values = {int(m): design.held_out_outputs[:, j] for j, m in enumerate(design.held_out_ids)}
        values |= {int(m): design.training_outputs[:, j] for j, m in enumerate(design.training_ids)}
        failed: dict[int, str] = dict.fromkeys(
            design.failed_ids, "its stored forecast is not finite at every key"
        )
        seconds: Mapping[int, float] = {}
        simulator_runs = 0
    else:
        journal_before = len(directory.journal())
        if isinstance(plan, BuiltinDesign):
            members = _members(design.held_out_ids, design.held_out) + _members(
                design.training_ids, design.training
            )
            outcome = run_members(
                directory, case.simulator, case.executor, members, case.keys, label="design",
                keep_simulations=keep, max_runs=max_runs, progress=progress,
            )  # fmt: skip
        else:
            design, outcome = _run_adaptive(
                config, case, directory, plan, design, plan_provenance,
                keep=keep, max_runs=max_runs, progress=progress,
            )  # fmt: skip
        if not outcome.complete:
            # An adaptive design's later batches are pending too (not drawn yet).
            later = max(config.design.training - len(design.training_ids), 0)
            summary = _summary(config, design, case, outcome.failed, len(outcome.pending) + later,
                               len(outcome.values) + len(outcome.failed))  # fmt: skip
            directory.write_json(SUMMARY, json.loads(summary.model_dump_json()))
            directory.write_state(complete=False)
            logger.info(
                "build stopped",
                extra={"event": "surrogate_stopped", "pending": len(outcome.pending)},
            )
            return BuildResult(directory.path, summary)
        values, seconds, failed = dict(outcome.values), outcome.seconds, dict(outcome.failed)
        simulator_runs = len(values) + len(failed)
        if case.coarse_simulator is not None:
            # Every design run on the coarse stack too, in a folder of its own,
            # within what is left of max_runs.
            budget = (
                None if max_runs is None
                else max(max_runs - (len(directory.journal()) - journal_before), 0)
            )  # fmt: skip
            coarse_outcome = run_members(
                SurrogateDirectory(directory.path / COARSE),
                case.coarse_simulator, case.executor,
                # Only the members whose fine run succeeded: a pair needs both.
                [m for m in _members(design.held_out_ids, design.held_out)
                 + _members(design.training_ids, design.training) if m[0] in values],
                case.keys, label="coarse", keep_simulations=keep, max_runs=budget,
                progress=progress,
            )  # fmt: skip
            if not coarse_outcome.complete:
                summary = _summary(
                    config, design, case, failed, len(coarse_outcome.pending),
                    simulator_runs + len(coarse_outcome.values) + len(coarse_outcome.failed),
                )  # fmt: skip
                directory.write_json(SUMMARY, json.loads(summary.model_dump_json()))
                directory.write_state(complete=False)
                logger.info(
                    "build stopped",
                    extra={"event": "surrogate_stopped", "pending": len(coarse_outcome.pending)},
                )  # fmt: skip
                return BuildResult(directory.path, summary)
            coarse_values = dict(coarse_outcome.values)
            for member, message in coarse_outcome.failed.items():
                # A run is a training or held-out pair: without its coarse run it is lost.
                failed.setdefault(member, f"its coarse run failed: {message}")
                values.pop(member, None)
            simulator_runs += len(coarse_outcome.values) + len(coarse_outcome.failed)
        for label, ids in (("held-out", design.held_out_ids), ("training", design.training_ids)):
            check_success(label, len(ids), sum(int(m) in failed for m in ids),
                          config.design.min_success_fraction)  # fmt: skip
    failed_ids = sorted(failed)
    held_out = _snapshot_set(case, design.held_out_ids, design.held_out, values,
                             plan_provenance, failed_ids, coarse_values)  # fmt: skip
    training = _snapshot_set(case, design.training_ids, design.training, values,
                             plan_provenance, failed_ids, coarse_values)  # fmt: skip
    # Every successful run, in id order: the snapshot set a refit starts from (phase 3).
    everything = _snapshot_set(
        case,
        list(design.held_out_ids) + list(design.training_ids),
        {k: np.concatenate([design.held_out[k], design.training[k]], axis=1)
         for k in design.held_out},
        values, plan_provenance, failed_ids, coarse_values,
    )  # fmt: skip
    directory.save_snapshots(everything)
    try:
        model = surrogate_models.create(section.model, **section.options)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[surrogate] {section.model}: {error}") from error
    try:
        fitted = fit_and_gate(
            model, training, held_out, streams["surrogate"],
            tolerances=config.gate.tolerances(), variances=case.variances,
            kinds=config.gate.kinds, extent=section.extent, failed=failed_ids,
        )  # fmt: skip
    except (TypeError, ValueError, ArithmeticError, LookupError, AttributeError) as error:
        # A plugin's own failure (an option of the wrong type, a shape it cannot take), named
        # with its options, never a traceback from inside the plugin.
        given = ", ".join(f"{k} = {v!r}" for k, v in section.options.items()) or "no options"
        raise ConfigError(
            f"the surrogate {section.model!r} ({given}) cannot be fitted: "
            f"{type(error).__name__}: {error}"
        ) from error
    stored = write_surrogate(
        directory.model,
        plugin=section.model,
        model=fitted.model,
        domain=fitted.domain,
        provenance=plan_provenance,
        options=options_relative_to(section.options, directory.model.parent),
        gate=fitted.gate,
        snapshots=SnapshotReference(path=f"../{SNAPSHOTS}", digest=everything.digest()),
        fit=FitRecord(seed=section.seed, stream_state=fitted.stream_state),
        apex_commit=plan_provenance.apex_commit,
        coarse=None
        if config.coarse is None
        else coarse_relative_to(config.coarse, directory.model.parent),
    )
    directory.write_json(GATE, json.loads(fitted.gate.model_dump_json()))
    # A changed plugin or gate refits on the runs made: the configuration follows the model
    # once it is written (a refit that fails leaves the last model's configuration).
    directory.write_config(config.model_dump_json(indent=2))
    times = [s for s in seconds.values() if math.isfinite(s)]
    if times:
        # Beside the model, outside its identity: what a run with it saves (decision 14).
        write_cost(stored.path, BuildCost(
            identity=stored.identity, runs=len(times), mean_seconds=float(np.mean(times)),
            median_seconds=float(np.median(times)), total_seconds=float(np.sum(times)),
            fit_seconds=fitted.fit_seconds,
        ))  # fmt: skip
    else:
        cost_path(stored.path).unlink(missing_ok=True)  # never an earlier build's
        if seconds:
            # Members run whole on other hosts come back untimed.
            logger.warning(
                "the build's full runs are not timed: runs with the model report no speed-up",
                extra={"event": "surrogate_untimed", "runs": len(seconds)},
            )
    summary = _summary(config, design, case, failed, 0, simulator_runs).model_copy(update={
        "status": "complete", "gate": fitted.gate, "identity": stored.identity,
        "model": str(MODEL),
        "domain": DomainSummary(
            coordinates=fitted.domain.n_coordinates, points=fitted.domain.n_points,
            extent=fitted.domain.extent, residual_bound=fitted.domain.residual_bound),
        "fit_seconds": fitted.fit_seconds, "prediction_seconds": fitted.prediction_seconds,
        "simulator_median_seconds": float(np.median(times)) if times else None,
    })  # fmt: skip
    directory.write_json(SUMMARY, json.loads(summary.model_dump_json()))
    directory.write_state(complete=True, fit_fingerprint=fit_key, identity=stored.identity)
    logger.info(
        "surrogate built",
        extra={"event": "surrogate_built", "identity": stored.identity,
               "passed": fitted.gate.passed},
    )  # fmt: skip
    return BuildResult(directory.path, summary)


def _simulator_files(case: HistoryMatchConfig) -> dict[str, Path]:
    return {
        f"simulator.{key}": Path(value)
        for key, value in case.simulator.options.items()
        if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
    }


def _summary(
    config: SurrogateConfig,
    design: StoredDesign,
    case: SurrogateCase,
    failed: Mapping[int, str],
    pending: int,
    simulator_runs: int,
) -> SurrogateBuildSummary:
    return SurrogateBuildSummary(
        name=config.surrogate.name,
        plugin=config.surrogate.model,
        plan=design.plan,
        status="stopped",
        training_runs=len(design.training_ids),
        held_out_runs=len(design.held_out_ids),
        failed=sorted(int(m) for m in failed),
        pending=pending,
        simulator_runs=simulator_runs,
        keys=len(case.keys),
        datatypes=sorted({k.datatype for k in case.keys}),
    )


def build_summary(path: Path) -> SurrogateBuildSummary:
    """Return a build's summary (``summary.json``).

    Raises
    ------
    ConfigError
        If ``path`` holds no build, or no summary yet (still drawing its design).
    """
    directory = SurrogateDirectory(path)
    if not SurrogateDirectory.holds_build(path):
        raise ConfigError(f"{path} holds no surrogate build")
    content = directory.read_json(SUMMARY)
    if content is None:
        raise ConfigError(f"{path}: the build has not finished a batch of runs yet")
    return SurrogateBuildSummary.model_validate(content)


# ------------------------------------------------------------------------------- checks


@dataclass(frozen=True)
class CheckResult:
    """A check's outcome: its directory and summary."""

    path: Path
    summary: SurrogateCheckSummary


def checks_folder(model_path: Path) -> Path:
    """Return where a model's checks go: its build's ``checks/``, or ``checks/`` beside it."""
    model_path = Path(model_path).resolve()
    build_dir = model_path.parent.parent
    if model_path.parent.name == "model" and SurrogateDirectory.holds_build(build_dir):
        return build_dir / "checks"
    return model_path.parent / "checks"


def _complete_checks(model_path: Path) -> list[SurrogateCheckSummary]:
    folder = checks_folder(model_path)
    found = []
    for path in sorted(folder.glob(f"*/{CHECK}")) if folder.is_dir() else []:
        summary = SurrogateCheckSummary.model_validate_json(path.read_text())
        if summary.status == "complete":
            found.append(summary)
    return found


def record_check(
    model_path: Path, folder: Path, summary: SurrogateCheckSummary
) -> SurrogateCheckSummary:
    """Write a completed check's ``check.json`` with its order number; return it numbered.

    The number is one past every complete check of the model, assigned and written under a
    lock on the model's ``checks/`` folder, so checks completing at once (each holding only
    its own folder's lock) never share one. A check completed again takes a new number,
    unless it is the latest already (it keeps its number).
    """
    with serialized(checks_folder(model_path), ".sequence.lock"):
        others = [c for c in _complete_checks(model_path) if c.id != summary.id]
        numbered = summary.model_copy(
            update={"sequence": max((c.sequence for c in others), default=0) + 1}
        )
        SurrogateDirectory(folder).write_json(CHECK, json.loads(numbered.model_dump_json()))
    return numbered


def last_check(model_path: Path, identity: str | None = None) -> SurrogateCheckSummary | None:
    """Return the latest complete check of a model (of ``identity``, if given), or ``None``.

    The latest by its order number (``sequence``, not a timestamp of whole seconds), passed
    or failed: a check that failed on too few successful runs is recorded too. History
    matching refuses a model whose last check failed.
    """
    found = [c for c in _complete_checks(model_path) if identity is None or c.identity == identity]
    return max(found, key=lambda c: c.sequence) if found else None


def _model_snapshots(model_path: Path, manifest: SurrogateManifest) -> SnapshotSet:
    """Return the snapshots a model was fitted and gated on (digest-checked).

    Raises
    ------
    ConfigError
        If the manifest names none, or the file is missing or not the one it records.
    """
    reference = manifest.snapshots
    if reference is None or reference.path is None:
        raise ConfigError(
            "the model records no snapshot set, so a check cannot keep its runs apart from "
            "the runs it was fitted and gated on"
        )
    path = Path(model_path).resolve().parent / reference.path
    snapshots = load_snapshots(path)
    if snapshots.digest() != reference.digest:
        raise ConfigError(f"the snapshots {path} are not the ones the model records")
    return snapshots


def check_surrogate(
    model_path: Path,
    against: Path,
    *,
    runs: int,
    seed: int = 0,
    iteration: int | None = None,
    min_success_fraction: float = 0.9,
    min_inside: float = 0.8,
    progress: ProgressReporter | None = None,
    max_runs: int | None = None,
) -> CheckResult:
    """Gate an existing model on new full runs (``apex surrogate check``); never changes it.

    ``against`` is the case's history-matching configuration (``runs`` members drawn from
    its prior with ``seed``) or a history-matching run directory (``runs`` members of its
    posterior, or of ``iteration``, chosen with ``seed``), run by the simulator.

    The runs are kept apart from the model's own: a member equal to one of the model's
    snapshots (its training or held-out runs) is left out and reported (``trained_on``),
    and the check is refused when fewer than ``runs`` other members remain. Members outside
    the model's domain are reported, not gated (the model refuses them). The check passes
    only if the gate passes, at least ``min_success_fraction`` of the runs succeeded (else
    it fails as a build does) and at least ``min_inside`` of the successful ones lie inside
    the domain.

    Raises
    ------
    ConfigError
        If the model, its snapshots or the case cannot be read, the case's stack is not the
        model's, or too few members are new to the model.
    WorkflowError
        If fewer than ``min_success_fraction`` of the runs succeed.
    """
    if runs < 1:
        raise ConfigError("a check needs at least one run")
    if not (0 < min_success_fraction <= 1 and 0 <= min_inside <= 1):
        raise ConfigError("min_success_fraction must be in (0, 1] and min_inside in [0, 1]")
    loaded = load_surrogate(Path(model_path))
    manifest = loaded.manifest
    coarse_section = None
    if isinstance(loaded.model, CoarseCorrectedSurrogate):
        if manifest.coarse is None or manifest.provenance.coarse is None:
            raise ConfigError(
                f"the surrogate {model_path} corrects a coarse simulation but records no "
                "coarse stack: rebuild it (models record theirs since the step 2)"
            )
        coarse_section = coarse_resolved_in(manifest.coarse, Path(model_path).parent)
    trained = _model_snapshots(Path(model_path), manifest)
    against = Path(against).resolve()
    if against.is_dir():
        case_config = composition.snapshot_config(RunDirectory(against))
        source = "posterior" if iteration is None else f"iteration:{iteration}"
    else:
        case_config = _case_config(against)
        source = "prior"
    check_id = _digest({
        "identity": loaded.identity, "against": str(against), "runs": runs, "seed": seed,
        "source": source,
    })[:16]  # fmt: skip
    folder = checks_folder(Path(model_path)) / check_id
    keys: tuple[DataKey, ...] = tuple(loaded.model.keys)
    variances = None
    if manifest.gate is not None and any(
        d.measure == "sigma_ratio" for d in manifest.gate.datatypes
    ):
        # Gated against observation errors: checked against the case's, at the same keys.
        observed, observed_variances = case_keys(case_config, "observations")
        assert observed_variances is not None
        index = {key: row for row, key in enumerate(observed)}
        missing = [k for k in keys if k not in index]
        if missing:
            raise ConfigError(
                f"the case does not observe {len(missing)} of the model's keys (e.g. "
                f"{missing[0]}), which were gated against observation errors"
            )
        variances = np.ascontiguousarray(observed_variances[[index[k] for k in keys]])
    with exclusive(folder, "surrogate check"):
        case = surrogate_case(
            case_config, keys, variances, execution=None, output_dir=folder,
            coarse=coarse_section,
        )  # fmt: skip
        mismatches = manifest.provenance.stack_mismatches(case.provenance)
        if mismatches:
            raise ConfigError(
                "the model was built for another simulator stack or case "
                f"({', '.join(mismatches)} differ)"
            )
        directory = SurrogateDirectory(folder)
        directory.open(
            kind=CHECK_FINGERPRINT,
            config_text=json.dumps({"model": str(Path(model_path).resolve()),
                                    "against": str(against), "runs": runs, "seed": seed,
                                    "source": source}, indent=2),
            fingerprint=check_id,
        )  # fmt: skip
        composition.configure_logging(directory.path)
        design = directory.load_design()
        if design is None:
            design, trained_on = _check_design(case, trained, against, source, iteration,
                                               runs, seed)  # fmt: skip
            directory.save_design(design, case.layout)
            directory.write_state(trained_on=trained_on)
        trained_on = [int(m) for m in directory.read_state().get("trained_on", [])]
        made_before = len(directory.journal())
        outcome = run_members(
            directory, case.simulator, case.executor,
            _members(design.held_out_ids, design.held_out),
            case.keys, label="check", max_runs=max_runs, progress=progress,
        )  # fmt: skip
        # What is left of max_runs for the coarse stack: less the fine runs made now.
        left = (
            None if max_runs is None
            else max(max_runs - (len(directory.journal()) - made_before), 0)
        )  # fmt: skip
        created = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        base = SurrogateCheckSummary(
            id=check_id, identity=loaded.identity, model=str(Path(model_path).resolve()),
            against=str(against), source=source, status="stopped", runs=runs,
            failed=sorted(outcome.failed), trained_on=trained_on, created=created,
            min_success_fraction=min_success_fraction, min_inside=min_inside,
        )  # fmt: skip
        if not outcome.complete:
            directory.write_json(CHECK, json.loads(base.model_dump_json()))
            return CheckResult(directory.path, base)
        try:
            check_success("check", runs, len(outcome.failed), min_success_fraction)
        except WorkflowError as error:
            # Recorded as the model's latest check, failed, before the error is raised: a
            # model whose last check failed is refused (phase 3).
            failed_check = base.model_copy(update={
                "status": "complete", "succeeded": runs - len(outcome.failed),
                "passed": False, "notes": [str(error)],
            })  # fmt: skip
            record_check(Path(model_path), directory.path, failed_check)
            raise
        coarse_values = None
        failed_ids = dict(outcome.failed)
        if case.coarse_simulator is not None:
            # Each successful run on the model's coarse stack too, and the
            # corrected prediction gated (a member without its coarse run is lost).
            coarse_outcome = run_members(
                SurrogateDirectory(directory.path / COARSE), case.coarse_simulator,
                case.executor,
                [m for m in _members(design.held_out_ids, design.held_out)
                 if m[0] in outcome.values],
                case.keys, label="coarse",
                max_runs=left,
                progress=progress,
            )  # fmt: skip
            if not coarse_outcome.complete:
                directory.write_json(CHECK, json.loads(base.model_dump_json()))
                return CheckResult(directory.path, base)
            coarse_values = dict(coarse_outcome.values)
            failed_ids.update(coarse_outcome.failed)
            base = base.model_copy(update={"failed": sorted(failed_ids)})
            try:
                check_success("check", runs, len(failed_ids), min_success_fraction)
            except WorkflowError as error:
                record_check(Path(model_path), directory.path, base.model_copy(update={
                    "status": "complete", "succeeded": runs - len(failed_ids),
                    "passed": False, "notes": [f"with the coarse runs: {error}"],
                }))  # fmt: skip
                raise
        snapshots = _snapshot_set(case, design.held_out_ids, design.held_out, outcome.values,
                                  case.provenance, sorted(failed_ids),
                                  coarse_values)  # fmt: skip
        found = check_members(loaded.domain, loaded.model, snapshots.inputs, snapshots.run_ids,
                              case.keys, coarse=snapshots.coarse_outputs)  # fmt: skip
        outside = set(found.members)
        inside = [j for j, m in enumerate(snapshots.run_ids) if m not in outside]
        tolerances = manifest.gate.tolerances if manifest.gate is not None else GateTolerances()
        if inside:
            gated = snapshots.columns(inside)
            if isinstance(loaded.model, CoarseCorrectedSurrogate):
                assert gated.coarse_outputs is not None
                predicted = np.ascontiguousarray(loaded.model.predict_corrected(
                    gated.inputs, gated.coarse_outputs, case.keys
                ))  # fmt: skip
            else:
                predicted = np.ascontiguousarray(loaded.model.predict(gated.inputs, case.keys))
            kinds = {d.datatype: d.kind for d in (manifest.gate.datatypes if manifest.gate else [])
                     if d.kind is not None}  # fmt: skip
            verdict = gate(predicted, gated.outputs, case.keys, gated.run_ids,
                           tolerances=tolerances, variances=case.variances, kinds=kinds,
                           failed=sorted(failed_ids))  # fmt: skip
        else:
            verdict = GateResult(held_out=[], failed=sorted(failed_ids), datatypes=[],
                                 tolerances=tolerances, passed=False)  # fmt: skip
        notes = []
        share = len(inside) / snapshots.n_runs
        if share < min_inside:
            notes.append(
                f"only {len(inside)} of the {snapshots.n_runs} successful runs lie inside the "
                f"model's domain ({share:.0%} < {min_inside:.0%}): too few to judge it"
            )
        if not verdict.passed:
            notes += verdict.failures()
        directory.write_json(GATE, json.loads(verdict.model_dump_json()))
        summary = base.model_copy(update={
            "status": "complete", "outside": sorted(outside), "gate": verdict,
            "succeeded": snapshots.n_runs, "inside": len(inside),
            "passed": verdict.passed and share >= min_inside, "notes": notes,
        })  # fmt: skip
        summary = record_check(Path(model_path), directory.path, summary)
        logger.info(
            "surrogate checked",
            extra={"event": "surrogate_check", "identity": loaded.identity,
                   "passed": summary.passed, "outside": len(outside)},
        )  # fmt: skip
        return CheckResult(directory.path, summary)


def _check_design(
    case: SurrogateCase,
    trained: SnapshotSet,
    against: Path,
    source: str,
    iteration: int | None,
    runs: int,
    seed: int,
) -> tuple[StoredDesign, list[int]]:
    """Choose a check's members, none equal to one of the model's snapshots.

    Returns the design (its runs as ``held_out``) and the members left out as trained on.

    Raises
    ------
    ConfigError
        If fewer than ``runs`` members are new to the model.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if source == "prior":
        drawn = composition.draw_prior_ensemble(case.config, case.layout, runs, rng)
        blocks = dict(case.layout.split(drawn))
        ids = np.arange(blocks[case.layout.names[0]].shape[1])
        what = f"the {ids.size} members drawn from the prior"
    else:
        results = RunResults(against, case.layout)
        k = results.posterior_iteration if iteration is None else iteration
        try:
            iterate = results.iterate(k)
        except FileNotFoundError as error:
            raise ConfigError(f"{against}: iteration {k}: {error}") from error
        blocks = {n: np.ascontiguousarray(np.asarray(v)) for n, v in iterate.state.items()}
        ids = np.asarray(iterate.member_ids, dtype=np.int64)
        what = f"the {ids.size} members of {against}'s iteration {k}"
    names = list(trained.layout.names)
    inputs = case.properties.ensemble(blocks)
    repeats = repeated_columns({n: inputs[n] for n in names}, dict(trained.inputs))
    trained_on = sorted({int(ids[i]) for i, _ in repeats})
    fresh = [j for j in range(ids.size) if int(ids[j]) not in trained_on]
    if len(fresh) < runs:
        hint = (
            " (a stored prior gives its first columns whatever the seed)"
            if source == "prior"
            else ""
        )
        raise ConfigError(
            f"only {len(fresh)} of {what} are new to the model ({len(trained_on)} are among "
            f"the runs it was fitted and gated on), fewer than the {runs} asked for{hint}; "
            "check against other members"
        )
    columns = (
        np.asarray(fresh[:runs])
        if source == "prior"
        else np.sort(rng.choice(np.asarray(fresh), size=runs, replace=False))
    )
    chosen = {n: np.ascontiguousarray(v[:, columns]) for n, v in blocks.items()}
    design = StoredDesign(
        plan=source,
        held_out_ids=tuple(int(ids[c]) for c in columns),
        training_ids=(),
        held_out=chosen,
        training={n: v[:, :0] for n, v in chosen.items()},
    )
    return design, trained_on


# ================================================================ history matching (§5)


@functools.lru_cache(maxsize=8)
def fitted_model(path: str, identity: str) -> tuple[SurrogateModel, TrainingDomain]:
    """Load a model of ``identity`` once per process (a surrogate simulator's worker loader).

    Raises
    ------
    ConfigError
        If the files are not those of ``identity``, or are refused.
    """
    loaded = load_surrogate(Path(path), identity=identity)
    return loaded.model, loaded.domain


def active_model(path: Path, identity: str | None = None) -> ActiveModel:
    """Load a model as a run forecasts with it: a :class:`SurrogateSimulator` and its files.

    Raises
    ------
    ConfigError
        If the files are refused, or are not those of ``identity`` (when given).
    """
    loaded = load_surrogate(Path(path), identity=identity)
    simulator = SurrogateSimulator(
        loaded.model,
        loaded.domain,
        identity=loaded.identity,
        loader=functools.partial(fitted_model, str(Path(path).resolve()), loaded.identity),
        guard=refuse_outside,
    )
    return ActiveModel(Path(path), loaded.identity, loaded.manifest, simulator)


def _field_data(config: HistoryMatchConfig) -> bool:
    return bool(config.observations.fields) or config.geophysics is not None


def refusals(config: HistoryMatchConfig) -> list[str]:
    """Return why ``config``'s ``[surrogate]`` is refused, from the configuration alone.

    "Refused": ``[model_error]``, fidelity levels, ``intersect_visage`` (and ERT's
    batches), and ``[[data_transforms]]`` on field data (§3, until tested); also
    ``variance_in_errors`` with data transforms (the variance is per raw datum). The
    adjoint update and executors running members remotely are refused when the run is built
    (:func:`refuse_parts`); the model's own refusals by :func:`refuse_model`.
    """
    section = config.surrogate
    if section is None:
        return []
    found = []
    if config.model_error is not None:
        found.append(
            "[surrogate] with [model_error]: the model-error model reads each run's case "
            "files, which a surrogate has none of"
        )
    if config.method.name == "multilevel_esmda" or "levels" in config.method.options:
        found.append(
            "[surrogate] with fidelity levels (multilevel ES-MDA): a surrogate is not a level"
        )
    name = config.simulator.name
    if name in REFUSED_SIMULATORS:
        found.append(f"[surrogate] over {name!r}: {REFUSED_SIMULATORS[name]}")
    if config.execution.name == "ert":
        found.append("[surrogate] with the 'ert' executor: ERT runs a case's members in batches")
    if config.data_transforms and _field_data(config):
        found.append(
            "[surrogate] with [[data_transforms]] on field data: the checks would gate them "
            "in a space not yet tested"
        )
    if section.variance_in_errors and config.data_transforms:
        found.append(
            "[surrogate] variance_in_errors with [[data_transforms]]: the predictive variance "
            "is per raw datum"
        )
    if section.on_check_failure == "refit" and config.data_transforms:
        found.append(
            '[surrogate] on_check_failure = "refit" with [[data_transforms]]: a refit is '
            "gated in the raw space, the checks in the transformed one"
        )
    return found


def refuse_parts(method: object, executor: object, section: SurrogateRunSection) -> None:
    """Refuse what a method or executor cannot do with a surrogate.

    The adjoint update; an executor running members whole on other hosts (the remote
    mode: the surrogate's forecasts stay local, and the checks' runs would come back
    transformed there); ``variance_in_errors`` with a method that does not use a forecast's
    error variance (``uses_error_variance``: ES-MDA and ES; the iterative methods draw their
    perturbations once and judge trials on ``C_d``, where the variance would only reweight
    the step); a fallback (``refit``, ``simulator``) with an iterative method that cannot
    judge its trials against its accepted iterate forecast again (``rescore_accepted``).

    Raises
    ------
    ConfigError
        Naming what is refused.
    """
    if section.variance_in_errors and not getattr(method, "uses_error_variance", False):
        raise ConfigError(
            f"[surrogate] variance_in_errors with {type(method).__name__}: only ES-MDA and ES "
            "add a forecast's variance to the observation errors (an iterative method draws "
            "its perturbations once and judges its trials on C_d)"
        )
    if (
        section.on_check_failure != "stop"
        and getattr(method, "iterative", False)
        and not callable(getattr(method, "rescore_accepted", None))
    ):
        raise ConfigError(
            f'[surrogate] on_check_failure = "{section.on_check_failure}" with '
            f"{type(method).__name__}: it cannot judge its trials against its accepted "
            'iterate forecast by the new source; use "stop"'
        )
    if getattr(method, "uses_sensitivities", False):
        raise ConfigError(
            "[surrogate] with the adjoint update (sensitivities): surrogate gradients are not "
            "part yet"
        )
    if getattr(executor, "remote", False):
        raise ConfigError(
            "[surrogate] with an executor running members on other hosts (remote mode): "
            "surrogate forecasts stay on local workers; use the executor's other mode "
            '(e.g. mode = "files") for the simulator\'s runs'
        )


def refuse_model(
    config: HistoryMatchConfig,
    model: ActiveModel,
    observations: ObservationSet,
    stack: SnapshotProvenance,
) -> None:
    """Refuse a model a run cannot use ("Refused").

    A model whose gate failed or whose last check failed; one built for another simulator
    stack or case; one whose domain lacks an observed key (every role: history matching,
    cv, test, field keys); a ``variance_in_errors`` model without ``predict_std``; a refit
    policy without the snapshots to refit from.

    Raises
    ------
    ConfigError
        Naming what is refused and why.
    """
    section = config.surrogate
    assert section is not None
    manifest = model.manifest
    name = model.path
    if manifest.gate is None or not manifest.gate.passed:
        failures = [] if manifest.gate is None else manifest.gate.failures()
        raise ConfigError(
            f"the surrogate {name} did not pass its gate ({'; '.join(failures) or 'none'}): "
            "rebuild it"
        )
    check = last_check(name, model.identity)
    if check is not None and not check.passed:
        raise ConfigError(
            f"the surrogate {name} failed its last check ({check.id}: "
            f"{'; '.join(check.notes) or 'not passed'}): rebuild or refit it"
        )
    mismatches = manifest.provenance.stack_mismatches(stack)
    if mismatches:
        raise ConfigError(
            f"the surrogate {name} was built for another simulator stack or case "
            f"({', '.join(mismatches)} differ)"
        )
    outside = check_request(model.domain, observations.keys)
    if not outside.inside:
        raise ConfigError(f"the surrogate {name} cannot forecast this run: {outside.message()}")
    if section.variance_in_errors and not callable(getattr(model.model, "predict_std", None)):
        raise ConfigError(
            f"[surrogate] variance_in_errors: the plugin {manifest.plugin!r} gives no "
            "predictive variance (predict_std)"
        )
    if section.on_check_failure == "refit":
        if manifest.snapshots is None or manifest.snapshots.path is None:
            raise ConfigError(
                f'[surrogate] on_check_failure = "refit": the surrogate {name} records no '
                "snapshot set to refit from"
            )
        sigma = any(d.measure == "sigma_ratio" for d in manifest.gate.datatypes)
        observed = set(observations.keys)
        if sigma and any(k not in observed for k in model.model.keys):
            raise ConfigError(
                f'[surrogate] on_check_failure = "refit": the surrogate {name} was gated '
                "against observation errors at keys this run does not observe"
            )


def surrogate_forecasting(
    config: HistoryMatchConfig,
    observations: ObservationSet,
    stack: SnapshotProvenance,
    *,
    layout: StateLayout | None = None,
    executor: object = None,
) -> SurrogateForecasting:
    """Compose what a run with ``[surrogate]`` holds, after every refusal.

    ``observations`` are the run's raw observations (every role), ``stack`` its simulator
    stack's provenance (:func:`stack_provenance`). A coarse-corrected model's coarse stack
    (from its manifest) is composed over the run's case, on the run's
    ``layout``, and run by the run's ``executor``; it must be the stack the model was
    built with.

    Raises
    ------
    ConfigError
        If the configuration or the model is refused.
    """
    section = config.surrogate
    assert section is not None
    found = refusals(config)
    if found:
        raise ConfigError("; ".join(found))
    original = active_model(section.model_file)
    refuse_model(config, original, observations, stack)
    manifest = original.manifest
    coarse = run_coarse_stack(config, original, layout, executor)
    assert manifest.gate is not None  # refused otherwise
    try:
        tolerances = (
            manifest.gate.tolerances
            if section.sigma_ratio is None
            else manifest.gate.tolerances.tightened(sigma_ratio=section.sigma_ratio)
        )
    except ValueError as error:
        raise ConfigError(f"[surrogate] sigma_ratio: {error}") from error
    kinds = {d.datatype: d.kind for d in manifest.gate.datatypes if d.kind is not None}
    variances = None
    if any(d.measure == "sigma_ratio" for d in manifest.gate.datatypes):
        index = {key: row for row, key in enumerate(observations.keys)}
        rows = [index[k] for k in original.model.keys if k in index]
        if len(rows) == len(original.model.keys):
            variances = np.ascontiguousarray(np.asarray(observations.variances)[rows])
    executor = (
        executors.create("serial")
        if section.workers is None
        else executors.create("process", workers=section.workers)
    )
    snapshots = (
        None
        if manifest.snapshots is None or manifest.snapshots.path is None
        else functools.partial(_model_snapshots, Path(section.model_file), manifest)
    )
    # The manifest's file options are relative to the model's folder; a refit reads them
    # there and records them relative to its own.
    resolved = options_resolved_in(manifest.options, Path(section.model_file).parent)
    return SurrogateForecasting(
        section,
        original=original,
        executor=executor,
        tolerances=tolerances,
        variances=variances,
        kinds=kinds,
        loader=active_model,
        plugin=functools.partial(surrogate_models.create, manifest.plugin, **resolved),
        snapshots=snapshots,
        cost=model_cost(Path(section.model_file), original.identity),
        stored_options=functools.partial(options_relative_to, resolved),
        coarse=None if coarse is None else coarse.simulator,
        stored_coarse=None
        if manifest.coarse is None
        else functools.partial(
            coarse_relative_to,
            coarse_resolved_in(manifest.coarse, Path(section.model_file).parent),
        ),
    )


def run_coarse_stack(
    config: HistoryMatchConfig,
    model: ActiveModel,
    layout: StateLayout | None,
    executor: object,
) -> CoarseStack | None:
    """Return the coarse stack a run with a coarse-corrected model runs.

    ``None`` for any other model. The stack is the manifest's (its file options relative
    to the model's folder) over the run's case: it must be the one the model was built
    with, compared as the fine stack is (:meth:`SnapshotProvenance.stack_mismatches`).

    Raises
    ------
    ConfigError
        If the manifest records no coarse stack, the stack is refused or invalid, or it is
        not the model's.
    """
    if not isinstance(model.model, CoarseCorrectedSurrogate):
        return None
    manifest = model.manifest
    if manifest.coarse is None or manifest.provenance.coarse is None:
        raise ConfigError(
            f"the surrogate {model.path} corrects a coarse simulation but records no coarse "
            "stack: rebuild it (models record theirs since the step 2)"
        )
    if layout is None:
        layout = composition.build_layout(config)
    section = coarse_resolved_in(manifest.coarse, model.path.parent)
    built = coarse_stack(config, section, layout, executor)
    mismatches = manifest.provenance.coarse.stack_mismatches(built.provenance)
    if mismatches:
        raise ConfigError(
            f"the surrogate {model.path} was built for another coarse stack "
            f"({', '.join(mismatches)} differ)"
        )
    return built


def model_cost(model_file: Path, identity: str) -> BuildCost | None:
    """Return what the build of the model ``identity`` cost (decision 14), ``None`` if unknown.

    The file beside the manifest, if it is this model's; for a model built before it was
    written, the journal of the build the model is still in (``<build>/model/surrogate.json``,
    whose state names the model). The cost only informs: a file that cannot be read, or
    is another model's, is logged and passed over, never stopping a run.
    """
    try:
        cost = read_cost(model_file)
    except ConfigError as error:
        logger.warning("surrogate cost unreadable", extra={"event": "surrogate_cost_skipped",
                                                           "reason": str(error)})  # fmt: skip
        cost = None
    if cost is not None and cost.identity == identity:
        return cost
    if cost is not None:
        logger.warning(
            "surrogate cost of another model",
            extra={"event": "surrogate_cost_skipped", "path": str(cost_path(model_file)),
                   "reason": f"its model is {cost.identity[:12]}..., not {identity[:12]}..."},
        )  # fmt: skip
    build = Path(model_file).resolve().parent.parent
    if not SurrogateDirectory.holds_build(build):
        return None
    if SurrogateDirectory(build).read_state().get("identity") != identity:
        return None
    times = [
        float(r.seconds)
        for r in SurrogateDirectory(build).journal().values()
        if r.ok and r.seconds is not None and math.isfinite(r.seconds)
    ]
    if not times:
        return None
    return BuildCost(
        identity=identity, runs=len(times), mean_seconds=float(np.mean(times)),
        median_seconds=float(np.median(times)), total_seconds=float(np.sum(times)),
    )  # fmt: skip


def planned_evaluations(
    section: SurrogateRunSection,
    ensemble: int | None,
    single: int | None,
    members: int | None,
    *,
    iterative: bool = False,
) -> int | None:
    """Return the surrogate evaluations a run may take: its forecasts' members, and refits'.

    A refit repeats its iteration's forecast (``members``); an iterative method's accepted
    iterate and its mean model are forecast again by the refit too (``members + 1``).
    ``None`` when the method cannot bound its forecasts or the ensemble's size is unknown.
    """
    if ensemble is None or members is None:
        return None
    refits = section.max_refits if section.on_check_failure == "refit" else 0
    per_refit = 2 * members + 1 if iterative else members
    return ensemble * members + (single or 0) + refits * per_refit


def planned_simulator_runs(
    section: SurrogateRunSection, ensemble: int | None, members: int | None
) -> tuple[int | None, list[str]]:
    """Return the simulator runs a run with ``[surrogate]`` may take, and what to say of them.

    The checks (at most one per check iteration among the ``ensemble`` forecasts), the
    posterior's forecast and, where the simulator may take over (``outside`` or
    ``on_check_failure`` = ``"simulator"``), every forecast as the worst case.
    """
    if ensemble is None or members is None:
        return None, []
    checked = members if section.check_members == "all" else min(section.check_members, members)
    checks = 0 if section.check_every == 0 else (ensemble - 1) // section.check_every
    runs = checks * checked + members
    notes = []
    if section.outside == "simulator" or section.on_check_failure == "simulator":
        # Every forecast, and an iterative method's accepted iterate and mean model forecast
        # again when the simulator takes over.
        runs += ensemble * members + (members + 1 if section.on_check_failure == "simulator" else 0)
        notes.append(
            "the simulator may take over members outside the domain or the rest of the run: "
            "counted as if it forecast everything"
        )
    return runs, notes
