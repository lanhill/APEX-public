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

import dataclasses
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
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from apex.algorithms.optimization.protocols import ResponseObjective
from apex.algorithms.surrogates.designs import (
    BUILTIN_PLANS,
    ensemble_design,
    latin_hypercube,
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
from apex.core.plugins import executors, priors, simulators, surrogate_designs, surrogate_models
from apex.core.progress import ProgressReporter
from apex.core.simulation import (
    CaseFileSimulator,
    Executor,
    SeriesRequest,
    Simulator,
    match_vectors,
)
from apex.core.surrogates import (
    AdaptiveDesign,
    BuiltinDesign,
    CoarseCorrectedSurrogate,
    EarlyStartSurrogate,
    FieldSurrogate,
    SurrogateModel,
)
from apex.domain.config import HistoryMatchConfig, PluginSection, SurrogateRunSection
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.optimization import OptimizationConfig
from apex.domain.prediction import PredictionSurrogateSection
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
from apex.services import optimization as optimization_services
from apex.services.identity import code_version, tree_hash
from apex.services.plugins import check_options
from apex.simulation.deck import deck_layout, restart_schedule
from apex.simulation.geophysics import GeophysicsSimulator
from apex.simulation.sensitivity import SensitivityRun
from apex.simulation.state_fields import StateFieldModel, field_keys, split_fields
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
from apex.workflows.prediction import PredictionSurrogate
from apex.workflows.simulated_objective import SimulatedObjective
from apex.workflows.surrogate import (
    MembersOutcome,
    check_success,
    column_objectives,
    early_data,
    fit_and_gate,
    gated_rows,
    run_members,
)
from apex.workflows.surrogate_forecasts import ActiveModel, SurrogateForecasting
from apex.workflows.surrogate_optimization import SurrogateObjective

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSurrogate:
    """A fitted surrogate ready to predict, with what its files say about it."""

    path: Path
    model: SurrogateModel
    manifest: SurrogateManifest
    domain: TrainingDomain
    identity: str


def current_options(plugin: str, options: Mapping[str, object]) -> dict[str, object]:
    """Return a manifest's options without those its plugin has retired.

    A plugin lists in ``retired_options`` the options it no longer takes but older
    manifests still name (e.g. a setting that became a run-time one), so that those models
    load and refit; the rest are checked against the plugin's options as before.
    """
    retired = frozenset(getattr(surrogate_models.factory(plugin), "retired_options", ()))
    dropped = sorted(set(options) & retired)
    if dropped:
        logger.info(
            "retired surrogate options ignored",
            extra={"event": "surrogate_retired_options", "plugin": plugin, "options": dropped},
        )
    return {k: v for k, v in options.items() if k not in retired}


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
    problems = check_options(
        "surrogate_models", manifest.plugin, current_options(manifest.plugin, manifest.options)
    )
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


def load_surrogate_config(path: Path, *, place: bool = True) -> SurrogateConfig:
    """Read and validate a build's configuration, resolving its relative paths.

    ``place``: give a build without ``output_dir`` its folder beside the file (the CLI's
    ``runs/<name>``); the service places it itself (decision 2).

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
    return resolve_surrogate_config(config, Path(path).resolve().parent, place=place)


def resolve_surrogate_config(
    config: SurrogateConfig, base: Path, *, place: bool = True
) -> SurrogateConfig:
    """Return ``config`` with its paths resolved against ``base`` and its output folder set.

    ``place=False`` leaves a missing ``output_dir`` unset (the service places it).
    """

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
            "output_dir": None if section.output_dir is None and not place
            else resolve(section.output_dir or Path("runs") / section.name),
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


def stack_provenance(
    case: HistoryMatchConfig | OptimizationConfig, simulator: Simulator
) -> SnapshotProvenance:
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
    assert case.simulator is not None  # an optimization without one is refused before
    options = json.loads(case.simulator.model_dump_json())["options"]
    template = case.simulator.options.get("deck_template_file")
    version = getattr(simulator, "version", None)
    # An optimization has neither section: its case digest is that of an empty document.
    document = (
        json.loads(case.model_dump_json(include={"geophysics", "model_error"}))
        if isinstance(case, HistoryMatchConfig)
        else {}
    )
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
        The case's history-matching or optimization configuration (resolved).
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
    fields, field_times, field_cells
        The state fields every run also saves, the dates and the cells.
    objective
        An optimization's response objective, when the keys are its responses: the gate
        ranks the held-out runs' objectives too.
    realizations
        An optimization's ``[model]`` arrays ``(n_values, n_realizations)`` by name (robust
        optimization): a design member ``m`` is given realization ``m mod n_realizations``.
    run_sections
        The configuration's sections that fix its runs (the build's fingerprint).
    gradients
        Whether every run also returns its values' derivatives with respect to its inputs
        (``SnapshotSet.gradients``), for a plugin with ``uses_gradients``.
    """

    config: HistoryMatchConfig | OptimizationConfig
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
    fields: tuple[str, ...] = ()
    field_times: tuple[str, ...] = ()
    field_cells: int = 0
    objective: ResponseObjective | None = None
    realizations: Mapping[str, NDArray[np.float64]] = dataclasses.field(default_factory=dict)
    #: Every run also returns its sensitivities (a plugin with ``uses_gradients``):
    #: ``simulator`` is wrapped in :class:`~apex.simulation.sensitivity.SensitivityRun`.
    gradients: bool = False
    run_sections: tuple[str, ...] = ()

    @property
    def optimization(self) -> bool:
        """Whether the case is an optimization's (its design is over its controls)."""
        return isinstance(self.config, OptimizationConfig)

    @property
    def history_matching(self) -> HistoryMatchConfig:
        """The case's history-matching configuration (refused for an optimization's).

        Raises
        ------
        ConfigError
            If the case is an optimization's.
        """
        if not isinstance(self.config, HistoryMatchConfig):
            raise ConfigError(
                "this needs a history-matching case (a prior, observations); the case is an "
                "optimization's"
            )
        return self.config

    def objective_of(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]] | None:
        """Return the objective of one column of outputs, when the keys are its responses."""
        objective = self.objective
        if objective is None:
            return None
        return lambda column: np.asarray(objective.evaluate(column), dtype=np.float64)

    def with_realizations(
        self, ids: Sequence[int], controls: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return design blocks: ``controls`` ``{name: (size, n)}`` and each member's model.

        Member ``ids[j]`` gets realization ``ids[j] mod n_realizations`` (none: unchanged).
        """
        blocks = {k: np.ascontiguousarray(v) for k, v in controls.items()}
        if not self.realizations:
            return blocks
        width = next(iter(self.realizations.values())).shape[1]
        columns = np.asarray([int(m) % width for m in ids], dtype=np.int64)
        for name, array in self.realizations.items():
            blocks[name] = np.ascontiguousarray(array[:, columns])
        return blocks

    @property
    def run_keys(self) -> tuple[DataKey, ...]:
        """The keys each full run answers: the surrogate's, then the fields' (field-major)."""
        if not self.fields:
            return self.keys
        return (*self.keys, *field_keys(self.fields, self.field_times, self.field_cells))


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
    case: HistoryMatchConfig | OptimizationConfig, outputs: object
) -> tuple[tuple[DataKey, ...], NDArray[np.float64] | None]:
    """Return a build's keys and, for the case's observations, their variances.

    An optimization's are its objective's response keys (``responses``) or listed series,
    always gated by relative errors.

    Raises
    ------
    ConfigError
        If the case transforms field data that would be gated, or a key's time is neither
        a step index nor an ISO date, or the outputs do not suit the kind of case.
    """
    if isinstance(case, OptimizationConfig):
        if outputs == "observations":
            raise ConfigError(
                "the case is an optimization, which has no observations: [case] outputs = "
                "'responses' (its objective's response keys) or { vectors, times }"
            )
        if outputs == "responses":
            keys = tuple(optimization_services.response_objective(case).response_keys())
        else:
            assert isinstance(outputs, SurrogateSeries)
            keys = tuple(DataKey(t, v) for v in outputs.vectors for t in outputs.times)
        check_key_times(keys)
        return keys, None
    if outputs == "responses":
        raise ConfigError(
            "[case] outputs = 'responses' are an optimization's objective's keys; the case is "
            "a history-matching configuration: use 'observations' or { vectors, times }"
        )
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
    case: HistoryMatchConfig | OptimizationConfig,
    keys: Sequence[DataKey],
    variances: NDArray[np.float64] | None,
    *,
    execution: PluginSection | None,
    output_dir: Path,
    coarse: CoarseSection | None = None,
    fields: Sequence[str] = (),
    field_times: Sequence[str] = (),
    field_cells: int | None = None,
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
    if isinstance(case, OptimizationConfig):
        if coarse is not None or fields:
            raise ConfigError(
                "a surrogate of an optimization takes no [coarse] stack and no [case] fields "
                "yet (its keys are the objective's responses)"
            )
        return _optimization_case(case, keys, execution=execution)
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
        inner, cells = stack.simulator, 0
        if fields:
            inner, cells = _field_stack(case, stack.simulator, fields, field_times, field_cells)
        simulator = parameterized(inner, properties)
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
        fields=tuple(str(f).upper() for f in fields),
        field_times=tuple(field_times),
        field_cells=cells,
        run_sections=_RUN_SECTIONS,
    )


#: An optimization's sections that fix its runs: what each member is given (its controls'
#: blocks and bounds, the model arrays, the parameterizations) and what runs it.
_OPTIMIZATION_RUN_SECTIONS = ("controls", "model", "parameterizations", "simulator")


def _optimization_case(
    case: OptimizationConfig, keys: Sequence[DataKey], *, execution: PluginSection | None
) -> SurrogateCase:
    """Compose the stack an optimization's surrogate is built (or checked) on.

    The design is over the controls (with ``[model]`` arrays, each member is given one
    realization); the surrogate's inputs are what the simulations are given, through the
    optimization's ``[[parameterizations]]``. The objective ranks the held-out runs when the
    keys are its response keys.

    Raises
    ------
    ConfigError
        If the optimization has no simulator, a plugin is unknown or refuses its options.
    """
    if case.simulator is None:
        raise ConfigError(
            "the optimization's objective is analytic (no [simulator]): a surrogate stands in "
            "for a simulator"
        )
    if case.simulator.name in REFUSED_SIMULATORS:
        raise ConfigError(
            f"a surrogate cannot stand in for {case.simulator.name!r}: "
            f"{REFUSED_SIMULATORS[case.simulator.name]}"
        )
    section = execution or case.execution
    if section.name == "ert":
        raise ConfigError("the 'ert' executor runs ERT's batches; use another [execution]")
    model = optimization_services.model_arrays(case)
    properties = optimization_services.control_properties(case, model)
    layout = optimization_services.control_layout(case, model)
    try:
        raw = simulators.create(case.simulator.name, **case.simulator.options)
        executor = executors.create(section.name, **section.options)
        simulator = raw if properties.identity else parameterized(raw, properties)
        composition.check_executor(executor, simulator)
    except (TypeError, ValueError, PluginError) as error:
        raise ConfigError(f"invalid case for a surrogate: {error}") from error
    objective = optimization_services.response_objective(case)
    responses = tuple(objective.response_keys())
    return SurrogateCase(
        config=case.model_copy(update={"execution": section}),
        layout=layout,
        properties=properties,
        input_layout=_input_layout(layout, properties),
        keys=tuple(keys),
        variances=None,
        simulator=simulator,
        executor=executor,
        provenance=stack_provenance(case, raw),
        # Ranked only when the keys are exactly what the objective reads.
        objective=objective if tuple(keys) == responses else None,
        realizations={k: np.ascontiguousarray(v) for k, v in model.items()},
        run_sections=_OPTIMIZATION_RUN_SECTIONS,
    )


def _field_stack(
    case: HistoryMatchConfig,
    simulator: Simulator,
    fields: Sequence[str],
    times: Sequence[str],
    cells: int | None,
) -> tuple[Simulator, int]:
    """Return the stack that also answers the state fields' keys, and the fields' cells.

    A deck simulator is wrapped with :class:`~apex.simulation.state_fields.StateFieldModel`
    (restart arrays on the deck's grid, ``DIMENS``/``SPECGRID``); any other simulator must
    answer the field keys itself (``field_cells`` given), e.g. ``linear`` with ``fields``.

    Raises
    ------
    ConfigError
        If the case has ``[geophysics]`` (its field keys would clash), or the cells cannot be
        known.
    """
    if case.geophysics is not None:
        raise ConfigError(
            "[case] fields with a [geophysics] case: build field surrogates on its flow stack "
            "(a case without [geophysics])"
        )
    if isinstance(simulator, CaseFileSimulator):
        template = case.simulator.options.get("deck_template_file")
        grid: list[int] | None = None
        case_dir = case.simulator.options.get("case_dir")
        if template is not None:
            found = deck_layout(
                Path(str(template)), None if case_dir is None else Path(str(case_dir))
            ).get("grid")
            grid = None if not isinstance(found, list | tuple) else [int(v) for v in found]
        if grid is None and cells is None:
            raise ConfigError(
                "[case] fields: the deck's grid (DIMENS or SPECGRID) cannot be read from its "
                "template; give field_cells"
            )
        if template is not None:
            # Before any run (as [geophysics]' vintages): a date the deck certainly writes no
            # restart at would fail every member.
            schedule = restart_schedule(
                Path(str(template)), None if case_dir is None else Path(str(case_dir))
            )
            missing = schedule.missing([dt.date.fromisoformat(t) for t in times])
            if missing and not schedule.unresolved:
                raise ConfigError(
                    "[case] fields: the deck writes no restart at "
                    + ", ".join(d.isoformat() for d in missing)
                    + " (give each field time a report time and a restart, e.g. RPTRST BASIC=2)"
                )
        total = int(np.prod(grid)) if grid is not None else int(cells or 0)
        if cells is not None and cells != total:
            raise ConfigError(f"[case] field_cells = {cells}, but the deck's grid has {total}")
        return GeophysicsSimulator(simulator, StateFieldModel(fields, times, total)), total
    if cells is None:
        raise ConfigError(
            f"[case] fields with the {case.simulator.name!r} simulator, which has no restart "
            "files: give field_cells, and the simulator must answer the field keys itself"
        )
    return simulator, int(cells)


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


def case_config(path: Path) -> HistoryMatchConfig | OptimizationConfig:
    """Return the configuration a build's ``[case] config_file`` names (resolved).

    Raises
    ------
    ConfigError
        If it is neither a history-matching nor an optimization configuration.
    """
    return _case_config(path)


def _case_config(path: Path) -> HistoryMatchConfig | OptimizationConfig:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read the case configuration {path}: {error}") from error
    if optimization_services.is_optimization(document):
        # An optimization's simulator stack, over its controls.
        return optimization_services.load_optimization_config(path, place=False)
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
    stored = _stored_generators(case.history_matching, layout)
    if stored:
        raise ConfigError(
            f"plan {plan!r} draws from the prior, but {', '.join(stored)} gives stored "
            "columns whatever the random stream: the held-out runs would repeat the "
            "training runs; use plan = 'file' with its ensemble_file"
        )

    def draw(n: int, generator: np.random.Generator) -> dict[str, NDArray[np.float64]]:
        drawn = composition.draw_prior_ensemble(case.history_matching, layout, n, generator)
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
    # decision 13: a plugin that learns from coarse runs in training only.
    trains = bool(getattr(factory, "trains_on_coarse", False))
    if corrects and config.coarse is None:
        raise ConfigError(
            f"the surrogate {config.surrogate.model!r} corrects a coarse simulation: it needs "
            "the runs' coarse outputs, from a [coarse] stack"
        )
    if trains and config.coarse is None:
        raise ConfigError(
            f"the surrogate {config.surrogate.model!r} trains on coarse runs: it needs a "
            "[coarse] stack (and [design] coarse_only for more coarse runs than fine ones)"
        )
    if factory is not None and not trains and config.design.coarse_only:
        raise ConfigError(
            f"[design] coarse_only = {config.design.coarse_only}: the surrogate "
            f"{config.surrogate.model!r} does not train on coarse runs (trains_on_coarse), so "
            "its fit would not read them: remove coarse_only, or use a plugin that does"
        )
    if factory is not None and not (corrects or trains) and config.coarse is not None:
        raise ConfigError(
            f"[coarse] doubles the runs, but the surrogate {config.surrogate.model!r} does not "
            "correct a coarse simulation: remove [coarse] or use a coarse-corrected plugin"
        )
    if config.case.fields:
        # The state fields are saved for a field surrogate, and gated.
        if factory is not None and not callable(getattr(factory, "predict_fields", None)):
            raise ConfigError(
                f"[case] fields: the surrogate {config.surrogate.model!r} predicts no fields "
                "(predict_fields); use a field surrogate (unet_fields) or remove fields"
            )
        if config.coarse is not None:
            raise ConfigError("[case] fields with a [coarse] stack are not supported yet")
        if name == "run":
            raise ConfigError("plan 'run' reuses a run's forecasts, which hold no fields")
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
    if config.coarse is not None or config.case.fields:
        raise ConfigError(
            f"the adaptive plan {name!r} with a [coarse] stack or [case] fields is not "
            "supported yet: its fits between batches would need them too"
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


#: The automatic calibration runs of a surrogate's spread (the user, 2026-11-10): a case of
#: up to SMALL_CASE_CELLS grid cells gets CALIBRATION_RUNS_SMALL, a larger one (dearer runs)
#: CALIBRATION_RUNS_LARGE. Measured on a known response (2026-11-10): z RMS on new runs
#: 0.84-1.50 (10th-90th percentile) with 10 runs, 0.91-1.27 with 30, where the validation runs
#: left the spread about 20 % too narrow.
SMALL_CASE_CELLS = 20_000
CALIBRATION_RUNS_SMALL = 30
CALIBRATION_RUNS_LARGE = 20


@dataclass(frozen=True)
class CalibrationCount:
    """How many calibration runs a build draws (``[design] calibration``), and why.

    Attributes
    ----------
    runs
        The count (0: none; a plugin that wants them then calibrates on its validation runs).
    rule
        How it was chosen, for the plan and the log (``None`` when the plugin takes none).
    automatic
        Whether ``[design]`` left it unset (the count then enters the build's identity).
    wanted
        Whether the plugin calibrates its spread on such runs (``uses_calibration_runs``):
        a Laplace spread's scales or an ensemble's factors, say.
    known
        Whether the plugin could be asked (``False``: not installed; a complete build is
        then taken as made).
    free
        Whether they are taken from stored forecasts (plan ``run``): no simulator runs.
    """

    runs: int
    rule: str | None
    automatic: bool
    wanted: bool
    known: bool = True
    free: bool = False

    def notice(self) -> str | None:
        """Return the line the plan and the build's log state (``None`` without the runs)."""
        if not self.wanted:
            return None
        if self.runs == 0:
            return (
                "no calibration runs: the surrogate's spread is calibrated on its validation "
                "runs, which also chose its training's stopping (a Laplace spread is then about "
                f"20 % too narrow on new runs); set [design] calibration = {CALIBRATION_RUNS_SMALL}"
                f" (small cases) or {CALIBRATION_RUNS_LARGE} (large ones) to add runs for it"
                + (f" ({self.rule})" if self.rule else "")
            )
        what = (
            f"{self.runs} calibration runs from the run's stored forecasts (no simulator runs)"
            if self.free
            else f"{self.runs} extra calibration runs"
        )
        return (
            f"{what} for the surrogate's spread ({self.rule}); set [design] calibration = K to "
            "change, 0 to calibrate on the validation runs instead"
        )


def _wants_calibration(config: SurrogateConfig) -> bool | None:
    """Whether the configured plugin calibrates its spread on runs of their own.

    ``None`` when it cannot be told because the plugin is not installed (a complete build
    needs no plugin). Other failures of its creation are not hidden: a plugin refusing its
    options is a :class:`ConfigError` with its message, as the build gives; anything else
    propagates.

    Raises
    ------
    ConfigError
        If the plugin refuses its options (``TypeError``, ``ValueError``).
    """
    try:
        model = surrogate_models.create(config.surrogate.model, **config.surrogate.options)
    except (PluginError, ImportError):
        return None
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[surrogate] {config.surrogate.model}: {error}") from error
    return bool(getattr(model, "uses_calibration_runs", False))


#: The most numbers a build's gradients may hold (``n_keys x n_inputs x n_runs``, the
#: training and held-out runs): 400 MB of float64 in ``snapshots.npz``; the build holds a few
#: copies while it assembles them. Gradient-enhanced models suit few inputs (a field of
#: thousands of cells is far beyond what their kernels use), so more is refused.
MAX_GRADIENT_VALUES = 50_000_000


def _wants_gradients(config: SurrogateConfig) -> bool:
    """Whether the configured plugin learns from the runs' gradients (``uses_gradients``).

    When the plugin cannot say (it is not installed, or now refuses its options) and the
    build's folder recorded a fingerprint, that fingerprint's ``gradients`` part: a complete
    build is found without its plugin, as :func:`_completed_build` finds any.

    Raises
    ------
    ConfigError
        If the plugin refuses its options and the folder recorded nothing: refused before
        anything is stamped there (as :func:`_wants_calibration`).
    """
    try:
        model = surrogate_models.create(config.surrogate.model, **config.surrogate.options)
    except (PluginError, ImportError, TypeError, ValueError) as error:
        parts = None
        if config.surrogate.output_dir is not None:
            parts = (
                SurrogateDirectory(config.surrogate.output_dir)
                .read_state()
                .get("fingerprint_parts")
            )
        if isinstance(parts, dict):
            return "gradients" in parts
        if isinstance(error, TypeError | ValueError):
            raise ConfigError(f"[surrogate] {config.surrogate.model}: {error}") from error
        return False
    return bool(getattr(model, "uses_gradients", False))


def _also_calibrates(config: SurrogateConfig) -> bool:
    """Whether the plugin also calibrates a spread on calibration runs.

    ``False`` when the plugin cannot be created (not installed, or now refusing its
    options): the build then rests on its recorded fingerprint (:func:`_wants_gradients`).
    """
    try:
        model = surrogate_models.create(config.surrogate.model, **config.surrogate.options)
    except (PluginError, ImportError, TypeError, ValueError):
        return False
    return bool(getattr(model, "uses_calibration_runs", False))


def _with_gradients(config: SurrogateConfig, case: SurrogateCase) -> SurrogateCase:
    """Return ``case`` whose runs return their sensitivities.

    Raises
    ------
    ConfigError
        If the stack cannot give gradients of the surrogate's inputs: a simulator without
        ``run_with_sensitivities``, parameterizations (the derivatives would be the state's,
        not the inputs'), state fields, a coarse stack, a plan other than ``prior``, ``lhs``
        and ``file``, or more than :data:`MAX_GRADIENT_VALUES` numbers.
    """
    plugin = config.surrogate.model
    # What the build itself shapes first: a field stack or a coarse one would also hide
    # the simulator's capability, and their own reason is the one to give.
    refused = [
        reason for reason, present in (
            ("state fields ([case] fields)", bool(case.fields)),
            ("a [coarse] stack", config.coarse is not None),
            (f"the plan {config.design.plan!r} (prior, lhs or file only)",
             config.design.plan not in ("prior", "lhs", "file")),
        ) if present
    ]  # fmt: skip
    if refused:
        raise ConfigError(
            f"[surrogate] {plugin} learns from the runs' gradients, which are not made with "
            + ", ".join(refused)
        )
    if not case.properties.identity:
        raise ConfigError(
            f"[surrogate] {plugin} learns from gradients with respect to its inputs, and the "
            "case's [[parameterizations]] map the state to them: not supported"
        )
    if _also_calibrates(config):
        # Its calibration runs would be run with sensitivities never used (and outside the
        # size limit's count): no plugin has both capabilities yet, so the pair is refused.
        raise ConfigError(
            f"[surrogate] {plugin} learns from the runs' gradients and calibrates a spread on "
            "calibration runs: the two together are not supported"
        )
    if not hasattr(case.simulator, "run_with_sensitivities"):
        raise ConfigError(
            f"[surrogate] {plugin} learns from the runs' gradients, and the "
            f"{case.provenance.simulator!r} simulator gives none (no run_with_sensitivities): "
            "use a simulator with sensitivities, or another surrogate"
        )
    runs = config.design.training + config.design.held_out
    values = len(case.keys) * case.input_layout.n_state * runs
    if values > MAX_GRADIENT_VALUES:
        raise ConfigError(
            f"[surrogate] {plugin}: the gradients of {len(case.keys)} keys over "
            f"{case.input_layout.n_state} inputs for {runs} runs are {values:,} numbers, over "
            f"the limit of {MAX_GRADIENT_VALUES:,}: fewer keys, inputs or runs"
        )
    return dataclasses.replace(
        case, simulator=SensitivityRun(case.simulator, case.layout.n_state), gradients=True
    )


def _spare_columns(config: SurrogateConfig, case: SurrogateCase) -> int | None:
    """Return the columns a finite ensemble leaves after the other runs (``None``: drawn).

    Plans ``file`` (its ensemble's columns) and ``run`` (its members with finite forecasts),
    after the held-out, training and coarse-only runs; never below zero.
    """
    section = config.design
    if section.plan == "file" and section.ensemble_file is not None:
        layout = (
            StateLayout([v for v in case.layout.variables if v.name not in case.realizations])
            if case.optimization
            else case.layout
        )
        columns = next(iter(_ensemble_file(section.ensemble_file, layout).values())).shape[-1]
    elif section.plan == "run" and section.run_dir is not None:
        _, _, finite = _run_columns(section.run_dir, section.iterations, case)
        columns = int(np.count_nonzero(finite))
    else:
        return None
    return max(int(columns) - section.training - section.held_out - section.coarse_only, 0)


def _case_cells(case: SurrogateCase) -> tuple[int, str]:
    """Return the case's size in cells for the calibration rule, and what it was read from.

    The deck's grid (``DIMENS``/``SPECGRID``, every cell, active or not: ``ACTNUM`` often sits
    in includes); without one, the largest state variable (a field has one entry per cell).
    """
    simulator = case.config.simulator
    options = {} if simulator is None else simulator.options
    template = options.get("deck_template_file")
    if template is not None:
        case_dir = options.get("case_dir")
        try:
            found = deck_layout(
                Path(str(template)), None if case_dir is None else Path(str(case_dir))
            ).get("grid")
        except (OSError, ValueError):
            found = None
        if isinstance(found, list | tuple) and found:
            dims = [int(v) for v in found]
            return int(np.prod(dims)), "grid " + " x ".join(str(d) for d in dims) + " ="
    largest = max((v.size for v in case.layout.variables), default=1)
    return int(largest), "the largest state variable,"


def calibration_count(config: SurrogateConfig, case: SurrogateCase) -> CalibrationCount:
    """Return the build's calibration runs: ``[design] calibration``, else automatic.

    Automatic (unset) with a plugin that uses them: :data:`CALIBRATION_RUNS_SMALL` for a case
    of up to :data:`SMALL_CASE_CELLS` cells (:func:`_case_cells`), else
    :data:`CALIBRATION_RUNS_LARGE`; none for an adaptive plan (its runs are chosen by the
    surrogate; not supported yet). Without such a plugin, none. Deterministic from the
    configuration and the case, so a resume draws the same.

    Raises
    ------
    ConfigError
        If ``calibration > 0`` is given for a plugin that does not use the runs, or with an
        adaptive plan.
    """
    given = config.design.calibration
    told = _wants_calibration(config)
    wanted = bool(told)
    adaptive = config.design.plan not in BUILTIN_PLANS
    if given is not None:
        if given > 0 and told is False:
            raise ConfigError(
                f"[design] calibration = {given}: the surrogate {config.surrogate.model!r} "
                "does not calibrate a spread on runs of their own with these options "
                "(uses_calibration_runs), so they would not be used: remove calibration"
            )
        if given > 0 and adaptive:
            raise ConfigError(
                f"[design] calibration = {given} with the adaptive plan "
                f"{config.design.plan!r} is not supported yet: give calibration = 0 (the "
                "spread is then calibrated on the validation runs) or a core plan"
            )
        spare = _spare_columns(config, case) if given > 0 else None
        if spare is not None and given > spare:
            raise ConfigError(
                f"[design] calibration = {given}: the design's ensemble leaves {spare} "
                "column(s) after the held-out, training and coarse-only runs; give at most "
                f"{spare}, or 0"
            )
        return CalibrationCount(given, f"[design] calibration = {given}", False, wanted,
                                told is not None, config.design.plan == "run")  # fmt: skip
    if not wanted:
        return CalibrationCount(0, None, True, False, told is not None)
    if adaptive:
        return CalibrationCount(
            0, f"the adaptive plan {config.design.plan!r} takes none yet", True, True
        )
    cells, source = _case_cells(case)
    small = cells <= SMALL_CASE_CELLS
    runs = CALIBRATION_RUNS_SMALL if small else CALIBRATION_RUNS_LARGE
    rule = (
        f"{source} {cells:,} cells {'<=' if small else '>'} {SMALL_CASE_CELLS:,}: "
        f"{runs} runs ({'small' if small else 'large'} case)"
    )
    spare = _spare_columns(config, case)
    if spare is not None and spare < runs:
        # A finite ensemble (plans file, run): the automatic count takes what it leaves, so
        # a design that fitted before still does.
        rule += f", capped at the {spare} column(s) the design's ensemble leaves"
        runs = spare
    return CalibrationCount(runs, rule, True, True, True, config.design.plan == "run")


def draw_design(
    config: SurrogateConfig,
    case: SurrogateCase,
    rng: np.random.Generator,
    counted: CalibrationCount | None = None,
) -> StoredDesign:
    """Draw the design ``[design]`` describes, from the stream ``design``.

    Member ids: for drawn designs, the held-out runs are ``0..H-1``, the training runs
    ``H..H+T-1``, the coarse-only ones (``[design] coarse_only = N``) ``H+T..H+T+N-1`` and
    the calibration runs (:func:`calibration_count`, ``C``) ``H+T+N..H+T+N+C-1``; for
    ``file`` and ``run``, the ensemble's columns. The coarse-only and then the calibration
    members are drawn last, so the others are those of a design without them.

    Raises
    ------
    ConfigError
        If the design cannot be drawn (unbounded box, a stored prior, too few columns).
    """
    section = config.design
    n_train, n_held, n_only = section.training, section.held_out, section.coarse_only
    n_cal = (calibration_count(config, case) if counted is None else counted).runs
    layout = case.layout
    if case.optimization:
        return _control_design(config, case, rng, n_cal)
    try:
        if section.plan == "prior":
            design = sampled_design(_prior_draw(case, "prior"), n_train, n_held, rng, n_only, n_cal)
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
            design = lhs_design(box, n_train, n_held, rng, n_only, n_cal)
        elif section.plan == "file":
            assert section.ensemble_file is not None
            ensemble = _ensemble_file(section.ensemble_file, layout)
            design = ensemble_design("file", ensemble, n_held, rng, n_train, n_only, n_cal)
        else:
            assert section.run_dir is not None
            ensemble, outputs, finite = _run_columns(section.run_dir, section.iterations, case)
            # The design chooses among the members whose forecasts are finite; ids are the
            # pooled columns, so the others are recorded as failed by theirs.
            usable = np.flatnonzero(finite)
            design = ensemble_design(
                "run", {k: v[:, usable] for k, v in ensemble.items()}, n_held, rng, n_train,
                n_calibration=n_cal,
            )  # fmt: skip
            assert design.training_columns is not None
            assert design.held_out_columns is not None
            held = [int(usable[c]) for c in design.held_out_columns]
            train = [int(usable[c]) for c in design.training_columns]
            calibrating = [int(usable[c]) for c in design.calibration_columns or ()]
            return StoredDesign(
                plan="run",
                held_out_ids=tuple(held),
                training_ids=tuple(train),
                held_out=design.held_out,
                training=design.training,
                held_out_outputs=np.ascontiguousarray(outputs[:, held]),
                training_outputs=np.ascontiguousarray(outputs[:, train]),
                failed_ids=tuple(int(i) for i in np.flatnonzero(~finite)),
                # The run's forecasts again (free): further members, none trained or gated on.
                calibration_ids=tuple(calibrating),
                calibration=design.calibration,
                calibration_outputs=np.ascontiguousarray(outputs[:, calibrating])
                if calibrating
                else None,
            )
    except ValueError as error:
        raise ConfigError(f"the design cannot be drawn: {error}") from error
    if design.held_out_columns is not None and design.training_columns is not None:
        held_ids, train_ids = design.held_out_columns, design.training_columns
    else:
        held_ids = tuple(range(n_held))
        train_ids = tuple(range(n_held, n_held + n_train))
    only_ids: tuple[int, ...] = ()
    if design.coarse_only is not None:
        only_ids = design.coarse_only_columns or tuple(
            range(n_held + n_train, n_held + n_train + n_only)
        )
    calibration_ids: tuple[int, ...] = ()
    if design.calibration is not None:
        first = n_held + n_train + n_only
        calibration_ids = design.calibration_columns or tuple(range(first, first + n_cal))
    return StoredDesign(
        plan=section.plan,
        held_out_ids=held_ids,
        training_ids=train_ids,
        held_out=design.held_out,
        training=design.training,
        coarse_only_ids=only_ids,
        coarse_only=design.coarse_only,
        calibration_ids=calibration_ids,
        calibration=design.calibration,
    )


def _control_design(
    config: SurrogateConfig, case: SurrogateCase, rng: np.random.Generator, n_cal: int = 0
) -> StoredDesign:
    """Draw an optimization's design over its controls, from the stream ``design``.

    ``lhs``: a Latin hypercube over the controls' bounds (or ``[design.box]``); ``file``: the
    columns of a stored ensemble of controls. With ``[model]`` arrays each member is then
    given its realization (:meth:`SurrogateCase.with_realizations`). Member ids as for any
    drawn design: held-out ``0..H-1``, training ``H..H+T-1``.

    Raises
    ------
    ConfigError
        If the plan does not suit an optimization, or a control is unbounded.
    """
    section = config.design
    controls = [v for v in case.layout.variables if v.name not in case.realizations]
    if section.plan not in ("lhs", "file"):
        raise ConfigError(
            f"[design] plan = {section.plan!r} does not suit an optimization, which has no "
            "prior and no history-matching run: use 'lhs' over the controls' bounds, or "
            "'file' with an ensemble of controls"
        )
    unknown = sorted(set(section.box) - {v.name for v in controls})
    if unknown:
        raise ConfigError(f"[design.box] names no control: {unknown}")
    try:
        if section.plan == "lhs":
            box = {}
            for variable in controls:
                lower, upper = section.box.get(variable.name, (variable.lower, variable.upper))
                if not (math.isfinite(lower) and math.isfinite(upper)):
                    raise ConfigError(
                        f"plan 'lhs' needs bounded controls: {variable.name!r} has bounds "
                        f"[{lower}, {upper}]; give it lower and upper, or [design.box] "
                        f"{variable.name} = [lo, hi]"
                    )
                box[variable.name] = (np.full(variable.size, lower), np.full(variable.size, upper))
            design = lhs_design(box, section.training, section.held_out, rng, n_calibration=n_cal)
        else:
            assert section.ensemble_file is not None
            ensemble = _ensemble_file(section.ensemble_file, StateLayout(controls))
            design = ensemble_design(
                "file", ensemble, section.held_out, rng, section.training, n_calibration=n_cal
            )
    except ValueError as error:
        raise ConfigError(f"the design cannot be drawn: {error}") from error
    if design.held_out_columns is not None and design.training_columns is not None:
        held_ids, train_ids = design.held_out_columns, design.training_columns
    else:
        held_ids = tuple(range(section.held_out))
        train_ids = tuple(range(section.held_out, section.held_out + section.training))
    calibration_ids: tuple[int, ...] = ()
    if design.calibration is not None:
        first = section.held_out + section.training
        calibration_ids = design.calibration_columns or tuple(range(first, first + n_cal))
    return StoredDesign(
        plan=section.plan,
        held_out_ids=held_ids,
        training_ids=train_ids,
        held_out=case.with_realizations(held_ids, design.held_out),
        training=case.with_realizations(train_ids, design.training),
        calibration_ids=calibration_ids,
        calibration=None
        if design.calibration is None
        else case.with_realizations(calibration_ids, design.calibration),
    )


# ------------------------------------------------------------------------------ building


#: The case's sections that fix its runs (the method, the observations' values, the run's
#: name and executor do not).
_RUN_SECTIONS = ("state", "prior", "parameterizations", "simulator", "geophysics", "model_error")


def fingerprint_parts(
    config: SurrogateConfig, case: SurrogateCase, *, legacy: bool = False
) -> dict[str, str]:
    """Return what fixes a build's runs, part by part (a digest or a value each).

    The case's sections that shape its runs (``case.<section>``, files by content), the
    stack's provenance (``stack.<field>``: simulator, its version, its options' and deck's
    digests), the keys, the design and the seed. The plugin, its options, the gate and the
    domain's extent are left out: changing them refits on the runs already made (the runs are
    the cost). The exception: a plugin that learns from the runs' gradients
    (``uses_gradients``) adds the part ``gradients``, so switching between it and one without
    the capability is refused in the same folder (its runs were made with sensitivities, or
    not). A resume refused names the parts that differ.

    The design's files (``ensemble_file``, files its plan's options name) count by their
    content, as the stack's, so a build moved or copied with its files is the same build
    (2026-11-03; the variant smoke's cluster builds had to restamp it). A run directory
    (plan ``run``) stays by its path, as does a folder a plan's option names: their content is
    mostly outputs, which a folder's digest leaves out. ``legacy``: the design by its paths, as
    builds made before were fingerprinted (:func:`_open_build` upgrades them).
    """
    sections = case.run_sections or _RUN_SECTIONS
    document = json.loads(case.config.model_dump_json(include=set(sections)))
    parts = {f"case.{name}": _digest(_with_contents(document.get(name))) for name in sections}
    stack = case.provenance.model_dump(
        include={"simulator", "simulator_version", "options_digest", "deck_digest", "case_digest"}
    )
    parts |= {f"stack.{name}": str(value) for name, value in sorted(stack.items())}
    parts["keys"] = _digest([key_document(k) for k in case.keys])
    design = json.loads(config.design.model_dump_json())
    # The calibration runs are never in the fingerprint: drawn after every other run, they
    # change none of them, and the build reconciles its saved design with the count instead
    # (:func:`_reconciled_design`), so the fingerprint needs no plugin (a complete build is
    # found without it) and builds without calibration runs keep theirs.
    design.pop("calibration", None)
    if not design.get("options"):
        # Absent before the plans' options existed: builds keep their digest.
        design.pop("options", None)
    if not legacy:
        if design.get("ensemble_file") is not None:
            design["ensemble_file"] = _file_contents(design["ensemble_file"])
        if design.get("options"):
            # Only options naming files (resolved against the configuration); a folder stays
            # by its path, as a run directory: a folder's digest leaves its outputs out.
            design["options"] = {
                k: _file_contents(v) if _is_path_option(k, v) else v
                for k, v in sorted(design["options"].items())
            }
    # coarse_only enters with the design only when > 0 (DesignSection's serializer leaves
    # it out otherwise), so builds without coarse-only members keep their fingerprints.
    parts["design"] = _digest(design)
    parts["seed"] = str(config.surrogate.seed)
    if case.fields:
        # The runs save the fields too: absent without them.
        parts["fields"] = _digest({"fields": list(case.fields), "times": list(case.field_times),
                                   "cells": case.field_cells})  # fmt: skip
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
    if case.gradients:
        # Every run also returns its sensitivities (absent otherwise: builds keep theirs).
        parts["gradients"] = "sensitivities"
    if config.design.plan not in BUILTIN_PLANS:
        # An adaptive plan's runs are chosen by the surrogate (the user,
        # 2026-11-02): its plugin and options are part of what the runs are.
        parts["surrogate"] = _digest({
            "model": config.surrogate.model,
            "options": _with_contents(config.surrogate.options),
        })  # fmt: skip
    return parts


def _file_contents(value: object) -> object:
    """Return a file's content digest for a path naming an existing file; else the value."""
    if isinstance(value, str) and Path(value).is_file():
        return {"file_sha256": provenance.file_hash(Path(value))}
    return value


def _open_build(
    directory: SurrogateDirectory, config: SurrogateConfig, case: SurrogateCase
) -> dict[str, str]:
    """Open (or resume) a build's directory; return its fingerprint's parts.

    A build fingerprinted by its design's paths (before 2026-11-03) resumes when it matches
    them, and its state is upgraded to the content fingerprint, so it is found again from
    any copy of its files. (A file changed in place at its old path cannot be told from the
    one the runs used: the upgrade takes the file as it is now.)

    A build resumed from moved design files records their new paths in its ``config.json``
    (the design section only), so that a later ``--resume`` of the directory reads them there.

    Raises
    ------
    ConfigError
        If the directory holds another build, or anything else.
    """
    parts = fingerprint_parts(config, case)
    fingerprint = _fingerprint(parts)
    saved = directory.read_state().get(BUILD_FINGERPRINT)
    if saved is not None and saved != fingerprint:
        legacy = fingerprint_parts(config, case, legacy=True)
        if saved == _fingerprint(legacy):
            directory.write_state(**{BUILD_FINGERPRINT: fingerprint}, fingerprint_parts=parts)
            logger.info(
                "build fingerprint upgraded",
                extra={"event": "surrogate_fingerprint_upgraded", "part": "design"},
            )
    resumed = directory.open(
        kind=BUILD_FINGERPRINT,
        config_text=config.model_dump_json(indent=2),
        fingerprint=fingerprint,
        parts=parts,
    )
    if resumed:
        stored = json.loads((directory.path / "config.json").read_text())
        design = json.loads(config.design.model_dump_json())
        if stored.get("design") != design:
            # The same build (its fingerprint matched), its design's files elsewhere now.
            directory.write_config(json.dumps({**stored, "design": design}, indent=2))
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
    problems = _model_problems(config)
    # The case's keys and the design plan are checked whether or not the plugin is
    # installed, so that validation says everything a build would refuse, not only the first.
    loaded = None
    try:
        case_config = _case_config(config.case.config_file)
        keys, variances = case_keys(case_config, config.case.outputs)
        loaded = (case_config, keys, variances)
    except (ConfigError, PluginError) as error:
        problems.append(str(error))
    except (TypeError, ValueError) as error:  # e.g. a response objective's options
        problems.append(f"invalid case for a surrogate: {error}")
    try:
        design_plan(config)  # an unknown plan, its options, the surrogate it needs
    except ConfigError as error:
        problems.append(str(error))
    if problems or loaded is None:
        raise ConfigError("\n".join(problems))
    assert config.surrogate.output_dir is not None
    case = _build_case(config, loaded)
    counted = calibration_count(config, case)
    warnings = []
    notice = counted.notice()
    if notice is not None:
        # The extra runs' cost (or the calibration without them), stated wherever the plan is
        # read: apex validate, REST, the web UI, the assistant's tools and the planner.
        warnings.append(notice)
    if config.design.plan == "lhs" and case.layout.n_state > 50:
        warnings.append(
            f"a Latin hypercube over {case.layout.n_state} state entries spreads few runs over "
            "many coordinates; a field's design is better drawn from its prior (plan 'prior')"
        )
    if config.design.plan == "run":
        assert config.design.run_dir is not None
        _, _, finite = _run_columns(config.design.run_dir, config.design.iterations, case)
        available = int(np.count_nonzero(finite))
        wanted = config.design.training + config.design.held_out + counted.runs
        if available < wanted:
            raise ConfigError(
                f"{config.design.run_dir} holds {available} members with finite forecasts, "
                f"fewer than the {wanted} the design asks for"
                + (f" (with {counted.runs} calibration runs)" if counted.runs else "")
            )
    runs = (
        0
        if config.design.plan == "run"
        else config.design.training + config.design.held_out + counted.runs
    )
    if config.coarse is not None:
        # Every run also on the coarse stack, the calibration runs excepted
        # (their spread is the fine model's), and the members run on it only.
        runs = 2 * runs - (0 if config.design.plan == "run" else counted.runs)
        runs += config.design.coarse_only
    return SurrogateBuildPlan(
        name=config.surrogate.name,
        model=config.surrogate.model,
        plan=config.design.plan,
        training=config.design.training,
        held_out=config.design.held_out,
        simulator_runs=runs,
        coarse_only=config.design.coarse_only,
        calibration=counted.runs,
        calibration_rule=counted.rule,
        keys=len(case.keys),
        datatypes=sorted({k.datatype for k in case.keys}),
        measure="relative" if case.variances is None else "sigma_ratio",
        simulator=case.provenance.simulator,
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
    path: Path,
    *,
    progress: ProgressReporter | None = None,
    max_runs: int | None = None,
    stopped: Callable[[], bool] | None = None,
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
    return build_config(config, progress=progress, max_runs=max_runs, stopped=stopped)


def build_config(
    config: SurrogateConfig,
    *,
    progress: ProgressReporter | None = None,
    max_runs: int | None = None,
    stopped: Callable[[], bool] | None = None,
) -> BuildResult:
    """Start or resume a build from a resolved configuration (one process per directory).

    ``stopped`` (a job's stop decision 4) is checked between waves of
    member runs; once true the build ends as stopped, its finished members kept.

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
        return _build(
            config, progress, max_runs, _with_stop_file(config.surrogate.output_dir, stopped)
        )


#: The file ``apex stop`` and a job's stop write in the folder of what they stop.
STOP_FILE = "stop_requested"


def _with_stop_file(folder: Path, stopped: Callable[[], bool] | None) -> Callable[[], bool] | None:
    """Return ``stopped`` that also reads ``folder``'s stop file, cleared first (a job's).

    A stop asked before this start (it ended an earlier job, or was never read) is not one
    for it, so the file is removed; the caller holds the folder's lock. Without ``stopped``
    (the CLI) nothing changes: one batch, as before.
    """
    if stopped is None:
        return None
    stop_file = Path(folder) / STOP_FILE
    stop_file.unlink(missing_ok=True)
    return lambda: stopped() or stop_file.exists()


def _model_problems(config: SurrogateConfig) -> list[str]:
    """Return what is wrong with ``[surrogate] model`` and its options (empty when valid)."""
    plugin = check_options("surrogate_models", config.surrogate.model, config.surrogate.options)
    return ["[surrogate]: " + "; ".join(plugin)] if plugin else []


def _build_case(
    config: SurrogateConfig,
    loaded: tuple[
        HistoryMatchConfig | OptimizationConfig,
        tuple[DataKey, ...],
        NDArray[np.float64] | None,
    ]
    | None = None,
) -> SurrogateCase:
    """Compose a build's stack; ``loaded`` is its case and keys when already read."""
    if loaded is None:
        case_config = _case_config(config.case.config_file)
        keys, variances = case_keys(case_config, config.case.outputs)
    else:
        case_config, keys, variances = loaded
    assert config.surrogate.output_dir is not None
    case = surrogate_case(
        case_config, keys, variances, execution=config.execution,
        output_dir=config.surrogate.output_dir, coarse=config.coarse,
        fields=config.case.fields, field_times=config.case.field_times,
        field_cells=config.case.field_cells,
    )  # fmt: skip
    return _with_gradients(config, case) if _wants_gradients(config) else case


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
    coarse_only: tuple[Mapping[str, NDArray[np.float64]], NDArray[np.float64]] | None = None,
    calibration: tuple[Mapping[str, NDArray[np.float64]], NDArray[np.float64], tuple[int, ...]]
    | None = None,
) -> SnapshotSet:
    """Return the successful members of ``ids`` as snapshots of the surrogate's inputs.

    With ``coarse`` (a coarse stack's values by member), only members with both, and the
    coarse values as the set's ``coarse_outputs``; ``coarse_only`` (the coarse-only
    members' inputs and outputs, :func:`_coarse_only_snapshots`) and ``calibration`` (the
    calibration runs', :func:`_calibration_snapshots`) go with the set whole.
    """
    columns = [
        j for j, m in enumerate(ids)
        if int(m) in outputs and (coarse is None or int(m) in coarse)
    ]  # fmt: skip
    state = {k: np.ascontiguousarray(v[:, columns]) for k, v in blocks.items()}
    inputs = case.properties.ensemble(state)
    n_keys = len(case.keys)
    stacked = np.stack([outputs[int(ids[j])] for j in columns], axis=-1)
    gradients = None
    if case.gradients:
        # Each run's output is [values, Jacobian.ravel()] (SensitivityRun); no fields here.
        n_state = case.layout.n_state
        if stacked.shape[0] != n_keys * (1 + n_state):
            raise WorkflowError(
                f"the runs' outputs hold {stacked.shape[0]} numbers, not {n_keys} values and "
                f"their {n_keys} x {n_state} gradients: runs made without sensitivities"
            )
        gradients = np.ascontiguousarray(stacked[n_keys:].reshape(n_keys, n_state, len(columns)))
        stacked = stacked[:n_keys]
    values = stacked.reshape(len(case.run_keys), len(columns))
    return SnapshotSet(
        layout=case.input_layout,
        inputs={k: np.ascontiguousarray(v) for k, v in inputs.items()},
        keys=case.keys,
        outputs=np.ascontiguousarray(values[:n_keys]),
        fields=split_fields(values[n_keys:], case.fields, case.field_times, case.field_cells)
        if case.fields
        else {},
        field_times=case.field_times,
        run_ids=tuple(int(ids[j]) for j in columns),
        provenance=provenance_,
        failed=tuple(int(m) for m in failed),
        coarse_outputs=None
        if coarse is None
        else np.stack([coarse[int(ids[j])] for j in columns], axis=-1).reshape(
            len(case.keys), len(columns)
        ),
        coarse_only_inputs=None if coarse_only is None else coarse_only[0],
        coarse_only_outputs=None if coarse_only is None else coarse_only[1],
        gradients=gradients,
        calibration_inputs=None if calibration is None else calibration[0],
        calibration_outputs=None if calibration is None else calibration[1],
        calibration_ids=() if calibration is None else calibration[2],
    )


def _calibration_snapshots(
    case: SurrogateCase, design: StoredDesign, values: Mapping[int, NDArray[np.float64]]
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], tuple[int, ...]] | None:
    """Return the successful calibration runs as the surrogate's inputs, outputs and ids.

    ``({variable: (size, n)}, (n_keys, n), (n,))`` in the design's order, the inputs mapped
    by the case's parameterizations (the scalar keys only: no fields); ``None`` if none
    succeeded or there are none.
    """
    if design.calibration is None:
        return None
    columns = [j for j, m in enumerate(design.calibration_ids) if int(m) in values]
    if not columns:
        return None
    state = {k: np.ascontiguousarray(v[:, columns]) for k, v in design.calibration.items()}
    inputs = case.properties.ensemble(state)
    ids = tuple(int(design.calibration_ids[j]) for j in columns)
    n_keys = len(case.keys)
    # A run's values cover the run keys (fields after the scalar keys): the scalar ones only.
    outputs = np.stack([np.asarray(values[m])[:n_keys] for m in ids], axis=-1)
    return (
        {k: np.ascontiguousarray(v) for k, v in inputs.items()},
        np.ascontiguousarray(outputs),
        ids,
    )


def _coarse_only_snapshots(
    case: SurrogateCase, design: StoredDesign, values: Mapping[int, NDArray[np.float64]]
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]] | None:
    """Return the successful coarse-only members as the surrogate's inputs and outputs.

    ``({variable: (size, n)}, (n_keys, n))`` in the design's order, the inputs mapped by
    the case's parameterizations (not the coarse stack's: the surrogate's inputs are the
    fine stack's); ``None`` if none succeeded.
    """
    if design.coarse_only is None:
        return None
    columns = [j for j, m in enumerate(design.coarse_only_ids) if int(m) in values]
    if not columns:
        return None
    state = {k: np.ascontiguousarray(v[:, columns]) for k, v in design.coarse_only.items()}
    inputs = case.properties.ensemble(state)
    outputs = np.stack([values[int(design.coarse_only_ids[j])] for j in columns], axis=-1).reshape(
        len(case.keys), len(columns)
    )
    return (
        {k: np.ascontiguousarray(v) for k, v in inputs.items()},
        np.ascontiguousarray(outputs),
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
    stopped: Callable[[], bool] | None = None,
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
            directory, case.simulator, case.executor, members, case.run_keys, label="design",
            keep_simulations=keep, max_runs=budget, progress=progress,
            stopped=stopped,
        )  # fmt: skip
        if budget is not None:
            budget = max(budget - (len(directory.journal()) - before), 0)
        if not outcome.complete or len(design.training_ids) >= n_train:
            return design, outcome
        if stopped is not None and stopped():
            # Before the next fit and batch (a fit may take hours): the resume continues.
            return design, dataclasses.replace(outcome, stopped=True)
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


def _model_files(directory: SurrogateDirectory) -> tuple[Path, Path]:
    return directory.model, directory.model.with_suffix(".npz")


def _calibration_digest(design: StoredDesign) -> str:
    """Return a digest of a design's calibration runs: their ids and inputs, in order."""
    digest = hashlib.sha256(b"apex-calibration-runs-1\n")
    digest.update(np.asarray(design.calibration_ids, dtype="<i8").tobytes())
    for name in sorted(design.calibration or {}):
        value = np.ascontiguousarray((design.calibration or {})[name], dtype="<f8")
        digest.update(name.encode() + repr(value.shape).encode() + value.tobytes())
    return digest.hexdigest()


def _reconciled_design(
    config: SurrogateConfig,
    case: SurrogateCase,
    directory: SurrogateDirectory,
    design: StoredDesign,
    counted: CalibrationCount,
) -> StoredDesign:
    """Return the saved design with ``counted.runs`` calibration runs (saved if it changed).

    The calibration runs are drawn last, so a design with another count has the same other
    runs: the design is drawn again from the seed (checked to give the same other runs) with
    the new count, or its calibration runs are dropped (``0``). Members are reused by id, so
    a drawn plan (``prior``, ``lhs``) refuses the change once a member of the old or the new
    calibration ids was made (journalled, or saved by a worker before a stop): the id would
    name other inputs. With ``file`` and ``run`` an id is its ensemble's column, so the
    change is safe.

    Raises
    ------
    ConfigError
        If the count changes on a drawn plan after calibration runs were made there, or the
        design drawn again does not give the saved runs.
    """
    runs = counted.runs
    have = len(design.calibration_ids)
    if have == runs:
        return design
    if runs == 0:
        changed = dataclasses.replace(
            design, calibration_ids=(), calibration=None, calibration_outputs=None
        )
        if have:
            # What was made, so that drawing the same runs again later is known to be safe.
            directory.write_state(calibration_dropped=_calibration_digest(design))
    else:
        drawn = draw_design(config, case, build_streams(config.surrogate.seed)["design"], counted)
        same = drawn.held_out_ids == design.held_out_ids and drawn.training_ids == (
            design.training_ids
        ) and drawn.coarse_only_ids == design.coarse_only_ids and all(
            np.array_equal(drawn.held_out[k], design.held_out[k])
            and np.array_equal(drawn.training[k], design.training[k])
            for k in design.held_out
        )  # fmt: skip
        if not same:
            raise ConfigError(
                f"{directory.path}: the design drawn again does not give its saved runs; "
                "build in another output_dir"
            )
        if config.design.plan not in ("file", "run"):
            journal = directory.journal()
            ids = set(design.calibration_ids) | set(drawn.calibration_ids)
            made = sorted(
                m for m in ids if m in journal or directory.member_file(m).is_file()
            )  # fmt: skip
            # The runs dropped before, drawn again exactly (the count they were made with):
            # their ids name the same inputs.
            again = not have and directory.read_state().get(
                "calibration_dropped"
            ) == _calibration_digest(drawn)
            if made and not again:
                raise ConfigError(
                    f"{directory.path}: calibration runs of another count were made here "
                    f"(members {', '.join(map(str, made[:5]))}"
                    + (", ..." if len(made) > 5 else "")
                    + f"); {runs} would reuse their outputs for other inputs: keep the count "
                    "they were made with, give [design] calibration = 0, or build in another "
                    "output_dir"
                )
        changed = dataclasses.replace(
            design, calibration_ids=drawn.calibration_ids, calibration=drawn.calibration,
            calibration_outputs=drawn.calibration_outputs,
        )  # fmt: skip
    directory.save_design(changed, case.layout)
    logger.info(
        "the design's calibration runs changed",
        extra={"event": "surrogate_design_calibration", "before": have, "after": runs},
    )
    return changed


def _lenient_count(config: SurrogateConfig, case: SurrogateCase) -> CalibrationCount:
    """Return the calibration count for a completion check, never failing on the plugin.

    A plugin that refuses its options (since the build: say, an update) or a design source
    that cannot be read (a moved ensemble) leaves the count unknown, so a complete build is
    still found; a build that has to run reports those failures itself. Any other exception
    from the plugin's constructor (a ``RuntimeError``, ``KeyError``: a bug) propagates, loudly,
    as it would in the build. It fails open: a
    complete build of plan ``run`` whose run directory is gone is returned as made, even one
    that, read, would now be given calibration runs (``plan_build`` refuses it).
    """
    try:
        return calibration_count(config, case)
    except (ConfigError, OSError):
        return CalibrationCount(0, None, True, False, known=False)


def _same_calibration(
    config: SurrogateConfig, directory: SurrogateDirectory, counted: CalibrationCount
) -> bool:
    """Whether a complete build's saved design has the calibration runs ``counted`` asks for.

    A plugin that is not installed cannot be asked: a complete build is then taken as made
    (unless ``[design] calibration`` gives another count than it has). So a Laplace build
    made before calibration runs existed is not complete once it would get them: its next
    build adds them and refits, as its plan and charge count them.
    """
    design = directory.load_design()
    have = 0 if design is None else len(design.calibration_ids)
    if not counted.known:
        given = config.design.calibration
        return given is None or given == have
    return have == counted.runs


def runs_made(build_dir: Path) -> int:
    """Return the simulator runs a build has made of its current design (resume's charge).

    Members journalled whose ids are the saved design's (held-out, training, calibration;
    the coarse stack's, coarse-only too): runs of a design changed since (calibration runs
    dropped) are not left to pay for.
    """
    directory = SurrogateDirectory(build_dir)
    design = directory.load_design()
    if design is None:
        return 0
    fine = {*design.held_out_ids, *design.training_ids, *design.calibration_ids}
    made = sum(1 for m in directory.journal() if m in fine)
    if (build_dir / COARSE).is_dir():
        coarse = {*design.held_out_ids, *design.training_ids, *design.coarse_only_ids}
        made += sum(1 for m in SurrogateDirectory(build_dir / COARSE).journal() if m in coarse)
    return made


def _completed_build(directory: SurrogateDirectory, config: SurrogateConfig) -> BuildResult | None:
    """Return the complete build of ``config`` in ``directory``, read only; else ``None``.

    The marker of :func:`_build`'s early return (complete, the same fit, its summary and
    model files), and the same runs as :func:`_open_build` would accept: their fingerprint,
    computed only once the marker is there (it composes the case's stack).
    """
    state = directory.read_state()
    if not (state.get("complete") and state.get("fit_fingerprint") == _fit_fingerprint(config)):
        return None
    saved = directory.read_json(SUMMARY)
    if saved is None or not all(f.is_file() for f in _model_files(directory)):
        return None
    case = _build_case(config)
    if not _same_calibration(config, directory, _lenient_count(config, case)):
        return None
    parts = fingerprint_parts(config, case)
    if state.get(BUILD_FINGERPRINT) != _fingerprint(parts):
        return None
    return BuildResult(directory.path, SurrogateBuildSummary.model_validate(saved))


def _build(
    config: SurrogateConfig,
    progress: ProgressReporter | None,
    max_runs: int | None,
    stopped: Callable[[], bool] | None = None,
) -> BuildResult:
    section = config.surrogate
    assert section.output_dir is not None
    directory = SurrogateDirectory(section.output_dir)
    # The model is created only at the fit: an unknown plugin or a misspelt option is
    # refused here, before the design's runs (and on resume, before the rest of them).
    # A complete build needs no plugin: its saved summary is returned as before, though
    # the plugin has since been uninstalled.
    problems = _model_problems(config)
    if problems:
        done = _completed_build(directory, config)
        if done is not None:
            return done
        raise ConfigError("\n".join(problems))
    case = _build_case(config)
    _open_build(directory, config, case)
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
    model_files = _model_files(directory)
    lenient = _lenient_count(config, case)
    if (
        state.get("complete")
        and state.get("fit_fingerprint") == fit_key
        and _same_calibration(config, directory, lenient)
    ):
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
    # Once for the build (it may read a run's or an ensemble's columns: plans run, file):
    # the completion check's, unless it could not ask the plugin (then strictly, to say why).
    counted = lenient if lenient.known else calibration_count(config, case)
    streams = build_streams(section.seed)
    design = directory.load_design()
    if design is not None and adaptive and "adaptive" not in directory.read_state():
        design = None  # saved, but its state was never written: draw it again (same draws)
    if design is None:
        if isinstance(plan, BuiltinDesign):
            design = draw_design(config, case, streams["design"], counted)
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
    if isinstance(plan, BuiltinDesign):
        design = _reconciled_design(config, case, directory, design, counted)
    notice = counted.notice()
    if notice is not None:
        logger.info(
            notice,
            extra={"event": "surrogate_calibration_runs", "runs": len(design.calibration_ids),
                   "rule": counted.rule, "automatic": counted.automatic},
        )  # fmt: skip
    plan_provenance = case.provenance.model_copy(update={
        "plan": design.plan, "seed": section.seed,
        "source": None if config.design.run_dir is None else str(config.design.run_dir),
    })  # fmt: skip
    keep = config.output.keep_simulations
    calibration_values: dict[int, NDArray[np.float64]] = {}
    calibration_failed: dict[int, str] = {}
    fallback: str | None = None
    coarse_values: dict[int, NDArray[np.float64]] | None = None
    coarse_only: tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]] | None = None
    coarse_only_failed: dict[int, str] = {}
    if design.plan == "run":
        assert design.held_out_outputs is not None
        assert design.training_outputs is not None
        values = {int(m): design.held_out_outputs[:, j] for j, m in enumerate(design.held_out_ids)}
        values |= {int(m): design.training_outputs[:, j] for j, m in enumerate(design.training_ids)}
        if design.calibration_outputs is not None:
            calibration_values = {
                int(m): design.calibration_outputs[:, j]
                for j, m in enumerate(design.calibration_ids)
            }
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
            if design.calibration is not None:
                # Full runs like the others, last (a resume reuses them as the others).
                members += _members(design.calibration_ids, design.calibration)
            outcome = run_members(
                directory, case.simulator, case.executor, members, case.run_keys, label="design",
                keep_simulations=keep, max_runs=max_runs, progress=progress, stopped=stopped,
            )  # fmt: skip
        else:
            design, outcome = _run_adaptive(
                config, case, directory, plan, design, plan_provenance,
                keep=keep, max_runs=max_runs, progress=progress, stopped=stopped,
            )  # fmt: skip
        if not outcome.complete:
            # An adaptive design's later batches are pending too (not drawn yet).
            later = max(config.design.training - len(design.training_ids), 0)
            lost = {m: v for m, v in outcome.failed.items() if m not in design.calibration_ids}
            summary = _summary(config, design, case, lost, len(outcome.pending) + later,
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
        # The calibration runs apart: never trained or gated on, their failures recorded
        # apart from the runs' (as the coarse-only members').
        for member in design.calibration_ids:
            if member in values:
                calibration_values[member] = values.pop(member)
            elif member in failed:
                calibration_failed[member] = failed.pop(member)
                logger.warning(
                    "a calibration run failed",
                    extra={"event": "surrogate_calibration_failed", "member": member,
                           "reason": calibration_failed[member]},
                )  # fmt: skip
        if case.coarse_simulator is not None:
            # Every design run on the coarse stack too, in a folder of its own,
            # within what is left of max_runs; then the coarse-only members ([design]
            # coarse_only), in the same folder, so a resume reuses them as the others.
            budget = (
                None if max_runs is None
                else max(max_runs - (len(directory.journal()) - journal_before), 0)
            )  # fmt: skip
            only_members = (
                [] if design.coarse_only is None
                else _members(design.coarse_only_ids, design.coarse_only)
            )  # fmt: skip
            coarse_outcome = run_members(
                SurrogateDirectory(directory.path / COARSE),
                case.coarse_simulator, case.executor,
                # Only the members whose fine run succeeded: a pair needs both.
                [m for m in _members(design.held_out_ids, design.held_out)
                 + _members(design.training_ids, design.training) if m[0] in values]
                + only_members,
                case.keys, label="coarse", keep_simulations=keep, max_runs=budget,
                progress=progress, stopped=stopped,
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
            only_ids = set(design.coarse_only_ids)
            coarse_values = {m: v for m, v in coarse_outcome.values.items() if m not in only_ids}
            for member, message in coarse_outcome.failed.items():
                if member in only_ids:
                    # Recorded apart from the runs' failures (``failed`` are fine runs lost to
                    # the fit or the gate), never silently dropped.
                    coarse_only_failed[member] = message
                    logger.warning(
                        "a coarse-only run failed",
                        extra={"event": "surrogate_coarse_only_failed", "member": member,
                               "reason": message},
                    )  # fmt: skip
                    continue
                # A run is a training or held-out pair: without its coarse run it is lost.
                failed.setdefault(member, f"its coarse run failed: {message}")
                values.pop(member, None)
            simulator_runs += len(coarse_outcome.values) + len(coarse_outcome.failed)
            if design.coarse_only is not None:
                check_success("coarse-only", len(only_ids), len(coarse_only_failed),
                              config.design.min_success_fraction)  # fmt: skip
                coarse_only = _coarse_only_snapshots(case, design, coarse_outcome.values)
        for label, ids in (("held-out", design.held_out_ids), ("training", design.training_ids)):
            check_success(label, len(ids), sum(int(m) in failed for m in ids),
                          config.design.min_success_fraction)  # fmt: skip
        total = len(design.calibration_ids)
        if total and (total - len(calibration_failed)) / total < config.design.min_success_fraction:
            # Not the build's runs: a spread calibrated on the validation runs (the plugin
            # warns) rather than a build lost after all its good runs.
            logger.warning(
                "%d of the %d calibration runs failed: fewer than %.0f%% succeeded; the "
                "spread is calibrated on the validation runs instead",
                len(calibration_failed), total, 100 * config.design.min_success_fraction,
                extra={"event": "surrogate_calibration_runs_failed",
                       "failed": sorted(calibration_failed)},
            )  # fmt: skip
            calibration_values = {}
            fallback = (
                f"{len(calibration_failed)} of the {total} calibration runs failed: the spread "
                "is calibrated on the validation runs"
            )
    if stopped is not None and stopped():
        # Every run made, the fit not started (it may take hours): ended as stopped, nothing
        # pending; the resume fits.
        summary = _summary(config, design, case, failed, 0, simulator_runs).model_copy(
            update={"calibration_fallback": fallback}
        )
        directory.write_json(SUMMARY, json.loads(summary.model_dump_json()))
        directory.write_state(complete=False)
        logger.info("build stopped", extra={"event": "surrogate_stopped", "pending": 0})
        return BuildResult(directory.path, summary)
    failed_ids = sorted(failed)
    calibration = _calibration_snapshots(case, design, calibration_values)
    held_out = _snapshot_set(case, design.held_out_ids, design.held_out, values,
                             plan_provenance, failed_ids, coarse_values, coarse_only,
                             calibration)  # fmt: skip
    training = _snapshot_set(case, design.training_ids, design.training, values,
                             plan_provenance, failed_ids, coarse_values, coarse_only,
                             calibration)  # fmt: skip
    # Every successful run, in id order: the snapshot set a refit starts from (phase 3).
    everything = _snapshot_set(
        case,
        list(design.held_out_ids) + list(design.training_ids),
        {k: np.concatenate([design.held_out[k], design.training[k]], axis=1)
         for k in design.held_out},
        values, plan_provenance, failed_ids, coarse_values, coarse_only, calibration,
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
            objective=case.objective_of(),
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
        # An optimization's: the box its checks draw within, kept with the
        # model so that a copied model can be checked without its build directory.
        control_box=_box_of(design, case) if case.optimization else None,
        # And the inputs the model saw (decision 6): a prediction's scenario is checked
        # against them without the build's snapshots.
        input_sizes=None
        if not case.optimization
        else {v.name: v.size for v in everything.layout.variables},
        control_inputs=None
        if not case.optimization
        else {
            name: (np.min(values, axis=1), np.max(values, axis=1))
            for name, values in everything.inputs.items()
            if name in _control_inputs(case)
        },
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
        "status": "complete",
        "gate": fitted.gate, "identity": stored.identity,
        "coarse_only_runs": 0 if coarse_only is None else coarse_only[1].shape[1],
        "coarse_only_failed": sorted(coarse_only_failed),
        "calibration_runs": len(calibration_values),
        "calibration_failed": sorted(calibration_failed),
        "calibration_fallback": fallback,
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


def _simulator_files(case: HistoryMatchConfig | OptimizationConfig) -> dict[str, Path]:
    if case.simulator is None:
        return {}
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


@dataclass(frozen=True)
class PredictionCost:
    """What predicting one member costs, against a full run (a build's facts and summary).

    Attributes
    ----------
    simulation
        What the model runs per member besides itself: ``none``, ``coarse`` (a
        coarse-corrected model's coarse forecast) or ``early`` (an early-start
        model's simulation up to its date).
    seconds
        One member's prediction, with the coarse run's median time when it needs one;
        ``None`` when unknown (an early-start model: its early run is not timed by the build).
    speedup
        The median full run over ``seconds``; ``None`` when either is unknown.
    """

    simulation: Literal["none", "coarse", "early"]
    seconds: float | None
    speedup: float | None


def prediction_cost(path: Path, summary: SurrogateBuildSummary) -> PredictionCost:
    """Return a build's cost per predicted member (:class:`PredictionCost`)."""
    try:
        plugin = surrogate_models.factory(summary.plugin)
    except PluginError:  # not installed here: judged by its folder only
        plugin = None
    full = summary.simulator_median_seconds
    own = summary.prediction_seconds
    if plugin is None:  # what it simulates per member is unknown here: no speedup
        return PredictionCost("none", own, None)
    if callable(getattr(plugin, "predict_from_early", None)):
        return PredictionCost("early", None, None)
    coarse = path / COARSE
    # Coarse runs count only for a model that corrects a coarse forecast (a coarse/ folder of
    # another model's build holds runs its predictions never make).
    if coarse.is_dir() and callable(getattr(plugin, "predict_corrected", None)):
        times = [r.seconds for r in SurrogateDirectory(coarse).journal().values()
                 if r.ok and r.seconds is not None]  # fmt: skip
        seconds = None if own is None or not times else own + float(np.median(times))
        return PredictionCost("coarse", seconds, full / seconds if full and seconds else None)
    return PredictionCost("none", own, full / own if full and own else None)


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
    stopped: Callable[[], bool] | None = None,
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
    case_config: HistoryMatchConfig | OptimizationConfig
    if against.is_dir():
        if optimization_services.holds_optimization(against):
            raise ConfigError(
                f"{against} is an optimization run: check an optimization's surrogate against "
                "its configuration (controls drawn within their bounds)"
            )
        case_config = composition.snapshot_config(RunDirectory(against))
        source = "posterior" if iteration is None else f"iteration:{iteration}"
    else:
        case_config = _case_config(against)
        source = "controls" if isinstance(case_config, OptimizationConfig) else "prior"
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
            # The fields it was trained and gated on, on the same cells.
            fields=manifest.fields, field_times=manifest.field_times,
            field_cells=_field_cells(trained) if manifest.fields else None,
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
            box = _design_box(Path(model_path), case, manifest) if source == "controls" else None
            if source == "controls" and loaded.domain.extent != "box":
                # The hypercube fills the box; a hull's or an ellipsoid's domain is narrower,
                # the more so the more controls, and runs outside it are simulated but not
                # gated (too few inside fail the check, min_inside).
                logger.warning(
                    "surrogate_check_box_wider_than_domain",
                    extra={
                        "extent": loaded.domain.extent,
                        "runs": runs,
                        "controls": sum(
                            v.size for v in case.layout.variables if v.name not in case.realizations
                        ),
                    },
                )
            design, trained_on = _check_design(case, trained, against, source, iteration,
                                               runs, seed, design_box=box)  # fmt: skip
            directory.save_design(design, case.layout)
            directory.write_state(trained_on=trained_on)
        trained_on = [int(m) for m in directory.read_state().get("trained_on", [])]
        made_before = len(directory.journal())
        outcome = run_members(
            directory, case.simulator, case.executor,
            _members(design.held_out_ids, design.held_out),
            case.run_keys, label="check", max_runs=max_runs, progress=progress, stopped=stopped,
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
                progress=progress, stopped=stopped,
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
        # An early-start model's early data: the check runs' own values up to its date.
        companion = (
            early_data(loaded.model, snapshots)
            if isinstance(loaded.model, EarlyStartSurrogate)
            else snapshots.coarse_outputs
        )
        found = check_members(loaded.domain, loaded.model, snapshots.inputs, snapshots.run_ids,
                              case.keys, coarse=companion)  # fmt: skip
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
            elif isinstance(loaded.model, EarlyStartSurrogate):
                predicted = np.ascontiguousarray(loaded.model.predict_from_early(
                    gated.inputs, early_data(loaded.model, gated), case.keys
                ))  # fmt: skip
            else:
                predicted = np.ascontiguousarray(loaded.model.predict(gated.inputs, case.keys))
            kinds = {d.datatype: d.kind for d in (manifest.gate.datatypes if manifest.gate else [])
                     if d.kind is not None}  # fmt: skip
            fields = None
            if case.fields and isinstance(loaded.model, FieldSurrogate):
                names = tuple(sorted(gated.fields))
                rebuilt = loaded.model.predict_fields(gated.inputs, names, gated.field_times)
                fields = {n: (np.asarray(rebuilt[n]), gated.fields[n]) for n in names}
            objective = case.objective_of()
            objectives = (
                None if objective is None
                else (column_objectives(objective, predicted),
                      column_objectives(objective, gated.outputs))
            )  # fmt: skip
            rows = gated_rows(loaded.model, case.keys)
            scored = (
                (predicted, gated.outputs, tuple(case.keys), case.variances) if rows is None
                else (np.ascontiguousarray(predicted[rows]),
                      np.ascontiguousarray(gated.outputs[rows]),
                      tuple(case.keys[r] for r in rows),
                      None if case.variances is None else case.variances[rows])
            )  # fmt: skip
            verdict = gate(scored[0], scored[1], scored[2], gated.run_ids, fields=fields,
                           tolerances=tolerances, variances=scored[3], kinds=kinds,
                           failed=sorted(failed_ids), objectives=objectives)  # fmt: skip
        else:
            verdict = GateResult(held_out=[], failed=sorted(failed_ids), datatypes=[],
                                 tolerances=tolerances, passed=False)  # fmt: skip
        notes = []
        if source == "controls" and loaded.domain.extent != "box":
            notes.append(
                f"the controls were drawn over the design's box, and the model's domain is its "
                f"{loaded.domain.extent} (narrower): {len(outside)} of {snapshots.n_runs} runs "
                "fell outside it and were not gated"
            )
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


def _field_cells(trained: SnapshotSet) -> int | None:
    """Return the cells of the fields a model's snapshots hold (``None``: none recorded)."""
    if not trained.fields:
        return None
    return int(next(iter(trained.fields.values())).shape[0])


def _design_box(
    model_path: Path, case: SurrogateCase, manifest: SurrogateManifest
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None:
    """Return each control's entrywise range over a model's design runs.

    ``(lower (size,), upper (size,))`` per control: what the build's Latin hypercube (or
    stored ensemble) spanned, held-out and training runs together. From the manifest's
    ``control_box`` (models built since it was recorded), else from the build's
    ``design.npz`` two folders above the manifest; ``None`` without either.
    """
    if manifest.control_box is not None:
        return {
            name: (np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
            for name, (low, high) in manifest.control_box.items()
        }
    build = Path(model_path).resolve().parent.parent
    if not (build / "design.npz").is_file():
        return None
    design = SurrogateDirectory(build).load_design()
    if design is None:
        return None
    return _box_of(design, case)


def _box_of(
    design: StoredDesign, case: SurrogateCase
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Return each control's entrywise range over a design's held-out and training runs."""
    box = {}
    for name in design.held_out:
        if name in case.realizations:
            continue
        both = np.concatenate([design.held_out[name], design.training[name]], axis=1)
        box[name] = (np.min(both, axis=1), np.max(both, axis=1))
    return box


def _check_design(
    case: SurrogateCase,
    trained: SnapshotSet,
    against: Path,
    source: str,
    iteration: int | None,
    runs: int,
    seed: int,
    design_box: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
) -> tuple[StoredDesign, list[int]]:
    """Choose a check's members, none equal to one of the model's snapshots.

    Returns the design (its runs as ``held_out``) and the members left out as trained on.

    Raises
    ------
    ConfigError
        If fewer than ``runs`` members are new to the model.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    if source == "controls":
        # An optimization's: a Latin hypercube over the box its design spanned
        # (the build's [design.box], not the controls' wider bounds: runs outside the model's
        # domain are not gated, and too few inside would fail the check), each member given
        # its realization as in a build.
        controls = [v for v in case.layout.variables if v.name not in case.realizations]
        if design_box is None:
            raise ConfigError(
                "a check of an optimization's surrogate draws its controls within the box its "
                "build's design spanned, which the model records neither in its manifest (built "
                "before it did) nor beside it (its build's design.npz, two folders above the "
                "manifest): rebuild it, or check it in its build directory"
            )
        missing = [v.name for v in controls if v.name not in design_box]
        if missing:
            raise ConfigError(f"the model's design holds no controls {missing}")
        for variable in controls:
            # The model's box against this optimization's controls (a manifest's, or its
            # build's design.npz): a block of another size, or bounds out of order, would
            # draw a misaligned box (the review of 2026-11-03).
            low, high = (np.asarray(b, dtype=np.float64) for b in design_box[variable.name])
            if low.shape != (variable.size,) or high.shape != (variable.size,):
                raise ConfigError(
                    f"the model's design box of {variable.name!r} has {low.size} lower and "
                    f"{high.size} upper bounds, and the optimization's control has "
                    f"{variable.size} entries: the model was built on other controls; rebuild "
                    "it on this optimization"
                )
            if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))
                    and np.all(low <= high)):  # fmt: skip
                raise ConfigError(
                    f"the model's design box of {variable.name!r} is not finite and ordered "
                    "(lower <= upper): its manifest or design.npz is damaged; rebuild it"
                )
        lower = np.concatenate([design_box[v.name][0] for v in controls])
        upper = np.concatenate([design_box[v.name][1] for v in controls])
        sample = latin_hypercube(lower, upper, runs, rng)
        offsets = np.cumsum([0] + [v.size for v in controls])
        drawn_controls = {
            v.name: np.ascontiguousarray(sample[offsets[i] : offsets[i + 1]])
            for i, v in enumerate(controls)
        }
        ids = np.arange(runs)
        blocks = case.with_realizations(ids.tolist(), drawn_controls)
        what = f"the {runs} controls drawn within the model's design box"
    elif source == "prior":
        drawn = composition.draw_prior_ensemble(case.history_matching, case.layout, runs, rng)
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
    policy without the snapshots to refit from, or for a model that trains on coarse runs.

    Raises
    ------
    ConfigError
        Naming what is refused and why.
    """
    section = config.surrogate
    assert section is not None
    manifest = model.manifest
    name = model.path
    if isinstance(model.model, EarlyStartSurrogate) and isinstance(
        model.model, CoarseCorrectedSurrogate
    ):
        raise ConfigError(
            f"the surrogate {name} both corrects a coarse simulation and continues each "
            "member's early run: a run makes one kind of companion run, not both"
        )
    if isinstance(model.model, EarlyStartSurrogate):
        # Each member runs to the early date, then the model continues it.
        early = set(model.model.early_keys())
        if not any(key not in early for key in observations.keys):
            raise ConfigError(
                f"the surrogate {name} continues each member's run from "
                f"{model.model.early_until}, and every observation of this run lies up to "
                "that date: the simulator's early runs would be the whole forecast"
            )
    if section.on_check_failure == "refit" and getattr(model.model, "uses_gradients", False):
        # A run's checks are plain simulator runs (no sensitivities): a refit set would have
        # no gradients for such a model to learn from.
        raise ConfigError(
            f'[surrogate] on_check_failure = "refit": the surrogate {name} learns from the '
            'runs\' gradients, which a run\'s checks do not make; use "stop" or "simulator", '
            "or rebuild it"
        )
    if section.on_check_failure == "refit" and getattr(model.model, "trains_on_coarse", False):
        # A run makes no coarse runs for such a model, and its refit set would need them.
        raise ConfigError(
            f'[surrogate] on_check_failure = "refit": the surrogate {name} trains on coarse '
            'runs, which a run does not make; use "stop" or "simulator", or rebuild it'
        )
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
    resolved = options_resolved_in(
        current_options(manifest.plugin, manifest.options), Path(section.model_file).parent
    )
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


# ===================================================================== prediction (§6)


def prediction_surrogate(
    section: PredictionSurrogateSection,
    case: HistoryMatchConfig,
    simulator: Simulator,
    executor: object,
    property_map: PropertyMap,
    request: SeriesRequest,
    execution: str,
    *,
    layout: StateLayout | None = None,
    controls: Mapping[str, NDArray[np.float64]] | None = None,
) -> tuple[PredictionSurrogate, SeriesRequest]:
    """Compose a prediction's surrogate and resolve its axis against the model's keys.

    ``case`` is the run's configuration as the prediction runs it (its simulator, nothing
    that wraps it), ``simulator`` the prediction's stack before its parameterizations,
    ``executor`` (named ``execution``) where its simulator runs (an early-start model's
    early runs). The axis:
    the requested vectors matched against the model's scalar datatypes, and its times (by
    default the model's report times), every one a key of the model: none beyond its last
    report date (its horizon).

    With a control scenario (decision 6, §3): ``controls`` the blocks every
    member is given (after ``[[controls.parameterizations]]``), ``layout`` the run's state.
    The model must have been built over controls (an optimization-shaped case of the
    forecast: its manifest records the controls' box): on inputs that are the run's state
    after its parameterizations and these controls, by name and size, each control within
    the range the design spanned (:func:`_refuse_other_controls`). A model built over
    controls is refused without a scenario, and one built without them with one.

    Raises
    ------
    ConfigError
        If the model is refused (its gate or last check failed, another simulator stack or
        case, a coarse-corrected model), the axis lies outside its keys, or an early-start
        model's early runs could not run.
    """
    path = Path(section.model_file)
    loaded = active_model(path)
    manifest = loaded.manifest
    model = loaded.model
    if isinstance(model, CoarseCorrectedSurrogate):
        raise ConfigError(
            f"the surrogate {path} corrects a coarse simulation: predictions with it are not "
            "supported (its coarse stack would have to run to the horizon)"
        )
    if manifest.gate is None or not manifest.gate.passed:
        failures = [] if manifest.gate is None else manifest.gate.failures()
        raise ConfigError(
            f"the surrogate {path} did not pass its gate ({'; '.join(failures) or 'none'}): "
            "rebuild it"
        )
    check = last_check(path, loaded.identity)
    if check is not None and not check.passed:
        raise ConfigError(
            f"the surrogate {path} failed its last check ({check.id}): rebuild or refit it"
        )
    controls = dict(controls or {})
    built_over = manifest.control_box is not None
    if not built_over and manifest.provenance.case_digest == _OPTIMIZATION_CASE_DIGEST:
        raise ConfigError(
            f"the surrogate {path} was built on an optimization's case before its controls' "
            "box was recorded: rebuild it to use it in a prediction"
        )
    if built_over and not controls:
        raise ConfigError(
            f"the surrogate {path} was built over controls "
            f"({', '.join(sorted(manifest.control_box or {}))}): give the prediction's scenario "
            "as [controls]"
        )
    if controls and not built_over:
        raise ConfigError(
            f"the surrogate {path} was built without controls, and this prediction sets "
            f"[controls] ({', '.join(sorted(controls))}): it would predict as if the controls "
            "were its build's; build one over the controls, on an optimization-shaped case of "
            "the forecast (decision 6)"
        )
    # The prediction's deck must be the build's: a changed schedule (another deck, other
    # controls) is another stack, which the model has not seen.
    provenance = stack_provenance(case, simulator)
    if built_over:
        # Its case must have been an optimization's, whose case digest is that of an empty
        # document (no [geophysics] or [model_error]); a prediction's case never has them
        # either (remote_config). Anything else (a hand-edited manifest, another kind of
        # case) is refused; then the digest is left out, and the simulator, its options and
        # the deck are compared as for any model.
        if manifest.provenance.case_digest != _OPTIMIZATION_CASE_DIGEST:
            raise ConfigError(
                f"the surrogate {path} records controls but was not built on an "
                "optimization's case (its case digest is another's): rebuild it on an "
                "optimization-shaped case of the forecast"
            )
        provenance = provenance.model_copy(update={"case_digest": None})
    mismatches = manifest.provenance.stack_mismatches(provenance)
    if mismatches:
        raise ConfigError(
            f"the surrogate {path} was built for another simulator stack or case than this "
            f"prediction's ({', '.join(mismatches)} differ): build it on the prediction's deck "
            "and simulator options, from a case without [geophysics] or [model_error] (a "
            "prediction forecasts the simulator's series alone)"
        )
    scalar = [key for key in model.keys if key.index is None]
    datatypes = sorted({key.datatype for key in scalar})
    # In time order: ISO dates by date, step labels by number ("10" after "9").
    times = sorted({key.time for key in scalar}, key=time_position)
    try:
        vectors = match_vectors(request.vectors, datatypes)
    except ValueError as error:
        raise ConfigError(
            f"[prediction.outputs] vectors: {error} among the surrogate's ({', '.join(datatypes)})"
        ) from error
    wanted = list(times) if request.times is None else list(request.times)
    beyond = [t for t in wanted if t not in times]
    if beyond:
        raise ConfigError(
            f"[prediction.outputs] times: {beyond[0]} is not a report time of the surrogate "
            f"{path} (its keys run {times[0]} to {times[-1]}, its horizon)"
        )
    known = set(scalar)
    missing = [DataKey(t, v) for v in vectors for t in wanted if DataKey(t, v) not in known]
    if missing:
        raise ConfigError(f"the surrogate {path} does not answer {missing[0]}")
    # The run's own check, now: its domain's keys and times (the gated ones), before any
    # early run spends simulator time (the coder's review).
    outside = check_request(loaded.domain, [DataKey(t, v) for v in vectors for t in wanted])
    if not outside.inside:
        raise ConfigError(f"the surrogate {path} cannot forecast this prediction: "
                          f"{outside.message()}")  # fmt: skip
    if built_over:
        if layout is None:
            raise ValueError("prediction_surrogate needs the run's state layout with controls")
        _refuse_other_controls(path, manifest, layout, property_map, controls)
    if isinstance(model, EarlyStartSurrogate):
        composition.refuse_early_runs(path, model.early_until, executor, simulator, execution)
    names = set(controls)

    def inputs(blocks: Mapping[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        # The state by the run's parameterizations; the scenario's controls as they are
        # (mapped by their own already, as every member is given them).
        state = {k: v for k, v in blocks.items() if k not in names}
        properties = state if property_map.identity else property_map.ensemble(state)
        properties = {**properties, **{k: v for k, v in blocks.items() if k in names}}
        return {
            name: np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            for name, value in properties.items()
        }

    assert manifest.gate is not None
    surrogate = PredictionSurrogate(
        simulator=loaded.simulator,
        model=model,
        domain=loaded.domain,
        identity=loaded.identity,
        executor=executors.create("serial")
        if section.workers is None
        else executors.create("process", workers=section.workers),
        inputs=inputs,
        simulator_of=functools.partial(parameterized, properties=property_map),
        outside=section.outside,
        tolerances=manifest.gate.tolerances,
        kinds={d.datatype: d.kind for d in manifest.gate.datatypes if d.kind is not None},
    )
    return surrogate, SeriesRequest(vectors, tuple(wanted))


def _control_inputs(case: SurrogateCase) -> frozenset[str]:
    """Return the surrogate's inputs that come from an optimization's controls.

    The controls, followed through the case's parameterizations in order: an output of a
    parameterization reading a control is one (``control_schedule``'s steps), an output of
    one reading only realizations is not (a realization mapped to another name, e.g. ln k
    to ``permx``).
    """
    found = {v.name for v in case.layout.variables if v.name not in case.realizations}
    for parameterization in case.properties.parameterizations:
        if set(parameterization.variables) & found:
            found |= set(parameterization.outputs)
    return frozenset(found) & frozenset(case.input_layout.names)


def _refuse_other_controls(
    path: Path,
    manifest: SurrogateManifest,
    layout: StateLayout,
    property_map: PropertyMap,
    controls: Mapping[str, NDArray[np.float64]],
) -> None:
    """Refuse a model built over other inputs or controls than a prediction's scenario.

    decision 6, §3: the model's inputs must be the run's state after its
    parameterizations and the scenario's controls after theirs, by name and size, and each
    control, as the model sees it (after its parameterizations), must lie within the range
    the design spanned entry by entry (a surrogate does not extrapolate the controls; other
    knots of a schedule can put the same values elsewhere). From the manifest's
    ``input_sizes`` and ``control_inputs``; for a model built before it recorded them, from
    its build's snapshots.

    Raises
    ------
    ConfigError
        Naming what differs, or why an older model's snapshots are needed and unreadable.
    """
    given = {v.name: v.size for v in _input_layout(layout, property_map).variables}
    given |= {name: int(np.asarray(v).size) for name, v in controls.items()}
    if manifest.input_sizes is not None and manifest.control_inputs is not None:
        fitted = dict(manifest.input_sizes)
        ranges = {
            name: (np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64))
            for name, (low, high) in manifest.control_inputs.items()
        }
        for name, (low, high) in ranges.items():
            # As control_box's (_check_design): a damaged manifest is named, not indexed.
            size = fitted.get(name)
            if (
                size is None
                or low.shape != (size,)
                or high.shape != (size,)
                or not (np.all(np.isfinite(low)) and np.all(np.isfinite(high)))
                or not np.all(low <= high)
            ):
                raise ConfigError(
                    f"the surrogate {path}'s recorded range of the input {name!r} is not "
                    "finite, ordered and of the input's size: its manifest is damaged; "
                    "rebuild it"
                )
    else:
        try:
            trained = _model_snapshots(path, manifest)
        except ConfigError as error:
            raise ConfigError(
                f"the surrogate {path} was built before its manifest recorded its inputs; a "
                "prediction's scenario is then checked against its build's snapshots, which "
                f"cannot be read ({error}): rebuild it, or use it in its build directory"
            ) from error
        fitted = {v.name: v.size for v in trained.layout.variables}
        ranges = {
            name: (np.min(values, axis=1), np.max(values, axis=1))
            for name, values in trained.inputs.items()
            if name in controls
        }
    if given != fitted:
        raise ConfigError(
            f"the surrogate {path} was fitted on inputs {_sized(fitted)}, and this prediction "
            f"gives its members {_sized(given)} (the run's state after its parameterizations, "
            "and the scenario's controls after theirs): build it on the run's realizations "
            "and these controls"
        )
    for name, values in sorted(controls.items()):
        if name not in ranges:
            raise ConfigError(
                f"the surrogate {path} takes {name!r} as a realization, not a control: this "
                "prediction's scenario sets it"
            )
        low, high = ranges[name]
        value = np.asarray(values, dtype=np.float64).reshape(-1)
        outside = [i for i in range(value.size) if not low[i] <= value[i] <= high[i]]
        if outside:
            i = outside[0]
            raise ConfigError(
                f"the scenario's {name}[{i}] = {value[i]:g} (after its parameterizations) lies "
                f"outside the range the surrogate {path} was built over ({low[i]:g} to "
                f"{high[i]:g}; {len(outside)} of its {value.size} entries): a surrogate does "
                "not extrapolate the controls; widen the build's design"
            )


#: The case digest of an optimization's case (an empty document: ``stack_provenance``).
_OPTIMIZATION_CASE_DIGEST = _digest(_with_contents({}))


def _sized(sizes: Mapping[str, int]) -> str:
    """Return ``{name: size}`` as ``name (size), ...`` in name order."""
    return ", ".join(f"{name} ({size})" for name, size in sorted(sizes.items()))


# ===================================================================== optimization (§6)


def optimization_surrogate(
    config: OptimizationConfig, simulation: SimulatedObjective
) -> SurrogateObjective:
    """Return the objective an optimization with ``[surrogate]`` evaluates.

    ``simulation`` is the optimization's own simulated objective; it becomes the source of
    the checks and of the reported optimum (rows recorded ``simulator``), and the surrogate's
    objective shares its journal under a context that includes the model's identity.

    Raises
    ------
    ConfigError
        If the model cannot be used: its gate or its last check failed, it was built for
        another simulator stack or other inputs (controls, parameterizations, model arrays),
        or it does not answer the objective's keys.
    """
    section = config.surrogate
    assert section is not None
    assert config.simulator is not None
    path = Path(section.model_file)
    loaded = load_surrogate(path)
    manifest = loaded.manifest
    if isinstance(loaded.model, EarlyStartSurrogate):
        # Each evaluation would need its own early simulation first.
        raise ConfigError(
            f"the surrogate {path} continues each member's early simulation (an early-start "
            "model): optimizations cannot use it"
        )
    if manifest.gate is None or not manifest.gate.passed:
        failures = [] if manifest.gate is None else manifest.gate.failures()
        raise ConfigError(
            f"the surrogate {path} did not pass its gate ({'; '.join(failures) or 'none'}): "
            "rebuild it"
        )
    check = last_check(path, loaded.identity)
    if check is not None and not check.passed:
        raise ConfigError(
            f"the surrogate {path} failed its last check ({check.id}: "
            f"{'; '.join(check.notes) or 'not passed'}): rebuild it"
        )
    raw = simulators.create(config.simulator.name, **config.simulator.options)
    mismatches = manifest.provenance.stack_mismatches(stack_provenance(config, raw))
    if mismatches:
        raise ConfigError(
            f"the surrogate {path} was built for another simulator stack "
            f"({', '.join(mismatches)} differ)"
        )
    model = dict(simulation.model)
    properties = optimization_services.control_properties(config, model)
    inputs = _input_layout(optimization_services.control_layout(config, model), properties)
    trained = _model_snapshots(path, manifest)
    if tuple(trained.layout.names) != tuple(inputs.names) or [
        v.size for v in trained.layout.variables
    ] != [v.size for v in inputs.variables]:
        raise ConfigError(
            f"the surrogate {path} was fitted on inputs {list(trained.layout.names)}, and this "
            f"optimization gives its simulations {list(inputs.names)} (controls, "
            "parameterizations and [model] arrays must be the build's)"
        )
    outside = check_request(loaded.domain, simulation.keys)
    if not outside.inside:
        raise ConfigError(
            f"the surrogate {path} cannot evaluate this objective: {outside.message()}"
        )
    tolerances = (
        manifest.gate.tolerances.tightened(relative=section.relative)
        if section.relative
        else manifest.gate.tolerances
    )
    kinds = {d.datatype: d.kind for d in manifest.gate.datatypes if d.kind is not None}
    surrogate_simulator: Simulator = SurrogateSimulator(
        loaded.model,
        loaded.domain,
        identity=loaded.identity,
        loader=functools.partial(fitted_model, str(path.resolve()), loaded.identity),
        guard=refuse_outside,
    )
    if not properties.identity:
        surrogate_simulator = parameterized(surrogate_simulator, properties)
    workers = section.workers
    executor = (
        executors.create("serial")
        if workers is None
        else executors.create("process", workers=workers)
    )
    context = _digest({"plain": simulation.context, "surrogate": loaded.identity})
    surrogate = SimulatedObjective(
        simulation.objective,
        surrogate_simulator,
        executor,
        simulation.run_dir,
        simulation.blocks,
        model,
        keep_simulations="none",
        progress=simulation.progress,
        context=context,
        journal=simulation.journal,
        source="surrogate",
    )
    simulation.source = "simulator"
    blocks = simulation.blocks

    def inside(controls: NDArray[np.float64], reals: NDArray[np.int64]) -> NDArray[np.bool_]:
        state = {name: np.ascontiguousarray(controls[part]) for name, part in blocks}
        state.update({n: np.ascontiguousarray(a[:, reals]) for n, a in model.items()})
        mapped = properties.ensemble(state)
        ids = list(range(controls.shape[1]))
        found = check_members(loaded.domain, loaded.model, mapped, ids)
        flagged = set(found.members)
        return np.array([j not in flagged for j in ids], dtype=bool)

    assert simulation.run_dir is not None
    return SurrogateObjective(
        surrogate,
        simulation,
        simulation.run_dir,
        identity=loaded.identity,
        inside=inside,
        outside=section.outside,
        check_every=section.check_every,
        on_check_failure=section.on_check_failure,
        tolerances=tolerances,
        kinds=kinds,
        max_evaluations=section.max_evaluations,
    )
