"""Composition root: build a history-matching run from a configuration file.

Configuration names are turned into concrete simulators, executors and methods here, in
the services layer, by name through the plugin registries; the layers below never choose
an implementation.
"""

import csv
import datetime as dt
import functools
import importlib
import inspect
import itertools
import json
import logging
import math
import os
import re
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

# Imported for their side effect of registering plugins (the OPM and Eclipse adapters
# need the reservoir extra only when they run).
import apex.algorithms.data_compression
import apex.algorithms.facies
import apex.algorithms.geophysics
import apex.algorithms.geostatistics.variogram
import apex.algorithms.metrics
import apex.algorithms.priors
import apex.algorithms.validation
import apex.execution.ert
import apex.execution.local
import apex.reporting.figures
import apex.services.geomodeling_skills
import apex.simulation.analytic
import apex.simulation.conversion
import apex.simulation.eclipse
import apex.simulation.ert
import apex.simulation.opm_flow
import apex.storage.pet_skill
import apex.storage.priors
import apex.workflows.model_error  # noqa: F401
from apex.algorithms.data_assimilation import (
    DampingSchedule,
    EnsembleAlgorithm,
    LMDampingSchedule,
    damping_schemes,
    localizations,
    methods,
    update_strategies,
)
from apex.algorithms.geophysics import (
    AVOAttribute,
    DisplacementKernel,
    PPReflectivity,
    avo_attributes,
    displacement_kernels,
    petro_elastic_models,
    reflectivity,
)
from apex.core.diagnostics import DiagnosticsPlan
from apex.core.errors import (
    ConfigError,
    PluginError,
    RunStopped,
    WorkflowError,
    validation_problems,
)
from apex.core.metrics import PerformanceMetric
from apex.core.plugins import (
    data_transforms,
    executors,
    metrics,
    model_error_models,
    parameterizations,
    priors,
    simulators,
    stopping_rules,
)
from apex.core.progress import ProgressReporter
from apex.core.simulation import (
    DataTransform,
    FieldResponses,
    ModelErrorModel,
    Parameterization,
    PriorGenerator,
    RemoteSpec,
    Simulator,
)
from apex.core.validation import StoppingRule
from apex.domain.config import (
    AVOOptions,
    DisplacementOptions,
    FileWaveletSection,
    GeophysicsSection,
    GravityOptions,
    HistoryMatchConfig,
    MethodSection,
    MetricSection,
    OverburdenPressureFile,
    PluginSection,
    RegionAroundWellsSection,
    RegionFileSection,
    StationFileSection,
)
from apex.domain.data_split import DataSplit, SplitExploration, resolve
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.results import HistoryMatchResult
from apex.domain.state import StateLayout, StateVariable
from apex.reporting.summary import run_summary
from apex.simulation.deck import (
    case_dir_files,
    case_folder,
    deck_contents,
    deck_layout,
    missing_summary_wells,
    restart_schedule,
)
from apex.simulation.geophysics import GeophysicsSimulator, Grid
from apex.simulation.model_error import ModelErrorSimulator
from apex.storage.inputs import read_observation_section, read_truth
from apex.storage.locks import DirectoryLocked, exclusive, in_use
from apex.storage.priors import read_array
from apex.storage.run_directory import RunDirectory, upgrade_metric_name
from apex.storage.run_results import Iterate, RunResults
from apex.workflows.geophysics import (
    PARAMETER_TRANSFORMS,
    CellTemperature,
    GeophysicsModel,
    PemParameter,
    build_geophysics_model,
)
from apex.workflows.history_matching import DATA_MISMATCH, HistoryMatchingWorkflow, RandomStreams
from apex.workflows.parameterization import PropertyMap, StackedProperties

logger = logging.getLogger(__name__)

#: Option names treated as paths and resolved against the configuration's directory.
PATH_OPTION_SUFFIXES = ("_file", "_dir", "_path")


def load_config(path: Path) -> HistoryMatchConfig:
    """Read and validate a TOML configuration, resolving relative paths.

    Raises
    ------
    ConfigError
        If the file cannot be read or does not match the schema.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    try:
        config = HistoryMatchConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid configuration {path}:\n{validation_problems(error)}") from error
    return _resolve_paths(config, path.resolve().parent)


def _resolve_paths(config: HistoryMatchConfig, base: Path) -> HistoryMatchConfig:
    """Return the configuration with its paths resolved against ``base``, and its defaults.

    Every route a configuration takes (a file, a REST or tool submission's document) goes
    through here: remote mode's defaults are set before any identity (run ids, the
    catalogue) is taken of it.
    """
    return executor_defaults(_resolved_paths(config, base))


def executor_defaults(config: HistoryMatchConfig) -> HistoryMatchConfig:
    """Return the configuration with its executor's defaults.

    An executor may set defaults of the configuration (``config_defaults``; the ssh
    executor's remote mode keeps only failed members' files). Every route a configuration
    takes goes through here before any identity is taken of it. An unknown executor is
    left to validation.
    """
    try:
        factory = executors.factory(config.execution.name)
    except PluginError:
        # Left to validation; logged (once), since an executor's package missing here
        # would change the defaulted configuration, and so a run's id.
        _warn_unknown_executor(config.execution.name)
        return config
    defaults = getattr(factory, "config_defaults", None)
    return config if defaults is None else defaults(config)


@functools.cache
def _warn_unknown_executor(name: str) -> None:
    from apex.services.daemon import stale_entry_points

    stale = "; this environment lacks entry points its checkout declares: run uv sync"
    hint = f"{stale} --inexact --all-packages" if stale_entry_points() else ""
    logger.warning("executor %r is not installed: its defaults are not applied%s", name, hint,
                   extra={"event": "executor_unknown"})  # fmt: skip


def _resolved_paths(config: HistoryMatchConfig, base: Path) -> HistoryMatchConfig:
    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    def resolve_options(section: PluginSection) -> dict[str, Any]:
        return {
            key: str(resolve(Path(value)))
            if key.endswith(PATH_OPTION_SUFFIXES) and isinstance(value, str)
            else value
            for key, value in section.options.items()
        }

    output_dir = resolve(config.run.output_dir or Path("runs") / config.run.name)
    prior = config.prior.model_copy(
        update={
            "generator": config.prior.generator.model_copy(
                update={"options": resolve_options(config.prior.generator)}
            ),
            "variables": {
                name: section.model_copy(update={"options": resolve_options(section)})
                for name, section in config.prior.variables.items()
            },
            "groups": {
                name: section.model_copy(update={"options": resolve_options(section)})
                for name, section in config.prior.groups.items()
            },
        }
    )
    return config.model_copy(
        update={
            "run": config.run.model_copy(update={"output_dir": output_dir}),
            "prior": prior,
            "observations": config.observations.model_copy(
                update={
                    "file": None
                    if config.observations.file is None
                    else resolve(config.observations.file),
                    "fields": [
                        field.model_copy(update={"file": resolve(field.file)})
                        for field in config.observations.fields
                    ],
                }
            ),
            "data_transforms": [
                section.model_copy(update={"options": resolve_options(section)})
                for section in config.data_transforms
            ],
            "parameterizations": [
                section.model_copy(update={"options": resolve_options(section)})
                for section in config.parameterizations
            ],
            "case": config.case.model_copy(
                update={
                    "truth_file": None
                    if config.case.truth_file is None
                    else resolve(config.case.truth_file)
                }
            ),
            "metrics": None
            if config.metrics is None
            else [
                # constraint_cost's function may read a file.
                section.model_copy(
                    update={
                        "options": _resolve_nested(resolve_options(section), resolve, "function")
                    }
                )
                for section in config.metrics
            ],
            "simulator": config.simulator.model_copy(
                update={"options": resolve_options(config.simulator)}
            ),
            "model_error": None
            if config.model_error is None
            else config.model_error.model_copy(
                update={"options": resolve_options(config.model_error)}
            ),
            "geophysics": None
            if config.geophysics is None
            else _resolve_geophysics(config.geophysics, resolve),
            # e.g. the full update's state_scaling_file (none of the other strategies'
            # options is a path, so existing configurations resolve as before).
            "method": config.method.model_copy(
                update={
                    **{
                        part: None
                        if getattr(config.method, part) is None
                        else getattr(config.method, part).model_copy(
                            # A localization's entries may name files (distance_entries).
                            update={
                                "options": _resolve_nested(
                                    _resolve_levels(
                                        resolve_options(getattr(config.method, part)),
                                        resolve,
                                        "entries",
                                    ),
                                    resolve,
                                    "constraints",
                                )
                            }
                        )
                        for part in ("strategy", "localization")
                    },
                    # A multilevel method's levels may name files (coarse decks).
                    "options": _resolve_levels(config.method.options, resolve),
                }
            ),
        }
    )


def _resolve_levels(
    options: Mapping[str, Any], resolve: Callable[[Path], Path], table: str = "levels"
) -> dict[str, Any]:
    """Resolve the ``*_file`` keys inside the list of tables ``table`` of plugin options.

    A multilevel method's ``levels``, a localization's ``entries``.
    """
    result = dict(options)
    levels = options.get(table)
    if isinstance(levels, list):
        result[table] = [
            {
                key: str(resolve(Path(value)))
                if key.endswith(PATH_OPTION_SUFFIXES) and isinstance(value, str)
                else value
                for key, value in level.items()
            }
            if isinstance(level, dict)
            else level
            for level in levels
        ]
    return result


def _resolve_nested(
    options: Mapping[str, Any], resolve: Callable[[Path], Path], key: str
) -> dict[str, Any]:
    """Resolve the ``*_file`` keys at any depth under ``options[key]``.

    For gmac's constraints' functions.
    """

    def walk(value: object, name: str = "") -> object:
        if isinstance(value, Mapping):
            return {k: walk(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str) and name.endswith(PATH_OPTION_SUFFIXES):
            return str(resolve(Path(value)))
        return value

    result = dict(options)
    if key in options:
        result[key] = walk(options[key])
    return result


#: Keys of a method's level table that are for the simulator.
LEVEL_SIMULATOR_KEYS = ("upscale", "deck_template_file", "forward_file", "matrix")


def coarse_member_ids(config: HistoryMatchConfig) -> frozenset[int]:
    """Member ids of a multilevel run's levels that run another model than the fine one.

    A level is coarse if it has simulator keys (``upscale``, its own deck or operator).
    Empty for other methods.
    """
    levels = config.method.options.get("levels")
    if not isinstance(levels, list):
        return frozenset()
    ids: set[int] = set()
    first = 0
    for level in levels:
        size = int(level.get("members", 0))
        if any(key in level for key in LEVEL_SIMULATOR_KEYS):
            ids.update(range(first, first + size))
        first += size
    return frozenset(ids)


def _simulator_options(config: HistoryMatchConfig) -> dict[str, Any]:
    """Return the simulator's options, plus the method's fidelity levels.

    A multilevel method's levels become the simulator's ``levels`` option (each level's
    first member id, size and simulator keys).

    Raises
    ------
    ConfigError
        If the levels' members do not add up to the ensemble size, or they need a
        simulator that cannot run fidelity levels.
    """
    options = dict(config.simulator.options)
    levels = config.method.options.get("levels")
    if not isinstance(levels, list):
        return options
    for index, level in enumerate(levels):
        # Before the sums below: a level without members read as 0 (the release tester).
        if not isinstance(level, dict) or "members" not in level:
            raise ConfigError(
                f"[method] levels: level {index} needs members (its number of members) and "
                "weight, e.g. levels = [{ members = 20, weight = 1.0 }, ...], coarsest first"
            )
    sizes = [int(level["members"]) for level in levels]
    if config.prior.ensemble_size is not None and sum(sizes) != config.prior.ensemble_size:
        raise ConfigError(
            f"[method] levels have {sum(sizes)} members, [prior] ensemble_size is "
            f"{config.prior.ensemble_size}"
        )
    firsts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    simulator_levels = [
        {
            "first": int(first),
            "size": size,
            **{k: v for k, v in level.items() if k in LEVEL_SIMULATOR_KEYS},
        }
        for first, size, level in zip(firsts, sizes, levels, strict=True)
    ]
    needs_simulator = any(len(level) > 2 for level in simulator_levels)
    accepts = "levels" in inspect.signature(simulators.factory(config.simulator.name)).parameters
    if needs_simulator and not accepts:
        raise ConfigError(
            f"simulator {config.simulator.name!r} cannot run fidelity levels (upscale, "
            "deck_template_file); use eclipse, opm_flow or linear"
        )
    if accepts:
        options["levels"] = simulator_levels
    return options


@dataclass
class Run:
    """A configured run, ready to start or resume."""

    config: HistoryMatchConfig
    workflow: HistoryMatchingWorkflow
    method: EnsembleAlgorithm
    run_dir: RunDirectory
    #: What [geophysics] cannot confirm before the run (for the plan).
    warnings: tuple[str, ...] = ()

    def execute(self, prior: np.ndarray | None, max_steps: int | None) -> HistoryMatchResult:
        """Start (with ``prior``) or continue (``prior=None``) the run."""
        return self.workflow.run(self.method, prior, max_steps=max_steps)


def build_layout(config: HistoryMatchConfig) -> StateLayout:
    """Build the state layout ``[[state.variables]]`` describes."""
    return StateLayout(
        [
            StateVariable(
                v.name,
                v.size,
                -math.inf if v.lower is None else v.lower,
                math.inf if v.upper is None else v.upper,
                None if v.grid is None else tuple(v.grid),
            )
            for v in config.state.variables
        ]
    )


@dataclass(frozen=True)
class ForecastStack:
    """The simulator a run forecasts with, and what building it found."""

    simulator: Simulator
    warnings: tuple[str, ...]
    #: State variables the forecast reads that have no true value (MEC's, the PEM's).
    auxiliary: frozenset[str]
    #: The geophysics model the stack forecasts with, if any (for the transforms' layout).
    geophysics: "GeophysicsModel | None" = None


def forecast_stack(
    config: HistoryMatchConfig,
    layout: StateLayout | None = None,
    observations: ObservationSet | None = None,
    *,
    checks: bool = True,
) -> ForecastStack:
    """Build the simulator stack a run forecasts with: the simulator and its wrappers.

    The one place that builds it: the run here, and a remote member's worker on its host
    (``checks = False``: the run already made the deck's checks and logged its
    warnings), so both run the same stack.

    Raises
    ------
    ConfigError
        If a plugin is unknown or refuses its options.
    """
    layout = layout if layout is not None else build_layout(config)
    if observations is None and (checks or config.model_error is not None):
        # Only the checks and the model-error model read them: a remote worker on a stack
        # without either needs no copy of the observations.
        observations = read_observation_section(config.observations)
    simulator: Simulator = simulators.create(config.simulator.name, **_simulator_options(config))
    # geophysics wraps the deck simulator, and model error wraps the result
    # (its residuals cover scalar keys only, so the field keys pass through).
    geophysics = build_geophysics(config)
    warnings: list[str] = []
    # a coupled simulator computes field data itself; the rest of the
    # field keys are [geophysics]'s. Asked of the registry's simulator, before any wrapper
    # (a wrapper would hide the declaration).
    provided = simulator.field_datatypes() if isinstance(simulator, FieldResponses) else {}
    if any(size < 1 for size in provided.values()):
        raise ConfigError(f"the simulator declares an empty field: {dict(provided)}")
    if geophysics is not None:
        both = sorted(d for d in provided if geophysics.owns(DataKey("", d, 0)))
        if both:
            # [geophysics] would take these keys at run time and drop the simulator's values.
            raise ConfigError(
                f"field data {', '.join(both)} are computed by the simulator and named by "
                "[geophysics] too: rename the [geophysics] field"
            )
    other_fields: list[DataKey] = []
    if checks:
        assert observations is not None  # read above for the checks
        # Undeclared field data stay the simulator's or [geophysics]'s to answer, as before
        # (the analytic and ERT simulators answer field keys without declaring them).
        other_fields = check_field_responses(observations.keys, provided)
    if geophysics is not None:
        if checks:
            geophysics.check_observations(other_fields)
            # Raises when a vintage certainly has none, or every member would fail on
            # the deck's output; the rest (what the text cannot confirm, aliasing) is
            # logged and planned.
            warnings = [
                *geophysics_restart_warnings(config),
                *geophysics_deck_warnings(config, geophysics),
            ]
            warnings += geophysics.warnings()
            for warning in warnings:
                logger.warning("geophysics_unconfirmed", extra={"warning": warning})
        simulator = GeophysicsSimulator(simulator, geophysics)
    model_error = None if observations is None else build_model_error(config, layout, observations)
    if model_error is not None and checks:
        check_model_error_summary(config, model_error)
    # the model-error parameters have no true value; metrics leave them out.
    auxiliary = frozenset() if model_error is None else frozenset({model_error.state_variable})
    if geophysics is not None:
        # The uncertain petro-elastic parameters have no true value either.
        auxiliary |= frozenset(geophysics.state_variables)
    if model_error is not None:
        simulator = ModelErrorSimulator(simulator, model_error)
    return ForecastStack(simulator, tuple(warnings), auxiliary, geophysics)


def check_field_responses(keys: Sequence[DataKey], provided: Mapping[str, int]) -> list[DataKey]:
    """Check observed field keys against a simulator's own field data.

    Returns the field keys of datatypes the simulator does not compute (left to
    ``[geophysics]``), in order.

    Only the indices are checked here; a field observed at a date the simulator does not
    report fails when it is read, in every member, as a ``ConfigError``.

    Raises
    ------
    ConfigError
        If a field the simulator computes is not observed whole at a time: every index
        from 0 to its size - 1, once.
    """
    others: list[DataKey] = []
    indices: dict[tuple[str, str], list[int]] = {}
    for key in keys:
        if key.index is None:
            continue
        if key.datatype in provided:
            indices.setdefault((key.datatype, key.time), []).append(key.index)
        else:
            others.append(key)
    problems = [
        f"{datatype}@{time} is observed at {len(observed)} indices ({min(observed)} to "
        f"{max(observed)}, {len(set(observed))} distinct); the simulator's field has indices "
        f"0 to {provided[datatype] - 1}, each once"
        for (datatype, time), observed in indices.items()
        if sorted(observed) != list(range(provided[datatype]))
    ]
    if problems:
        more = f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""
        raise ConfigError(
            "observed field data the simulator cannot forecast: " + "; ".join(problems[:5]) + more
        )
    return others


def build_run(config: HistoryMatchConfig, progress: ProgressReporter | None = None) -> Run:
    """Build the workflow and method a configuration describes.

    ``progress`` receives the run's progress; by default nothing is shown.

    Raises
    ------
    ConfigError
        If a named plugin is unknown or its options are invalid.
    """
    config = executor_defaults(config)
    layout = build_layout(config)
    observations = read_observation_section(config.observations)
    assert config.run.output_dir is not None  # set by load_config
    run_dir = RunDirectory(config.run.output_dir)
    try:
        stack = forecast_stack(config, layout, observations)
        simulator, geophysics_warnings = stack.simulator, list(stack.warnings)
        auxiliary = stack.auxiliary
        executor = executors.create(config.execution.name, **config.execution.options)
        check_executor(executor, simulator)
        prepare_remote(executor, simulator, config)
        parameterizations = build_parameterizations(config)
        method = _build_method(config, layout, parameterizations)
        transforms = build_transforms(config)
        # ERT runs an ERT case's members in batches, so they go together.
        if (config.simulator.name == "ert") != (config.execution.name == "ert"):
            raise ConfigError(
                "the 'ert' simulator and the 'ert' executor go together (ERT runs the "
                "case's members in batches); use both or neither"
            )
        _check_pgs_rules(config)
        if parameterizations and getattr(method, "uses_sensitivities", False):
            raise ConfigError(
                "the adjoint update needs gradients with respect to the state; a "
                "parameterization's simulator gives them with respect to its properties"
            )
        if getattr(method, "uses_sensitivities", False):
            # The adjoint update needs the simulator's gradients, on raw data rows.
            if not hasattr(simulator, "run_with_sensitivities"):
                raise ConfigError(
                    f"the adjoint update (sensitivities = true) needs a simulator with "
                    f"gradients; {config.simulator.name!r} has none"
                )
            if transforms:
                raise ConfigError("the adjoint update does not support data transforms yet")
        # A strategy that cannot work with the ensemble's size says so now, not at its
        # first update (#216: adaptive_rows' default threshold with 7 members).
        # The same for its localization (a fixed 3/sqrt(N) threshold needs 10 members).
        strategy = getattr(method, "strategy", None)
        for part in (strategy, getattr(strategy, "localization", None)):
            check_members = getattr(part, "check_members", None)
            if callable(check_members) and config.prior.ensemble_size is not None:
                check_members(config.prior.ensemble_size)
        performance = build_metrics(config)
        plan = build_diagnostics_plan(config, method)
        split, stopping = build_validation(config, observations, method)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid plugin configuration: {error}") from error
    truth = None
    saved_truth = run_dir.checkpoints / "truth.npz"
    if saved_truth.exists():  # a started run scores against the truth it began with
        truth = run_dir.load_arrays("truth")["truth"]
    elif config.case.truth_file is not None:
        truth = read_truth(config.case.truth_file, layout, auxiliary)
    try:
        workflow = HistoryMatchingWorkflow(
            layout=layout,
            observations=observations,
            simulator=simulator,
            executor=executor,
            run_dir=run_dir,
            streams=RandomStreams.from_seed(config.run.seed),
            min_success_fraction=config.failures.min_success_fraction,
            transforms=transforms,
            save_rejected_trials=config.output.save_rejected_trials,
            keep_simulations=config.output.keep_simulations,
            save_raw_forecasts=config.output.save_raw_forecasts,
            metrics=performance,
            truth=truth,
            diagnostics_plan=plan,
            split=split,
            stopping=stopping,
            progress=progress,
            parameterizations=parameterizations,
            auxiliary_variables=auxiliary,
        )
    except ValueError as error:  # a transform or metric that does not fit the case
        raise ConfigError(f"invalid configuration: {error}") from error
    return Run(config, workflow, method, run_dir, tuple(geophysics_warnings))


def build_validation(
    config: HistoryMatchConfig, observations: ObservationSet, method: EnsembleAlgorithm
) -> tuple[DataSplit | None, StoppingRule | None]:
    """Resolve ``[validation]`` into data roles and a CV stopping rule.

    Raises
    ------
    ConfigError
        If cross validation is configured for a method with a fixed schedule (ES-MDA).
    ValueError
        If a role selects nothing, selects field data, or roles overlap.
    """
    section = config.validation
    if section is None:
        return None, None
    iterative = getattr(method, "iterative", None)
    if section.cv is not None and iterative is None:
        raise ConfigError(
            f"method {config.method.name!r} does not declare `iterative`; "
            "EnsembleAlgorithm plugins must set it to say whether cross validation applies"
        )
    if section.cv is not None and not iterative:
        raise ConfigError(
            f"method {config.method.name!r} runs a fixed schedule; cross validation "
            "([validation] cv) needs an iterative method such as gies"
        )
    split = resolve(
        observations,
        cv=section.cv,
        test=section.test,
        history_matching=section.history_matching,
    )
    if section.cv is None:
        return split, None
    rule = section.stopping_rule()
    try:
        return split, stopping_rules.create(rule.name, **rule.options)
    except PluginError as error:
        raise ConfigError(f"[validation] stopping: {error}") from error


def build_diagnostics_plan(
    config: HistoryMatchConfig, method: EnsembleAlgorithm
) -> DiagnosticsPlan:
    """Resolve which diagnostics the run saves.

    From ``[output] diagnostics`` and the ``diagnostics`` tables of the method, strategy
    and localization sections; a plugin's own table overrides ``[output]`` for it.

    Raises
    ------
    ValueError
        For a name no configured plugin produces, or a ``diagnostics`` table on a
        section whose plugin produces none.
    """
    parts = {
        "method": (config.method, method),
        "strategy": (config.method.strategy, getattr(method, "strategy", None)),
        "localization": (
            config.method.localization,
            getattr(getattr(method, "strategy", None), "localization", None),
        ),
    }
    available: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    for role, (section, plugin) in parts.items():
        declared = dict(getattr(plugin, "available_diagnostics", {}) or {})
        if declared:
            available[role] = declared
        if section is not None and section.diagnostics is not None:
            if not declared:
                raise ValueError(f"[method] {role} {section.name!r} produces no diagnostics")
            overrides[role] = section.diagnostics
    return DiagnosticsPlan.build(
        available,
        config.output.diagnostics,
        overrides,
        max_mb=config.output.diagnostics_max_mb,
        every=config.output.diagnostics_every,
    )


#: Metrics a run evaluates when it configures none. SSIM needs its grid, so it
#: is only evaluated when configured.
DEFAULT_METRICS = {
    "real": ("normalized_data_mismatch",),
    "synthetic": ("normalized_data_mismatch", "rmse"),
}


def _resolve_geophysics(
    section: GeophysicsSection, resolve: Callable[[Path], Path]
) -> GeophysicsSection:
    """Return ``[geophysics]`` with its files resolved against the configuration's folder."""
    pressure = section.overburden_pressure
    if isinstance(pressure, OverburdenPressureFile):
        pressure = pressure.model_copy(update={"file": resolve(pressure.file)})
    data = []
    for datum in section.data:
        options: AVOOptions | GravityOptions | DisplacementOptions
        if isinstance(datum.options, AVOOptions):
            wavelet = datum.options.wavelet
            if isinstance(wavelet, FileWaveletSection):
                wavelet = wavelet.model_copy(update={"file": resolve(wavelet.file)})
            region = datum.options.region
            if isinstance(region, RegionFileSection):
                region = region.model_copy(update={"file": resolve(region.file)})
            options = datum.options.model_copy(update={"wavelet": wavelet, "region": region})
        else:
            stations = datum.options.stations
            if isinstance(stations, StationFileSection):
                stations = stations.model_copy(update={"file": resolve(stations.file)})
            elif stations.seabed_file is not None:
                stations = stations.model_copy(
                    update={"seabed_file": resolve(stations.seabed_file)}
                )
            options = datum.options.model_copy(update={"stations": stations})
        data.append(datum.model_copy(update={"options": options}))
    return section.model_copy(
        update={
            "grid_file": resolve(section.grid_file),
            "overburden_pressure": pressure,
            "data": data,
        }
    )


def build_geophysics(config: HistoryMatchConfig) -> GeophysicsModel | None:
    """Create the forward model of ``[geophysics]`` from its plugins.

    Raises
    ------
    ConfigError
        If a plugin is unknown or refuses its options, a map attribute is not one, or
        geophysics is combined with what it cannot support (fidelity levels, the adjoint
        update).
    """
    section = config.geophysics
    if section is None:
        return None
    if "levels" in (config.method.options or {}):
        # A coarse level's grid has other columns, so its traces are not the observed ones.
        raise ConfigError(
            "[geophysics] cannot be combined with fidelity levels (a multilevel method): "
            "the traces are the fine grid's columns"
        )
    if (config.method.options or {}).get("sensitivities") or (
        config.method.strategy is not None
        and (config.method.strategy.options or {}).get("sensitivities")
    ):
        raise ConfigError("[geophysics] cannot be combined with the adjoint update")
    try:
        pem_options, parameters = _pem_parameters(config)
        cell_temperature = None
        deck_gas = False
        if pem_options.get("temperature") == "deck":
            pem_options, cell_temperature, deck_gas = _deck_temperature(config, pem_options)
        pem = (
            None
            if section.pem is None
            else petro_elastic_models.create(section.pem.name, **pem_options)
        )
        reflectivities: list[PPReflectivity | None] = []
        attributes: list[AVOAttribute | None] = []
        kernels: list[DisplacementKernel | None] = []
        for d in section.data:
            if d.kind != "avo":
                reflectivities.append(None)
                attributes.append(None)
                if d.kind == "seafloor_displacement":
                    kernel = d.displacement().kernel
                    kernels.append(displacement_kernels.create(kernel.name, **kernel.options))
                else:
                    kernels.append(None)
                continue
            avo = d.avo()
            kernels.append(None)
            reflectivities.append(
                reflectivity.create(avo.reflectivity.name, **avo.reflectivity.options)
            )
            name = "trace" if avo.form == "traces" else avo.attribute
            attribute = avo_attributes.create(name)
            if avo.form == "maps" and not attribute.is_map:
                raise ConfigError(
                    f"[geophysics] {d.datatype}: attribute {name!r} is not a map attribute; "
                    "use form = 'traces' for whole traces"
                )
            attributes.append(attribute)
        model = build_geophysics_model(
            section, pem, reflectivities, attributes, _completions(config), kernels,
            brine_as_oil=_co2store_oil_gas(config), parameters=parameters,
            temperature=cell_temperature,
        )  # fmt: skip
        if cell_temperature is not None and deck_gas:
            # CO₂'s table over the cells' temperatures, built (and checked) before any
            # member runs, also when the deck's phases are unknown, and carried to the
            # members with the model (a thermal run without temperature_range is refused).
            prepare = getattr(pem, "prepare", None)
            if callable(prepare):
                prepare()
    except PluginError as error:
        raise ConfigError(f"[geophysics]: {error}") from error
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[geophysics]: invalid settings: {error}") from error
    _check_field_size(config, model)
    return model


def _deck_temperature(
    config: HistoryMatchConfig, options: dict[str, object]
) -> tuple[dict[str, object], CellTemperature | None, bool]:
    """Resolve ``temperature = "deck"`` from the deck's text.

    One temperature for the whole deck becomes ``temperature_c`` (as before, bit for bit);
    depth tables (``RTEMPVD``/``TEMPVD``, per ``EQLNUM``) give each cell's at its depth; a
    thermal run gives each cell's restart ``TEMP``. ``temperature_range`` is, unless given,
    the tables' temperatures over ``grid_file``'s active cells (their tops to their bases,
    every table: the regions are not known before a run). Given, it must cover them with one
    table (with several it is the user's: a cell outside it fails its member, naming it). A
    thermal run's range cannot be known before it runs: CO₂'s table then needs one given.

    Returns
    -------
    tuple
        The options, the cells' temperature source (``None``: one temperature), and whether
        the deck may have CO₂ (its phases unknown or with gas), for the setup to build
        CO₂'s table before any member runs.

    Raises
    ------
    ConfigError
        If the simulator has no deck, ``temperature_c`` is given too, the deck's temperature
        cannot be read from its text, or ``temperature_range`` is malformed, misses the
        cells' temperatures (one table), or is needed and missing (tables beyond 0-100 °C).
    """
    where = "[geophysics] pem temperature = 'deck'"
    template = config.simulator.options.get("deck_template_file")
    if not isinstance(template, str):
        raise ConfigError(f"{where}: the simulator has no deck (deck_template_file)")
    if "temperature_c" in options:
        raise ConfigError(f"{where}: give temperature_c or 'deck', not both")
    contents = deck_contents(Path(template), case_folder(config.simulator.options))
    gas = contents.phases is None or "gas" in contents.phases
    out = dict(options)
    given = options.get("temperature_range")
    if given is not None and not (isinstance(given, list | tuple) and len(given) == 2):
        raise ConfigError(f"{where}: temperature_range must be [low, high] (°C), got {given}")
    if contents.thermal:
        return out, CellTemperature(thermal=True), gas
    tables = contents.temperature_tables
    if contents.temperature_varies and tables is not None:
        low, high = _cells_temperature_span(config, tables)
        if given is None:
            if low < 0.0 or high > 100.0:
                raise ConfigError(
                    f"{where}: the deck's tables give {low:g}-{high:g} °C over grid_file's "
                    "cells, beyond Batzle & Wang's 0-100 °C; give temperature_range, the "
                    "reservoir cells' own range (a cell outside it fails its member)"
                )
            out["temperature_range"] = [low, high]
        elif len(tables) == 1 and not (float(given[0]) <= low and high <= float(given[1])):
            raise ConfigError(
                f"{where}: temperature_range {list(given)} misses the deck's {low:g}-{high:g} "
                "°C over grid_file's cells (from their tops to their bases), where every "
                "member's cells lie"
            )
        return out, CellTemperature(tables=tables), gas
    if not contents.temperature_varies and contents.temperature is not None:
        if given is not None and not float(given[0]) <= contents.temperature <= float(given[1]):
            raise ConfigError(
                f"{where}: the deck's one temperature, {contents.temperature:.4g} °C, is outside "
                f"temperature_range {list(given)}"
            )
        # One temperature: the model runs as with temperature_c, its scalar path.
        out["temperature_c"] = contents.temperature
        out.pop("temperature_range", None)
        out.pop("temperature")
        return out, None, gas
    raise ConfigError(
        f"{where}: the deck's temperature cannot be read from its text (RTEMP or RTEMPVD "
        "templated, defaulted or in an unread include, or a templated RUNSPEC); write it as "
        "plain text, or give temperature_c"
    )


def _cells_temperature_span(
    config: HistoryMatchConfig, tables: tuple[tuple[tuple[float, float], ...], ...]
) -> tuple[float, float]:
    """Return the tables' lowest and highest temperature (°C) over grid_file's active cells.

    From the shallowest top to the deepest base, every table (the cells' regions are known
    only from a run's INIT): a table is linear between its depths and constant beyond them,
    so its extremes there are at those two depths or at its own depths between them.
    """
    assert config.geophysics is not None
    grid = Grid.read(config.geophysics.grid_file)
    shallow = float(grid.top[grid.actnum].min())
    deep = float(grid.bottom[grid.actnum].max())
    values: list[float] = []
    for table in tables:
        depths, temperatures = (np.array(v, dtype=np.float64) for v in zip(*table, strict=True))
        inside = depths[(depths > shallow) & (depths < deep)]
        values += list(np.interp(np.concatenate(([shallow, deep], inside)), depths, temperatures))
    return min(values), max(values)


def _pem_parameters(
    config: HistoryMatchConfig,
) -> tuple[dict[str, object], tuple[PemParameter, ...]]:
    """Return the petro-elastic options without state references, and the references.

    An option written ``{ state = "<variable>", transform = ..., value = <fallback> }``
    (a table with ``state``) is set per member by that state variable: an
    option the model declares uncertain, with a transform it allows (its first by
    default), the variable one value, auxiliary (no true value). Any other table is an
    ordinary option (e.g. ``standard``'s ``fluid_fits``).

    Raises
    ------
    ConfigError
        If the reference is malformed, names an option the model does not declare
        uncertain, a transform it does not allow, or a state variable that is missing or
        not one value.
    """
    section = config.geophysics
    if section is None or section.pem is None:
        return {}, ()
    options: dict[str, object] = {}
    parameters: list[PemParameter] = []
    variables = {v.name: v for v in config.state.variables}
    uncertain = _uncertain_options(section.pem.name)
    for name, value in section.pem.options.items():
        if not isinstance(value, dict) or "state" not in value:
            options[name] = value
            continue
        where = f"[geophysics] pem {name}"
        if name not in uncertain:
            settable = ", ".join(sorted(uncertain))
            raise ConfigError(
                f"{where}: a state variable may set only {settable} of {section.pem.name}"
                if settable
                else f"{where}: {section.pem.name} has no option a state variable may set"
            )
        allowed = uncertain[name]
        unknown = sorted(set(value) - {"state", "transform", "value"})
        state, transform = value.get("state"), value.get("transform", allowed[0])
        if unknown or not isinstance(state, str) or transform not in PARAMETER_TRANSFORMS:
            raise ConfigError(
                f'{where}: a state reference is {{ state = "<variable>", transform = '
                f"{' | '.join(repr(t) for t in allowed)}, value = <fallback> }}"
            )
        if transform not in allowed:
            # Another transform lets an update push the option out of its range, and
            # every such member would fail after its simulation.
            raise ConfigError(
                f"{where}: transform {transform!r} can leave {name}'s valid range; use "
                f"{' or '.join(repr(t) for t in allowed)} (the default)"
            )
        if state not in variables or variables[state].size != 1:
            raise ConfigError(f"{where}: state variable {state!r} must be declared, of size 1")
        fallback = value.get("value")
        if fallback is not None and (
            isinstance(fallback, bool) or not isinstance(fallback, int | float)
        ):
            raise ConfigError(f"{where}: value must be a number")
        parameters.append(
            PemParameter(name, state, str(transform), None if fallback is None else float(fallback))
        )
        if fallback is not None:
            # The setup's model takes the fallback, so its own checks see it.
            options[name] = float(fallback)
    return options, tuple(parameters)


def _uncertain_options(name: str) -> Mapping[str, tuple[str, ...]]:
    """Return the options a state variable may set, each with the transforms it allows.

    Those a petro-elastic model declares (``uncertain_options``): an option its setup
    depends on (e.g. the temperature, which CO₂'s table and the deck's check take) cannot
    change per member.
    """
    # An unknown model's PluginError propagates: build_geophysics reports it as such.
    cls = petro_elastic_models.factory(name)
    declared: Mapping[str, tuple[str, ...]] = getattr(cls, "uncertain_options", {})
    return declared


def _co2store_oil_gas(config: HistoryMatchConfig) -> bool | None:
    """Return whether the deck is OPM Flow's CO2STORE in its oil-gas form (brine as oil).

    ``None`` when its RUNSPEC is templated, so that the text cannot tell.
    """
    template = config.simulator.options.get("deck_template_file")
    if not isinstance(template, str) or config.geophysics is None or config.geophysics.pem is None:
        return False  # without a petro-elastic model the brine's phase is never read
    contents = deck_contents(Path(template), case_folder(config.simulator.options))
    if contents.co2store is None:
        return None
    return contents.co2store and contents.phases == frozenset({"oil", "gas"})


def _completions(config: HistoryMatchConfig) -> dict[str, list[tuple[int, int]]] | None:
    """Return the deck's completed columns per well, for regions around wells; else None.

    Raises
    ------
    ConfigError
        If a region around wells is configured but the deck's completions cannot all be
        read from its text (no template, or COMPDAT records with Mako expressions).
    """
    section = config.geophysics
    if section is None or not any(
        isinstance(d.options, AVOOptions) and isinstance(d.options.region, RegionAroundWellsSection)
        for d in section.data
    ):
        return None
    options = config.simulator.options or {}
    template = options.get("deck_template_file")
    if not isinstance(template, str):
        raise ConfigError(
            "[geophysics] a region around wells reads the completions from the simulator's "
            "deck_template_file, which it has none of"
        )
    layout = deck_layout(Path(template), case_folder(options))
    if layout["skipped"]:
        raise ConfigError(
            "[geophysics] a region around wells needs every completion, but these COMPDAT "
            f"records cannot be read from the deck's text: {layout['skipped']}; give the "
            "region as a file instead"
        )
    wells = layout["wells"]
    if not isinstance(wells, dict):  # deck_layout's documented form
        raise TypeError("deck_layout returned no wells table")
    return {str(name): [(int(c[0]), int(c[1])) for c in cells] for name, cells in wells.items()}


def _iso_date(text: str) -> str:
    """Return ``text`` as ``YYYY-MM-DD`` if it is an ISO date, else as it is."""
    try:
        return dt.date.fromisoformat(text).isoformat()
    except ValueError:
        return text


def _check_field_size(config: HistoryMatchConfig, model: GeophysicsModel) -> None:
    """Refuse more untransformed field values per member than ``max_field_values``.

    A field counts as transformed at the vintages a ``[[data_transforms]]`` entry names it
    (its ``datatype`` option, at its ``times`` or every time), as ``wavelet`` selects them.
    Only observed vintages count: the others are never forecast as data (a rolling
    baseline's data-less vintage is computed in passing, as the next one's reference).
    """
    assert config.geophysics is not None
    covered: dict[str, set[str] | None] = {}  # datatype -> its transformed times (None: all)
    for transform in config.data_transforms:
        datatype = transform.options.get("datatype")
        if not isinstance(datatype, str):
            continue
        times = transform.options.get("times")
        if times is None or covered.get(datatype, set()) is None:
            covered[datatype] = None
        else:
            known = covered.setdefault(datatype, set())
            assert known is not None
            known.update(_iso_date(str(t)) for t in times)
    sizes = model.field_sizes()
    # Dates compared as dates: a field's time is any ISO form date.fromisoformat takes
    # ("20200701"), the model's vintages its isoformat().
    observed = {(f.datatype, _iso_date(f.time)) for f in config.observations.fields}
    bare: dict[str, int] = {}
    for name, vintages in model.field_vintages().items():
        times = covered.get(name, set())
        seen = [v for v in vintages if (name, v) in observed]
        untransformed = 0 if times is None else sum(v not in times for v in seen)
        if untransformed:
            bare[name] = sizes[name] * untransformed
    limit = config.geophysics.max_field_values
    if sum(bare.values()) > limit:
        raise ConfigError(
            f"[geophysics]: {sum(bare.values()):,} untransformed field values per member "
            f"({', '.join(sorted(bare))}), more than max_field_values = {limit:,}; add a "
            "[[data_transforms]] (e.g. wavelet) for them, or raise max_field_values"
        )


def geophysics_restart_warnings(config: HistoryMatchConfig) -> list[str]:
    """Check from the deck's text that it writes a restart at every vintage.

    Returns warnings when the text cannot tell (template expressions, Mako lines, unread
    includes); the run's own restart files decide then.

    Raises
    ------
    ConfigError
        If the deck certainly writes no restart at a vintage, naming what to add.
    """
    section = config.geophysics
    if section is None:
        return []
    options = config.simulator.options
    template = options.get("deck_template_file")
    if not isinstance(template, str):
        return ["[geophysics]: the simulator has no deck_template_file, so the vintages' "
                "restarts are not checked before the run"]  # fmt: skip
    schedule = restart_schedule(Path(template), case_folder(options))
    # A difference's baseline is read from its restart as the vintages are.
    vintages = sorted(
        {date.fromisoformat(v) for d in section.data for v in d.vintages}
        | {date.fromisoformat(b) for d in section.data if (b := d.baseline()) is not None}
    )
    missing = schedule.missing(vintages)
    notes = "; ".join(schedule.unresolved[:3])
    if missing:
        reported = {t.date() for t in schedule.report_times}
        start = schedule.report_times[0].date() if schedule.report_times else None

        def reason(d: date) -> str:
            if d in reported:
                return "a report time without a restart"
            return "before START" if start is not None and d < start else "not a report time"

        reasons = [f"{d.isoformat()} ({reason(d)})" for d in missing]
        message = (
            f"[geophysics]: the deck writes no restart at {', '.join(reasons)}; give each "
            "vintage a report time (DATES) and a restart there (e.g. RPTRST BASIC=2, a "
            "restart at every report time)"
        )
        if not schedule.unresolved:
            raise ConfigError(message)
        return [f"{message}, as far as its text tells ({notes})"]
    warnings = []
    if schedule.unresolved:
        warnings.append(f"[geophysics]: the vintages' restarts could not be confirmed from the "
                        f"deck's text ({notes}); the run's restart files decide")  # fmt: skip
    if config.simulator.name == "eclipse":
        warnings.append(
            "[geophysics]: the restarts are checked by OPM Flow's rules (Eclipse's may "
            "differ; separate restarts under BASIC=1 alone are refused)"
        )
    return warnings


def geophysics_deck_warnings(config: HistoryMatchConfig, model: GeophysicsModel) -> list[str]:
    """Check, before any member runs, what every member would fail on in its read.

    From the deck's text (:func:`~apex.simulation.deck.deck_contents`): formatted output
    files (APEX reads unformatted ones), the grid's size against ``grid_file``, the
    RUNSPEC phases against the petro-elastic model, the restart arrays the data need
    (densities, pore volumes), and Eclipse's separate restarts under ``BASIC=1``. A member raises
    these too, but only once its run is read, and an executor finishes the members it
    has started first; here they stop the run before it starts.

    Returns
    -------
    list[str]
        What the text cannot tell.

    Raises
    ------
    ConfigError
        For what the text shows every member would fail on.
    """
    template = config.simulator.options.get("deck_template_file")
    if not isinstance(template, str):
        return []  # noted by geophysics_restart_warnings
    contents = deck_contents(Path(template), case_folder(config.simulator.options))
    if contents.formatted is True:
        raise ConfigError(
            "[geophysics]: the deck writes formatted files (FMTOUT); APEX reads the INIT and "
            "restarts unformatted, so remove FMTOUT"
        )
    if contents.grid is not None and contents.grid != model.reference.shape:
        raise ConfigError(
            f"[geophysics]: the deck's grid is {contents.grid} (DIMENS), grid_file's "
            f"{model.reference.shape}"
        )
    if contents.co2store and config.simulator.name == "eclipse":
        raise ConfigError(
            "[geophysics]: Eclipse 300's CO2STORE is compositional, with other restart arrays "
            "(DENW, DENG, ...), and is not supported; run the deck on opm_flow"
        )
    if contents.phases is not None:
        model.check_phases(sorted(contents.phases))
    if config.simulator.name == "eclipse" and contents.basic_one and contents.unified is False:
        raise ConfigError(
            "[geophysics]: Eclipse with RPTRST BASIC=1 and separate restart files keeps only "
            "the last restart; use BASIC=2, or UNIFOUT"
        )
    warnings: list[str] = model.check_temperature(
        contents.temperature,
        contents.temperature_varies,
        None if contents.phases is None else model.petro_elastic_phases(sorted(contents.phases)),
    )
    if contents.co2store is None and getattr(model.pem, "gas", None) == "co2":
        warnings.append(
            "[geophysics]: the deck's RUNSPEC is templated, so whether it is CO2STORE's oil-gas "
            "form (whose oil phase is brine) cannot be told; APEX reads oil as oil, and refuses "
            "an oil-gas run when it reads each member, after its simulation"
        )
    if contents.formatted is None:
        warnings.append(
            "[geophysics]: the deck's RUNSPEC is templated (Mako, expressions or includes): "
            "it cannot confirm that FMTOUT (formatted output, which APEX cannot read) is absent"
        )
    if config.simulator.name == "eclipse" and contents.basic_one and contents.unified is None:
        warnings.append(
            "[geophysics]: under RPTRST BASIC=1 Eclipse keeps only the last of separate "
            "restarts, and the deck's templated RUNSPEC cannot confirm UNIFOUT"
        )
    if config.simulator.name == "eclipse" and contents.basic_one is None and not contents.unified:
        warnings.append(
            "[geophysics]: a restart control is templated, so whether only BASIC=1 holds (with "
            "which Eclipse keeps only the last of separate restarts) cannot be told"
        )
    found = {"DEN": contents.densities, "RPORV": contents.pore_volumes}
    for mnemonic, reason in model.restart_needs().items():
        if found[mnemonic] is False:
            raise ConfigError(
                f"[geophysics] {reason} needs {mnemonic} in the restarts, but no RPTRST asks "
                f"for it; add it (e.g. RPTRST BASIC=2 {mnemonic} /)"
            )
        if found[mnemonic] is None:
            warnings.append(
                f"[geophysics] {reason} needs {mnemonic} in the restarts (RPTRST), which the "
                "deck's text cannot confirm; the run's restarts decide"
            )
    return warnings


def build_model_error(
    config: HistoryMatchConfig, layout: StateLayout, observations: ObservationSet
) -> ModelErrorModel | None:
    """Create the configured model-error model, fitted to the observations.

    Raises
    ------
    ConfigError
        If the model is unknown, its parameters are not a state variable of its size, or
        it is combined with what it cannot support (parameterizations, the adjoint update).
    """
    section = config.model_error
    if section is None:
        return None
    try:
        model = model_error_models.create(section.name, **section.options)
    except PluginError as error:
        raise ConfigError(f"[model_error]: {error}") from error
    sizes = {v.name: v.size for v in layout.variables}
    name = model.state_variable
    if name not in sizes:
        raise ConfigError(
            f"[model_error]: its parameters need the state variable {name!r} "
            f"([[state.variables]] name = {name!r}, size = {model.n_params})"
        )
    if sizes[name] != model.n_params:
        raise ConfigError(
            f"[model_error]: {name!r} has {sizes[name]} values, the model {model.n_params}"
        )
    if config.parameterizations:
        raise ConfigError("[model_error] cannot be combined with [[parameterizations]] yet")
    if "levels" in config.method.options:
        # One schedule's block indices serve every level, but a coarse level's grid and
        # properties differ, so one theta maps to different residuals per level: the
        # multilevel gain then mixes inconsistent models (the 5Spots smoke run's data
        # mismatch rose). Refused until each level has its own networks.
        raise ConfigError(
            "[model_error] cannot be combined with fidelity levels (a multilevel method): "
            "its networks read the fine grid's well blocks, which a coarse level does not have"
        )
    if (config.method.options or {}).get("sensitivities") or (
        config.method.strategy is not None
        and (config.method.strategy.options or {}).get("sensitivities")
    ):
        raise ConfigError("[model_error] cannot be combined with the adjoint update")
    model.fit(observations)
    return model


def check_model_error_summary(config: HistoryMatchConfig, model: ModelErrorModel) -> None:
    """Refuse, before any member runs, a deck whose SUMMARY lacks what the model reads.

    ``pg_lstm`` reads every listed well's ``WBHP`` (its ``PRESS_DIFF`` feature) whether or
    not it is observed; a deck without it failed only once each member's run was read,
    after the whole prior was simulated (the release tester, 2026-10-19). Read from the
    deck's text (:func:`~apex.simulation.deck.missing_summary_wells`); what the text
    cannot tell is left to the run.

    Raises
    ------
    ConfigError
        If the deck's SUMMARY section certainly lacks a vector the model reads.
    """
    template = config.simulator.options.get("deck_template_file")
    vectors = getattr(model, "summary_vectors", None)
    if not isinstance(template, str) or not vectors:
        return
    by_keyword: dict[str, list[str]] = {}
    for keyword, well in vectors:
        by_keyword.setdefault(keyword, []).append(well)
    for keyword, wells in by_keyword.items():
        missing = missing_summary_wells(
            Path(template), keyword, wells, case_folder(config.simulator.options)
        )
        if missing:
            raise ConfigError(
                f"[model_error] {config.model_error.name if config.model_error else ''}: "
                f"the deck's SUMMARY section has no {keyword} of "
                f"{', '.join(missing)}, which the model reads for every listed well (observed "
                f"or not); add, e.g., {keyword} / (every well) to SUMMARY"
            )


def build_metrics(config: HistoryMatchConfig) -> dict[str, PerformanceMetric]:
    """Create the configured performance metrics (or the defaults), by result name.

    An ``actnum_file`` option (with an optional ``actnum_key``) is loaded into ``actnum``.
    ``model_change`` is always recorded (the user's standing choice, 2026-09-30): a run
    that does not configure it gets it, over the ``rmse`` metric's ``actnum`` if it has one.
    By default a synthetic case whose parameterizations produce ``facies`` also records
    ``facies_mismatch``.

    Raises
    ------
    ConfigError
        If a metric is unknown, named twice, or needs the true model in a real case.
    ValueError, TypeError
        If its options are invalid.
    """
    sections = config.metrics
    if sections is None:
        sections = [MetricSection(name=name) for name in DEFAULT_METRICS[config.case.kind]]
        # a synthetic case whose parameterization produces facies scores them.
        produces_facies = any(
            "facies" in getattr(p, "outputs", ()) for p in build_parameterizations(config)
        )
        if config.case.kind == "synthetic" and produces_facies:
            sections += [
                MetricSection(name=n)
                for n in ("facies_mismatch", "facies_brier", "facies_log_score")
            ]
    built: dict[str, PerformanceMetric] = {}
    for section in sections:
        if section.name == DATA_MISMATCH:
            # The objective is always recorded under this name: nothing to build.
            logger.warning(
                # Its meaning changed on 2026-09-30: it was the mean square.
                "[[metrics]] data_mismatch is the objective the methods minimise (the "
                "error-weighted sum of squares per member), recorded in every run; for the "
                "mean square per datum, use normalized_data_mismatch",
                extra={"event": "metric_meaning_changed", "metric": DATA_MISMATCH},
            )
            if section.options:
                raise ConfigError("[[metrics]]: data_mismatch (the objective) takes no options")
            continue
        if section.key == DATA_MISMATCH or section.key.startswith(("cv_", "test_")):
            # Reserved: the objective, and the held-out data's scores.
            raise ConfigError(
                f"[[metrics]]: the name {section.key!r} is reserved (the objective, or the "
                "cv_/test_ scores of held-out data); give the metric another label"
            )
        if section.key in built:
            raise ConfigError(f"[[metrics]]: {section.key!r} twice; give one a label")
        options = dict(section.options)
        if "actnum_file" in options:
            options["actnum"] = read_array(
                Path(options.pop("actnum_file")), options.pop("actnum_key", None)
            )
        try:
            metric = metrics.create(section.name, **options)
        except PluginError as error:
            raise ConfigError(f"[[metrics]]: {error}") from error
        if metric.requires_truth and config.case.kind == "real":
            raise ConfigError(
                f"[[metrics]]: {section.name!r} compares with the true model, which a real "
                "case study does not have"
            )
        built[section.key] = metric
    if "model_change" not in built and not any(s.name == "model_change" for s in sections):
        actnum = next(
            (
                getattr(metric, "active", None)
                for section in sections
                if section.name == "rmse"
                for metric in [built[section.key]]
            ),
            None,
        )
        built["model_change"] = metrics.create("model_change", actnum=actnum)
    return built


def build_transforms(
    config: HistoryMatchConfig, geophysics: "GeophysicsModel | None" = None
) -> list[DataTransform]:
    """Create the configured data transforms, unfitted, in order.

    A ``mask_file`` option (with an optional ``mask_key``) is loaded into ``mask``.
    ``geophysics`` is the run's model when it is built already (a remote worker's stack):
    an AVO field's layout is taken from it instead of building it again.

    Raises
    ------
    ConfigError
        If a transform is unknown.
    ValueError, TypeError
        If its options are invalid.
    """
    transforms = []
    geophysics_fields = (
        set()
        if config.geophysics is None
        else {name for datum in config.geophysics.data for name in datum.datatypes()}
    )
    for section in config.data_transforms:
        options = dict(section.options)
        given = {"grid", "mask_file", "mask", "order"} & set(options)
        if section.name == "wavelet" and options.get("datatype") in geophysics_fields:
            name = str(options["datatype"])
            if given and "grid" not in given:
                raise ConfigError(
                    f"[[data_transforms]] wavelet of {name}: give its layout in full (grid, "
                    "with mask_file and order), or none to take the traces' own"
                )
        if section.name == "wavelet" and options.get("datatype") in geophysics_fields and not given:
            # An AVO field's grid and mask are its traces' own (a layout
            # given in full is used as it is, as before).
            if geophysics is None:
                geophysics = build_geophysics(config)
            assert geophysics is not None
            grid, mask = geophysics.wavelet_layout(name)
            if mask.mean() < 0.5:  # for traces too: their mask is the map's, repeated
                # The fill gives exactly zero detail coefficients; more than half of them,
                # and the median-based noise estimates (white or per band) collapse to
                # their floor.
                raise ConfigError(
                    f"[[data_transforms]] wavelet of {name}: its traces cover "
                    f"{100 * mask.mean():.0f} % of their {' x '.join(map(str, grid))} lattice, too "
                    "few for the noise estimate (the filled cells' zero coefficients dominate it); "
                    "give a layout (grid, mask_file, order)"
                )
            options.update(grid=list(grid), mask=mask, order="F")
        if "mask_file" in options:
            options["mask"] = read_array(
                Path(options.pop("mask_file")), options.pop("mask_key", None)
            )
        try:
            transforms.append(data_transforms.create(section.name, **options))
        except PluginError as error:
            raise ConfigError(f"[[data_transforms]]: {error}") from error
    return transforms


def _check_pgs_rules(config: HistoryMatchConfig) -> None:
    """Refuse a plurigaussian prior and truncation that disagree.

    The prior conditions its latent fields with its rule, and the truncation maps them with
    its own: different rules (or fields in another order) would give facies that the wells'
    conditioning never meant.

    Raises
    ------
    ConfigError
        If a ``gstools_pgs`` group and a ``pgs_truncation`` of the same fields differ in their
        rule, or name the same fields in another order.
    """
    keys = ("rule_file", "proportions", "extent")
    truncations = [s for s in config.parameterizations if s.name == "pgs_truncation"]
    for name, group in config.prior.groups.items():
        if group.name != "gstools_pgs":
            continue
        for section in truncations:
            fields = list(section.options.get("fields", []))
            if set(fields) != set(group.variables):
                continue
            if fields != list(group.variables):
                raise ConfigError(
                    f"prior group {name!r} and pgs_truncation name the latent fields in different "
                    f"orders ({group.variables} and {fields}); the rule's axes follow the order"
                )
            mine = {k: group.options.get(k, 3.0 if k == "extent" else None) for k in keys}
            theirs = {k: section.options.get(k, 3.0 if k == "extent" else None) for k in keys}
            if mine != theirs:
                raise ConfigError(
                    f"prior group {name!r} and pgs_truncation use different rules ({mine} and "
                    f"{theirs}); give both the same rule_file or proportions and extent"
                )


def build_parameterizations(config: HistoryMatchConfig) -> list[Parameterization]:
    """Create ``[[parameterizations]]`` in order.

    Raises
    ------
    ConfigError
        If a parameterization is unknown or its options are invalid.
    """
    built = []
    for section in config.parameterizations:
        try:
            built.append(parameterizations.create(section.name, **section.options))
        except (PluginError, TypeError, ValueError) as error:
            raise ConfigError(f"[[parameterizations]] {section.name}: {error}") from error
    return built


def _build_method(
    config: HistoryMatchConfig,
    layout: StateLayout,
    parameterizations: Sequence[Parameterization] | None = None,
) -> EnsembleAlgorithm:
    if parameterizations is None:
        parameterizations = build_parameterizations(config)
    return build_method(config.method, layout, parameterizations)


def _wrapped_method(
    section: MethodSection,
    layout: StateLayout,
    parameterizations: Sequence[Parameterization],
) -> EnsembleAlgorithm | None:
    """Build a method that wraps another (CHOP), or return ``None``.

    Such a method takes ``inner_method``: ``[method.options] inner`` names the wrapped
    method, which is built from the rest of ``[method]`` (its strategy, schedule, damping
    and options); the wrapper's own options are those its signature names.
    """
    if section.name not in methods.names():
        return None
    parameters = inspect.signature(methods.factory(section.name)).parameters
    if "inner_method" not in parameters:
        return None
    options = dict(section.options)
    inner = options.pop("inner", "gies")
    if not isinstance(inner, str):
        raise ConfigError(f"[method.options] inner must name a method, got {inner!r}")
    own = {name: options.pop(name) for name in list(options) if name in parameters}
    if "block_sizes" in parameters:
        if "block_sizes" in own:
            raise ConfigError("[method.options] block_sizes come from [state]; remove it")
        own["block_sizes"] = [v.size for v in layout.variables]
    inner_section = section.model_copy(update={"name": inner, "options": options})
    inner_method = build_method(inner_section, layout, parameterizations)
    try:
        return methods.create(section.name, inner_method=inner_method, **own)
    except (TypeError, ValueError, PluginError) as error:
        raise ConfigError(f"[method] {section.name}: {error}") from error


def build_method(
    section: MethodSection,
    layout: StateLayout,
    parameterizations: Sequence[Parameterization] = (),
) -> EnsembleAlgorithm:
    """Build ``[method]`` (with its strategy, localization and damping) for ``layout``.

    For callers that run a method's analysis outside the workflow, such as ERT's
    ``APEX_UPDATE``.

    Raises
    ------
    ConfigError
        If a named plugin is unknown, or the parts do not go together.
    """
    wrapper = _wrapped_method(section, layout, parameterizations)
    if wrapper is not None:
        return wrapper
    options = dict(section.options)
    strategy_options: dict[str, Any] = (
        dict(section.strategy.options) if section.strategy is not None else {}
    )
    if section.localization is not None:
        if section.strategy is None:
            raise ConfigError("[method.localization] needs an explicit [method.strategy]")
        for supplied in ("block_sizes", "variable_names"):
            if supplied in section.localization.options:
                raise ConfigError(f"[method.localization] {supplied} come from [state]; remove it")
        strategy_class = update_strategies.factory(section.strategy.name)
        if "localization" not in inspect.signature(strategy_class).parameters:
            raise ConfigError(f"strategy {section.strategy.name!r} does not support localization")
        supplied_options: dict[str, Any] = {"block_sizes": [v.size for v in layout.variables]}
        # Localizations that name state variables (distance_entries) get their names.
        localization_class = localizations.factory(section.localization.name)
        if "variable_names" in inspect.signature(localization_class).parameters:
            supplied_options["variable_names"] = [v.name for v in layout.variables]
        strategy_options["localization"] = localizations.create(
            section.localization.name, **supplied_options, **section.localization.options
        )
    if section.strategy is not None:
        # Strategies that scale per state variable (the full update) get the blocks.
        strategy_class = update_strategies.factory(section.strategy.name)
        if "block_sizes" in inspect.signature(strategy_class).parameters:
            strategy_options.setdefault("block_sizes", [v.size for v in layout.variables])
        # Strategies whose terms see variables by name and grid (gmac) get them.
        if "state_variables" in inspect.signature(strategy_class).parameters:
            sources = {"state_variables": "[state]", "property_map": "[[parameterizations]]"}
            for supplied, source in sources.items():
                if supplied in strategy_options:
                    raise ConfigError(
                        f"[method.strategy] {supplied} comes from {source}; remove it"
                    )
            strategy_options["state_variables"] = [
                {"name": v.name, "size": v.size, "grid": v.grid} for v in layout.variables
            ]
            # ... and, with parameterizations, the map to the properties.
            if parameterizations:
                try:
                    properties = PropertyMap(parameterizations, layout)
                except ValueError as error:
                    raise ConfigError(f"[[parameterizations]]: {error}") from error
                strategy_options["property_map"] = StackedProperties(properties, layout)
        options["strategy"] = update_strategies.create(section.strategy.name, **strategy_options)
    lower, upper = layout.bounds()
    if np.isfinite(lower).any() or np.isfinite(upper).any():
        options.setdefault("lower_bounds", lower)
        options.setdefault("upper_bounds", upper)
    if section.name not in methods.names():
        raise ConfigError(
            f"unknown method {section.name!r}; available: {', '.join(sorted(methods.names()))}"
        )
    # [method] schedule and max_iterations go to the methods that take them (iterative ones).
    parameters = inspect.signature(methods.factory(section.name)).parameters
    if section.schedule is not None:
        if "schedule" not in parameters:
            raise ConfigError(
                f"method {section.name!r} has a fixed schedule; remove [method.schedule]"
            )
        # Each iterative method names its schedule's type (GIES: DampingSchedule).
        schedule_type = getattr(methods.factory(section.name), "schedule_type", DampingSchedule)
        options["schedule"] = schedule_type(**section.schedule)
    if section.damping is not None:
        # A damping scheme of its own: an auxiliary technique, for the methods
        # that adapt a damping (GIES, LM-EnRML; not GN-EnRML's step length).
        if not getattr(methods.factory(section.name), "accepts_damping", False):
            raise ConfigError(f"method {section.name!r} adapts no damping; remove [method.damping]")
        if section.schedule is not None:
            raise ConfigError(
                "give the damping's options in [method.damping] options, not also "
                "[method.schedule] (it holds the method's default scheme's options)"
            )
        try:
            options["schedule"] = damping_schemes.create(
                section.damping.name, **section.damping.options
            )
        except PluginError as error:
            raise ConfigError(f"[method.damping]: {error}") from error
    if "max_iterations" in parameters:
        options.setdefault("max_iterations", section.max_iterations)
    return methods.create(section.name, **options)


def _draw_groups(run: Run) -> dict[str, np.ndarray]:
    """Draw the variables of ``[prior.groups]`` with their joint generators.

    Groups are drawn in name order, before the per-variable generators, from the same
    ``prior`` stream.

    Raises
    ------
    ConfigError
        If a group names an unknown variable, or its generator is not a joint one or fails.
    """
    layout = run.workflow.layout
    variables = {v.name: v for v in layout.variables}
    blocks: dict[str, np.ndarray] = {}
    for name, section in sorted(run.config.prior.groups.items()):
        context = f"prior group {name!r} ({section.name})"
        unknown = sorted(set(section.variables) - set(variables))
        if unknown:
            raise ConfigError(f"{context} names unknown state variables: {', '.join(unknown)}")
        try:
            generator = priors.create(section.name, **section.options)
        except (TypeError, ValueError, PluginError) as error:
            raise ConfigError(f"{context}: {error}") from error
        if not hasattr(generator, "generate_joint"):
            raise ConfigError(f"{context}: {section.name!r} draws one variable at a time")
        members = [variables[v] for v in section.variables]
        try:
            drawn = generator.generate_joint(
                members, run.config.prior.ensemble_size, run.workflow.streams["prior"]
            )
        except ValueError as error:
            raise ConfigError(f"{context}: {error}") from error
        for variable in members:
            block = np.asarray(drawn[variable.name], dtype=np.float64)
            if getattr(generator, "clip", False):
                block = np.clip(block, variable.lower, variable.upper)
            blocks[variable.name] = block
    return blocks


def draw_prior(run: Run) -> np.ndarray:
    """Create the prior ensemble the configuration describes.

    Variables are generated in layout order from the ``prior`` stream, then clipped to
    their bounds (as PET clips at generation), except for generators with ``clip = False``
    such as ``file``: a stored ensemble is used as is (as PET's ``importstate``).

    Raises
    ------
    ConfigError
        If a generator is unknown or misconfigured, or variables disagree in ensemble size.
    """
    prior = run.config.prior
    layout = run.workflow.layout
    unknown = sorted(set(prior.variables) - set(layout.names))
    if unknown:
        raise ConfigError(f"[prior.variables] names unknown state variables: {', '.join(unknown)}")
    blocks = _draw_groups(run)
    for variable in layout.variables:
        if variable.name in blocks:
            continue
        section = prior.generator_for(variable.name)
        options = dict(section.options)
        context = f"prior of {variable.name!r} ({section.name})"
        try:
            if "mean_file" in options:  # a per-cell mean, e.g. PET's priormean.npz
                options["mean"] = read_array(
                    Path(options.pop("mean_file")), options.pop("mean_key", None)
                )
            # TypeError here means unknown or missing options; inside generate it is a bug.
            generator = priors.create(section.name, **options)
        except (TypeError, ValueError, PluginError) as error:
            raise ConfigError(f"{context}: {error}") from error
        if not isinstance(generator, PriorGenerator):
            raise ConfigError(
                f"{context}: {section.name!r} draws variables jointly; name them in "
                "[prior.groups.<name>] variables"
            )
        try:
            block = generator.generate(variable, prior.ensemble_size, run.workflow.streams["prior"])
        except ValueError as error:
            raise ConfigError(f"{context}: {error}") from error
        if getattr(generator, "clip", True):
            block = np.clip(block, variable.lower, variable.upper)
        blocks[variable.name] = block
    sizes = {name: block.shape[-1] for name, block in blocks.items()}
    if len(set(sizes.values())) != 1:
        raise ConfigError(f"prior variables have different ensemble sizes: {sizes}")
    return layout.stack(blocks)


def start(
    config_path: Path, max_steps: int | None = None, progress: ProgressReporter | None = None
) -> HistoryMatchResult:
    """Start a new run from ``config_path``, reporting to ``progress``."""
    return start_config(load_config(config_path), max_steps, progress)


def start_config(
    config: HistoryMatchConfig,
    max_steps: int | None = None,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> HistoryMatchResult:
    """Start a new run from a resolved configuration (e.g. from :func:`load_config`).

    ``stopped`` says whether the job running it was asked to stop (:func:`resume`).
    """
    return start_built(build_run(config, progress), max_steps=max_steps, stopped=stopped)


def start_built(
    run: Run,
    prior: np.ndarray | None = None,
    max_steps: int | None = None,
    stopped: Callable[[], bool] | None = None,
) -> HistoryMatchResult:
    """Start a run that is already built, from ``prior`` (default: the configured prior).

    For callers that build a run and change its algorithm or simulator before starting it,
    such as the import of ERT's iterations, which replays them through the
    workflow.
    """
    # Held until the run ends: no second process may start or resume it meanwhile.
    with exclusive(run.run_dir.path):
        run.run_dir.initialise(config_text=run.config.model_dump_json(indent=2))
        _stop_if_asked(run.run_dir, stopped)
        configure_logging(run.run_dir.path)
        write_run_provenance(run, resumed=False)
        try:
            result = run.execute(draw_prior(run) if prior is None else prior, max_steps)
        except RunStopped:
            # Stopped before its first checkpoint (while it waited for a verified
            # copy): it is started again, and its stop must not stop that start.
            if not (run.run_dir.path / "state.json").exists():
                run.run_dir.clear_stop()
            raise
        _write_summary(run, result)
    return result


def resume(
    run_path: Path,
    max_steps: int | None = None,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> HistoryMatchResult:
    """Continue the run stored in ``run_path`` with its saved configuration.

    A stop request is withdrawn first: resuming means continuing. A job's stop
    asked for meanwhile (``stopped``: while the run was built, before its lock was taken)
    is requested again, so the run stops at its next checkpoint.

    Raises
    ------
    WorkflowError
        If the run has no checkpoint (it ended before its first step did).
    """
    run_dir = RunDirectory(run_path)
    if not (run_dir.path / "state.json").exists() and in_use(run_dir.path):
        # Started by another process, still in its first step: "in use", not "no run".
        raise DirectoryLocked(
            f"the run in {run_dir.path} is in use by another process, before its first "
            "checkpoint; wait for it, or stop it (apex stop) before resuming it"
        )
    if not (run_dir.path / "state.json").exists():
        raise WorkflowError(
            f"{run_dir.path} has no checkpoint to resume from: the run ended before its "
            "first step did; start it again from its configuration"
        )
    config = snapshot_config(run_dir)
    run = build_run(config, progress)
    # Before anything is written: a run that another process is continuing stays its own.
    with exclusive(run_dir.path):
        run_dir.clear_stop()
        _stop_if_asked(run_dir, stopped)
        configure_logging(run.run_dir.path)
        write_run_provenance(run, resumed=True)
        result = run.execute(None, max_steps)
        _write_summary(run, result)
    return result


def _stop_if_asked(run_dir: RunDirectory, stopped: Callable[[], bool] | None) -> None:
    """Request a stop in ``run_dir`` if the job running it was asked to stop.

    Checked once this process holds the run's lock: before it, the job's stop is kept in
    the job store only, since the run's stop file is read by whichever process runs it.
    """
    if stopped is not None and stopped():
        run_dir.request_stop()


def _input_files(config: HistoryMatchConfig) -> dict[str, Path]:
    """Return the files a run reads, by where they are configured, for provenance."""
    files: dict[str, Path] = {}
    if config.observations.file is not None:
        files["observations"] = config.observations.file
    for field in config.observations.fields:
        files[f"observations.fields.{field.datatype}@{field.time}"] = field.file
    if config.case.truth_file is not None:
        files["truth"] = config.case.truth_file
    sections = {"simulator": config.simulator, "prior": config.prior.generator}
    sections.update({f"prior.{k}": v for k, v in config.prior.variables.items()})
    for label, section in sections.items():
        for key, value in section.options.items():
            if key.endswith(PATH_OPTION_SUFFIXES) and isinstance(value, str):
                files[f"{label}.{key}"] = Path(value)
    if config.geophysics is not None:  # the files the forward model reads
        files["geophysics.grid_file"] = config.geophysics.grid_file
        pressure = config.geophysics.overburden_pressure
        if isinstance(pressure, OverburdenPressureFile):
            files["geophysics.overburden_pressure"] = pressure.file
        for datum in config.geophysics.data:
            name = f"geophysics.{datum.datatype}"
            if isinstance(datum.options, AVOOptions):
                if isinstance(datum.options.wavelet, FileWaveletSection):
                    files[f"{name}.wavelet"] = datum.options.wavelet.file
                if isinstance(datum.options.region, RegionFileSection):
                    files[f"{name}.region"] = datum.options.region.file
                continue
            stations = datum.options.stations
            if isinstance(stations, StationFileSection):
                files[f"{name}.stations"] = stations.file
            elif stations.seabed_file is not None:
                files[f"{name}.seabed"] = stations.seabed_file
    return files


def prepare_remote(executor: object, simulator: Simulator, config: HistoryMatchConfig) -> None:
    """Give an executor that runs members whole on other hosts the stack's spec.

    Nothing is deployed here: the spec's ``code`` prepares the copies of APEX at the
    executor's first forecast, so a run built only to be validated or read costs nothing.

    Raises
    ------
    ConfigError
        If the run cannot run remotely (parameterizations, for now) or its inputs cannot
        be mirrored.
    """
    attach = getattr(executor, "attach_remote", None)
    if attach is None or not getattr(executor, "remote", False):
        return
    if config.parameterizations:
        raise ConfigError(
            "remote members do not run [[parameterizations]] yet (the workflow wraps the "
            'simulator); use the executor\'s other mode (e.g. mode = "files")'
        )
    # The executor names the code that prepares its hosts' copies of APEX ("module:name"),
    # resolved here: composition names no executor's module.
    provider = getattr(executor, "remote_code_provider", None)
    if not isinstance(provider, str) or ":" not in provider:
        raise ConfigError(f"{type(executor).__name__} runs members remotely but names no "
                          "remote_code_provider")  # fmt: skip
    module, _, name = provider.partition(":")
    try:
        run_code = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as error:
        raise ConfigError(f"{type(executor).__name__}'s remote_code_provider {provider!r} "
                          f"cannot be loaded: {error}") from error  # fmt: skip

    template = config.simulator.options.get("deck_template_file")
    includes = None if template is None else case_dir_files(Path(str(template)))
    attach(
        RemoteSpec(
            simulator=simulator,
            config=json.loads(config.model_dump_json()),
            includes=None if includes is None else tuple(includes),
            code=run_code,
            # A job's stop while the executor waits for a verified copy.
            stopped=None
            if config.run.output_dir is None
            else RunDirectory(config.run.output_dir).stop_requested,
        )
    )


def check_executor(executor: object, simulator: Simulator) -> None:
    """Let an executor refuse a simulator it cannot run (``ssh`` needs a program).

    Raises
    ------
    ValueError
        If the executor's ``check_simulator`` refuses the simulator.
    """
    check = getattr(executor, "check_simulator", None)
    if check is not None:
        check(simulator)


def write_run_provenance(run: "Run", *, resumed: bool) -> None:
    """Write (at start) or extend (on resume) ``provenance.json``.

    A resume on another host, APEX commit or CPU class logs a warning: bit-identical
    resume depends on all three (another commit may compute differently, and NumPy and
    OpenBLAS round differently on AVX-512 CPUs).
    """
    from apex.services.identity import run_id
    from apex.storage import provenance

    path = run.run_dir.path / "provenance.json"
    version = getattr(run.workflow.simulator, "version", None)
    record = provenance.collect(
        _input_files(run.config),
        simulator={
            "name": run.config.simulator.name,
            "version": None if version is None else version(),
        },
    )
    describe = getattr(run.workflow.executor, "describe", None)
    if describe is not None:  # where members run, when it is not only here
        record["execution"] = describe(run.workflow.simulator)
    if resumed and path.exists():
        saved = json.loads(path.read_text())
        first = saved["start"]
        for what, old, new in (
            ("host", first["host"]["name"], record["host"]["name"]),
            ("APEX commit", first["apex"]["commit"], record["apex"]["commit"]),
            ("AVX-512", first["host"].get("avx512"), record["host"].get("avx512")),
        ):
            if old != new:
                logger.warning(
                    "resuming on another %s (%s, was %s): results may not be bit-identical "
                    "to an uninterrupted run",
                    what,
                    new,
                    old,
                    extra={"event": "resume_elsewhere", "what": what},
                )
        saved["resumes"].append(record)
    else:
        saved = {"run_id": run_id(run.config), "start": record, "resumes": []}
    path.write_text(json.dumps(saved, indent=2, default=str) + "\n")


def snapshot_config(run_dir: RunDirectory) -> HistoryMatchConfig:
    """Read the configuration a run was started with (``config.json``).

    Snapshots written before ``[case]`` existed are read as real case studies: they have
    no true model, and their runs never evaluated metrics against one.

    Raises
    ------
    ConfigError
        If the snapshot is missing or invalid.
    """
    try:
        raw = json.loads(run_dir.config_text())
        if isinstance(raw, dict) and "case" not in raw:
            raw["case"] = {"kind": "real"}
        if isinstance(raw, dict) and run_dir.snapshot_version() < 2:
            _migrate_localization_name(raw)
        if isinstance(raw, dict) and run_dir.snapshot_version() < 3:
            _migrate_metric_names(raw)
        if isinstance(raw, dict) and run_dir.snapshot_version() < 4:
            _migrate_damping_reduction(raw)
        if isinstance(raw, dict) and run_dir.snapshot_version() < 5:
            _migrate_lm_confirmation(raw)
        return HistoryMatchConfig.model_validate(raw)
    except (OSError, ValueError, ValidationError) as error:
        raise ConfigError(
            f"{run_dir.path} has no readable configuration snapshot: {error}"
        ) from error


def _migrate_metric_names(raw: dict[str, Any]) -> None:
    """Read a snapshot older than 2026-09-30's ``data_mismatch`` metric by its new name.

    It was the mean square, now ``normalized_data_mismatch``; ``data_mismatch`` names the
    objective. A resume of such a run keeps its metrics.
    """
    for section in raw.get("metrics") or []:
        if isinstance(section, dict) and isinstance(section.get("name"), str):
            section["name"] = upgrade_metric_name(section["name"])


def _migrate_damping_reduction(raw: dict[str, Any]) -> None:
    """Keep the rule of a snapshot older than 2026-10-13 that left rlmmac's reduction unset.

    It divided the damping by ``factor`` after an accepted step; the default is now 0.9
    ``reduction = None`` keeps the division, so a resume stays
    bit-identical. Only the rlmmac scheme: ``[method.damping] name = "rlmmac"``, or no
    damping section and a method whose schedule is :class:`DampingSchedule` (GIES, and
    the method CHOP wraps).
    """
    if _pin_scheme_option(raw, "rlmmac", DampingSchedule, "reduction", None):
        # Every read of an old snapshot migrates it (summaries, figures, the monitor): debug.
        logger.debug(
            "run from before 2026-10-13: its damping is divided by the factor after an "
            "accepted step",
            extra={"event": "snapshot_migrated", "damping_reduction": None},
        )


def _migrate_lm_confirmation(raw: dict[str, Any]) -> None:
    """Keep PET's stopping rule in a snapshot older than 2026-10-19 with the ``lm`` scheme.

    The first improving trial within the tolerance stopped such a run; the default is now
    ``confirm_convergence = true``. Pinning it false keeps a resume identical.
    """
    if _pin_scheme_option(raw, "lm", LMDampingSchedule, "confirm_convergence", False):
        logger.debug(
            "run from before 2026-10-19: the first small improvement stops it",
            extra={"event": "snapshot_migrated", "confirm_convergence": False},
        )


def _pin_scheme_option(
    raw: dict[str, Any], scheme: str, schedule_type: type, key: str, value: object
) -> bool:
    """Set ``key`` to ``value`` in an old snapshot's damping scheme unless it gave one.

    The scheme is ``[method.damping] name = scheme``, or no damping section and a method
    whose own schedule is ``schedule_type`` (for CHOP, that of the method it wraps).
    Malformed sections are left to the configuration's validation. Returns whether the
    snapshot names that scheme.
    """
    method = raw.get("method")
    if not isinstance(method, dict):
        return False
    damping = method.get("damping")
    if isinstance(damping, dict):
        options = damping.get("options") or {}
        if damping.get("name") == scheme and isinstance(options, dict):
            damping["options"] = {key: value, **options}
            return True
        return False
    name = method.get("name")
    if not isinstance(name, str) or name not in methods.names():
        return False
    factory = methods.factory(name)
    if "inner_method" in inspect.signature(factory).parameters:
        own = method.get("options") or {}
        inner = own.get("inner", "gies") if isinstance(own, dict) else None
        if not isinstance(inner, str) or inner not in methods.names():
            return False
        factory = methods.factory(inner)
    if "schedule" not in inspect.signature(factory).parameters:
        return False
    if getattr(factory, "schedule_type", DampingSchedule) is not schedule_type:
        return False
    schedule = method.get("schedule") or {}
    if not isinstance(schedule, dict):
        return False
    method["schedule"] = {key: value, **schedule}
    return True


def _migrate_localization_name(raw: dict[str, Any]) -> None:
    """Read ``"adaptive"`` in a snapshot older than 2026-09-28 as ``"adaptive_per_update"``.

    It named the per-update method then; it names the prior-only one now. A resume of such
    a run must keep its method, to stay bit-identical.
    """
    localization = (raw.get("method") or {}).get("localization")
    if isinstance(localization, dict) and localization.get("name") == "adaptive":
        localization["name"] = "adaptive_per_update"
        logger.info(
            "run from before 2026-09-28: its 'adaptive' localization is the per-update one",
            extra={"event": "snapshot_migrated", "localization": "adaptive_per_update"},
        )


def _write_summary(run: Run, result: HistoryMatchResult) -> None:
    summary = run_summary(
        result.history,
        name=run.config.run.name,
        method=run.config.method.name,
        converged=result.converged,
        complete=result.complete,
    )
    if result.selected_iteration is not None:
        summary["validation"] = {
            "selected_iteration": result.selected_iteration,
            "stopped_by_cv": result.stopped_by_cv,
            "last_iteration": result.history[-1].iteration,
        }
    (run.run_dir.path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


class _JsonLines(logging.Formatter):
    """One JSON object per record, including the structured ``extra`` fields."""

    _standard: ClassVar[frozenset[str]] = frozenset(vars(logging.makeLogRecord({})))

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in vars(record).items() if k not in self._standard}
        return json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "message": record.getMessage(),
                **fields,
            },
            default=str,
        )


#: Environment of a job's worker: the services root and the job, whose event
#: stream in the store receives a copy of every event.
JOB_ROOT_ENV, JOB_ID_ENV = "APEX_JOB_ROOT", "APEX_JOB_ID"


class _JobEvents(logging.Handler):
    """Copies events into a job's stream in the services store (a job worker)."""

    def __init__(self, root: Path, job_id: str) -> None:
        super().__init__()
        from apex.storage.sqlite_store import SQLiteStore

        self.store = SQLiteStore(root)
        self.job_id = job_id
        self.formatter_ = _JsonLines()

    def emit(self, record: logging.LogRecord) -> None:
        """Store the event (failures to store never stop the run)."""
        try:
            self.store.add_event_soon(self.job_id, json.loads(self.formatter_.format(record)))
        except Exception:  # noqa: BLE001 (logging must never raise)
            self.handleError(record)


def configure_logging(run_path: Path, level: int = logging.INFO) -> None:
    """Log structured events to ``<run>/events.jsonl`` and warnings to stderr."""
    root = logging.getLogger("apex")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(run_path / "events.jsonl", encoding="utf-8")
    file_handler.setFormatter(_JsonLines())
    job_root, job_id = os.environ.get(JOB_ROOT_ENV), os.environ.get(JOB_ID_ENV)
    if job_root and job_id:
        # A job worker also collects every event in its job's stream.
        root.addHandler(_JobEvents(Path(job_root), job_id))
    console = logging.StreamHandler(sys.stderr)
    console.addFilter(_for_console)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)


#: INFO events also shown on the console: what the run will save beyond the defaults.
CONSOLE_EVENTS = frozenset({"diagnostics_plan"})


def _for_console(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING or getattr(record, "event", None) in CONSOLE_EVENTS


def load_results(run_path: Path) -> RunResults:
    """Load the saved results of the run in ``run_path`` (see :class:`RunResults`).

    Raises
    ------
    ConfigError
        If the run has no configuration snapshot or no scored forecast yet.
    """
    config = snapshot_config(RunDirectory(run_path))
    try:
        return RunResults(run_path, build_layout(config))
    except (OSError, ValueError) as error:
        raise ConfigError(f"{run_path} has no results yet: {error}") from error


def analysis_observations(run_path: Path) -> ObservationSet:
    """Rebuild a run's observations in its analysis data space (after its transforms)."""
    config = snapshot_config(RunDirectory(run_path))
    observations = read_observation_section(config.observations)
    for transform in build_transforms(config):
        observations = transform.fit(observations)
    return observations


def explore_data_split(
    config_path: Path,
    *,
    prior_run: Path | None = None,
    max_cv: int | None = None,
    output_dir: Path | None = None,
    font_size: float = 12.0,
) -> tuple[SplitExploration, list[Path]]:
    """Explore how to split a case's production data for cross validation.

    Writes the explorer's tables and figures to ``output_dir`` (default
    ``<config dir>/data_split``) and returns the report and the file paths.

    Parameters
    ----------
    config_path
        The case's configuration (its ``[validation]`` section, if any, is ignored).
    prior_run
        A run of the case that has forecast its prior (``iter_0``), for the
        prior-ensemble redundancy of each well.
    max_cv
        Largest candidate CV set, in wells.

    Raises
    ------
    ConfigError
        If the case has fewer than two wells, or ``prior_run`` has no prior forecast.
    """
    from apex.reporting.data_split_report import write_exploration
    from apex.workflows.data_split import explore, prior_forecast

    config = load_config(config_path)
    observations = read_observation_section(config.observations)
    prior_keys = prior_predicted = None
    if prior_run is not None:
        run = build_run(snapshot_config(RunDirectory(prior_run)))
        keys = {"history_matching": run.workflow.observations.keys}
        keys.update({r: o.keys for r, o in run.workflow.held_out_observations.items()})
        try:
            prior_keys, prior_predicted = prior_forecast(RunDirectory(prior_run), keys)
        except FileNotFoundError as error:
            raise ConfigError(f"{prior_run} has no prior forecast (iter_0) yet") from error
    try:
        report = explore(
            observations, prior_keys=prior_keys, prior_predicted=prior_predicted, max_cv=max_cv
        )
    except ValueError as error:
        raise ConfigError(f"data split: {error}") from error
    output = output_dir or config_path.resolve().parent / "data_split"
    return report, write_exploration(report, output, font_size=font_size)


def write_report_figures(
    run_path: Path,
    output_dir: Path | None = None,
    *,
    truth_response: NDArray[np.float64] | None = None,
    font_size: float = 12.0,
) -> list[Path]:
    """Write a run's result figures (PNG and PDF, one figure per file) and metric table.

    In ``output_dir`` (default ``<run>/report``), as PET-fork's ``Figures/`` folders:

    - ``metrics/``: ``data_mismatch`` and ``metric_<metric>_<label>``, per-member distributions at
      every iteration (box plots), and ``metrics_summary.csv`` (mean, std, median, p10,
      p90 of every metric per kept iteration);
    - ``data/``: per data type, the prior (``_initial``) and final (``_final``) ensemble
      forecasts against the observations (scalar data), with ``truth_response`` (the true
      model's forecast at the observation keys, in the analysis data space) when given;
    - ``fields/``: per variable with a ``grid`` and per layer, the true field (synthetic
      cases), the prior and final means and standard deviations, and their differences;
    - ``histograms/``: per such variable, the prior and final value distributions;
    - ``health/``: ensemble health and convergence (the run monitor's facts):
      ``spread`` (each variable's spread relative to the prior's, with the collapse
      threshold of ``[agents.monitor]``), ``data_mismatch_change`` (the relative change of the
      accepted mean data mismatch, with the method's tolerance), ``members`` (members simulated
      per forecast) and, for iterative methods, ``damping`` (every trial's).

    Returns the PNG paths (each has a PDF twin) and the CSV path.

    Raises
    ------
    ConfigError
        If the run has no configuration snapshot or no scored forecast yet.
    """
    from apex.reporting.result_plots import (
        field_layer_figures,
        histogram_figures,
        metric_figure,
        production_figures,
    )

    results = load_results(run_path)
    output = output_dir or results.path / "report"
    figures: list[Path] = []
    for metric, labels in results.metric_labels().items():
        for label in labels:
            iterations, values, accepted = results.metric(metric, label)
            # The total keeps its name; each data type's share has a file of its own (they
            # once shared one, the last overwriting the total).
            total = metric == "data_mismatch" and label == "all"
            name = "data_mismatch" if total else f"metric_{_slug(metric)}_{_slug(label)}"
            figures.append(
                metric_figure(
                    values,
                    iterations,
                    output / "metrics" / name,
                    title="Data mismatch per iteration" if total else f"{metric} ({label})",
                    ylabel=f"{metric} per member",
                    accepted=accepted,
                    log=metric
                    in (
                        "data_mismatch",
                        "normalized_data_mismatch",
                        "cv_normalized_data_mismatch",
                        "test_normalized_data_mismatch",
                    ),
                    selected=results.selected_iteration,
                    font_size=font_size,
                )
            )
    table = output / "metrics" / "metrics_summary.csv"
    rows = results.metric_summary()
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures.append(table)

    # "final" is the posterior: with cross validation the minimum-CV iterate.
    prior, final = results.iterate(0), results.iterate(results.posterior_iteration)
    run = build_run(snapshot_config(results.run_dir))
    figures += _health_figures(results, run, output / "health", font_size)
    workflow = run.workflow
    observations = workflow.analysis_observations
    role_rows = workflow.role_rows()
    prior_all, final_all = (
        all_roles(prior, role_rows, observations.n_data),
        all_roles(final, role_rows, observations.n_data),
    )
    assigned = np.zeros(observations.n_data, dtype=bool)
    for indices in role_rows.values():
        assigned[indices] = True
    # Field components and wavelet coefficients are not time series; plot scalars only
    # (and not data a [validation] section leaves unused).
    scalar = [i for i, key in enumerate(observations.keys) if key.index is None and assigned[i]]
    if scalar:
        figures += production_figures(
            [observations.keys[i].time for i in scalar],
            [observations.keys[i].datatype for i in scalar],
            observations.values[scalar],
            observations.variances[scalar],
            prior_all[scalar],
            final_all[scalar],
            output / "data",
            truth=None if truth_response is None else np.asarray(truth_response)[scalar],
            font_size=font_size,
        )
    truth = results.truth()
    for variable in results.layout.variables:
        if variable.grid is None:
            continue
        true_field = None if truth is None else truth[variable.name]
        figures += field_layer_figures(
            variable.name,
            variable.grid,
            prior.state[variable.name],
            final.state[variable.name],
            output / "fields",
            truth=true_field,
            font_size=font_size,
        )
        figures += histogram_figures(
            variable.name,
            prior.state[variable.name],
            final.state[variable.name],
            output / "histograms",
            truth=true_field,
            font_size=font_size,
        )
    properties = workflow.property_map
    if not properties.identity:
        # The properties the simulator saw (facies, porosity, ...), mapped from
        # the prior, the final iterate and the truth.
        before, after = properties.ensemble(prior.state), properties.ensemble(final.state)
        true_properties = None if truth is None else properties.member(truth)
        for name, grid in sorted(properties.grids.items()):
            if name in results.layout.names:
                continue  # a state variable passed through: plotted above
            true_field = None if true_properties is None else true_properties[name]
            figures += field_layer_figures(
                name, grid, before[name], after[name], output / "properties",
                truth=true_field, font_size=font_size,
            )  # fmt: skip
            figures += histogram_figures(
                name, before[name], after[name], output / "histograms",
                truth=true_field, font_size=font_size,
            )  # fmt: skip
    return figures


def all_roles(iterate: Iterate, rows: dict[str, NDArray[np.intp]], n_data: int) -> np.ndarray:
    """Return an iterate's forecast of every role in the analysis data space (NaN: unused)."""
    predicted = np.full((n_data, iterate.member_ids.size), np.nan)
    predicted[rows["history_matching"]] = iterate.predicted
    for role, values in (("cv", iterate.cv_predicted), ("test", iterate.test_predicted)):
        if role in rows and values is not None:
            predicted[rows[role]] = values
    return predicted


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _health_figures(results: RunResults, run: Run, output: Path, font_size: float) -> list[Path]:
    """Write the ensemble-health and convergence figures of :func:`write_report_figures`."""
    from apex.reporting.result_plots import (
        damping_figure,
        data_mismatch_change_figure,
        members_figure,
        spread_figures,
    )
    from apex.services.settings import MonitorSection

    history = results.history
    accepted = [r for r in history if r.accepted]
    prior = results.iterate(0)
    before = {n: float(np.mean(np.std(v, axis=-1, ddof=1))) for n, v in prior.state.items()}
    ratios: dict[str, list[float]] = {name: [] for name in before}
    for record in accepted:
        state = results.iterate(record.iteration).state
        for name, spread in before.items():
            now = float(np.mean(np.std(state[name], axis=-1, ddof=1)))
            ratios[name].append(now / spread if spread > 0 else float("nan"))
    kept = [r.iteration for r in accepted]
    figures = spread_figures(
        kept, ratios, output, threshold=MonitorSection().collapse_spread_ratio, font_size=font_size
    )
    if len(accepted) > 1:
        means = [r.data_mismatch_mean for r in accepted]
        changes = [abs(1 - b / a) if a > 0 else float("nan") for a, b in itertools.pairwise(means)]
        schedule = (
            getattr(run.method, "step_schedule", None)
            or getattr(run.method, "lm_schedule", None)
            or getattr(run.method, "schedule", None)
        )
        tolerance = getattr(schedule, "data_mismatch_tolerance", None)
        figures.append(
            data_mismatch_change_figure(
                kept[1:],
                changes,
                output / "data_mismatch_change",
                tolerance=tolerance,
                font_size=font_size,
            )
        )
    iterations = [r.iteration for r in history]
    size = run.config.prior.ensemble_size or history[0].n_members
    figures.append(
        members_figure(
            iterations,
            [r.n_members for r in history],
            size,
            output / "members",
            font_size=font_size,
        )
    )
    trials = [r for r in history[1:] if r.damping is not None]
    if trials:
        figures.append(
            damping_figure(
                [r.iteration for r in trials],
                [float(r.damping or 0.0) for r in trials],
                [r.accepted for r in trials],
                output / "damping",
                font_size=font_size,
                label=getattr(run.method, "control_name", "damping"),
            )
        )
    return figures
