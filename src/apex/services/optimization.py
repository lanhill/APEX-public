"""Composition of optimization runs: configuration to optimizer, objective and problem.

Plugins are chosen by name through the registries (``optimizers``, ``objectives``,
``response_objectives`` with ``simulators`` and ``executors``); the workflow
(``workflows/optimization.py``) runs them with checkpoints. A response objective (phase 2)
is wrapped in :class:`~apex.workflows.simulated_objective.SimulatedObjective`, which
simulates each evaluation once and journals it.
"""

import hashlib
import json
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError

import apex.algorithms.optimization  # registers the built-in plugins
import apex.services.composition  # also registers the simulators and executors
from apex.algorithms.optimization import (
    GeneratorDraws,
    OptimizerBase,
    Problem,
    objectives,
    optimizers,
    response_objectives,
)
from apex.algorithms.optimization.combined import StackedResponses, stack_objectives
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    FloatArray,
    Objective,
    ResponseObjective,
    VectorObjective,
)
from apex.algorithms.optimization.statistics import needs_realizations
from apex.core.errors import ApexError, ConfigError, PluginError, validation_problems
from apex.core.plugins import executors, parameterizations, simulators
from apex.core.progress import ProgressReporter
from apex.core.simulation import Simulator
from apex.domain.config import PluginSection
from apex.domain.optimization import OptimizationConfig
from apex.domain.services import OptimizationPlan, OptimizationSummary
from apex.domain.state import StateLayout, StateVariable
from apex.storage import optimization_directory as layout
from apex.workflows.optimization import OptimizationRun, run_optimization
from apex.workflows.parameterization import PropertyMap, parameterized
from apex.workflows.simulated_objective import SimulatedObjective

#: Options naming files, resolved against the configuration's directory (as for runs).
PATH_OPTION_SUFFIXES = ("_file", "_dir", "_path")


def load_optimization_config(path: Path, *, place: bool = True) -> OptimizationConfig:
    """Read and validate an optimization configuration, resolving its paths.

    With ``place`` an unset output directory becomes ``runs/<name>`` next to the file;
    without, it stays unset (the job service places it).

    Raises
    ------
    ConfigError
        If the file cannot be read or does not match the schema.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read optimization configuration {path}: {error}") from error
    return resolve_optimization_config(raw, path.resolve().parent, place=place, source=path)


def resolve_optimization_config(
    config: OptimizationConfig | Mapping[str, Any],
    base: Path,
    *,
    place: bool = True,
    source: Path | None = None,
) -> OptimizationConfig:
    """Validate a configuration and resolve its relative paths against ``base``.

    Raises
    ------
    ConfigError
        If it does not match the schema.
    """
    if not isinstance(config, OptimizationConfig):
        try:
            config = OptimizationConfig.model_validate(config)
        except ValidationError as error:
            where = f" {source}" if source is not None else ""
            raise ConfigError(
                f"invalid optimization configuration{where}:\n{validation_problems(error)}"
            ) from error

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    def resolve_options(section: PluginSection) -> PluginSection:
        options = {
            key: str(resolve(Path(value)))
            if key.endswith(PATH_OPTION_SUFFIXES) and isinstance(value, str)
            else value
            for key, value in section.options.items()
        }
        return section.model_copy(update={"options": options})

    output = config.optimization.output_dir
    if output is None and place:
        output = Path("runs") / config.optimization.name
    realizations = config.model.realizations_file
    return config.model_copy(
        update={
            "optimization": config.optimization.model_copy(
                update={"output_dir": output and resolve(output)}
            ),
            "simulator": None if config.simulator is None else resolve_options(config.simulator),
            "objective": None if config.objective is None else resolve_options(config.objective),
            "objectives": [resolve_options(entry) for entry in config.objectives],
            "parameterizations": [resolve_options(p) for p in config.parameterizations],
            "model": config.model.model_copy(
                update={"realizations_file": realizations and resolve(realizations)}
            ),
        }
    )


def is_optimization(raw: Mapping[str, Any]) -> bool:
    """Return whether a parsed TOML document is an optimization configuration."""
    return "optimization" in raw and "optimizer" in raw


def plan_optimization(config: OptimizationConfig) -> OptimizationPlan:
    """Return what an optimization will cost; nothing is evaluated.

    Raises
    ------
    ConfigError
        If the configuration cannot be built.
    """
    from apex.services.identity import optimization_id

    if config.optimization.output_dir is None:  # a plan needs a place to build its objective
        config = config.model_copy(
            update={"optimization": config.optimization.model_copy(
                update={"output_dir": Path("/nonexistent/plan")}
            )}
        )  # fmt: skip
    optimizer, problem, _ = build(config)
    evaluations = optimizer.max_evaluations(problem.n_realizations)
    simulated = config.simulator is not None
    workers = config.execution.options.get("workers")
    warnings = []
    if simulated and evaluations is None:
        warnings.append(f"{config.optimizer.name} cannot bound its evaluations")
    statistics = [e.statistic for e in config.objectives if e.statistic != "mean"]
    if statistics and problem.n_realizations == 1:
        warnings.append(
            f"with one realization the statistics {', '.join(statistics)} equal the mean: those "
            "objectives repeat it (give realizations, [model] realizations_file)"
        )
    return OptimizationPlan(
        optimization_id=optimization_id(config),
        name=config.optimization.name,
        optimizer=config.optimizer.name,
        objective=config.objective_label,
        n_controls=int(problem.x0.size),
        n_realizations=problem.n_realizations,
        max_evaluations=evaluations,
        max_simulator_runs=(evaluations if simulated else 0),
        simulator=None if config.simulator is None else config.simulator.name,
        parallel_runs=int(workers) if simulated and workers else (1 if simulated else None),
        output_dir=None
        if config.optimization.output_dir is None
        else str(config.optimization.output_dir),
        warnings=warnings,
    )


def holds_optimization(run_dir: Path) -> bool:
    """Return whether a directory holds an optimization run (not a history match's).

    From its configuration alone: the history's first row comes only once the start is
    evaluated, which for a simulated objective may take a while.
    """
    if not (run_dir / layout.CONFIG).exists():
        return False
    try:
        return "optimizer" in layout.read_config(run_dir)
    except (OSError, ValueError):
        return False


def write_optimization_figures(
    run_dir: Path, output: Path | None = None, *, font_size: float = 12.0
) -> list[Path]:
    """Write an optimization's figures (``<run>/report`` by default); return the PNG paths.

    The objective's history, and for several objectives the front and the
    hypervolume over the generations (``reporting/optimization_plots.py``).
    """
    from apex.reporting.optimization_plots import optimization_figures

    return optimization_figures(
        layout.read_history(run_dir), layout.read_pareto(run_dir), output or run_dir / "report",
        font_size=font_size,
    )  # fmt: skip


def optimization_summary(run_dir: Path) -> OptimizationSummary:
    """Return an optimization run's progress and result.

    Raises
    ------
    ConfigError
        If ``run_dir`` holds no optimization run.
    """
    from apex.storage.locks import locked

    try:
        raw = layout.read_config(run_dir)
        config = OptimizationConfig.model_validate(raw)
    except (OSError, ValueError) as error:
        raise ConfigError(f"no optimization run in {run_dir}: {error}") from error
    result_path = run_dir / layout.RESULT
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    history = layout.read_history(run_dir)
    pareto = layout.read_pareto(run_dir)
    journal = layout.read_evaluations(run_dir)
    state: Literal["new", "running", "stopped", "finished"]
    if locked(run_dir):
        state = "running"
    elif result is not None:
        state = "finished"
    else:
        state = "stopped" if history else "new"
    problem_names = [
        variable.name if variable.size == 1 else f"{variable.name}[{i}]"
        for variable in config.controls.variables
        for i in range(variable.size)
    ]
    multi = _multi_objective(config)
    return OptimizationSummary(
        path=str(run_dir),
        name=config.optimization.name,
        state=state,
        optimizer=config.optimizer.name,
        objective=config.objective_label,
        controls=problem_names,
        history=history,
        result=result,
        pareto=pareto,
        evaluations=sum(1 for row in journal if _evaluated(row)),
        failed_evaluations=sum(1 for row in journal if not _evaluated(row)),
        simulated=config.simulator is not None,
        penalised=bool(config.optimizer.options.get("epf")),
        # Several objectives have scalings of their own: the summary shows them minimised.
        objective_scaling=None
        if config.objective is None
        else _scaling(config.objective.options.get("obj_scaling")),
        multi_objective=multi,
        objective_names=_objective_names(config, result, pareto) if multi else None,
        objective_scalings=[_scaling(e.options.get("obj_scaling")) for e in config.objectives]
        or None,
    )


def _multi_objective(config: OptimizationConfig) -> bool:
    """Whether the run has several objectives: ``[[objectives]]`` or a multi-objective optimizer."""
    if config.objectives:
        return True
    try:
        return bool(getattr(optimizers.factory(config.optimizer.name), "multi_objective", False))
    except ApexError:  # an optimizer no longer registered, or broken: none is assumed
        return False


def _objective_names(
    config: OptimizationConfig, result: Mapping[str, Any] | None, pareto: Mapping[str, Any] | None
) -> list[str] | None:
    """Return a multi-objective run's objective names, as results show them; ``None`` if unknown.

    Called only for a multi-objective run, so no plugin is created for others. From
    ``[[objectives]]``, else what the run wrote (its result, its front), else a vector
    objective plugin's own ``objective_names`` (analytic plugins only: cheap to create).
    """
    if config.objectives:
        return [entry.shown for entry in config.objectives]
    for written in (result, pareto):
        names = (written or {}).get("objective_names")
        if isinstance(names, list) and len(names) > 1:
            return [str(n) for n in names]
    section = config.objective
    if section is None or section.name not in objectives.names():
        return None
    try:
        plugin = objectives.create(section.name, **section.options)
    except (ApexError, ValueError, TypeError):  # options a newer plugin refuses: unknown
        return None
    names = getattr(plugin, "objective_names", None)
    return [str(n) for n in names] if names and len(names) > 1 else None


def _evaluated(row: Mapping[str, Any]) -> bool:
    """Whether a journal row holds a value: ``value``, or ``values`` for several objectives."""
    return row.get("value") is not None or row.get("values") is not None


def _scaling(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def control_blocks(config: OptimizationConfig) -> list[tuple[str, slice]]:
    """Return each control block's name and slice of the control vector."""
    blocks, start = [], 0
    for variable in config.controls.variables:
        blocks.append((variable.name, slice(start, start + variable.size)))
        start += variable.size
    return blocks


def build_problem(
    config: OptimizationConfig, objective: Objective, n_realizations: int | None = None
) -> tuple[Problem, FloatArray | None]:
    """Return the problem and, if every block has a variance, the initial covariance.

    ``n_realizations`` overrides ``[controls] realizations`` (the model's count).
    """
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    variances: list[float | None] = []
    names: list[str] = []
    for variable in config.controls.variables:
        values = (
            [float(variable.initial)]
            if isinstance(variable.initial, float | int)
            else [float(v) for v in variable.initial]
        )
        x0 += values
        lower += [-np.inf if variable.lower is None else variable.lower] * len(values)
        upper += [np.inf if variable.upper is None else variable.upper] * len(values)
        variances += [variable.variance] * len(values)
        names += (
            [variable.name]
            if len(values) == 1
            else [f"{variable.name}[{i}]" for i in range(len(values))]
        )
    low, high = np.array(lower), np.array(upper)
    problem = Problem(
        objective,
        np.array(x0),
        lower=None if np.all(np.isinf(low)) else low,
        upper=None if np.all(np.isinf(high)) else high,
        n_realizations=config.controls.realizations if n_realizations is None else n_realizations,
        names=tuple(names),
    )
    cov = None if any(v is None for v in variances) else np.diag(np.array(variances, dtype=float))
    return problem, cov


def build(
    config: OptimizationConfig, progress: ProgressReporter | None = None
) -> tuple[OptimizerBase, Problem, FloatArray | None]:
    """Create the optimizer and the problem a configuration names.

    Raises
    ------
    ConfigError
        If a plugin is unknown, rejects its options, or the problem lacks what it needs.
    """
    sections = config.objective_sections
    kinds = {section.name in response_objectives.names() for section in sections}
    if len(kinds) > 1:
        raise ConfigError("[[objectives]] mixes simulated and analytic objectives: use one kind")
    simulated = kinds.pop()
    label = config.objective_label
    if simulated and config.simulator is None:
        raise ConfigError(f"objective {label!r} is computed from simulations: add [simulator]")
    if not simulated and config.simulator is not None:
        raise ConfigError(
            f"objective {label!r} is analytic; [simulator] is for objectives "
            f"of simulated responses ({', '.join(response_objectives.names())})"
        )
    n_realizations: int | None = None
    try:
        optimizer = optimizers.create(config.optimizer.name, **config.optimizer.options)
        if simulated:
            simulation = _simulated_objective(config, progress)
            n_realizations = simulation.n_realizations
            objective: Objective = simulation
        elif config.objective is not None:
            objective = objectives.create(config.objective.name, **config.objective.options)
        else:
            objective = stack_objectives(
                [entry.shown for entry in config.objectives],
                [objectives.create(entry.name, **entry.options) for entry in config.objectives],
                [entry.statistic for entry in config.objectives],
            )
    except PluginError as error:
        raise ConfigError(str(error)) from error
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid optimizer or objective options: {error}") from error
    problem, cov = build_problem(config, objective, n_realizations)
    # The optimizer refuses a mismatch only when it starts; validation, plans and studies
    # must see it first (a reported issue, 2026-10-06: a planned nsga2 cell on a scalar case).
    several = isinstance(objective, VectorObjective)
    if several and not optimizer.multi_objective:
        raise ConfigError(
            f"optimizer {config.optimizer.name!r} optimizes one objective, and {label!r} has "
            "several: use nsga2, or one objective"
        )
    if several and optimizer.multi_objective:
        # NSGA-III and MOEA/D size their directions and population by the objectives (ADR
        # 0056), before the plan's bound is computed.
        set_objectives = getattr(optimizer, "set_objectives", None)
        if callable(set_objectives):
            try:
                set_objectives(len(cast(VectorObjective, objective).objective_names))
            except ValueError as error:
                raise ConfigError(f"invalid optimizer options: {error}") from error
    if several:
        # What the optimizer refuses only when it starts, refused here for validation, plans
        # and studies: a statistic needing more realizations, and MOEA/D's
        # constraints without EPF.
        statistics = tuple(getattr(objective, "objective_statistics", ()))
        if problem.n_realizations < needs_realizations(statistics):
            raise ConfigError(
                "the statistic std needs at least two realizations ([model] realizations_file "
                f"or [controls] realizations), and this optimization has {problem.n_realizations}"
            )
        if (getattr(optimizer, "constraints_by_epf_only", False)
                and isinstance(objective, ConstrainedObjective)
                and getattr(optimizer, "constraint_handling", None) != "epf"):  # fmt: skip
            raise ConfigError(
                f"optimizer {config.optimizer.name!r} handles constraints by the external "
                'penalty only: set constraint_handling = "epf" and epf.r, or use '
                "nsga2 or nsga3"
            )
    if optimizer.multi_objective and not several:
        raise ConfigError(
            f"optimizer {config.optimizer.name!r} needs several objectives, and {label!r} is "
            "one: list them in [[objectives]], or use a single-objective optimizer"
        )
    if optimizer.needs_covariance and cov is None:
        raise ConfigError(
            f"optimizer {config.optimizer.name!r} needs a variance for every control block"
        )
    return optimizer, problem, cov


def _simulated_objective(
    config: OptimizationConfig, progress: ProgressReporter | None
) -> SimulatedObjective:
    """Wrap the configuration's response objective in its simulator and executor."""
    assert config.simulator is not None  # checked by build
    run_dir = config.optimization.output_dir
    if run_dir is None:
        raise ConfigError("the optimization's output directory is not resolved")
    model: dict[str, np.ndarray] = {}
    if config.model.realizations_file is not None:
        try:
            with np.load(config.model.realizations_file) as stored:
                model = {name: np.asarray(stored[name], dtype=np.float64) for name in stored.files}
        except (OSError, ValueError) as error:
            raise ConfigError(f"cannot read [model] realizations_file: {error}") from error
        flat = [name for name, array in model.items() if array.ndim != 2]
        if flat:
            raise ConfigError(
                f"[model] arrays {flat} must be (n_values, n_realizations), one column each"
            )
    response: ResponseObjective
    if config.objective is not None:
        response = response_objectives.create(config.objective.name, **config.objective.options)
    else:
        response = StackedResponses(
            [entry.shown for entry in config.objectives],
            [response_objectives.create(e.name, **e.options) for e in config.objectives],
            [e.statistic for e in config.objectives],
        )
    simulator = _parameterized(
        simulators.create(config.simulator.name, **config.simulator.options), config, model
    )
    executor = executors.create(config.execution.name, **config.execution.options)
    try:
        apex.services.composition.check_executor(executor, simulator)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    objective = SimulatedObjective(
        response,
        simulator,
        executor,
        run_dir,
        control_blocks(config),
        model,
        keep_simulations=config.output.keep_simulations,
        progress=progress,
        context=evaluation_context(config),
    )
    explicit = "realizations" in config.controls.model_fields_set
    if model and explicit and config.controls.realizations != objective.n_realizations:
        raise ConfigError(
            f"[controls] realizations is {config.controls.realizations}, the [model] file "
            f"holds {objective.n_realizations}; leave realizations out"
        )
    if not model and config.controls.realizations > 1:
        raise ConfigError(
            "[controls] realizations > 1 needs the realizations' models: [model] realizations_file"
        )
    return objective


def _parameterized(
    simulator: Simulator, config: OptimizationConfig, model: Mapping[str, np.ndarray]
) -> Simulator:
    """Wrap the simulator in ``[[parameterizations]]``, if any (the controls as the state).

    Raises
    ------
    ConfigError
        If a parameterization is unknown, its options are invalid, or it consumes a
        variable that neither the controls, the model arrays nor an earlier one provides.
    """
    if not config.parameterizations:
        return simulator
    built = []
    for section in config.parameterizations:
        try:
            built.append(parameterizations.create(section.name, **section.options))
        except (PluginError, TypeError, ValueError) as error:
            raise ConfigError(f"[[parameterizations]] {section.name}: {error}") from error
    variables = [StateVariable(v.name, v.size) for v in config.controls.variables]
    variables += [StateVariable(name, a.shape[0]) for name, a in model.items()]
    try:
        properties = PropertyMap(built, StateLayout(variables))
        # Sizes are checked only when a member is mapped: map the initial controls (with
        # each model array's first realization) now, not in the first simulation.
        state = {
            v.name: np.broadcast_to(np.asarray(v.initial, dtype=np.float64), (v.size,)).copy()
            for v in config.controls.variables
        }
        state.update({name: np.asarray(a[:, 0], dtype=np.float64) for name, a in model.items()})
        properties.member(state)
    except ValueError as error:
        raise ConfigError(f"[[parameterizations]]: {error}") from error
    return parameterized(simulator, properties)


def evaluation_context(config: OptimizationConfig) -> str:
    """Return a hash of what decides a simulated evaluation besides its controls.

    The objective, simulator and parameterization sections, and the bytes of the files they
    name (the deck template, an eCalc model, a rule) and of the model file. It is part of
    every journal key, so a resume after any of them changed simulates again instead of
    reusing stale values.
    """
    digest = hashlib.sha256()
    sections = [*config.objective_sections, config.simulator, *config.parameterizations]
    # A statistic decides what is ranked, not what is simulated: left out, so the
    # journals of runs from before it keep their keys.
    digest.update(json.dumps([s.model_dump(mode="json", exclude={"statistic"})
                              for s in sections if s]).encode())  # fmt: skip
    files = [Path(v) for s in sections if s for k, v in sorted(s.options.items())
             if k.endswith(PATH_OPTION_SUFFIXES) and isinstance(v, str)]  # fmt: skip
    if config.model.realizations_file is not None:
        files.append(config.model.realizations_file)
    for path in files:
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _draws(config: OptimizationConfig) -> GeneratorDraws:
    # One stream, spawned from the seed as the history-matching streams are.
    (child,) = np.random.SeedSequence(config.optimization.seed).spawn(1)
    return GeneratorDraws(np.random.default_rng(child))


def optimize(
    config_path: Path,
    max_iterations: int | None = None,
    progress: ProgressReporter | None = None,
) -> OptimizationRun:
    """Start the optimization a configuration describes."""
    config = load_optimization_config(config_path)
    return _run(config, resume=False, max_iterations=max_iterations, progress=progress)


def optimize_config(
    config: OptimizationConfig,
    max_iterations: int | None = None,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> OptimizationRun:
    """Start (or, if its directory holds one, continue) a resolved, placed optimization.

    A finished optimization is returned as it is (a job may find its run done).
    ``stopped``: whether the job running it was asked to stop (``run_optimization``).
    """
    run_dir = config.optimization.output_dir
    if run_dir is None:
        raise ConfigError("the optimization's output directory is not resolved")
    if (run_dir / layout.CONFIG).exists():
        return resume(run_dir, max_iterations, progress, stopped)
    return _run(
        config, resume=False, max_iterations=max_iterations, progress=progress, stopped=stopped
    )


def resume(
    run_dir: Path,
    max_iterations: int | None = None,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> OptimizationRun:
    """Continue an optimization run from its last checkpoint.

    Raises
    ------
    ConfigError
        If ``run_dir`` holds no optimization run.
    """
    try:
        raw = layout.read_config(run_dir)
    except (OSError, ValueError) as error:
        raise ConfigError(f"no optimization run in {run_dir}: {error}") from error
    config = OptimizationConfig.model_validate(raw)
    section = config.optimization.model_copy(update={"output_dir": run_dir.resolve()})
    return _run(
        config.model_copy(update={"optimization": section}),
        resume=True,
        max_iterations=max_iterations,
        progress=progress,
        stopped=stopped,
    )


def _run(
    config: OptimizationConfig,
    *,
    resume: bool,
    max_iterations: int | None,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> OptimizationRun:
    optimizer, problem, cov = build(config, progress)
    run_dir = config.optimization.output_dir
    if run_dir is None:
        raise ConfigError("the optimization's output directory is not resolved")
    serialised: dict[str, Any] = config.model_dump(mode="json")
    try:
        return run_optimization(
            run_dir,
            optimizer,
            problem,
            _draws(config),
            cov=cov,
            config=serialised,
            resume=resume,
            max_iterations=max_iterations,
            stopped=stopped,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error


def row_evaluations(row: Mapping[str, object]) -> int:
    """Return the objective evaluations a history row (or the result) counts so far.

    Its members' evaluations (ensemble and population optimizers), else its function
    evaluations (deterministic optimizers, whose members are not counted); 0 if neither.
    """
    for key in ("member_evaluations", "nfev"):
        value = row.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return int(value)
    return 0


def objective_evaluations(summary: OptimizationSummary) -> int:
    """Return an optimization's objective evaluations so far, analytic or simulated.

    The result's when finished (its last iteration included, which may have found no
    improving step), else the last history row's (:func:`row_evaluations`).
    """
    for source in ([summary.result] if summary.result else []) + list(reversed(summary.history)):
        if count := row_evaluations(source):
            return count
    return 0


def accepted_steps(summary: OptimizationSummary) -> int:
    """Return the steps an optimization accepted after its start (its history's step rows)."""
    return sum(1 for row in summary.history if row.get("kind", "step") == "step")
