"""Composition of predictions: configuration, ensembles, workflow, reports.

``apex predict PREDICTION.toml`` and ``apex report PRED_DIR`` call these functions; the
case-study runner and notebooks may too.
"""

import hashlib
import json
import logging
import os
import tomllib
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

import apex.algorithms.metrics  # noqa: F401  (registers the forecast scores)
from apex.algorithms import optimization as _optimization  # noqa: F401  (control_schedule)
from apex.core.errors import ApexError, ConfigError, PluginError, validation_problems
from apex.core.forecast import ForecastScore
from apex.core.plugins import (
    executors,
    forecast_scores,
    model_error_models,
    parameterizations,
    simulators,
)
from apex.core.progress import ProgressReporter
from apex.core.simulation import (
    FORECAST_STATE_OPTION,
    ExtrapolatingModel,
    ModelErrorModel,
    SeriesRequest,
    extrapolates,
)
from apex.domain.config import HistoryMatchConfig, PluginSection
from apex.domain.observations import ObservationSet
from apex.domain.prediction import PredictionConfig, PredictionControlsSection, PredictionResult
from apex.domain.state import StateLayout, StateVariable
from apex.services import composition
from apex.simulation.model_error import RESIDUAL_PREFIX, ModelErrorSimulator
from apex.storage import provenance
from apex.storage.forecast_state import write_forecast_state
from apex.storage.inputs import read_observation_section
from apex.storage.locks import LOCK_FILE, exclusive
from apex.storage.prediction_directory import FORECAST_STATE_PREFIX, PredictionDirectory
from apex.storage.run_directory import SURROGATE, RunDirectory
from apex.storage.run_results import RunResults
from apex.workflows.parameterization import PropertyMap, parameterized
from apex.workflows.prediction import (
    EnsembleInput,
    PredictionWorkflow,
    StoredForecast,
    first_prediction_time,
)

logger = logging.getLogger(__name__)


def load_prediction_config(path: Path) -> PredictionConfig:
    """Read and validate a prediction configuration, resolving relative paths.

    Raises
    ------
    ConfigError
        If the file cannot be read or does not match the schema.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read prediction configuration {path}: {error}") from error
    try:
        config = PredictionConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(
            f"invalid prediction configuration {path}:\n{validation_problems(error)}"
        ) from error
    return resolve_prediction_config(config, path.resolve().parent)


def resolve_prediction_config(config: PredictionConfig, base: Path) -> PredictionConfig:
    """Resolve a prediction configuration's relative paths against ``base``."""

    def resolve(value: Path) -> Path:
        return value if value.is_absolute() else (base / value).resolve()

    def resolve_options(options: dict[str, Any]) -> dict[str, Any]:
        return {
            key: str(resolve(Path(value)))
            if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
            else value
            for key, value in options.items()
        }

    section = config.prediction
    run_dir = resolve(section.run_dir)
    output_dir = (
        resolve(section.output_dir)
        if section.output_dir
        else run_dir / "predictions" / section.name
    )
    return config.model_copy(
        update={
            "prediction": section.model_copy(update={"run_dir": run_dir, "output_dir": output_dir}),
            "simulator": config.simulator.model_copy(
                update={"options": resolve_options(config.simulator.options)}
            ),
            "execution": None
            if config.execution is None
            else config.execution.model_copy(
                update={"options": resolve_options(config.execution.options)}
            ),
            "surrogate": None
            if config.surrogate is None
            else config.surrogate.model_copy(
                update={"model_file": resolve(config.surrogate.model_file)}
            ),
            "controls": None
            if config.controls is None
            else config.controls.model_copy(
                update={
                    "parameterizations": [
                        p.model_copy(update={"options": resolve_options(p.options)})
                        for p in config.controls.parameterizations
                    ]
                }
            ),
        }
    )


def simulator_section(config: PredictionConfig, run: HistoryMatchConfig) -> PluginSection:
    """Return the prediction's simulator: the run's, with the prediction's overrides."""
    override = config.simulator
    if override.name is not None and override.name != run.simulator.name:
        return PluginSection(name=override.name, options=override.options)
    return run.simulator.model_copy(
        update={"options": {**run.simulator.options, **override.options}}
    )


def _execution(executor: object, simulator: object) -> dict[str, object]:
    """Return what the executor says of where members run (its mode, hosts)."""
    describe = getattr(executor, "describe", None)
    if describe is None:
        return {}
    try:
        return {"execution": describe(simulator)}
    except (ApexError, OSError) as error:
        return {"execution": {"error": str(error)}}


def remote_config(
    run: HistoryMatchConfig,
    simulator: PluginSection,
    output_dir: Path,
    *,
    model_error: bool = False,
) -> HistoryMatchConfig:
    """Return the configuration a remote worker builds the prediction's simulator from.

    The run's, with the prediction's simulator and output folder, and nothing that wraps
    the run's simulator (geophysics, transforms: a prediction forecasts series), but its
    model-error model with ``model_error`` (one that forecasts: the members are g + r
    there too). A multilevel method's levels go too: every member of
    a prediction runs on the fine model here, so it must on the hosts.
    """
    options = {k: v for k, v in run.method.options.items() if k != "levels"}
    return run.model_copy(update={
        "simulator": simulator, "geophysics": None,
        "model_error": run.model_error if model_error else None,
        "data_transforms": [],
        "method": run.method.model_copy(update={"options": options}),
        "run": run.run.model_copy(update={"output_dir": output_dir}),
    })  # fmt: skip


def as_predicted(config: PredictionConfig, run: HistoryMatchConfig) -> HistoryMatchConfig:
    """Return the run's configuration as a prediction runs it.

    Its simulator and executor with the prediction's changes: what a check of a prediction
    must look at.
    """
    return run.model_copy(update={
        "simulator": simulator_section(config, run),
        "execution": config.execution or run.execution,
    })  # fmt: skip


def _fingerprint(
    config: PredictionConfig,
    run_dir: RunDirectory,
    simulator: PluginSection,
    model: str = "",
    extra: Mapping[str, Path] | None = None,
) -> str:
    # The configuration, the run it reads and the content of the simulator's files (e.g.
    # the deck template): a changed deck must not resume from members of the old one; nor
    # a rebuilt surrogate (``model``: its identity) from the old model's.
    digest = hashlib.sha256()
    digest.update(config.model_dump_json().encode())
    digest.update(model.encode())
    digest.update(run_dir.config_text().encode())
    # The controls' parameterizations' files too: none without [controls], so a plain
    # prediction's fingerprint is unchanged.
    # ``extra``: a forecasting model-error model's files and the observations it is fitted
    # to: a changed one must not resume from the old model's members;
    # none for a plain prediction, whose fingerprint is unchanged.
    found = {**_input_files(simulator), **_control_files(config), **(extra or {})}
    for name, path in sorted(found.items()):
        digest.update(name.encode())
        digest.update(provenance.file_hash(path).encode() if path.is_file() else b"missing")
    return digest.hexdigest()


def _input_files(
    simulator: PluginSection, model_error: PluginSection | None = None
) -> dict[str, Path]:
    """Return the input files the prediction's forecasts depend on, by option.

    The simulator's, and a forecasting model-error model's (its schedules).
    """
    found = {
        f"simulator.{key}": Path(value)
        for key, value in simulator.options.items()
        if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
    }
    if model_error is not None:
        found |= {
            f"model_error.{key}": Path(value)
            for key, value in model_error.options.items()
            if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
        }
    return found


def _control_files(config: PredictionConfig) -> dict[str, Path]:
    """Return the files ``[[controls.parameterizations]]`` name, as the simulator's."""
    if config.controls is None:
        return {}
    return {
        f"controls.parameterizations[{i}].{key}": Path(value)
        for i, section in enumerate(config.controls.parameterizations)
        for key, value in section.options.items()
        if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
    }


def _stored(
    results: RunResults,
    iteration: int,
    keys_rows: tuple[ObservationSet, dict[str, Any]],
    coarse: frozenset[int] = frozenset(),
) -> StoredForecast:
    """Return the run's forecast of an iterate at its scalar data (all roles).

    Members in ``coarse`` (a multilevel run's coarse levels) are left out: the
    run forecast them with a coarse model, the prediction runs every member on the fine
    one, so their history cannot be reproduced.
    """
    observations, role_rows = keys_rows
    iterate = results.iterate(iteration)
    # a surrogate run's posterior is the simulator's forecast of it (full_<k>);
    # other members a surrogate forecast cannot be reproduced exactly, so they are left out
    # of the check (phase 4 compares them within the model's gate).
    full = results.full(iteration)
    if full is not None:
        iterate = full
    predicted = composition.all_roles(iterate, role_rows, observations.n_data)
    scalar = [i for i, key in enumerate(observations.keys) if key.index is None]
    variances = np.ascontiguousarray(np.asarray(observations.variances, dtype=np.float64))
    keep = np.array(
        [
            int(m) not in coarse and source != SURROGATE
            for m, source in zip(iterate.member_ids, iterate.forecast_source, strict=True)
        ],
        dtype=bool,
    )
    if iterate.by_surrogate:
        logger.warning(
            "members forecast by a surrogate are left out of the history check",
            extra={"event": "history_check_surrogate", "iteration": iteration,
                   "left_out": int(np.count_nonzero(~keep))},
        )  # fmt: skip
    return StoredForecast(
        tuple(observations.keys[i] for i in scalar),
        predicted[scalar][:, keep],
        iterate.member_ids[keep],
        # The observation errors, for a surrogate prediction's gate.
        variances[scalar],
    )


def _ensembles(
    config: PredictionConfig, run_config: HistoryMatchConfig, results: RunResults
) -> tuple[list[EnsembleInput], np.ndarray | None]:
    workflow = composition.build_run(run_config, surrogate=False).workflow
    keys_rows = (workflow.analysis_observations, workflow.role_rows())
    # Every member is forecast on the prediction's (fine) model; a multilevel run's coarse
    # members were not, so the history check skips them.
    coarse = composition.coarse_member_ids(run_config)
    ensembles, truth = [], None
    for name in config.prediction.ensembles:
        if name == "truth":
            true_model = results.truth()
            if true_model is None:
                raise ConfigError(
                    f"{results.path} has no true model: 'truth' needs a synthetic case"
                )
            truth = results.layout.stack({k: v[:, None] for k, v in true_model.items()})[:, 0]
            continue
        if name == "posterior":
            iteration = results.posterior_iteration
        elif name == "prior":
            iteration = 0
        else:
            iteration = int(name.split(":")[1])
        try:
            iterate = results.iterate(iteration)
        except FileNotFoundError as error:
            raise ConfigError(f"ensemble {name!r}: {error}") from error
        ensembles.append(
            EnsembleInput(
                name,
                results.layout.stack(dict(iterate.state)),
                iterate.member_ids,
                _stored(results, iteration, keys_rows, coarse),
            )
        )
    return ensembles, truth


def predict(config_path: Path, progress: ProgressReporter | None = None) -> PredictionResult:
    """Start or resume the prediction ``config_path`` describes.

    Raises
    ------
    ConfigError
        If the configuration, the run or a plugin is invalid, or the output directory
        holds a prediction with another configuration.
    """
    return predict_config(load_prediction_config(config_path), progress)


def predict_config(
    config: PredictionConfig,
    progress: ProgressReporter | None = None,
    stopped: Callable[[], bool] | None = None,
) -> PredictionResult:
    """Start or resume a prediction from a resolved configuration.

    ``stopped`` says whether the job running it was asked to stop: requested again once
    its folder's earlier stop is withdrawn (it reads stops while it waits for a verified
    copy of APEX).

    Raises
    ------
    DirectoryLocked
        If another process is writing the same prediction.
    """
    assert config.prediction.output_dir is not None
    folder = config.prediction.output_dir
    # Whether this call made the folder comes from the lock's own mkdir: an exists() taken
    # before the lock could let a process that lost the race remove the winner's files.
    with exclusive(folder, "prediction") as new:
        try:
            return _predict(config, progress, stopped)
        except ConfigError:
            if new:
                _remove_if_only_lock(folder)
            raise


def _remove_if_only_lock(folder: Path) -> None:
    """Remove a refused prediction's new folder if it holds nothing but its lock.

    A prediction refused before anything was written left its folder and lock behind
    (release tester). Called while the lock is still held, so no other process can hold
    it yet; ``os.rmdir`` removes only an empty folder, so nothing else is ever deleted.
    The folder is new, so a model-error forecast state in it (``forecast_state-*.npz``,
    written before the refusal) is this call's own and goes too.
    """
    try:
        names = {p.name for p in folder.iterdir()}
        states = {n for n in names if n.startswith(FORECAST_STATE_PREFIX) and n.endswith(".npz")}
        if names - states <= {LOCK_FILE}:
            for name in sorted(states):
                (folder / name).unlink()
            (folder / LOCK_FILE).unlink(missing_ok=True)
            os.rmdir(folder)
    except OSError:
        pass  # something else is there now: keep the folder


#: Names a deck template is given besides the member's blocks (simulation/deck.py
#: render_deck: ``member_id``, ``case_dir``; the deck simulators' fidelity ``level`` and
#: ``coarse``), with which a control of that name would clash unseen.
RESERVED_CONTROL_NAMES = frozenset({"member_id", "case_dir", "level", "coarse"})


def control_blocks(
    controls: PredictionControlsSection,
    state: PropertyMap,
    layout: StateLayout,
    template_values: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Return a control scenario's blocks ``{name: (n,)}``, after its parameterizations.

    decision 6: what every member is given besides its state. Each
    parameterization maps the controls only, as an optimization's (``control_schedule``).

    Raises
    ------
    ConfigError
        If a parameterization is unknown or invalid, or a control (or what its
        parameterizations make of it) is named as one of the run's state variables or
        properties, which it would hide, or as a name the deck template is given otherwise
        (``member_id``, ``case_dir``, the simulator's ``template_values``).
    """
    variables = [
        StateVariable(
            v.name, v.size,
            -np.inf if v.lower is None else float(v.lower),
            np.inf if v.upper is None else float(v.upper),
        )
        for v in controls.variables
    ]  # fmt: skip
    built = []
    for section in controls.parameterizations:
        try:
            built.append(parameterizations.create(section.name, **section.options))
        except (PluginError, TypeError, ValueError) as error:
            raise ConfigError(f"[[controls.parameterizations]] {section.name}: {error}") from error
    try:
        mapped = PropertyMap(built, StateLayout(variables)).member(
            {v.name: v.array() for v in controls.variables}
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[controls]: {error}") from error
    clashing = sorted(set(mapped) & (set(layout.names) | set(state.names)))
    if clashing:
        raise ConfigError(
            f"[controls] names {clashing}, which the run's state or its parameterizations "
            "already give each member: name the controls otherwise"
        )
    reserved = sorted(set(mapped) & (RESERVED_CONTROL_NAMES | set(template_values)))
    if reserved:
        raise ConfigError(
            f"[controls] names {reserved}, which the deck template is given otherwise "
            "(member_id, case_dir, level, coarse, the simulator's template_values): name the "
            "controls otherwise"
        )
    return {name: np.ascontiguousarray(np.ravel(v), dtype=np.float64) for name, v in mapped.items()}


def _scenario(
    config: PredictionConfig,
    base: object,
    simulator_config: PluginSection,
    property_map: PropertyMap,
    layout: StateLayout,
) -> dict[str, np.ndarray]:
    """Return the prediction's control scenario's blocks, refusing one that cannot run.

    ``{}`` without ``[controls]``. ``base`` is the prediction's simulator before any
    wrapper. Used at the start (:func:`_predict`) and before submission
    (:func:`check_prediction`), so that ``apex validate`` refuses what the start would.

    Raises
    ------
    ConfigError
        As :func:`control_blocks` and :func:`_check_controls_read`, and on a batch
        simulator that writes only its declared parameters (ERT's).
    """
    if config.controls is None:
        return {}
    controls = control_blocks(
        config.controls, property_map, layout,
        dict(simulator_config.options.get("template_values") or {}),
    )  # fmt: skip
    if controls and getattr(base, "ert_batches", False):
        # ERT's batches write only the parameters they declare (simulation/ert.py
        # write_parameters): the scenario would never reach the forward model (the review).
        raise ConfigError(
            f"a prediction with [controls] cannot run on the {simulator_config.name} "
            "simulator: its batches write only their declared parameters, not the controls"
        )
    if controls and config.surrogate is None:
        _check_controls_read(base, simulator_config.name, controls)
    return controls


def _check_controls_read(simulator: object, name: str, controls: Mapping[str, np.ndarray]) -> None:
    """Refuse a control scenario the simulator would ignore (release tester H35).

    A simulator that declares the state names it reads (``state_names_read``) must
    read every control; one that cannot tell is warned about.

    Raises
    ------
    ConfigError
        If the simulator reads none of some controls' names.
    """
    read = getattr(simulator, "state_names_read", None)
    names = read() if callable(read) else None
    if names is None:
        logger.warning(
            "the %s simulator does not say which named values it reads: the control "
            "scenario (%s) may not reach its forecast",
            name, ", ".join(sorted(controls)),
            extra={"event": "controls_unchecked", "simulator": name},
        )  # fmt: skip
        return
    unread = sorted(set(controls) - names)
    if unread:
        raise ConfigError(
            f"[controls] gives {unread}, which the {name} simulator does not read: the "
            "forecast would be the same without them. A deck template reads a control by "
            "naming it (e.g. ${" + unread[0] + "[0]}); other simulators read only their "
            "own state names"
        )


def _predict(
    config: PredictionConfig,
    progress: ProgressReporter | None,
    stopped: Callable[[], bool] | None = None,
) -> PredictionResult:
    section = config.prediction
    run_dir = RunDirectory(section.run_dir)
    run_config = _with_forecast_options(config, composition.snapshot_config(run_dir))
    layout = composition.build_layout(run_config)
    try:
        results = RunResults(section.run_dir, layout)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{section.run_dir} has no results yet: {error}") from error
    simulator_config = simulator_section(config, run_config)
    execution = config.execution or run_config.execution
    model_error = _forecasting_model_error(config, run_config, simulator_config, layout)
    _refuse_for_model(config, run_config, model_error)
    forecast: dict[str, object] = {}
    if model_error is not None and extrapolates(model_error):
        # The model's parameters past the history, fitted once here
        # and read from a file wherever the model is built (workers, remote hosts).
        run_config, model_error, forecast = _prepare_extrapolation(
            config, run_config, simulator_config, layout, results, model_error
        )
    state = forecast.get("forecast_state")
    state_file = str(state["file"]) if isinstance(state, dict) else None
    fitted = isinstance(state, dict) and state.get("fitted_by_this_start") is True
    try:
        try:
            base = simulators.create(simulator_config.name, **simulator_config.options)
            # the run's parameterizations map each member's state, as in the run.
            property_map = PropertyMap(composition.build_parameterizations(run_config), layout)
            simulator = parameterized(base, property_map)
            if model_error is not None:
                # The run's forecasts were g + r, and so are the
                # prediction's (a model-error run has no parameterizations: refused).
                simulator = ModelErrorSimulator(simulator, model_error)
            executor = executors.create(execution.name, **execution.options)
            composition.check_executor(executor, simulator)
            # an executor running members whole on other hosts rebuilds the
            # prediction's simulator there: the run's configuration with the prediction's
            # simulator, and nothing that wraps the run's (no geophysics, model error or
            # transforms: a prediction forecasts series).
            assert section.output_dir is not None
            composition.prepare_remote(
                executor, simulator,
                remote_config(run_config, simulator_config, section.output_dir,
                              model_error=model_error is not None),
            )  # fmt: skip
            scores: dict[str, ForecastScore] = {
                name: forecast_scores.create(name) for name in section.scores
            }
        except (TypeError, ValueError, PluginError) as error:
            raise ConfigError(f"invalid prediction plugin configuration: {error}") from error
        ensembles, truth = _ensembles(config, run_config, results)
        assert section.output_dir is not None
        times = section.outputs.times
        request = SeriesRequest(
            tuple(section.outputs.vectors), None if times == "report" else tuple(times)
        )
        controls = _scenario(config, base, simulator_config, property_map, layout)
        surrogate = None
        if config.surrogate is not None:
            # The members (not the truth) by a fitted model, its keys the axis.
            from apex.services import surrogates

            try:
                surrogate, request = surrogates.prediction_surrogate(
                    config.surrogate,
                    remote_config(run_config, simulator_config, section.output_dir),
                    base, executor, property_map, request, execution.name,
                    layout=layout, controls=controls,
                )  # fmt: skip
            except (TypeError, ValueError, PluginError) as error:
                raise ConfigError(f"invalid prediction [surrogate]: {error}") from error
        directory = PredictionDirectory(section.output_dir)
        directory.open(
            forecast_state=state_file,
            config_text=config.model_dump_json(indent=2),
            fingerprint=_fingerprint(
                config,
                run_dir,
                simulator_config,
                "" if surrogate is None else surrogate.identity,
                None if model_error is None else _model_error_files(run_config),
            ),
        )
    except Exception:
        # Refused or failed between the fit and the folder's opening (another prediction's
        # folder, an invalid [surrogate], an unknown simulator, ...): a state file this start
        # fitted is its own (derived, named by its inputs) and is not left behind.
        # Safe: _predict runs only inside predict_config's lock on this folder (the fit
        # included), so no other start can be using a file this one found missing.
        if state_file is not None and fitted and section.output_dir is not None:
            (section.output_dir / state_file).unlink(missing_ok=True)
        raise
    if stopped is not None and stopped():
        (directory.path / "stop_requested").write_text("stop at the next checkpoint\n")
    composition.configure_logging(directory.path)
    directory.write_json(
        "provenance",
        provenance.collect(
            {
                "run_config": run_dir.path / "config.json",
                **_input_files(
                    simulator_config, None if model_error is None else run_config.model_error
                ),
            },
            simulator={
                "name": simulator_config.name,
                "version": _simulator_version(simulator),
            },
        )
        | _execution(executor, simulator)
        | (
            {}
            if model_error is None or run_config.model_error is None
            else {
                "model_error": {
                    "name": run_config.model_error.name,
                    "truth_is_simulator": section.truth_is_simulator,
                }
                | forecast
            }
        )
        # The scenario every member was given, after its parameterizations (release
        # tester H35: it was only in config.json).
        | ({"controls": {k: v.tolist() for k, v in controls.items()}} if controls else {})
        | (
            {}
            if surrogate is None or config.surrogate is None
            else {
                "surrogate": {
                    "model": str(config.surrogate.model_file),
                    "identity": surrogate.identity,
                    "early_until": surrogate.early_until,
                    "outside": surrogate.outside,
                }
            }
        ),
    )
    observations = read_observation_section(run_config.observations)
    try:
        workflow = PredictionWorkflow(
            layout=layout,
            simulator=simulator,
            executor=executor,
            directory=directory,
            request=request,
            observation_times=[key.time for key in observations.keys],
            scores=scores,
            include_mean_model=section.include_mean_model,
            check_history=section.check_history,
            percentile_convention=section.percentile_convention,
            min_success_fraction=run_config.failures.min_success_fraction,
            progress=progress,
            surrogate=surrogate,
            controls=controls,
        )
    except ValueError as error:
        raise ConfigError(f"invalid prediction: {error}") from error
    return workflow.run(ensembles, truth)


def _forecasting_model_error(
    config: PredictionConfig,
    run: HistoryMatchConfig,
    simulator: PluginSection,
    layout: StateLayout,
) -> ModelErrorModel | None:
    """Return the run's model-error model for the prediction's members, if the run has one.

    Built as composition builds it, against the prediction's simulator
    (its ``check_simulator`` and the deck's SUMMARY checked), fitted to the run's
    observations; only a model that declares ``forecasts`` (its residuals apply past the
    history) is taken.

    Raises
    ------
    ConfigError
        If the model does not forecast (``pg_lstm``: an output layer per history report
        time), or the prediction also asks for ``[surrogate]``.
    """
    section = run.model_error
    if section is None:
        return None
    # The class says whether it forecasts: a model that does not is refused before it is
    # built (and its files read), with the reason (the review).
    try:
        forecasts = getattr(model_error_models.factory(section.name), "forecasts", False)
    except PluginError as error:
        raise ConfigError(f"[model_error]: {error}") from error
    if forecasts is not True:
        raise ConfigError(
            f"{config.prediction.run_dir} ran with a model-error model ([model_error] "
            f"{section.name}) that does not forecast (its residuals apply at the history's "
            "report times only, e.g. one output layer per time); predictions beyond its "
            "history need a model that declares forecasts"
        )
    observations = read_observation_section(run.observations)
    as_run = run.model_copy(update={"simulator": simulator})
    try:
        model = composition.build_model_error(as_run, layout, observations)
    except (ValueError, TypeError) as error:
        # The model's constructor refusing its options: since a prediction may set its
        # forecast options, these are a client's (the coder's re-review), reported as a
        # configuration error (validation invalid, the new folder removed).
        raise ConfigError(
            f"[model_error] {section.name} cannot be built with its options (the run's, and "
            f"a prediction's [model_error] options): {type(error).__name__}: {error}"
        ) from error
    assert model is not None
    if config.surrogate is not None:
        raise ConfigError(
            "a prediction of a run with a model-error model cannot use [surrogate]: the "
            "surrogate would forecast g alone, the run's forecasts were g + r"
        )
    composition.check_model_error_summary(as_run, model)
    return model


def check_prediction(config: PredictionConfig) -> None:
    """Refuse, before submission, what a prediction cannot run and needs no forecast to know.

    ``apex validate``, the tools' validation and :meth:`ApexService.submit_prediction` call
    it; :func:`_predict` applies the same refusals. A run that does not exist yet (a
    study's prediction waits for it) is not checked here. The model-error model, if any,
    is built (its files read, no simulation) to know whether it extrapolates, and with a
    control scenario the simulator is built (nothing simulated) to know whether it reads it.

    Raises
    ------
    ConfigError
        As :func:`_with_forecast_options`, :func:`_forecasting_model_error`,
        :func:`_refuse_for_model` and :func:`_scenario`.
    """
    if not (Path(config.prediction.run_dir) / "config.json").is_file():
        return
    run = _with_forecast_options(
        config, composition.snapshot_config(RunDirectory(config.prediction.run_dir))
    )
    layout = composition.build_layout(run)
    simulator_config = simulator_section(config, run)
    try:
        model = _forecasting_model_error(config, run, simulator_config, layout)
    except OSError as error:
        raise ConfigError(f"cannot build the run's model-error model: {error}") from error
    _refuse_for_model(config, run, model)
    if config.controls is not None:
        try:
            base = simulators.create(simulator_config.name, **simulator_config.options)
            property_map = PropertyMap(composition.build_parameterizations(run), layout)
        except (OSError, TypeError, ValueError, PluginError) as error:
            raise ConfigError(f"invalid prediction plugin configuration: {error}") from error
        _scenario(config, base, simulator_config, property_map, layout)


def _refuse_for_model(
    config: PredictionConfig, run: HistoryMatchConfig, model: ModelErrorModel | None
) -> None:
    """Refuse what the run's model-error model, or its absence, makes impossible.

    Raises
    ------
    ConfigError
        The truth or ``MEC:`` vectors of a model-error run (:func:`_refuse_model_error_
        requests`); with a model that extrapolates its parameters,
        the truth (its parameters are not stored: an auxiliary variable's are zero, which
        no posterior sequence extrapolates) and ``[[data_transforms]]`` (they replace the
        assimilated keys the fit reads); without one, a ``seed`` or ``[model_error]
        options`` (they would change nothing).
    """
    section = config.prediction
    if model is not None:
        _refuse_model_error_requests(config)
    if model is not None and extrapolates(model):
        if "truth" in section.ensembles:
            raise ConfigError(
                "a prediction of a run whose model-error model extrapolates its parameters "
                "past the history cannot forecast 'truth': the truth has no stored "
                "model-error parameters (an auxiliary variable), so there is no sequence to "
                "extrapolate"
            )
        if run.data_transforms:
            raise ConfigError(
                "a model-error model that extrapolates its parameters needs the keys the run "
                "assimilated, which [[data_transforms]] replace; not supported"
            )
        return
    if section.seed is not None:
        raise ConfigError(
            "[prediction] seed is given, but nothing in this prediction draws (only a "
            "model-error model that extrapolates its parameters past the history does, ADR "
            "0058 revision 4); remove it"
        )
    if config.model_error is not None:
        raise ConfigError(
            "[model_error] options are given, but the run's model-error model does not "
            "extrapolate its parameters past the history in this configuration, so they "
            "would change nothing; remove them"
        )


def _with_forecast_options(config: PredictionConfig, run: HistoryMatchConfig) -> HistoryMatchConfig:
    """Return the run's configuration with the prediction's ``[model_error] options``.

    Only the options the run's model declares in ``forecast_options``
    (how its parameters past the history are built). The merged configuration is what the
    model, the fingerprint, the provenance and remote hosts see.

    Raises
    ------
    ConfigError
        If the run has no model-error model, or an option is not one of its forecast
        options.
    """
    if config.model_error is None:
        return run
    if run.model_error is None:
        raise ConfigError(
            f"[model_error] options are given, but {config.prediction.run_dir} ran without a "
            "model-error model"
        )
    try:
        factory = model_error_models.factory(run.model_error.name)
    except PluginError as error:
        raise ConfigError(f"[model_error]: {error}") from error
    allowed = frozenset(getattr(factory, "forecast_options", frozenset()))
    other = sorted(set(config.model_error.options) - allowed)
    if other:
        raise ConfigError(
            f"a prediction's [model_error] options may set only the model's forecast options "
            f"({', '.join(sorted(allowed)) or 'none'} for {run.model_error.name}), not "
            f"{', '.join(other)}: the others shape the history the prediction reproduces"
        )
    options = {**run.model_error.options, **config.model_error.options}
    return run.model_copy(
        update={"model_error": run.model_error.model_copy(update={"options": options})}
    )


def _prediction_seed(config: PredictionConfig, run: HistoryMatchConfig) -> np.random.SeedSequence:
    """Return the prediction's seed sequence.

    ``[prediction] seed`` if given; else the run's seed with a spawn key from the
    prediction's name (the first 31 bits of its SHA-256), so two predictions of one run
    draw differently and a resume draws the same.
    """
    section = config.prediction
    if section.seed is not None:
        return np.random.SeedSequence(section.seed)
    name = int.from_bytes(hashlib.sha256(section.name.encode()).digest()[:4], "big") >> 1
    return np.random.SeedSequence(run.run.seed, spawn_key=(name,))


def _prepare_extrapolation(
    config: PredictionConfig,
    run: HistoryMatchConfig,
    simulator: PluginSection,
    layout: StateLayout,
    results: RunResults,
    model: ModelErrorModel,
) -> tuple[HistoryMatchConfig, ModelErrorModel, dict[str, object]]:
    """Fit an extrapolating model's forecast state; return the run as predicted with it.

    The model's :meth:`forecast_state` from the run's prior (iterate
    0) and posterior (the CV-selected or last iterate) of its state variable, their common
    members paired by id, and the keys the run assimilated; written with the prediction's
    seed to a file in the prediction's folder named by a digest of those inputs (so a
    resume, or a configuration replacing an unfinished one, reuses it instead of fitting
    again: a refit in another environment may round differently); the run's
    configuration with that file as the model's option ``forecast_state_file`` (so the
    fingerprint, the provenance and remote hosts have it), and the model rebuilt from it.
    Every ensemble's members then extrapolate from their own parameters with what was
    fitted on the prior-to-posterior update.

    Raises
    ------
    ConfigError
        If the prediction asks for the truth (its parameters are not stored: an auxiliary
        variable's are zero, which no posterior sequence extrapolates), the run transforms
        its data, the two iterates share fewer than 2 members, or the model refuses them.
    """
    section = config.prediction
    assert run.model_error is not None
    assert section.output_dir is not None
    assert isinstance(model, ExtrapolatingModel)
    variable = model.state_variable
    prior = results.iterate(0)
    posterior = results.iterate(results.posterior_iteration)
    # Members that failed are dropped from later iterates: pair the two by their ids.
    common, in_prior, in_posterior = np.intersect1d(
        prior.member_ids, posterior.member_ids, return_indices=True
    )
    if common.size < 2:
        raise ConfigError(
            f"the run's prior and posterior share {common.size} member(s); the "
            "extrapolation fits on at least 2"
        )
    before = np.ascontiguousarray(np.asarray(prior.state[variable])[:, in_prior])
    after = np.ascontiguousarray(np.asarray(posterior.state[variable])[:, in_posterior])
    keys = list(composition.build_run(run, surrogate=False).workflow.observations.keys)
    seed = _prediction_seed(config, run)
    # The file is named by what the fit reads: the same inputs find the same file.
    inputs = hashlib.sha256()
    inputs.update(run.model_error.model_dump_json().encode())
    inputs.update(json.dumps([str(key) for key in keys]).encode())
    for array in (common, before, after):
        inputs.update(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes())
    inputs.update(str(seed.entropy).encode() + repr(seed.spawn_key).encode())
    path = section.output_dir / f"forecast_state-{inputs.hexdigest()[:16]}.npz"
    saved = section.output_dir / "config.json"
    if (
        not path.is_file()
        and PredictionDirectory(section.output_dir).finished_members()
        and saved.is_file()
        and saved.read_text(encoding="utf-8") != config.model_dump_json(indent=2)
    ):
        # The folder's open() would refuse it: refused before fitting, so no new state
        # file lands in another prediction's folder (the coder's review). Only a changed
        # configuration is caught here; a changed run file with the same configuration
        # is refused by open()'s fingerprint, after the fit.
        raise ConfigError(
            f"{section.output_dir} holds a prediction with another configuration and "
            "finished members; choose another name or output_dir, or delete it to start "
            f"again with this configuration: rm -r '{section.output_dir}'"
        )
    fitted = not path.is_file()
    try:
        if fitted:
            write_forecast_state(path, model.forecast_state(before, after, keys), seed)
    except ValueError as error:
        raise ConfigError(f"[model_error] {run.model_error.name}: {error}") from error
    options = {**run.model_error.options, FORECAST_STATE_OPTION: str(path)}
    predicted = run.model_copy(
        update={"model_error": run.model_error.model_copy(update={"options": options})}
    )
    rebuilt = _forecasting_model_error(config, predicted, simulator, layout)
    assert rebuilt is not None
    record: dict[str, object] = {
        "forecast_state": {"file": path.name, "sha256": provenance.file_hash(path),
                           "posterior_iteration": int(results.posterior_iteration),
                           "members": int(common.size), "fitted_by_this_start": fitted},
        "seed": {"entropy": str(seed.entropy), "spawn_key": list(seed.spawn_key),
                 "given": section.seed is not None},
    }  # fmt: skip
    logger.info(
        "the model-error model's parameters past the history are extrapolated",
        extra={"event": "prediction_model_error_extrapolation", "file": path.name},
    )
    return predicted, rebuilt, record


def _model_error_files(run: HistoryMatchConfig) -> dict[str, Path]:
    """Return a model-error model's files and the observations it is fitted to."""
    assert run.model_error is not None
    found = {
        f"model_error.{key}": Path(value)
        for key, value in run.model_error.options.items()
        if key.endswith(composition.PATH_OPTION_SUFFIXES) and isinstance(value, str)
    }
    if run.observations.file is not None:
        found["observations.file"] = Path(run.observations.file)
    return found


def _refuse_model_error_requests(config: PredictionConfig) -> None:
    """Refuse, before anything runs, what a model-error run's prediction cannot answer.

    The truth (unless ``truth_is_simulator``: a model error means the data came from
    another model, which the simulator does not reproduce, and scores against it would be
    against the wrong reference), and the residuals' own vectors (``MEC:``, not summary
    vectors).

    Raises
    ------
    ConfigError
        Saying why.
    """
    section = config.prediction
    if "truth" in section.ensembles and not section.truth_is_simulator:
        raise ConfigError(
            "a prediction of a run with a model-error model refuses 'truth' by default: the "
            "model corrects the simulator because the data came from another model (the "
            "real system, or a finer one), which this simulator's forecast of the truth's "
            "parameters does not reproduce, so scores against it would use the wrong "
            "reference; set [prediction] truth_is_simulator = true only if the observations "
            "came from this very simulator, without model error"
        )
    residual = [v for v in section.outputs.vectors if v.startswith(RESIDUAL_PREFIX)]
    if residual:
        raise ConfigError(
            f"a prediction cannot request the model-error residuals ({', '.join(residual)}): "
            "they are not summary vectors; the run's iterates hold them "
            "(auxiliary_predicted)"
        )
    if section.truth_is_simulator and "truth" in section.ensembles:
        logger.info(
            "the truth forecast by the simulator alone",
            extra={"event": "prediction_truth_is_simulator"},
        )


def _simulator_version(simulator: object) -> str | None:
    version = getattr(simulator, "version", None)
    return None if version is None else str(version())


def load_prediction(path: Path) -> tuple[PredictionConfig, PredictionResult]:
    """Load a finished prediction: its configuration and its result.

    Raises
    ------
    ConfigError
        If ``path`` is not a finished prediction.
    """
    directory = PredictionDirectory(path)
    if not PredictionDirectory.holds_prediction(path) or not directory.read_state().get("complete"):
        raise ConfigError(f"{path} is not a finished prediction")
    config = PredictionConfig.model_validate_json((directory.path / "config.json").read_text())
    names = [name for name in config.prediction.ensembles if name != "truth"]
    forecasts = {name: directory.load_forecast(name) for name in names}
    scores_path = directory.path / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    run_config = composition.snapshot_config(RunDirectory(config.prediction.run_dir))
    observations = read_observation_section(run_config.observations)
    some = next(iter(forecasts.values()))
    first = first_prediction_time(some.times, [k.time for k in observations.keys])
    return config, PredictionResult(forecasts, directory.load_truth(), first, scores)


def write_prediction_figures(
    path: Path, output_dir: Path | None = None, *, font_size: float = 12.0
) -> list[Path]:
    """Write a finished prediction's figures (default ``<path>/report``).

    Raises
    ------
    ConfigError
        If ``path`` is not a finished prediction.
    """
    from apex.reporting.prediction_plots import (
        distribution_figures,
        forecast_figures,
        score_figures,
    )

    config, result = load_prediction(path)
    output = output_dir or Path(path) / "report"
    run_config = composition.snapshot_config(RunDirectory(config.prediction.run_dir))
    observations = read_observation_section(run_config.observations)
    observed = [
        (key.time, key.datatype, float(value), float(variance))
        for key, value, variance in zip(
            observations.keys, observations.values, observations.variances, strict=True
        )
        if key.index is None
    ]
    convention = config.prediction.percentile_convention
    figures = forecast_figures(
        result.forecasts,
        output,
        truth=result.truth,
        first_prediction_time=result.first_prediction_time,
        observations=observed,
        convention=convention,
        font_size=font_size,
    )
    figures += distribution_figures(
        result.forecasts, output, truth=result.truth, font_size=font_size
    )
    figures += score_figures(result.scores, output, font_size=font_size)
    return figures
