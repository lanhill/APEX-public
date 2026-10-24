"""Composition of predictions: configuration, ensembles, workflow, reports.

``apex predict PREDICTION.toml`` and ``apex report PRED_DIR`` call these functions; the
case-study runner and notebooks may too.
"""

import hashlib
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

import apex.algorithms.metrics  # noqa: F401  (registers the forecast scores)
from apex.core.errors import ApexError, ConfigError, PluginError, validation_problems
from apex.core.forecast import ForecastScore
from apex.core.plugins import executors, forecast_scores, simulators
from apex.core.progress import ProgressReporter
from apex.core.simulation import SeriesRequest
from apex.domain.config import HistoryMatchConfig, PluginSection
from apex.domain.observations import ObservationSet
from apex.domain.prediction import PredictionConfig, PredictionResult
from apex.services import composition
from apex.storage import provenance
from apex.storage.inputs import read_observation_section
from apex.storage.locks import exclusive
from apex.storage.prediction_directory import PredictionDirectory
from apex.storage.run_directory import RunDirectory
from apex.storage.run_results import RunResults
from apex.workflows.parameterization import PropertyMap, parameterized
from apex.workflows.prediction import (
    EnsembleInput,
    PredictionWorkflow,
    StoredForecast,
    first_prediction_time,
)


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
    run: HistoryMatchConfig, simulator: PluginSection, output_dir: Path
) -> HistoryMatchConfig:
    """Return the configuration a remote worker builds the prediction's simulator from.

    The run's, with the prediction's simulator and output folder, and nothing that wraps
    the run's simulator (geophysics, model error, transforms: a prediction forecasts
    series). A multilevel method's levels go too: every member of a prediction runs on
    the fine model here, so it must on the hosts.
    """
    options = {k: v for k, v in run.method.options.items() if k != "levels"}
    return run.model_copy(update={
        "simulator": simulator, "geophysics": None, "model_error": None,
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


def _fingerprint(config: PredictionConfig, run_dir: RunDirectory, simulator: PluginSection) -> str:
    # The configuration, the run it reads and the content of the simulator's files (e.g.
    # the deck template): a changed deck must not resume from members of the old one.
    digest = hashlib.sha256()
    digest.update(config.model_dump_json().encode())
    digest.update(run_dir.config_text().encode())
    for name, path in sorted(_input_files(simulator).items()):
        digest.update(name.encode())
        digest.update(provenance.file_hash(path).encode() if path.is_file() else b"missing")
    return digest.hexdigest()


def _input_files(simulator: PluginSection) -> dict[str, Path]:
    return {
        f"simulator.{key}": Path(value)
        for key, value in simulator.options.items()
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
    predicted = composition.all_roles(iterate, role_rows, observations.n_data)
    scalar = [i for i, key in enumerate(observations.keys) if key.index is None]
    keep = np.array([int(m) not in coarse for m in iterate.member_ids], dtype=bool)
    return StoredForecast(
        tuple(observations.keys[i] for i in scalar),
        predicted[scalar][:, keep],
        iterate.member_ids[keep],
    )


def _ensembles(
    config: PredictionConfig, run_config: HistoryMatchConfig, results: RunResults
) -> tuple[list[EnsembleInput], np.ndarray | None]:
    workflow = composition.build_run(run_config).workflow
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
    with exclusive(config.prediction.output_dir, "prediction"):
        return _predict(config, progress, stopped)


def _predict(
    config: PredictionConfig,
    progress: ProgressReporter | None,
    stopped: Callable[[], bool] | None = None,
) -> PredictionResult:
    section = config.prediction
    run_dir = RunDirectory(section.run_dir)
    run_config = composition.snapshot_config(run_dir)
    if run_config.model_error is not None:
        # The networks have one output layer per history report time, and the
        # run's forecasts are g + r, so a forecast of g alone would not reproduce them.
        raise ConfigError(
            f"{section.run_dir} ran with a model-error model ([model_error] "
            f"{run_config.model_error.name}); predictions beyond its history are not supported"
        )
    layout = composition.build_layout(run_config)
    try:
        results = RunResults(section.run_dir, layout)
    except (OSError, ValueError) as error:
        raise ConfigError(f"{section.run_dir} has no results yet: {error}") from error
    simulator_config = simulator_section(config, run_config)
    execution = config.execution or run_config.execution
    try:
        simulator = simulators.create(simulator_config.name, **simulator_config.options)
        # the run's parameterizations map each member's state, as in the run.
        simulator = parameterized(
            simulator, PropertyMap(composition.build_parameterizations(run_config), layout)
        )
        executor = executors.create(execution.name, **execution.options)
        composition.check_executor(executor, simulator)
        # an executor running members whole on other hosts rebuilds the
        # prediction's simulator there: the run's configuration with the prediction's
        # simulator, and nothing that wraps the run's (no geophysics, model error or
        # transforms: a prediction forecasts series).
        assert section.output_dir is not None
        composition.prepare_remote(
            executor, simulator, remote_config(run_config, simulator_config, section.output_dir)
        )
        scores: dict[str, ForecastScore] = {
            name: forecast_scores.create(name) for name in section.scores
        }
    except (TypeError, ValueError, PluginError) as error:
        raise ConfigError(f"invalid prediction plugin configuration: {error}") from error
    ensembles, truth = _ensembles(config, run_config, results)
    assert section.output_dir is not None
    directory = PredictionDirectory(section.output_dir)
    directory.open(
        config_text=config.model_dump_json(indent=2),
        fingerprint=_fingerprint(config, run_dir, simulator_config),
    )
    if stopped is not None and stopped():
        (directory.path / "stop_requested").write_text("stop at the next checkpoint\n")
    composition.configure_logging(directory.path)
    directory.write_json(
        "provenance",
        provenance.collect(
            {"run_config": run_dir.path / "config.json", **_input_files(simulator_config)},
            simulator={
                "name": simulator_config.name,
                "version": _simulator_version(simulator),
            },
        )
        | _execution(executor, simulator),
    )
    times = section.outputs.times
    request = SeriesRequest(
        tuple(section.outputs.vectors), None if times == "report" else tuple(times)
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
        )
    except ValueError as error:
        raise ConfigError(f"invalid prediction: {error}") from error
    return workflow.run(ensembles, truth)


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
