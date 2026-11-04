"""Summaries of runs across case studies: performance metrics at every step.

:func:`summarize` finds the run directories under the given paths and summarises each
with the first registered summarizer (:data:`apex.core.plugins.summaries`) that applies.
History matching is summarised one step per accepted iterate (the prior is step 0), a
prediction one step per ensemble forecast, an optimization one step
per row of its history (the start, each accepted step, each EPF update), a surrogate build
its training and its held-out runs, with the gate.

Every workflow that writes a result directory has a summarizer (the user's rule,
2026-10-10; future ones such as risk assessment too): register one here, and
``tests/workflows/test_summaries.py`` checks that each such workflow has one.
"""

import itertools
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from apex.algorithms.metrics.performance import ModelChange
from apex.core.errors import ApexError, ConfigError
from apex.core.metrics import MetricContext
from apex.core.plugins import summaries
from apex.core.summaries import Summarizer
from apex.domain.config import HistoryMatchConfig
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.summary import MetricStatistics, RunSummary, StepSummary
from apex.services import composition, optimization, prediction, surrogates
from apex.storage.prediction_directory import PredictionDirectory
from apex.storage.priors import read_array
from apex.storage.run_directory import RunDirectory
from apex.storage.run_results import RunResults
from apex.storage.surrogate_directory import SurrogateDirectory

logger = logging.getLogger(__name__)


@summaries.register("history_matching")
class HistoryMatchSummarizer:
    """Summarise a history-matching run: its accepted iterates and their metrics.

    Every metric the run recorded is summarised (mean, std, median, P10, P90 over the
    members), plus the data mismatch the method minimises (``data_mismatch``). A run that did not
    record ``model_change`` gets it computed from its saved iterates (with the ``rmse``
    metric's ``actnum``, if it has one), and a note says so.

    A run with a surrogate labels each step with its forecasts' source
    (``"3 (surrogate)"``), and adds the simulator's forecast of the posterior as the metrics
    ``simulator:<metric>`` at the result step, with a note comparing the two.
    """

    def applies(self, path: Path) -> bool:
        """Whether ``path`` holds a history-matching run (``config.json``, ``history.json``)."""
        return (path / "config.json").is_file() and (path / "history.json").is_file()

    def summarize(self, path: Path) -> RunSummary:
        """Summarise the run in ``path``.

        Raises
        ------
        ConfigError
            If it has no configuration snapshot or no scored forecast.
        """
        config = composition.snapshot_config(RunDirectory(path))
        results = composition.load_results(path)
        history = results.history
        surrogate = any(r.forecast_source is not None for r in history)
        steps: list[StepSummary] = []
        rejected = 0
        for record in history:
            if not record.accepted:
                rejected += 1
                continue
            label = f"{len(steps)} ({record.forecast_source or 'simulator'})" if surrogate else None
            steps.append(
                StepSummary(
                    record.iteration,
                    record.n_members,
                    rejected_before=rejected,
                    control=record.damping,
                    label=label,
                )
            )
            rejected = 0
        rows = [
            MetricStatistics(
                metric=str(r["metric"]),
                label=str(r["label"]),
                iteration=int(str(r["iteration"])),
                mean=float(str(r["mean"])),
                std=float(str(r["std"])),
                median=float(str(r["median"])),
                p10=float(str(r["p10"])),
                p90=float(str(r["p90"])),
            )
            for r in results.metric_summary()
        ]
        notes = []
        if "model_change" not in results.metric_labels():
            rows.extend(_model_change(results, config))
            notes.append("model_change computed from the saved iterates (not recorded by the run)")
        if surrogate:
            rows.extend(_simulator_posterior(results, notes))
        return RunSummary(
            workflow="history_matching",
            name=config.run.name,
            path=path,
            method=_method(config),
            status=_status(path, results),
            steps=tuple(steps),
            metrics=tuple(rows),
            selected_iteration=results.posterior_iteration,
            forecasts=len(history),
            notes=tuple(notes),
        )


def _simulator_posterior(results: RunResults, notes: list[str]) -> list[MetricStatistics]:
    """Return a surrogate run's posterior as the simulator forecast it.

    The metrics of ``full_<k>.npz`` as ``simulator:<metric>`` at the posterior's step, and
    a note comparing the surrogate's and the simulator's data mismatch.
    """
    iteration = results.posterior_iteration
    report = (results.surrogate or {}).get("posterior") or {}
    notes.append(
        "forecasts by a surrogate: steps are labelled with their source; the "
        "simulator's forecast of the posterior is the simulator:* metrics"
    )
    timing = _surrogate_timing((results.surrogate or {}).get("timing"))
    if timing:
        notes.append(timing)
    for rescore in (results.surrogate or {}).get("rescores", []):
        notes.append(
            f"the forecasts' source changed at iteration {rescore['iteration']}: accepted "
            f"iterate {rescore['accepted']} was forecast again by the {rescore['source']}, "
            "and cross validation (if any) compared the iterates from it on"
        )
    if report.get("forecast") == "simulator already":
        notes.append("the posterior was forecast by the simulator already (it took over)")
        return []
    path = results.run_dir.checkpoints / f"full_{iteration}.npz"
    if not path.exists():
        notes.append("the posterior has no forecast by the simulator yet (the run is not finished)")
        return []
    full = results.run_dir.load_arrays(f"full_{iteration}")
    rows = []
    for key, values in full.items():
        if not key.startswith("metric:"):
            continue
        _, metric, label = key.split(":", 2)
        v = np.asarray(values, dtype=np.float64)
        rows.append(MetricStatistics(
            f"simulator:{metric}", label, iteration, float(v.mean()),
            float(v.std(ddof=1)) if v.size > 1 else 0.0, float(np.median(v)),
            float(np.percentile(v, 10)), float(np.percentile(v, 90)),
        ))  # fmt: skip
    surrogate_side = report.get("surrogate") or {}
    simulator_side = report.get("simulator") or {}
    if "data_mismatch" in surrogate_side and "data_mismatch" in simulator_side:
        flag = " (beyond the surrogate's gate)" if report.get("beyond_gate") else ""
        notes.append(
            f"posterior data mismatch: simulator {simulator_side['data_mismatch']:.4g}, "
            f"surrogate {surrogate_side['data_mismatch']:.4g}{flag}"
        )
    return rows


def _surrogate_timing(timing: Mapping[str, Any] | None) -> str | None:
    """Return a surrogate run's cost as one note (decision 14), ``None`` without it."""
    if not timing:
        return None
    side = timing.get("surrogate") or {}
    if not side.get("members"):
        return None
    per_member = side.get("worker_seconds_per_member")
    if per_member is None:
        per_member = side.get("seconds_per_member")
    text = (
        f"surrogate forecasts: {side['members']} members in {side['seconds']:.3g} s "
        f"({per_member:.3g} s per member)"
    )
    full = timing.get("full_model")
    if full:
        text += (
            f"; full model: {full['mean_seconds']:.3g} s mean, {full['median_seconds']:.3g} s "
            f"median per run over the build's {full['runs']} runs"
        )
        if timing.get("speedup"):
            text += f"; speed-up {timing['speedup']:.3g}x"
        equivalent = timing.get("simulator_equivalent_worker_seconds")
        run = timing.get("run_worker_seconds")
        if equivalent is not None and run is not None:  # like with like only
            text += (
                f"; as full runs they would take {equivalent:.3g} worker-s, against "
                f"{run:.3g} worker-s for the run's forecasts, plus the build's "
                f"{full['total_seconds']:.3g} s once"
            )
    simulator = timing.get("simulator") or {}
    if simulator.get("members"):
        text += (
            f"; the run's own simulator runs: {simulator['members']} in "
            f"{simulator['seconds']:.3g} s"
        )
    return text


@summaries.register("prediction")
class PredictionSummarizer:
    """Summarise a prediction: one step per ensemble forecast.

    Per ensemble (the step's label, e.g. ``posterior``, ``prior``, ``iteration:3``): the
    members forecast, and per vector the forecast at the last time over the members
    (``final_value``: mean, std, median, P10, P90). With a truth forecast, the forecast
    scores over the prediction period (``coverage_p10_p90``, ``crps``, ...), per vector and
    averaged over the vectors (``all``), as single values. The result is the posterior.
    """

    def applies(self, path: Path) -> bool:
        """Whether ``path`` holds a finished prediction."""
        return PredictionDirectory.holds_prediction(path) and bool(
            PredictionDirectory(path).read_state().get("complete")
        )

    def summarize(self, path: Path) -> RunSummary:
        """Summarise the prediction in ``path``.

        Raises
        ------
        ConfigError
            If it is not a finished prediction.
        """
        config, result = prediction.load_prediction(path)
        steps, rows, notes = [], [], []
        for index, (name, forecast) in enumerate(result.forecasts.items()):
            steps.append(StepSummary(index, int(forecast.values.shape[-1]), label=name))
            if forecast.failed_member_ids.size:
                notes.append(f"{name}: {forecast.failed_member_ids.size} member(s) failed")
            for v, vector in enumerate(forecast.vectors):
                final = forecast.values[v, -1, :]
                rows.append(
                    MetricStatistics(
                        "final_value",
                        vector,
                        index,
                        float(final.mean()),
                        float(final.std(ddof=1)) if final.size > 1 else 0.0,
                        float(np.median(final)),
                        float(np.percentile(final, 10)),
                        float(np.percentile(final, 90)),
                    )
                )
            per_score: dict[str, list[float]] = {}
            for vector, scores in result.scores.get(name, {}).items():
                for score, value in scores.items():
                    rows.append(_single(score, vector, index, value))
                    per_score.setdefault(score, []).append(float(value))
            for score, values in per_score.items():
                rows.append(_single(score, "all", index, float(np.mean(values))))
        names = list(result.forecasts)
        section = config.prediction
        sources = PredictionDirectory(path).read_state().get("member_sources", {})
        if config.surrogate is not None:
            # Which members the surrogate forecast, the rest the simulator's.
            for name, by_member in sources.items():
                counts = {s: list(by_member.values()).count(s) for s in set(by_member.values())}
                notes.append(
                    f"{name}: {counts.get('surrogate', 0)} member(s) by the surrogate "
                    f"{config.surrogate.model_file}, "
                    f"{counts.get('simulator', 0)} by the simulator"
                    + ("; the truth by the simulator" if result.truth is not None else "")
                )
        return RunSummary(
            workflow="prediction",
            name=section.name,
            path=path,
            method=f"prediction of {section.run_dir.name}",
            status="complete",
            steps=tuple(steps),
            metrics=tuple(rows),
            selected_iteration=names.index("posterior") if "posterior" in names else None,
            forecasts=len(names),
            notes=tuple(notes),
        )


@summaries.register("optimization")
class OptimizationSummarizer:
    """Summarise an optimization run: one step per row of its history.

    Per step: the objective (the optimizer's value, with several objectives the
    representative's first; label ``all``) and, with several objectives, each
    objective by name, the hypervolume and the front's size, as single values. The step's
    members are the objective evaluations it took, its control the step size where the
    optimizer has one. The finished run's objective over the realizations (robust
    optimization) is a statistic at the last step (label ``realizations``). The result
    is the last step of a finished run.
    """

    def applies(self, path: Path) -> bool:
        """Whether ``path`` holds an optimization run (``config.json``, ``history.jsonl``)."""
        return optimization.holds_optimization(path)

    def summarize(self, path: Path) -> RunSummary:
        """Summarise the optimization in ``path``.

        Raises
        ------
        ConfigError
            If it is not an optimization run, or it has no history row yet.
        """
        try:
            summary = optimization.optimization_summary(path)
        except (OSError, ValueError) as error:  # result.json being written, say
            raise ConfigError(f"{path}: the optimization could not be read: {error}") from error
        if not summary.history:
            raise ConfigError(f"{path}: the optimization has not finished a step yet")
        steps, rows = [], []
        evaluated = 0
        for index, row in enumerate(summary.history):
            kind = str(row.get("kind", "step"))
            members = _evaluations(row)
            outer = int(row.get("epf_iteration") or 0)
            # With an external penalty the iterations restart each outer iteration.
            label = (f"step {row.get('iteration', index)}" if kind == "step" else kind) + (
                f" (epf {outer})" if summary.penalised and outer else "")  # fmt: skip
            step = row.get("step_size")
            control = float(step) if isinstance(step, int | float) else None
            steps.append(StepSummary(
                index, max(members - evaluated, 0), label=label,
                control=control if control is not None and math.isfinite(control) else None,
            ))  # fmt: skip
            evaluated = max(evaluated, members)
            for key, value in row.items():
                if (not isinstance(value, int | float) or isinstance(value, bool)
                        or not math.isfinite(value)):  # fmt: skip
                    continue
                if key == "objective":
                    rows.append(_single("objective", "all", index, float(value)))
                elif key.startswith("objective:"):
                    rows.append(_single("objective", key.split(":", 1)[1], index, float(value)))
                elif key in ("hypervolume", "front_size"):
                    rows.append(_single(key, "all", index, float(value)))
        last = len(steps) - 1
        notes = []
        if summary.result is not None:
            spread = np.asarray(summary.result.get("objective_per_realization") or [], float)
            if spread.size > 1:
                rows.append(MetricStatistics(
                    "objective", "realizations", last, float(spread.mean()),
                    float(spread.std(ddof=1)), float(np.median(spread)),
                    float(np.percentile(spread, 10)), float(np.percentile(spread, 90)),
                ))  # fmt: skip
            if summary.result.get("message"):
                notes.append(f"stopped: {summary.result['message']}")
            after = optimization.objective_evaluations(summary) - evaluated
            if after > 0:  # the last iteration's, which found no step to accept
                notes.append(f"{after} more objective evaluations after the last accepted step")
        if summary.objective_scaling is not None and summary.objective_scaling < 0:
            notes.append("the objective is minimised: a scaled, negated quantity (e.g. NPV)")
        if summary.penalised:
            notes.append("the objective includes the external penalty (epf)")
        if summary.failed_evaluations:
            notes.append(f"{summary.failed_evaluations} evaluation(s) failed")
        status = {"finished": "complete", "stopped": "stopped",
                  "running": "running or interrupted", "new": "running or interrupted"}  # fmt: skip
        return RunSummary(
            workflow="optimization",
            name=summary.name,
            path=path,
            method=f"{summary.optimizer} on {summary.objective}",
            status=status[summary.state],
            steps=tuple(steps),
            metrics=tuple(rows),
            selected_iteration=last if summary.result is not None else None,
            forecasts=max(evaluated, 0),
            notes=tuple(notes),
        )


@summaries.register("surrogate")
class SurrogateSummarizer:
    """Summarise a surrogate build: its runs, its gate, its cost.

    Two steps: the training runs (step 0) and the held-out runs (step 1). As single values:
    at step 1, the gate's error and tolerance per datatype (``gate_error``,
    ``gate_tolerance``, label the datatype) and its verdict (``gate_passed``, 1 or 0); at
    step 0, the failed runs, the domain's size (``domain_coordinates``,
    ``domain_points``), the fit's time, one prediction's time, the simulator's median run
    and their ratio (``speedup``). With checks (``apex surrogate check``), the latest
    complete one of the model at step 1: ``check_passed`` and ``check_gated`` (label
    ``latest``), and a note. The result is the held-out step of a finished build.
    """

    def applies(self, path: Path) -> bool:
        """Whether ``path`` holds a surrogate build."""
        return SurrogateDirectory.holds_build(path)

    def summarize(self, path: Path) -> RunSummary:
        """Summarise the build in ``path``.

        Raises
        ------
        ConfigError
            If it has not finished a batch of runs yet.
        """
        summary = surrogates.build_summary(path)
        failed = set(summary.failed)
        steps = (
            StepSummary(0, summary.training_runs, label="training"),
            StepSummary(1, summary.held_out_runs, label="held-out"),
        )
        rows = [_single("failed_runs", "all", 0, float(len(failed)))]
        notes = []
        if summary.domain is not None:
            rows.append(_single("domain_coordinates", "all", 0, summary.domain.coordinates))
            rows.append(_single("domain_points", "all", 0, summary.domain.points))
        timing = {
            "fit_seconds": summary.fit_seconds,
            "prediction_seconds": summary.prediction_seconds,
            "simulator_median_seconds": summary.simulator_median_seconds,
        }
        for metric, value in timing.items():
            if value is not None:
                rows.append(_single(metric, "all", 0, value))
        if summary.prediction_seconds and summary.simulator_median_seconds:
            speedup = summary.simulator_median_seconds / summary.prediction_seconds
            rows.append(_single("speedup", "all", 0, speedup))
        if summary.gate is not None:
            for entry in summary.gate.datatypes:
                error = math.inf if entry.error is None else entry.error
                rows.append(_single("gate_error", entry.datatype, 1, error))
                rows.append(_single("gate_tolerance", entry.datatype, 1, entry.tolerance))
            rows.append(_single("gate_passed", "all", 1, float(summary.gate.passed)))
            notes += [f"gate failed: {line}" for line in summary.gate.failures()]
        if summary.model is not None:
            latest = surrogates.last_check(path / summary.model, summary.identity)
            if latest is not None:
                rows.append(_single("check_passed", "latest", 1, float(latest.passed)))
                rows.append(_single("check_gated", "latest", 1, float(latest.inside)))
                verdict = "passed" if latest.passed else "FAILED"
                notes.append(
                    f"latest check {latest.id} ({latest.source}, {latest.created}): {verdict}, "
                    f"{latest.inside} of {latest.runs} runs gated"
                )
        if summary.pending:
            notes.append(f"{summary.pending} run(s) still to go")
        if failed:
            notes.append(f"{len(failed)} design run(s) failed")
        return RunSummary(
            workflow="surrogate",
            name=summary.name,
            path=path,
            method=f"{summary.plugin} ({summary.plan} design)",
            status=summary.status,
            steps=steps,
            metrics=tuple(rows),
            selected_iteration=1 if summary.status == "complete" else None,
            forecasts=summary.simulator_runs,
            notes=tuple(notes),
        )


def _evaluations(row: Mapping[str, object]) -> int:
    """Return an optimization's objective evaluations up to a history row."""
    return optimization.row_evaluations(row)


def _single(metric: str, label: str, index: int, value: float) -> MetricStatistics:
    """Return a value that is not a statistic over members (a forecast score)."""
    nan = float("nan")
    return MetricStatistics(metric, label, index, float(value), nan, nan, nan, nan)


def find_runs(paths: Iterable[Path]) -> list[Path]:
    """Return the run directories at or under ``paths``, in order, without duplicates.

    A directory is a run if a summarizer applies to it. The search does not descend into
    runs, except for their ``predictions/``, whose predictions follow their run.
    """
    found: list[Path] = []

    def add(directory: Path) -> None:
        if directory.resolve() not in {p.resolve() for p in found}:
            found.append(directory)

    for root in paths:
        pending = [Path(root)]
        while pending:
            directory = pending.pop(0)
            if not directory.is_dir():
                continue
            if _summarizer(directory) is not None:
                add(directory)
                predictions = directory / "predictions"
                if predictions.is_dir():
                    for child in sorted(p for p in predictions.iterdir() if p.is_dir()):
                        if _summarizer(child) is not None:
                            add(child)
                continue
            pending.extend(sorted(p for p in directory.iterdir() if p.is_dir()))
    return found


def summarize(paths: Sequence[Path]) -> list[RunSummary]:
    """Summarise every run at or under ``paths`` (see :func:`find_runs`).

    Runs that cannot be read (no scored forecast yet) are skipped with a warning.

    Raises
    ------
    ConfigError
        If no run is found.
    """
    runs = find_runs(paths)
    if not runs:
        raise ConfigError(f"no run directory at or under {', '.join(map(str, paths))}")
    result = []
    for run in runs:
        summarizer = _summarizer(run)
        assert summarizer is not None
        try:
            result.append(summarizer.summarize(run))
        except ApexError as error:
            logger.warning(
                "run not summarised: %s", error, extra={"event": "summary_skipped", "run": str(run)}
            )
    return result


def summarize_one(path: Path) -> RunSummary:
    """Summarise the run (or prediction) in ``path``.

    Raises
    ------
    ConfigError
        If ``path`` holds no run a summarizer recognises.
    """
    summarizer = _summarizer(Path(path))
    if summarizer is None:
        raise ConfigError(f"{path} holds no run, prediction or optimization to summarise")
    return summarizer.summarize(Path(path))


def _summarizer(path: Path) -> Summarizer | None:
    for name in summaries.names():
        candidate = summaries.create(name)
        if candidate.applies(path):
            return candidate
    return None


def _method(config: HistoryMatchConfig) -> str:
    method = config.method
    parts = [p.name for p in (method.strategy, method.localization) if p is not None]
    return method.name + (f" ({', '.join(parts)})" if parts else "")


def _status(path: Path, results: RunResults) -> str:
    summary = path / "summary.json"
    if not summary.exists():
        return "running or interrupted"
    content = json.loads(summary.read_text())
    validation = content.get("validation") or {}
    if validation.get("stopped_by_cv"):
        return "stopped by cross validation"
    if content.get("converged"):
        return "converged"
    return "complete" if content.get("complete") else "stopped"


def _model_change(results: RunResults, config: HistoryMatchConfig) -> list[MetricStatistics]:
    """PET-fork's model change between consecutive accepted iterates, after the run."""
    actnum = None
    for section in config.metrics or []:
        if section.name == "rmse" and "actnum_file" in section.options:
            actnum = read_array(
                Path(str(section.options["actnum_file"])),
                section.options.get("actnum_key"),
            )
            break
    metric = ModelChange(actnum=actnum)
    rows: list[MetricStatistics] = []
    accepted = results.accepted_iterations
    for before, after in itertools.pairwise(accepted):
        old, new = results.iterate(before), results.iterate(after)
        position = {int(m): i for i, m in enumerate(old.member_ids)}
        columns = [position[int(m)] for m in new.member_ids]
        context = MetricContext(
            state=new.state,
            # Model change reads the states only; a stand-in datum fills the rest.
            predicted=np.zeros((1, len(columns))),
            observations=ObservationSet([DataKey("-", "-")], [0.0], [1.0]),
            previous_state={k: v[:, columns] for k, v in old.state.items()},
        )
        for label, values in metric.evaluate(context).items():
            rows.append(
                MetricStatistics(
                    "model_change",
                    label,
                    after,
                    float(values.mean()),
                    float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    float(np.median(values)),
                    float(np.percentile(values, 10)),
                    float(np.percentile(values, 90)),
                )
            )
    return rows
