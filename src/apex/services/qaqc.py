"""QA/QC of a history-matching run: PET's forecast and analysis diagnostics (``apex qaqc``).

For one saved iterate (the posterior by default), over the history-matching data:

- **coverage**: per data type, the observations that lie outside the forecast range (the
  forecast misses them);
- **Mahalanobis** (Oliver 2020): each datum scored against the forecast perturbed with
  its observation error, the data types scored on their series' leading component, and
  (for up to :data:`MAX_PAIRS_DATA` data) pairs of data; the largest scores are the data
  the model cannot reach;
- **Kalman gains**: per data type and parameter group, the gain the data would apply to
  the ensemble mean, ranked by its largest and its mean entry: which data move what;
- **movement**: per parameter group, the spread before and now and the share of
  parameters that moved more than 1, 2 and 3 prior standard deviations.

The computations are :mod:`apex.algorithms.qaqc` (PET parity); the report is written by
:mod:`apex.reporting.qaqc_report`.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.algorithms import qaqc
from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.services import composition
from apex.storage.run_directory import RunDirectory

#: Pairs are scored only up to this many point data (the pairs grow as n^2 / 2).
MAX_PAIRS_DATA = 400


@dataclass(frozen=True, slots=True)
class TypeCoverage:
    """One data type's forecast against its observations, for the report and its figure.

    ``forecast`` is ``(n_times, N)``; ``below``/``above`` flag observations outside it.
    """

    datatype: str
    times: tuple[str, ...]
    forecast: NDArray[np.float64]
    observations: NDArray[np.float64]
    below: NDArray[np.bool_]
    above: NDArray[np.bool_]
    #: The observations' error variances ``(n_times,)``, when known.
    variances: NDArray[np.float64] | None = None

    @property
    def outside(self) -> int:
        """Observations outside the forecast range."""
        return int(self.below.sum() + self.above.sum())

    def outside_with_noise(self, widths: float = 2.0) -> int | None:
        """Observations outside the forecast range widened by ``widths`` error deviations.

        An observation carries its own error, which the forecasts (model outputs) do not:
        a consistent posterior leaves many observations outside the bare range, more so
        the more it narrowed. Widened by the error, an observation outside is one the
        ensemble cannot explain. ``None`` without the variances.
        """
        if self.variances is None:
            return None
        sd = widths * np.sqrt(self.variances)
        low = self.forecast.min(axis=-1) - sd
        high = self.forecast.max(axis=-1) + sd
        return int(((self.observations < low) | (self.observations > high)).sum())


@dataclass(frozen=True, slots=True)
class QAQCReport:
    """The diagnostics of one iterate (see the module docstring)."""

    run: str
    iteration: int
    coverage: tuple[TypeCoverage, ...]
    data_scores: tuple[tuple[str, float], ...]
    type_scores: tuple[tuple[str, float], ...]
    pair_scores: tuple[tuple[str, float], ...]
    gains_max: tuple[tuple[str, float], ...]
    gains_mean: tuple[tuple[str, float], ...]
    movement: dict[str, qaqc.Movement]
    notes: tuple[str, ...] = field(default_factory=tuple)


def qaqc_report(
    run_path: Path, iteration: int | None = None, *, damping: float = 0.0, top: int = 10
) -> QAQCReport:
    """Compute the QA/QC diagnostics of a run's iterate (the posterior by default).

    ``damping`` is the gains' ``lambda`` (0: the ES form; PET uses the scheme's own).

    Raises
    ------
    ConfigError
        If the run cannot be read.
    """
    config = composition.snapshot_config(RunDirectory(run_path))
    results = composition.load_results(run_path)
    step = results.posterior_iteration if iteration is None else iteration
    kept = results.accepted_iterations
    if step not in kept:
        raise ConfigError(
            f"{run_path} has no saved iteration {step}; its iterations are "
            f"{', '.join(map(str, kept)) or 'none yet'}"
        )
    iterate, prior = results.iterate(step), results.iterate(0)
    observations = composition.build_run(config).workflow.observations
    keys, values, variances = observations.keys, observations.values, observations.variances
    predicted = iterate.predicted
    notes: list[str] = []

    types: dict[str, list[int]] = {}
    for row, key in enumerate(keys):
        types.setdefault(key.datatype, []).append(row)
    point = {t: [r for r in rows if keys[r].index is None] for t, rows in types.items()}

    coverage = []
    for datatype, rows in point.items():
        if not rows:
            continue
        result = qaqc.coverage(predicted[rows], values[rows])
        coverage.append(
            TypeCoverage(
                datatype,
                tuple(str(keys[r].time) for r in rows),
                predicted[rows],
                values[rows],
                result.below,
                result.above,
                variances[rows],
            )
        )

    # One stream across the passes, in PET's order: data, pairs, then data types.
    rng = np.random.default_rng(qaqc.MAHALANOBIS_SEED)
    rows = [r for t in point for r in point[t]]
    labels = [_label(keys[r]) for r in rows]
    data_scores: list[tuple[str, float]] = []
    pair_scores: list[tuple[str, float]] = []
    if rows:
        perturbed = qaqc.perturbed_forecast(predicted[rows], variances[rows], rng)
        scores = qaqc.mahalanobis_scores(perturbed, values[rows])
        data_scores = _top(zip(labels, scores.tolist(), strict=True), top)
        # The pairs' perturbation is drawn even when they are not scored, so the data-type
        # pass draws what PET's does.
        perturbed = qaqc.perturbed_forecast(predicted[rows], variances[rows], rng)
        if len(rows) <= MAX_PAIRS_DATA:
            joint = qaqc.joint_mahalanobis(perturbed, values[rows], 2)
            pair_scores = _top(
                ((f"{labels[i]} & {labels[j]}", s) for (i, j), s in joint.items()), top
            )
        else:
            notes.append(f"pairs not scored: {len(rows)} point data (> {MAX_PAIRS_DATA})")
    projected = []
    for datatype, type_rows in point.items():
        if type_rows:  # every non-empty type, as PET (one time: its only value)
            series = qaqc.perturbed_forecast(predicted[type_rows], variances[type_rows], rng)
            projected.append((datatype, *qaqc.leading_projection(series, values[type_rows])))
    type_scores: list[tuple[str, float]] = []
    if projected:
        scores = qaqc.mahalanobis_scores(
            np.array([p for _, p, _ in projected]), np.array([o for _, _, o in projected])
        )
        type_scores = _top(zip([t for t, _, _ in projected], scores.tolist(), strict=True), top)

    gains_max, gains_mean = [], []
    for datatype, type_rows in types.items():
        for name, block in iterate.state.items():
            if block.shape[0] == 1:
                continue  # PET ranks field parameters only
            gain = qaqc.kalman_gain(
                block, predicted[type_rows], values[type_rows], variances[type_rows], damping
            )
            if gain is None:
                continue
            gains_max.append((f"{datatype} -> {name}", float(np.abs(gain).max())))
            gains_mean.append((f"{datatype} -> {name}", float(abs(gain.mean()))))

    movement = {
        name: qaqc.parameter_movement(prior.state[name], iterate.state[name])
        for name in iterate.state
    }
    return QAQCReport(
        run=config.run.name,
        iteration=step,
        coverage=tuple(coverage),
        data_scores=tuple(data_scores),
        type_scores=tuple(type_scores),
        pair_scores=tuple(pair_scores),
        gains_max=tuple(_top(gains_max, top)),
        gains_mean=tuple(_top(gains_mean, top)),
        movement=movement,
        notes=tuple(notes),
    )


def _label(key: DataKey) -> str:
    return f"{key.datatype} @ {key.time}"


def _top(items: Iterable[tuple[str, float]], n: int) -> list[tuple[str, float]]:
    # NaN (a datum without spread) ranks last rather than anywhere.
    ranked = sorted(items, key=lambda item: -np.inf if np.isnan(item[1]) else item[1], reverse=True)
    return [(str(label), float(value)) for label, value in ranked[:n]]
