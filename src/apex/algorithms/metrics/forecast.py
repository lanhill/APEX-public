"""Forecast scores of an ensemble against the true model's forecast.

Every kernel takes one vector's ensemble forecast over the prediction period,
``(n_times, n_members)`` (ensemble axis last), and the truth's forecast ``(n_times,)``:

- ``coverage_p10_p90``: the fraction of times at which the truth lies within the
  ensemble's [10th, 90th] percentile band (NumPy's linear interpolation). For an ensemble
  of ``n`` members exchangeable with the truth (a calibrated one), the expected value is
  about ``0.8 (n - 1) / (n + 1)``, not 0.8: the band between the empirical percentiles
  holds ``0.8 (n - 1)`` of the ``n + 1`` equally likely ranks of the truth (0.65 for 10
  members, 0.72 for 20, 0.78 for 100; numerics review, confirmed by simulation). It is
  exact in rank; within a gap between two members the truth's position depends on the
  distribution (Gaussian: about 0.015 higher at 10 members, negligible from 20 on).
  :func:`nominal_coverage` gives it; well below nominal means over-confidence.
- ``crps``: the continuous ranked probability score of the ensemble's empirical
  distribution, averaged over time (Gneiting & Raftery, 2007, eq. 21, the "energy form"):
  ``CRPS = mean_j |y_j - t| - 1/2 mean_{j,k} |y_j - y_k|``. It is a proper score, in the
  units of the vector; lower is better, 0 for an ensemble collapsed on the truth.
- ``mean_error``: the RMSE over time of the ensemble mean against the truth.
- ``spread``: the ensemble standard deviation (``ddof = 1``), averaged over time; to be
  read with ``coverage`` (a small spread with low coverage is a collapsed ensemble).

Reference: Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules,
prediction, and estimation. Journal of the American Statistical Association, 102(477),
359-378.
"""

import numpy as np
from numpy.typing import NDArray

from apex.core.plugins import forecast_scores


def _check(
    ensemble: NDArray[np.float64], truth: NDArray[np.float64], min_members: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ensemble, truth = np.asarray(ensemble, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if ensemble.ndim != 2 or truth.shape != (ensemble.shape[0],):
        raise ValueError(
            f"expected ensemble (n_times, n_members) and truth (n_times,), got "
            f"{ensemble.shape} and {truth.shape}"
        )
    if ensemble.shape[0] == 0 or ensemble.shape[1] < min_members:
        raise ValueError(
            f"need at least one time and {min_members} member(s), got {ensemble.shape}"
        )
    return ensemble, truth


def coverage(
    ensemble: NDArray[np.float64],
    truth: NDArray[np.float64],
    low: float = 10.0,
    high: float = 90.0,
) -> float:
    """Fraction of times with the truth in the ensemble's ``[low, high]`` percentile band."""
    ensemble, truth = _check(ensemble, truth)
    lower, upper = np.percentile(ensemble, [low, high], axis=-1)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def nominal_coverage(n_members: int, low: float = 10.0, high: float = 90.0) -> float:
    """Return the expected coverage of a calibrated ensemble: ``(high-low)% (n-1)/(n+1)``."""
    if n_members < 2:
        raise ValueError(f"need at least 2 members, got {n_members}")
    return (high - low) / 100.0 * (n_members - 1) / (n_members + 1)


def crps_per_time(ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensemble CRPS at each time, ``(n_times,)``.

    ``mean_{j,k} |y_j - y_k|`` is evaluated from the sorted members in ``O(n log n)``:
    ``2 / n^2 sum_i (2 i - n - 1) y_(i)``, with ``i = 1..n`` the rank.
    """
    ensemble, truth = _check(ensemble, truth)
    n = ensemble.shape[1]
    sorted_members = np.sort(ensemble, axis=-1)
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    pairwise = 2.0 / n**2 * (sorted_members @ weights)
    result: NDArray[np.float64] = (
        np.mean(np.abs(ensemble - truth[:, None]), axis=-1) - 0.5 * pairwise
    )
    return result


def crps(ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """Ensemble CRPS averaged over time."""
    return float(np.mean(crps_per_time(ensemble, truth)))


def mean_error(ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """RMSE over time of the ensemble mean against the truth."""
    ensemble, truth = _check(ensemble, truth)
    return float(np.sqrt(np.mean((ensemble.mean(axis=-1) - truth) ** 2)))


def spread(ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
    """Ensemble standard deviation (``ddof = 1``) averaged over time; ``truth`` checks shapes."""
    ensemble, _ = _check(ensemble, truth, min_members=2)
    return float(np.mean(ensemble.std(axis=-1, ddof=1)))


@forecast_scores.register("coverage_p10_p90")
class CoverageP10P90:
    """Fraction of times at which the truth lies in the ensemble's [p10, p90] band."""

    def score(self, ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """Return the coverage, in [0, 1]."""
        return coverage(ensemble, truth)


@forecast_scores.register("crps")
class CRPS:
    """Continuous ranked probability score, averaged over time (lower is better)."""

    def score(self, ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """Return the mean CRPS, in the vector's units."""
        return crps(ensemble, truth)


@forecast_scores.register("mean_error")
class MeanError:
    """RMSE over time of the ensemble mean against the truth."""

    def score(self, ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """Return the RMSE, in the vector's units."""
        return mean_error(ensemble, truth)


@forecast_scores.register("spread")
class Spread:
    """Ensemble standard deviation, averaged over time."""

    def score(self, ensemble: NDArray[np.float64], truth: NDArray[np.float64]) -> float:
        """Return the mean standard deviation, in the vector's units."""
        return spread(ensemble, truth)
