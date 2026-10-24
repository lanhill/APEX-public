"""Quality assurance of forecasts and quality control of analyses (PET's QA/QC).

Ported from PET
---------------
PET module : src/pipt/misc_tools/qaqc_tools.py (QAQC.calc_coverage, calc_mahalanobis,
             _joint_scores, calc_kg, _projection, _gain, calc_da_stat)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Oliver, D. S. (2020). Diagnosing reservoir model deficiency for model
             improvement. Journal of Petroleum Science and Engineering, 193, 107367 (the
             Mahalanobis diagnostic).
Differences: the computations only; PET's plots, grid output and seismic coverage maps
             are left to the reporting layer (``apex qaqc``). Data types are
             ``DataKey.datatype`` (PET's data-frame columns), report points the keys'
             times. ``kalman_gain`` does not apply PET's optional auto-adaptive
             localization.

All arrays keep the ensemble axis last: forecasts ``(n_data, n_ensemble)``.

- :func:`coverage`: observations below or above every member (the forecast misses them);
- :func:`mahalanobis_scores` and :func:`joint_mahalanobis`: each observation (or pair or
  triple) scored against the forecast perturbed with its observation error, ``(d - mean)^2
  / var`` (Oliver 2020): a large score is a datum the model cannot reach;
- :func:`leading_projection`: a data type's series projected onto its leading principal
  component, to score data types rather than data;
- :func:`kalman_gain`: the ES-style gain each data type would apply to a parameter group,
  in the subspace of the forecast anomalies, for ranking which data move what;
- :func:`parameter_movement`: how far each parameter moved from the prior, in prior
  standard deviations.
"""

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.kernels import truncated_svd

#: PET's energy for the gain ranking's subspace (``_projection``).
GAIN_ENERGY = 0.99
#: PET's seed for the Mahalanobis perturbations, so repeated reports agree.
MAHALANOBIS_SEED = 50


@dataclass(frozen=True, slots=True)
class Coverage:
    """Which observations lie outside the forecast range, ``(n_data,)`` each."""

    below: NDArray[np.bool_]
    above: NDArray[np.bool_]

    @property
    def outside(self) -> int:
        """Number of observations outside the range."""
        return int(self.below.sum() + self.above.sum())


def coverage(predicted: NDArray[np.float64], observations: NDArray[np.float64]) -> Coverage:
    """Observations under every member (``below``) or over every member (``above``)."""
    obs = observations[:, None]
    return Coverage(
        below=np.asarray((obs < predicted).all(axis=1)),
        above=np.asarray((obs > predicted).all(axis=1)),
    )


def perturbed_forecast(
    predicted: NDArray[np.float64], variances: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return the forecast plus observation noise ``(n_data, N)`` (PET's ``fcst_pert``)."""
    noise = rng.standard_normal(predicted.shape)
    perturbed: NDArray[np.float64] = predicted + np.sqrt(variances)[:, None] * noise
    return perturbed


def mahalanobis_scores(
    perturbed: NDArray[np.float64], observations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Level-1 scores ``(d_i - mean_i)^2 / var_i`` of each datum ``(n,)`` (population var)."""
    scores: NDArray[np.float64] = (observations - perturbed.mean(axis=1)) ** 2 / perturbed.var(
        axis=1
    )
    return scores


def joint_mahalanobis(
    perturbed: NDArray[np.float64], observations: NDArray[np.float64], level: int
) -> dict[tuple[int, ...], float]:
    """Level-2 (pairs) or level-3 (triples) scores ``r^T C^-1 r / 2`` (PET ``_joint_scores``).

    ``C`` is the sample covariance (``ddof = 1``) of the combination; singular ones are
    skipped, as in PET.

    Raises
    ------
    ValueError
        For a level other than 2 or 3.
    """
    if level not in (2, 3):
        raise ValueError(f"level must be 2 or 3, got {level}")
    n = perturbed.shape[0]
    scores: dict[tuple[int, ...], float] = {}
    for combination in itertools.combinations(range(n), level):
        rows = list(combination)
        x = perturbed[rows]
        mean = x.mean(axis=1)
        diff = x - mean[:, None]
        cov = diff @ diff.T / (x.shape[1] - 1)
        residual = observations[rows] - mean
        try:
            scores[combination] = float(residual @ np.linalg.solve(cov, residual)) / 2
        except np.linalg.LinAlgError:
            continue
    return scores


def leading_projection(
    perturbed: NDArray[np.float64], observations: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Project a data type's series onto its leading principal component.

    The perturbed series ``(n_t, N)`` and observations ``(n_t,)`` are projected (PET's
    ``combine = 'time'``).
    """
    _, _, vt = np.linalg.svd((perturbed - perturbed.mean(axis=1, keepdims=True)).T,
                             full_matrices=False)  # fmt: skip
    leading = vt[:1, :]  # (1, n_t)
    return (leading @ perturbed).ravel(), float((leading @ observations[:, None]).item())


def kalman_gain(
    state: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    variances: NDArray[np.float64],
    damping: float = 0.0,
) -> NDArray[np.float64] | None:
    """Return the gain of the ensemble mean of ``state`` ``(n_b, N)`` from one data type.

    PET's ``_gain``: with the forecast anomalies ``A = Y - mean(Y)`` (not scaled) and their
    truncated SVD ``A = U s V^T`` (energy 0.99), ``X0 = (N - 1) s^-1 U^T C_d U s^-1`` (``C_d``
    diagonal), ``X0 = Z Lambda Z^T`` and ``X1 = U s^-1 Z``, the operator ``A^T X1 ((1 + lambda)
    + Lambda)^-1 X1^T`` takes the mean residual ``mean(d - Y)`` to member weights. Returns
    the parameters' gain ``(n_b,)``, or ``None`` if the forecast has no spread.
    """
    anomaly = predicted - predicted.mean(axis=1, keepdims=True)
    u, s, _ = truncated_svd(anomaly, GAIN_ENERGY)
    if s.size == 0 or not np.any(s):
        return None
    inverse = 1.0 / s
    n = predicted.shape[1]
    x0 = (n - 1) * ((inverse[:, None] * u.T) @ (variances[:, None] * u)) * inverse[None, :]
    eigenvalues, z = np.linalg.eigh(x0)
    x1 = (u * inverse[None, :]) @ z  # (n_data, r)
    operator = (anomaly.T @ x1) / ((damping + 1) + eigenvalues)[None, :] @ x1.T  # (N, n_data)
    residual = (observations[:, None] - predicted).mean(axis=1)
    gain: NDArray[np.float64] = state @ (operator @ residual)
    return gain


@dataclass(frozen=True, slots=True)
class Movement:
    """How far a parameter group moved from the prior (PET ``calc_da_stat``).

    ``prior_std`` and ``std``: the mean over parameters of the member standard deviation
    (population), before and now; ``beyond``: the percentage of parameters whose mean moved
    by more than 1, 2 and 3 prior standard deviations.
    """

    prior_std: float
    std: float
    beyond: tuple[float, float, float]


def parameter_movement(prior: NDArray[np.float64], current: NDArray[np.float64]) -> Movement:
    """Compare a parameter group's current ensemble ``(n_b, N')`` with its prior ``(n_b, N)``."""
    std0 = prior.std(axis=1)
    moved = current.mean(axis=1) - prior.mean(axis=1)
    beyond = tuple(float(100 * np.mean(np.abs(moved) > k * std0)) for k in (1, 2, 3))
    return Movement(float(std0.mean()), float(current.std(axis=1).mean()), beyond)  # type: ignore[arg-type]
