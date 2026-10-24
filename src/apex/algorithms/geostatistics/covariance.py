"""Variogram covariance models, anisotropic grid covariances and correlated sampling.

See the package docstring for provenance and the differences from PET.
"""

import logging
import math

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

#: Covariance models as functions of the normalised lag ``u = distance / range``. The
#: factor 3 in the exponential family makes ``range`` the practical range (5% of the sill).
VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "cubic", "stable")


def variogram_covariance(
    u: NDArray[np.float64], model: str, sill: float = 1.0, exponent: float = 1.0
) -> NDArray[np.float64]:
    """Covariance ``sill * rho(u)`` for normalised lags ``u >= 0``.

    Parameters
    ----------
    u
        Lag divided by the range (anisotropy already applied).
    model
        One of :data:`VARIOGRAM_MODELS`. ``"stable"`` is ``exp(-3 u^exponent)`` with
        ``0 < exponent <= 2``; PET's ``'exp'`` is ``"stable"`` with ``exponent = 1.9``.
    sill
        Variance at zero lag.
    exponent
        Exponent of the ``"stable"`` model.

    Raises
    ------
    ValueError
        If the model is unknown or its parameters are invalid.
    """
    u = np.abs(u)
    inside = u < 1
    if model == "spherical":
        rho = np.where(inside, 1 - 1.5 * u + 0.5 * u**3, 0.0)
    elif model == "cubic":
        rho = np.where(inside, 1 - 7 * u**2 + 35 / 4 * u**3 - 7 / 2 * u**5 + 3 / 4 * u**7, 0.0)
    elif model == "exponential":
        rho = np.exp(-3 * u)
    elif model == "gaussian":
        rho = np.exp(-3 * u**2)
    elif model == "stable":
        if not 0 < exponent <= 2:
            raise ValueError(f"stable exponent must be in (0, 2], got {exponent}")
        rho = np.exp(-3 * u**exponent)
    else:
        raise ValueError(f"unknown variogram model {model!r}; expected one of {VARIOGRAM_MODELS}")
    if sill <= 0:
        raise ValueError(f"sill must be positive, got {sill}")
    covariance: NDArray[np.float64] = sill * rho
    return covariance


def layer_covariance(
    nx: int,
    ny: int,
    *,
    model: str,
    sill: float,
    major_range: float,
    minor_range: float | None = None,
    azimuth: float = 0.0,
    exponent: float = 1.0,
) -> NDArray[np.float64]:
    """Covariance of one grid layer, ``(nx*ny, nx*ny)``, cells in Eclipse order (i fastest).

    Distances are in cells. The major axis points at ``azimuth`` degrees counter-clockwise
    from the i-axis; ``minor_range`` defaults to ``major_range`` (isotropic). A range of 0
    gives an uncorrelated (diagonal) covariance, as in PET.
    """
    if nx < 1 or ny < 1:
        raise ValueError(f"grid layer must be at least 1x1, got {nx}x{ny}")
    minor = major_range if minor_range is None else minor_range
    if major_range < 0 or minor < 0:
        raise ValueError("ranges must be non-negative")
    n = nx * ny
    if major_range == 0 or minor == 0:
        return np.diag(np.full(n, float(sill)))
    j, i = np.divmod(np.arange(n), nx)  # Eclipse order: index = i + nx * j
    di = i[:, None] - i[None, :]
    dj = j[:, None] - j[None, :]
    theta = math.radians(azimuth)
    along = math.cos(theta) * di + math.sin(theta) * dj
    across = -math.sin(theta) * di + math.cos(theta) * dj
    u = np.hypot(along / major_range, across / minor)
    return variogram_covariance(u, model, sill, exponent)


def covariance_factor(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``F`` with ``F F^T = covariance``: the lower Cholesky factor if possible.

    Numerically singular covariances (smooth models with long ranges) fall back to
    ``V sqrt(max(w, 0))`` from a symmetric eigen-decomposition, with a warning.
    """
    try:
        factor: NDArray[np.float64] = scipy.linalg.cholesky(covariance, lower=True)
        return factor
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(covariance)
        logger.warning(
            "covariance is not numerically positive definite (smallest eigenvalue %.3g); "
            "sampling with eigenvalues clipped at 0",
            float(w.min()),
        )
        clipped: NDArray[np.float64] = v * np.sqrt(np.clip(w, 0.0, None))[None, :]
        return clipped


def correlated_sample(
    mean: ArrayLike, factor: NDArray[np.float64], standard_normal: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return ``mean + F z`` for ``z`` ``(n, n_ensemble)``; ``mean`` is a scalar or ``(n,)``."""
    mean_vector = np.broadcast_to(np.asarray(mean, dtype=np.float64), (factor.shape[0],))
    sample: NDArray[np.float64] = mean_vector[:, None] + factor @ standard_normal
    return sample
