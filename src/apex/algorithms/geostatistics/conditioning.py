"""Conditioning Gaussian fields to hard data by kriging their residuals.

An unconditional realisation ``z`` with covariance ``C`` becomes a conditional one by

    z_c = z + C_xd (C_dd + R)^-1 (d - z_d - e),    e ~ N(0, R),

with ``d`` the data at their cells, ``z_d`` the realisation there, ``C_xd`` the covariance
between every cell and the data's cells, and ``R`` the data's error variance (diagonal).
For a Gaussian ``z`` this is an exact draw from the field conditioned on the data: its mean
is the simple-kriging estimate and its covariance the kriging covariance (Journel &
Huijbregts 1978, §V.C; Chilès & Delfiner 2012, §7.3.1). With ``R = 0`` every realisation
takes the data's values at their cells. The mean is the generator's, so the kriging is
simple kriging around it.

Reference : Journel, A. G. & Huijbregts, C. J. (1978). Mining Geostatistics, Academic Press,
            §V.C ("conditioning by kriging"); Chilès, J.-P. & Delfiner, P. (2012).
            Geostatistics: Modeling Spatial Uncertainty, 2nd ed., Wiley, §7.3.1.
"""

import logging
import math

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from apex.algorithms.geostatistics.covariance import variogram_covariance

logger = logging.getLogger(__name__)

#: The data's correlation matrix (their covariance scaled to unit variances, so that
#: variables of different scales, ln k and porosity, count alike) worse conditioned than
#: this gets :data:`JITTER`. Close cells of a smooth model (Gaussian, long ranges) make it
#: near singular while Cholesky still succeeds, and data such a model cannot fit (rough
#: values a few cells apart) then blow the field up by orders of magnitude (1e4 x the
#: sill, measured), with nothing failing.
MAX_CONDITION = 1e8
#: The nugget added then, relative to each datum's own variance: it bounds the correlation
#: matrix's condition number by about ``n_d / JITTER`` and the kriging weights with it.
#: Data the model can fit are still met closely; data it cannot are met only as nearly as
#: the smooth field allows, which is logged with the largest miss.
JITTER = 1e-6
#: A kriged correction larger than this many prior standard deviations means the data are
#: improbable under the variogram (rough values close together under a smooth model): the
#: conditional field swings far outside the prior. The generators warn; the geomodeler
#: refuses such data (:func:`kriged_excursion`). The harm itself, not a chi-square test of
#: the data: that refused data drawn from the prior whenever the sill was 15 % off, which
#: estimated sills routinely are (measured on 500 wells' data).
LARGE_CORRECTION = 10.0


def layer_cross_covariance(
    nx: int,
    ny: int,
    columns: NDArray[np.int64],
    *,
    model: str,
    sill: float,
    major_range: float,
    minor_range: float | None = None,
    azimuth: float = 0.0,
    exponent: float = 1.0,
) -> NDArray[np.float64]:
    """Covariance between every cell of a layer and some of its cells, ``(nx*ny, n_c)``.

    ``columns`` are 0-based cell indices within the layer, in Eclipse order (``i + nx*j``).
    The model's conventions are :func:`~apex.algorithms.geostatistics.layer_covariance`'s:
    distances in cells, the major axis at ``azimuth`` degrees counter-clockwise from the
    i-axis, and a range of 0 uncorrelated.
    """
    minor = major_range if minor_range is None else minor_range
    if major_range < 0 or minor < 0:
        raise ValueError("ranges must be non-negative")
    n = nx * ny
    columns = np.asarray(columns, dtype=np.int64)
    if major_range == 0 or minor == 0:
        same = np.arange(n)[:, None] == columns[None, :]
        return np.where(same, float(sill), 0.0)
    j, i = np.divmod(np.arange(n), nx)
    di = i[:, None] - i[columns][None, :]
    dj = j[:, None] - j[columns][None, :]
    theta = math.radians(azimuth)
    along = math.cos(theta) * di + math.sin(theta) * dj
    across = -math.sin(theta) * di + math.cos(theta) * dj
    u = np.hypot(along / major_range, across / minor)
    return variogram_covariance(u, model, sill, exponent)


def kriging_factor(
    data_covariance: NDArray[np.float64],
) -> tuple[tuple[NDArray[np.float64], NDArray[np.bool_]], float, bool]:
    """Return the Cholesky factor the kriging uses, the condition number, and the jitter.

    ``data_covariance`` is ``(n_d, n_d)``, the variogram's at the data plus their error
    variances. Its correlation matrix's condition number decides :data:`JITTER`, which is
    then added relative to each datum's variance (see :data:`MAX_CONDITION`).

    Raises
    ------
    ValueError
        If it cannot be factored even so.
    """
    covariance = 0.5 * (data_covariance + data_covariance.T)
    variance = np.diag(covariance).copy()
    if np.any(variance <= 0):
        raise ValueError("the hard data's variances must be positive")
    scale = np.sqrt(variance)
    condition = float(np.linalg.cond(covariance / np.outer(scale, scale)))
    jittered = not math.isfinite(condition) or condition > MAX_CONDITION
    if jittered:
        covariance = covariance + np.diag(JITTER * variance)
    try:
        factor = scipy.linalg.cho_factor(covariance, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError(
            "the hard data's covariance cannot be factored; give the data an error_std"
        ) from None
    return factor, condition, jittered


def kriged_excursion(
    cross: NDArray[np.float64],
    rows: NDArray[np.int64],
    departure: NDArray[np.float64],
    error_variance: NDArray[np.float64],
    prior_variance: NDArray[np.float64],
) -> float:
    """Return the kriged mean's largest departure from the prior mean, in prior stds.

    ``departure`` is the data minus the prior mean at their cells, ``(n_d,)``;
    ``prior_variance`` each entry's prior variance, ``(n,)``. This is how far conditioning
    moves the field on average: data improbable under the variogram make it large (see
    :data:`LARGE_CORRECTION`). Computed on the matrix the kriging factors.
    """
    factor, _, _ = kriging_factor(cross[rows] + np.diag(error_variance))
    mean_shift = cross @ scipy.linalg.cho_solve(factor, departure)
    return float(np.max(np.abs(mean_shift) / np.sqrt(prior_variance), initial=0.0))


def condition_by_kriging(
    fields: NDArray[np.float64],
    cross: NDArray[np.float64],
    rows: NDArray[np.int64],
    values: NDArray[np.float64],
    error_variance: NDArray[np.float64],
    noise: NDArray[np.float64] | None = None,
    prior_variance: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return the realisations conditioned to the data (see the module).

    Parameters
    ----------
    fields
        Unconditional realisations, ``(n, n_ensemble)``.
    cross
        ``C_xd``, the covariance between every entry and the data's entries, ``(n, n_d)``.
    rows
        The data's entries in ``fields``, ``(n_d,)``, distinct.
    values
        The data, ``(n_d,)``.
    error_variance
        ``R``'s diagonal, ``(n_d,)``, non-negative.
    noise
        Standard normals ``(n_d, n_ensemble)`` for the data's errors; needed (and only
        used) where ``error_variance`` is positive.
    prior_variance
        Each entry's prior variance, ``(n,)``, against which a large correction is
        warned about (:data:`LARGE_CORRECTION`); default the largest at the data.

    Returns
    -------
    numpy.ndarray
        ``(n, n_ensemble)``, C-contiguous.

    Raises
    ------
    ValueError
        If the data's covariance cannot be factored (with :data:`JITTER` when near singular).
    """
    residual = values[:, None] - fields[rows]
    if np.any(error_variance > 0):
        if noise is None:
            raise ValueError("data with errors need noise draws")
        residual = residual - np.sqrt(error_variance)[:, None] * noise
    factor, condition, jittered = kriging_factor(cross[rows] + np.diag(error_variance))
    correction = cross @ scipy.linalg.cho_solve(factor, residual)
    conditioned: NDArray[np.float64] = np.ascontiguousarray(fields + correction)
    if prior_variance is None:
        prior_variance = np.full(cross.shape[0], float(np.max(np.diag(cross[rows]))))
    largest = float(np.max(np.abs(correction) / np.sqrt(prior_variance)[:, None], initial=0.0))
    if largest > LARGE_CORRECTION:
        logger.warning(
            "conditioning moves the field by up to %.3g prior standard deviations: the hard "
            "data are improbable under this variogram (rough values close together under a "
            "smooth model?); give them an error_std, or a shorter range or rougher model",
            largest,
        )
    if jittered:
        exact = error_variance == 0
        miss = np.abs(conditioned[rows[exact]] - values[exact, None]).max(axis=1, initial=0.0)
        relative = float(np.max(miss / np.sqrt(prior_variance[rows[exact]]), initial=0.0))
        logger.warning(
            "the hard data's covariance is near singular (correlation condition number "
            "%.2g: close cells of a smooth variogram); a nugget of %.0e of each datum's "
            "variance was added, and the members miss the data by up to %.3g prior stds. "
            "If that is large, the data vary more than the variogram allows: give them an "
            "error_std, or a rougher variogram or a shorter range",
            condition, JITTER, relative,
        )  # fmt: skip
    return conditioned
