"""Array kernels for performance metrics: normalized L2 distance and SSIM.

Definitions from PET-fork (``not public`` @ ``1348861``,
``Plotting/metric_analyses.py``: ``normalized_l2``, ``calculate_ssim``), where they are
used by the Brugge study (``Examples/.../Brugge/metric_analyses_brugge.py``).
SSIM is scikit-image's ``structural_similarity`` algorithm with its defaults (uniform
window, sample covariance, ``K1 = 0.01``, ``K2 = 0.03``), re-implemented here on scipy and
checked against scikit-image in the tests.

Reference: Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. (2004). Image quality
assessment: from error visibility to structural similarity. IEEE Transactions on Image
Processing, 13(4), 600-612.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

#: SSIM stabilising constants (Wang et al. 2004; scikit-image defaults).
K1, K2 = 0.01, 0.03


def normalized_l2(
    reference: NDArray[np.float64], ensemble: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the RMSE ``||reference - m_j||_2 / sqrt(n)`` of every member ``j``.

    Parameters
    ----------
    reference
        ``(n,)``.
    ensemble
        ``(n, n_members)``.

    Returns
    -------
    ``(n_members,)``.
    """
    if reference.ndim != 1 or ensemble.ndim != 2 or ensemble.shape[0] != reference.size:
        raise ValueError(
            f"expected reference (n,) and ensemble (n, n_members), got {reference.shape} "
            f"and {ensemble.shape}"
        )
    if reference.size == 0:
        raise ValueError("normalized_l2 of empty vectors")
    result: NDArray[np.float64] = np.sqrt(np.mean((ensemble - reference[:, None]) ** 2, axis=0))
    return result


def ssim(
    reference: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    data_range: float,
    win_size: int = 7,
) -> NDArray[np.float64]:
    """Mean structural similarity of each member's field to ``reference``.

    For windows ``w`` of ``win_size`` cells per axis, with local means ``mu``, sample
    variances ``s^2`` and covariance ``s_xy`` (normalized by ``N_w - 1``)::

        SSIM_w = (2 mu_x mu_y + C1)(2 s_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(s_x^2 + s_y^2 + C2))
        C1 = (K1 R)^2, C2 = (K2 R)^2, R = data_range

    averaged over the windows that fit inside the grid (a border of ``(win_size - 1) / 2``
    cells is excluded; filters reflect at the edges), as scikit-image does.

    Parameters
    ----------
    reference
        The true field on the grid, shape ``grid`` (1-3 dimensions).
    ensemble
        Member fields, ``(*grid, n_members)``.
    data_range
        ``R``, the value range the constants refer to (> 0).
    win_size
        Odd window size >= 3 (scikit-image fails for 1), at most every grid dimension.

    Returns
    -------
    ``(n_members,)``, each in ``[-1, 1]``; 1 for identical fields.

    Notes
    -----
    Computed in float64 whatever the input; scikit-image keeps float32 inputs in float32
    (differences ~1e-8 then).
    """
    grid = reference.shape
    if ensemble.ndim != reference.ndim + 1 or ensemble.shape[:-1] != grid:
        raise ValueError(f"ensemble must be (*{grid}, n_members), got {ensemble.shape}")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError(f"win_size must be odd and >= 3, got {win_size}")
    if any(n < win_size for n in grid):
        raise ValueError(
            f"win_size {win_size} exceeds the grid {grid}; use a smaller odd win_size or "
            "per-layer SSIM"
        )
    if not data_range > 0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    ndim = reference.ndim
    size = (win_size,) * ndim + (1,)  # never across members
    n_window = win_size**ndim
    cov_norm = n_window / (n_window - 1)

    x = reference.astype(np.float64)
    y = ensemble.astype(np.float64)
    # The reference's local moments are filtered once and broadcast over the members.
    ux = uniform_filter(x, size=win_size)[..., None]
    uxx = uniform_filter(x * x, size=win_size)[..., None]
    uy, uyy = uniform_filter(y, size=size), uniform_filter(y * y, size=size)
    uxy = uniform_filter(x[..., None] * y, size=size)
    vx, vy = cov_norm * (uxx - ux * ux), cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1, c2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    pad = (win_size - 1) // 2
    inner = s[tuple(slice(pad, n - pad) for n in grid)]
    result: NDArray[np.float64] = inner.reshape(-1, ensemble.shape[-1]).mean(axis=0)
    return result
