"""Pure array kernels shared by ensemble data-assimilation methods.

Ported from PET
---------------
PET module : src/pipt/misc_tools/analysis_tools.py (truncSVD),
             src/pipt/ensembles/ensemble_base.py (proj)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y. & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
             ensemble smoother for efficient history matching and uncertainty
             quantification. Computational Geosciences, 17(4), 689-703.
Differences: anomalies subtract the mean instead of multiplying by PET's (ne, ne) projection
             matrix (same result, O(n ne) instead of O(n ne^2)). truncated_svd takes the
             energy as a fraction only; PET's percentage form belongs to the config importer.
"""

import math

import numpy as np
from numpy.typing import NDArray


def anomalies(ensemble: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scaled ensemble anomalies ``(A - mean(A)) / sqrt(N - 1)``.

    Parameters
    ----------
    ensemble
        Ensemble matrix, shape ``(n, n_ensemble)``.

    Returns
    -------
    NDArray
        Anomalies, shape ``(n, n_ensemble)``, such that ``anomalies @ anomalies.T`` is the
        unbiased sample covariance.

    Raises
    ------
    ValueError
        If ``ensemble`` is not 2-D or has fewer than two members.
    """
    if ensemble.ndim != 2 or ensemble.shape[-1] < 2:
        raise ValueError(
            f"expected shape (n, n_ensemble) with n_ensemble >= 2, got {ensemble.shape}"
        )
    n_ensemble = ensemble.shape[-1]
    centred: NDArray[np.float64] = ensemble - ensemble.mean(axis=-1, keepdims=True)
    return centred / math.sqrt(n_ensemble - 1)


def truncated_svd(
    matrix: NDArray[np.float64], energy: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Thin SVD truncated to the smallest rank retaining ``energy`` of the singular-value sum.

    Follows PET's rule exactly: the retained fraction accumulates the singular values
    themselves (a fraction of the nuclear norm, not of the squared Frobenius energy), and
    the kept rank is the first at which the cumulative fraction *reaches* ``energy``.

    Parameters
    ----------
    matrix
        Matrix to decompose, shape ``(m, n)``.
    energy
        Fraction of the singular-value sum to retain, in ``(0, 1]``.

    Returns
    -------
    tuple of NDArray
        ``U`` ``(m, r)``, singular values ``s`` ``(r,)`` and ``Vt`` ``(r, n)``, with ``r >= 1``.

    Raises
    ------
    ValueError
        If ``energy`` is outside ``(0, 1]`` or ``matrix`` is not 2-D.
    """
    if not 0.0 < energy <= 1.0:
        raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    total = s.sum()
    if total == 0.0:
        # A zero spectrum carries no ranking information; keep everything.
        rank = s.size
    else:
        # First index where the cumulative fraction reaches `energy`, kept, hence +1. The
        # clamp absorbs rounding in the last cumulative entry when energy == 1.
        rank = min(int(np.searchsorted(np.cumsum(s) / total, energy)) + 1, s.size)
    return u[:, :rank], s[:rank], vt[:rank, :]


def clip_to_bounds(
    state: NDArray[np.float64],
    lower: NDArray[np.float64] | None,
    upper: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Clip each state row to its bounds, returning a new array.

    Parameters
    ----------
    state
        State ensemble, shape ``(n_state, n_ensemble)``.
    lower, upper
        Per-row bounds, shape ``(n_state,)``; ``None`` or ``-inf``/``inf`` means unbounded.

    Returns
    -------
    NDArray
        Clipped state, shape ``(n_state, n_ensemble)``.
    """
    low = -np.inf if lower is None else lower[:, None]
    high = np.inf if upper is None else upper[:, None]
    return np.clip(state, low, high)
