"""Observation-error covariance used to whiten data and draw perturbations.

Ported from PET
---------------
PET module : src/misc/sampling.py (gen_real), src/pipt/update_schemes/analysis/base.py (solve)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple
             data assimilation. Computers & Geosciences, 55, 3-15.
Differences: PET samples with the upper Cholesky factor U (C = U^T U, perturbations U^T z)
             but whitens with U^-1, which is only correct for a diagonal C. For a
             correlated C this gives the gain an effective covariance U U^T != C. APEX uses
             the lower factor L = U^T for both, so C = L L^T holds throughout. Perturbations
             drawn from the same z equal PET's: bit-for-bit for diagonal C (both use
             sqrt(alpha * var)), up to rounding for dense C.
"""

from typing import Self

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


class DataErrorCovariance:
    """Observation-error covariance ``C_d``, stored as variances or as a dense matrix.

    Build instances with :meth:`diagonal` or :meth:`dense`. The square-root factor
    ``L`` with ``C_d = L L^T`` (``sqrt(variances)`` or the lower Cholesky factor) is
    computed once and used both to whiten data and to colour perturbations.
    """

    __slots__ = ("_factor", "_values")

    def __init__(self, values: NDArray[np.float64], factor: NDArray[np.float64]) -> None:
        # Private: callers use the validating constructors below.
        self._values = values
        self._factor = factor

    @classmethod
    def diagonal(cls, variances: ArrayLike) -> Self:
        """Uncorrelated errors with the given variances, shape ``(n_data,)``.

        Raises
        ------
        ValueError
            If ``variances`` is not a 1-D array of finite, strictly positive values.
        """
        values = np.asarray(variances, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"variances must be a non-empty 1-D array, got shape {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("variances must be finite and strictly positive")
        return cls(values, np.sqrt(values))

    @classmethod
    def dense(cls, matrix: ArrayLike) -> Self:
        """Correlated errors with the given covariance matrix, shape ``(n_data, n_data)``.

        Raises
        ------
        ValueError
            If ``matrix`` is not square, symmetric and positive definite.
        """
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
            raise ValueError(f"covariance must be a non-empty square matrix, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("covariance must be finite")
        # Absolute tolerance scaled to the variances, so round-off next to a zero entry
        # (e.g. 0 vs 1e-300) is not reported as asymmetry.
        scale = float(np.abs(np.diagonal(values)).max())
        if not np.allclose(values, values.T, rtol=1e-12, atol=1e-12 * scale):
            raise ValueError("covariance must be symmetric")
        try:
            factor = scipy.linalg.cholesky(values, lower=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        return cls(values, factor)

    @property
    def n_data(self) -> int:
        """Number of observations."""
        return int(self._values.shape[0])

    @property
    def is_diagonal(self) -> bool:
        """Whether the covariance is stored as variances only."""
        return self._values.ndim == 1

    @property
    def values(self) -> NDArray[np.float64]:
        """The variances ``(n_data,)`` or covariance matrix ``(n_data, n_data)`` (a copy)."""
        return self._values.copy()

    def inflated(self, alpha: float) -> Self:
        """Return ``alpha * C_d``.

        The factor is ``sqrt(alpha * variances)`` for a diagonal covariance, the same
        arithmetic as PET, and ``sqrt(alpha) * L`` for a dense one, which avoids a new
        Cholesky factorisation.

        Raises
        ------
        ValueError
            If ``alpha`` is not finite and strictly positive.
        """
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"inflation factor must be finite and positive, got {alpha}")
        values = alpha * self._values
        factor = np.sqrt(values) if self.is_diagonal else np.sqrt(alpha) * self._factor
        return type(self)(values, factor)

    def whiten(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``L^-1 @ matrix`` for a data-space ``matrix`` of shape ``(n_data, k)``."""
        self._check_rows(matrix)
        if self.is_diagonal:
            return matrix / self._factor[:, None]
        return scipy.linalg.solve_triangular(self._factor, matrix, lower=True)

    def correlate(self, standard_normal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``L @ z``, turning standard-normal draws ``(n_data, k)`` into errors ~ ``C_d``."""
        self._check_rows(standard_normal)
        if self.is_diagonal:
            return self._factor[:, None] * standard_normal
        return self._factor @ standard_normal

    def data_mismatch(self, residuals: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-column data mismatch ``r_j^T C_d^-1 r_j`` of residuals ``(n_data, k)``: ``(k,)``.

        Equals PET's ``calc_objectivefun`` for each ensemble member.
        """
        whitened = self.whiten(residuals)
        data_mismatch: NDArray[np.float64] = np.einsum("ij,ij->j", whitened, whitened)
        return data_mismatch

    def perturbations(self, rng: np.random.Generator, n_ensemble: int) -> NDArray[np.float64]:
        """Draw ``n_ensemble`` error realisations, shape ``(n_data, n_ensemble)``."""
        return self.correlate(rng.standard_normal((self.n_data, n_ensemble)))

    def _check_rows(self, matrix: NDArray[np.float64]) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != self.n_data:
            raise ValueError(f"expected shape ({self.n_data}, k), got {matrix.shape}")
