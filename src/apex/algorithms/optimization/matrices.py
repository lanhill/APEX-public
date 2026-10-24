"""Small matrix helpers of the optimizers: eigenvalue flooring and correlation.

Ported from PET
---------------
PET module : src/popt/misc_tools/optim_tools.py (get_sym_pos_semidef, cov2corr)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : none (standard linear algebra)
Differences: ``cov_to_corr`` gives a zero-variance component unit variance and no
             correlation, where PET returns NaN.
"""

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray

#: Eigenvalues are floored here when a matrix is not positive definite (PET's ``rtol``).
EIGENVALUE_FLOOR = 1e-5


def floor_eigenvalues(a: FloatArray, floor: float = EIGENVALUE_FLOOR) -> FloatArray:
    """Return the symmetric ``a`` ``(d, d)`` with its eigenvalues floored at ``floor``.

    Only a matrix with a non-positive eigenvalue is changed (then every eigenvalue below
    ``floor`` is raised to it); a positive definite one is returned as it is, as in PET.
    """
    eigenvalues, vectors = np.linalg.eigh(a)
    if np.all(eigenvalues > 0):
        return a
    eigenvalues = np.clip(eigenvalues, floor, None)
    return np.asarray((vectors * eigenvalues) @ vectors.T, dtype=float)


def cov_to_corr(cov: FloatArray) -> FloatArray:
    """Return the correlation matrix ``(d, d)`` of the covariance ``cov`` ``(d, d)``.

    A component with zero variance (e.g. an evolution-path component a bound projection set
    to zero) is uncorrelated with the others and has unit variance; PET returns NaN there.
    """
    std = np.sqrt(np.diag(cov))
    scale = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(scale > 0, cov / np.where(scale > 0, scale, 1.0), 0.0)
    corr[np.diag_indices_from(corr)] = np.where(std > 0, np.diag(corr), 1.0)
    return np.asarray(corr, dtype=float)
