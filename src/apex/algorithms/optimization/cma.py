"""Covariance matrix adaptation (CMA): the rank-one and rank-mu updates.

Ported from PET
---------------
PET module : src/popt/optimization_methods/subroutines/cma.py (CMA)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Hansen, N. (2006). The CMA evolution strategy: a comparing review. In Towards a
             New Evolutionary Computation, 75-102. Springer.
Differences: the samples are ``(d, N)`` (PET: ``(N, d)``). Otherwise none.

``C <- (1 - a_1 - a_mu) C + a_1 p p^T + a_mu sum_i w_i y_i y_i^T`` over the ``n_mu`` best
samples ``y_i`` (centred on the ensemble mean), with the evolution path
``p <- (1 - a_c) p + sqrt(a_c (2 - a_c) mu_eff) step``. With ``corr_update`` both terms are
turned into correlation matrices, so that a correlation stays one.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np

from apex.algorithms.optimization.matrices import cov_to_corr
from apex.algorithms.optimization.protocols import FloatArray, OptimizerState, StateValue


class CMA:
    """Hansen's (2006) covariance update with PET's defaults.

    Parameters
    ----------
    ne
        Ensemble size.
    dim
        Number of controls ``d``.
    alpha_mu, alpha_1, alpha_c
        Learning rates of the rank-mu and rank-one updates and of the evolution path
        (Hansen's defaults when ``None``).
    n_mu
        Number of best samples in the rank-mu update (default ``ne // 2``).
    corr_update
        Adapt a correlation matrix instead of a covariance.
    equal_weights
        Weigh the ``n_mu`` samples equally; else ``log(n_mu + 1) - log(i)``, normalised.
    """

    def __init__(
        self,
        ne: int,
        dim: int,
        alpha_mu: float | None = None,
        n_mu: int | None = None,
        alpha_1: float | None = None,
        alpha_c: float | None = None,
        *,
        corr_update: bool = False,
        equal_weights: bool = True,
    ) -> None:
        if ne < 1:
            raise ValueError(f"ne (the ensemble size) must be at least 1, got {ne}")
        self.ne = ne
        self.dim = dim
        self.corr_update = corr_update
        self.n_mu = int(ne / 2) if n_mu is None else n_mu
        if not 0 < self.n_mu <= ne:
            raise ValueError(f"n_mu must be in [1, {ne}], got {self.n_mu}")
        if equal_weights:
            self.weights = np.ones(self.n_mu) / self.n_mu
        else:
            ranks = np.arange(1, self.n_mu + 1)
            self.weights = np.log(self.n_mu + 1) - np.log(ranks)
            self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.c_cov = 1 / self.mu_eff * 2 / (dim + 2**0.5) ** 2 + (1 - 1 / self.mu_eff) * min(
            1, (2 * self.mu_eff - 1) / ((dim + 2) ** 2 + self.mu_eff)
        )
        self.alpha_1 = self.c_cov / self.mu_eff if alpha_1 is None else alpha_1
        self.alpha_mu = self.c_cov * (1 - 1 / self.mu_eff) if alpha_mu is None else alpha_mu
        self.alpha_c = 4 / (dim + 4) if alpha_c is None else alpha_c
        self.evo_path: FloatArray | float = 0.0

    def rank_mu(self, samples: FloatArray, values: FloatArray) -> FloatArray:
        """Return the rank-mu matrix ``(d, d)`` of the ``n_mu`` best of ``samples`` ``(d, N)``."""
        best = values.argsort()[: self.n_mu]
        centred = samples[:, best] - samples.mean(axis=1, keepdims=True)
        c_mu = (centred * self.weights) @ centred.T
        return cov_to_corr(c_mu) if self.corr_update else np.asarray(c_mu)

    def rank_one(self, step: FloatArray) -> FloatArray:
        """Advance the evolution path by ``step`` ``(d,)``; return ``p p^T`` ``(d, d)``."""
        s = self.alpha_c
        self.evo_path = (1 - s) * self.evo_path + np.sqrt(s * (2 - s) * self.mu_eff) * step
        c_one = np.outer(self.evo_path, self.evo_path)
        return cov_to_corr(c_one) if self.corr_update else np.asarray(c_one)

    def __call__(
        self, cov: FloatArray, step: FloatArray, samples: FloatArray, values: FloatArray
    ) -> FloatArray:
        """Return the updated covariance (or correlation) ``(d, d)``.

        Parameters
        ----------
        cov
            Current matrix ``(d, d)``.
        step
            The accepted step direction ``(d,)`` (GenOpt: the step over the step size).
        samples
            The ensemble's samples ``(d, N)`` (GenOpt: the copula's normals).
        values
            Their objectives ``(N,)``.
        """
        c_mu = self.rank_mu(samples, values)
        c_one = self.rank_one(step)
        return np.asarray(
            (1 - self.alpha_1 - self.alpha_mu) * cov + self.alpha_1 * c_one + self.alpha_mu * c_mu
        )

    def export_state(self) -> OptimizerState:
        """Return the evolution path."""
        return {"evo_path": self.evo_path}

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Restore the evolution path."""
        path = state["evo_path"]
        if isinstance(path, np.ndarray):
            self.evo_path = np.array(path, dtype=float)
        else:
            self.evo_path = float(cast(float, path))
