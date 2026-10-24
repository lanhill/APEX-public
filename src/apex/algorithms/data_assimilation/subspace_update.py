"""The ensemble subspace update: the stochastic iterative ensemble smoother in weight space.

Ported from PET
---------------
PET module : src/pipt/update_schemes/analysis/subspace.py (subspace_update.update),
             src/pipt/update_schemes/core/scheme_base.py (propose_state, weight branch),
             src/pipt/update_schemes/enrml.py (IterativeEnRML.score_and_commit: the
             weights are committed on acceptance)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Evensen, G., Raanes, P. N., Stordal, A. S. & Hove, J. (2019). Efficient
             implementation of an iterative ensemble smoother for data assimilation and
             reservoir history matching. Frontiers in Applied Mathematics and Statistics,
             5, 47. Raanes, P. N., Stordal, A. S. & Evensen, G. (2019). Revising the
             stochastic iterative ensemble smoother. Nonlinear Processes in Geophysics,
             26(3), 325-338.
Differences: none in the step (the parity test pins two iterations, the second with
             non-zero weights). Numerically null singular values of ``S`` (``s <= eps *
             max(shape) * s_max``) are dropped, as in APEX's RLM-MAC port: PET keeps them
             when ``energy`` is 1 and ``n_data >= N``, and then divides by rounding noise.
             The weights couple every member, so members can be dropped only in the prior
             forecast; a member failing in a later trial stops the run (PET cannot drop
             members at all). A correlated ``C_d``: PET whitens with the upper
             Cholesky factor's inverse ``U^-1`` (``C_d = U^T U``; the ES-MDA whitening bug),
             APEX with ``L^-1`` (``L = U^T``). That cancels only when ``S`` spans the data
             space (``n_data <= N - 1``, no truncation: the inverse is then exact and every
             term is whitened by the same matrix). Otherwise, the usual case (``n_data >=
             N``, or truncation), the step projects on the kept range of ``S``, which
             depends on the whitening, and the steps differ (21% and 31% on the parity
             cases). Bounds clip the trial after it is formed and
             do not change ``W``, as in PET: a clipped iterate is not ``X_p (I + W /
             sqrt(N - 1))``, and the next trial is formed from ``W`` again.

Each member is a combination of the *prior* members, ``X = X_p (I + W / sqrt(N - 1))``,
with the weights ``W`` ``(N, N)`` starting at 0. With ``Pi = (I - 1 1^T / N) / sqrt(N - 1)``,
whitened anomalies ``Y = L^-1 g(X) Pi``, ``Omega = I + W Pi`` and ``S = Y Omega^-1`` (the
linearised sensitivity in weight space), ``E = L^-1 D Pi`` (the perturbations of the fixed
perturbed observations ``D``) and the whitened residuals ``R = L^-1 (g(X) - D)``, one step is

    ``dW = -(1 + lambda)^-1 [W + S^T (S S^T + (1 + lambda) E E^T)^+ (R - S W)]``,

with the pseudo-inverse formed in the truncated SVD of ``S = U s V^T``: ``X = s^-1 U^T E``,
``X X^T = Q Lambda Q^T``, ``X2 = U s^-1 Q`` and ``X3 = S^T X2 = V Q``, so that the inverse is
``X2 (I + (1 + lambda) Lambda)^-1 X2^T``. The trial is ``X_p (I + (W + gamma dW) / sqrt(N -
1))``, with GN-EnRML's step length ``gamma`` (1 otherwise); its weights become ``W`` only if
the loop accepts it. At ``lambda = 0`` and ``gamma = 1`` this is Evensen et al.'s
Gauss-Newton iteration.
"""

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.registries import update_strategies
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.errors import WorkflowError


@update_strategies.register("subspace")
class SubspaceUpdate(EmitsDiagnostics):
    """Ensemble subspace update: an iterative update of each member's weights.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum of ``S`` kept, in ``(0, 1]`` (PET ``energy``;
        its iterative smoothers default to 0.95, which ``apex pet-import`` passes).
    """

    #: Centres on the ensemble mean: no mean-model forecast.
    requires_reference_prediction = False
    relative_damping = False
    #: Each member's prior state: the trial is a combination of the prior members.
    requires_prior_state = True
    #: The step length multiplies the weight step, so the strategy applies it itself.
    applies_step_length = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "singular_values": "retained singular values s of S = Y Omega^-1, (r,)",
        "weights": "the trial's ensemble weights W, (N, N)",
    }

    def __init__(self, energy: float = 0.98) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        self.energy = energy
        self._weights: NDArray[np.float64] | None = None  # committed W
        self._pending: NDArray[np.float64] | None = None  # the trial's W

    # ------------------------------------------------------------------ loop state

    def prepare(self, prior_state: NDArray[np.float64]) -> None:
        """Start from ``W = 0`` for the prior members that survived their forecast."""
        n = prior_state.shape[-1]
        self._weights, self._pending = np.zeros((n, n)), None

    def accept_trial(self) -> None:
        """Commit the last trial's weights (the loop accepted it).

        Raises
        ------
        RuntimeError
            If no trial is pending.
        """
        if self._pending is None:
            raise RuntimeError("SubspaceUpdate has no pending trial to accept")
        self._weights, self._pending = self._pending, None

    def drop_members(self, ok: NDArray[np.bool_]) -> None:
        """Refuse to drop members once the weights exist (they couple every member).

        Raises
        ------
        WorkflowError
            If a member is dropped.
        """
        if not np.all(ok):
            raise WorkflowError(
                f"{int(np.size(ok) - np.count_nonzero(ok))} member(s) failed in an update: "
                "the subspace analysis writes every member as a combination of all the prior "
                "members, so it cannot drop members after the prior forecast; rerun with "
                "retries for transient failures, or use the approx or full analysis"
            )

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the committed and pending weights, for resume."""
        arrays = {} if self._weights is None else {"weights": self._weights}
        if self._pending is not None:
            arrays["pending_weights"] = self._pending
        return arrays

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore :meth:`export_state`'s arrays."""
        if arrays:
            self._weights = arrays["weights"]
            self._pending = arrays.get("pending_weights")

    # ------------------------------------------------------------------ the step

    def step(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        *,
        damping: float = 0.0,
        reference_prediction: NDArray[np.float64] | None = None,
        rng: np.random.Generator | None = None,
        prior_state: NDArray[np.float64] | None = None,
        step_length: float = 1.0,
    ) -> NDArray[np.float64]:
        """Return the increment from ``state`` to the trial ``(n_state, N)``.

        The trial is ``X_p (I + (W + step_length dW) / sqrt(N - 1))`` (module docstring);
        its weights are pending until :meth:`accept_trial`.

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``damping`` is negative, ``step_length`` is not
            positive, the strategy was not prepared, ``prior_state`` is missing, or the
            predicted data have no spread.
        """
        if reference_prediction is not None:
            raise ValueError("SubspaceUpdate centres data anomalies on the ensemble mean")
        check_shapes(state, predicted, perturbed_observations, data_error)
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if not step_length > 0:
            raise ValueError(f"step_length must be positive, got {step_length}")
        if prior_state is None or prior_state.shape != state.shape:
            raise ValueError("SubspaceUpdate needs each member's prior state, shaped as the state")
        n = state.shape[-1]
        weights = self._weights
        if weights is None or weights.shape != (n, n):
            raise ValueError("SubspaceUpdate is not prepared for this ensemble: call prepare()")

        # Pi = (I - 1 1^T / N) / sqrt(N - 1): anomalies(A) = A Pi.
        projection = (np.eye(n) - np.full((n, n), 1.0 / n)) / np.sqrt(n - 1)
        y = data_error.whiten(anomalies(predicted))  # L^-1 g(X) Pi, (n_data, N)
        omega = np.eye(n) + weights @ projection  # (N, N)
        s_matrix = np.linalg.solve(omega.T, y.T).T  # S = Y Omega^-1, (n_data, N)
        residuals = data_error.whiten(predicted - perturbed_observations)  # R, (n_data, N)
        e = data_error.whiten(anomalies(perturbed_observations))  # E, (n_data, N)

        u, s, _ = truncated_svd(s_matrix, self.energy)
        # Drop numerically null singular values: their inverse would be rounding noise.
        keep = s > np.finfo(np.float64).eps * max(s_matrix.shape) * s[0]
        u, s = u[:, keep], s[keep]
        if s.size == 0:
            raise ValueError("the predicted data have no spread: the subspace step is undefined")
        self._diagnostics.add("strategy", "singular_values", s)

        x = (u.T @ e) / s[:, None]  # s^-1 U^T E, (r, N)
        eigenvalues, q = np.linalg.eigh(x @ x.T)  # symmetric: eigh
        x2 = (u / s[None, :]) @ q  # U s^-1 Q, (n_data, r)
        x3 = s_matrix.T @ x2  # S^T X2, (N, r)
        denominator = (1.0 + (1.0 + damping) * eigenvalues)[:, None]  # I + (1 + lam) Lambda
        delta_m = x3 @ ((x3.T @ weights) / denominator)
        delta_d = x3 @ ((x2.T @ residuals) / denominator)
        weight_step = -(weights + delta_d - delta_m) / (1.0 + damping)

        pending = weights + step_length * weight_step
        self._pending = pending
        self._diagnostics.add("strategy", "weights", pending)
        # X_p (I + W / sqrt(N - 1)) - X
        trial = prior_state + prior_state @ (pending / np.sqrt(n - 1))
        increment: NDArray[np.float64] = trial - state
        return increment
