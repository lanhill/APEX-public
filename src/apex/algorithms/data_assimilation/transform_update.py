"""Ensemble-transform updates: Gauss-Newton on the ``N x N`` transform ``W`` (``W_0 = I``).

Ported from PET
---------------
PET module : src/pipt/update_schemes/analysis/subspace2.py (subspace2_update),
             src/pipt/update_schemes/analysis/margis.py (margIS_update),
             src/pipt/update_schemes/core/scheme_base.py (propose_state, matrix branch),
             src/pipt/update_schemes/enrml.py (score_and_commit: ``W`` committed on
             acceptance)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Raanes, P. N., Stordal, A. S. & Evensen, G. (2019). Revising the stochastic
             iterative ensemble smoother. Nonlinear Processes in Geophysics, 26(3),
             325-338 (``subspace2``). Stordal, A. S., Lorentzen, R. J. & Fossum, K.
             (2023). Marginalized iterative ensemble smoothers for data assimilation.
             Computational Geosciences, 27, 975-986 (``margis``).
Differences: none in the steps (the parity tests pin two iterations of each). PET's
             margIS caches the whitened perturbed observations at the first iteration;
             APEX whitens them at every step, which is the same for the fixed
             perturbations of LM-/GN-EnRML. As for the subspace update: members can be
             dropped only in the prior forecast (``W`` couples every member), bounds
             clip a trial without changing ``W``, and ES/ES-MDA refuse these strategies
             (PET's ES-MDA re-imposes the prior with them). A correlated ``C_d``: PET
             whitens with the upper Cholesky factor's inverse (the ES-MDA bug), which
             enters here as ``C_d^-1`` replaced by ``(U U^T)^-1``, so the steps differ;
             parity is for a diagonal ``C_d`` (the only one a run builds today; with
             correlations across types, whitening would also mix margis's types).

Each member is ``x_j = mean(X_p) + A_p w_j`` with ``A_p = X_p - mean(X_p)`` the prior
anomalies and ``w_j`` the ``j``-th column of ``W``. With the whitened predictions
``sY = L^-1 g(X)`` and perturbed observations ``D = L^-1 D_obs``, the linearised
sensitivity in transform space is ``Y = (sY W^-1)(I - 1 1^T / N)``, and one Gauss-Newton
step on ``J(W) = (N - 1)/2 ||W - I||_F^2 + 1/2 ||D - sY||^2`` is

    ``dW = (Y^T R Y + (N - 1) I)^-1 [(N - 1)(I - W) + Y^T R (D - sY)] / (1 + lambda)``.

``R`` weights the data: the identity for ``subspace2``; for ``margis`` it is, per data type
``k`` with ``M_k`` data, ``r_k = (M_k + nu) / (chi_k + nu s^2)``: the data-error scale is
marginalised over an inverse-chi-squared prior (Stordal et al., Eq. 37), and ``nu -> inf``
with ``s = 1`` gives ``subspace2``. ``chi_k`` is the *ensemble mean* of the members' sums of
the type's squared whitened residuals, as in PET: the exact gradient of the marginalised
likelihood weights each member by its own ``chi_kj``; the mean keeps one Hessian for the
whole transform.

``lambda`` (LM-EnRML) only divides the step here, it does not regularise the Hessian as
it does for ``approx``: LM-EnRML with these strategies is GN-EnRML with ``gamma = 1 / (1 +
lambda)``. Either loop accepts a trial on the Gaussian ``C_d`` data mismatch, not on margis's
marginalised objective (as PET).
The trial is ``mean(X_p) + A_p (W + gamma dW)``, ``gamma`` being GN-EnRML's step length.
For a linear model one full step (``lambda = 0``, ``gamma = 1``) of ``subspace2`` gives each
member's RML solution with the prior's sample covariance, whatever the ensemble size.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.registries import update_strategies
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.errors import WorkflowError
from apex.domain.observations import DataKey


class _TransformUpdate(EmitsDiagnostics):
    """What ``subspace2`` and ``margis`` share: the transform, its commit and its step."""

    #: Centres on the ensemble mean: no mean-model forecast.
    requires_reference_prediction = False
    relative_damping = False
    #: Each member's prior state: the trial is a transform of the prior ensemble.
    requires_prior_state = True
    #: The step length multiplies the transform step, so the strategy applies it itself.
    applies_step_length = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "weights": "the trial's ensemble transform W, (N, N)",
        "data_weights": "the data weight of each row (margis: per type; 1 otherwise), (n_data,)",
    }

    def __init__(self) -> None:
        self._weights: NDArray[np.float64] | None = None  # committed W
        self._pending: NDArray[np.float64] | None = None  # the trial's W

    def prepare(self, prior_state: NDArray[np.float64]) -> None:
        """Start from ``W = I`` for the prior members that survived their forecast."""
        self._weights, self._pending = np.eye(prior_state.shape[-1]), None

    def accept_trial(self) -> None:
        """Commit the last trial's transform (the loop accepted it).

        Raises
        ------
        RuntimeError
            If no trial is pending.
        """
        if self._pending is None:
            raise RuntimeError(f"{type(self).__name__} has no pending trial to accept")
        self._weights, self._pending = self._pending, None

    def drop_members(self, ok: NDArray[np.bool_]) -> None:
        """Refuse to drop members once the transform exists (it couples every member).

        Raises
        ------
        WorkflowError
            If a member is dropped.
        """
        if not np.all(ok):
            raise WorkflowError(
                f"{int(np.size(ok) - np.count_nonzero(ok))} member(s) failed in an update: "
                f"the {self.name} analysis writes every member as a transform of all the prior "
                "members, so it cannot drop members after the prior forecast; rerun with "
                "retries for transient failures, or use the approx or full analysis"
            )

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the committed and pending transforms, for resume."""
        arrays = {} if self._weights is None else {"weights": self._weights}
        if self._pending is not None:
            arrays["pending_weights"] = self._pending
        return arrays

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore :meth:`export_state`'s arrays."""
        if arrays:
            self._weights = arrays["weights"]
            self._pending = arrays.get("pending_weights")

    name: ClassVar[str] = "transform"

    def _data_weights(
        self, residuals: NDArray[np.float64], sensitivity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the weight of each data row, ``(n_data,)``: 1 (``subspace2``)."""
        return np.ones(residuals.shape[0])

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

        The trial is ``mean(X_p) + A_p (W + step_length dW)`` (module docstring); its
        transform is pending until :meth:`accept_trial`.

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``damping`` is negative, ``step_length`` is not
            positive, the strategy was not prepared, or ``prior_state`` is missing.
        """
        name = type(self).__name__
        if reference_prediction is not None:
            raise ValueError(f"{name} centres data anomalies on the ensemble mean")
        check_shapes(state, predicted, perturbed_observations, data_error)
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if not step_length > 0:
            raise ValueError(f"step_length must be positive, got {step_length}")
        if prior_state is None or prior_state.shape != state.shape:
            raise ValueError(f"{name} needs each member's prior state, shaped as the state")
        n = state.shape[-1]
        weights = self._weights
        if weights is None or weights.shape != (n, n):
            raise ValueError(f"{name} is not prepared for this ensemble: call prepare()")

        whitened = data_error.whiten(predicted)  # sY, (n_data, N)
        observed = data_error.whiten(perturbed_observations)  # D, (n_data, N)
        # Y = (sY W^-1)(I - 1 1^T / N): the sensitivity seen through the transform.
        sensitivity = np.linalg.solve(weights.T, whitened.T).T
        sensitivity = sensitivity - sensitivity.mean(axis=-1, keepdims=True)
        residuals = observed - whitened  # D - sY
        rows = self._data_weights(residuals, sensitivity)
        self._diagnostics.add("strategy", "data_weights", rows)
        weighted = sensitivity * rows[:, None]  # R Y
        gradient = (n - 1) * (np.eye(n) - weights) + weighted.T @ residuals
        hessian = weighted.T @ sensitivity + (n - 1) * np.eye(n)
        transform_step = np.linalg.solve(hessian, gradient) / (1.0 + damping)

        pending = weights + step_length * transform_step
        self._pending = pending
        self._diagnostics.add("strategy", "weights", pending)
        mean = prior_state.mean(axis=-1, keepdims=True)
        trial = mean + (prior_state - mean) @ pending
        increment: NDArray[np.float64] = trial - state
        return increment


@update_strategies.register("subspace2")
class TransformUpdate(_TransformUpdate):
    """Ensemble-transform IES of Raanes et al. (2019), PET's ``subspace2``.

    It uses the analytic ``C_d`` throughout (no ``E E^T``), so there is no SVD and no
    ``energy`` option. See the module docstring.
    """

    name: ClassVar[str] = "subspace2"


@update_strategies.register("margis")
class MarginalISUpdate(_TransformUpdate):
    """Marginalised IES of Stordal et al. (2023), PET's ``margis``.

    The data-error scale of each data type is marginalised over an inverse-chi-squared
    prior with ``nu`` degrees of freedom and scale ``s``; each type's data are weighted by
    ``(M_k + nu) / (chi_k + nu s^2)`` (module docstring).

    Parameters
    ----------
    nu
        Degrees of freedom of the prior on each type's error scale; default ``N - 1``
        (PET's). Larger trusts ``C_d`` more; ``inf`` with ``s = 1`` gives ``subspace2``.
    s
        Prior scale of the whitened errors, 1 (PET's): ``C_d`` is right on average.
    """

    name: ClassVar[str] = "margis"

    def __init__(self, nu: float | None = None, s: float = 1.0) -> None:
        super().__init__()
        if nu is not None and not nu > 0:
            raise ValueError(f"nu must be positive, got {nu}")
        if not s > 0:
            raise ValueError(f"s must be positive, got {s}")
        self.nu, self.s = nu, float(s)
        self._types: list[NDArray[np.intp]] | None = None

    def use_data_keys(self, keys: Sequence[DataKey]) -> None:
        """Group the data rows by type (``DataKey.datatype``), in first-seen order."""
        rows: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            rows.setdefault(key.datatype, []).append(i)
        self._types = [np.asarray(r, dtype=np.intp) for r in rows.values()]

    def _data_weights(
        self, residuals: NDArray[np.float64], sensitivity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self._types is None or sum(t.size for t in self._types) != residuals.shape[0]:
            raise ValueError("MarginalISUpdate needs the data keys: call use_data_keys()")
        n = residuals.shape[-1]
        nu = float(n - 1) if self.nu is None else float(self.nu)
        weights = np.empty(residuals.shape[0])
        for rows in self._types:
            # chi_k: the member mean of the type's squared whitened residuals (Eq. 37).
            chi = float(np.mean(np.sum(residuals[rows] ** 2, axis=0)))
            weights[rows] = (rows.size + nu) / (chi + nu * self.s**2)
        return weights
