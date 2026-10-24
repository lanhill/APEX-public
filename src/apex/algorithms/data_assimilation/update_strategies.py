"""Update strategies: how an analysis inverts the ensemble-approximated sensitivity.

Ported from PET
---------------
PET module : src/pipt/update_schemes/analysis/approx.py (approx_update.update)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y. & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
             ensemble smoother for efficient history matching and uncertainty
             quantification. Computational Geosciences, 17(4), 689-703.
Localized : PET-fork (not public) 1348861b08314b8b3f3e1ccf886c7c095273e202,
             pipt/update_schemes/update_methods_ns/approx_update.py (``autoadaloc`` branch);
             upstream's ``autoadaloc`` branch with ``projection = "rank-r"`` (e65fb10) for a
             projected localization (``adaptive_projected``).
Differences: the analytic-covariance path is ported, unlocalized (upstream) and localized
             (PET-fork), with ``emp_cov`` (``empirical_data_error``) and the adjoint
             branch (``sensitivities``, not localized). Truncation follows
             upstream's rule (keep ranks up to the first reaching ``energy``) also when
             localized; PET-fork keeps one rank fewer (``cumsum(s)/sum(s) <= energy``),
             so localized parity is tested at ``energy = 1``. PET's state scaling is
             omitted: it divides the state anomalies by the prior standard deviation and
             multiplies the step back, which cancels exactly when every prior variance is
             positive (a zero variance gives NaN in PET, a finite step here). Whitening uses
             the lower Cholesky factor (see ``data_error``).
"""

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.localization import Localization, ProjectedLocalization
from apex.algorithms.data_assimilation.registries import update_strategies
from apex.core.diagnostics import EmitsDiagnostics


@update_strategies.register("approx")
class ApproxUpdate(EmitsDiagnostics):
    """Approximate (TSVD) Levenberg-Marquardt update, the ES/ES-MDA analysis at ``damping=0``.

    With whitened predicted-data anomalies ``dY_w = L^-1 dY = U S V^T`` (truncated SVD),

    ``step = dX V S (I (1 + lambda) + S^2)^-1 U^T L^-1 (D - Y)``

    which for ``lambda = 0`` and no truncation equals ``C_xy (C_yy + C_d)^-1 (D - Y)``.

    With a ``localization``, the step is ``localization.localized_step(dX, G, r)`` with the
    gain factor ``G = V S (I (1 + lambda) + S^2)^-1 U^T`` and ``r = L^-1 (D - Y)``: PET-fork's
    ``approx_update`` ``autoadaloc`` branch (``1348861``), which gives ``auto_ada_loc`` the
    physical anomalies and ``G r``. PET-fork's approx update has no prior-only variant;
    with ``adaptive_prior`` the prior taper multiplies ``dX G`` as in its RLM-MAC
    ``ini_only`` branch (an APEX extension). A projected localization
    (``adaptive_projected``, upstream PET's ``autoadaloc`` with ``projection = "rank-r"``)
    gets ``S V^T`` and ``X2 = (I (1 + lambda) + S^2)^-1 U^T L^-1 (D - Y)`` instead and tapers
    in that ``r``-dimensional space.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum of ``dY_w`` to retain, in ``(0, 1]``
        (PET keyword ``energy``, default 0.98).
    localization
        Optional localization plugin (``adaptive``, prior-only, the default; its alias
        ``adaptive_prior``; ``adaptive_per_update``; ``distance``; ``adaptive_projected``).
    empirical_data_error
        PET's ``emp_cov``: use the perturbations' own covariance ``E E^T`` (``E = L^-1 D Pi``)
        for ``C_d`` in the inversion, in the subspace of ``U``: with ``X0 = s^-1 U^T E``,
        ``X0 X0^T = Q Lambda Q^T`` and ``d = (1 + lambda) Lambda + 1``, the step is
        ``dX V Q d^-1 Q^T s^-1 U^T L^-1 (D - Y)`` (PET ``approx_update``, ``e65fb10``). Its
        numerically null singular values are dropped (they would be divided by). PET's
        distance-localized branch multiplies the anomalies by ``sqrt(N - 1)`` here, unlike its
        unlocalized step; APEX uses the same gain factor for both.
    sensitivities
        PET's adjoint branch: the data anomalies are the simulator's member-mean
        sensitivity times the state anomalies, ``dY = mean_j(dg/dm)_j dX``, in place of the
        forecast's own. The method requests the sensitivity with each forecast and passes it
        as ``sensitivity`` ``(n_data, n_state)``. Not with localization.
    """

    #: Centres on the ensemble mean, so no extra forecast is needed.
    requires_reference_prediction = False
    relative_damping = False

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "singular_values": "retained singular values s of L^-1 dY, (r,)",
        "gain": (
            "Kalman-gain-like dX G acting on whitened data L^-1 d, (n_state, n_data); "
            "without localization's taper (with one, the step is not gain @ L^-1 d); "
            "dense, formed only when requested"
        ),
    }

    def __init__(
        self,
        energy: float = 0.98,
        localization: Localization | None = None,
        empirical_data_error: bool = False,
        sensitivities: bool = False,
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        if sensitivities and localization is not None:
            raise ValueError("the adjoint (sensitivity) update does not take a localization")
        self.energy = energy
        self.localization = localization
        self.empirical_data_error = bool(empirical_data_error)
        self.sensitivities = bool(sensitivities)

    @property
    def uses_sensitivities(self) -> bool:
        """Whether each step needs the simulator's sensitivity."""
        return self.sensitivities

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
        sensitivity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Return the state increment, shape ``(n_state, n_ensemble)``.

        See :class:`~apex.algorithms.data_assimilation.protocols.UpdateStrategy` for the
        parameters; ``sensitivity`` ``(n_data, n_state)`` is the member-mean ``dg/dm`` the
        adjoint update (``sensitivities``) needs.

        Raises
        ------
        ValueError
            If the array shapes are inconsistent, ``damping`` is negative, or a
            ``reference_prediction`` is given (this strategy centres on the ensemble mean).
        """
        if reference_prediction is not None:
            raise ValueError(
                "ApproxUpdate centres data anomalies on the ensemble mean; "
                "use RLMMACUpdate to centre on a reference prediction"
            )
        check_shapes(state, predicted, perturbed_observations, data_error)
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")

        state_anomalies = anomalies(state)  # dX, (n_state, n_ensemble)
        if self.sensitivities:
            if sensitivity is None or sensitivity.shape != (predicted.shape[0], state.shape[0]):
                raise ValueError(
                    "the adjoint update needs sensitivity, (n_data, n_state) = "
                    f"({predicted.shape[0]}, {state.shape[0]})"
                )
            # PET: Y = mean(enAdj) @ enX @ PI, the linearised data anomalies.
            data_anomalies = sensitivity @ state_anomalies
        else:
            data_anomalies = anomalies(predicted)
        whitened_anomalies = data_error.whiten(data_anomalies)  # L^-1 dY, (n_data, n_ensemble)
        whitened_innovations = data_error.whiten(perturbed_observations - predicted)  # L^-1 (D - Y)

        u, s, vt = truncated_svd(whitened_anomalies, self.energy)  # (n_data, r), (r,), (r, n_ens)
        self._diagnostics.add("strategy", "singular_values", s)
        if self.empirical_data_error:
            return self._empirical_step(
                state_anomalies, u, s, vt, perturbed_observations, data_error,
                whitened_innovations, damping, rng,
            )  # fmt: skip
        if self._diagnostics.wants("strategy", "gain"):
            self._diagnostics.add(
                "strategy",
                "gain",
                lambda: state_anomalies @ (((vt.T * s) / (1.0 + damping + s**2)) @ u.T),
                nbytes=state.shape[0] * u.shape[0] * 8,
            )
        if isinstance(self.localization, ProjectedLocalization):
            if rng is None:
                raise ValueError("a localized update needs rng (the localization stream)")
            # PET (rank-r): Y_anom_proj = Sr VrT, X2 = (1 + lambda + Sr^2)^-1 Ur^T D_anom.
            projected = (u.T @ whitened_innovations) / (1.0 + damping + s**2)[:, None]
            return self.localization.projected_step(
                state_anomalies, s[:, None] * vt, projected, rng
            )
        if self.localization is not None:
            if rng is None:
                raise ValueError("a localized update needs rng (the localization stream)")
            # G = V S (I (1 + lambda) + S^2)^-1 U^T, (n_ens, n_data).
            gain_factor = ((vt.T * s) / (1.0 + damping + s**2)) @ u.T
            return self.localization.localized_step(
                state_anomalies, gain_factor, whitened_innovations, rng
            )
        # (I (1 + lambda) + S^2)^-1 U^T L^-1 (D - Y): diagonal, so a row scaling.
        projected = (u.T @ whitened_innovations) / (1.0 + damping + s**2)[:, None]  # (r, n_ens)
        # dX V S: scale the columns of V by S instead of forming diag(S).
        return state_anomalies @ ((vt.T * s) @ projected)

    def _empirical_step(
        self,
        state_anomalies: NDArray[np.float64],
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vt: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        whitened_innovations: NDArray[np.float64],
        damping: float,
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return the ``emp_cov`` step (see ``empirical_data_error``)."""
        perturbations = data_error.whiten(anomalies(perturbed_observations))  # E
        if isinstance(self.localization, ProjectedLocalization):
            if rng is None:
                raise ValueError("a localized update needs rng (the localization stream)")
            s, vt, inner = _empirical_projection(u, s, vt, perturbations, damping)
            # PET's emp_cov X2 with the rank-r projection S V^T.
            return self.localization.projected_step(
                state_anomalies, s[:, None] * vt, inner @ whitened_innovations, rng
            )
        factor = _empirical_gain_factor(u, s, vt, perturbations, damping)  # (N, n_data)
        if self._diagnostics.wants("strategy", "gain"):
            self._diagnostics.add(
                "strategy", "gain", lambda: state_anomalies @ factor,
                nbytes=state_anomalies.shape[0] * u.shape[0] * 8,
            )  # fmt: skip
        if self.localization is not None:
            if rng is None:
                raise ValueError("a localized update needs rng (the localization stream)")
            return self.localization.localized_step(
                state_anomalies, factor, whitened_innovations, rng
            )
        step: NDArray[np.float64] = state_anomalies @ (factor @ whitened_innovations)
        return step


def _empirical_gain_factor(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vt: NDArray[np.float64],
    perturbations: NDArray[np.float64],
    damping: float,
) -> NDArray[np.float64]:
    """PET's ``emp_cov`` gain factor ``V Q d^-1 Q^T s^-1 U^T`` ``(n_ensemble, n_data)``.

    ``perturbations`` is ``E = L^-1 D Pi`` ``(n_data, n_ensemble)``.
    """
    keep = s > np.finfo(np.float64).eps * max(u.shape[0], vt.shape[1]) * s[0]
    u, s, vt = u[:, keep], s[keep], vt[keep]
    x0 = (u.T @ perturbations) / s[:, None]  # s^-1 U^T E, (r, N)
    eigenvalues, q = np.linalg.eigh(x0 @ x0.T)  # symmetric: eigh
    d = (1.0 + damping) * eigenvalues + 1.0
    factor: NDArray[np.float64] = (vt.T @ q) @ ((q.T @ (u.T / s[:, None])) / d[:, None])
    return factor


def _empirical_projection(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vt: NDArray[np.float64],
    perturbations: NDArray[np.float64],
    damping: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return the kept ``s``, ``V^T`` and PET's ``emp_cov`` ``X2`` operator ``(r, n_data)``.

    ``X2 = s^-1 Q d^-1 Q^T s^-1 U^T L^-1 (D - Y)`` (PET ``approx_update``), so that
    ``(V S) X2`` is the unlocalized ``emp_cov`` step's factor (``_empirical_gain_factor``).
    """
    keep = s > np.finfo(np.float64).eps * max(u.shape[0], vt.shape[1]) * s[0]
    u, s, vt = u[:, keep], s[keep], vt[keep]
    x0 = (u.T @ perturbations) / s[:, None]  # s^-1 U^T E, (r, N)
    eigenvalues, q = np.linalg.eigh(x0 @ x0.T)
    d = (1.0 + damping) * eigenvalues + 1.0
    inner: NDArray[np.float64] = (q @ ((q.T @ (u.T / s[:, None])) / d[:, None])) / s[:, None]
    return s, np.ascontiguousarray(vt), inner


def check_shapes(
    state: NDArray[np.float64],
    predicted: NDArray[np.float64],
    perturbed_observations: NDArray[np.float64],
    data_error: DataErrorCovariance,
) -> None:
    """Validate the shapes and finiteness of an analysis's inputs.

    Raises
    ------
    ValueError
        If the arrays are not 2-D, disagree in ensemble size or data rows, or contain
        non-finite values.
    """
    if state.ndim != 2 or predicted.ndim != 2:
        raise ValueError(
            f"state and predicted must be 2-D, got {state.shape} and {predicted.shape}"
        )
    if state.shape[-1] != predicted.shape[-1]:
        raise ValueError(
            f"ensemble sizes differ: state has {state.shape[-1]}, "
            f"predicted has {predicted.shape[-1]} members"
        )
    if perturbed_observations.shape != predicted.shape:
        raise ValueError(
            f"perturbed_observations shape {perturbed_observations.shape} "
            f"must equal predicted shape {predicted.shape}"
        )
    for name, array in (
        ("state", state),
        ("predicted", predicted),
        ("perturbed_observations", perturbed_observations),
    ):
        # A failed forward run typically shows up as NaN in `predicted`; name it here
        # rather than let the SVD fail to converge.
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
    if predicted.shape[0] != data_error.n_data:
        raise ValueError(
            f"predicted has {predicted.shape[0]} data rows but data_error has {data_error.n_data}"
        )
