"""The full Levenberg-Marquardt update: the approx step plus the prior-increment term.

Ported from PET
---------------
PET module : src/pipt/update_schemes/analysis/full.py (full_update.update, ext_Am),
             src/pipt/misc_tools/analysis_tools.py (calc_scaling)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y. & Oliver, D. S. (2013). Levenberg-Marquardt forms of the iterative
             ensemble smoother for efficient history matching and uncertainty
             quantification. Computational Geosciences, 17(4), 689-703.
Differences: none in the step (the parity test pins it at PET's state scaling). PET's
             ``state_scaling`` is the square root of each variable's *configured* prior
             variance (``calc_scaling``); APEX priors need not declare one (a ``file`` prior
             does not), so the default is each variable's prior-ensemble standard deviation
             (``sqrt`` of its mean per-cell sample variance). Only the ratios between the
             variables' scalings matter (a common factor cancels), and those differ from
             PET's for clipped priors, per-layer variances or ``file`` priors whose spread
             is not the declared variance: ``state_scaling`` or ``state_scaling_file`` give
             PET's (``apex pet-import`` writes it). ``Am`` and the default scaling come from
             the members that survived the prior forecast; PET uses the whole prior. No
             localization (PET has none for this update), ``emp_cov`` or adjoint branches.
             Diagonal ``C_d`` only for parity: PET whitens with the upper Cholesky factor
             (as for ES-MDA), so a correlated ``C_d`` diverges.

With the state scaling ``S`` (``scx``), scaled anomalies ``X = S^-1 dX`` (``dX`` the
``1/sqrt(N-1)`` state anomalies), whitened data anomalies ``L^-1 dY = U s V^T`` (truncated
SVD) and ``D = L^-1 (d - g(m))`` against the fixed perturbed observations, the step of
member ``j`` is ``dm1 + dm2``:

- ``dm1 = S X V s (1 + lambda + s^2)^-1 U^T D``: the approx update (the data term);
- ``dm2 = -S X V (1 + lambda + s^2)^-1 V^T X^T Am Am^T S^-1 (m - m_prior)``: the pull of
  each member toward its own prior realisation, with ``Am = U_p s_p^-1`` from the truncated
  SVD of the scaled *prior* anomalies, so that ``Am Am^T`` approximates the inverse of the
  scaled prior covariance. ``Am`` is built once, from the prior (PET's ``ext_Am``).

Unlike the approx update, the state scaling does not cancel entirely: the truncated
pseudo-inverse ``Am`` depends on the ratios between the variables' scalings.
"""

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.registries import update_strategies
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.options import InputFile


@update_strategies.register("full")
class FullUpdate(EmitsDiagnostics):
    """Full LM update with the prior-increment term; needs each member's prior state.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum kept, for both the data anomalies and the prior
        anomalies of ``Am``, in ``(0, 1]`` (PET ``energy``). The default, 0.98, is the
        approx update's; PET's LM-EnRML uses 0.95, which ``apex pet-import`` passes.
    block_sizes
        Sizes of the state variables, in state order (the composition passes them), for
        the per-variable default scaling; one block without them.
    state_scaling, state_scaling_file
        Explicit scaling per state row ``(n_state,)`` (PET's ``state_scaling``), or an
        ``.npz`` holding it as ``scaling`` (``apex pet-import`` writes one), in place of the
        prior-ensemble standard deviation per variable.
    """

    #: Centres on the ensemble mean: no mean-model forecast.
    requires_reference_prediction = False
    relative_damping = False
    #: The loop passes each member's prior state (``prior_state``) and calls :meth:`prepare`.
    requires_prior_state = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "singular_values": "retained singular values s of L^-1 dY, (r,)",
        "prior_term": "the prior-increment part dm2 of the step, (n_state, N)",
    }

    def __init__(
        self,
        energy: float = 0.98,
        block_sizes: Sequence[int] | None = None,
        state_scaling: ArrayLike | None = None,
        state_scaling_file: InputFile | None = None,
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        self.energy = energy
        self.block_sizes = None if block_sizes is None else tuple(int(b) for b in block_sizes)
        if state_scaling is not None and state_scaling_file is not None:
            raise ValueError("give state_scaling or state_scaling_file, not both")
        if state_scaling_file is not None:
            with np.load(Path(state_scaling_file)) as data:
                state_scaling = data["scaling"]
        self.state_scaling = (
            None if state_scaling is None else np.asarray(state_scaling, dtype=np.float64)
        )
        self._scaling: NDArray[np.float64] | None = None
        self._am: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ preparation

    def prepare(self, prior_state: NDArray[np.float64]) -> None:
        """Form the scaling and ``Am`` from the prior ensemble ``(n_state, N)``, once.

        Raises
        ------
        ValueError
            If the scaling has the wrong length or is not positive.
        """
        n_state = prior_state.shape[0]
        if self.state_scaling is not None:
            scaling = self.state_scaling
        else:
            sizes = self.block_sizes or (n_state,)
            if sum(sizes) != n_state:
                raise ValueError(f"block_sizes {sizes} do not add up to {n_state} states")
            variances = np.var(prior_state, axis=-1, ddof=1)
            edges = np.cumsum((0, *sizes))
            scaling = np.concatenate(
                [
                    np.full(b - a, np.sqrt(variances[a:b].mean()))
                    for a, b in itertools.pairwise(edges)
                ]
            )
        if scaling.shape != (n_state,) or not np.all(np.isfinite(scaling) & (scaling > 0)):
            raise ValueError(f"state scaling must be ({n_state},) and positive")
        # Am = U_p s_p^-1 of the scaled prior anomalies (PET ext_Am, the same truncation).
        u, s, _ = truncated_svd(anomalies(prior_state) / scaling[:, None], self.energy)
        self._scaling, self._am = scaling, u / s[None, :]

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the prepared arrays, for resume."""
        if self._scaling is None or self._am is None:
            return {}
        return {"scaling": self._scaling, "am": self._am}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore :meth:`export_state`'s arrays."""
        if arrays:
            self._scaling, self._am = arrays["scaling"], arrays["am"]

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
    ) -> NDArray[np.float64]:
        """Return the state increment ``(n_state, N)`` (see the module docstring).

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``damping`` is negative, the strategy was not
            prepared, or ``prior_state`` is missing.
        """
        if reference_prediction is not None:
            raise ValueError("FullUpdate centres data anomalies on the ensemble mean")
        check_shapes(state, predicted, perturbed_observations, data_error)
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if self._scaling is None or self._am is None:
            raise ValueError("FullUpdate is not prepared: call prepare(prior_state) first")
        if prior_state is None or prior_state.shape != state.shape:
            raise ValueError("FullUpdate needs each member's prior state, shaped as the state")
        scaling, am = self._scaling, self._am

        state_anomalies = anomalies(state)  # S X: physical dX, (n_state, N)
        scaled = state_anomalies / scaling[:, None]  # X, (n_state, N)
        u, s, vt = truncated_svd(data_error.whiten(anomalies(predicted)), self.energy)
        self._diagnostics.add("strategy", "singular_values", s)
        denominator = (1.0 + damping + s**2)[:, None]  # (r, 1)

        # dm1 = S X V s (1 + lambda + s^2)^-1 U^T L^-1 (d - g(m))
        innovations = data_error.whiten(perturbed_observations - predicted)
        data_term = state_anomalies @ ((vt.T * s) @ ((u.T @ innovations) / denominator))
        # dm2 = -S X V (1 + lambda + s^2)^-1 V^T X^T Am Am^T S^-1 (m - m_prior)
        pulled = am @ (am.T @ ((state - prior_state) / scaling[:, None]))  # (n_state, N)
        prior_term = -state_anomalies @ (vt.T @ ((vt @ (scaled.T @ pulled)) / denominator))
        self._diagnostics.add("strategy", "prior_term", prior_term)
        step: NDArray[np.float64] = data_term + prior_term
        return step
