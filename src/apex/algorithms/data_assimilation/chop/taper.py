"""CHOP's per-member tapers on a shared gain.

:class:`ChopTaper` is the model update's localization: member ``j``'s increment is

``dm_j = (F_j o K) r_j``, ``K = dX G`` ``(n_state, n_data)``,

with ``F_j`` the product of the components' factors at member ``j``'s hyper-parameters,
``G`` the strategy's gain factor ``(n_ensemble, n_data)`` and ``r_j`` member ``j``'s
whitened innovation. One gain for every member, one taper per member: the transpose of
:class:`~apex.algorithms.data_assimilation.localization.MemberLocalization` (one taper,
a gain per member). ``K`` is formed in row blocks within ``max_batch_mb``.

:class:`HyperTaper` is the hyper-parameter update's localization (``same``): each
component tapers its own rows of ``K_theta`` at member ``j``'s hyper-parameters, from the
correlations of the hyper-parameters with the new forecast.

Shapes: ensemble last; ``n_theta`` the stacked hyper-parameters per member.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.chop.components import ChopComponent, ComponentContext
from apex.core.diagnostics import EmitsDiagnostics
from apex.domain.observations import DataKey


def _normalised(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rows centred and scaled as ``localization.correlation`` does (std floored at 1e-6)."""
    std = np.maximum(m.std(axis=-1), 1e-6)
    result: NDArray[np.float64] = (m - m.mean(axis=-1, keepdims=True)) / std[:, None]
    return np.ascontiguousarray(result)


class ChopTaper(EmitsDiagnostics):
    """The model update's per-member Schur factors (see the module docstring).

    Parameters
    ----------
    components
        The estimated components, in their stacked order.
    max_batch_mb
        The memory of one row block of the gain, its correlations and factors.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {}

    def __init__(self, components: Sequence[ChopComponent], max_batch_mb: float = 256.0) -> None:
        if max_batch_mb <= 0:
            raise ValueError(f"max_batch_mb must be positive, got {max_batch_mb}")
        self.components = list(components)
        self.max_batch_mb = max_batch_mb
        self.context: ComponentContext | None = None
        self.slices: list[slice] = []
        #: Every member's natural hyper-parameters, ``(n_theta, n_ensemble)``.
        self.theta: NDArray[np.float64] | None = None
        self._prior: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        self._current: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    @property
    def wants(self) -> set[str]:
        """The correlations the components need: ``prior`` and/or ``current``."""
        return {
            str(getattr(c, "correlation", "")) for c in self.components if c.needs_correlation
        } - {""}

    def configure(self, context: ComponentContext, slices: Sequence[slice]) -> None:
        """Set the run's sizes and each component's rows of the stacked vector."""
        self.context, self.slices = context, list(slices)

    # ------------------------------------------------------------------ correlations

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Keep the initial ensemble's correlation factors (``correlation = "prior"``).

        rho^0 belongs to the initial ensemble: members that fail later do not change it.
        """
        if "prior" in self.wants:
            self._prior = (_normalised(prior_state), _normalised(prior_predicted))

    def set_current(self, state: NDArray[np.float64], predicted: NDArray[np.float64]) -> None:
        """Use these correlations for ``correlation = "current"`` (before each update)."""
        if "current" in self.wants:
            self._current = (_normalised(state), _normalised(predicted))

    def _correlation(self, which: str, rows: slice) -> NDArray[np.float64]:
        source = self._prior if which == "prior" else self._current
        if source is None:
            raise RuntimeError(f"CHOP's {which} correlations were not set before the update")
        state, data = source
        corr: NDArray[np.float64] = state[rows] @ data.T / state.shape[1]
        return corr

    # ------------------------------------------------------------------ the update

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``dm_j = (F_j o (dX G)) r_j`` for every member, ``(n_state, n_ensemble)``."""
        context, theta = self.context, self.theta
        if context is None or theta is None:
            raise RuntimeError("CHOP's taper has no hyper-parameters (configure it first)")
        n_state, n_ensemble = state_anomalies.shape
        if theta.shape[1] != n_ensemble or whitened_innovations.shape[1] != n_ensemble:
            raise ValueError(
                f"hyper-parameters for {theta.shape[1]} members, innovations for "
                f"{whitened_innovations.shape[1]}, anomalies for {n_ensemble}"
            )
        n_data = gain_factor.shape[1]
        # A block holds K, the correlations and the product of the factors.
        rows_per_block = max(1, int(self.max_batch_mb * 2**20 // (3 * 8 * max(n_data, 1))))
        step = np.empty((n_state, n_ensemble))
        for start in range(0, n_state, rows_per_block):
            rows = slice(start, min(start + rows_per_block, n_state))
            gain = state_anomalies[rows] @ gain_factor  # K rows, (rows, n_data)
            correlations = {w: self._correlation(w, rows) for w in self.wants}
            for j in range(n_ensemble):
                factor: NDArray[np.float64] = np.ones((1, 1))
                for component, part in zip(self.components, self.slices, strict=True):
                    which = str(getattr(component, "correlation", ""))
                    factor = factor * component.factor(
                        theta[part, j], rows, context, correlations.get(which)
                    )
                step[rows, j] = (factor * gain) @ whitened_innovations[:, j]
        return step

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the initial ensemble's correlation factors (``prior``), for resume."""
        if self._prior is None:
            return {}
        return {"chop_prior_state": self._prior[0], "chop_prior_data": self._prior[1]}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore :meth:`export_state`'s arrays."""
        if "chop_prior_state" in arrays:
            self._prior = (
                np.ascontiguousarray(arrays["chop_prior_state"]),
                np.ascontiguousarray(arrays["chop_prior_data"]),
            )


class HyperTaper(EmitsDiagnostics):
    """The hyper-parameter update's per-member taper (``hyper_localization = "same"``).

    Before each hyper-parameter update, :attr:`theta` holds the members' natural values
    (``l^i``) and :attr:`correlation` the correlations of the updated hyper-parameters with
    the new forecast, ``(n_theta, n_data)``.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {}

    def __init__(self, components: Sequence[ChopComponent]) -> None:
        self.components = list(components)
        self.context: ComponentContext | None = None
        self.slices: list[slice] = []
        self.theta: NDArray[np.float64] | None = None
        self.correlation: NDArray[np.float64] | None = None

    def configure(self, context: ComponentContext, slices: Sequence[slice]) -> None:
        """Set the run's sizes and each component's rows."""
        self.context, self.slices = context, list(slices)

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Nothing to set up: the taper follows the hyper-parameters."""

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``dtheta_j = (T_theta(theta_j) o (dTheta G)) r_j``, ``(n_theta, N)``."""
        context, theta, correlation = self.context, self.theta, self.correlation
        if context is None or theta is None or correlation is None:
            raise RuntimeError("CHOP's hyper taper was not set before the update")
        gain = state_anomalies @ gain_factor  # K_theta, (n_theta, n_data)
        step = np.empty_like(state_anomalies)
        for j in range(state_anomalies.shape[1]):
            taper = np.ones_like(gain)
            for component, part in zip(self.components, self.slices, strict=True):
                taper[part] = component.hyper_taper(
                    theta[part, j], correlation[part], context, state_anomalies.shape[1]
                )
            step[:, j] = (taper * gain) @ whitened_innovations[:, j]
        return step

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return nothing: the taper is rebuilt from the hyper-parameters."""
        return {}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Nothing to restore."""
