"""Local analysis: each group of parameters updated with only its selected data.

A new method (PET's ``localanalysis`` does not run at ``e65fb10``). References: Chen, Y. &
Oliver, D. S. (2017). Localization and regularization for iterative ensemble smoothers.
Computational Geosciences, 21, 13-30. Emerick, A. A. (2016). Analysis of the performance
of ensemble-based assimilation of production and seismic data. J. Petroleum Sci. Eng.,
139, 219-239. Luo, X. & Bhakta, T. (2020). Automatic and adaptive localization for
ensemble-based history matching. J. Petroleum Sci. Eng., 184, 106559,
doi:10.1016/j.petrol.2019.106559 (the correlation-based selection).

A selection mask ``M = (T > SUPPORT)`` ``(n_state, n_data)`` comes from a localization's
taper ``T``, built once from the prior: by default ``adaptive_prior`` with its hard taper
(datum ``j`` updates parameter ``i`` when their prior correlation exceeds the noise
level), or ``distance`` (Gaspari-Cohn or Furrer-Bengtsson; their support is twice the
localization length). An adaptive soft or sigmoid taper is refused: it is positive for
nearly every pair (the soft taper wherever ``|rho| > 2c - 1``, i.e. everywhere for a noise
level ``c < 0.5``; the sigmoid everywhere), so it selects no subset. Parameters whose
rows of ``M`` are identical form a group ``G_g`` with data ``J_g``; each group gets the
approx update restricted to its data,

    ``L_g^-1 dY[J_g] = U s V^T`` (truncated SVD),
    ``step[G_g] = dX[G_g] V s ((1 + lambda) I + s^2)^-1 U^T L_g^-1 (D - Y)[J_g]``,

with ``L_g`` the Cholesky factor of ``C_d[J_g, J_g]`` (already inflated by ES-MDA) and
``lambda`` the method's damping. Grouping is exact: rows with the same data subset get the
same local gain. Parameters that select no data are not updated. With ``weight = true``
the local increment is also multiplied by ``T`` (soft tapers act inside the local update).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.localization import (
    AdaptiveLocalization,
    Localization,
    PriorAdaptiveLocalization,
    refuse_projected,
)
from apex.algorithms.data_assimilation.registries import update_strategies
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics

logger = logging.getLogger(__name__)

#: Taper values at or below this are outside the support (Gaspari-Cohn is ~1e-15 near
#: its edge in floating point).
SUPPORT = 1e-12


def selection_groups(mask: NDArray[np.bool_]) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Group the rows of a selection mask ``(n_state, n_data)`` by their data subset.

    Returns ``[(state rows, data rows), ...]`` for every distinct non-empty row pattern, in
    order of first appearance; rows that select no data are left out.
    """
    mask = np.asarray(mask, dtype=bool)  # its support: packbits counts any nonzero as 1
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        return []
    # Rows as packed bits viewed as one opaque item each: np.unique then sorts short
    # byte strings instead of rows of n_data booleans (np.unique(axis=0)), about 90 times
    # faster on 5Spots (0.43 s of a 1 s update; 2026-10-04), same groups.
    packed = np.ascontiguousarray(np.packbits(mask, axis=1))  # (rows, ceil(n_data / 8))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, inverse = np.unique(keys, return_inverse=True)
    inverse = inverse.reshape(-1)
    # Rows of each pattern in one stable sort, O(rows log rows) (per-row adaptive selection
    # makes nearly every row its own pattern; the numerics review of 2026-10-04).
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    groups = []
    for rows in sorted(members, key=lambda r: int(r[0])):
        data = np.flatnonzero(mask[rows[0]])
        if data.size:
            groups.append((rows.astype(np.intp), data.astype(np.intp)))
    return groups


@update_strategies.register("local_analysis")
class LocalAnalysisUpdate(EmitsDiagnostics):
    """Local analysis: groups of state updated with their locally selected data.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum of each group's whitened data anomalies kept,
        in ``(0, 1]`` (as the approx update's, default 0.98).
    localization
        The data selection: a localization with a fixed taper from the prior
        (``adaptive_prior``/``adaptive``, ``distance``), from ``[method.localization]``.
        Without one, ``adaptive_prior`` with a hard taper over ``block_sizes``.
    block_sizes
        State variable sizes (the composition passes them), for the default selection.
    weight
        Also multiply each local increment by the taper ``T`` (default: use only its
        support).
    """

    #: Centres on the ensemble mean: no mean-model forecast.
    requires_reference_prediction = False
    relative_damping = False
    #: The localization selects data; it does not taper a global gain (LM-EnRML allows it).
    localization_role: ClassVar[str] = "selection"

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "groups": "number of parameter groups (distinct data subsets), ()",
        "selected": "data selected per state row, (n_state,)",
    }

    def __init__(
        self,
        energy: float = 0.98,
        localization: Localization | None = None,
        block_sizes: Sequence[int] | None = None,
        weight: bool = False,
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        refuse_projected(localization, "local_analysis")
        if isinstance(localization, AdaptiveLocalization):
            raise ValueError(
                "local analysis needs a selection fixed from the prior; adaptive_per_update "
                "changes its taper at every update: use adaptive_prior or distance"
            )
        if getattr(localization, "taper", "hard") in ("soft", "sigmoid") and isinstance(
            localization, PriorAdaptiveLocalization
        ):
            raise ValueError(
                "correlation-based selection needs the hard taper: a soft or sigmoid adaptive "
                "taper is positive for nearly every parameter-datum pair (no subset)"
            )
        if localization is None:
            if block_sizes is None:
                raise ValueError("give block_sizes (the composition does) or a localization")
            localization = PriorAdaptiveLocalization(block_sizes, taper="hard")
        self.energy = energy
        self.localization: Localization = localization
        self.weight = bool(weight)
        self._groups: list[tuple[NDArray[np.intp], NDArray[np.intp]]] | None = None
        self._groups_of: NDArray[np.float64] | None = None  # the taper they came from

    def _taper(self, n_state: int, n_data: int) -> NDArray[np.float64]:
        taper = getattr(self.localization, "taper_matrix", None)
        if taper is None:
            raise ValueError(
                "the selection's localization is not prepared (the loop prepares it from "
                "the prior), or it has no fixed taper"
            )
        if taper.shape != (n_state, n_data):
            raise ValueError(f"selection taper {taper.shape} does not match ({n_state}, {n_data})")
        result: NDArray[np.float64] = taper
        return result

    def groups(self, n_state: int, n_data: int) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Return the parameter groups ``[(state rows, data rows), ...]`` of the selection."""
        taper = self._taper(n_state, n_data)
        if self._groups is None or self._groups_of is not taper:
            self._groups, self._groups_of = selection_groups(taper > SUPPORT), taper
            if len(self._groups) == 1 and self._groups[0][1].size == n_data:
                logger.warning(
                    "local analysis: every parameter selects every datum, so the update is "
                    "the global one",
                    extra={"event": "local_analysis_global"},
                )
        return self._groups

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
    ) -> NDArray[np.float64]:
        """Return the state increment ``(n_state, N)`` (see the module docstring).

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``damping`` is negative, a
            ``reference_prediction`` is given, or the selection is not prepared.
        """
        check_shapes(state, predicted, perturbed_observations, data_error)
        data_anomalies = self._data_anomalies(predicted, reference_prediction)  # dY
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        n_state, n_data = state.shape[0], predicted.shape[0]
        groups = self.groups(n_state, n_data)
        taper = self._taper(n_state, n_data)
        self._diagnostics.add("strategy", "groups", np.array(len(groups)))
        self._diagnostics.add("strategy", "selected", (taper > SUPPORT).sum(axis=1))

        state_anomalies = anomalies(state)  # dX, (n_state, N)
        innovations = perturbed_observations - predicted  # D - Y
        step = np.zeros_like(state)
        # C_d is read once per step (a copy), not once per group; each group factors only
        # its own block.
        values, diagonal = data_error.values, data_error.is_diagonal
        for rows, data in groups:
            local_error = _restricted(values, diagonal, data)
            u, s, vt = truncated_svd(local_error.whiten(data_anomalies[data]), self.energy)
            u, w, vt = self._weights(u, s, vt, damping)  # G = V diag(w) U^T
            whitened = local_error.whiten(innovations[data])
            if self.weight:
                # T o (dX G) on the group.
                gain = state_anomalies[rows] @ ((vt.T * w) @ u.T)
                local = (taper[np.ix_(rows, data)] * gain) @ whitened
            else:
                local = state_anomalies[rows] @ (vt.T @ ((u.T @ whitened) * w[:, None]))
            step[rows] = local
        return step

    def _data_anomalies(
        self, predicted: NDArray[np.float64], reference_prediction: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        """``dY``: the forecast's ``1/sqrt(N - 1)`` anomalies about its mean (approx)."""
        if reference_prediction is not None:
            raise ValueError("LocalAnalysisUpdate centres data anomalies on the ensemble mean")
        return anomalies(predicted)

    def _weights(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vt: NDArray[np.float64],
        damping: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the local gain ``V diag(w) U^T``: ``w = s / (1 + lambda + s^2)`` (approx)."""
        return u, s / (1.0 + damping + s**2), vt


def _restricted(
    values: NDArray[np.float64], diagonal: bool, rows: NDArray[np.intp]
) -> DataErrorCovariance:
    """``C_d[J, J]`` from ``C_d``'s variances or matrix, as a diagonal or a dense covariance."""
    if diagonal:
        return DataErrorCovariance.diagonal(values[rows])
    return DataErrorCovariance.dense(values[np.ix_(rows, rows)])


@update_strategies.register("local_rlmmac")
class LocalRLMMACUpdate(LocalAnalysisUpdate):
    """Local analysis with GIES's RLM-MAC update in each group (for GIES).

    As :class:`LocalAnalysisUpdate`, with the RLM-MAC local gain: data anomalies centred on
    the mean-model forecast, ``dY = (Y - g(x_mean)) / sqrt(N - 1)``, and per group
    ``w = s / (s^2 + gamma_g)`` with ``gamma_g = lambda * mean(s^2)`` over the group's kept
    spectrum (numerically null singular values dropped), as
    :class:`~apex.algorithms.data_assimilation.gies.RLMMACUpdate` does globally. GIES's
    default ``energy`` 0.95 applies.
    """

    #: RLM-MAC centres on the forecast of the ensemble-mean state.
    requires_reference_prediction = True
    #: As RLM-MAC's: ``gamma_g = lambda * mean(s^2)``, no regularisation at ``lambda = 0``.
    relative_damping = True

    def __init__(
        self,
        energy: float = 0.95,
        localization: Localization | None = None,
        block_sizes: Sequence[int] | None = None,
        weight: bool = False,
    ) -> None:
        super().__init__(energy, localization, block_sizes, weight)

    def _data_anomalies(
        self, predicted: NDArray[np.float64], reference_prediction: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        if reference_prediction is None or reference_prediction.shape != (predicted.shape[0],):
            raise ValueError("LocalRLMMACUpdate needs reference_prediction, (n_data,)")
        if not np.all(np.isfinite(reference_prediction)):
            raise ValueError("reference_prediction contains non-finite values")
        centred: NDArray[np.float64] = (predicted - reference_prediction[:, None]) / np.sqrt(
            predicted.shape[-1] - 1
        )
        return centred

    def _weights(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vt: NDArray[np.float64],
        damping: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if s[0] == 0.0:
            return u[:, :0], s[:0], vt[:0]  # no sensitivity in this group: no update
        keep = s > np.finfo(np.float64).eps * max(u.shape[0], vt.shape[1]) * s[0]
        u, s, vt = u[:, keep], s[keep], vt[keep]
        gamma = damping * float(np.mean(s**2))
        return u, s / (s**2 + gamma), vt


@update_strategies.register("adaptive_rows")
class AdaptiveRowsUpdate(LocalAnalysisUpdate):
    """Adaptive localization per state row, recomputed at every update.

    Adapted from ERT
    ----------------
    Library   : iterative_ensemble_smoother 1.3.0 (github.com/equinor/iterative_ensemble_smoother,
                tag v1.3.0, commit e4dcca0de3f50201e8e790db8153e3e758e9dc8f; GPL-3.0),
                ``AdaptiveESMDA``: the method, not its code (APEX's grouped local update of
                does the algebra).
    Reference : Vossepoel, F. C., Evensen, G. & van Leeuwen, P. J. (2025), Monthly Weather
                Review, eq. 6 (the ``3/sqrt(N)`` threshold); Luo, X. & Bhakta, T. (2020).
                Automatic and adaptive localization for ensemble-based history matching.
                J. Petroleum Sci. Eng., 184, 106559. doi:10.1016/j.petrol.2019.106559.
    Differences: none in the update: the truncation rule is ERT's too (the fraction of
                the singular-value sum), save ERT's cap at rank N-1, which drops only a
                rounding-level value (tested against the library at several truncations).
                The default threshold is refused below 10 members, where ERT keeps nothing.

    At each update, state row ``i`` keeps the data ``j`` with ``|corr(x_i, y_j)| >
    threshold`` over the current ensemble (plain sample correlations; rows or data without
    spread keep nothing); rows with the same data subset form a group, and each group gets
    the approx update restricted to its data (see the module docstring). Rows that keep no
    data are not updated.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum of each group's whitened data anomalies kept,
        in ``(0, 1]`` (default 0.98, as ``approx``).
    threshold
        Correlation threshold in ``[0, 1)``; default ``3 / sqrt(N)`` (refused when that is
        1 or more, below 10 members).
    max_batch_mb
        Memory budget of one batch's ``(rows, n_data)`` correlations.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "groups": "number of parameter groups (distinct data subsets) at this update, ()",
        "selected": "data selected per state row at this update, (n_state,)",
    }

    def __init__(
        self, energy: float = 0.98, threshold: float | None = None, max_batch_mb: float = 256.0
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        if threshold is not None and not 0.0 <= threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {threshold}")
        if max_batch_mb <= 0:
            raise ValueError(f"max_batch_mb must be positive, got {max_batch_mb}")
        self.energy = energy
        self.threshold = threshold
        self.max_batch_mb = max_batch_mb
        self.weight = False

    # No localization (``[method.localization]`` is refused): the parent's fixed taper and
    # groups don't exist here, so their accessors fail loudly instead of reading one.
    def _taper(self, n_state: int, n_data: int) -> NDArray[np.float64]:
        raise TypeError("adaptive_rows has no taper: it selects data per row at every update")

    def groups(self, n_state: int, n_data: int) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Not fixed: the groups change at every update (see :meth:`step`).

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError("adaptive_rows forms its groups at every update, from the ensemble")

    def selection(
        self, state: NDArray[np.float64], predicted: NDArray[np.float64], rows: NDArray[np.intp]
    ) -> NDArray[np.bool_]:
        """Return the selection ``(len(rows), n_data)`` of the given state rows.

        Raises
        ------
        ValueError
            If the default threshold ``3/sqrt(N)`` keeps nothing (``N < 10``).
        """
        y = predicted - predicted.mean(axis=-1, keepdims=True)
        return self._select(state, rows, y, np.sqrt((y * y).sum(axis=-1)))

    def check_members(self, n: int) -> None:
        """Refuse an ensemble of ``n`` members the default threshold keeps no data with.

        Called when a run is validated, so the refusal comes before any simulation (a
        reported issue, #216: a 7-member study passed validation and failed at its update).

        Raises
        ------
        ValueError
            If the default threshold ``3/sqrt(n)`` is 1 or more.
        """
        self._threshold(n)

    def _threshold(self, n: int) -> float:
        """Return the correlation threshold for ``n`` members (see :meth:`selection`)."""
        threshold = self.threshold
        if threshold is None:
            threshold = 3.0 / np.sqrt(n)
            if threshold >= 1.0:
                raise ValueError(
                    f"the default threshold 3/sqrt(N) = {threshold:.3g} keeps no data with {n} "
                    "members: give a threshold below 1"
                )
        return threshold

    def _select(
        self,
        state: NDArray[np.float64],
        rows: NDArray[np.intp],
        y: NDArray[np.float64],
        sy: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Select from the data anomalies ``y`` ``(n_data, N)`` and their norms ``sy``.

        ``step`` forms ``y`` and ``sy`` once for all batches.
        """
        threshold = self._threshold(state.shape[-1])
        x = state[rows] - state[rows].mean(axis=-1, keepdims=True)
        sx = np.sqrt((x * x).sum(axis=-1))
        corr = x @ y.T  # (rows, n_data), normalised in place
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            corr /= sy
            corr /= sx[:, None]
        # No spread: no correlation, so nothing is selected (and nothing updated). That
        # includes 0/0 (NaN) and a spread whose square underflowed to 0 (±inf; review).
        corr[~np.isfinite(corr)] = 0.0
        np.abs(corr, out=corr)
        return np.asarray(corr > threshold)

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
    ) -> NDArray[np.float64]:
        """Return the state increment ``(n_state, N)``, the selection taken from this ensemble.

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``damping`` is negative, a
            ``reference_prediction`` is given, or the default threshold keeps nothing.
        """
        check_shapes(state, predicted, perturbed_observations, data_error)
        data_anomalies = self._data_anomalies(predicted, reference_prediction)  # dY
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        n_state, n_data = state.shape[0], predicted.shape[0]
        state_anomalies = anomalies(state)  # dX
        innovations = perturbed_observations - predicted  # D - Y
        values, diagonal = data_error.values, data_error.is_diagonal
        step = np.zeros_like(state)
        selected = np.zeros(n_state, dtype=np.int64)
        subsets: set[bytes] = set()
        # The correlations (rows, n_data) and about as much again in temporaries.
        batch = max(1, int(self.max_batch_mb * 2**20 / (2 * 8 * max(n_data, 1))))
        y = predicted - predicted.mean(axis=-1, keepdims=True)
        sy = np.sqrt((y * y).sum(axis=-1))
        self._threshold(state.shape[-1])  # refuse a useless default before any work
        for start in range(0, n_state, batch):
            rows = np.arange(start, min(start + batch, n_state), dtype=np.intp)
            mask = self._select(state, rows, y, sy)
            selected[rows] = mask.sum(axis=1)
            for local_rows, data in selection_groups(mask):
                subsets.add(data.tobytes())
                group = rows[local_rows]
                local_error = _restricted(values, diagonal, data)
                u, s, vt = truncated_svd(local_error.whiten(data_anomalies[data]), self.energy)
                u, w, vt = self._weights(u, s, vt, damping)
                whitened = local_error.whiten(innovations[data])
                step[group] = state_anomalies[group] @ (vt.T @ ((u.T @ whitened) * w[:, None]))
        self._diagnostics.add("strategy", "groups", np.array(len(subsets)))
        self._diagnostics.add("strategy", "selected", selected)
        return step
