"""Localization of ensemble updates: three methods behind one interface.

Ported from PET
---------------
PET module : pipt/misc_tools/cov_regularization.py (localization.auto_ada_loc,
             tapering_function, rational_function, rational_function_sigmoid,
             get_corr_mtx, _gen_loc_mask, _repos_mask), pipt/update_schemes/gies/
             rlmmac_update.py (localized branches), pipt/loop/assimilation.py (ini_only)
PET commit : 1348861b08314b8b3f3e1ccf886c7c095273e202 (not public)
Reference  : Luo, X. & Bhakta, T. (2020). Automatic and adaptive localization for
             ensemble-based history matching. J. Petroleum Sci. Eng., 184, 106559.
             doi:10.1016/j.petrol.2019.106559.
             Luo, X., Bhakta, T. & Naevdal, G. (2018). Correlation-based adaptive
             localization with applications to ensemble-based 4D-seismic history
             matching. SPE Journal, 23(2), 396-427. doi:10.2118/185936-PA.
             Gaspari, G. & Cohn, S. E. (1999), Q. J. R. Meteorol. Soc., 125, 723-757.
             Furrer, R. & Bengtsson, T. (2007), J. Multivariate Anal., 98, 227-255.
Differences: see each class; randomness comes from an explicit Generator.

Ported from PET (upstream)
--------------------------
PET module : src/pipt/localization/auto_ada_loc.py (AutoAdaptiveLocalization: corr_matrix,
             tapering_function, rational_function, rational_function_sigmoid) and its use in
             src/pipt/update_schemes/analysis/approx.py (``projection = "rank-r"``)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Luo, X. & Bhakta, T. (2020), as above.
Differences: see :class:`ProjectedAdaptiveLocalization` (``"adaptive_projected"``).

A localization turns the unlocalized increment ``dX (X r)`` of a strategy into a
localized one. ``X`` is the strategy's gain factor ``(n_ensemble, n_data)`` acting on the
whitened innovations ``r`` ``(n_data, n_ensemble)``:

- **ensemble-space** tapering (``AdaptiveLocalization``, ``"adaptive_per_update"``):
  ``(T o dX) (X r)`` with ``T`` ``(n_state, n_ensemble)`` recomputed at every update;
- **gain-space** tapering (``PriorAdaptiveLocalization``, ``"adaptive"`` = the default, alias
  ``"adaptive_prior"``; ``DistanceLocalization``, ``"distance"``): ``(T o (dX X)) r`` with a
  fixed ``T`` ``(n_state, n_data)`` set up once by ``prepare``;
- **projected-data** tapering (``ProjectedAdaptiveLocalization``, ``"adaptive_projected"``,
  upstream PET's ``autoadaloc``): the approx update's ``T o (dX (S V^T)^T)`` in the space of
  the rank-``r`` projected data ``S V^T``, recomputed at every update. Only ``ApproxUpdate``
  can apply it (:class:`ProjectedLocalization`).

Until 2026-09-28 ``"adaptive"`` named the per-update method; run snapshots written before
then are read with ``"adaptive_per_update"`` (``services.composition.snapshot_config``).
"""

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation.registries import localizations
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.options import TEXT
from apex.domain.observations import DataKey

logger = logging.getLogger(__name__)

Taper = Literal["hard", "soft", "sigmoid"]
#: Upstream PET's ``threshold`` rules (``auto_ada_loc.py``).
Threshold = Literal["adaptive", "fixed", "universal"]


class Localization(Protocol):
    """Localizes the increment of an update strategy."""

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Set up from the prior ensemble and its forecast, once, before the first update."""
        ...

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return the localized increment ``(n_state, n_ensemble)``."""
        ...

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Arrays that :meth:`prepare` computed, for checkpoints."""
        ...

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore what :meth:`export_state` returned (on resume)."""
        ...


@runtime_checkable
class MemberLocalization(Protocol):
    """A localization that can also taper per-member gains.

    Strategies whose gain differs per member (GMAC with a member-dependent metric) give
    each member's gain factor ``G_j`` ``(n_ensemble, n_data)``, one at a time, and the
    localization returns the increments ``(n_state, n_ensemble)``. With every ``G_j``
    equal it is :meth:`Localization.localized_step`.
    """

    def localized_member_steps(
        self,
        state_anomalies: NDArray[np.float64],
        member_gains: Iterable[NDArray[np.float64]],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return the localized increments ``(n_state, n_ensemble)``, column ``j`` by ``G_j``.

        It must draw every gain (the strategy records diagnostics as they are drawn) and
        raise ``ValueError`` if there are not ``n_ensemble`` of them.
        """
        ...


@runtime_checkable
class SourceLocalization(Protocol):
    """A localization with one taper per source of a constrained update.

    ``projected`` holds each source's projected innovations ``(n_ensemble, n_ensemble)``
    (data, then each constraint), column ``j`` for member ``j``.
    """

    def localized_source_steps(
        self,
        state_anomalies: NDArray[np.float64],
        projected: Sequence[NDArray[np.float64]],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``sum_s (T_s o dX) P_s`` ``(n_state, n_ensemble)``."""
        ...


# ---------------------------------------------------------------------- kernels


def correlation(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sample correlations between the rows of ``a`` ``(n_a, n)`` and ``b`` ``(n_b, n)``.

    PET's ``get_corr_mtx``: population standard deviations floored at an absolute 1e-6 (in
    the variables' units), so constant rows correlate as 0 and rows with a tiny spread have
    their correlations shrunk by ``std / 1e-6``.
    """
    n = a.shape[-1]

    def normalised(m: NDArray[np.float64]) -> NDArray[np.float64]:
        std = np.maximum(m.std(axis=-1), 1e-6)
        result: NDArray[np.float64] = (m - m.mean(axis=-1, keepdims=True)) / std[:, None]
        return result

    corr: NDArray[np.float64] = normalised(a) @ normalised(b).T / n
    return corr


def derangement(rng: np.random.Generator, n: int) -> NDArray[np.int64]:
    """Return a permutation of ``range(n)`` with no fixed points.

    Draws a permutation, then moves any member left in place: several fixed points are
    rotated among themselves (as PET does); a single one is swapped with the last (or
    first) position. PET-fork overwrites instead of swapping in that case, which can
    duplicate one member and drop another; APEX swaps.
    """
    return _fix_fixed_points(rng.permutation(n))


def _fix_fixed_points(permutation: NDArray[np.int64]) -> NDArray[np.int64]:
    perm = permutation.copy()
    n = perm.size
    fixed = np.flatnonzero(perm == np.arange(n))
    if fixed.size > 1:
        # Rotate the fixed positions' values one step left (PET's index shuffle).
        perm[fixed] = perm[np.roll(fixed, -1)]
    elif fixed.size == 1:
        position = int(fixed[0])
        other = n - 1 if position < n - 1 else 0
        perm[position], perm[other] = perm[other], position
    return perm


def gaspari_cohn(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gaspari-Cohn fifth-order piecewise rational function of ``z = distance / c``."""
    z = np.abs(z)
    y = np.zeros_like(z)
    inner, outer = z <= 1, (z > 1) & (z <= 2)
    zi, zo = z[inner], z[outer]
    y[inner] = 1 - zi**5 / 4 + zi**4 / 2 + 5 * zi**3 / 8 - 5 * zi**2 / 3
    y[outer] = zo**5 / 12 - zo**4 / 2 + 5 * zo**3 / 8 + 5 * zo**2 / 3 - 5 * zo - 2 / (3 * zo) + 4
    return y


def adaptive_taper(
    corr: NDArray[np.float64],
    corr_shuffled: NDArray[np.float64] | None,
    block_sizes: Sequence[int],
    threshold_scale: float,
    taper: Taper,
    fixed: float | None = None,
) -> NDArray[np.float64]:
    """Taper correlations against their noise level, per state block and column.

    For each state variable (row block) and column, the noise level is estimated from the
    correlations with a shuffled ensemble, ``sigma = median|rho_s| / 0.6745``, and the
    threshold is ``threshold_scale * sqrt(2 ln n_block) * sigma`` (universal threshold);
    with ``fixed`` the threshold is that value everywhere and ``corr_shuffled`` is not
    needed (e.g. ``3/sqrt(N)``: three standard deviations of a sample correlation with
    ``N`` members when there is none; Vossepoel, Evensen & van Leeuwen 2025, eq. 6, as
    ``adaptive_rows``). ``hard`` keeps correlations above it; ``soft`` applies
    Gaspari-Cohn to ``(1 - |rho|) / (1 - threshold)``; ``sigmoid`` is PET's
    ``expit(50 (|rho| - (1 - threshold_scale)))``, which does not use the universal
    threshold (with ``fixed``: ``expit(50 (|rho| - fixed))``).
    """
    if sum(block_sizes) != corr.shape[0]:
        raise ValueError(f"block sizes {list(block_sizes)} do not add up to {corr.shape[0]} rows")
    result = np.empty_like(corr)
    start = 0
    for size in block_sizes:
        rows = slice(start, start + size)
        magnitude = np.abs(corr[rows])
        if fixed is not None:
            cutoff = np.full(corr.shape[1], float(fixed))
        else:
            if corr_shuffled is None:
                raise ValueError("the universal threshold needs the shuffled correlations")
            noise = np.median(np.abs(corr_shuffled[rows]), axis=0) / 0.6745  # (n_columns,)
            cutoff = (
                threshold_scale * math.sqrt(2 * math.log(size)) * noise if size > 1 else noise * 0
            )
        # A correlation magnitude never exceeds 1, so a cutoff >= 1 means nothing in the column
        # is distinguishable from noise. PET-fork's soft taper uses |1 - cutoff| there and
        # reverses (more noise, less localization); APEX sets such columns to 0.
        hopeless = cutoff >= 1
        if np.any(hopeless):
            logger.warning(
                "adaptive localization: %d of %d columns have a noise cutoff >= 1 "
                "(ensemble too small for block of %d); their updates are zeroed",
                int(hopeless.sum()),
                hopeless.size,
                size,
            )
        if taper == "hard":
            result[rows] = (magnitude > cutoff[None, :]).astype(np.float64)
        elif taper == "soft":
            result[rows] = gaspari_cohn((1 - magnitude) / (1 - cutoff[None, :]))
        elif taper == "sigmoid":
            centre = fixed if fixed is not None else 1 - threshold_scale
            result[rows] = 1 / (1 + np.exp(-50 * (magnitude - centre)))
        else:
            raise ValueError(f"taper must be 'hard', 'soft' or 'sigmoid', got {taper!r}")
        result[rows][:, hopeless] = 0.0
        start += size
    return result


def pearson_correlation(
    a: NDArray[np.float64], b: NDArray[np.float64], eps: float = 1e-6
) -> NDArray[np.float64]:
    """Sample correlations of the rows of ``a`` ``(n_a, n)`` and ``b`` ``(n_b, n)``, ``(n_a, n_b)``.

    Upstream PET's ``corr_matrix``: ``np.corrcoef``, with the rows (of ``a``) and columns
    (of ``b``) whose population standard deviation is below ``eps`` set to 0. Unlike
    :func:`correlation`, nearly constant rows are not shrunk.
    """
    std_a, std_b = a.std(axis=-1), b.std(axis=-1)
    centred_a = a - a.mean(axis=-1, keepdims=True)
    centred_b = b - b.mean(axis=-1, keepdims=True)
    # Dividing by the std only where it clears eps leaves the others' rows at 0 (PET zeroes
    # them after np.corrcoef, which gives NaN or noise there).
    scale_a = np.divide(1.0, std_a, out=np.zeros_like(std_a), where=std_a >= eps)
    scale_b = np.divide(1.0, std_b, out=np.zeros_like(std_b), where=std_b >= eps)
    corr: NDArray[np.float64] = (
        (centred_a * scale_a[:, None]) @ (centred_b * scale_b[:, None]).T / a.shape[-1]
    )
    return corr


def threshold_taper(
    corr: NDArray[np.float64],
    corr_shuffled: NDArray[np.float64],
    block_sizes: Sequence[int],
    threshold: Threshold,
    cutoff: float,
    taper: Taper,
) -> NDArray[np.float64]:
    """Taper correlations against their noise level, per state block and column (upstream PET).

    Upstream PET's ``tapering_function``. For each state block and column the noise level is
    ``sigma = median|rho_s| / 0.6745`` over the block's shuffled correlations, and the
    threshold is ``cutoff * sigma`` (``"adaptive"``), ``cutoff`` (``"fixed"``) or
    ``sqrt(2 ln n_block) * sigma`` (``"universal"``). ``hard`` keeps ``|rho| > threshold``;
    ``soft`` is Gaspari-Cohn of ``(1 - |rho|) / (1 - threshold)``; ``sigmoid`` is
    ``expit(50 (|rho| - threshold))``.

    Parameters
    ----------
    corr, corr_shuffled
        Correlations with the ensemble and with a shuffled one, ``(n_state, n_columns)``.
    block_sizes
        State block sizes; they add up to ``n_state``.

    Returns
    -------
    NDArray
        The taper, ``(n_state, n_columns)``, in ``[0, 1]``.
    """
    if corr.shape != corr_shuffled.shape:
        raise ValueError(f"correlation shapes differ: {corr.shape} and {corr_shuffled.shape}")
    if sum(block_sizes) != corr.shape[0]:
        raise ValueError(f"block sizes {list(block_sizes)} do not add up to {corr.shape[0]} rows")
    result = np.empty_like(corr)
    start = 0
    for size in block_sizes:
        rows = slice(start, start + size)
        magnitude = np.abs(corr[rows])
        noise = np.median(np.abs(corr_shuffled[rows]), axis=0) / 0.6745  # (n_columns,)
        if threshold == "fixed":
            level = np.full(noise.shape, cutoff)
        elif threshold == "universal":
            level = math.sqrt(2 * math.log(size)) * noise
        elif threshold == "adaptive":
            level = cutoff * noise
        else:
            raise ValueError(
                f"threshold must be 'adaptive', 'fixed' or 'universal', got {threshold!r}"
            )
        # |rho| <= 1, so a threshold >= 1 keeps nothing distinguishable from noise. Upstream's
        # soft taper then takes Gaspari-Cohn's inner branch at z <= 0, giving values above 1
        # (it amplifies), and its sigmoid up to 0.5; APEX zeroes the column. The universal
        # threshold reaches this at ordinary sizes (field blocks of 1e5 cells, N <~ 25).
        hopeless = level >= 1
        if np.any(hopeless):
            logger.warning(
                "adaptive_projected localization: %d of %d columns have a threshold >= 1 "
                "(ensemble too small for a block of %d); their updates are zeroed",
                int(hopeless.sum()),
                hopeless.size,
                size,
            )
        block = result[rows]
        if taper == "hard":
            block[:] = magnitude > level[None, :]
        elif taper == "soft":
            with np.errstate(divide="ignore", invalid="ignore"):
                block[:] = gaspari_cohn((1 - magnitude) / (1 - level[None, :]))
        elif taper == "sigmoid":
            block[:] = 1 / (1 + np.exp(-50 * (magnitude - level[None, :])))
        else:
            raise ValueError(f"taper must be 'hard', 'soft' or 'sigmoid', got {taper!r}")
        block[:, hopeless] = 0.0
        start += size
    return result


# ---------------------------------------------------------------------- methods


class _AdaptiveBase(EmitsDiagnostics):
    def __init__(
        self,
        block_sizes: Sequence[int],
        threshold_scale: float = 1.0,
        taper: Taper = "hard",
        threshold: Literal["universal", "fixed"] = "universal",
        cutoff: float | None = None,
    ) -> None:
        if not block_sizes or any(size < 1 for size in block_sizes):
            raise ValueError(
                f"block_sizes must be positive state-variable sizes, got {block_sizes}"
            )
        if threshold_scale <= 0:
            raise ValueError(f"threshold_scale must be positive, got {threshold_scale}")
        if taper not in ("hard", "soft", "sigmoid"):
            raise ValueError(f"taper must be 'hard', 'soft' or 'sigmoid', got {taper!r}")
        if threshold not in ("universal", "fixed"):
            raise ValueError(f"threshold must be 'universal' or 'fixed', got {threshold!r}")
        if threshold == "universal" and cutoff is not None:
            raise ValueError("cutoff is the fixed threshold: give it with threshold = 'fixed'")
        if threshold == "fixed" and threshold_scale != 1.0:
            raise ValueError(
                "threshold_scale scales the universal threshold: with threshold = 'fixed' give "
                "the threshold itself as cutoff"
            )
        if cutoff is not None and not 0 < cutoff < 1:
            raise ValueError(f"cutoff must be in (0, 1), got {cutoff}")
        self.block_sizes = tuple(int(size) for size in block_sizes)
        self.threshold_scale = threshold_scale
        self.taper: Taper = taper
        self.threshold = threshold
        self.cutoff = cutoff

    def _taper(
        self, a: NDArray[np.float64], b: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        # The shuffle is drawn either way, so the localization stream is the same (the
        # fixed threshold does not need the shuffled copy itself).
        order = derangement(rng, a.shape[-1])
        if self.threshold == "fixed":
            return adaptive_taper(
                correlation(a, b), None, self.block_sizes, 1.0, self.taper,
                fixed=self._fixed_cutoff(a.shape[-1]),
            )  # fmt: skip
        return adaptive_taper(
            correlation(a, b),
            correlation(a[:, order], b),
            self.block_sizes,
            self.threshold_scale,
            self.taper,
        )

    def check_members(self, n: int) -> None:
        """Refuse an ensemble of ``n`` members a fixed ``3/sqrt(N)`` threshold keeps nothing with.

        Called when a run is validated, so the refusal comes before any simulation (as
        ``adaptive_rows``', #216). A run whose failed members leave fewer than 10 stops at
        its next taper with the same message.

        Raises
        ------
        ValueError
            If ``threshold = "fixed"`` without ``cutoff`` and ``3/sqrt(n)`` is 1 or more.
        """
        if self.threshold == "fixed":
            self._fixed_cutoff(n)

    def _fixed_cutoff(self, members: int) -> float:
        """Return the fixed threshold: ``cutoff``, or ``3/sqrt(N)`` for ``N`` members.

        Raises
        ------
        ValueError
            If ``3/sqrt(N)`` is 1 or more (fewer than 10 members): nothing would be kept.
        """
        if self.cutoff is not None:
            return self.cutoff
        value = 3.0 / math.sqrt(members)
        if value >= 1.0:
            raise ValueError(
                f"the fixed threshold 3/sqrt(N) = {value:.3g} keeps nothing with {members} "
                "members: give cutoff below 1"
            )
        return value


@localizations.register("adaptive_per_update")
class AdaptiveLocalization(_AdaptiveBase):
    """Automatic and adaptive localization, recomputed at every update (PET ``autoadaloc``).

    Correlates the state anomalies with the ensemble-projected innovations ``X r``
    ``(n_ensemble, n_ensemble)`` and applies ``(T o dX) (X r)``.

    Parameters
    ----------
    block_sizes
        Sizes of the state variables (set from the state layout).
    threshold_scale
        Multiplier of the universal threshold (PET ``autoadaloc`` value, ``nstd``).
    taper
        ``"hard"`` (default), ``"soft"`` or ``"sigmoid"`` (PET ``type``).
    threshold
        ``"universal"`` (default): ``threshold_scale * sqrt(2 ln n_block) * sigma`` from the
        shuffled correlations; ``"fixed"``: ``cutoff`` for every datum and cell (the user,
        2026-10-20). The universal threshold grows with the block's size, so with many
        cells and few members it can keep almost nothing (0.45 for 18,000 cells and 100
        members); a fixed one does not.
    cutoff
        The fixed threshold, in ``(0, 1)``; by default ``3/sqrt(N)`` for ``N`` members, three
        standard deviations of a sample correlation where there is none (Vossepoel, Evensen &
        van Leeuwen 2025, eq. 6; 0.3 at 100). Only with ``threshold = "fixed"``, which also
        takes no ``threshold_scale``. The ``hard`` taper keeps exactly ``|rho| > cutoff``;
        ``soft`` (Gaspari-Cohn of ``(1 - |rho|) / (1 - cutoff)``) reaches 0 only at
        ``|rho| = 2 cutoff - 1``, so below 0.5 it keeps a little of every correlation.

    Differences from PET: the shuffle comes from the run's ``localization`` stream (a
    true derangement); thresholds are per state variable of the layout (PET: per
    ``prod(FIELD)`` rows); columns whose cutoff is >= 1 are zeroed rather than reversed.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "taper": (
            "ensemble-space taper of dX, (n_state, N), per update; with constraints one per "
            "source, (n_sources, n_state, N)"
        )
    }

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Nothing to set up: the taper is recomputed at every update."""

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``(T o dX) (X r)`` with ``T`` from correlations of ``dX`` and ``X r``."""
        if rng is None:
            raise ValueError("AdaptiveLocalization needs a random generator for its shuffle")
        projected = gain_factor @ whitened_innovations  # X r, (n_ensemble, n_ensemble)
        taper = self._taper(state_anomalies, projected, rng)
        self.last_taper = taper  # for diagnostics and plots, not checkpointed
        self._diagnostics.add("localization", "taper", taper)
        step: NDArray[np.float64] = (taper * state_anomalies) @ projected
        return step

    def localized_member_steps(
        self,
        state_anomalies: NDArray[np.float64],
        member_gains: Iterable[NDArray[np.float64]],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``(T o dX) P`` with ``P[:, j] = G_j r_j`` and ``T`` from ``dX`` and ``P``.

        The per-member form of :meth:`localized_step` (whose ``X r`` has columns ``X r_j``).
        """
        if rng is None:
            raise ValueError("AdaptiveLocalization needs a random generator for its shuffle")
        columns = [g @ whitened_innovations[:, j] for j, g in enumerate(member_gains)]
        if len(columns) != whitened_innovations.shape[1]:
            raise ValueError(
                f"{len(columns)} member gains for {whitened_innovations.shape[1]} members"
            )
        projected = np.stack(columns, axis=-1)  # (n_ensemble, n_ensemble)
        taper = self._taper(state_anomalies, projected, rng)
        self.last_taper = taper
        self._diagnostics.add("localization", "taper", taper)
        step: NDArray[np.float64] = (taper * state_anomalies) @ projected
        return step

    def localized_source_steps(
        self,
        state_anomalies: NDArray[np.float64],
        projected: Sequence[NDArray[np.float64]],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``sum_s (T_s o dX) P_s``, one taper per source.

        C-GIES's localization (Luo & Cruz 2022; manuscript eq.
        ``GMAC_constrained_update_formula_corrLoc``):
        ``P_s`` ``(n_ensemble, n_ensemble)`` holds source ``s``'s projected innovations
        (data, each constraint), column ``j`` for member ``j``, and ``T_s`` comes from the
        correlations of ``dX`` with ``P_s``, as :meth:`localized_step`'s.
        """
        if rng is None:
            raise ValueError("AdaptiveLocalization needs a random generator for its shuffle")
        step = np.zeros_like(state_anomalies)
        tapers = []
        for source in projected:
            taper = self._taper(state_anomalies, source, rng)
            tapers.append(taper)
            step += (taper * state_anomalies) @ source
        if tapers:
            self.last_taper = tapers[0]
        # One taper per source (data, then each constraint): (n_sources, n_state, N).
        self._diagnostics.add("localization", "taper", lambda: np.stack(tapers))
        return step

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """No state to checkpoint."""
        return {}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """No state to restore."""


class _GainTaper(EmitsDiagnostics):
    taper_matrix: NDArray[np.float64] | None = None

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``(T o (dX X)) r`` with the fixed taper ``T`` ``(n_state, n_data)``."""
        if self.taper_matrix is None:
            raise ValueError(f"{type(self).__name__}.prepare() must run before the first update")
        gain = state_anomalies @ gain_factor  # Kalman-gain-like, (n_state, n_data)
        if self.taper_matrix.shape != gain.shape:
            raise ValueError(
                f"taper shape {self.taper_matrix.shape} does not match gain {gain.shape}"
            )
        self._diagnostics.add("localization", "taper", self.taper_matrix)
        step: NDArray[np.float64] = (self.taper_matrix * gain) @ whitened_innovations
        return step

    def localized_member_steps(
        self,
        state_anomalies: NDArray[np.float64],
        member_gains: Iterable[NDArray[np.float64]],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return column ``j`` as ``(T o (dX G_j)) r_j`` with the fixed taper ``T``.

        One ``(n_state, n_data)`` gain at a time.
        """
        if self.taper_matrix is None:
            raise ValueError(f"{type(self).__name__}.prepare() must run before the first update")
        n_members = whitened_innovations.shape[1]
        step = np.empty((state_anomalies.shape[0], n_members))
        given = 0
        for j, gain_factor in enumerate(member_gains):
            given += 1
            gain = state_anomalies @ gain_factor
            if self.taper_matrix.shape != gain.shape:
                raise ValueError(
                    f"taper shape {self.taper_matrix.shape} does not match gain {gain.shape}"
                )
            step[:, j] = (self.taper_matrix * gain) @ whitened_innovations[:, j]
        if given != n_members:
            raise ValueError(f"{given} member gains for {n_members} members")
        self._diagnostics.add("localization", "taper", self.taper_matrix)
        return step

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return the fixed taper matrix."""
        return {} if self.taper_matrix is None else {"taper": self.taper_matrix}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore the fixed taper matrix."""
        if "taper" in arrays:
            self.taper_matrix = np.asarray(arrays["taper"], dtype=np.float64)


@localizations.register("adaptive")
class PriorAdaptiveLocalization(_AdaptiveBase, _GainTaper):
    """Adaptive localization with the taper built once from the prior (PET ``ini_only``).

    APEX's default adaptive localization (``"adaptive"``, the user's decision of
    2026-09-28; ``"adaptive_prior"`` is the same plugin). The per-update variant is
    ``"adaptive_per_update"`` (:class:`AdaptiveLocalization`).

    ``T`` ``(n_state, n_data)`` tapers the prior's state-data correlations against their
    noise level; every update applies ``(T o (dX X)) r``.

    Parameters
    ----------
    block_sizes
        Sizes of the state variables (set from the state layout).
    threshold_scale
        Multiplier of the universal threshold (positive, default 1); larger keeps fewer
        correlations. Not with ``threshold = "fixed"``.
    taper
        How correlations are kept: ``"hard"`` (default) keeps those above the threshold
        and drops the rest, ``"soft"`` scales them down smoothly with Gaspari-Cohn, and
        ``"sigmoid"`` uses a steep logistic step.
    threshold
        ``"universal"`` (default): a threshold per state variable and datum, estimated
        from correlations with a shuffled ensemble and growing with the variable's size;
        ``"fixed"``: ``cutoff`` everywhere. With many cells and few members the universal
        threshold can keep almost nothing; a fixed one does not.
    cutoff
        The fixed threshold on ``|correlation|``, in ``(0, 1)``; by default ``3/sqrt(N)``
        for ``N`` members (0.3 at 100), three standard deviations of a sample correlation
        where there is none. Only with ``threshold = "fixed"``.

    Differences from PET: PET-fork's ``ini_only`` branch multiplies the taper with gain
    built from state anomalies divided by the prior standard deviation and never scales
    back, so its step is off by that factor unless every prior std is 1. APEX uses the
    unscaled anomalies.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "taper": "gain taper from the prior, (n_state, n_data); dense"
    }

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Build the taper from the prior ensemble and its forecast."""
        self.taper_matrix = self._taper(prior_state, prior_predicted, rng)


#: The same plugin under its earlier name (kept: configurations use it).
localizations.add("adaptive_prior", PriorAdaptiveLocalization)


@localizations.register("distance")
class DistanceLocalization(_GainTaper):
    """Distance-based localization around well locations (PET ``dist_loc``).

    Each datum ``"<VECTOR>:<WELL>"`` tapers the state by its distance to the well's
    completion cells in the same layer, with the maximum taken over cells. Distances are in
    grid cells; ``anisotropy = [ratio, angle]`` rotates the (y, x) offset by ``angle``
    degrees and divides its first component by ``ratio`` (PET's convention). Data of wells
    without positions raise an error unless ``unlisted = "zero"`` (PET silently zeroes
    them). In 3-D, each layer is tapered only from that well's completions in the same
    layer (no vertical distance), an APEX extension: PET's masks are single-layer.

    Parameters
    ----------
    grid
        ``[nx, ny, nz]``; each state variable is one grid property in Eclipse order
        (``i`` fastest), so every block size must equal ``nx * ny * nz``.
    positions
        Well name to completion cells ``[[i, j, k], ...]``, 1-based as in Eclipse decks.
    radius
        Localization length ``c`` in cells (the taper vanishes beyond ``2c`` for
        Gaspari-Cohn, beyond ``c`` for Furrer-Bengtsson).
    taper
        ``"gaspari_cohn"`` (PET ``gc``) or ``"furrer_bengtsson"`` (PET ``fb``: spherical
        correlation ``rho`` turned into ``N rho^2 / ((N + 1) rho^2 + 1)``).
    anisotropy
        ``[ratio, angle in degrees]``; ``[1, 0]`` is isotropic.
    unlisted
        What to do with data of wells missing from ``positions``: ``"error"`` (default)
        refuses the run, so a misspelt well cannot silently drop its data; ``"zero"``
        leaves the state untouched by them.
    block_sizes
        Sizes of the state variables (set from the state layout).

    Differences from PET: the taper is evaluated directly from cell-to-well distances
    instead of cropping, padding and re-centring a precomputed stencil; see
    docs/migration/pet_parity.md for where PET's stencil placement differs.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "taper": "distance-based gain taper, (n_state, n_data); dense"
    }

    def __init__(
        self,
        grid: Sequence[int],
        positions: Annotated[Mapping[str, Sequence[Sequence[int]]], TEXT],
        radius: float,
        block_sizes: Sequence[int],
        taper: Literal["gaspari_cohn", "furrer_bengtsson"] = "gaspari_cohn",
        anisotropy: ArrayLike = (1.0, 0.0),
        unlisted: Literal["error", "zero"] = "error",
    ) -> None:
        self.grid = tuple(int(n) for n in grid)
        if len(self.grid) != 3 or min(self.grid) < 1:
            raise ValueError(f"grid must be [nx, ny, nz] with positive sizes, got {grid}")
        cells = math.prod(self.grid)
        if any(size != cells for size in block_sizes):
            raise ValueError(
                f"every state variable must have nx*ny*nz = {cells} entries, "
                f"got {list(block_sizes)}"
            )
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if taper not in ("gaspari_cohn", "furrer_bengtsson"):
            raise ValueError(f"taper must be 'gaspari_cohn' or 'furrer_bengtsson', got {taper!r}")
        ratio, angle = (float(v) for v in np.asarray(anisotropy, dtype=np.float64))
        if ratio <= 0:
            raise ValueError(f"anisotropy ratio must be positive, got {ratio}")
        self.positions = {
            well.upper(): [tuple(int(v) for v in cell) for cell in cells_]
            for well, cells_ in positions.items()
        }
        for well, completions in self.positions.items():
            for i, j, k in completions:
                if not (
                    1 <= i <= self.grid[0] and 1 <= j <= self.grid[1] and 1 <= k <= self.grid[2]
                ):
                    raise ValueError(
                        f"well {well} cell {(i, j, k)} is outside the grid {self.grid}"
                    )
        self.radius = float(radius)
        self.block_sizes = tuple(int(size) for size in block_sizes)
        self.taper = taper
        self.anisotropy = (ratio, angle)
        if unlisted not in ("error", "zero"):
            raise ValueError(f"unlisted must be 'error' or 'zero', got {unlisted!r}")
        self.unlisted = unlisted
        self._layout: tuple[list[str | None], int] | None = None

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Build the taper ``(n_state, n_data)`` from the data keys' wells.

        Raises
        ------
        ValueError
            If some data have no well positions and ``unlisted`` is ``"error"``: a
            misspelt well would otherwise silently drop its data from the update. Also if
            any datum is a field component, which has no well.
        """
        fields = sorted({f"{k.datatype}@{k.time}" for k in keys if k.index is not None})
        if fields:
            # Field components (or their wavelet coefficients) have no single well; tapering
            # them by well distance would be meaningless.
            raise ValueError(
                f"distance localization cannot localize field data: {', '.join(fields)}"
            )
        wells = [_well_of(key) for key in keys]
        missing = sorted({str(w) for w in wells if w not in self.positions})
        if missing and self.unlisted == "error":
            raise ValueError(
                f"distance localization has no positions for {', '.join(missing)}; add them, "
                "or set unlisted = 'zero' to leave those data out of the update"
            )
        unused = sorted(set(self.positions) - {w for w in wells if w is not None})
        if unused:
            logger.warning("distance localization: wells without data: %s", ", ".join(unused))
        self._build(wells, prior_state.shape[-1])

    def _build(self, wells: Sequence[str | None], n_ensemble: int) -> None:
        # Dense (n_state, n_data); fine for moderate sizes. The gain it multiplies,
        # dX X, is dense anyway (as in PET).
        field = np.column_stack([self.well_taper(well, n_ensemble) for well in wells])
        self.taper_matrix = np.vstack([field] * len(self.block_sizes))
        self._layout = (list(wells), n_ensemble)

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """Return what rebuilds the taper: the wells of the data and the ensemble size.

        The taper itself is a deterministic function of the geometry, so it is not stored.
        """
        if self._layout is None:
            return {}
        wells, n_ensemble = self._layout
        return {
            "wells": np.array(["" if w is None else w for w in wells]),
            "n_ensemble": np.array(float(n_ensemble)),
        }

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Rebuild the taper from the saved wells and ensemble size."""
        if "wells" in arrays:
            wells = [str(w) or None for w in np.asarray(arrays["wells"]).tolist()]
            self._build(wells, int(np.asarray(arrays["n_ensemble"])))

    def well_taper(self, well: str | None, n_ensemble: int) -> NDArray[np.float64]:
        """Return the taper over the grid, Eclipse order, for ``well`` (zeros if unknown)."""
        nx, ny, nz = self.grid
        taper = np.zeros((nz, ny, nx))
        if well is None or well not in self.positions:
            return taper.ravel()
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")  # (ny, nx) 0-based
        ratio, angle = self.anisotropy
        theta = math.radians(angle)
        for wi, wj, wk in self.positions[well]:
            dy, dx = (wj - 1) - j, (wi - 1) - i  # PET's (row, column) offset
            ry = math.cos(theta) * dy + math.sin(theta) * dx
            rx = -math.sin(theta) * dy + math.cos(theta) * dx
            distance = np.hypot(ry / ratio, rx) / self.radius
            if self.taper == "gaspari_cohn":
                values = gaspari_cohn(distance)
            else:
                rho = np.where(distance < 1, 1 - 1.5 * distance + 0.5 * distance**3, 0.0)
                values = n_ensemble * rho**2 / (rho**2 * (n_ensemble + 1) + 1)
            taper[wk - 1] = np.maximum(taper[wk - 1], values)
        return taper.ravel()


def _well_of(key: DataKey) -> str | None:
    _, _, well = key.datatype.partition(":")
    return well.upper() or None


@runtime_checkable
class ProjectedLocalization(Protocol):
    """A localization in the space of the approx update's projected data (``ApproxUpdate``).

    The approx update's truncated SVD ``L^-1 dY = U S V^T`` projects the data onto ``r``
    directions; such a localization tapers the state-projected-data covariance there.
    """

    def projected_step(
        self,
        state_anomalies: NDArray[np.float64],
        projected_anomalies: NDArray[np.float64],
        projected_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return the localized increment ``(n_state, n_ensemble)``.

        ``projected_anomalies`` is ``S V^T`` ``(r, n_ensemble)``, the data anomalies in the
        projected space (``U^T L^-1 dY``); ``projected_innovations`` is the update's
        ``(r, n_ensemble)`` coefficients, so that the unlocalized step is
        ``dX (S V^T)^T projected_innovations``.
        """
        ...


def refuse_projected(localization: object, strategy: str) -> None:
    """Raise if ``localization`` works only in the approx update's projected space.

    Raises
    ------
    ValueError
        If ``localization`` is a :class:`ProjectedLocalization`.
    """
    if isinstance(localization, ProjectedLocalization):
        raise ValueError(
            f"adaptive_projected localizes in the approx update's projected-data space; "
            f"strategy {strategy!r} has no such space: use strategy 'approx', or another "
            "localization (adaptive, adaptive_per_update, distance)"
        )


@localizations.register("adaptive_projected")
class ProjectedAdaptiveLocalization(EmitsDiagnostics):
    """Upstream PET's adaptive localization (``autoadaloc``) in the projected-data space.

    At every update of the approx analysis, with ``L^-1 dY = U S V^T`` (truncated SVD) and
    the projected data anomalies ``P = S V^T`` ``(r, N)``, the taper ``T`` ``(n_state, r)``
    compares the correlations of ``dX`` with the rows of ``P`` against their noise level
    (correlations with a shuffled ``dX``), and the step is

    ``(T o (dX P^T)) X2``,   ``X2 = (I (1 + lambda) + S^2)^-1 U^T L^-1 (D - Y)``

    (``T = 1`` gives the unlocalized step). PET's ``approx_update`` with
    ``localization = {name = "autoadaloc", projection = "rank-r"}``; the ES, ES-MDA,
    LM-EnRML and GN-EnRML approx analyses, also with ``empirical_data_error`` (PET
    ``emp_cov``, whose ``X2`` it takes).

    Parameters
    ----------
    block_sizes
        Sizes of the state variables (set from the state layout); each has its own noise
        levels (PET: per ``parameters`` block).
    threshold
        ``"adaptive"`` (default): ``cutoff * sigma``; ``"fixed"``: ``cutoff``;
        ``"universal"``: ``sqrt(2 ln n_block) * sigma`` (PET ``threshold``).
    cutoff
        The multiplier of ``sigma`` (``"adaptive"``) or the threshold itself (``"fixed"``,
        in ``(0, 1)``); PET ``cutoff`` (also ``nstd``), default 0.3. Unused by
        ``"universal"``.
    taper
        ``"hard"`` (default), ``"soft"`` or ``"sigmoid"`` (PET ``type``: ``hard``,
        ``soft``, ``sigm``).

    Differences from PET: noise levels are per state variable of the layout (PET: per
    ``parameters`` block of ``prod(field)`` or ``active`` rows, one block by default);
    columns whose threshold is >= 1 are zeroed (PET's soft taper amplifies there, its
    sigmoid gives up to 0.5); rows zeroed by PET's absolute ``1e-6`` std floor (a small
    or collapsed spread) are logged; the shuffle is ``rng.permutation`` from the run's
    ``localization`` stream, as PET's (fixed points allowed). PET's ``projection =
    "ensemble"`` is not ported: it multiplies ``X2`` ``(r, N)`` by ``D_anom`` ``(n_data,
    N)``, which fails unless ``r == N == n_data`` (recorded in pet_parity.md);
    ``adaptive_per_update`` tapers in the ensemble space.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "taper": "projected-data taper, (n_state, r), per update"
    }
    #: Upstream PET localizes LM-EnRML's and GN-EnRML's approx update this way.
    enrml_localization: ClassVar[bool] = True

    def __init__(
        self,
        block_sizes: Sequence[int],
        threshold: Threshold = "adaptive",
        cutoff: float = 0.3,
        taper: Taper = "hard",
    ) -> None:
        if not block_sizes or any(size < 1 for size in block_sizes):
            raise ValueError(
                f"block_sizes must be positive state-variable sizes, got {block_sizes}"
            )
        if threshold not in ("adaptive", "fixed", "universal"):
            raise ValueError(
                f"threshold must be 'adaptive', 'fixed' or 'universal', got {threshold!r}"
            )
        if not cutoff > 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if threshold == "fixed" and not cutoff < 1:
            # A correlation magnitude never exceeds 1: a fixed threshold >= 1 keeps nothing.
            raise ValueError(f"a fixed threshold must be in (0, 1), got cutoff = {cutoff}")
        if taper not in ("hard", "soft", "sigmoid"):
            raise ValueError(f"taper must be 'hard', 'soft' or 'sigmoid', got {taper!r}")
        self.block_sizes = tuple(int(size) for size in block_sizes)
        self.threshold: Threshold = threshold
        self.cutoff = float(cutoff)
        self.taper: Taper = taper

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Nothing to set up: the taper is recomputed at every update."""

    def projected_step(
        self,
        state_anomalies: NDArray[np.float64],
        projected_anomalies: NDArray[np.float64],
        projected_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Return ``(T o (dX P^T)) X2`` (see the class docstring).

        Parameters
        ----------
        state_anomalies
            ``dX`` ``(n_state, N)``.
        projected_anomalies
            ``P = S V^T`` ``(r, N)``.
        projected_innovations
            ``X2`` ``(r, N)``.
        rng
            The ``localization`` stream, for the shuffle.
        """
        if rng is None:
            raise ValueError("adaptive_projected needs a random generator for its shuffle")
        n_ensemble = state_anomalies.shape[-1]
        rank = projected_anomalies.shape[0]
        if projected_anomalies.shape != (rank, n_ensemble) or projected_innovations.shape != (
            rank,
            n_ensemble,
        ):
            raise ValueError(
                f"projected anomalies {projected_anomalies.shape} and innovations "
                f"{projected_innovations.shape} must both be (r, {n_ensemble})"
            )
        # PET's eps (1e-6) is absolute, on anomalies in the state's units / sqrt(N - 1): a
        # variable with a small spread (e.g. compressibility, SI permeability) or one that
        # has collapsed gets a zero taper row and is not updated. Kept for parity; told.
        spread = state_anomalies.std(axis=-1)
        frozen = (spread > 0) & (spread < 1e-6)
        if np.any(frozen):
            logger.warning(
                "adaptive_projected localization: %d state rows have an anomaly std below "
                "PET's absolute 1e-6 and are not updated (rescale such variables)",
                int(frozen.sum()),
            )
        # PET: corr_matrix(X, Y) and corr_matrix(X[:, rng.permutation(ne)], Y).
        shuffled = state_anomalies[:, rng.permutation(n_ensemble)]
        taper = threshold_taper(
            pearson_correlation(state_anomalies, projected_anomalies),
            pearson_correlation(shuffled, projected_anomalies),
            self.block_sizes,
            self.threshold,
            self.cutoff,
            self.taper,
        )  # (n_state, r)
        self._diagnostics.add("localization", "taper", taper)
        # C_xp = dX P^T, the state-projected-data covariance (n_state, r).
        step: NDArray[np.float64] = (taper * (state_anomalies @ projected_anomalies.T)) @ (
            projected_innovations
        )
        return step

    def localized_step(
        self,
        state_anomalies: NDArray[np.float64],
        gain_factor: NDArray[np.float64],
        whitened_innovations: NDArray[np.float64],
        rng: np.random.Generator | None,
    ) -> NDArray[np.float64]:
        """Refuse: this localization needs the approx update's projected space.

        Raises
        ------
        ValueError
            Always; strategies other than ``approx`` refuse it when constructed.
        """
        raise ValueError(
            "adaptive_projected localizes only the approx update (its projected-data space)"
        )

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        """No state to checkpoint (the taper is recomputed at every update)."""
        return {}

    def restore_state(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """No state to restore."""
