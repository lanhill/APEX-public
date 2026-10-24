"""Multilevel ES-MDA with the hybrid gain.

Ported from PET
---------------
PET module : src/pipt/update_schemes/multilevel.py (esmda_hybrid, MultilevelEnsemble),
             src/pipt/update_schemes/analysis/hybrid.py (hybrid_update),
             src/pipt/misc_tools/extract_tools.py (extract_multilevel_info)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Fossum, K., Mannseth, T. & Stordal, A. S. (2020). Assessment of multilevel
             ensemble-based data assimilation for reservoir history matching.
             Computational Geosciences, 24(1), 217-239.
Differences: none in the step (the parity test pins it). PET's state scaling multiplies
             and divides and cancels exactly, so it is omitted (as in the approx update).
             PET draws each level's perturbations with its own ``gen_real`` call; APEX
             draws them for the whole ensemble at once from the step's stream and gives
             each level its members' columns (the same distribution). Members that fail
             are dropped from their own level (PET cannot drop members). PET scores the
             data mismatch over all levels together against the un-inflated perturbations, as
             APEX's ES-MDA does.

The ensemble is split into fidelity levels ``l`` (by member id, in order); level ``l`` runs
a coarser, cheaper model. At every ES-MDA step each level's forecast gives its gain factor,
from the truncated SVD of its whitened data anomalies ``L^-1 dY_l = U_l s_l V_l^T``, and the
*hybrid* gain combines them with the level weights ``w_l``:

    ``K = sum_l w_l dX_l V_l s_l ((1 + lambda) I + s_l^2)^-1 U_l^T``   (``(n_state, n_data)``),

``dX_l`` the level's ``1/sqrt(N_l - 1)`` state anomalies. Each level is stepped with its own
innovations, ``step_l = K L^-1 (D_l - Y_l)``, ``C_d`` inflated by the step's ``alpha``.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Required, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from apex.algorithms.data_assimilation.algorithm import RunSetup, Status, Step
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.esmda import ESMDA
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.registries import methods, update_strategies
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.errors import WorkflowError
from apex.core.options import LEVELS


def hybrid_gain(
    states: Sequence[NDArray[np.float64]],
    predicted: Sequence[NDArray[np.float64]],
    data_error: DataErrorCovariance,
    weights: Sequence[float],
    *,
    energy: float,
    damping: float = 0.0,
) -> NDArray[np.float64]:
    """Return the hybrid gain ``K`` ``(n_state, n_data)`` on whitened innovations.

    ``states[l]`` ``(n_state, N_l)`` and ``predicted[l]`` ``(n_data, N_l)`` are level ``l``'s
    ensemble and forecast; ``weights`` the level weights (summing to 1).
    """
    gain: NDArray[np.float64] | None = None
    for state, forecast, weight in zip(states, predicted, weights, strict=True):
        u, s, vt = truncated_svd(data_error.whiten(anomalies(forecast)), energy)
        level = anomalies(state) @ (((vt.T * s) / (1.0 + damping + s**2)) @ u.T)
        gain = weight * level if gain is None else gain + weight * level
    assert gain is not None
    return gain


@update_strategies.register("hybrid")
class HybridUpdate(EmitsDiagnostics):
    """The multilevel hybrid update: one gain from every fidelity level's anomalies.

    Used by :class:`MultilevelESMDA`, which passes each member's level and the weights.

    Parameters
    ----------
    energy
        Fraction of the singular-value sum kept per level, in ``(0, 1]`` (default 0.98).
    """

    requires_reference_prediction = False
    relative_damping = False
    #: Needs each member's level: only multilevel ES-MDA provides them.
    requires_levels = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "gain": "the hybrid gain K on whitened innovations, (n_state, n_data); dense"
    }

    def __init__(self, energy: float = 0.98) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        self.energy = energy

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
        levels: NDArray[np.int64] | None = None,
        weights: Sequence[float] = (),
    ) -> NDArray[np.float64]:
        """Return the increment ``(n_state, N)``; ``levels`` ``(N,)`` gives each member's level.

        Raises
        ------
        ValueError
            If the shapes are inconsistent, ``levels`` or ``weights`` are missing or do not
            fit, a level has fewer than 2 members, or a ``reference_prediction`` is given.
        """
        if reference_prediction is not None:
            raise ValueError("HybridUpdate centres data anomalies on each level's mean")
        check_shapes(state, predicted, perturbed_observations, data_error)
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        if levels is None or levels.shape != (state.shape[-1],):
            raise ValueError("HybridUpdate needs each member's level, (n_ensemble,)")
        if not weights or int(levels.max()) >= len(weights) or int(levels.min()) < 0:
            raise ValueError(
                f"levels {sorted(set(levels.tolist()))} do not fit {len(weights)} weights"
            )
        columns = [np.flatnonzero(levels == level) for level in range(len(weights))]
        small = [level for level, c in enumerate(columns) if c.size < 2]
        if small:
            raise ValueError(f"levels {small} have fewer than 2 members")
        if self._diagnostics.wants("strategy", "gain"):
            gain = hybrid_gain(
                [state[:, c] for c in columns],
                [predicted[:, c] for c in columns],
                data_error,
                weights,
                energy=self.energy,
                damping=damping,
            )
            self._diagnostics.add("strategy", "gain", gain, nbytes=gain.nbytes)
        # K r = sum_l w_l dX_l [V_l s_l ((1 + lambda) I + s_l^2)^-1 (U_l^T r)] without forming
        # K (n_state x n_data): O(n_state N) memory, as the approx update (PET batches rows).
        innovations = data_error.whiten(perturbed_observations - predicted)  # r, (n_data, N)
        step = np.zeros_like(state)
        for c, weight in zip(columns, weights, strict=True):
            u, s, vt = truncated_svd(data_error.whiten(anomalies(predicted[:, c])), self.energy)
            projected = (u.T @ innovations) / (1.0 + damping + s**2)[:, None]  # (r, N)
            step += weight * (anomalies(state[:, c]) @ ((vt.T * s) @ projected))
        return step


class Level(TypedDict, total=False):
    """One fidelity level: its members and weight, and what its simulator runs.

    Levels are listed coarsest first and hold consecutive member ids; their members add up
    to ``[prior] ensemble_size``. A level with no simulator key runs the fine model.
    """

    members: Required[
        Annotated[int, Field(description="The level's number of members N_l (at least 2).")]
    ]
    weight: Required[
        Annotated[
            float,
            Field(
                description="The level's weight w_l in the hybrid gain (non-negative; the "
                "weights are normalised to sum 1, as PET's)."
            ),
        ]
    ]
    upscale: Annotated[
        Mapping[str, Any],
        Field(description="The simulator's grid coarsening (method, dim, max_trunc, ...)."),
    ]
    deck_template_file: Annotated[
        str, Field(description="A deck template of the level's own (a coarse model).")
    ]
    forward_file: Annotated[
        str, Field(description="The linear simulator's G of the level (an .npz).")
    ]
    matrix: Annotated[
        Sequence[Sequence[float]], Field(description="The linear simulator's G, inline.")
    ]


@methods.register("multilevel_esmda")
class MultilevelESMDA(ESMDA):
    """Multilevel ES-MDA: ES-MDA's schedule on an ensemble split over fidelity levels.

    Parameters
    ----------
    inflation
        The ES-MDA inflation factors ``alpha_i`` (sum of ``1 / alpha_i`` equal to 1).
    levels
        One table per level (:class:`Level`), coarsest first as PET's: ``members`` (``N_l``)
        and ``weight`` (``w_l``; normalised to sum 1, as PET). Other keys (``upscale``,
        ``deck_template_file``) are for the simulator (the composition passes them).
    strategy
        ``HybridUpdate()`` by default (the only one).
    lower_bounds, upper_bounds
        Optional per-state bounds ``(n_state,)``.
    """

    status: Status = "parity"
    provides_levels: ClassVar[bool] = True

    def __init__(
        self,
        inflation: Sequence[float],
        levels: Annotated[Sequence[Level], LEVELS],
        strategy: HybridUpdate | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> None:
        if not levels:
            raise ValueError("multilevel ES-MDA needs at least one level")
        for index, level in enumerate(levels):
            # Named here: a missing key used to surface as 0 members or a NaN weight.
            missing = [key for key in ("members", "weight") if key not in level]
            if missing:
                raise ValueError(
                    f"level {index} needs {' and '.join(missing)}: every level is a table "
                    "{ members = N_l, weight = w_l }, coarsest first"
                )
        sizes = [int(level["members"]) for level in levels]
        if any(size < 2 for size in sizes):
            raise ValueError(f"every level needs at least 2 members, got {sizes}")
        raw = [float(level["weight"]) for level in levels]
        if not all(math.isfinite(w) and w >= 0 for w in raw) or sum(raw) <= 0:
            raise ValueError(f"level weights must be non-negative with a positive sum, got {raw}")
        if strategy is not None and not isinstance(strategy, HybridUpdate):
            raise ValueError(
                f"multilevel ES-MDA runs the hybrid update only (PET's), not "
                f"{type(strategy).__name__}: leave out [method.strategy], or name hybrid"
            )
        super().__init__(inflation, HybridUpdate() if strategy is None else strategy,
                         lower_bounds, upper_bounds)  # fmt: skip
        self.sizes = tuple(sizes)
        total = sum(raw)
        # PET normalises the weights unless they already sum to exactly 1.
        self.weights = tuple(raw) if total == 1.0 else tuple(w / total for w in raw)
        self._bounds = np.cumsum(self.sizes)

    def start(self, setup: RunSetup) -> Step:
        """Check that the prior has the levels' members, then start as ES-MDA.

        Raises
        ------
        ValueError
            If the prior's size is not the sum of the levels' members (a ``file`` prior of
            another size would otherwise shrink or overrun the last level).
        """
        n = setup.prior.shape[-1]
        if n != int(self._bounds[-1]):
            raise ValueError(
                f"the prior has {n} members but the levels {list(self.sizes)} need "
                f"{int(self._bounds[-1])}"
            )
        return super().start(setup)

    def level_of(self, member_ids: NDArray[np.int64]) -> NDArray[np.int64]:
        """Return the level of each member id (levels hold consecutive ids, coarsest first).

        Raises
        ------
        ValueError
            If an id is beyond the levels' total size.
        """
        ids = np.asarray(member_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= self._bounds[-1]):
            raise ValueError(f"member ids beyond the {int(self._bounds[-1])} members of the levels")
        levels: NDArray[np.int64] = np.searchsorted(self._bounds, ids, side="right").astype(
            np.int64
        )
        return levels

    def _update(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed: NDArray[np.float64],
        inflated: DataErrorCovariance,
        rng: np.random.Generator | None = None,
        *,
        sensitivity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        run = self._require_run()
        levels = self.level_of(run.member_ids)
        counts = np.bincount(levels, minlength=len(self.sizes))
        if (counts < 2).any():
            raise WorkflowError(
                f"multilevel ES-MDA: members per level {counts.tolist()} after failures; every "
                "level needs at least 2"
            )
        step = self.strategy.step(  # type: ignore[call-arg]
            state, predicted, perturbed, inflated, levels=levels, weights=self.weights
        )
        self._diagnostics.add("method", "step", step)
        return self.bounds.apply(state + step)
