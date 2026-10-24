"""NSGA-III (``nsga3``): reference-direction survival for many objectives.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/algorithms/moo/nsga3.py (``NSGA3``, ``ReferenceDirectionSurvival``,
             ``niching``, ``associate_to_niches``, ``calc_niche_count``,
             ``HyperplaneNormalization``, ``get_extreme_points_c``, ``get_nadir_point``,
             ``comp_by_cv_then_random``), pymoo/functions/standard/
             calc_perpendicular_distance.py
Reference  : Deb, K. & Jain, H. (2014). An evolutionary many-objective optimization algorithm
             using reference-point-based nondominated sorting approach, part I: solving
             problems with box constraints. IEEE TEC, 18, 577-601.
Differences: as ``nsga2``: ``x0`` is a member, robust objectives are reduced over
             the realizations (by their statistic), EPF is offered, and the stall is
             the front's movement. Defaults are smaller than the papers'
             (:mod:`.reference_directions`). Two infeasible members of equal violation are
             drawn by the run's generator; pymoo's ``comp_by_cv_then_random`` calls
             ``compare`` without its generator, so pymoo draws that tie from a fresh, unseeded
             one (not reproducible). The front reported is every non-dominated feasible
             member (as ``nsga2``); pymoo's ``opt`` keeps only those nearest a direction.

Survival (pymoo's ``ReferenceDirectionSurvival`` within ``Survival.do``): the feasible
members are sorted into fronts, up to the front that reaches the number to keep; the
hyperplane normalisation is updated from them (ideal, worst and extreme points, the nadir
from the extreme points' hyperplane, its state kept across generations); each member is
associated with its nearest reference line in normalised objectives; the last front is cut by
niching (the least crowded lines first, the nearest member for an empty line, else a random
one). Infeasible members follow, least violation first. Parents are picked by binary
tournaments: the smaller violation if either is infeasible, else at random.

Arrays follow APEX's convention internally (objectives ``(n_obj, n)``, members last); the
normalisation's points are vectors ``(n_obj,)`` and the extreme points rows ``(n_obj, n_obj)``
as pymoo's.
"""

import logging
import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from apex.algorithms.optimization.base import EpfOptions
from apex.algorithms.optimization.multiobjective import fronts_of, pareto_ranks
from apex.algorithms.optimization.nsga2 import NSGA2
from apex.algorithms.optimization.population import ConstraintHandling, Initialization
from apex.algorithms.optimization.protocols import (
    FloatArray,
    IntArray,
    OptimizerState,
    StateValue,
    Status,
)
from apex.algorithms.optimization.reference_directions import directions_for
from apex.algorithms.optimization.registries import optimizers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- kernels


def perpendicular_distance(N: FloatArray, directions: FloatArray) -> FloatArray:
    """Return each point's distance to each reference line ``(n, k)``.

    ``N`` normalised points ``(n, n_obj)`` (rows, as pymoo's), ``directions`` ``(k, n_obj)``.
    """
    norm = np.linalg.norm(directions, axis=1)  # (k,)
    scalar = (N @ directions.T) / norm[None, :]  # (n, k): projections' lengths
    proj = scalar[:, :, None] * (directions / norm[:, None])[None, :, :]  # (n, k, n_obj)
    return np.asarray(np.linalg.norm(proj - N[:, None, :], axis=2), dtype=float)


def associate_to_niches(
    F: FloatArray, directions: FloatArray, ideal: FloatArray, nadir: FloatArray
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Return each member's niche, its distance to it and the distance matrix.

    ``F`` rows ``(n, n_obj)``; normalised by the ideal point and the intercepts (a zero range
    divides by 1e-12, as pymoo's).
    """
    denom = nadir - ideal
    denom = np.where(denom == 0, 1e-12, denom)
    matrix = perpendicular_distance((F - ideal) / denom, directions)
    niche = np.argmin(matrix, axis=1).astype(np.int64)
    return niche, matrix[np.arange(F.shape[0]), niche], matrix


def niche_counts(n_niches: int, niche: IntArray) -> IntArray:
    """Return how many members each niche holds ``(n_niches,)``."""
    counts = np.zeros(n_niches, dtype=np.int64)
    index, count = np.unique(niche, return_counts=True)
    counts[index] = count
    return counts


def niching(
    rng: np.random.Generator,
    n_remaining: int,
    counts: IntArray,
    niche: IntArray,
    distance: FloatArray,
) -> list[int]:
    """Return the last front's survivors (pymoo's ``niching``; ``counts`` is updated).

    ``niche`` and ``distance`` are the last front's members'.
    """
    survivors: list[int] = []
    mask = np.full(niche.size, True)
    while len(survivors) < n_remaining:
        n_select = n_remaining - len(survivors)
        candidates = np.unique(niche[mask])
        least = counts[candidates].min()
        chosen = candidates[np.where(counts[candidates] == least)[0]]
        chosen = chosen[rng.permutation(len(chosen))[:n_select]]
        for next_niche in chosen:
            members = np.where(np.logical_and(niche == next_niche, mask))[0]
            rng.shuffle(members)  # breaks ties at random
            pick = int(members[np.argmin(distance[members])]) if counts[next_niche] == 0 \
                else int(members[0])  # fmt: skip
            mask[pick] = False
            survivors.append(pick)
            counts[next_niche] += 1
    return survivors


def extreme_points(F: FloatArray, ideal: FloatArray, previous: FloatArray | None) -> FloatArray:
    """Return the extreme points ``(n_obj, n_obj)``: per axis, the least achievement scalar.

    ``F`` rows ``(n, n_obj)``; the previous extreme points are candidates too, so none is lost.
    """
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6
    candidates = F if previous is None else np.concatenate([previous, F], axis=0)
    shifted = candidates - ideal
    shifted[shifted < 1e-3] = 0
    asf = np.max(shifted * weights[:, None, :], axis=2)  # (n_obj, n_candidates)
    return np.ascontiguousarray(candidates[np.argmin(asf, axis=1), :])


def nadir_point(
    extremes: FloatArray,
    ideal: FloatArray,
    worst: FloatArray,
    worst_of_front: FloatArray,
    worst_of_population: FloatArray,
) -> FloatArray:
    """Return the nadir from the extreme points' hyperplane, with pymoo's fallbacks."""
    try:
        M = extremes - ideal
        b = np.ones(extremes.shape[1])
        plane = np.linalg.solve(M, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            intercepts = 1 / plane
        nadir = ideal + intercepts
        if not np.allclose(M @ plane, b) or np.any(intercepts <= 1e-6):
            raise np.linalg.LinAlgError
        # pymoo, unlike the paper: never beyond the worst point found so far.
        nadir = np.where(nadir > worst, worst, nadir)
    except np.linalg.LinAlgError:
        nadir = np.array(worst_of_front, dtype=float)
    small = nadir - ideal <= 1e-6
    return np.asarray(np.where(small, worst_of_population, nadir), dtype=float)


class HyperplaneNormalization:
    """NSGA-III's normalisation, kept across generations (pymoo's)."""

    def __init__(self, n_obj: int) -> None:
        self.ideal = np.full(n_obj, np.inf)
        self.worst = np.full(n_obj, -np.inf)
        self.nadir: FloatArray | None = None
        self.extremes: FloatArray | None = None

    def update(self, F: FloatArray, nds: IntArray) -> None:
        """Update from the feasible members' rows ``(n, n_obj)`` and the first front ``nds``."""
        self.ideal = np.min(np.vstack((self.ideal, F)), axis=0)
        self.worst = np.max(np.vstack((self.worst, F)), axis=0)
        self.extremes = extreme_points(F[nds, :], self.ideal, self.extremes)
        self.nadir = nadir_point(
            self.extremes, self.ideal, self.worst, np.max(F[nds, :], axis=0), np.max(F, axis=0)
        )


def reference_survival(
    rng: np.random.Generator,
    F: FloatArray,
    n_survive: int,
    directions: FloatArray,
    norm: HyperplaneNormalization,
) -> tuple[IntArray, IntArray]:
    """Return NSGA-III's survivors among feasible members and every member's rank.

    ``F`` ``(n_obj, n)``; the survivors in pymoo's order (the fronts before the last, then the
    last front's niching picks).
    """
    rows = np.ascontiguousarray(F.T)
    ranks = pareto_ranks(F)
    fronts: list[IntArray] = []
    total = 0
    for front in fronts_of(ranks):  # pymoo's n_stop_if_ranked: up to the one reaching n
        fronts.append(front)
        total += front.size
        if total >= n_survive:
            break
    norm.update(rows, fronts[0])
    assert norm.nadir is not None
    order = np.concatenate(fronts)
    niche, distance, _ = associate_to_niches(rows[order], directions, norm.ideal, norm.nadir)
    if order.size <= n_survive:
        return order, ranks
    if len(fronts) == 1:
        until = np.array([], dtype=np.int64)
        counts = np.zeros(len(directions), dtype=np.int64)
    else:
        until = np.arange(order.size - fronts[-1].size)
        counts = niche_counts(len(directions), niche[until])
    last = np.arange(order.size - fronts[-1].size, order.size)
    picks = niching(rng, n_survive - until.size, counts, niche[last], distance[last])
    chosen = np.concatenate((until, last[picks])).astype(np.int64)
    return order[chosen], ranks


def survive_by_references(
    rng: np.random.Generator,
    F: FloatArray,
    cv: FloatArray,
    n_survive: int,
    directions: FloatArray,
    norm: HyperplaneNormalization,
) -> tuple[IntArray, IntArray]:
    """Return the survivors under constraints (pymoo's ``Survival.do``) and ranks.

    The feasible members go through :func:`reference_survival` (at most ``n_survive``), then
    infeasible ones fill up, least violation first (NumPy's default sort, as pymoo's).
    """
    n = F.shape[1]
    ranks = np.full(n, -1, dtype=np.int64)
    feasible = np.flatnonzero(cv <= 0.0)
    survivors: list[int] = []
    if feasible.size:
        kept, r = reference_survival(rng, F[:, feasible], min(feasible.size, n_survive),
                                     directions, norm)  # fmt: skip
        ranks[feasible] = r
        survivors = [int(i) for i in feasible[kept]]
    infeasible = np.flatnonzero(cv > 0.0)
    infeasible = infeasible[np.argsort(cv[infeasible])]
    survivors.extend(int(i) for i in infeasible[: max(n_survive - len(survivors), 0)])
    return np.asarray(survivors, dtype=np.int64), ranks


def tournament_by_violation(rng: np.random.Generator, cv: FloatArray, n_select: int) -> IntArray:
    """Return ``n_select`` winners of pymoo's ``comp_by_cv_then_random`` tournaments.

    Competitors from concatenated permutations; the smaller violation if either is infeasible
    (a tie drawn from ``rng``; see the module docstring), else one at random.
    """
    n = cv.size
    needed = n_select * 2
    perms = np.concatenate([rng.permutation(n) for _ in range(int(np.ceil(needed / n)))])
    pairs = perms[:needed].reshape(n_select, 2)
    winners = np.empty(n_select, dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        if cv[a] > 0.0 or cv[b] > 0.0:
            if cv[a] < cv[b]:
                winners[i] = a
            elif cv[a] > cv[b]:
                winners[i] = b
            else:
                winners[i] = int(rng.choice([a, b]))
        else:
            winners[i] = int(rng.choice([a, b]))
    return winners


# ---------------------------------------------------------------------------- the optimizer


@optimizers.register("nsga3")
class NSGA3(NSGA2):
    """NSGA-III: many-objective NSGA with survival by reference directions.

    Parameters
    ----------
    n_partitions, inner_partitions
        Das-Dennis divisions of each axis, and of an inner layer shrunk halfway to the centre
        (0: none). Default by objectives: 2 -> 19, 3 -> 6, 4 -> 4 (20, 28, 35 directions);
        from 5, two layers, 2 and 1 (20 directions for 5, 27 for 6, 44 for 8), as a single
        layer with fewer partitions than objectives has no interior direction. Deb & Jain
        (2014) take 12 for three objectives (91 directions), as pymoo's examples do; APEX's
        are smaller because an evaluation is a simulation.
    pop_size
        Members (default: the number of directions, rounded up to a multiple of 4; pymoo: the
        number of directions, warning below it).
    n_offsprings
        Children per generation (default ``pop_size``).
    eta_c, crossover_prob
        SBX's distribution index and probability (pymoo's NSGA-III: 30 and 1).
    crossover_prob_var, mutation_prob, mutation_prob_var, eta_m, eliminate_duplicates, ref_point,
    stall, ftol, init, constraint_handling, eq_tolerance, maxiter, xtol, transform, epf
        As ``nsga2``.
    """

    status: ClassVar[Status] = "experimental"
    label: ClassVar[str] = "NSGA-III"

    def __init__(
        self,
        *,
        n_partitions: int | None = None,
        inner_partitions: int | None = None,
        pop_size: int | None = None,
        n_offsprings: int | None = None,
        crossover_prob: float = 1.0,
        crossover_prob_var: float = 0.5,
        eta_c: float = 30.0,
        mutation_prob: float = 0.9,
        mutation_prob_var: float | None = None,
        eta_m: float = 20.0,
        eliminate_duplicates: bool = True,
        ref_point: list[float] | None = None,
        stall: int = 30,
        init: Initialization = "uniform",
        constraint_handling: ConstraintHandling = "feasibility",
        eq_tolerance: float = 1e-4,
        maxiter: int = 100,
        ftol: float = 0.0025,
        xtol: float = 1e-8,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        if n_partitions is not None and n_partitions < 1:
            raise ValueError(f"n_partitions must be positive, got {n_partitions}")
        if inner_partitions is not None and inner_partitions < 0:
            raise ValueError(f"inner_partitions must not be negative, got {inner_partitions}")
        # The population is sized once the objectives are known (set_objectives); until then
        # a placeholder that the base class accepts.
        super().__init__(
            pop_size=pop_size or 4, n_offsprings=n_offsprings, crossover_prob=crossover_prob,
            crossover_prob_var=crossover_prob_var, eta_c=eta_c, mutation_prob=mutation_prob,
            mutation_prob_var=mutation_prob_var, eta_m=eta_m,
            eliminate_duplicates=eliminate_duplicates, ref_point=ref_point, stall=stall,
            init=init, constraint_handling=constraint_handling, eq_tolerance=eq_tolerance,
            maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf,
        )  # fmt: skip
        self.n_partitions = n_partitions
        self.inner_partitions = inner_partitions
        self.given_pop_size = pop_size
        self.given_offsprings = n_offsprings
        self.directions: FloatArray | None = None
        self.norm: HyperplaneNormalization | None = None

    def set_objectives(self, n_obj: int) -> None:
        """Make the reference directions for ``n_obj`` objectives and size the population."""
        self.directions = directions_for(n_obj, self.n_partitions, self.inner_partitions)
        k = self.directions.shape[0]
        self.pop_size = self.given_pop_size or 4 * math.ceil(k / 4)
        if self.pop_size < k:
            # pymoo warns too: fewer members than directions leave directions without one.
            logger.warning(
                "nsga3: pop_size %d is below the %d reference directions",
                self.pop_size,
                k,
                extra={"event": "nsga3_small_population"},
            )
        if self.pop_size < 2:
            raise ValueError(f"pop_size must be at least 2, got {self.pop_size}")
        self.n_new = self.given_offsprings or self.pop_size
        if self.norm is None or self.norm.ideal.size != n_obj:
            self.norm = HyperplaneNormalization(n_obj)

    def _setup(self, problem: Any, draws: Any, cov: FloatArray | None) -> None:  # noqa: ANN401
        names = getattr(problem.objective, "objective_names", None)
        if names is not None and (self.directions is None
                                  or self.directions.shape[1] != len(names)):  # fmt: skip
            self.set_objectives(len(names))
        super()._setup(problem, draws, cov)

    def _refresh_epf_function_value(self) -> None:
        # A new penalty factor changes every penalised objective: the normalisation's ideal,
        # worst and extreme points, kept across generations, came from the old factor's values
        # and may no longer be reachable (numerics review, 2026-10-06), so it starts afresh.
        if self.epf_iteration > 1 and self.iteration == 0 and self.norm is not None:
            self.norm = HyperplaneNormalization(self.norm.ideal.size)
        super()._refresh_epf_function_value()

    def survivors(
        self, F: FloatArray, cv: FloatArray, n_survive: int
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Return the survivors by reference directions, ranks, and no crowding (unused)."""
        assert self.directions is not None
        assert self.norm is not None
        keep, ranks = survive_by_references(self.rng, F, cv, n_survive, self.directions,
                                            self.norm)  # fmt: skip
        return keep, ranks, np.zeros(F.shape[1])

    def parents(self, F: FloatArray, cv: FloatArray, n_select: int) -> IntArray:
        """Return parents by pymoo's ``comp_by_cv_then_random`` tournaments."""
        return tournament_by_violation(self.rng, cv, n_select)

    # ------------------------------------------------------------------ state

    def _export_population_extra(self) -> OptimizerState:
        state = dict(super()._export_population_extra())
        norm = self.norm
        assert norm is not None
        state.update({
            "norm_ideal": norm.ideal, "norm_worst": norm.worst,
            "norm_nadir": None if norm.nadir is None else norm.nadir,
            "norm_extremes": None if norm.extremes is None else norm.extremes,
        })  # fmt: skip
        return state

    def _restore_population_extra(self, state: Mapping[str, StateValue]) -> None:
        super()._restore_population_extra(state)
        assert self.norm is not None
        self.norm.ideal = np.array(state["norm_ideal"], dtype=float)
        self.norm.worst = np.array(state["norm_worst"], dtype=float)
        nadir, extremes = state.get("norm_nadir"), state.get("norm_extremes")
        self.norm.nadir = None if nadir is None else np.array(nadir, dtype=float)
        self.norm.extremes = (
            None if extremes is None
            else np.array(extremes, dtype=float).reshape(self.n_obj, self.n_obj)
        )  # fmt: skip
