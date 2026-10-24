"""Multi-objective kernels: dominance, crowding, NSGA-II survival, hypervolume, IGD.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/util/nds/non_dominated_sorting.py and fast_non_dominated_sort.py (the
             fronts), pymoo/operators/survival/rank_and_crowding/metrics.py
             (``calc_crowding_distance``) and classes.py (``RankAndCrowding``),
             pymoo/core/survival.py (infeasible members after the feasible ones, by violation),
             pymoo/util/randomized_argsort.py, pymoo/algorithms/moo/nsga2.py
             (``binary_tournament``, ``comp_by_dom_and_crowding``)
Reference  : Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist
             multiobjective genetic algorithm: NSGA-II. IEEE TEC, 6, 182-197. Deb, K. (2000).
             An efficient constraint handling method for genetic algorithms. CMAME, 186,
             311-338. Fonseca, C. M., Paquete, L. & Lopez-Ibanez, M. (2006). An improved
             dimension-sweep algorithm for the hypervolume indicator. Proc. IEEE CEC,
             1157-1163. Coello Coello, C. A. & Reyes Sierra, M. (2004). MICAI (IGD).
Differences: objectives are rows ``(n_obj, n)``, members columns (pymoo: ``(n, n_obj)``); the
             kernels draw pymoo's random numbers in its order. The hypervolume is computed
             exactly by slicing (two objectives: a sort; three and four: recursive slicing
             over the sorted first objective), not by pymoo's (moocore's) algorithm; fronts
             here are small. Five or more objectives are refused for it.

Every objective is minimised. A member ``a`` dominates ``b`` when it is no worse in every
objective and better in at least one.
"""

from collections.abc import Sequence

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray, IntArray

#: The most objectives the exact hypervolume takes (cost grows as n^(n_obj - 1)).
MAX_HYPERVOLUME_OBJECTIVES = 4


def dominates(a: FloatArray, b: FloatArray) -> bool:
    """Return whether ``a`` dominates ``b`` (objective vectors ``(n_obj,)``)."""
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_ranks(F: FloatArray) -> IntArray:
    """Return each member's front (0: non-dominated) ``(n,)`` for objectives ``(n_obj, n)``.

    Deb et al.'s fast non-dominated sort; the fronts are pymoo's (``np.where(rank == r)``).
    """
    n = F.shape[1]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    # better[i, j]: i is no worse than j everywhere and better somewhere (i dominates j).
    le = np.all(F[:, :, None] <= F[:, None, :], axis=0)
    lt = np.any(F[:, :, None] < F[:, None, :], axis=0)
    dominates_matrix = le & lt
    dominated_by = dominates_matrix.sum(axis=0)  # how many dominate each member
    ranks = np.full(n, -1, dtype=np.int64)
    current = np.flatnonzero(dominated_by == 0)
    rank = 0
    remaining = dominated_by.copy()
    while current.size:
        ranks[current] = rank
        for i in current:
            remaining[dominates_matrix[i]] -= 1
        remaining[current] = -1  # ranked
        current = np.flatnonzero(remaining == 0)
        rank += 1
    return ranks


def fronts_of(ranks: IntArray) -> list[IntArray]:
    """Return the fronts in order, each its members' indices ascending (pymoo's order)."""
    if ranks.size == 0:
        return []
    return [np.flatnonzero(ranks == r) for r in range(int(ranks.max()) + 1)]


def crowding_distance(F: FloatArray) -> FloatArray:
    """Return NSGA-II's crowding distance of each member of one front ``(m,)``.

    pymoo's ``calc_crowding_distance`` (without filtering duplicates, as its ``"cd"``): per
    objective, the gap between the neighbours in sorted order over the objective's range,
    infinite at the extremes, averaged over the objectives. A front of two or fewer is
    infinite everywhere.
    """
    n_obj, m = F.shape
    if m <= 2:
        return np.full(m, np.inf)
    rows = F.T  # pymoo's (m, n_obj), inside the kernel
    order = np.argsort(rows, axis=0, kind="mergesort")
    sorted_rows = rows[order, np.arange(n_obj)]
    gaps = np.vstack([sorted_rows, np.full(n_obj, np.inf)]) - np.vstack(
        [np.full(n_obj, -np.inf), sorted_rows]
    )
    norm = np.max(sorted_rows, axis=0) - np.min(sorted_rows, axis=0)
    norm[norm == 0] = np.nan
    with np.errstate(invalid="ignore"):
        to_last, to_next = gaps[:-1] / norm, gaps[1:] / norm
    to_last[np.isnan(to_last)] = 0.0
    to_next[np.isnan(to_next)] = 0.0
    back = np.argsort(order, axis=0)
    total = to_last[back, np.arange(n_obj)] + to_next[back, np.arange(n_obj)]
    return np.asarray(np.sum(total, axis=1) / n_obj, dtype=float)


def randomized_argsort(rng: np.random.Generator, values: FloatArray, descending: bool) -> IntArray:
    """Pymoo's ``randomized_argsort`` (``method = "numpy"``): ties broken at random."""
    perm = rng.permutation(len(values))
    order = perm[np.argsort(values[perm], kind="quicksort")]
    return np.asarray(order[::-1] if descending else order, dtype=np.int64)


def rank_and_crowding(
    rng: np.random.Generator, F: FloatArray, n_survive: int
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return NSGA-II's survivors among feasible members, with their rank and crowding.

    pymoo's ``RankAndCrowding._do``: whole fronts in order while they fit, then the best
    crowded of the front that overflows (``randomized_argsort``, descending).

    Parameters
    ----------
    rng
        Draws for the ties of the last front, as pymoo's.
    F
        Objectives ``(n_obj, n)``.
    n_survive
        How many survive.

    Returns
    -------
    tuple
        The survivors' indices into ``F`` ``(<= n_survive,)``, and every member's rank
        ``(n,)`` and crowding distance ``(n,)`` (those of fronts not reached: rank ``-1``,
        crowding ``nan``).
    """
    n = F.shape[1]
    ranks = pareto_ranks(F)
    crowding = np.full(n, np.nan)
    out_rank = np.full(n, -1, dtype=np.int64)
    survivors: list[int] = []
    for k, front in enumerate(fronts_of(ranks)):
        if len(survivors) >= n_survive:
            break
        keep = np.arange(len(front))
        if len(survivors) + len(front) > n_survive:
            n_remove = len(survivors) + len(front) - n_survive
            distance = crowding_distance(F[:, front])
            keep = randomized_argsort(rng, distance, descending=True)[:-n_remove]
        else:
            distance = crowding_distance(F[:, front])
        crowding[front] = distance
        out_rank[front] = k
        survivors.extend(int(i) for i in front[keep])
    return np.asarray(survivors, dtype=np.int64), out_rank, crowding


def survive(
    rng: np.random.Generator, F: FloatArray, cv: FloatArray, n_survive: int
) -> tuple[IntArray, IntArray, FloatArray]:
    """Return NSGA-II's survivors under constraints (pymoo's ``Survival.do``).

    With constraints, the feasible members go through :func:`rank_and_crowding` (at most
    ``n_survive``), then infeasible ones fill up, least violation first. Without any
    infeasible member, all of them do.

    Returns
    -------
    tuple
        The survivors' indices ``(n_survive,)`` (or fewer), each member's rank (infeasible:
        ``-1``) and crowding (infeasible: ``nan``).
    """
    n = F.shape[1]
    feasible = np.flatnonzero(cv <= 0.0)
    ranks = np.full(n, -1, dtype=np.int64)
    crowding = np.full(n, np.nan)
    if feasible.size:
        chosen, r, c = rank_and_crowding(rng, F[:, feasible], min(feasible.size, n_survive))
        ranks[feasible], crowding[feasible] = r, c
        survivors = list(feasible[chosen])
    else:
        survivors = []
    infeasible = np.flatnonzero(cv > 0.0)
    # pymoo's split_by_feasibility(sort_infeas_by_cv=True): NumPy's default sort, as pymoo's
    # (not stable: equal violations keep pymoo's order, not their index order).
    infeasible = infeasible[np.argsort(cv[infeasible])]
    survivors.extend(int(i) for i in infeasible[: max(n_survive - len(survivors), 0)])
    return np.asarray(survivors, dtype=np.int64), ranks, crowding


def tournament(
    rng: np.random.Generator,
    F: FloatArray,
    cv: FloatArray,
    crowding: FloatArray,
    n_select: int,
) -> IntArray:
    """Return ``n_select`` winners of NSGA-II's binary tournaments (pymoo's).

    Competitors come from concatenated random permutations. Of two: if either is infeasible,
    the smaller violation (ties drawn); else the dominating one; else the larger crowding
    distance (ties drawn). ``F`` ``(n_obj, n)``, ``cv`` and ``crowding`` ``(n,)``.
    """
    n = F.shape[1]
    needed = n_select * 2
    perms = np.concatenate([rng.permutation(n) for _ in range(int(np.ceil(needed / n)))])
    pairs = perms[:needed].reshape(n_select, 2)
    winners = np.empty(n_select, dtype=np.int64)
    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        if cv[a] > 0.0 or cv[b] > 0.0:
            winners[i] = _smaller(rng, a, cv[a], b, cv[b])
        elif dominates(F[:, a], F[:, b]):
            winners[i] = a
        elif dominates(F[:, b], F[:, a]):
            winners[i] = b
        else:
            winners[i] = _smaller(rng, a, -crowding[a], b, -crowding[b])
    return winners


def _smaller(rng: np.random.Generator, a: int, a_val: float, b: int, b_val: float) -> int:
    """Pymoo's ``compare(..., "smaller_is_better", return_random_if_equal=True)``."""
    if a_val < b_val:
        return a
    if a_val > b_val:
        return b
    return int(rng.choice([a, b]))


# ---------------------------------------------------------------------------- indicators


def non_dominated(F: FloatArray) -> IntArray:
    """Return the indices of the non-dominated members of ``F`` ``(n_obj, n)``."""
    return np.flatnonzero(pareto_ranks(F) == 0) if F.shape[1] else np.empty(0, dtype=np.int64)


def hypervolume(F: FloatArray, reference: Sequence[float] | FloatArray) -> float:
    """Return the hypervolume dominated by ``F`` ``(n_obj, n)`` and bounded by ``reference``.

    Exact. Points not strictly better than the reference in every objective add nothing.
    Two objectives: a sweep over the first objective; three and four: slices over the first
    objective, each the hypervolume of one fewer dimension (Fonseca et al. 2006's
    dimension sweep, without its data structures).

    Raises
    ------
    ValueError
        With more than :data:`MAX_HYPERVOLUME_OBJECTIVES` objectives.
    """
    ref = np.asarray(reference, dtype=float)
    n_obj = F.shape[0]
    if ref.shape != (n_obj,):
        raise ValueError(f"the reference point has {ref.size} values for {n_obj} objectives")
    if n_obj > MAX_HYPERVOLUME_OBJECTIVES:
        raise ValueError(
            f"the exact hypervolume takes at most {MAX_HYPERVOLUME_OBJECTIVES} objectives"
        )
    inside = F[:, np.all(ref[:, None] > F, axis=0)]
    if inside.shape[1] == 0:
        return 0.0
    return _hv(inside[:, non_dominated(inside)], ref)


def _hv(F: FloatArray, ref: FloatArray) -> float:
    n_obj, n = F.shape
    if n == 0:
        return 0.0
    if n_obj == 1:
        return float(ref[0] - np.min(F[0]))
    if n_obj == 2:
        order = np.argsort(F[0], kind="stable")
        x, y = F[0, order], F[1, order]
        # Strips between consecutive x, each as high as the lowest y so far (to the right of
        # a point, everything above its y and below the reference is dominated).
        volume, best_y = 0.0, ref[1]
        for i in range(n):
            best_y = min(best_y, y[i])
            right = x[i + 1] if i + 1 < n else ref[0]
            volume += (right - x[i]) * (ref[1] - best_y)
        return float(volume)
    # Slice along the first objective: between consecutive values, the cross-section is the
    # (n_obj - 1)-dimensional hypervolume of the points at or below the slice.
    order = np.argsort(F[0], kind="stable")
    F = F[:, order]
    volume = 0.0
    for i in range(n):
        right = F[0, i + 1] if i + 1 < n else ref[0]
        if right <= F[0, i]:
            continue
        section = F[1:, : i + 1]
        volume += (right - F[0, i]) * _hv(section[:, non_dominated(section)], ref[1:])
    return float(volume)


def igd(F: FloatArray, front: FloatArray) -> float:
    """Return the inverted generational distance of ``F`` to a reference ``front``.

    The mean, over the reference front's points ``(n_obj, m)``, of the Euclidean distance to
    the nearest member of ``F`` ``(n_obj, n)``; 0 when ``F`` covers the front.
    """
    distance = np.linalg.norm(front[:, :, None] - F[:, None, :], axis=0)  # (m, n)
    return float(np.mean(np.min(distance, axis=1)))


def front_movement(previous: FloatArray, current: FloatArray) -> float:
    """Return how far a front moved since the previous generation, in normalised objectives.

    pymoo's ``MultiObjectiveSpaceTermination._delta`` (``termination/ftol.py``): with the
    current front's ideal and nadir, the largest of the ideal's and the nadir's changes, each
    objective divided by the current range (1 where the range is degenerate), and the IGD of
    the previous front, normalised by the same ideal and nadir, to the current one. Unlike
    the hypervolume, it needs no reference point, so neither a tight nor a loose one can stop
    a run early.

    Parameters
    ----------
    previous, current
        The fronts' objectives ``(n_obj, m_prev)`` and ``(n_obj, m)``, both non-empty.
    """
    ideal, nadir = np.min(current, axis=1), np.max(current, axis=1)
    old_ideal, old_nadir = np.min(previous, axis=1), np.max(previous, axis=1)
    norm = nadir - ideal
    norm = np.where(norm < 1e-32, 1.0, norm)
    delta_ideal = float(np.max(np.abs((ideal - old_ideal) / norm)))
    delta_nadir = float(np.max(np.abs((nadir - old_nadir) / norm)))
    # pymoo's normalize leaves a degenerate objective unscaled (shifted by the ideal only).
    scale = np.where(nadir > ideal, nadir - ideal, 1.0)[:, None]
    current_n = (current - ideal[:, None]) / scale
    previous_n = (previous - ideal[:, None]) / scale
    return max(delta_ideal, delta_nadir, igd(previous_n, current_n))


def representative(F: FloatArray) -> int:
    """Return the front member closest to the ideal point in normalised objectives.

    Each objective is scaled by the front's range (ideal 0, nadir 1); the member with the
    smallest Euclidean distance to the origin is the compromise a single view shows
    A single member is itself.
    """
    if F.shape[1] == 1:
        return 0
    ideal, nadir = np.min(F, axis=1), np.max(F, axis=1)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    scaled = (F - ideal[:, None]) / span[:, None]
    return int(np.argmin(np.linalg.norm(scaled, axis=0)))
