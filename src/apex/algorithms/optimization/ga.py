"""A real-coded genetic algorithm (``ga``): tournaments, SBX, polynomial mutation, elitism.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/algorithms/soo/nonconvex/ga.py (GA, ``comp_by_cv_and_fitness``,
             ``FitnessSurvival``), pymoo/operators/crossover/sbx.py (``cross_sbx`` and
             ``SimulatedBinaryCrossover._do``), pymoo/core/crossover.py (``Crossover.do``),
             pymoo/operators/mutation/pm.py (``mut_pm``), pymoo/core/mutation.py
             (``Mutation.do``), pymoo/operators/selection/tournament.py, pymoo/core/mating.py
             and pymoo/core/infill.py (the mating loop with duplicate elimination)
Reference  : Deb, K. & Agrawal, R. B. (1995). Simulated binary crossover for continuous search
             space. Complex Systems, 9, 115-148. Deb, K. & Goyal, M. (1996). A combined genetic
             adaptive search (GeneAS) for engineering design. Computer Science and
             Informatics, 26, 30-45. Deb, K. (2000). An efficient constraint handling method
             for genetic algorithms. CMAME, 186, 311-338.
Differences: members are columns ``(n_x, P)`` (pymoo: rows); the operators draw exactly
             pymoo's random numbers in pymoo's order and shapes, then transpose, so given the
             same ``numpy.random.Generator`` they return pymoo's children (the operator
             reference tests). ``pop_size`` defaults to 20 (pymoo: 100), because
             evaluations here are simulator runs; crossover and mutation probabilities are
             options. A tournament of more than two members (``tournament > 2``) takes the
             best under the ranking (pymoo's comparison is binary). The mating loop stops after
             100 attempts, as pymoo's, with fewer offspring if duplicates remain. Duplicates
             are judged by the largest coordinate difference (pymoo: the Euclidean distance;
             at its 1e-16 the two agree).

One generation: select ``ceil(n_offsprings / 2)`` pairs by tournament; SBX each pair with
probability ``crossover_prob`` (each variable with ``crossover_prob_var``); mutate each child
with probability ``mutation_prob`` (each variable with ``mutation_prob_var``); drop
duplicates; evaluate; keep the best ``pop_size`` of parents and offspring (feasible first).
"""

import math
from typing import ClassVar

import numpy as np

from apex.algorithms.optimization.base import EpfOptions
from apex.algorithms.optimization.population import (
    ConstraintHandling,
    Initialization,
    PopulationOptimizer,
    rank_order,
)
from apex.algorithms.optimization.protocols import FloatArray, IntArray, Status
from apex.algorithms.optimization.registries import optimizers

#: pymoo's tolerance for SBX on equal parents, and for duplicates.
SBX_EPS = 1.0e-14
DUPLICATE_EPS = 1.0e-16
#: pymoo's limit on mating attempts per generation.
MATING_ATTEMPTS = 100


def sbx(
    rng: np.random.Generator,
    p1: FloatArray,
    p2: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    eta: float = 15.0,
    prob: float = 0.9,
    prob_var: float = 0.5,
    prob_exch: float = 1.0,
    prob_bin: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """Return SBX's two children of each pair of parents (pymoo's ``Crossover.do`` with SBX).

    Parameters
    ----------
    rng
        The generator; drawn from as pymoo does (order and shapes).
    p1, p2
        The pairs' parents ``(n_x, m)``.
    lower, upper
        Bounds ``(n_x,)``.
    eta
        Distribution index.
    prob
        Probability that a pair is crossed at all (else the children are the parents).
    prob_var
        Probability that a variable of a crossed pair takes part.
    prob_exch, prob_bin
        pymoo's exchange of the two children (per pair, then per variable).

    Returns
    -------
    tuple of FloatArray
        The children ``(n_x, m)`` each, clamped to the bounds.
    """
    X = np.stack([p1.T, p2.T]).astype(float)  # pymoo's (2, m, n_x)
    m = X.shape[1]
    cross = rng.random(m) < prob  # Crossover.do
    Q = np.copy(X)
    if np.any(cross):
        # SimulatedBinaryCrossover._do
        eta_ = np.full((m, 1), float(eta))
        prob_var_ = np.full((m, 1), float(prob_var))
        prob_bin_ = np.full((m, 1), float(prob_bin))
        rand = rng.random((m, 1))
        prob_bin_[rand > prob_exch] = 0.0
        crossed = _cross_sbx(rng, X, lower, upper, eta_, prob_var_, prob_bin_)
        Q[:, cross] = crossed[:, cross]
    return np.ascontiguousarray(Q[0].T), np.ascontiguousarray(Q[1].T)


def _cross_sbx(
    rng: np.random.Generator,
    X: FloatArray,
    xl: FloatArray,
    xu: FloatArray,
    eta: FloatArray,
    prob_var: FloatArray,
    prob_bin: FloatArray,
) -> FloatArray:
    """Pymoo's ``cross_sbx`` on ``(2, m, n_x)`` parents (its layout, inside the operator)."""
    _, m, n_x = X.shape
    cross = rng.random((m, n_x)) < prob_var
    cross[np.abs(X[0] - X[1]) <= SBX_EPS] = False
    cross[:, xl == xu] = False
    p1, p2 = X[0][cross], X[1][cross]
    sm = p1 < p2
    y1, y2 = np.where(sm, p1, p2), np.where(sm, p2, p1)
    _xl = np.repeat(xl[None, :], m, axis=0)[cross]
    _xu = np.repeat(xu[None, :], m, axis=0)[cross]
    eta_ = eta.repeat(n_x, axis=1)[cross]
    prob_bin_ = prob_bin.repeat(n_x, axis=1)[cross]
    rand = rng.random(len(eta_))

    def betaq_of(beta: FloatArray) -> FloatArray:
        alpha = 2.0 - np.power(beta, -(eta_ + 1.0))
        inside = rand <= 1.0 / alpha
        betaq = np.zeros(inside.shape)
        betaq[inside] = np.power(rand * alpha, 1.0 / (eta_ + 1.0))[inside]
        betaq[~inside] = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (eta_ + 1.0))[~inside]
        return betaq

    delta = y2 - y1
    with np.errstate(divide="ignore", invalid="ignore"):
        c1 = 0.5 * ((y1 + y2) - betaq_of(1.0 + 2.0 * (y1 - _xl) / delta) * delta)
        c2 = 0.5 * ((y1 + y2) + betaq_of(1.0 + 2.0 * (_xu - y2) / delta) * delta)
    child1, child2 = np.where(sm, c1, c2), np.where(sm, c2, c1)
    swap = np.bitwise_xor(rng.random(len(prob_bin_)) < prob_bin_, X[0, cross] > X[1, cross])
    child1, child2 = np.where(swap, child2, child1), np.where(swap, child1, child2)
    Q = np.copy(X)
    Q[0, cross] = child1
    Q[1, cross] = child2
    return np.clip(Q, xl, xu)  # pymoo's repair_clamp


def polynomial_mutation(
    rng: np.random.Generator,
    X: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    eta: float = 20.0,
    prob: float = 0.9,
    prob_var: float | None = None,
) -> FloatArray:
    """Return the members ``(n_x, n)`` after pymoo's polynomial mutation (``Mutation.do``).

    Each member is mutated with probability ``prob``; each of its variables with
    ``prob_var`` (default pymoo's ``min(0.5, 1 / n_x)``). Draws as pymoo does.
    """
    rows = np.ascontiguousarray(X.T, dtype=float)  # pymoo's (n, n_x)
    n, n_x = rows.shape
    p_var = min(0.5, 1.0 / n_x) if prob_var is None else prob_var
    mutated = _mut_pm(rng, rows, lower, upper, eta, p_var)
    chosen = rng.random(n) <= prob  # Mutation.do, after _do
    out = np.copy(rows)
    out[chosen] = mutated[chosen]
    return np.ascontiguousarray(out.T)


def _mut_pm(
    rng: np.random.Generator,
    X: FloatArray,
    xl: FloatArray,
    xu: FloatArray,
    eta: float,
    prob_var: float,
) -> FloatArray:
    """Pymoo's ``mut_pm`` on ``(n, n_x)`` members (its layout, inside the operator)."""
    n, n_x = X.shape
    mut = rng.random((n, n_x)) < prob_var  # mut_binomial, at_least_once = False
    mut[:, xl == xu] = False
    Xp = np.copy(X)
    _xl = np.repeat(xl[None, :], n, axis=0)[mut]
    _xu = np.repeat(xu[None, :], n, axis=0)[mut]
    x = X[mut]
    delta1 = (x - _xl) / (_xu - _xl)
    delta2 = (_xu - x) / (_xu - _xl)
    mut_pow = 1.0 / (eta + 1.0)
    rand = rng.random(x.shape)
    low = rand <= 0.5
    deltaq = np.zeros(x.shape)
    with np.errstate(invalid="ignore"):
        val = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(1.0 - delta1, eta + 1.0)
        deltaq[low] = (np.power(val, mut_pow) - 1.0)[low]
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(1.0 - delta2, eta + 1.0)
        deltaq[~low] = (1.0 - np.power(val, mut_pow))[~low]
    y = np.clip(x + deltaq * (_xu - _xl), _xl, _xu)
    Xp[mut] = y
    return np.clip(Xp, xl, xu)


def tournament(
    rng: np.random.Generator,
    value: FloatArray,
    cv: FloatArray,
    n_select: int,
    *,
    size: int = 2,
) -> IntArray:
    """Return ``n_select`` winners' indices (pymoo's ``TournamentSelection`` and ranking).

    Competitors come from concatenated random permutations of the population; of two, the
    smaller violation wins if either is infeasible, else the smaller value; ties are drawn
    (pymoo's ``comp_by_cv_and_fitness``). Of more than two, the best under the ranking.
    """
    n = value.size
    needed = n_select * size
    perms = np.concatenate([rng.permutation(n) for _ in range(math.ceil(needed / n))])
    pairs = perms[:needed].reshape(n_select, size)
    winners = np.empty(n_select, dtype=np.int64)
    for i, group in enumerate(pairs):
        if size == 2:
            a, b = int(group[0]), int(group[1])
            if cv[a] > 0.0 or cv[b] > 0.0:
                ka, kb = cv[a], cv[b]
            else:
                ka, kb = value[a], value[b]
            if ka < kb:
                winners[i] = a
            elif ka > kb:
                winners[i] = b
            else:
                winners[i] = int(rng.choice([a, b]))
        else:
            order = rank_order(value[group], cv[group])
            winners[i] = int(group[order[0]])
    return winners


def new_columns(new: FloatArray, *others: FloatArray) -> IntArray:
    """Return the columns of ``new`` that are new.

    Equal (within pymoo's 1e-16) to none of ``others`` nor to an earlier column of ``new``.
    """
    keep: list[int] = []
    seen = [o for o in others if o.size]
    for j in range(new.shape[1]):
        column = new[:, j : j + 1]
        if any(np.any(np.max(np.abs(o - column), axis=0) <= DUPLICATE_EPS) for o in seen):
            continue
        if keep and np.any(np.max(np.abs(new[:, keep] - column), axis=0) <= DUPLICATE_EPS):
            continue
        keep.append(j)
    return np.asarray(keep, dtype=np.int64)


@optimizers.register("ga")
class GeneticAlgorithm(PopulationOptimizer):
    """A real-coded genetic algorithm: SBX crossover, polynomial mutation, tournaments.

    Parameters
    ----------
    n_offsprings
        Children per generation (default ``pop_size``).
    tournament
        Tournament size (pymoo: 2).
    crossover_prob, crossover_prob_var, eta_c
        SBX: per pair, per variable, the distribution index.
    mutation_prob, mutation_prob_var, eta_m
        Polynomial mutation: per member, per variable (default ``min(0.5, 1/n_x)``), index.
    eliminate_duplicates
        Redraw children equal to a member or to each other.
    pop_size, stall, init, constraint_handling, eq_tolerance, maxiter, ftol, xtol,
    transform, epf
        See :class:`~apex.algorithms.optimization.population.PopulationOptimizer`.
    """

    status: ClassVar[Status] = "experimental"
    label: ClassVar[str] = "GA"

    def __init__(
        self,
        *,
        pop_size: int = 20,
        n_offsprings: int | None = None,
        tournament: int = 2,
        crossover_prob: float = 0.9,
        crossover_prob_var: float = 0.5,
        eta_c: float = 15.0,
        mutation_prob: float = 0.9,
        mutation_prob_var: float | None = None,
        eta_m: float = 20.0,
        eliminate_duplicates: bool = True,
        stall: int = 20,
        init: Initialization = "uniform",
        constraint_handling: ConstraintHandling = "feasibility",
        eq_tolerance: float = 1e-4,
        maxiter: int = 100,
        ftol: float = 1e-6,
        xtol: float = 1e-8,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        super().__init__(
            pop_size=pop_size, stall=stall, init=init,
            constraint_handling=constraint_handling, eq_tolerance=eq_tolerance,
            maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf,
        )  # fmt: skip
        if tournament < 2:
            raise ValueError(f"tournament must be at least 2, got {tournament}")
        for name, p in (("crossover_prob", crossover_prob),
                        ("crossover_prob_var", crossover_prob_var),
                        ("mutation_prob", mutation_prob)):  # fmt: skip
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {p}")
        if mutation_prob_var is not None and not 0.0 <= mutation_prob_var <= 1.0:
            raise ValueError(f"mutation_prob_var must lie in [0, 1], got {mutation_prob_var}")
        if eta_c <= 0 or eta_m <= 0:
            raise ValueError("eta_c and eta_m must be positive")
        self.n_new = pop_size if n_offsprings is None else n_offsprings
        if self.n_new < 1:
            raise ValueError(f"n_offsprings must be positive, got {self.n_new}")
        self.tournament = tournament
        self.crossover_prob = crossover_prob
        self.crossover_prob_var = crossover_prob_var
        self.eta_c = eta_c
        self.mutation_prob = mutation_prob
        self.mutation_prob_var = mutation_prob_var
        self.eta_m = eta_m
        self.eliminate_duplicates = eliminate_duplicates

    def generation(self) -> None:
        """Mate, evaluate the children, and keep the best ``pop_size`` (elitist survival)."""
        children = self.offspring()
        if children.shape[1] == 0:
            return  # every child a duplicate: the population stays (it will stall)
        F, c_eq, c_iq = self.evaluate_population(children)
        X = np.column_stack([self.X, children])
        F = np.column_stack([self.F, F])
        c_eq, c_iq = np.column_stack([self.c_eq, c_eq]), np.column_stack([self.c_iq, c_iq])
        value, cv, _ = self.scores(F, c_eq, c_iq)
        keep = rank_order(value, cv)[: self.pop_size]  # pymoo's FitnessSurvival
        self._set_population(X[:, keep], F[:, keep], c_eq[:, keep], c_iq[:, keep])

    def offspring(self) -> FloatArray:
        """Return this generation's children ``(n_x, <= n_offsprings)`` (pymoo's mating loop)."""
        value, cv, _ = self.scores(self.F, self.c_eq, self.c_iq)
        children = np.empty((self.X.shape[0], 0))
        for _ in range(MATING_ATTEMPTS):
            remaining = self.n_new - children.shape[1]
            if remaining <= 0:
                break
            n_matings = math.ceil(remaining / 2)
            parents = tournament(self.rng, value, cv, 2 * n_matings, size=self.tournament)
            parents = parents.reshape(n_matings, 2)
            c1, c2 = sbx(
                self.rng, self.X[:, parents[:, 0]], self.X[:, parents[:, 1]], self.xl,
                self.xu, eta=self.eta_c, prob=self.crossover_prob,
                prob_var=self.crossover_prob_var,
            )  # fmt: skip
            new = np.column_stack([c1, c2])  # pymoo's order: the first children, then the second
            new = polynomial_mutation(
                self.rng, new, self.xl, self.xu, eta=self.eta_m, prob=self.mutation_prob,
                prob_var=self.mutation_prob_var,
            )  # fmt: skip
            if self.eliminate_duplicates:
                new = new[:, new_columns(new, self.X, children)]
            children = np.column_stack([children, new[:, :remaining]])
        return np.ascontiguousarray(children)
