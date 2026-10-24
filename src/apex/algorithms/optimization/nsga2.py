"""NSGA-II (``nsga2``): elitist non-dominated sorting with crowding.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/algorithms/moo/nsga2.py (NSGA2: binary tournaments by dominance and
             crowding, RankAndCrowding survival, SBX and polynomial mutation), with the
             kernels of :mod:`.multiobjective` and :mod:`.ga`
Reference  : Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist
             multiobjective genetic algorithm: NSGA-II. IEEE TEC, 6, 182-197.
Differences: ``pop_size`` defaults to 40 (pymoo: 100), for simulator cost; ``x0`` is a member
             of the initial population; the run ends when the front has stalled (``stall``
             generations in a row in which it moved less than ``ftol``, measured as pymoo's
             ``MultiObjectiveSpaceTermination``) or at ``maxiter`` (pymoo's default
             termination applies the same measure every sixth generation with
             ``ftol = 0.005`` over a window of 50).
             The external penalty (``constraint_handling = "epf"``) adds the same penalty to
             every objective; pymoo has none. A robust objective's value is each objective's
             mean over the realizations.

The answer is a front: :meth:`NSGA2.front` returns the non-dominated feasible members (or the
least infeasible member while none is feasible). ``x_k`` is its representative, the member
closest to the ideal point in normalised objectives (:func:`.multiobjective.representative`),
so that single-value views keep working. The hypervolume is measured against a
reference point fixed when the first feasible members appear (``ref_point``, or their nadir
plus 10 % of each objective's range) and checkpointed, so a run's values are comparable.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import numpy as np

from apex.algorithms.optimization.base import EpfOptions, external_penalty
from apex.algorithms.optimization.ga import (
    MATING_ATTEMPTS,
    new_columns,
    polynomial_mutation,
    sbx,
)
from apex.algorithms.optimization.multiobjective import (
    MAX_HYPERVOLUME_OBJECTIVES,
    front_movement,
    hypervolume,
    non_dominated,
    representative,
    survive,
    tournament,
)
from apex.algorithms.optimization.population import (
    ConstraintHandling,
    Initialization,
    PopulationOptimizer,
    violation,
)
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    Draws,
    FloatArray,
    IntArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
    VectorObjective,
)
from apex.algorithms.optimization.registries import optimizers
from apex.algorithms.optimization.statistics import (
    MEAN,
    needs_realizations,
    reduce_realizations,
)

#: The reference point's margin beyond the first feasible members' nadir (fraction of range).
REFERENCE_MARGIN = 0.1


@optimizers.register("nsga2")
class NSGA2(PopulationOptimizer):
    """NSGA-II: multi-objective optimization by non-dominated sorting and crowding.

    Parameters
    ----------
    pop_size
        Members (default 40).
    n_offsprings
        Children per generation (default ``pop_size``).
    crossover_prob, crossover_prob_var, eta_c, mutation_prob, mutation_prob_var, eta_m
        SBX and polynomial mutation, as ``ga``'s.
    eliminate_duplicates
        Redraw children equal to a member or to each other.
    ref_point
        The hypervolume's reference point, one value per objective (default: from the first
        feasible members).
    stall, ftol
        Stop after ``stall`` generations in a row (default 30) in which the feasible front
        moved less than ``ftol`` (default 0.0025, pymoo's tolerance of the measure):
        :func:`.multiobjective.front_movement`, its ideal and nadir points and its IGD to
        the previous front, in normalised objectives. Before any member is feasible, a
        generation counts as stalled unless the least violation fell.
    init, constraint_handling, eq_tolerance, maxiter, xtol, transform, epf
        See :class:`~apex.algorithms.optimization.population.PopulationOptimizer`.
    """

    status: ClassVar[Status] = "experimental"
    label: ClassVar[str] = "NSGA-II"
    multi_objective: ClassVar[bool] = True

    def __init__(
        self,
        *,
        pop_size: int = 40,
        n_offsprings: int | None = None,
        crossover_prob: float = 0.9,
        crossover_prob_var: float = 0.5,
        eta_c: float = 15.0,
        mutation_prob: float = 0.9,
        mutation_prob_var: float | None = None,
        eta_m: float = 20.0,
        eliminate_duplicates: bool = True,
        ref_point: Sequence[float] | None = None,
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
        super().__init__(
            pop_size=pop_size, stall=stall, init=init,
            constraint_handling=constraint_handling, eq_tolerance=eq_tolerance,
            maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf,
        )  # fmt: skip
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
        self.crossover_prob = crossover_prob
        self.crossover_prob_var = crossover_prob_var
        self.eta_c = eta_c
        self.mutation_prob = mutation_prob
        self.mutation_prob_var = mutation_prob_var
        self.eta_m = eta_m
        self.eliminate_duplicates = eliminate_duplicates
        self.given_ref = None if ref_point is None else [float(v) for v in ref_point]

    # ------------------------------------------------------------------ set-up

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        if not isinstance(problem.objective, VectorObjective):
            raise ValueError(
                f"{self.label} needs several objectives: an [objective] plugin that returns "
                "several "
                "(e.g. zdt1), or a list [[objectives]]"
            )
        super()._setup(problem, draws, cov)
        self.names = tuple(problem.objective.objective_names)
        self.n_obj = len(self.names)
        if self.n_obj < 2:
            raise ValueError(f"{self.label} needs at least two objectives")
        # Each objective's statistic over the realizations (default the mean).
        self.statistics = tuple(
            getattr(problem.objective, "objective_statistics", (MEAN,) * self.n_obj)
        )
        if problem.n_realizations < needs_realizations(self.statistics):
            raise ValueError(
                "the statistic std needs at least two realizations, and this problem has "
                f"{problem.n_realizations}"
            )
        if self.given_ref is not None and len(self.given_ref) != self.n_obj:
            raise ValueError(
                f"ref_point has {len(self.given_ref)} values for {self.n_obj} objectives"
            )
        n_r = problem.n_realizations
        self.F = np.empty((self.n_obj, n_r, 0))
        self.ranks = np.empty(0, dtype=np.int64)
        self.crowding = np.empty(0)
        self.ref: FloatArray | None = None if self.given_ref is None else np.array(self.given_ref)
        self.hv = 0.0
        # The last generation's feasible front, for the stall; and before any member is
        # feasible, the least violation so far (progress then, as phase 1's).
        self.previous_front: FloatArray | None = None
        self.best_cv = math.inf
        # This generation's movement, logged for the monitor; none
        # without two feasible fronts in a row. Display only: it decides nothing.
        self.movement: float | None = None

    # ------------------------------------------------------------------ evaluation

    def evaluate_population(self, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate members ``(n_x, n)`` on every realization, in one batch of ``n n_r``.

        Returns the objectives ``(n_obj, n_r, n)`` and the constraints ``(n_eq, n)`` and
        ``(n_iq, n)``.
        """
        n = X.shape[1]
        n_r = self.problem.n_realizations
        physical = self.bounds.to_physical(self.bounds.project(X))
        controls = np.ascontiguousarray(np.repeat(physical, n_r, axis=1))
        realizations = np.tile(np.arange(n_r, dtype=np.int64), n)
        values = np.asarray(self.problem.objective(controls, realizations), dtype=float)
        if values.shape != (self.n_obj, n * n_r):
            raise ValueError(
                f"the objectives returned shape {values.shape} for {self.n_obj} objectives and "
                f"{n * n_r} points"
            )
        if not np.all(np.isfinite(values)):
            bad = int(np.sum(~np.isfinite(values)))
            raise ValueError(
                f"the objectives returned {bad} non-finite value(s) for {self.label}'s members; "
                "check the objectives (or their simulations) at those controls"
            )
        self.member_evaluations += n * n_r
        self.nfev += n
        F = np.ascontiguousarray(values.reshape(self.n_obj, n, n_r).transpose(0, 2, 1))
        if self.constrained:
            objective = cast(ConstrainedObjective, self.problem.objective)
            c_eq, c_iq = (np.asarray(c, dtype=float).reshape(-1, n)
                          for c in objective.constraints(physical))  # fmt: skip
        else:
            c_eq, c_iq = np.empty((0, n)), np.empty((0, n))
        return F, np.ascontiguousarray(c_eq), np.ascontiguousarray(c_iq)

    def means(
        self, F: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return each member's ranked objectives ``(n_obj, n)`` and violation ``(n,)``.

        The objectives are each one's statistic over the realizations (the mean by
        default), plus the penalty with EPF (the same for every objective); the violation is
        pymoo's sum, zero with EPF.
        """
        mean = self.reduce(F)
        n = mean.shape[1]
        if self.constraint_handling == "epf":
            penalty = external_penalty(self.epf["r"], c_eq, c_iq) if self.constrained \
                else np.zeros(n)  # fmt: skip
            return mean + penalty[None, :], np.zeros(n)
        cv = violation(c_eq, c_iq, self.eq_tolerance) if self.constrained else np.zeros(n)
        return mean, cv

    def set_objectives(self, n_obj: int) -> None:
        """Learn the number of objectives before the run (NSGA-III sizes itself from it)."""

    def survivors(
        self, F: FloatArray, cv: FloatArray, n_survive: int
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Return the survivors, each member's rank and crowding (NSGA-II's survival)."""
        return survive(self.rng, F, cv, n_survive)

    def parents(self, F: FloatArray, cv: FloatArray, n_select: int) -> IntArray:
        """Return ``n_select`` parents by NSGA-II's binary tournaments."""
        return tournament(self.rng, F, cv, self.crowding, n_select)

    def reduce(self, F: FloatArray) -> FloatArray:
        """Return each objective's statistic over the realizations ``(n_obj, n)``."""
        statistics = getattr(self, "statistics", (MEAN,) * F.shape[0])
        return reduce_realizations(F, statistics)

    def scores(
        self, F: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Refuse: the base's scalar ranking does not apply (``means`` and the fronts do)."""
        raise NotImplementedError("nsga2 ranks by fronts: see means()")

    def _set_population(
        self, X: FloatArray, F: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> None:
        self.X, self.F = np.ascontiguousarray(X), np.ascontiguousarray(F)
        self.c_eq, self.c_iq = np.ascontiguousarray(c_eq), np.ascontiguousarray(c_iq)

    # ------------------------------------------------------------------ the loop

    def initialize(self) -> None:
        """Rank the first population and order it as pymoo's survival after the infill does."""
        mean, cv = self.means(self.F, self.c_eq, self.c_iq)
        keep, ranks, crowding = self.survivors(mean, cv, mean.shape[1])
        self._set_population(self.X[:, keep], self.F[:, :, keep], self.c_eq[:, keep],
                             self.c_iq[:, keep])  # fmt: skip
        self.ranks, self.crowding = ranks[keep], crowding[keep]

    def true_violation(self) -> FloatArray:
        """Return each member's constraint violation ``(n,)``, under EPF too (pymoo's sum)."""
        n = self.X.shape[1]
        if not self.constrained:
            return np.zeros(n)
        return violation(self.c_eq, self.c_iq, self.eq_tolerance)

    def generation(self) -> None:
        """Mate by tournaments, evaluate the children, keep the best by rank and crowding."""
        children = self.offspring()
        if children.shape[1] == 0:
            return
        F, c_eq, c_iq = self.evaluate_population(children)
        X = np.column_stack([self.X, children])
        F = np.concatenate([self.F, F], axis=2)
        c_eq, c_iq = np.column_stack([self.c_eq, c_eq]), np.column_stack([self.c_iq, c_iq])
        mean, cv = self.means(F, c_eq, c_iq)
        keep, ranks, crowding = self.survivors(mean, cv, self.pop_size)
        self._set_population(X[:, keep], F[:, :, keep], c_eq[:, keep], c_iq[:, keep])
        self.ranks, self.crowding = ranks[keep], crowding[keep]

    def offspring(self) -> FloatArray:
        """Return this generation's children (pymoo's mating loop, NSGA-II's tournaments)."""
        mean, cv = self.means(self.F, self.c_eq, self.c_iq)
        children = np.empty((self.X.shape[0], 0))
        for _ in range(MATING_ATTEMPTS):
            remaining = self.n_new - children.shape[1]
            if remaining <= 0:
                break
            n_matings = math.ceil(remaining / 2)
            parents = self.parents(mean, cv, 2 * n_matings)
            parents = parents.reshape(n_matings, 2)
            c1, c2 = sbx(
                self.rng, self.X[:, parents[:, 0]], self.X[:, parents[:, 1]], self.xl,
                self.xu, eta=self.eta_c, prob=self.crossover_prob,
                prob_var=self.crossover_prob_var,
            )  # fmt: skip
            new = polynomial_mutation(
                self.rng, np.column_stack([c1, c2]), self.xl, self.xu, eta=self.eta_m,
                prob=self.mutation_prob, prob_var=self.mutation_prob_var,
            )  # fmt: skip
            if self.eliminate_duplicates:
                new = new[:, new_columns(new, self.X, children)]
            children = np.column_stack([children, new[:, :remaining]])
        return np.ascontiguousarray(children)

    def front_indices(self) -> np.ndarray:
        """Return the front's indices.

        The non-dominated feasible members, each control vector once, else (none feasible)
        the least infeasible one. Feasible means the constraints hold, under EPF too: a
        penalised member that violates them is never reported as Pareto-optimal.
        """
        mean, _ = self.means(self.F, self.c_eq, self.c_iq)
        cv = self.true_violation()
        feasible = np.flatnonzero(cv <= 0.0)
        if feasible.size == 0:
            return np.array([int(np.argmin(cv))], dtype=np.int64)
        front = feasible[non_dominated(mean[:, feasible])]
        # Each member once: MOEA/D copies a child into every neighbour it improves.
        _, first = np.unique(self.X[:, front].T, axis=0, return_index=True)
        return np.asarray(front[np.sort(first)], dtype=np.int64)

    def _take_best(self, *, first: bool) -> None:
        """Commit the front's representative as ``x_k``; measure the hypervolume and stall.

        The stall needs no reference point (review, 2026-10-06): a hypervolume against a
        fixed one stalled early when the point was too loose (a wide first population) or
        too tight (a first front of one point), so the front's own movement is measured.
        """
        mean, _ = self.means(self.F, self.c_eq, self.c_iq)
        cv = self.true_violation()
        front = self.front_indices()
        best = int(front[representative(mean[:, front])])
        feasible = bool(cv[best] <= 0.0)
        if self.ref is None and feasible:
            # Every feasible member's nadir, not only the front's: a first front of one or
            # two points would give a box later fronts outgrow, and add nothing.
            self.ref = self._reference(mean[:, cv <= 0.0])
        self.hv = (hypervolume(mean[:, front], self.ref)
                   if self.ref is not None and feasible and self.n_obj <= MAX_HYPERVOLUME_OBJECTIVES
                   else 0.0)  # fmt: skip
        least = float(np.min(cv))
        current = np.ascontiguousarray(mean[:, front]) if feasible else None
        self.movement = (None if current is None or self.previous_front is None
                         else front_movement(self.previous_front, current))  # fmt: skip
        if not first:
            if current is None:
                improved = least < self.best_cv
            elif self.movement is None:
                improved = True  # the first feasible front
            else:
                improved = self.movement > self.ftol
            self.stalled = 0 if improved else self.stalled + 1
        self.previous_front = current
        self.best_cv = min(self.best_cv, least)
        x = np.ascontiguousarray(self.X[:, best].copy())
        fk = np.ascontiguousarray(self.F[:, :, best].copy())  # (n_obj, n_r)
        if self.epf:
            penalty = float(external_penalty(self.epf["r"], self.c_eq[:, [best]],
                                             self.c_iq[:, [best]])[0])  # fmt: skip
            self.last_penalty = np.full(self.problem.n_realizations, penalty)
        if first:
            self.xk, self.fk = x, fk
        else:
            self._commit_step(x, fk)

    def _reference(self, front: FloatArray) -> FloatArray:
        """Return the reference point: the front's nadir plus 10 % of each objective's range."""
        ideal, nadir = np.min(front, axis=1), np.max(front, axis=1)
        span = nadir - ideal
        span = np.where(span > 0, span, np.maximum(np.abs(nadir), 1.0))
        return np.asarray(nadir + REFERENCE_MARGIN * span, dtype=float)

    def _refresh_epf_function_value(self) -> None:
        # A new penalty factor re-ranks the stored members (nothing is evaluated) and, as the
        # objectives changed, starts the hypervolume afresh under the new penalty.
        if self.epf_iteration <= 1 or self.iteration != 0:
            return
        self.initialize()
        self.previous_front = None
        self.best_cv = math.inf
        self.stalled = 0
        self._take_best(first=True)

    def best_index(self) -> int:
        """Return the representative's index."""
        mean, _ = self.means(self.F, self.c_eq, self.c_iq)
        front = self.front_indices()
        return int(front[representative(mean[:, front])])

    # ------------------------------------------------------------------ convergence and logs

    def check_function_convergence(self) -> bool:
        """Converged when the front has moved less than ``ftol`` for ``stall`` generations."""
        if self.stalled >= self.stall:
            self.conv_msg = (
                f"The front moved less than {self.ftol} (normalised) for {self.stall} generations"
            )
            return True
        return False

    def front(self) -> dict[str, Any]:
        """Return the front as JSON values: physical controls, objectives, violation.

        ``x`` ``[m][n_x]``, ``objectives`` ``[m][n_obj]`` (each one's statistic over the
        realizations, the mean by default),
        ``objectives_per_realization`` ``[m][n_obj][n_r]``, ``violation`` ``[m]``, the
        representative's position, the objectives' names, the hypervolume and its
        reference point.
        """
        mean, _ = self.means(self.F, self.c_eq, self.c_iq)
        front = self.front_indices()
        raw = self.reduce(self.F)
        cv = violation(self.c_eq, self.c_iq, self.eq_tolerance) if self.constrained \
            else np.zeros(self.X.shape[1])  # fmt: skip
        physical = self.bounds.to_physical(self.bounds.project(self.X[:, front]))
        return {
            "objective_names": list(self.names),
            "objective_statistics": list(self.statistics),
            "x": physical.T.tolist(),
            "objectives": raw[:, front].T.tolist(),
            "objectives_per_realization": self.F[:, :, front].transpose(2, 0, 1).tolist(),
            "violation": cv[front].tolist(),
            "representative": int(representative(mean[:, front])),
            "hypervolume": self.hv,
            "reference_point": None if self.ref is None else self.ref.tolist(),
        }

    def log_columns(self) -> dict[str, float]:
        """Return the representative's objectives (statistics), hypervolume and front size."""
        columns: dict[str, float] = {"iteration": self.iteration}
        if not self.F.size:
            return columns
        rep = self.reduce(cast(FloatArray, self.fk)[:, :, None])[:, 0]
        columns["objective"] = float(rep[0])
        for name, value in zip(self.names, rep, strict=True):
            columns[f"objective:{name}"] = float(value)
        columns["hypervolume"] = self.hv
        columns["front_size"] = float(self.front_indices().size)
        columns["spread"] = self.spread()
        # The stall's own measures, for the monitor: the front's movement and the
        # generations in a row without progress (what the stall counts to ``stall``).
        if self.movement is not None:
            columns["front_movement"] = self.movement
        columns["stalled_generations"] = float(self.stalled)
        if self.constrained:
            feasible = violation(self.c_eq, self.c_iq, self.eq_tolerance) <= 0.0
            columns["feasible_share"] = float(np.mean(feasible))
        return columns

    def result_extra(self) -> dict[str, float]:
        """Return the hypervolume, the front's size, and the representative's violation."""
        if not self.F.size:
            return {}
        extra = {"hypervolume": self.hv, "front_size": float(self.front_indices().size)}
        if self.constrained:
            cv = violation(self.c_eq, self.c_iq, self.eq_tolerance)
            extra["violation"] = float(cv[self.best_index()])
            extra["feasible"] = float(extra["violation"] <= 0.0)
        return extra

    # ------------------------------------------------------------------ state

    def _export_population_extra(self) -> OptimizerState:
        return {
            "ranks": self.ranks.astype(np.float64),
            "crowding": np.nan_to_num(self.crowding, nan=-1.0, posinf=np.finfo(float).max),
            "crowding_inf": np.isposinf(self.crowding).astype(np.float64),
            "crowding_nan": np.isnan(self.crowding).astype(np.float64),
            "reference": None if self.ref is None else self.ref,
            "previous_front": self.previous_front,
            "best_cv": None if math.isinf(self.best_cv) else self.best_cv,
            "hv": self.hv,
        }

    def _restore_population_extra(self, state: Mapping[str, StateValue]) -> None:
        def array(key: str) -> FloatArray:
            return np.array(state[key], dtype=float)

        self.ranks = array("ranks").astype(np.int64)
        crowding = array("crowding")
        crowding[array("crowding_inf") > 0] = np.inf
        crowding[array("crowding_nan") > 0] = np.nan
        self.crowding = crowding
        reference = state.get("reference")
        self.ref = None if reference is None else array("reference")
        previous = state.get("previous_front")
        self.previous_front = (
            None
            if previous is None
            else np.ascontiguousarray(array("previous_front").reshape(self.n_obj, -1))
        )
        self.hv = float(cast(float, state["hv"]))
        best_cv = state.get("best_cv")
        self.best_cv = math.inf if best_cv is None else float(cast(float, best_cv))
        n_r = self.problem.n_realizations
        self.F = np.ascontiguousarray(self.F.reshape(self.n_obj, n_r, -1))
