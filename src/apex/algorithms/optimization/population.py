"""What the population optimizers share: members, ranking, convergence and state.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/operators/sampling/rnd.py (FloatRandomSampling), pymoo/core/individual.py
             (the constraint violation ``CV``), pymoo/algorithms/soo/nonconvex/ga.py
             (``FitnessSurvival``: feasible first, then the objective)
Reference  : Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
             Computer Methods in Applied Mechanics and Engineering, 186, 311-338.
             Blank, J. & Deb, K. (2020). pymoo: Multi-objective optimization in Python. IEEE
             Access, 8, 89497-89509.
Differences: the population is ``(n_x, P)`` (the ensemble axis last, APEX's convention;
             pymoo's is ``(P, n_x)``); a robust objective's value of a member is its mean over
             the model realizations; the external penalty (``constraint_handling = "epf"``)
             is APEX's, a second constraint method pymoo lacks. The start includes ``x0``.

A generation evaluates its new members in one batch, over every realization: ``(n_x, n_new
n_r)`` columns. Each member keeps its raw objective per realization and its constraints, so
an EPF update re-ranks without evaluating anything. Every generation is an accepted step
(``OptimizerBase`` ends a run on a step that is not, and checkpoints only accepted ones); an
improvement of the best only resets the stall counter.

The ranking (Deb 2000, as pymoo's): with ``"feasibility"``, a smaller constraint violation
first, then a smaller objective; the violation is pymoo's sum
``sum max(-c_iq, 0) + sum max(|c_eq| - eq_tolerance, 0)``. With ``"epf"`` the objective plus
the penalty alone.
"""

import logging
from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Literal, cast

import numpy as np

from apex.algorithms.optimization.base import (
    EpfOptions,
    OptimizerBase,
    StepReport,
    external_penalty,
)
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    Draws,
    FloatArray,
    GeneratorDraws,
    OptimizerState,
    Problem,
    StateValue,
)

logger = logging.getLogger(__name__)

type ConstraintHandling = Literal["feasibility", "epf"]
type Initialization = Literal["uniform", "lhs"]


def violation(c_eq: FloatArray, c_iq: FloatArray, eq_tolerance: float) -> FloatArray:
    """Return pymoo's constraint violation of each member ``(n,)``.

    Parameters
    ----------
    c_eq, c_iq
        Equality ``(n_eq, n)`` (zero when met) and inequality ``(n_iq, n)`` (non-negative
        when met) constraints; either may have no rows.
    eq_tolerance
        Equalities within it count as met.
    """
    eq = np.sum(np.maximum(np.abs(c_eq) - eq_tolerance, 0.0), axis=0)
    iq = np.sum(np.maximum(-c_iq, 0.0), axis=0)
    return np.asarray(eq + iq, dtype=float)


def rank_order(values: FloatArray, cv: FloatArray) -> FloatArray:
    """Return the members' indices from best to worst: smaller ``cv``, then smaller value.

    ``np.lexsort`` is stable, as pymoo's ``FitnessSurvival`` relies on (ties keep their order).
    """
    return np.asarray(np.lexsort([values, cv]), dtype=np.int64)


def uniform_population(
    rng: np.random.Generator, lower: FloatArray, upper: FloatArray, n: int
) -> FloatArray:
    """Return ``n`` members drawn uniformly in the box ``(n_x, n)`` (pymoo's sampling)."""
    draws = rng.random((n, lower.size))  # pymoo's layout and order of draws
    return np.ascontiguousarray((lower + draws * (upper - lower)).T)


def latin_hypercube(
    rng: np.random.Generator, lower: FloatArray, upper: FloatArray, n: int
) -> FloatArray:
    """Return ``n`` members of a random Latin hypercube in the box ``(n_x, n)``.

    Each control's range is cut into ``n`` equal strata, one member in each, at a uniform
    position within it; the strata are paired across controls by independent permutations.
    """
    n_x = lower.size
    cells = np.stack([rng.permutation(n) for _ in range(n_x)])  # (n_x, n)
    unit = (cells + rng.random((n_x, n))) / n
    return np.ascontiguousarray(lower[:, None] + unit * (upper - lower)[:, None])


class PopulationOptimizer(OptimizerBase):
    """The loop of a population optimizer: generations of a population, evaluated together.

    Parameters
    ----------
    pop_size
        Members (P).
    stall
        Generations without a relative improvement above ``ftol`` that count as converged.
    init
        The initial population besides ``x0``: ``"uniform"`` or ``"lhs"``.
    constraint_handling
        ``"feasibility"`` (Deb's rules) or ``"epf"`` (the external penalty, with ``epf``).
    eq_tolerance
        Equality constraints within it count as met (feasibility rules).
    maxiter
        Generations at most (per outer EPF iteration).
    ftol
        Relative improvement of the best objective a generation needs; ``stall`` generations
        without it count as converged.
    xtol
        The population's spread (the largest standard deviation of a control, optimizer
        coordinates) that counts as converged.
    transform
        Optimize in the unit cube of the bounds.
    epf
        External penalty options: ``r`` (required), ``r_factor``, ``tol_factor``,
        ``conv_crit``, ``max_epf_iter``. Only with ``constraint_handling = "epf"``.
    """

    label: ClassVar[str] = "population"
    #: The members a generation evaluates (``pop_size`` unless a subclass says otherwise).
    n_new: int

    def __init__(
        self,
        *,
        pop_size: int = 20,
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
        super().__init__(maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf)
        if pop_size < 2:
            raise ValueError(f"pop_size must be at least 2, got {pop_size}")
        if stall < 1:
            raise ValueError(f"stall must be positive, got {stall}")
        if init not in ("uniform", "lhs"):
            raise ValueError(f"init must be 'uniform' or 'lhs', got {init!r}")
        if constraint_handling not in ("feasibility", "epf"):
            raise ValueError(
                f"constraint_handling must be 'feasibility' or 'epf', got {constraint_handling!r}"
            )
        # One constraint method in force: EPF options mean the penalty, and only with it.
        if constraint_handling == "epf" and not self.epf:
            raise ValueError('constraint_handling = "epf" needs the epf options (at least r)')
        if constraint_handling == "feasibility" and self.epf:
            raise ValueError('epf options need constraint_handling = "epf"')
        if eq_tolerance < 0:
            raise ValueError(f"eq_tolerance must be non-negative, got {eq_tolerance}")
        self.pop_size = pop_size
        self.n_new = pop_size
        self.stall = stall
        self.init: Initialization = init
        self.constraint_handling: ConstraintHandling = constraint_handling
        self.eq_tolerance = eq_tolerance

    # ------------------------------------------------------------------ set-up

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        if not isinstance(draws, GeneratorDraws):
            # The operators draw as pymoo's do, from a seeded numpy Generator (its state is
            # checkpointed with the draws).
            raise ValueError(f"{self.label} needs a seeded random generator (GeneratorDraws)")
        if problem.lower is None or problem.upper is None:
            raise ValueError(f"{self.label} needs finite lower and upper bounds on every control")
        if not (np.all(np.isfinite(problem.lower)) and np.all(np.isfinite(problem.upper))):
            raise ValueError(f"{self.label} needs finite lower and upper bounds on every control")
        self.rng = draws.rng
        n_x = self.xk.size
        if self.transform:
            self.xl, self.xu = np.zeros(n_x), np.ones(n_x)
        else:
            self.xl = np.asarray(problem.lower, dtype=float)
            self.xu = np.asarray(problem.upper, dtype=float)
        self.constrained = isinstance(problem.objective, ConstrainedObjective)
        n_r = problem.n_realizations
        # The population, ensemble axis last; per member its raw objective per realization
        # and its constraints (an EPF update re-ranks from these).
        self.X = np.empty((n_x, 0))
        self.F = np.empty((n_r, 0))
        self.c_eq = np.empty((0, 0))
        self.c_iq = np.empty((0, 0))
        self.stalled = 0
        self.best_rank: tuple[float, float] | None = None

    # ------------------------------------------------------------------ evaluation

    def evaluate_population(self, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate members ``(n_x, n)`` (optimizer coordinates) on every realization.

        Returns the raw objective ``(n_r, n)`` and the constraints ``(n_eq, n)`` and
        ``(n_iq, n)`` (no rows without constraints). One batch: ``n n_r`` columns.
        """
        n = X.shape[1]
        n_r = self.problem.n_realizations
        physical = self.bounds.to_physical(self.bounds.project(X))
        # Member-major columns: member j's realizations are columns j n_r ... j n_r + n_r - 1.
        controls = np.ascontiguousarray(np.repeat(physical, n_r, axis=1))
        realizations = np.tile(np.arange(n_r, dtype=np.int64), n)
        values = np.asarray(self.problem.objective(controls, realizations), dtype=float)
        if values.shape != (n * n_r,):
            raise ValueError(f"the objective returned shape {values.shape} for {n * n_r} points")
        if not np.all(np.isfinite(values)):
            # A NaN would tie in tournaments and could win one: fail loudly.
            bad = int(np.sum(~np.isfinite(values)))
            raise ValueError(
                f"the objective returned {bad} non-finite value(s) for {self.label}'s members; "
                "check the objective (or its simulations) at those controls"
            )
        self.member_evaluations += n * n_r
        self.nfev += n
        F = np.ascontiguousarray(values.reshape(n, n_r).T)
        if self.constrained:
            objective = cast(ConstrainedObjective, self.problem.objective)
            c_eq, c_iq = (np.asarray(c, dtype=float).reshape(-1, n)
                          for c in objective.constraints(physical))  # fmt: skip
        else:
            c_eq, c_iq = np.empty((0, n)), np.empty((0, n))
        return F, np.ascontiguousarray(c_eq), np.ascontiguousarray(c_iq)

    def scores(
        self, F: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return each member's ranked value, violation and penalty ``(n,)`` each.

        The value is the mean over realizations (plus the penalty with EPF); the violation is
        zero with EPF (the penalty carries it), pymoo's sum otherwise.
        """
        mean = np.mean(F, axis=0)
        n = F.shape[1]
        if self.constraint_handling == "epf":
            penalty = external_penalty(self.epf["r"], c_eq, c_iq) if self.constrained \
                else np.zeros(n)  # fmt: skip
            return mean + penalty, np.zeros(n), penalty
        cv = violation(c_eq, c_iq, self.eq_tolerance) if self.constrained else np.zeros(n)
        return mean, cv, np.zeros(n)

    def _set_population(
        self, X: FloatArray, F: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> None:
        self.X, self.F = np.ascontiguousarray(X), np.ascontiguousarray(F)
        self.c_eq, self.c_iq = np.ascontiguousarray(c_eq), np.ascontiguousarray(c_iq)

    # ------------------------------------------------------------------ the loop

    def _start(self) -> None:
        self._started = True
        logger.info("optimizer_start", extra={"optimizer": self.label, "n_controls": self.xk.size})
        x0 = self.bounds.project(self.xk)
        sample = (latin_hypercube if self.init == "lhs" else uniform_population)(
            self.rng, self.xl, self.xu, self.pop_size - 1
        )
        X = np.ascontiguousarray(np.column_stack([x0, sample]))
        self._set_population(X, *self.evaluate_population(X))
        self.initialize()
        self._take_best(first=True)
        self._log_iteration()

    def initialize(self) -> None:
        """Set up the optimizer's own state once the first population is evaluated."""

    @abstractmethod
    def generation(self) -> None:
        """Make, evaluate and keep one generation (updating the population)."""

    def update_step(self) -> StepReport:
        """One generation: always an accepted step (see the module docstring)."""
        self.generation()
        self._take_best(first=False)
        return StepReport(True)

    def best_index(self) -> int:
        """Return the index of the best member under the ranking."""
        value, cv, _ = self.scores(self.F, self.c_eq, self.c_iq)
        return int(rank_order(value, cv)[0])

    def _take_best(self, *, first: bool) -> None:
        """Commit the best member as ``x_k``; count a stalled generation."""
        value, cv, penalty = self.scores(self.F, self.c_eq, self.c_iq)
        best = int(rank_order(value, cv)[0])
        rank = (float(cv[best]), float(value[best]))
        if not first and self.best_rank is not None:
            old_cv, old_value = self.best_rank
            improved = rank[0] < old_cv or (
                rank[0] == old_cv and rank[1] < old_value - self.ftol * abs(old_value)
            )
            self.stalled = 0 if improved else self.stalled + 1
        self.best_rank = rank
        n_r = self.problem.n_realizations
        fk = np.ascontiguousarray(self.F[:, best] + penalty[best])
        self.last_penalty = np.full(n_r, penalty[best]) if self.epf else None
        x = np.ascontiguousarray(self.X[:, best].copy())
        if first:
            self.xk, self.fk = x, fk
        else:
            self._commit_step(x, fk)

    def _refresh_epf_function_value(self) -> None:
        # A new penalty factor re-ranks the stored members: nothing is evaluated.
        if self.epf_iteration <= 1 or self.iteration != 0:
            return
        self.best_rank = None
        self._take_best(first=True)
        self.stalled = 0

    # ------------------------------------------------------------------ convergence

    def check_function_convergence(self) -> bool:
        """Converged when the best has not improved by more than ``ftol`` for ``stall``."""
        if self.stalled >= self.stall:
            self.conv_msg = (
                f"No relative improvement above {self.ftol} for {self.stall} generations"
            )
            return True
        return False

    def check_state_convergence(self) -> bool:
        """Converged when the population's spread (largest control std) is below ``xtol``."""
        spread = self.spread()
        if spread < self.xtol:
            self.conv_msg = f"Population spread {spread:.3g} < {self.xtol}"
            return True
        return False

    def check_convergence(self) -> bool:
        """No gradient test: the stall and spread tests decide."""
        return False

    def spread(self) -> float:
        """Return the largest standard deviation of a control over the population."""
        return float(np.max(np.std(self.positions(), axis=1))) if self.X.size else 0.0

    def positions(self) -> FloatArray:
        """Return the members whose spread measures convergence (the population)."""
        return self.X

    # ------------------------------------------------------------------ bounds

    def max_evaluations(self, n_realizations: int) -> int | None:
        """Return the start ``P n_r``, then ``epf_maxiter maxiter n_new n_r``.

        An EPF update evaluates nothing (the members' raw values are kept).
        """
        start = self.pop_size * n_realizations
        return start + self.epf_maxiter * self.maxiter * self.n_new * n_realizations

    def evaluations_per_iteration(self, n_realizations: int) -> int | None:
        """One generation's new members on every realization."""
        return self.n_new * n_realizations

    # ------------------------------------------------------------------ logs and state

    def log_columns(self) -> dict[str, float]:
        """Return the best's objective (mean), the population's mean, spread, feasible share."""
        columns = super().log_columns()
        if self.F.size:
            value, _, _ = self.scores(self.F, self.c_eq, self.c_iq)
            columns["population_mean"] = float(np.mean(value))
            columns["spread"] = self.spread()
            if self.constrained:
                # The constraints themselves, also under EPF (whose ranking has no violation).
                feasible = violation(self.c_eq, self.c_iq, self.eq_tolerance) <= 0.0
                columns["feasible_share"] = float(np.mean(feasible))
        return columns

    def result_extra(self) -> dict[str, float]:
        """Return the best's constraint violation and whether it is feasible."""
        if not self.F.size:
            return {}
        _, cv, _ = self.scores(self.F, self.c_eq, self.c_iq)
        best = self.best_index()
        extra = {"spread": self.spread()}
        if self.constrained:
            extra["violation"] = float(cv[best]) if self.constraint_handling == "feasibility" \
                else float(violation(self.c_eq, self.c_iq, self.eq_tolerance)[best])  # fmt: skip
            extra["feasible"] = float(extra["violation"] <= 0.0)
        return extra

    def _export_extra(self) -> OptimizerState:
        state: OptimizerState = {
            "X": self.X,
            "F": self.F,
            "c_eq": self.c_eq,
            "c_iq": self.c_iq,
            "stalled": self.stalled,
            "best_cv": None if self.best_rank is None else self.best_rank[0],
            "best_value": None if self.best_rank is None else self.best_rank[1],
        }
        state.update(self._export_population_extra())
        return state

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        def array(key: str) -> FloatArray:
            return np.ascontiguousarray(np.array(state[key], dtype=float))

        n_new = self._n_constraint_rows(state)
        self.X, self.F = array("X"), array("F")
        n = self.X.shape[1]
        self.c_eq = array("c_eq").reshape(n_new[0], n)
        self.c_iq = array("c_iq").reshape(n_new[1], n)
        self.stalled = int(cast(int, state["stalled"]))
        cv, value = state.get("best_cv"), state.get("best_value")
        self.best_rank = None if cv is None or value is None else (
            float(cast(float, cv)), float(cast(float, value)))  # fmt: skip
        self._restore_population_extra(state)

    @staticmethod
    def _n_constraint_rows(state: Mapping[str, StateValue]) -> tuple[int, int]:
        rows = []
        for key in ("c_eq", "c_iq"):
            value = np.array(state[key], dtype=float)
            rows.append(value.shape[0] if value.ndim == 2 else 0)
        return rows[0], rows[1]

    def _export_population_extra(self) -> OptimizerState:
        """Return the optimizer's own state beyond the population."""
        return {}

    def _restore_population_extra(self, state: Mapping[str, StateValue]) -> None:
        """Restore what :meth:`_export_population_extra` returned."""


#: Options a drafted configuration gives a population optimizer when none are given: few
#: generations in the unit cube. A draft is robust over every posterior member, and each
#: generation is ``pop_size`` simulations per member: 20 x 11 x 50 members = 11,000 runs, within
#: the approval threshold (20,000); the user raises them knowingly.
DRAFT_OPTIONS: dict[str, object] = {"pop_size": 20, "maxiter": 10, "transform": True}


def is_population_optimizer(name: str) -> bool:
    """Return whether a registered optimizer is a population optimizer (``ga``, ``pso``)."""
    from apex.algorithms.optimization.registries import optimizers

    try:
        factory = optimizers.factory(name)
    except KeyError:
        return False
    return isinstance(factory, type) and issubclass(factory, PopulationOptimizer)
