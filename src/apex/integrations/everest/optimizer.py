"""ropt's optimizers (SciPy, Dakota) as an APEX optimizer.

The reverse: there APEX's optimizers serve ropt (Everest); here ropt's backends
serve APEX. ``[optimizer] name = "ropt"`` with ``method = "scipy/slsqp"`` (or
``"scipy/l-bfgs-b"``, ``"dakota/optpp_q_newton"`` with ``ropt-dakota``, ...) runs ropt's
``BasicOptimizer`` on APEX's objective, analytic or simulated (APEX's simulators, executors
and evaluation journal).

Adapted from ropt
-----------------
Library   : ropt 0.28 (github.com/TNO-ropt/ropt; GPL-3.0), ``ropt.workflow.BasicOptimizer``
            and its batch evaluator protocol; called, not copied. ropt is installed with
            ERT 26 (the ``ert`` extra) and imported only when this optimizer is built or
            run: the entry point ``apex.optimizers`` is installed with APEX, and loading it
            must not need ropt.

How it maps:

- **Iterations.** ropt drives its own loop: its backend's iterations, line searches and
  ensemble gradients (``number_of_perturbations`` perturbed points per realization,
  ``perturbation_magnitude`` in the optimizer's coordinates). An APEX "accepted step" is a
  point ropt evaluated that ranks better than the current iterate (below): each one is
  committed, logged in ``history.jsonl`` and checkpointed, as the other optimizers' steps.
  ``maxiter`` is ropt's ``max_iterations`` for the run (a resume replays it).
- **Ranking and feasibility.** Without constraints a point ranks by its mean objective.
  With ``constraints = true`` it ranks by its violation first, ``v = max(|c_eq|,
  max(-c_iq, 0))``: a point with ``v <= constraint_tolerance`` (feasible) ranks above any
  infeasible one, feasible points by their mean objective, infeasible ones by ``v``. A run
  may start infeasible: its steps then reduce the violation until a feasible point is
  found, and objective improvements follow.
- **Coordinates.** ropt works in APEX's optimizer coordinates: the unit cube of the bounds
  with ``transform = true``, physical controls otherwise. Points are projected onto the
  bounds before they are evaluated.
- **Realizations.** Robust optimization weighs the realizations equally, as APEX's mean;
  each row ropt asks for is one realization's evaluation.
- **Constraints.** With ``constraints = true``, a constrained objective's ``c_eq = 0`` and
  ``c_iq >= 0`` become ropt's nonlinear constraints (methods that take them:
  ``scipy/slsqp``, ``scipy/cobyla``, ...); otherwise they are ignored, as APEX's other
  optimizers ignore them without ``epf``. The external penalty (``epf``) is refused.
- **Limits and approval.** ``max_functions`` and ``max_batches`` are ropt's. The evaluation
  bound APEX's approval needs is only known with ``max_batches``: the start, then at most
  ``max_batches`` batches of ``n_realizations (number_of_perturbations + 1)`` evaluations.
  Without it the bound is unknown (a simulated objective's plan says so). Measured on
  2026-10-05 (ropt 0.28.1, ropt-dakota 0.28.0) for SciPy's slsqp, l-bfgs-b, cobyla,
  nelder-mead, powell, trust-constr and differential_evolution and Dakota's optpp_q_newton,
  conmin_frcg, conmin_mfd, asynch_pattern_search, coliny_ea and mesh_adaptive_search: no
  batch asks for more (derivative-free and population methods: one point, so
  ``n_realizations``); ``test_every_batch_fits_the_evaluation_bound`` keeps it so.
  Dakota's genetic algorithms (``soga``, ``moga``) abort the whole process in Dakota, so
  they are refused (``REFUSED_METHODS``).
- **Stop.** A stop request (the stop file) is asked before every batch, except while a
  resume replays up to its checkpoint (``OptimizerBase.stop_check``, set by the
  workflow); ``--max-iterations`` is counted at accepted steps. Either aborts ropt
  (``USER_ABORT``); the last checkpoint stands.
- **Resume.** ropt cannot continue from a checkpoint, so a resume replays its run from the
  start with the checkpointed seed: ropt is deterministic and a simulated objective's
  evaluations come from the journal, so nothing already journaled is simulated again (an
  interruption re-spends at most the batch it was in, plus failed members). At the
  checkpointed step the point must be bit-identical and the counters equal to the
  checkpoint's (else a ``WorkflowError``); the run then continues as the uninterrupted one
  would.
- **Randomness.** ropt's perturbation seed is drawn from APEX's stream (``[optimization]
  seed``) when the run starts, unless ``seed`` is given.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.optimization import GeneratorDraws, OptimizerBase, StepReport
from apex.algorithms.optimization.base import EpfOptions, IterationCallback, as_float
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    Draws,
    OptimizerState,
    Problem,
    StateValue,
)
from apex.core.errors import WorkflowError
from apex.core.options import TEXT, mapping

if TYPE_CHECKING:
    from ropt.evaluation import EvaluationBatchContext, EvaluationBatchResult

type FloatArray = NDArray[np.float64]

#: How ropt's exit codes read as the run's message (by name: ropt is imported lazily).
EXIT_MESSAGES = {
    "OPTIMIZER_FINISHED": "ropt: the optimizer finished",
    "MAX_FUNCTIONS_REACHED": "ropt: max_functions reached",
    "MAX_BATCHES_REACHED": "ropt: max_batches reached",
    "TOO_FEW_REALIZATIONS": "ropt: too few realizations succeeded",
}


def _ropt() -> None:
    """Fail clearly when ropt is missing."""
    import importlib.util

    if importlib.util.find_spec("ropt") is None:
        raise ValueError("the ropt optimizer needs ropt: install APEX's 'ert' extra")


class RoptOptimizer(OptimizerBase):
    """ropt's optimizer backends (SciPy, Dakota) run on APEX's objectives.

    Parameters
    ----------
    method
        ropt's backend method, e.g. ``scipy/slsqp``, ``scipy/l-bfgs-b``, ``scipy/cobyla``,
        ``dakota/optpp_q_newton`` (with ``ropt-dakota``).
    maxiter
        ropt's ``max_iterations`` (the backend's iterations) of the run (a resume replays it).
    max_functions, max_batches
        ropt's limits on function evaluations and evaluation batches; ``max_batches``
        bounds the evaluations (approval).
    number_of_perturbations
        Perturbed points per realization of ropt's ensemble gradient.
    perturbation_magnitude
        Their standard deviation in the optimizer's coordinates (one, or one per control);
        default ropt's.
    convergence_tolerance
        The backend's tolerance (ropt's ``convergence_tolerance``).
    seed
        ropt's perturbation seed (default: drawn from the optimization's seed).
    options
        Passed to the backend as its ``options`` (e.g. SciPy's ``ftol``).
    constraints
        Hand a constrained objective's constraints to ropt (default: ignore them, as APEX's
        optimizers without ``epf``).
    constraint_tolerance
        The violation below which a point counts as feasible.
    transform
        Optimize in the unit cube of the bounds.
    epf
        Refused: ropt handles constraints itself.
    """

    status = "experimental"
    #: ropt comes with ERT 26 (the ``ert`` extra; checked by ``apex plugins``).
    requires: ClassVar[Mapping[str, str]] = MappingProxyType({"ropt": "ert"})
    label = "ropt"

    def __init__(
        self,
        method: Annotated[str, TEXT] = "scipy/slsqp",
        maxiter: int = 100,
        max_functions: int | None = None,
        max_batches: int | None = None,
        number_of_perturbations: int = 5,
        perturbation_magnitude: float | Sequence[float] | None = None,
        convergence_tolerance: float | None = None,
        seed: int | None = None,
        options: Annotated[Mapping[str, Any] | None, mapping()] = None,
        constraints: bool = False,
        constraint_tolerance: float = 1e-8,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        if epf:
            raise ValueError(
                "the ropt optimizer takes no epf: ropt handles constraints itself "
                "(constraints = true)"
            )
        super().__init__(maxiter=maxiter, transform=transform)
        backend, _, name = method.partition("/")
        if not backend or not name:
            raise ValueError(
                f"method must be '<backend>/<method>', e.g. scipy/slsqp; got {method!r}"
            )
        _ropt()
        _check_method(method)
        for key, value in (("max_functions", max_functions), ("max_batches", max_batches)):
            if value is not None and value < 1:
                raise ValueError(f"{key} must be positive, got {value}")
        if number_of_perturbations < 1:
            raise ValueError("number_of_perturbations must be at least 1")
        if constraint_tolerance < 0:
            raise ValueError("constraint_tolerance must be non-negative")
        self.method = method
        self.max_functions = max_functions
        self.max_batches = max_batches
        self.number_of_perturbations = number_of_perturbations
        self.perturbation_magnitude = perturbation_magnitude
        self.convergence_tolerance = convergence_tolerance
        self.seed = seed
        self.backend_options = dict(options or {})
        self.use_constraints = constraints
        self.constraint_tolerance = constraint_tolerance

    # ------------------------------------------------------------------ bounds

    def max_evaluations(self, n_realizations: int) -> int | None:
        """Return the start plus ``max_batches`` full batches, or ``None`` without it."""
        if self.max_batches is None:
            return None
        per_batch = n_realizations * (self.number_of_perturbations + 1)
        return n_realizations + self.max_batches * per_batch

    def max_evaluations_basis(self, n_realizations: int) -> str | None:
        """Return :meth:`max_evaluations` as its terms with their values (``None`` without)."""
        total = self.max_evaluations(n_realizations)
        if total is None:
            return None
        n = n_realizations
        realizations = f"{n} realization{'' if n == 1 else 's'}"
        return (f"{n} for the start + {self.max_batches} batches x {realizations} x "
                f"({self.number_of_perturbations} perturbations + 1) = {total}")  # fmt: skip

    def update_step(self) -> StepReport:  # pragma: no cover - ropt runs the loop
        """Not used: ropt drives the iterations (:meth:`_run`)."""
        raise NotImplementedError("ropt drives its own iterations")

    # ------------------------------------------------------------------ ranking

    def _violation(self, x: FloatArray) -> float:
        """Return the constraint violation ``v`` at ``x`` (optimizer coordinates; 0 without)."""
        if not self._n_eq + self._n_iq:
            return 0.0
        objective = cast(ConstrainedObjective, self.problem.objective)
        physical = self.bounds.to_physical(self.bounds.project(x))[:, None]
        c_eq, c_iq = (np.asarray(c, dtype=float).ravel() for c in objective.constraints(physical))
        parts = [np.abs(c_eq), np.maximum(-c_iq, 0.0)]
        return float(max((float(np.max(p)) for p in parts if p.size), default=0.0))

    def _feasibility(self) -> dict[str, float]:
        """With constraints: the iterate's ``violation`` and ``feasible`` (1 or 0)."""
        if not self._n_eq + self._n_iq:
            return {}
        violation = self._violation(self.xk)
        return {"violation": violation, "feasible": float(violation <= self.constraint_tolerance)}

    def log_columns(self) -> dict[str, float]:
        """One row of the history, with the violation and feasibility under constraints.

        An infeasible run's steps reduce the violation, not the objective, so its history
        need not be monotone in ``objective``: the ``violation`` column shows why.
        """
        return {**super().log_columns(), **self._feasibility()}

    def result_extra(self) -> dict[str, float]:
        """Return the result's violation and feasibility under constraints.

        A study, the agents and the web UI then see whether its objective belongs to a
        feasible point.
        """
        return {**super().result_extra(), **self._feasibility()}

    def _rank(self, values: FloatArray, violation: float) -> tuple[int, float]:
        """Feasible points first, by mean objective; infeasible ones by their violation."""
        if violation <= self.constraint_tolerance:
            return 0, float(np.mean(values))
        return 1, violation

    # ------------------------------------------------------------------ the run

    def _run(self, on_iteration: IterationCallback | None) -> bool:
        _ropt()
        from ropt.enums import ExitCode
        from ropt.results import FunctionResults
        from ropt.workflow import BasicOptimizer

        replay: tuple[int, FloatArray, tuple[int, int, int]] | None = None
        if not self._started:
            self._start()
            if self.seed is None:
                self.seed = self._draw_seed()
            if on_iteration is not None and on_iteration(self) is False:
                return True
        elif self.iteration > 0:
            # A resume: ropt cannot continue from a checkpoint, so its run is replayed from
            # the start with the same seed. It is deterministic, and a simulated objective's
            # evaluations come from the journal, so nothing is simulated twice; at the
            # checkpointed step the point and the counters must be exactly the checkpoint's.
            replay = (self.iteration, self.xk.copy(), self._counters())
            self.iteration, self.nfev, self.member_evaluations, self._batches = 0, 0, 0, 0
            self.xk = self.bounds.to_internal(self.problem.x0.copy())
            self.fk = self._objective_value(self.xk)
        self.epf_iteration = 1
        config = self._config(self.maxiter, self.max_batches)
        current = self._rank(cast(FloatArray, self.fk), self._violation(self.xk))
        abort: dict[str, Any] = {"stop": False, "diverged": None}

        def replaying() -> bool:
            return replay is not None and self.iteration < replay[0]

        def should_abort() -> bool:
            # A stop request is honoured before every batch, not only at accepted steps.
            stop = self.stop_check
            return bool(abort["stop"] or (not replaying() and stop is not None and stop()))

        optimizer = BasicOptimizer(config, self._evaluate)
        optimizer.set_abort_callback(should_abort)

        def improved(results: tuple[Any, ...]) -> None:
            nonlocal current
            for item in results:
                if not isinstance(item, FunctionResults) or abort["stop"]:
                    continue
                values = np.asarray(item.evaluations.objectives, dtype=float)[:, 0]
                if not np.all(np.isfinite(values)):
                    continue
                x = self.bounds.project(np.asarray(item.evaluations.variables, dtype=float))
                rank = self._rank(values, self._violation(x))
                if rank < current:
                    current = rank
                    self._commit_step(x, values)
                    self.iteration += 1
                    if replay is not None and self.iteration <= replay[0]:
                        if self.iteration == replay[0]:
                            same = np.array_equal(self.xk, replay[1])
                            if not same or self._counters() != replay[2]:
                                abort["diverged"] = (
                                    f"the replay of ropt's run differs from its checkpoint "
                                    f"at step {replay[0]} (counters {self._counters()} against "
                                    f"{replay[2]}): the objective, its files or ropt changed"
                                )
                                abort["stop"] = True
                        continue  # recorded already
                    self._log_iteration()
                    if on_iteration is not None and on_iteration(self) is False:
                        abort["stop"] = True

        optimizer.set_results_callback(improved)
        code = optimizer.run(self.xk)
        if abort["diverged"] is not None or replaying():
            raise WorkflowError(
                abort["diverged"]
                or f"the replay of ropt's run ended at step {self.iteration}, before its "
                f"checkpoint's {replay[0] if replay else 0}: the objective, its files or ropt "
                "changed"
            )
        if code == ExitCode.USER_ABORT or abort["stop"]:
            return True
        if code == ExitCode.ABORT_FROM_ERROR:
            raise RuntimeError("ropt aborted on an error in an evaluation")
        self.conv_msg = EXIT_MESSAGES.get(code.name, f"ropt: {code.name}")
        if current[0] == 1:
            self.conv_msg += f" (no feasible point found: violation {current[1]:.3g})"
        self._inner_done = True
        self._log_convergence()
        return False

    def _counters(self) -> tuple[int, int, int]:
        """Return the counters a replay must reproduce: evaluations, members, batches."""
        return self.nfev, self.member_evaluations, self._batches

    def _draw_seed(self) -> int:
        if not isinstance(self.draws, GeneratorDraws):
            raise ValueError(
                "the ropt optimizer draws its perturbation seed from the optimization's "
                "random stream: give seed, or run it through apex optimize"
            )
        return int(self.draws.rng.integers(1, 2**31 - 1))

    def _config(self, max_iterations: int, max_batches: int | None) -> dict[str, Any]:
        n_x = self.xk.size
        if not self.bounds.bounded:
            lower, upper = np.full(n_x, -np.inf), np.full(n_x, np.inf)
        elif self.bounds.transform:
            lower, upper = np.zeros(n_x), np.ones(n_x)
        else:
            lower, upper = self.bounds.lower, self.bounds.upper
        variables: dict[str, Any] = {
            "variable_count": n_x,
            "lower_bounds": lower,
            "upper_bounds": upper,
            "seed": (int(cast(int, self.seed)),),
        }
        if self.perturbation_magnitude is not None:
            variables["perturbation_magnitudes"] = np.broadcast_to(
                np.asarray(self.perturbation_magnitude, dtype=float), (n_x,)
            ).copy()
        backend: dict[str, Any] = {"method": self.method, "max_iterations": max_iterations}
        if self.convergence_tolerance is not None:
            backend["convergence_tolerance"] = self.convergence_tolerance
        if self.backend_options:
            backend["options"] = self.backend_options
        n_real = self.problem.n_realizations
        config: dict[str, Any] = {
            "variables": variables,
            "backend": backend,
            "gradient": {"number_of_perturbations": self.number_of_perturbations},
            "realizations": {"weights": np.full(n_real, 1.0 / n_real)},
        }
        limits = {"max_functions": self.max_functions, "max_batches": max_batches}
        config["optimizer"] = {k: v for k, v in limits.items() if v is not None}
        if self._n_eq + self._n_iq:
            config["nonlinear_constraints"] = {
                "lower_bounds": np.zeros(self._n_eq + self._n_iq),
                "upper_bounds": np.concatenate([np.zeros(self._n_eq), np.full(self._n_iq, np.inf)]),
            }
        return config

    def _constraint_counts(self) -> tuple[int, int]:
        objective = self.problem.objective
        if not self.use_constraints:
            return 0, 0
        if not isinstance(objective, ConstrainedObjective):
            raise ValueError("constraints = true needs an objective with constraints")
        physical = self.bounds.to_physical(self.bounds.project(self.xk))[:, None]
        # One column in: each constraint is one row (an empty c_eq has none).
        c_eq, c_iq = (
            np.asarray(c, dtype=float).reshape(-1, 1) for c in objective.constraints(physical)
        )
        return int(c_eq.shape[0]), int(c_iq.shape[0])

    def _evaluate(
        self, variables: FloatArray, context: "EvaluationBatchContext"
    ) -> "EvaluationBatchResult":
        """Evaluate the rows ropt asks for: one realization each (inactive rows stay 0)."""
        from ropt.evaluation import EvaluationBatchResult

        self._batches += 1
        rows = np.flatnonzero(context.active)
        points = self.bounds.project(np.asarray(variables, dtype=float)[rows].T)  # (n_x, n)
        controls = self.bounds.to_physical(points)
        realizations = np.asarray(context.realizations, dtype=np.int64)[rows]
        objectives = np.zeros((variables.shape[0], 1))
        objectives[rows, 0] = self.evaluate_members(controls, realizations)
        constraints = None
        if self._n_eq + self._n_iq:
            objective = cast(ConstrainedObjective, self.problem.objective)
            c_eq, c_iq = objective.constraints(controls)
            n = controls.shape[1]
            values = np.vstack([np.asarray(c_eq, dtype=float).reshape(self._n_eq, n),
                                np.asarray(c_iq, dtype=float).reshape(self._n_iq, n)])  # fmt: skip
            constraints = np.zeros((variables.shape[0], self._n_eq + self._n_iq))
            constraints[rows] = values.T
        return EvaluationBatchResult(objectives=objectives, constraints=constraints)

    # ------------------------------------------------------------------ state

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        self._batches = 0
        self._n_eq, self._n_iq = self._constraint_counts()

    def _export_extra(self) -> OptimizerState:
        return {"batches": self._batches, "seed": self.seed}

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self._batches = int(as_float(state.get("batches", 0)))
        if state.get("seed") is not None:
            self.seed = int(as_float(state["seed"]))


#: Methods refused although a backend offers them: Dakota's genetic algorithms (JEGA) abort
#: the whole process inside Dakota (measured 2026-10-05, ropt-dakota 0.28.0), which would kill
#: an optimization's worker; APEX's own population methods replace them.
REFUSED_METHODS = {
    "dakota/soga": "Dakota's single-objective GA aborts the process inside Dakota",
    "dakota/moga": "Dakota's multi-objective GA aborts the process inside Dakota",
}


def _check_method(method: str) -> None:
    """Refuse a method no installed ropt backend offers, or one known to crash."""
    from ropt.plugins.manager import get_plugin

    if method.lower() in REFUSED_METHODS:
        raise ValueError(
            f"{method} is refused: {REFUSED_METHODS[method.lower()]}; for a global search use "
            "scipy/differential_evolution or dakota/coliny_ea, or APEX's "
            '[optimizer] name = "genopt"'
        )

    try:
        get_plugin("backend", method)
    except (ValueError, KeyError) as error:
        raise ValueError(f"no ropt backend offers {method!r}: {error}") from error


__all__ = ["RoptOptimizer"]
