"""The iteration every optimizer shares: convergence checks, the EPF loop and checkpoints.

Ported from PET
---------------
PET module : src/popt/optimization_methods/optimizer_base.py (OptimizerBase), and
             src/popt/cost_functions/epf.py (the external penalty)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.), section
             17.1 (the quadratic penalty method). Springer.
Differences: the objective's constraints come through ``ConstrainedObjective`` and the loop
             adds the penalty itself, recording it for the EPF convergence test; PET's
             objectives write ``epf['penalty']`` themselves. Restart is through
             :meth:`OptimizerBase.export_state` and ``restore_state`` (the workflow's
             checkpoints) instead of PET's pickles. Logging is structured.

The loop (PET's ``run_optimization``)::

    start: evaluate f(x0) (and the derivatives the optimizer needs)
    for each EPF outer iteration (one without EPF):
        re-evaluate f(x_k) with the new penalty factor (from the second outer iteration)
        while iteration < maxiter:
            iteration += 1
            step: an accepted step commits x_k+1; a failed one ends the run
            converged if |f_k+1 - f_k| < ftol |f_k|, or ||x_k+1 - x_k|| < xtol,
                or the optimizer's own test (by default ||projected gradient||_inf < gtol)
        stop unless EPF; EPF converged if mean(penalty) / r <= conv_crit; else
            r *= r_factor, ftol *= tol_factor, iteration = 0

Objective values are kept per model realization (robust optimization); every comparison
uses their mean.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, ClassVar, cast

import numpy as np

from apex.algorithms.optimization.bounds import BoundTransform
from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    DifferentiableObjective,
    Draws,
    FloatArray,
    IntArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
    VectorObjective,
)
from apex.core.options import mapping

logger = logging.getLogger(__name__)

#: Called after the start, every accepted step and every EPF update; returning
#: ``False`` stops the run there (it can be resumed from :meth:`OptimizerBase.export_state`).
type IterationCallback = Callable[["OptimizerBase"], bool | None]

#: The ``epf`` option: penalty factors, all numbers (a mapping whose keys name
#: nothing on disk).
type EpfOptions = Annotated[Mapping[str, float] | None, mapping()]

#: The message of a run stopped by its ``on_iteration`` callback.
STOPPED = "stopped by the caller"

#: The EPF options and their PET defaults (``r`` has none: it must be given).
EPF_DEFAULTS: dict[str, float] = {
    "r_factor": 2.0,
    "tol_factor": 0.9,
    "conv_crit": 1e-5,
    "max_epf_iter": 10,
}


def external_penalty(r: float, c_eq: FloatArray, c_iq: FloatArray) -> FloatArray:
    r"""Return PET's external penalty of each column ``(n,)``.

    ``0.5 r (sum_i c_eq,i^2 + sum_j max(-c_iq,j, 0)^2)``: equality constraints are satisfied at
    zero, inequality constraints when non-negative.

    Parameters
    ----------
    r
        Penalty factor.
    c_eq, c_iq
        Constraints ``(n_eq, n)`` and ``(n_iq, n)`` (either may have zero rows).
    """
    return np.asarray(
        r * 0.5 * (np.sum(c_eq**2, axis=0) + np.sum(np.maximum(-c_iq, 0) ** 2, axis=0)),
        dtype=float,
    )


class StepReport:
    """What one step produced: accepted (the optimizer committed a new iterate), or why not."""

    __slots__ = ("accepted", "message")

    def __init__(self, accepted: bool, message: str = "") -> None:
        self.accepted = accepted
        self.message = message


@dataclass
class OptimizationResult:
    """The outcome of a run (PET's ``OptimizeResult``), in physical coordinates.

    Attributes
    ----------
    x
        Final controls ``(n_x,)``.
    fun
        Objective at ``x`` on each model realization ``(n_realizations,)``.
    jac, hess
        Last gradient ``(n_x,)`` and Hessian ``(n_x, n_x)``, if any.
    nit
        Iterations of the last outer (EPF) iteration.
    nfev, njev, nhev
        Single-point objective evaluations, gradient and Hessian evaluations.
    member_evaluations
        Objective evaluations of ensemble members.
    message
        Why the run stopped.
    epf_iteration
        Outer EPF iterations (1 without EPF).
    """

    x: FloatArray
    fun: FloatArray
    jac: FloatArray | None
    hess: FloatArray | None
    nit: int
    nfev: int
    njev: int
    nhev: int
    member_evaluations: int
    message: str
    epf_iteration: int
    extra: dict[str, float] = field(default_factory=dict)


class OptimizerBase(ABC):
    """The shared loop; a subclass supplies :meth:`update_step` (see the module docstring).

    Parameters
    ----------
    maxiter
        Iterations per outer EPF iteration.
    ftol
        Relative change of the mean objective that counts as converged.
    xtol
        Change of the controls (2-norm, optimizer coordinates) that counts as converged.
    gtol
        Infinity norm of the projected gradient that counts as converged.
    transform
        Optimize in the unit cube of the bounds.
    epf
        External penalty options: ``r`` (required), ``r_factor`` (default 2),
        ``tol_factor`` (0.9), ``conv_crit`` (1e-5), ``max_epf_iter`` (10). Needs a
        constrained objective.
    """

    status: ClassVar[Status] = "experimental"
    #: Shown in logs.
    label: ClassVar[str] = "optimizer"
    #: Whether the optimizer needs a :class:`DifferentiableObjective` (gradient; Hessian).
    needs_gradient: ClassVar[bool] = False
    #: Asked before each evaluation batch by optimizers that drive their own loop (ropt):
    #: True stops at once, the last checkpoint standing. Set by the workflow
    #: (a stop request); ``None`` outside it.
    stop_check: "Callable[[], bool] | None" = None
    #: Whether it needs the Hessian too (an instance attribute: it may depend on the options).
    needs_hessian: bool = False
    #: Whether the optimizer needs the initial perturbation covariance.
    needs_covariance: ClassVar[bool] = False
    #: Whether it optimizes several objectives at once; the others refuse them.
    multi_objective: ClassVar[bool] = False

    def __init__(
        self,
        *,
        maxiter: int = 100,
        ftol: float = 1e-5,
        xtol: float = 1e-8,
        gtol: float = 1e-5,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        if maxiter < 1:
            raise ValueError(f"maxiter must be positive, got {maxiter}")
        self.maxiter = maxiter
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.transform = transform
        self.epf: dict[str, float] = {}
        if epf:
            unknown = set(epf) - {"r", *EPF_DEFAULTS}
            if unknown:
                raise ValueError(f"unknown EPF options {sorted(unknown)}")
            if "r" not in epf:
                raise ValueError("EPF needs the initial penalty factor r")
            self.epf = {**EPF_DEFAULTS, **{k: float(v) for k, v in epf.items()}}
        self.epf_maxiter = int(self.epf["max_epf_iter"]) if self.epf else 1

    # ------------------------------------------------------------------ set-up

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        """Bind the problem; subclasses extend this to build their estimators."""
        if isinstance(problem.objective, VectorObjective) and not self.multi_objective:
            raise ValueError(
                f"{self.label} optimizes one objective, and this problem has several "
                f"({', '.join(problem.objective.objective_names)}): use nsga2, or one objective"
            )
        if self.needs_gradient and not isinstance(problem.objective, DifferentiableObjective):
            raise ValueError(f"{self.label} needs an objective with a gradient and a Hessian")
        if self.epf and not isinstance(problem.objective, ConstrainedObjective):
            raise ValueError("EPF needs an objective with constraints")
        if self.epf and self.needs_gradient:
            # The objective's exact gradient lacks the penalty's; PET leaves that to the
            # user's jac. The ensemble optimizers estimate the penalized gradient themselves.
            raise ValueError(
                f"{self.label} cannot use EPF: the objective's gradient does not include the "
                "penalty's; use an ensemble optimizer (enopt, genopt, smcopt)"
            )
        if self.needs_covariance and cov is None:
            raise ValueError(f"{self.label} needs the initial perturbation covariance")
        self.problem = problem
        self.draws = draws
        self.bounds = BoundTransform(problem.lower, problem.upper, transform=self.transform)
        self.xk: FloatArray = self.bounds.to_internal(problem.x0.copy())
        self.fk: FloatArray | None = None
        self.jk: FloatArray | None = None
        self.hk: FloatArray | None = None
        self.xk_old: FloatArray | None = None
        self.fk_old: FloatArray | None = None
        self.iteration = 0
        self.epf_iteration = 0
        self.conv_msg = ""
        self.nfev = self.njev = self.nhev = 0
        self.member_evaluations = 0
        self.last_penalty: FloatArray | None = None
        self._inner_done = False
        self._started = False
        self.stopped = False

    def minimize(
        self,
        problem: Problem,
        draws: Draws | None = None,
        *,
        cov: FloatArray | None = None,
        on_iteration: IterationCallback | None = None,
        state: Mapping[str, StateValue] | None = None,
    ) -> OptimizationResult:
        """Run (or, given ``state``, continue) the optimization.

        Parameters
        ----------
        problem
            The objective, ``x0``, the bounds and the model realizations.
        draws
            Source of random draws (the ensemble optimizers need one).
        cov
            Initial perturbation covariance ``(n_x, n_x)`` (physical units).
        on_iteration
            See :data:`IterationCallback`.
        state
            A state from :meth:`export_state` to continue from.
        """
        self._setup(problem, draws, cov)
        if state is not None:
            self.restore_state(state)
        self.stopped = self._run(on_iteration)
        return self.result()

    def _run(self, on_iteration: IterationCallback | None) -> bool:
        if not self._started:
            self._start()
            if on_iteration is not None and on_iteration(self) is False:
                return True
        if self.epf_iteration == 0:
            self.epf_iteration = 1
        while self.epf_iteration <= self.epf_maxiter:
            self._refresh_epf_function_value()
            failed = False
            if not self._inner_done:
                while self.iteration < self.maxiter:
                    self.iteration += 1
                    report = self.update_step()
                    if not report.accepted:
                        # PET reports the iteration limit even when the last step failed.
                        self.conv_msg = (
                            "Maximum number of iterations reached"
                            if self.iteration >= self.maxiter
                            else report.message
                        )
                        failed = True
                        break
                    self._log_iteration()
                    converged = (
                        self.check_function_convergence()
                        or self.check_state_convergence()
                        or self.check_convergence()
                    )
                    self._inner_done = converged or self.iteration >= self.maxiter
                    if self.iteration >= self.maxiter and not converged:
                        self.conv_msg = "Maximum number of iterations reached"
                    if on_iteration is not None and on_iteration(self) is False:
                        return True
                    if converged:
                        break
            if failed or not self.epf:
                break
            if self._check_epf_convergence():
                break
            self.iteration = 0
            self.epf_iteration += 1
            self._inner_done = False
            if on_iteration is not None and on_iteration(self) is False:
                return True
        self._log_convergence()
        return False

    def max_evaluations(self, n_realizations: int) -> int | None:
        """Return an upper bound on the objective's evaluations (one column each), if known.

        The start, then per EPF iteration a refresh and ``maxiter`` iterations of at most
        :meth:`evaluations_per_iteration`. With a simulated objective each evaluation is one
        simulator run; the evaluation journal makes the actual count smaller.
        """
        per_iteration = self.evaluations_per_iteration(n_realizations)
        if per_iteration is None:
            return None
        return n_realizations + self.epf_maxiter * (n_realizations + self.maxiter * per_iteration)

    def evaluations_per_iteration(self, n_realizations: int) -> int | None:
        """Return the most evaluations one iteration makes (``None``: not bounded here)."""
        return None

    def _start(self) -> None:
        self._started = True
        logger.info("optimizer_start", extra={"optimizer": self.label, "n_controls": self.xk.size})
        if self.fk is None:
            self.fk = self._objective_value(self.xk)
        self._evaluate_missing_derivatives()
        self._log_iteration()

    # ------------------------------------------------------------------ the step

    @abstractmethod
    def update_step(self) -> StepReport:
        """Take one step: commit a better point with :meth:`_commit_step`, or say why not."""

    def _objective_value(self, x: FloatArray) -> FloatArray:
        """Return the objective at ``x`` as this optimizer keeps it (per realization by default)."""
        return self.fun(x)

    def _evaluate_missing_derivatives(self) -> None:
        """Evaluate the gradient and Hessian at ``x_k`` when needed and missing."""
        if self.needs_gradient and self.jk is None:
            self.jk = self.jac(self.xk)
        if self.needs_hessian and self.hk is None:
            self.hk = self.hess(self.xk)

    def _commit_step(
        self,
        x_new: FloatArray,
        f_new: FloatArray,
        jac: FloatArray | None = None,
        hess: FloatArray | None = None,
    ) -> None:
        """Make ``x_new`` current; the iterate it replaces becomes ``x_old``."""
        self.xk_old, self.fk_old = self.xk, self.fk
        self.xk, self.fk = x_new, f_new
        if jac is not None:
            self.jk = jac
        if hess is not None:
            self.hk = hess

    # ------------------------------------------------------------------ evaluations

    def fun(self, x: FloatArray) -> FloatArray:
        """Evaluate one point (optimizer coordinates) on every realization ``(n_real,)``."""
        self.nfev += 1
        physical = self.bounds.to_physical(self.bounds.project(x))
        n_real = self.problem.n_realizations
        controls = np.repeat(physical[:, None], n_real, axis=1)
        values, penalty = self._penalized(controls, np.arange(n_real, dtype=np.int64))
        self.last_penalty = penalty
        return values

    def evaluate_members(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Evaluate ensemble members (physical controls ``(n_x, n)``); returns ``(n,)``."""
        self.member_evaluations += controls.shape[1]
        values, _ = self._penalized(controls, realizations)
        return values

    def jac(self, x: FloatArray) -> FloatArray:
        """Return the objective's gradient at ``x`` in optimizer coordinates."""
        self.njev += 1
        objective = cast(DifferentiableObjective, self.problem.objective)
        physical = self.bounds.to_physical(self.bounds.project(x))
        return self.bounds.gradient_to_internal(np.asarray(objective.gradient(physical), float))

    def hess(self, x: FloatArray) -> FloatArray:
        """Return the objective's Hessian at ``x`` in optimizer coordinates."""
        self.nhev += 1
        objective = cast(DifferentiableObjective, self.problem.objective)
        physical = self.bounds.to_physical(self.bounds.project(x))
        return self.bounds.hessian_to_internal(np.asarray(objective.hessian(physical), float))

    def _penalized(
        self, controls: FloatArray, realizations: IntArray
    ) -> tuple[FloatArray, FloatArray | None]:
        values = np.asarray(self.problem.objective(controls, realizations), dtype=float)
        if values.shape != (controls.shape[1],):
            raise ValueError(
                f"the objective returned shape {values.shape} for {controls.shape[1]} points"
            )
        if not self.epf:
            return values, None
        objective = cast(ConstrainedObjective, self.problem.objective)
        c_eq, c_iq = objective.constraints(controls)
        penalty = external_penalty(self.epf["r"], np.atleast_2d(c_eq), np.atleast_2d(c_iq))
        return values + penalty, penalty

    # ------------------------------------------------------------------ convergence

    def check_convergence(self) -> bool:
        """Run the optimizer's own test; by default the projected gradient against ``gtol``."""
        if self.jk is None:
            return False
        projected = self.bounds.project_gradient(self.xk, self.jk)
        if np.linalg.norm(projected, np.inf) < self.gtol:
            self.conv_msg = f"Projected gradient norm ‖g‖∞ < {self.gtol}."
            return True
        return False

    def check_function_convergence(self) -> bool:
        """Converged when the mean objective changed by less than ``ftol`` relative."""
        if self.fk_old is not None and self.fk is not None:
            change = np.mean(self.fk) - np.mean(self.fk_old)
            if abs(change) < self.ftol * np.abs(np.mean(self.fk_old)):
                self.conv_msg = f"Function change satisfies |Δf| < {self.ftol}·|f_prev|"
                return True
        return False

    def check_state_convergence(self) -> bool:
        """Converged when the controls moved by less than ``xtol`` (2-norm)."""
        if self.xk_old is not None and np.linalg.norm(self.xk - self.xk_old) < self.xtol:
            self.conv_msg = f"State change norm ‖Δx‖₂ < {self.xtol}"
            return True
        return False

    def _check_epf_convergence(self) -> bool:
        """End the EPF loop, or raise ``r`` and tighten ``ftol`` for another outer iteration."""
        if self.epf_iteration == self.epf_maxiter:
            logger.info("epf_max_iterations", extra={"epf_iteration": self.epf_iteration})
            return True
        if self.last_penalty is None or self.last_penalty.size == 0:
            raise RuntimeError("EPF convergence needs the penalty of the current iterate")
        # The violation itself: the penalty with the factor r divided back out.
        violation = float(np.mean(self.last_penalty)) / self.epf["r"]
        if violation > self.epf["conv_crit"]:
            self.epf["r"] *= self.epf["r_factor"]
            self.ftol *= self.epf["tol_factor"]
            logger.info(
                "epf_update",
                extra={"r": self.epf["r"], "ftol": self.ftol, "violation": violation},
            )
            return False
        logger.info("epf_converged", extra={"violation": violation})
        return True

    def _refresh_epf_function_value(self) -> None:
        # A new penalty factor changes the objective at the current iterate.
        if self.epf_iteration <= 1 or self.iteration != 0:
            return
        self.fk = self.fun(self.xk)

    # ------------------------------------------------------------------ results and logs

    def result(self) -> OptimizationResult:
        """Return the current result, in physical coordinates."""
        x = self.bounds.to_physical(self.bounds.project(self.xk))
        jac = None if self.jk is None else self.bounds.gradient_to_physical(self.jk)
        hess = None if self.hk is None else self.bounds.hessian_to_physical(self.hk)
        return OptimizationResult(
            x=x,
            fun=np.atleast_1d(np.asarray(self.fk, dtype=float)),
            jac=jac,
            hess=hess,
            nit=self.iteration,
            nfev=self.nfev,
            njev=self.njev,
            nhev=self.nhev,
            member_evaluations=self.member_evaluations,
            # A run stopped by the caller keeps its own message in the state, for the resume.
            message=STOPPED if self.stopped else self.conv_msg,
            epf_iteration=max(self.epf_iteration, 1),
            extra=self.result_extra(),
        )

    def result_extra(self) -> dict[str, float]:
        """Optimizer-specific scalars for the result and the history."""
        return {}

    def log_columns(self) -> dict[str, float]:
        """One row of the iteration history."""
        return {"iteration": self.iteration, "objective": float(np.mean(cast(FloatArray, self.fk)))}

    def _log_iteration(self) -> None:
        columns = self.log_columns()
        if self.epf:
            columns["epf_iteration"] = self.epf_iteration
        logger.info("optimizer_iteration", extra={"optimizer": self.label, **columns})

    def _log_convergence(self) -> None:
        logger.info(
            "optimizer_done",
            extra={
                "optimizer": self.label,
                "reason": self.conv_msg,
                "objective": float(np.mean(cast(FloatArray, self.fk))),
                "iterations": self.iteration,
                "nfev": self.nfev,
            },
        )

    # ------------------------------------------------------------------ checkpoints

    def export_state(self) -> OptimizerState:
        """Return everything the loop needs to continue bit-identically."""
        state: OptimizerState = {
            "xk": self.xk,
            "fk": self.fk,
            "jk": self.jk,
            "hk": self.hk,
            "xk_old": self.xk_old,
            "fk_old": self.fk_old,
            "iteration": self.iteration,
            "epf_iteration": self.epf_iteration,
            "ftol": self.ftol,
            "epf": json.dumps(self.epf),
            "conv_msg": self.conv_msg,
            "nfev": self.nfev,
            "njev": self.njev,
            "nhev": self.nhev,
            "member_evaluations": self.member_evaluations,
            "last_penalty": self.last_penalty,
            "inner_done": self._inner_done,
            "started": self._started,
            "draws": None if self.draws is None else json.dumps(self.draws.export_state()),
        }
        for key, value in self._export_extra().items():
            state[f"opt.{key}"] = value
        return state

    def restore_state(self, state: Mapping[str, StateValue]) -> None:
        """Continue from a state returned by :meth:`export_state` (after set-up)."""

        def array(key: str) -> FloatArray | None:
            value = state.get(key)
            return None if value is None else np.array(value, dtype=float)

        xk = array("xk")
        if xk is None:
            raise ValueError("the optimizer state has no iterate")
        self.xk = xk
        self.fk, self.jk, self.hk = array("fk"), array("jk"), array("hk")
        self.xk_old, self.fk_old = array("xk_old"), array("fk_old")
        self.last_penalty = array("last_penalty")
        self.iteration = int(cast(int, state["iteration"]))
        self.epf_iteration = int(cast(int, state["epf_iteration"]))
        self.ftol = float(cast(float, state["ftol"]))
        self.epf = {k: float(v) for k, v in json.loads(str(state["epf"])).items()}
        self.conv_msg = str(state["conv_msg"])
        self.nfev = int(cast(int, state["nfev"]))
        self.njev = int(cast(int, state["njev"]))
        self.nhev = int(cast(int, state["nhev"]))
        self.member_evaluations = int(cast(int, state["member_evaluations"]))
        self._inner_done = bool(state["inner_done"])
        self._started = bool(state["started"])
        draws = state.get("draws")
        if draws is not None and self.draws is not None:
            self.draws.restore_state(json.loads(str(draws)))
        self._restore_extra(
            {k.removeprefix("opt."): v for k, v in state.items() if k.startswith("opt.")}
        )

    def _export_extra(self) -> OptimizerState:
        """Return the subclass's own state."""
        return {}

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:  # noqa: B027
        """Restore the subclass's own state (none by default)."""


def as_float(value: StateValue) -> float:
    """Read a float from a state value."""
    return float(cast(float, value))


def as_array(value: StateValue) -> FloatArray:
    """Read an array from a state value."""
    return np.array(value, dtype=float)
