"""Gradient-based optimizers: line search (GD, BFGS, Newton-CG) and trust region.

Ported from PET
---------------
PET module : src/popt/optimization_methods/linesearch.py (LineSearch),
             src/popt/optimization_methods/trust_region.py (TrustRegion)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.), chapters
             3 (line search), 4 (trust region), 6 (quasi-Newton) and 7 (Newton-CG).
             Springer.
Differences: none. PET's unused ``normalize`` option and its initial inverse Hessian
             ``hess0_inv`` (always the identity in its examples) are not ported. Kept from
             PET: a rejected trust-region step cuts the radius by 0.25, whatever ``gam1``.

They need an objective with its gradient (and, except BFGS, its Hessian):
``DifferentiableObjective``.
"""

from collections.abc import Mapping
from typing import ClassVar, Literal, cast

import numpy as np

from apex.algorithms.optimization.base import EpfOptions, OptimizerBase, StepReport, as_float
from apex.algorithms.optimization.line_search import (
    backtracking_line_search,
    bfgs_update,
    newton_cg,
    trust_region_step,
    wolfe_line_search,
)
from apex.algorithms.optimization.protocols import (
    Draws,
    FloatArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
)
from apex.algorithms.optimization.registries import optimizers


def _mean(values: FloatArray | None) -> float:
    return float(np.mean(cast(FloatArray, values)))


@optimizers.register("line_search")
class LineSearch(OptimizerBase):
    """Line-search optimization; options are PET's.

    Parameters
    ----------
    method
        Search direction: ``GD`` (steepest descent), ``BFGS`` or ``Newton-CG``.
    step_size
        First step length (default ``0.25 / ||p||_inf`` at the first iteration).
    step_size_max
        Largest step length.
    step_size_adapt
        Later first trials: 0 keep ``step_size``, 1 from the last decrease
        (``2 (f_k - f_k-1) / (g^T p)``), 2 from the slope ratio.
    c1, c2
        Wolfe constants.
    rho
        Backtracking factor.
    lsmaxiter
        Trial step lengths per line search.
    lsmethod
        0 backtracking (Armijo), 1 strong Wolfe.
    recompute_jac
        Gradient re-evaluations (and retries) after a failed line search.
    maxiter, ftol, xtol, gtol, transform, epf
        See :class:`OptimizerBase`.
    """

    status: ClassVar[Status] = "parity"
    label: ClassVar[str] = "LineSearch"
    needs_gradient: ClassVar[bool] = True

    def __init__(
        self,
        *,
        method: Literal["GD", "BFGS", "Newton-CG"] = "GD",
        step_size: float | None = None,
        step_size_max: float = 1e5,
        step_size_adapt: Literal[0, 1, 2] = 1,
        c1: float = 1e-4,
        c2: float = 0.9,
        rho: float = 0.5,
        lsmaxiter: int = 10,
        lsmethod: Literal[0, 1] = 1,
        recompute_jac: int = 0,
        maxiter: int = 100,
        ftol: float = 1e-5,
        xtol: float = 1e-8,
        gtol: float = 1e-5,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        super().__init__(
            maxiter=maxiter, ftol=ftol, xtol=xtol, gtol=gtol, transform=transform, epf=epf
        )
        self.method = method
        self.needs_hessian = method == "Newton-CG"
        self.initial_step_size = step_size
        self.step_size_max = step_size_max
        self.step_size_adapt = step_size_adapt
        self.c1, self.c2, self.rho = c1, c2, rho
        self.lsmaxiter = lsmaxiter
        self.lsmethod = lsmethod
        self.recompute_jac = recompute_jac

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        self.step_size = self.initial_step_size
        self.step_taken: float | None = None
        self.jk_old: FloatArray | None = None
        self.pk_old: FloatArray | None = None
        self.bk = np.eye(self.xk.size)

    def update_step(self) -> StepReport:
        """Find a direction, search along it, and commit the point found."""
        retries = 0
        while retries <= self.recompute_jac:
            self._evaluate_missing_derivatives()
            pk = self._direction()
            step, f_new, j_new = self._line_search(pk)
            if step:
                self._accept(pk, step, cast(float, f_new), cast(FloatArray, j_new))
                return StepReport(True)
            if retries < self.recompute_jac:
                self.jk = None
                retries += 1
            else:
                break
        return StepReport(False, "Line search failed to find a suitable step size")

    iteration_evaluations: ClassVar[str] = (
        "per attempt, 1 + recompute_jac: up to lsmaxiter + 1 step lengths, each on every "
        "realization"
    )

    def evaluations_per_iteration(self, n_realizations: int) -> int:
        """Return the most evaluations of one iteration.

        Each of ``1 + recompute_jac`` attempts runs a line search, which evaluates each step
        length once: at most ``lsmaxiter + 1`` of them (Wolfe: the bracketing phase and the
        zoom share ``lsmaxiter + 1``; backtracking: ``lsmaxiter``), each on every
        realization. Gradients and Hessians are the objective's own, not evaluations.
        """
        return (1 + self.recompute_jac) * (self.lsmaxiter + 1) * n_realizations

    def _direction(self) -> FloatArray:
        gradient = cast(FloatArray, self.jk)
        if self.method == "GD":
            return -gradient
        if self.method == "BFGS":
            return -(self.bk @ gradient)
        return newton_cg(gradient, cast(FloatArray, self.hk))

    def _initial_step(self, pk: FloatArray) -> float:
        if self.step_size is None:
            norm = float(np.linalg.norm(pk, np.inf))
            if norm == 0:
                raise FloatingPointError("the search direction is zero: the gradient vanishes")
            self.step_size = 0.25 / norm
        alpha = float(self.step_size)
        if self.iteration > 1:
            slope = float(np.dot(pk, cast(FloatArray, self.jk)))
            if self.step_size_adapt == 1 and slope != 0:
                alpha = 2 * (_mean(self.fk) - _mean(self.fk_old)) / slope
            elif self.step_size_adapt == 2 and slope != 0:
                slope_old = float(
                    np.dot(cast(FloatArray, self.pk_old), cast(FloatArray, self.jk_old))
                )
                alpha = self.step_size * slope_old / slope
        alpha = abs(alpha)
        return 0.75 * self.step_size_max if alpha >= self.step_size_max else alpha

    def _line_search(self, pk: FloatArray) -> tuple[float | None, float | None, FloatArray | None]:
        step = self._initial_step(pk)

        def objective(x: FloatArray) -> float:
            return float(np.mean(self.fun(x)))

        jk = cast(FloatArray, self.jk)
        if self.lsmethod == 1:
            return wolfe_line_search(
                step,
                self.xk,
                pk,
                objective,
                self.jac,
                _mean(self.fk),
                jk,
                amax=self.step_size_max,
                maxiter=self.lsmaxiter,
                c1=self.c1,
                c2=self.c2,
            )
        return backtracking_line_search(
            step,
            self.xk,
            pk,
            objective,
            self.jac,
            _mean(self.fk),
            jk,
            rho=self.rho,
            maxiter=self.lsmaxiter,
            c1=self.c1,
        )

    def _accept(self, pk: FloatArray, step: float, f_new: float, j_new: FloatArray) -> None:
        self.jk_old, self.pk_old, self.step_taken = self.jk, pk, step
        self._commit_step(
            self.bounds.project(self.xk + step * pk), np.atleast_1d(float(f_new)), jac=j_new
        )
        if self.method == "BFGS":
            sk = self.xk - cast(FloatArray, self.xk_old)
            yk = cast(FloatArray, self.jk) - cast(FloatArray, self.jk_old)
            if self.iteration == 1 and np.dot(yk, yk) > 0:
                # Scale the first inverse Hessian (Nocedal & Wright, eq. 6.20).
                self.bk = np.dot(yk, sk) / np.dot(yk, yk) * np.eye(sk.size)
            self.bk = bfgs_update(self.bk, sk, yk)
        # The Hessian on hand is the previous iterate's.
        self.hk = None

    def log_columns(self) -> dict[str, float]:
        """Return the iteration, objective, gradient infinity norm and step length taken."""
        step = self.step_taken if self.step_taken is not None else self.step_size
        return {
            **super().log_columns(),
            "gradient_inf_norm": float(np.linalg.norm(cast(FloatArray, self.jk), np.inf)),
            "step_size": float("nan") if step is None else float(step),
        }

    def _export_extra(self) -> OptimizerState:
        return {
            "step_size": self.step_size,
            "step_taken": self.step_taken,
            "jk_old": self.jk_old,
            "pk_old": self.pk_old,
            "bk": self.bk,
        }

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self.step_size = None if state["step_size"] is None else as_float(state["step_size"])
        self.step_taken = None if state["step_taken"] is None else as_float(state["step_taken"])
        jk_old, pk_old = state["jk_old"], state["pk_old"]
        self.jk_old = None if jk_old is None else np.array(jk_old, dtype=float)
        self.pk_old = None if pk_old is None else np.array(pk_old, dtype=float)
        self.bk = np.array(state["bk"], dtype=float)


@optimizers.register("trust_region")
class TrustRegion(OptimizerBase):
    """Trust-region optimization; options are PET's.

    Parameters
    ----------
    method
        Subproblem solver: ``iterative`` (exact) or ``CG-Steihaug``.
    quasi_newton
        Approximate the Hessian by BFGS (PET's ``hess = "BFGS"``).
    trust_radius
        Initial radius.
    trust_radius_max, trust_radius_min
        Radius limits (default ``100`` and ``1/1000`` times the initial radius); the run
        stops at the minimum.
    trust_radius_cuts
        Radius cuts before a step is rejected.
    rho_tol
        Least ratio of actual to predicted decrease for a step to be accepted.
    eta1, eta2
        Ratios below which the radius shrinks by ``gam1``, and above which (at the boundary)
        it grows by ``gam2``.
    gam1, gam2
        Radius factors.
    resample
        Re-evaluate the gradient and the Hessian after a rejected step.
    maxiter, ftol, xtol, gtol, transform, epf
        See :class:`OptimizerBase`.
    """

    status: ClassVar[Status] = "parity"
    label: ClassVar[str] = "TrustRegion"
    needs_gradient: ClassVar[bool] = True

    def __init__(
        self,
        *,
        method: Literal["iterative", "CG-Steihaug"] = "iterative",
        quasi_newton: bool = False,
        trust_radius: float = 1.0,
        trust_radius_max: float | None = None,
        trust_radius_min: float | None = None,
        trust_radius_cuts: int = 4,
        rho_tol: float = 1e-6,
        eta1: float = 0.05,
        eta2: float = 0.5,
        gam1: float = 0.5,
        gam2: float = 1.5,
        resample: bool = False,
        maxiter: int = 100,
        ftol: float = 1e-5,
        xtol: float = 1e-8,
        gtol: float = 1e-5,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        super().__init__(
            maxiter=maxiter, ftol=ftol, xtol=xtol, gtol=gtol, transform=transform, epf=epf
        )
        self.method = method
        self.quasi_newton = quasi_newton
        self.needs_hessian = not quasi_newton
        self.initial_radius = trust_radius
        self.trust_radius_max = 100 * trust_radius if trust_radius_max is None else trust_radius_max
        self.trust_radius_min = (
            trust_radius / 1000 if trust_radius_min is None else trust_radius_min
        )
        self.trust_radius_cuts = trust_radius_cuts
        self.rho_tol, self.eta1, self.eta2 = rho_tol, eta1, eta2
        self.gam1, self.gam2 = gam1, gam2
        self.resample = resample

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        self.trust_radius = self.initial_radius
        self.rho = 0.0
        self.hits_boundary: bool | None = None
        self.jk_old: FloatArray | None = None

    def _objective_value(self, x: FloatArray) -> FloatArray:
        # The reduction ratio needs one number: the mean over the realizations.
        return np.atleast_1d(float(np.mean(self.fun(x))))

    def update_step(self) -> StepReport:
        """Try steps, cutting the radius, until one is accepted or the cuts run out."""
        self._evaluate_missing_derivatives()
        for _ in range(self.trust_radius_cuts + 1):
            report = self._attempt()
            if report is not None:
                return report
        return StepReport(False, "Trust-region step rejected after radius cut attempts.")

    iteration_evaluations: ClassVar[str] = (
        "up to 1 + trust_radius_cuts trial steps, each on every realization"
    )

    def evaluations_per_iteration(self, n_realizations: int) -> int:
        """Return the most evaluations of one iteration.

        Up to ``1 + trust_radius_cuts`` trial steps, each evaluated once on every
        realization. Gradients and Hessians are the objective's own, not evaluations.
        """
        return (1 + self.trust_radius_cuts) * n_realizations

    def check_convergence(self) -> bool:
        """Run the projected-gradient test, then stop at the minimum radius."""
        if super().check_convergence():
            return True
        if self.trust_radius <= self.trust_radius_min:
            self.conv_msg = f"Trust-region radius Δₖ <= {self.trust_radius_min}."
            return True
        return False

    def _attempt(self) -> StepReport | None:
        jk = cast(FloatArray, self.jk)
        gradient = self.bounds.project_gradient(self.xk, jk)
        first_bfgs = self.quasi_newton and self.hk is None and self.iteration == 1
        if first_bfgs:
            # No curvature yet: a steepest-descent step to the boundary (infinity norm).
            sk = -gradient
            norm = float(np.linalg.norm(sk, np.inf))
            if norm > 0:
                sk = sk / norm * self.trust_radius
            hits = True
        else:
            if self.hk is None:
                self.hk = self.hess(self.xk)
            sk, hits = trust_region_step(
                self.xk, _mean(self.fk), gradient, self.hk, self.trust_radius, self.method
            )
        x_new = self.bounds.project(self.xk + sk)
        f_new = self._objective_value(x_new)
        actual = _mean(self.fk) - _mean(f_new)
        if first_bfgs:
            predicted = -float(np.dot(gradient, sk))
        else:
            hk = self.hk if self.hk is not None else self.hess(self.xk)
            predicted = -float(np.dot(gradient, sk)) - 0.5 * float(np.dot(sk, hk @ sk))
        self.rho = actual / predicted if predicted != 0 else -np.inf
        if self.rho > self.rho_tol and _mean(f_new) < _mean(self.fk):
            self._accept(x_new, f_new, sk, hits)
            return StepReport(True)
        self.trust_radius *= 0.25
        if self.trust_radius < self.trust_radius_min:
            return StepReport(False, "Trust-region radius Δₖ below minimum.")
        if self.resample:
            self.jk = self.jac(self.xk)
            if not self.quasi_newton:
                self.hk = self.hess(self.xk)
        return None

    def _accept(self, x_new: FloatArray, f_new: FloatArray, sk: FloatArray, hits: bool) -> None:
        self.jk_old = self.jk
        self._commit_step(x_new, f_new)
        self.jk = self.jac(self.xk)
        if self.quasi_newton:
            yk = self.jk - cast(FloatArray, self.jk_old)
            if self.hk is None:
                curvature = float(np.dot(yk, sk))
                scale = np.dot(yk, yk) / curvature if curvature > 0 else 1.0
                self.hk = scale * np.eye(self.xk.size)
            self.hk = _bfgs_hessian_update(self.hk, sk, yk)
        else:
            self.hk = self.hess(self.xk)
        old = self.trust_radius
        if self.rho >= self.eta2 and hits:
            new = min(self.gam2 * old, self.trust_radius_max)
        elif self.rho < self.eta1:
            new = self.gam1 * old
        else:
            new = old
        self.trust_radius = float(np.clip(new, self.trust_radius_min, self.trust_radius_max))
        self.hits_boundary = hits

    def log_columns(self) -> dict[str, float]:
        """Return the iteration, objective, radius and reduction ratio."""
        return {**super().log_columns(), "trust_radius": self.trust_radius, "rho": self.rho}

    def _export_extra(self) -> OptimizerState:
        return {
            "trust_radius": self.trust_radius,
            "rho": self.rho,
            "jk_old": self.jk_old,
            "hits_boundary": self.hits_boundary,
        }

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self.trust_radius = as_float(state["trust_radius"])
        self.rho = as_float(state["rho"])
        jk_old = state["jk_old"]
        self.jk_old = None if jk_old is None else np.array(jk_old, dtype=float)
        hits = state["hits_boundary"]
        self.hits_boundary = None if hits is None else bool(hits)


def _bfgs_hessian_update(b: FloatArray, s: FloatArray, y: FloatArray) -> FloatArray:
    """Return the BFGS update of the Hessian ``b``; skipped without positive curvature."""
    s, y = s.reshape(-1, 1), y.reshape(-1, 1)
    ys = (y.T @ s).item()
    sbs = (s.T @ b @ s).item()
    if ys <= 0 or sbs <= 0:
        return b
    return np.asarray(b + (y @ y.T) / ys - (b @ s) @ (s.T @ b) / sbs)
