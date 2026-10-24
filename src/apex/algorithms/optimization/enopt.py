"""EnOpt: ensemble optimization with a Gaussian perturbation ensemble.

Ported from PET
---------------
PET module : src/popt/optimization_methods/enopt.py (EnOpt), with the gradient and Hessian
             of src/popt/ensembles/ensemble_gaussian.py (GaussianEnsemble)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Chen, Y., Oliver, D. S. & Zhang, D. (2009). Efficient ensemble-based closed-loop
             production optimization. SPE Journal, 14(4), 634-645. Zhang, Y., Stordal, A. S.
             & Lorentzen, R. J. (2023). A natural Hessian approximation for ensemble based
             optimization. Computational Geosciences, 27, 355-364.
Differences: the gradient's baseline is ``f(x_k)`` (ensemble_gradient.py). PET's ``hess``
             callable is the option ``ensemble_hessian``. Otherwise none.

One iteration (PET's ``update_step``)::

    for resampling attempt k = 0 .. resample:
        draw an ensemble with covariance cov_factor^k C (shrinking the step by
        sqrt(cov_factor)^k) and estimate the gradient g (and the Hessian H)
        g <- H^-1 C C g with ``hessian``; else g / ||g||_inf with ``normalize``
        for backtracking trial j = 0 .. alpha_maxiter:
            x_trial = step_rule(x_k, g); accept if mean f(x_k) - mean f(x_trial) > tol
            else shrink the step by 0.5
        resample only if the last trial was worse than x_k
    on acceptance, with a Hessian: C <- floor_eig(C - (alpha_cov H + beta dC_old))
"""

from collections.abc import Mapping
from typing import Annotated, ClassVar, cast

import numpy as np

from apex.algorithms.optimization.base import (
    EpfOptions,
    OptimizerBase,
    StepReport,
    as_array,
    as_float,
)
from apex.algorithms.optimization.ensemble_gradient import GaussianEnsemble
from apex.algorithms.optimization.matrices import floor_eigenvalues
from apex.algorithms.optimization.protocols import (
    Draws,
    FloatArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
)
from apex.algorithms.optimization.registries import optimizers, step_rules
from apex.algorithms.optimization.step_rules import StepRule
from apex.core.options import TEXT


def build_step_rule(name: str, step_size: float, momentum: float, trust_radius: float) -> StepRule:
    """Build a step rule by name (PET's ``GD``, ``Adam``, ``AdaMax``, ``Steihaug``; any case).

    The CG-Steihaug rule starts from ``trust_radius`` (PET's EnOpt: 3.0) instead of the step
    size.
    """
    key = name.lower()
    return step_rules.create(key, trust_radius if key == "steihaug" else step_size, momentum)


def objective_tolerances(tol: float | None, ftol: float | None) -> tuple[float, float]:
    """Return PET's improvement tolerance and ``ftol``: ``tol`` sets both when given."""
    return (1e-6 if tol is None else tol), (
        tol if tol is not None else (1e-5 if ftol is None else ftol)
    )


@optimizers.register("enopt")
class EnOpt(OptimizerBase):
    """Ensemble optimization (EnOpt): steps along an ensemble estimate of the gradient.

    Parameters
    ----------
    ne
        Ensemble size (PET's ``ne``; default 100).
    natural_gradient
        Use the natural gradient ``C grad f`` (PET's default) instead of ``grad f``.
    tol
        Improvement of the mean objective a trial needs; also ``ftol`` when given.
    alpha, step_size
        Initial step size (``step_size`` overrides ``alpha``).
    alpha_cov
        Step size of the covariance update.
    beta
        Momentum of the step and of the covariance update (Adam's first moment rate).
    nesterov
        Estimate the gradient at the momentum-extrapolated point.
    alpha_maxiter
        Backtracking trials per resampling attempt.
    resample
        Resampling attempts when backtracking fails.
    ensemble_hessian
        Estimate the ensemble Hessian (PET passes ``hess``): it adapts the covariance and
        feeds the Steihaug rule.
    hessian
        Precondition the gradient with the Hessian (needs ``ensemble_hessian``).
    normalize
        Scale the gradient and the Hessian by their infinity norms.
    cov_factor
        Covariance shrink factor per resampling attempt.
    optimizer
        Step rule: ``GD``, ``Adam``, ``AdaMax`` (turns ``normalize`` off) or ``Steihaug``
        (needs ``ensemble_hessian``), or any registered ``step_rules`` name.
    trust_radius
        Initial radius of the Steihaug rule.
    maxiter, ftol, xtol, gtol, transform, epf
        See :class:`OptimizerBase`.
    """

    status: ClassVar[Status] = "parity"
    label: ClassVar[str] = "EnOpt"
    needs_covariance: ClassVar[bool] = True

    def __init__(
        self,
        *,
        ne: int = 100,
        natural_gradient: bool = True,
        tol: float | None = None,
        alpha: float = 0.1,
        step_size: float | None = None,
        alpha_cov: float = 0.001,
        beta: float = 0.0,
        nesterov: bool = False,
        alpha_maxiter: int = 5,
        resample: int = 0,
        ensemble_hessian: bool = False,
        hessian: bool = False,
        normalize: bool = True,
        cov_factor: float = 0.5,
        optimizer: Annotated[str, TEXT] = "GD",
        trust_radius: float = 3.0,
        maxiter: int = 100,
        ftol: float | None = None,
        xtol: float = 1e-8,
        gtol: float = 1e-5,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        self.obj_func_tol, resolved_ftol = objective_tolerances(tol, ftol)
        super().__init__(
            maxiter=maxiter, ftol=resolved_ftol, xtol=xtol, gtol=gtol, transform=transform, epf=epf
        )
        if hessian and not ensemble_hessian:
            raise ValueError("hessian = true needs ensemble_hessian = true")
        if hessian and transform:
            # PET's H^-1 C C g mixes the unit-cube H and g with the physical C.
            raise ValueError("hessian = true cannot be combined with transform = true")
        if optimizer.lower() == "steihaug" and not ensemble_hessian:
            raise ValueError("the Steihaug step rule needs ensemble_hessian = true")
        if ne < 1:
            raise ValueError(f"ne (the ensemble size) must be at least 1, got {ne}")
        self.ne = ne
        self.natural_gradient = natural_gradient
        self.alpha = alpha if step_size is None else step_size
        self.alpha_cov = alpha_cov
        self.beta = beta
        self.nesterov = nesterov
        self.alpha_iter_max = alpha_maxiter
        self.max_resample = resample
        self.use_ensemble_hessian = ensemble_hessian
        self.use_hessian = hessian
        # PET's builder turns normalisation off for AdaMax, whose step is scale-free.
        self.normalize = normalize and optimizer.lower() != "adamax"
        self.cov_factor = cov_factor
        self.optimizer_name = optimizer
        self.trust_radius = trust_radius
        self.step_rule = build_step_rule(optimizer, self.alpha, beta, trust_radius)
        self.alpha_iter = 0

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        if draws is None:
            raise ValueError("EnOpt needs a source of random draws")
        self.cov = np.array(cov, dtype=float)
        if self.cov.shape != (self.xk.size, self.xk.size):
            raise ValueError(f"the covariance must be {(self.xk.size,) * 2}, got {self.cov.shape}")
        self.ensemble = GaussianEnsemble(
            self.evaluate_members,
            self.ne,
            draws,
            problem.lower,
            problem.upper,
            problem.n_realizations,
            natural_gradient=self.natural_gradient,
        )
        self.state_step = np.zeros_like(self.xk, dtype=float)
        self.cov_step = np.zeros_like(self.cov)
        self.step_rule = build_step_rule(
            self.optimizer_name, self.alpha, self.beta, self.trust_radius
        )

    def _evaluate_missing_derivatives(self) -> None:
        # The ensemble gradient needs the covariance; it is estimated inside every step.
        pass

    def evaluations_per_iteration(self, n_realizations: int) -> int:
        """Return the most evaluations of one iteration.

        Each of ``1 + resample`` attempts evaluates its ensemble and up to ``alpha_maxiter + 1``
        trial points, each on every realization.
        """
        trials = (self.alpha_iter_max + 1) * n_realizations
        return (1 + self.max_resample) * (self.ne + trials)

    def update_step(self) -> StepReport:
        """Run one EnOpt iteration with backtracking and resampling."""
        self.step_rule.restore_parameters()
        resampling = 0
        new_values = cast(FloatArray, self.fk)
        while resampling <= self.max_resample:
            shrink = self.cov_factor**resampling
            self.step_rule.apply_backtracking(np.sqrt(self.cov_factor) ** resampling)
            self.jk, self.hk = self._search_quantities(shrink)
            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state, new_step = self.step_rule.apply_update(
                    self.xk, self.jk, iteration=self.iteration, hessian=self.hk
                )
                new_state = self.bounds.project(new_state)
                new_values = self.fun(new_state)
                if np.mean(cast(FloatArray, self.fk)) - np.mean(new_values) > self.obj_func_tol:
                    self._accept(new_state, new_values, new_step, self.hk)
                    return StepReport(True)
                if self.alpha_iter < self.alpha_iter_max:
                    self.step_rule.apply_backtracking()
                    self.alpha_iter += 1
                else:
                    break
            if resampling < self.max_resample and np.mean(new_values) > np.mean(
                cast(FloatArray, self.fk)
            ):
                resampling += 1
                self.step_rule.restore_parameters()
                continue
            return StepReport(False, "EnOpt failed to find an improving step.")
        return StepReport(False, "EnOpt exhausted all resampling attempts.")

    def _search_quantities(self, shrink: float) -> tuple[FloatArray, FloatArray | None]:
        cov_step = self.beta * self.cov_step if self.nesterov else 0.0
        state_step = self.beta * self.state_step if self.nesterov else 0.0
        cov = shrink * (self.cov + cov_step)
        x_for_gradient = self.xk + state_step
        physical = self.bounds.to_physical(self.bounds.project(x_for_gradient))
        self.njev += 1
        gradient = self.bounds.gradient_to_internal(
            self.ensemble.gradient(physical, cov, cast(FloatArray, self.fk))
        )
        hessian: FloatArray | None = None
        if self.use_ensemble_hessian:
            self.nhev += 1
            hessian = self.bounds.hessian_to_internal(self.ensemble.hessian())
        if self.use_hessian and hessian is not None:
            # PET's direction H^-1 C C g. With H ~ C (hess f) C and g ~ C grad f it is the
            # Newton step (hess f)^-1 grad f only when C is proportional to I; it is kept
            # as PET's. A singular or non-finite H fails loudly rather than giving NaNs.
            try:
                gradient = np.linalg.solve(hessian, (self.cov @ self.cov) @ gradient)
            except np.linalg.LinAlgError as error:
                raise FloatingPointError(
                    "the ensemble Hessian is singular; use more members (ne) or hessian = false"
                ) from error
            if not np.all(np.isfinite(gradient)):
                raise FloatingPointError("the Hessian-preconditioned gradient is not finite")
        elif self.normalize:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, np.inf), 1e-12)
        if self.normalize and hessian is not None:
            hessian = hessian / np.maximum(np.linalg.norm(hessian, np.inf), 1e-12)
        return gradient, hessian

    def _accept(
        self,
        new_state: FloatArray,
        new_values: FloatArray,
        new_step: FloatArray,
        hessian: FloatArray | None,
    ) -> None:
        self._commit_step(new_state, new_values)
        self.state_step = new_step
        self.alpha = self.step_rule.current_step_size()
        if hessian is not None:
            # Covariance update with momentum (Zhang et al. 2023): C descends along H.
            self.cov_step = (
                self.alpha_cov * self.bounds.hessian_to_physical(hessian)
                + self.beta * self.cov_step
            )
            self.cov = floor_eigenvalues(self.cov - self.cov_step)
        if self.xk.size == 1:
            self.step_rule.halve_base_step_size()
        self.step_rule.restore_parameters()

    def log_columns(self) -> dict[str, float]:
        """Return the iteration, backtracking trials, objective, step size and ``C[0, 0]``."""
        return {
            **super().log_columns(),
            "alpha_iter": self.alpha_iter,
            "step_size": float(self.alpha),
            "cov00": float(self.cov[0, 0]),
        }

    def _export_extra(self) -> OptimizerState:
        state: OptimizerState = {
            "cov": self.cov,
            "state_step": self.state_step,
            "cov_step": self.cov_step,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
        }
        state.update({f"rule.{k}": v for k, v in self.step_rule.export_state().items()})
        return state

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self.cov = as_array(state["cov"])
        self.state_step = as_array(state["state_step"])
        self.cov_step = as_array(state["cov_step"])
        self.alpha = as_float(state["alpha"])
        self.alpha_iter = int(cast(int, state["alpha_iter"]))
        self.step_rule.restore_state(
            {k.removeprefix("rule."): v for k, v in state.items() if k.startswith("rule.")}
        )
