"""GenOpt: generalized ensemble optimization with an adapting mutation distribution.

Ported from PET
---------------
PET module : src/popt/optimization_methods/genopt.py (GenOpt), with
             src/popt/ensembles/ensemble_generalized.py and
             src/popt/optimization_methods/subroutines/cma.py
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Stordal, A. S., Szklarz, S. P. & Leeuwenburgh, O. (2016). A theoretical look at
             ensemble-based optimization in reservoir management. Mathematical Geosciences,
             48, 399-417. Hansen, N. (2006). The CMA evolution strategy: a comparing review.
             Springer.
Differences: one ensemble serves both gradients (PET draws and evaluates a second one for the
             mutation gradient; ``separate_mutation_ensemble = true`` restores that). The
             mutation gradient gets physical controls under ``transform`` (PET: unit-cube
             ones). The gradient's baseline is ``f(x_k)``. The correlation adapts by CMA only
             (PET's callable ``corr_adapt`` cannot come from a configuration). ``theta`` falls
             by at most half per step (PET can drive it to zero or below). Under EPF the
             members are penalized as the baseline is (PET's generalized ensemble evaluates
             its members without the penalty).

One iteration: as EnOpt's (backtracking, resampling with ``theta`` shrunk by
``cov_factor^k``), then on acceptance ``theta`` follows its own (normalised) gradient,
``theta <- theta + (beta dtheta_old - alpha_theta g_theta)``, and CMA adapts the copula's
correlation from the Gaussian samples and the step direction.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, ClassVar, Literal, cast

import numpy as np

from apex.algorithms.optimization.base import (
    EpfOptions,
    OptimizerBase,
    StepReport,
    as_array,
    as_float,
)
from apex.algorithms.optimization.cma import CMA
from apex.algorithms.optimization.enopt import build_step_rule, objective_tolerances
from apex.algorithms.optimization.generalized import GeneralizedEnsemble
from apex.algorithms.optimization.protocols import (
    Draws,
    FloatArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
)
from apex.algorithms.optimization.registries import optimizers
from apex.core.options import TEXT


@optimizers.register("genopt")
class GenOpt(OptimizerBase):
    """Generalized ensemble optimization (GenOpt): EnOpt with marginals and copulas, or CMA.

    Parameters
    ----------
    ne
        Ensemble size (default 100).
    marginal
        ``BetaMC`` (PET's default) or ``TruncGaussian``: the marginals with a ``theta``
        gradient.
    theta
        Initial marginal parameter per control (default from the initial variances).
    tol, alpha, step_size, beta, nesterov, alpha_maxiter, resample, normalize, cov_factor
        As for EnOpt; ``cov_factor`` shrinks ``theta`` per resampling attempt.
    alpha_theta
        Step size of ``theta``.
    optimizer
        Step rule: ``GD`` or ``Adam``.
    cma
        Adapt the copula's correlation by CMA.
    cma_n_mu, cma_alpha_mu, cma_alpha_1, cma_alpha_c, cma_equal_weights
        CMA's options (Hansen's defaults when unset); CMA adapts a correlation.
    separate_mutation_ensemble
        Draw a second ensemble for the mutation gradient, as PET does.
    maxiter, ftol, xtol, gtol, transform, epf
        See :class:`OptimizerBase`.
    """

    status: ClassVar[Status] = "parity"
    label: ClassVar[str] = "GenOpt"
    needs_covariance: ClassVar[bool] = True

    def __init__(
        self,
        *,
        ne: int = 100,
        marginal: Literal["BetaMC", "TruncGaussian"] = "BetaMC",
        theta: Sequence[float] | None = None,
        tol: float | None = None,
        alpha: float = 0.1,
        step_size: float | None = None,
        alpha_theta: float = 0.1,
        beta: float = 0.0,
        nesterov: bool = False,
        alpha_maxiter: int = 5,
        resample: int = 0,
        normalize: bool = True,
        cov_factor: float = 0.5,
        optimizer: Annotated[str, TEXT] = "GD",
        cma: bool = False,
        cma_n_mu: int | None = None,
        cma_alpha_mu: float | None = None,
        cma_alpha_1: float | None = None,
        cma_alpha_c: float | None = None,
        cma_equal_weights: bool = True,
        separate_mutation_ensemble: bool = False,
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
        if optimizer.lower() not in ("gd", "adam"):
            raise ValueError(f"GenOpt's step rule is GD or Adam, got {optimizer!r}")
        if ne < 1:
            raise ValueError(f"ne (the ensemble size) must be at least 1, got {ne}")
        self.ne = ne
        self.marginal = marginal
        self.initial_theta = None if theta is None else np.asarray(theta, dtype=float)
        self.alpha = alpha if step_size is None else step_size
        self.alpha_theta = alpha_theta
        self.beta = beta
        self.nesterov = nesterov
        self.alpha_iter_max = alpha_maxiter
        self.max_resample = resample
        self.normalize = normalize
        self.cov_factor = cov_factor
        self.optimizer_name = optimizer
        self.use_cma = cma
        self.cma_options = {
            "n_mu": cma_n_mu,
            "alpha_mu": cma_alpha_mu,
            "alpha_1": cma_alpha_1,
            "alpha_c": cma_alpha_c,
        }
        self.cma_equal_weights = cma_equal_weights
        self.separate_mutation_ensemble = separate_mutation_ensemble
        self.alpha_iter = 0

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        if draws is None:
            raise ValueError("GenOpt needs a source of random draws")
        cov = np.asarray(cov, dtype=float)
        self.ensemble = GeneralizedEnsemble(
            self.evaluate_members,
            problem.x0,
            cov,
            self.ne,
            draws,
            problem.lower,
            problem.upper,
            marginal=self.marginal,
            theta=self.initial_theta,
            n_realizations=problem.n_realizations,
        )
        self.theta = np.array(self.ensemble.theta, dtype=float)
        self.corr = np.array(self.ensemble.corr, dtype=float)
        self.state_step = np.zeros_like(self.xk, dtype=float)
        self.theta_step = np.zeros_like(self.theta)
        self.step_rule = build_step_rule(self.optimizer_name, self.alpha, self.beta, 1.0)
        self.cma: CMA | None = None
        if self.use_cma:
            self.cma = CMA(
                self.ne,
                self.xk.size,
                alpha_mu=self.cma_options["alpha_mu"],
                n_mu=cast(int | None, self.cma_options["n_mu"]),
                alpha_1=self.cma_options["alpha_1"],
                alpha_c=self.cma_options["alpha_c"],
                corr_update=True,
                equal_weights=self.cma_equal_weights,
            )

    def _evaluate_missing_derivatives(self) -> None:
        # Both gradients take theta and the correlation; they are estimated inside the step.
        pass

    iteration_evaluations: ClassVar[str] = (
        "per attempt, 1 + resample: ne ensemble members (twice with "
        "separate_mutation_ensemble; one realization each), then up to alpha_maxiter + 1 "
        "line-search points, each on every realization"
    )

    def evaluations_per_iteration(self, n_realizations: int) -> int:
        """Return the most evaluations of one iteration.

        Each of ``1 + resample`` attempts evaluates its ensemble (two with
        ``separate_mutation_ensemble``) and up to ``alpha_maxiter + 1`` trial points, each on
        every realization.
        """
        members = self.ne * (2 if self.separate_mutation_ensemble else 1)
        trials = (self.alpha_iter_max + 1) * n_realizations
        return (1 + self.max_resample) * (members + trials)

    def update_step(self) -> StepReport:
        """Run one GenOpt iteration: the controls by backtracking, then theta and correlation."""
        self.step_rule.restore_parameters()
        resampling = 0
        fk = cast(FloatArray, self.fk)
        new_values = fk
        while resampling <= self.max_resample:
            shrink = self.cov_factor**resampling
            self.jk, theta_gradient, mutation_ensemble = self._search_quantities(shrink)
            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state, new_step = self.step_rule.apply_update(
                    self.xk, self.jk, iteration=self.iteration, hessian=None
                )
                new_state = self.bounds.project(new_state)
                new_values = self.fun(new_state)
                if np.mean(fk) - np.mean(new_values) > self.obj_func_tol:
                    self._accept(new_state, new_values, new_step, theta_gradient, mutation_ensemble)
                    return StepReport(True)
                if self.alpha_iter < self.alpha_iter_max:
                    self.step_rule.apply_backtracking()
                    self.alpha_iter += 1
                else:
                    break
            if resampling < self.max_resample and np.mean(new_values) > np.mean(fk):
                resampling += 1
                self.step_rule.restore_parameters()
                continue
            return StepReport(False, "GenOpt failed to find an improving step.")
        return StepReport(False, "GenOpt exhausted all resampling attempts.")

    def _search_quantities(
        self, shrink: float
    ) -> tuple[FloatArray, FloatArray, Mapping[str, FloatArray]]:
        theta_step = self.beta * self.theta_step if self.nesterov else 0.0
        state_step = self.beta * self.state_step if self.nesterov else 0.0
        theta = shrink * (self.theta + theta_step)
        x_for_gradient = self.xk + state_step
        physical = self.bounds.to_physical(self.bounds.project(x_for_gradient))
        fk = cast(FloatArray, self.fk)
        self.njev += 1
        ensemble = self.ensemble.draw_and_evaluate(physical, theta, self.corr)
        gradient, _ = self.ensemble.gradient(physical, theta, self.corr, ensemble, fk)
        gradient = self.bounds.gradient_to_internal(gradient)
        mutation_ensemble = (
            self.ensemble.draw_and_evaluate(physical, theta, self.corr)
            if self.separate_mutation_ensemble
            else ensemble
        )
        theta_gradient, _ = self.ensemble.mutation_gradient(physical, theta, mutation_ensemble, fk)
        if self.normalize:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, np.inf), 1e-12)
            theta_gradient = theta_gradient / np.maximum(
                np.linalg.norm(theta_gradient, np.inf), 1e-12
            )
        return gradient, theta_gradient, mutation_ensemble

    def _accept(
        self,
        new_state: FloatArray,
        new_values: FloatArray,
        new_step: FloatArray,
        theta_gradient: FloatArray,
        mutation_ensemble: Mapping[str, FloatArray],
    ) -> None:
        self._commit_step(new_state, new_values)
        self.state_step = new_step
        self.alpha = self.step_rule.current_step_size()
        # theta is not backtracked: it follows its own gradient once the controls improved.
        self.theta_step = self.beta * self.theta_step - self.alpha_theta * theta_gradient
        # theta is a scale (sigma) or a concentration and must stay positive. PET has no
        # guard: one normalised step of alpha_theta can set sigma to 0 (then every trial fails)
        # or a concentration below 0 (NaN members). APEX lets theta fall to at most half its
        # value per step (docs/migration/pet_parity.md).
        new_theta = np.maximum(self.theta + self.theta_step, self.theta / 2)
        self.theta_step = new_theta - self.theta
        self.theta = new_theta
        if self.cma is not None:
            # step / alpha: the unit-length direction the evolution path wants.
            alpha = self.alpha if self.alpha else 1.0
            self.corr = self.cma(
                self.corr, new_step / alpha, mutation_ensemble["Z"], mutation_ensemble["F"]
            )
        if self.xk.size == 1:
            self.step_rule.halve_base_step_size()
        self.step_rule.restore_parameters()

    def log_columns(self) -> dict[str, float]:
        """Return the iteration, trials, objective, step size and the correlation's spread."""
        off_diagonal = self.corr - np.eye(self.corr.shape[0])
        return {
            **super().log_columns(),
            "alpha_iter": self.alpha_iter,
            "step_size": float(self.alpha),
            "max_corr": float(np.max(off_diagonal)),
            "min_corr": float(np.min(self.corr)),
        }

    def _export_extra(self) -> OptimizerState:
        state: OptimizerState = {
            "theta": self.theta,
            "corr": self.corr,
            "state_step": self.state_step,
            "theta_step": self.theta_step,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
        }
        state.update({f"rule.{k}": v for k, v in self.step_rule.export_state().items()})
        if self.cma is not None:
            state.update({f"cma.{k}": v for k, v in self.cma.export_state().items()})
        return state

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self.theta = as_array(state["theta"])
        self.corr = as_array(state["corr"])
        self.state_step = as_array(state["state_step"])
        self.theta_step = as_array(state["theta_step"])
        self.alpha = as_float(state["alpha"])
        self.alpha_iter = int(cast(int, state["alpha_iter"]))
        self.step_rule.restore_state(
            {k.removeprefix("rule."): v for k, v in state.items() if k.startswith("rule.")}
        )
        if self.cma is not None:
            self.cma.restore_state(
                {k.removeprefix("cma."): v for k, v in state.items() if k.startswith("cma.")}
            )
