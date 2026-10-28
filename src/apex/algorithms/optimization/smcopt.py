"""SmcOpt: sequential Monte Carlo optimization.

Ported from PET
---------------
PET module : src/popt/optimization_methods/smcopt.py (SmcOpt), with
             src/popt/ensembles/ensemble_gaussian.py (GaussianEnsemble.calc_ensemble_weights)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Stordal, A. S. & Elsheikh, A. H. (2015). Optimization using sequential Monte
             Carlo with an application to well placement. ECMOR XIV, extended abstract.
Differences: none

One iteration::

    for resampling attempt k = 0 .. resample:
        draw particles around x_k with covariance cov_factor^k C (the step shrinks by
        sqrt(cov_factor)^k), weight all particles by exp(-(F - min F) * inflate),
        inflate = 2 (inflation_factor + iteration), and resample the survivors
        for backtracking trial j = 0 .. alpha_maxiter:
            x_trial = (1 - alpha) x_k + alpha * (weighted particle mean)
            accept if it improves the mean objective, or if the best particle improves the
            best objective seen, by more than tol; else halve alpha
        resample only if the last trial was worse than x_k

SmcOpt works in physical coordinates (PET forces ``transform = False``).
"""

from collections.abc import Mapping
from typing import ClassVar, cast

import numpy as np

from apex.algorithms.optimization.base import EpfOptions, OptimizerBase, StepReport, as_float
from apex.algorithms.optimization.enopt import objective_tolerances
from apex.algorithms.optimization.ensemble_gradient import GaussianEnsemble
from apex.algorithms.optimization.protocols import (
    Draws,
    FloatArray,
    OptimizerState,
    Problem,
    StateValue,
    Status,
)
from apex.algorithms.optimization.registries import optimizers
from apex.algorithms.optimization.step_rules import GradientDescent


@optimizers.register("smcopt")
class SmcOpt(OptimizerBase):
    """Sequential Monte Carlo optimization (SmcOpt): resampled ensembles towards the optimum.

    Parameters
    ----------
    ne
        Number of particles (PET's ``ne``; default 100).
    tol
        Improvement a trial needs; also ``ftol`` when given.
    alpha
        Weight of the particle mean in a step.
    alpha_maxiter
        Backtracking trials per resampling attempt.
    resample
        Resampling attempts when backtracking fails.
    cov_factor
        Covariance shrink factor per resampling attempt.
    inflation_factor
        Initial tempering of the weights (it grows with the iteration).
    survival_factor
        Fraction of particles drawn anew each iteration (clipped to ``[0.1, 1]``).
    best_func
        Best objective seen before this run (default: the initial objective).
    maxiter, ftol, xtol, gtol, epf
        See :class:`OptimizerBase`.
    """

    status: ClassVar[Status] = "parity"
    label: ClassVar[str] = "SmcOpt"
    needs_covariance: ClassVar[bool] = True

    def __init__(
        self,
        *,
        ne: int = 100,
        tol: float | None = None,
        alpha: float = 0.1,
        alpha_maxiter: int = 5,
        resample: int = 0,
        cov_factor: float = 0.5,
        inflation_factor: float = 1.0,
        survival_factor: float = 1.0,
        best_func: float | None = None,
        maxiter: int = 100,
        ftol: float | None = None,
        xtol: float = 1e-8,
        gtol: float = 1e-5,
        epf: EpfOptions = None,
    ) -> None:
        self.obj_func_tol, resolved_ftol = objective_tolerances(tol, ftol)
        super().__init__(maxiter=maxiter, ftol=resolved_ftol, xtol=xtol, gtol=gtol, epf=epf)
        if ne < 1:
            raise ValueError(f"ne (the ensemble size) must be at least 1, got {ne}")
        self.ne = ne
        self.alpha = alpha
        self.alpha_iter_max = alpha_maxiter
        self.max_resample = resample
        self.cov_factor = cov_factor
        self.inflation_factor = inflation_factor
        self.survival_factor = float(np.clip(survival_factor, 0.1, 1.0))
        self.initial_best_func = best_func
        self.alpha_iter = 0

    def _setup(self, problem: Problem, draws: Draws | None, cov: FloatArray | None) -> None:
        super()._setup(problem, draws, cov)
        if draws is None:
            raise ValueError("SmcOpt needs a source of random draws")
        if problem.n_realizations > 1:
            # Survivors do not carry their realization, so new particles would all be drawn
            # on the first blocks and the weights would compare different models.
            raise ValueError("SmcOpt does not support robust optimization (n_realizations > 1)")
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
        )
        self.step_rule = GradientDescent(self.alpha, 0.0)
        self.best_func = np.inf
        self.best_state: FloatArray | None = None
        self.sens_njev = 0

    def _start(self) -> None:
        if self.fk is None:
            self.fk = self.fun(self.xk)
        start = self.fk if self.initial_best_func is None else self.initial_best_func
        self.best_func = float(np.mean(start))
        super()._start()

    def _evaluate_missing_derivatives(self) -> None:
        pass

    iteration_evaluations: ClassVar[str] = (
        "per attempt, 1 + resample: ne ensemble members (one realization each), then up to "
        "alpha_maxiter + 1 line-search points, each on every realization"
    )

    def evaluations_per_iteration(self, n_realizations: int) -> int:
        """Return the most evaluations of one iteration.

        Each of ``1 + resample`` attempts evaluates its particles and up to ``alpha_maxiter + 1``
        trial points, each on every realization.
        """
        trials = (self.alpha_iter_max + 1) * n_realizations
        return (1 + self.max_resample) * (self.ne + trials)

    def update_step(self) -> StepReport:
        """Run one SmcOpt iteration with backtracking and resampling."""
        self.step_rule.restore_parameters()
        resampling = 0
        inflate = 2.0 * (self.inflation_factor + self.iteration)
        fk = cast(FloatArray, self.fk)
        new_values = fk
        while resampling <= self.max_resample:
            shrink = self.cov_factor**resampling
            self.step_rule.apply_backtracking(np.sqrt(self.cov_factor) ** resampling)
            target, self.best_state, best_trial = self.ensemble.smc_sensitivity(
                self.xk, inflate, shrink * self.cov, self.survival_factor
            )
            self.sens_njev += 1
            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state = self.bounds.project(self.step_rule.apply_smc_update(self.xk, target))
                new_values = self.fun(new_state)
                improved_objective = np.mean(fk) - np.mean(new_values) > self.obj_func_tol
                improved_best = (self.best_func - best_trial) > self.obj_func_tol
                if improved_objective or improved_best:
                    self._commit_step(new_state, new_values)
                    if improved_best:
                        self.best_func = float(best_trial)
                    self.step_rule.restore_parameters()
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
            return StepReport(False, "SmcOpt failed to find an improving step.")
        return StepReport(False, "SmcOpt exhausted all resampling attempts.")

    def result_extra(self) -> dict[str, float]:
        """Return the best objective seen and the number of particle updates."""
        return {"best_func": float(self.best_func), "sens_njev": float(self.sens_njev)}

    def log_columns(self) -> dict[str, float]:
        """Return the iteration, backtracking trials, objective and best objective seen."""
        return {
            **super().log_columns(),
            "alpha_iter": self.alpha_iter,
            "best_func": float(self.best_func),
        }

    def _export_extra(self) -> OptimizerState:
        state: OptimizerState = {
            "cov": self.cov,
            "best_func": self.best_func,
            "best_state": self.best_state,
            "sens_njev": self.sens_njev,
            "alpha_iter": self.alpha_iter,
        }
        state.update({f"rule.{k}": v for k, v in self.step_rule.export_state().items()})
        state.update({f"ens.{k}": v for k, v in self.ensemble.export_state().items()})
        return state

    def _restore_extra(self, state: Mapping[str, StateValue]) -> None:
        self.cov = np.array(state["cov"], dtype=float)
        self.best_func = as_float(state["best_func"])
        best = state.get("best_state")
        self.best_state = None if best is None else np.array(best, dtype=float)
        self.sens_njev = int(cast(int, state["sens_njev"]))
        self.alpha_iter = int(cast(int, state["alpha_iter"]))
        self.step_rule.restore_state(
            {k.removeprefix("rule."): v for k, v in state.items() if k.startswith("rule.")}
        )
        self.ensemble.restore_state(
            {k.removeprefix("ens."): v for k, v in state.items() if k.startswith("ens.")}
        )
