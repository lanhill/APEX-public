"""Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

Ported from PET
---------------
PET module : src/pipt/update_schemes/esmda.py (ESMDA.calc_analysis, ESMDA._ext_inflation_param)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Emerick, A. A. & Reynolds, A. C. (2013). Ensemble smoother with multiple
             data assimilation. Computers & Geosciences, 55, 3-15.
Differences: the forecast loop, data mismatch logging and restart live outside this class
             (workflows). The sum(1/alpha) == 1 check uses a relative tolerance of 1e-9
             instead of machine epsilon, which rejected valid schedules such as
             [9.333333333, 7, 4, 2]. Perturbations come from an explicit Generator.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    Estimate,
    ForecastRequest,
    Forecasts,
    Record,
    RunSetup,
    Status,
    Step,
    keep,
)
from apex.algorithms.data_assimilation.bounds import StateBounds
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.localization import Localization
from apex.algorithms.data_assimilation.protocols import AnalysisResult, UpdateStrategy
from apex.algorithms.data_assimilation.registries import methods
from apex.algorithms.data_assimilation.update_strategies import ApproxUpdate
from apex.core.diagnostics import Diagnostics, EmitsDiagnostics
from apex.core.progress import IterationProgress
from apex.domain.observations import DataKey

#: Relative tolerance on sum(1/alpha) == 1.
INFLATION_SUM_RTOL = 1e-9


@methods.register("esmda")
class ESMDA(EmitsDiagnostics):
    """ES-MDA with a fixed schedule of covariance inflation factors ``alpha_i``.

    Step ``i`` perturbs the observations with the inflated covariance and updates

    ``d_j = d_obs + sqrt(alpha_i) L z_j``,
    ``x_j <- x_j + C_xy (C_yy + alpha_i C_d)^-1 (d_j - y_j)``,

    where the gain is evaluated by ``strategy``. The factors must satisfy
    ``sum(1 / alpha_i) = 1`` so that the schedule is equivalent to a single update with
    ``C_d`` in the linear-Gaussian case.

    Parameters
    ----------
    inflation
        Inflation factors, one per step (PET ``mda.inflation_param``).
    strategy
        Update strategy; ``ApproxUpdate()`` by default (PET ``analysis = "approx"``).
    lower_bounds, upper_bounds
        Optional per-state bounds, shape ``(n_state,)``, applied to the posterior
        (PET ``prior_<name>.limits``). Their length is checked against ``n_state`` at the
        first analysis, when the state is known.

    Raises
    ------
    ValueError
        If the schedule is empty, has a non-positive factor, or violates the sum rule.
    """

    def __init__(
        self,
        inflation: Sequence[float],
        strategy: UpdateStrategy | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> None:
        alpha = tuple(float(a) for a in inflation)
        if not alpha:
            raise ValueError("inflation must contain at least one factor")
        if any(not math.isfinite(a) or a <= 0 for a in alpha):
            raise ValueError(f"inflation factors must be finite and positive, got {alpha}")
        inverse_sum = math.fsum(1.0 / a for a in alpha)
        if not math.isclose(inverse_sum, 1.0, rel_tol=INFLATION_SUM_RTOL):
            raise ValueError(f"sum(1/alpha) must equal 1, got {inverse_sum!r} for {alpha}")
        # These strategies add the prior-increment term of an iterative method (each member
        # pulled back towards its prior), which ES-MDA's steps do not have: there the prior
        # would count at every step and the data with weight 1/alpha only (PET runs them so;
        # on a linear-Gaussian case the posterior is 1.4-3x too wide, pet_parity.md).
        if getattr(strategy, "requires_levels", False) and not self.provides_levels:
            raise ValueError("the hybrid update needs fidelity levels: use method multilevel_esmda")
        if getattr(strategy, "requires_prior_state", False):
            raise ValueError(
                f"{type(self).__name__} with the {type(strategy).__name__} analysis is not "
                "supported: it includes the prior-increment term of an iterative method "
                "(each member pulled back towards its prior), and ES-MDA's steps have none, "
                "each step updating the current ensemble; use the approx strategy, or this "
                "one with gies, lm_enrml or gn_enrml"
            )
        # RLM-MAC centres data anomalies on the forecast at the mean state, which only GIES's
        # loop (and the EnRML methods built on it) provides; without this check ES-MDA would
        # forecast the whole prior and then fail at its first analysis. GN-EnRML, though on
        # that loop, refuses these strategies (damping 0 leaves them unregularised), so it is
        # not offered (release tester N9, 2026-10-20).
        if getattr(strategy, "requires_reference_prediction", False):
            raise ValueError(
                f"{type(self).__name__} with the {type(strategy).__name__} analysis is not "
                "supported: it needs the forecast at the mean state, which only the methods "
                "built on GIES's loop make; use the approx strategy, or this one with gies or "
                "lm_enrml"
            )
        self.inflation = alpha
        self.strategy: UpdateStrategy = ApproxUpdate() if strategy is None else strategy
        self.bounds = StateBounds(lower_bounds, upper_bounds)
        self._run: _EsmdaRun | None = None

    #: PET parity of the analysis step.
    status: Status = "parity"
    #: A fixed schedule of n_steps: cross validation does not apply.
    iterative: bool = False
    #: Whether the method gives each member a fidelity level (multilevel ES-MDA).
    provides_levels: ClassVar[bool] = False

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "step": "state increment of the analysis, before state bounds are applied, (n_state, N)"
    }

    @classmethod
    def uniform(
        cls,
        n_steps: int,
        strategy: UpdateStrategy | None = None,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
    ) -> "ESMDA":
        """Schedule with ``alpha_i = n_steps`` for every step (PET's default)."""
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        return cls([float(n_steps)] * n_steps, strategy, lower_bounds, upper_bounds)

    @property
    def lower_bounds(self) -> NDArray[np.float64] | None:
        """Lower state bounds ``(n_state,)``, if any."""
        return self.bounds.lower

    @property
    def upper_bounds(self) -> NDArray[np.float64] | None:
        """Upper state bounds ``(n_state,)``, if any."""
        return self.bounds.upper

    @property
    def n_steps(self) -> int:
        """Number of assimilation steps."""
        return len(self.inflation)

    def analysis(
        self,
        step_index: int,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        rng: np.random.Generator,
        *,
        localization_rng: np.random.Generator | None = None,
        sensitivity: NDArray[np.float64] | None = None,
    ) -> AnalysisResult:
        """Perturb the observations and run analysis step ``step_index``.

        Parameters
        ----------
        step_index
            0-based step in the schedule.
        state
            State ensemble that produced ``predicted``, ``(n_state, n_ensemble)``.
        predicted
            Predicted data, ``(n_data, n_ensemble)``.
        observations
            Observed data, ``(n_data,)``.
        data_error
            Un-inflated observation-error covariance ``C_d``.
        rng
            Source of the observation perturbations, redrawn at every step.
        localization_rng
            Source of a localization's randomness (the run's ``localization`` stream).

        Returns
        -------
        AnalysisResult
            Posterior state and the perturbed observations used.
        """
        inflated = self._inflated(step_index, data_error)
        perturbed = self._perturb(observations, inflated, predicted.shape[-1], rng)
        posterior = self._update(
            state, predicted, perturbed, inflated, localization_rng, sensitivity=sensitivity
        )
        return AnalysisResult(posterior, perturbed)

    def update(
        self,
        step_index: int,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed_observations: NDArray[np.float64],
        data_error: DataErrorCovariance,
        *,
        rng: np.random.Generator | None = None,
        sensitivity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Deterministic part of step ``step_index`` with given perturbed observations.

        Returns the posterior ``(n_state, n_ensemble)``. ``data_error`` is the
        un-inflated ``C_d``; the step's inflation is applied here. ``rng`` feeds a
        localization that draws (``adaptive``'s shuffle).
        """
        inflated = self._inflated(step_index, data_error)
        return self._update(
            state, predicted, perturbed_observations, inflated, rng, sensitivity=sensitivity
        )

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Set up the strategy's localization, if any, from the prior (e.g. its taper)."""
        localization = self._localization()
        if localization is not None:
            localization.prepare(prior_state, prior_predicted, keys, rng)

    def export_prepared(self) -> dict[str, NDArray[np.float64]]:
        """Return the localization's prepared arrays (empty without localization)."""
        localization = self._localization()
        return {} if localization is None else localization.export_state()

    def restore_prepared(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        """Restore the localization's prepared arrays on resume."""
        localization = self._localization()
        if localization is not None:
            localization.restore_state(arrays)

    def attach_diagnostics(self, sink: Diagnostics) -> None:
        """Use ``sink`` here and in the strategy and localization this method owns."""
        self._diagnostics = sink
        for part in (self.strategy, self._localization()):
            attach = getattr(part, "attach_diagnostics", None)
            if attach is not None:
                attach(sink)

    def _localization(self) -> Localization | None:
        localization: Localization | None = getattr(self.strategy, "localization", None)
        return localization

    def _inflated(self, step_index: int, data_error: DataErrorCovariance) -> DataErrorCovariance:
        if not 0 <= step_index < self.n_steps:
            raise IndexError(f"step_index must be in [0, {self.n_steps}), got {step_index}")
        return data_error.inflated(self.inflation[step_index])

    @staticmethod
    def _perturb(
        observations: NDArray[np.float64],
        inflated: DataErrorCovariance,
        n_ensemble: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        if observations.shape != (inflated.n_data,):
            raise ValueError(
                f"observations must have shape ({inflated.n_data},), got {observations.shape}"
            )
        return observations[:, None] + inflated.perturbations(rng, n_ensemble)

    @property
    def uses_sensitivities(self) -> bool:
        """Whether the strategy takes the simulator's sensitivities."""
        return bool(getattr(self.strategy, "uses_sensitivities", False))

    def _update(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed: NDArray[np.float64],
        inflated: DataErrorCovariance,
        rng: np.random.Generator | None = None,
        *,
        sensitivity: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        extra: dict[str, Any] = {"sensitivity": sensitivity} if self.uses_sensitivities else {}
        step = self.strategy.step(state, predicted, perturbed, inflated, rng=rng, **extra)
        self._diagnostics.add("method", "step", step)
        return self.bounds.apply(state + step)

    # --------------------------------------------------------- EnsembleAlgorithm
    #
    # The loop: forecast the prior; then, n_steps times, analyse (re-perturbing the
    # observations with the step's inflated C_d) and forecast the posterior. Data mismatches are
    # scored against one un-inflated perturbation of the data, drawn once (PET's
    # enObs_conv), so they are comparable across the inflated steps.

    def start(self, setup: RunSetup) -> Step:
        """Draw the scoring perturbations and request the prior forecast."""
        self.attach_diagnostics(setup.diagnostics)
        prior = keep(setup.prior)
        member_ids = np.arange(prior.shape[-1])
        scoring = setup.observations.values[:, None] + setup.data_error.perturbations(
            setup.streams["perturbations"], prior.shape[-1]
        )
        self._run = _EsmdaRun(setup, keep(scoring), pending=prior, pending_ids=member_ids)
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble", 0, prior, member_ids, with_sensitivities=self.uses_sensitivities
                )
            ],
            iteration=0,
            artifacts={
                "perturbed_observations": {
                    "perturbed_observations": scoring,
                    "member_ids": member_ids,
                }
            },
        )

    def advance(self, forecasts: Forecasts) -> Step:
        """Record the forecast step and, unless the schedule is done, analyse the next one."""
        run = self._require_run()
        forecast = forecasts["ensemble"]
        iteration = run.iteration
        run.state = keep(run.pending, forecast.ok)
        run.member_ids = run.pending_ids[forecast.ok]
        run.scoring = keep(run.scoring, forecast.ok)
        run.predicted = forecast.predicted
        extras = {}
        if run.pending_perturbed is not None:
            extras["perturbed_observations"] = keep(run.pending_perturbed, forecast.ok)
        record = Record(
            iteration,
            "iterate",
            run.state,
            run.predicted,
            run.member_ids,
            run.scoring,
            raw=forecast.raw,
            extras=extras,
        )
        if iteration == 0:
            # One-off set-up from the prior (e.g. a localization taper).
            self.prepare(
                run.state,
                run.predicted,
                run.setup.observations.keys,
                run.setup.streams["localization"],
            )
        if iteration == self.n_steps:
            # Marked, so that resuming a finished run does not forecast it again.
            run.done = True
            return Step(requests=[], iteration=iteration, records=[record])
        result = self.analysis(
            iteration,
            run.state,
            run.predicted,
            run.setup.observations.values,
            run.setup.data_error,
            run.setup.streams["analysis"],
            localization_rng=run.setup.streams["localization"],
            sensitivity=forecast.sensitivity,
        )
        run.pending = keep(result.posterior)
        run.pending_ids = run.member_ids
        run.pending_perturbed = keep(result.perturbed_observations)
        run.iteration = iteration + 1
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble",
                    run.iteration,
                    run.pending,
                    run.pending_ids,
                    with_sensitivities=self.uses_sensitivities,
                )
            ],
            iteration=run.iteration,
            records=[record],
        )

    def export_state(self) -> AlgorithmState:
        """Return the last recorded ensemble, the pending posterior and the schedule position."""
        run = self._require_run()
        arrays: dict[str, NDArray[Any]] = {
            "state": run.state,
            "predicted": run.predicted,
            "member_ids": run.member_ids,
            "scoring": run.scoring,
            "pending": run.pending,
            "pending_ids": run.pending_ids,
        }
        if run.pending_perturbed is not None:
            arrays["pending_perturbed"] = run.pending_perturbed
        arrays.update({f"method_state:{k}": v for k, v in self.export_prepared().items()})
        return AlgorithmState(arrays=arrays, values={"iteration": run.iteration, "done": run.done})

    def restore_state(self, setup: RunSetup, state: AlgorithmState) -> Step:
        """Continue from :meth:`export_state`: request the pending posterior's forecast."""
        self.attach_diagnostics(setup.diagnostics)
        a = state.arrays
        self._run = _EsmdaRun(
            setup,
            a["scoring"],
            pending=a["pending"],
            pending_ids=a["pending_ids"],
            pending_perturbed=a.get("pending_perturbed"),
            state=a["state"],
            predicted=a["predicted"],
            member_ids=a["member_ids"],
            iteration=int(state.values["iteration"]),
        )
        self._run.done = bool(state.values.get("done", False))
        prefix = "method_state:"
        self.restore_prepared({k[len(prefix) :]: v for k, v in a.items() if k.startswith(prefix)})
        run = self._run
        if run.done:
            return Step(requests=[], iteration=run.iteration)
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble",
                    run.iteration,
                    run.pending,
                    run.pending_ids,
                    with_sensitivities=self.uses_sensitivities,
                )
            ],
            iteration=run.iteration,
        )

    def restore_legacy(self, setup: RunSetup, legacy: Mapping[str, Any]) -> Step:
        """Continue a run of the old fixed-schedule driver.

        It checkpointed after the forecast of step ``iteration``; the next analysis had not
        run, so it runs now, drawing from the restored streams as the old driver would.
        """
        self.attach_diagnostics(setup.diagnostics)
        self._run = _EsmdaRun(
            setup,
            legacy["scoring"],
            pending=legacy["state"],
            pending_ids=legacy["member_ids"],
            state=legacy["state"],
            predicted=legacy["predicted"],
            member_ids=legacy["member_ids"],
            iteration=int(legacy["iteration"]),
        )
        self.restore_prepared(legacy["method_state"])
        if self.uses_sensitivities:
            raise ValueError("runs of the old driver cannot resume with the adjoint update")
        run = self._run
        if run.iteration == self.n_steps:
            return Step(requests=[], iteration=run.iteration)
        result = self.analysis(
            run.iteration,
            run.state,
            run.predicted,
            setup.observations.values,
            setup.data_error,
            setup.streams["analysis"],
            localization_rng=setup.streams["localization"],
        )
        run.pending = keep(result.posterior)
        run.pending_perturbed = keep(result.perturbed_observations)
        run.iteration += 1
        return Step(
            requests=[
                ForecastRequest(
                    "ensemble",
                    run.iteration,
                    run.pending,
                    run.pending_ids,
                    with_sensitivities=self.uses_sensitivities,
                )
            ],
            iteration=run.iteration,
        )

    def max_forecasts(self) -> tuple[int, int]:
        """Return the forecasts a run takes: ``(ensemble, single-member)``."""
        return self.n_steps + 1, 0

    def progress(self) -> IterationProgress:
        """Return the schedule position: analysis steps whose forecast is done."""
        run = self._run
        if run is None or run.iteration == 0:
            return IterationProgress(0, self.n_steps, "prior")
        if run.done:
            return IterationProgress(self.n_steps, self.n_steps, "done")
        # Step k is pending until its forecast has been recorded.
        return IterationProgress(run.iteration - 1, self.n_steps, f"step {run.iteration}")

    def estimate(self) -> Estimate:
        """Return the last forecast ensemble."""
        run = self._require_run()
        return Estimate(run.state, run.predicted, run.member_ids)

    def _require_run(self) -> "_EsmdaRun":
        if self._run is None:
            raise RuntimeError("ES-MDA has not started: call start() or restore_state() first")
        return self._run


class _EsmdaRun:
    """ES-MDA's loop state (see :meth:`ESMDA.export_state`)."""

    def __init__(
        self,
        setup: RunSetup,
        scoring: NDArray[np.float64],
        *,
        pending: NDArray[np.float64],
        pending_ids: NDArray[np.int64],
        pending_perturbed: NDArray[np.float64] | None = None,
        state: NDArray[np.float64] | None = None,
        predicted: NDArray[np.float64] | None = None,
        member_ids: NDArray[np.int64] | None = None,
        iteration: int = 0,
    ) -> None:
        self.setup = setup
        self.scoring = scoring
        #: The last posterior has been forecast: the schedule is complete.
        self.done = False
        self.pending, self.pending_ids, self.pending_perturbed = (
            pending,
            pending_ids,
            pending_perturbed,
        )
        empty = np.empty((0, 0))
        self.state = empty if state is None else state
        self.predicted = empty if predicted is None else predicted
        self.member_ids = np.empty(0, dtype=np.int64) if member_ids is None else member_ids
        self.iteration = iteration
