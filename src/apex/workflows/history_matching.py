"""Ensemble history-matching workflow: one generic driver, checkpoints and resume.

:meth:`HistoryMatchingWorkflow.run` drives any
:class:`~apex.algorithms.data_assimilation.algorithm.EnsembleAlgorithm` (ES-MDA, GIES, and
new ones without changes here). The algorithm owns its loop and state; the workflow
forecasts what it requests (with the failure policy and data transforms), scores and saves
its records, and checkpoints it after every step marked ``checkpoint``.

Every checkpoint writes the algorithm's arrays, the history, then ``state.json`` last
(including the random streams), so :meth:`resume` continues a stopped or crashed run
exactly where it left off, bit-identically to an uninterrupted run. Every ensemble array
the loop holds is C-contiguous (:func:`keep`), which bit-identity needs on AVX-512 CPUs.
Run directories of the old drivers are converted on resume.

Saved results (``checkpoints/``): ``perturbed_observations.npz`` once (the draw iterative
methods use throughout, and the un-inflated draw fixed schedules score data mismatches against);
``iter_<k>.npz`` with the ensemble and its forecast for every step (fixed schedules, where
steps ``k >= 1`` also hold the perturbed observations that step's analysis drew, e.g.
ES-MDA's inflated ones, as PET's ``enObs``) or every accepted iterate (iterative methods,
``k`` the iteration that produced it; GIES adds its mean-model forecast); optionally
``trial_<k>.npz`` for rejected trials; ``data_mismatch_<k>.npz`` with per-member data mismatches and
metrics for every forecast; ``posterior.npz`` at the end. Bookkeeping for resume:
``algorithm_state_<n>.npz`` (the algorithm's state, the latest only) and ``state.json``.

**With a surrogate** (``[surrogate]``, :mod:`apex.workflows.surrogate_forecasts`)
every forecast the method asks for is the surrogate's, made on local workers; members
outside its validated domain stop the run (or are run by the simulator), and at the check
iterations some members are run by the simulator too and gated against the surrogate
(``check_<k>.npz``). Every decision of the loop (acceptance, damping, stopping, the
cross-validation rule) is made on the surrogate's forecasts, and each saved ensemble records
its members' ``forecast_source``. When the run completes, the posterior iterate is
forecast by the simulator (``full_<k>.npz``), and ``posterior.npz`` holds the simulator's
forecast. Without ``[surrogate]`` none of this runs: the loop is the one above.
"""

import dataclasses
import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.algorithm import (
    AlgorithmState,
    EnsembleAlgorithm,
    Forecast,
    ForecastRequest,
    Record,
    RunSetup,
    Step,
    keep,
)
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.surrogates.domain import check_members
from apex.algorithms.surrogates.gate import gate
from apex.core.diagnostics import DiagnosticsPlan, NoDiagnostics, RecordingDiagnostics
from apex.core.errors import ConfigError, WorkflowError
from apex.core.metrics import MetricContext, PerformanceMetric
from apex.core.progress import (
    GuardedProgress,
    IterationProgress,
    NullProgress,
    ProgressReporter,
)
from apex.core.simulation import (
    DataTransform,
    Executor,
    MemberFailure,
    MemberInput,
    Parameterization,
    Simulator,
)
from apex.core.surrogates import EarlyStartSurrogate
from apex.core.validation import StoppingRule
from apex.domain.data_split import DataSplit
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.results import HistoryMatchResult, IterationRecord
from apex.domain.state import StateLayout
from apex.domain.surrogate import GateResult
from apex.simulation.model_error import ModelErrorSimulator
from apex.simulation.sensitivity import SensitivityRun
from apex.storage.run_directory import SIMULATOR, SURROGATE, RunDirectory, upgrade_iterate
from apex.workflows.parameterization import PropertyMap, parameterized
from apex.workflows.surrogate_forecasts import (
    COARSE_LABEL,
    EARLY_LABEL,
    SURROGATE_LABEL,
    SurrogateCheckFailed,
    SurrogateForecasting,
    forecast_timing,
)

#: Deck runs tried, one member each, to check the deck's report dates before a surrogate
#: run with model error forecasts: a run that fails is followed by
#: another member's, up to this many.
DATE_CHECK_RUNS = 3

logger = logging.getLogger(__name__)

#: The name of the objective the methods minimise, ``(d_pert - y)^T C_d^-1 (d_pert - y)``
#: per member, as a metric (it was "misfit" before 2026-09-30).
DATA_MISMATCH = "data_mismatch"

#: Names of the independent random streams spawned from the run seed. New names are only
#: ever appended: spawned children depend on their position, so existing streams (and
#: results) do not change. ``surrogate``: the members a check runs by the
#: simulator, and refits.
STREAM_NAMES = ("prior", "perturbations", "analysis", "localization", "chop", "surrogate")

#: Member id used for the forecast of the ensemble-mean state.
MEAN_MODEL_ID = -1


@dataclass
class RandomStreams:
    """Independent generators for each random purpose, spawned from one seed.

    Separate streams keep, e.g., the prior draw unchanged when the method changes, and
    their states are checkpointed so a resumed run continues the same sequences.
    """

    generators: dict[str, np.random.Generator]

    @classmethod
    def from_seed(cls, seed: int) -> "RandomStreams":
        """Spawn one stream per name in :data:`STREAM_NAMES` from ``seed``."""
        children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
        return cls(
            {
                name: np.random.default_rng(child)
                for name, child in zip(STREAM_NAMES, children, strict=True)
            }
        )

    def __getitem__(self, name: str) -> np.random.Generator:
        """Return the stream called ``name``."""
        return self.generators[name]

    def state(self) -> dict[str, Any]:
        """Bit-generator states, JSON-serialisable."""
        return {name: gen.bit_generator.state for name, gen in self.generators.items()}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Set every stream back to a saved :meth:`state`.

        Streams added after the run was saved keep their freshly spawned state, which is
        deterministic (spawned children depend only on their index), so older runs resume.
        """
        for name, gen in self.generators.items():
            if name in state:
                gen.bit_generator.state = dict(state[name])


class HistoryMatchingWorkflow:
    """Runs a history-matching method against a simulator, with checkpoints.

    Parameters
    ----------
    layout
        State variables stacked into the ensemble rows.
    observations
        Observed data; predictions are requested at its keys, in its order.
    transforms
        Data transforms, fitted here to ``observations`` in order and applied
        to every forecast. Methods, localization and data mismatches see only the transformed
        data (:attr:`observations`); :attr:`raw_observations` keeps what was given.
    save_rejected_trials
        Iterative methods: also save rejected trial ensembles (``trial_<k>.npz``).
    save_raw_forecasts
        With transforms: also save the untransformed forecasts (``raw_predicted``).
    keep_simulations
        Retention of the simulator work directories (``[output] keep_simulations``):
        ``all``, ``failed``, ``last``, ``failed_and_last`` or ``none``.
    metrics
        Performance metrics evaluated at every forecast, by result name.
    truth
        The true model ``(n_state,)`` of a synthetic case study, for metrics that need it.
    diagnostics_plan
        Which diagnostics to save; default none.
    split
        Data roles: the algorithm sees only the history-matching data; the
        cross-validation (CV) and test data are forecast and scored at every record.
        Default: every datum is history-matching data.
    stopping
        Stopping rule on the CV normalized data mismatch of accepted iterates; required with CV
        data. The posterior is then the minimum-CV iterate.
    progress
        Receives the run's progress: iterations, forecasts, members, scores.
        Default: none shown. It never influences the results, and a reporter that raises
        is dropped with a warning (``progress_failed``) instead of stopping the run.
    simulator, executor
        Forward model and where its members run.
    run_dir
        Initialised run directory for checkpoints and simulator work directories.
    streams
        Random streams of the run.
    min_success_fraction
        Continue while at least this fraction of the current members simulate
        successfully, dropping the failed ones; otherwise raise :class:`WorkflowError`.
    auxiliary_variables
        State variables that are not reservoir properties, e.g. a model-error model's
        parameters: the quantity metrics leave them out unless named.
    surrogate
        A fitted surrogate makes the forecasts (``[surrogate]``); ``simulator``
        and ``executor`` then run only its checks, its fallbacks and the posterior's
        forecast. ``None`` (default): the simulator makes every forecast.

    A simulator with ``auxiliary_keys(keys)`` (a model-error simulator's residuals) is
    asked for those keys after the observations'; their values are saved with each saved
    ensemble (``auxiliary_predicted``, ``auxiliary_keys``) and never assimilated or scored.
    With a surrogate they are saved only where every member runs on
    the deck: the posterior's forecast by the simulator (``full_<k>``) and the iterates
    after a switch to the simulator (``on_check_failure = "simulator"``).
    """

    def __init__(
        self,
        *,
        layout: StateLayout,
        observations: ObservationSet,
        simulator: Simulator,
        executor: Executor,
        run_dir: RunDirectory,
        streams: RandomStreams,
        min_success_fraction: float = 0.9,
        transforms: Sequence[DataTransform] = (),
        save_rejected_trials: bool = False,
        keep_simulations: str = "all",
        save_raw_forecasts: bool = False,
        metrics: Mapping[str, PerformanceMetric] | None = None,
        truth: NDArray[np.float64] | None = None,
        diagnostics_plan: DiagnosticsPlan | None = None,
        split: DataSplit | None = None,
        stopping: StoppingRule | None = None,
        progress: ProgressReporter | None = None,
        parameterizations: Sequence[Parameterization] = (),
        auxiliary_variables: frozenset[str] = frozenset(),
        surrogate: SurrogateForecasting | None = None,
    ) -> None:
        if not 0 < min_success_fraction <= 1:
            raise ValueError(f"min_success_fraction must be in (0, 1], got {min_success_fraction}")
        self.layout = layout
        # Guarded: a failing display is dropped, never allowed to stop the run.
        self.progress: ProgressReporter = GuardedProgress(
            NullProgress() if progress is None else progress
        )
        self.raw_observations = observations
        self.transforms = tuple(transforms)
        for transform in self.transforms:
            observations = transform.fit(observations)
        self.split = split
        self._hm_rows, self._held_out = _roles(self.raw_observations, observations, split)
        self.analysis_observations = observations
        if self._hm_rows is not None:
            observations = observations.subset(self._hm_rows)
        self.observations = observations
        if split is not None and split.has_cv and stopping is None:
            raise ValueError("cross-validation data need a stopping rule")
        self.stopping = stopping if split is not None and split.has_cv else None
        # The rule is replayed from its initial state at every accepted iterate.
        self._rule_initial = {} if self.stopping is None else self.stopping.state()
        self._cv_stop_requested = self._cv_stopped = False
        # Held-out rows of each member forecast, by iteration, until its record is scored
        # (a record always follows the forecast of its own iteration).
        self._pending_held_out: dict[int, tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]]
        self._pending_held_out = {}
        # keys the simulator also answers (e.g. model-error residuals), saved with
        # each forecast's record but never assimilated.
        auxiliary_keys = getattr(simulator, "auxiliary_keys", None)
        self.auxiliary_keys: tuple[DataKey, ...] = (
            tuple(auxiliary_keys(self.raw_observations.keys)) if callable(auxiliary_keys) else ()
        )
        self._pending_auxiliary: dict[int, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
        self._last_auxiliary: NDArray[np.float64] | None = None
        unknown = sorted(auxiliary_variables - set(layout.names))
        if unknown:
            raise ValueError(f"auxiliary variables {', '.join(unknown)} are not state variables")
        self.auxiliary_variables = frozenset(auxiliary_variables)
        if self.transforms:
            logger.info(
                "data transformed",
                extra={
                    "event": "data_transformed",
                    "n_raw": self.raw_observations.n_data,
                    "n_data": observations.n_data,
                },
            )
        # the simulator, the metrics and the figures see the properties.
        self.property_map = PropertyMap(parameterizations, layout)
        self.simulator = parameterized(simulator, self.property_map)
        self.executor = executor
        self.run_dir = run_dir
        self.streams = streams
        self.min_success_fraction = min_success_fraction
        self.data_error = DataErrorCovariance.diagonal(observations.variances)
        self.save_rejected_trials = save_rejected_trials
        self.keep_simulations = keep_simulations
        # Without transforms the raw forecast is the forecast; nothing extra to save.
        self.save_raw_forecasts = save_raw_forecasts and bool(self.transforms)
        # an executor running members whole on other hosts may apply the
        # fitted transforms there. Each member then comes back transformed, followed by its
        # auxiliary rows and, when they are saved, its raw rows.
        use_transforms = getattr(executor, "use_transforms", None)
        # Asked even without transforms, so that an executor serving an earlier workflow
        # forgets its transforms.
        self._transformed_remotely = callable(use_transforms) and bool(
            use_transforms(
                self.transforms,
                n_in=self.raw_observations.n_data,
                n_out=self.analysis_observations.n_data,
                keep_raw=self.save_raw_forecasts,
            )
        )
        self._last_raw: NDArray[np.float64] | None = None
        self.surrogate = surrogate
        if surrogate is not None and self._transformed_remotely:
            # Composition refuses it first ("Refused"); kept here for other callers.
            raise ValueError(
                "a surrogate's forecasts stay on local workers: no executor transforming "
                "members remotely"
            )
        #: A surrogate run's model-error residual keys, saved with the posterior's forecast
        #: by the simulator only.
        self._surrogate_auxiliary_keys: tuple[DataKey, ...] = ()
        if surrogate is not None and self.auxiliary_keys:
            # With a surrogate, a model error's residuals are not saved
            # with the iterates (their members come from two sources, whose auxiliary rows
            # are not gathered); the forecasts g + r and p + r are unaffected.
            self._surrogate_auxiliary_keys, self.auxiliary_keys = self.auxiliary_keys, ()
        # A surrogate run's forecast sources of each iteration until its record is saved:
        # (member ids, their sources, the surrogate's identity, its predictive variance).
        self._pending_sources: dict[
            int,
            tuple[NDArray[np.int64], NDArray[np.str_], str, NDArray[np.float64] | None],
        ] = {}
        self._posterior_report: dict[str, Any] | None = None
        # The iteration whose check changed the forecasts' source (a switch or a refit).
        self._source_changed: int | None = None
        # A coarse-corrected surrogate's coarse runs of the last surrogate forecast, by member
        # id, taken at once by the caller (_surrogate_outputs) for the check: a
        # refit repeating the forecast, and its snapshot set, reuse them.
        self._last_coarse: dict[int, NDArray[np.float64] | MemberFailure] = {}
        rows: dict[str, list[int]] = {}
        for i, key in enumerate(observations.keys):
            rows.setdefault(key.datatype, []).append(i)
        self._type_rows = {name: np.array(r) for name, r in rows.items()}
        self.metrics = dict(metrics or {})
        self.diagnostics_plan = diagnostics_plan or DiagnosticsPlan({})
        self._diagnostics = (
            RecordingDiagnostics(self.diagnostics_plan) if self.diagnostics_plan.items else None
        )
        if truth is not None and np.shape(truth) != (layout.n_state,):
            raise ValueError(f"truth must be ({layout.n_state},), got {np.shape(truth)}")
        self._truth = None if truth is None else layout.split(np.asarray(truth)[:, None])
        needs_truth = sorted(k for k, m in self.metrics.items() if m.requires_truth)
        if needs_truth and truth is None:
            raise ValueError(f"metrics {', '.join(needs_truth)} need the true model")
        self._check_metrics()
        # Simulators that write files declare it; in-memory ones get no work directory.

    @property
    def held_out_observations(self) -> dict[str, ObservationSet]:
        """The cross-validation and test data, by role; empty without them."""
        return {role: observations for role, (_, observations) in self._held_out.items()}

    def role_rows(self) -> dict[str, NDArray[np.intp]]:
        """Rows of each role in :attr:`analysis_observations`.

        ``"history_matching"`` are the rows of ``predicted`` (every row without held-out
        data); ``"cv"`` and ``"test"`` those of ``cv_predicted`` and ``test_predicted``.
        """
        n = self.analysis_observations.n_data
        rows = {"history_matching": np.arange(n) if self._hm_rows is None else self._hm_rows}
        rows.update({role: indices for role, (indices, _) in self._held_out.items()})
        return rows

    # ------------------------------------------------------------------ generic driver

    def run(
        self,
        algorithm: EnsembleAlgorithm,
        prior: NDArray[np.float64] | None = None,
        *,
        max_steps: int | None = None,
    ) -> HistoryMatchResult:
        """Run (or continue) any :class:`EnsembleAlgorithm`.

        The algorithm owns its loop and state; this driver forecasts what it requests,
        applies the failure policy, scores and saves its records, and checkpoints it.

        Parameters
        ----------
        algorithm
            The algorithm.
        prior
            Prior ensemble ``(n_state, n_ensemble)`` for a new run; ``None`` to continue
            from the last checkpoint.
        max_steps
            Stop after this many further iterations (the run can be resumed later).
        """
        if prior is not None:
            prior = self._check_prior(prior)
            self._save_truth()
            self._save_split()
            step = algorithm.start(self._setup(prior))
            # The prior forecast is not counted: max_steps = 2 gives iterations 0, 1, 2.
            completed = 0
            sequence = 0
        else:
            saved = self._load()
            self._restore_validation(saved)
            self._restore_surrogate(saved)
            setup = self._setup(np.empty((self.layout.n_state, 0)))
            if saved["driver"] == "generic":
                arrays = self.run_dir.load_arrays(str(saved["algorithm_file"]))
                state = AlgorithmState(
                    arrays={k: keep(v) if v.ndim == 2 else v for k, v in arrays.items()},
                    values=saved["algorithm"],
                )
                step = algorithm.restore_state(setup, state)
                sequence = int(saved["sequence"])
            else:
                step = self._restore_legacy(algorithm, saved, setup)
                sequence = 0
                if self._diagnostics is not None:
                    # The old drivers checkpointed before the next update, so the restore
                    # has just run it; save its items under the iteration it produced.
                    self._diagnostics.iteration = step.iteration
                    self._save_diagnostics()
            completed = step.iteration - 1
        self._save_artifacts(step)
        self._announce_diagnostics()
        self.progress.run_started(
            algorithm=type(algorithm).__name__, progress=_progress_of(algorithm)
        )
        while step.requests and not self._cv_stopped:
            if step.checkpoint and max_steps is not None and step.iteration > completed + max_steps:
                # Recorded so that a report can say why the run stopped (#259); an event
                # only, so the run itself is unchanged.
                logger.info(
                    "step limit reached: stopping at this checkpoint",
                    extra={"event": "max_steps", "iteration": step.iteration,
                           "max_steps": max_steps},
                )  # fmt: skip
                break
            forecasts = {request.name: self._forecast_request(request) for request in step.requests}
            if self._source_changed is not None:
                # The trial is judged against its accepted iterate forecast by
                # the same source (after a switch to the simulator, or a refit).
                self._rescore_accepted(algorithm, self._source_changed)
                self._source_changed = None
            if self._diagnostics is not None:
                # Items offered during this step belong to the update it may produce.
                self._diagnostics.iteration = step.iteration + 1
            step = algorithm.advance(forecasts)
            self._save_artifacts(step)
            for record in step.records:
                self._save_record(record)
            position = _progress_of(algorithm)
            if position is not None:
                self.progress.iteration_progress(position)
            if step.checkpoint:
                self._save_diagnostics()
                sequence += 1
                self._checkpoint_generic(algorithm, step, sequence)
                # A CV stop takes effect only once a checkpoint records it, so that a
                # resume finishes instead of continuing (a stop decided on a record of a
                # non-checkpoint step waits for the next checkpoint).
                self._cv_stopped = self._cv_stop_requested
                if not self._cv_stopped and step.requests and self.run_dir.stop_requested():
                    # A graceful stop: after a checkpoint, as with max_steps, so
                    # a resume continues bit-identically.
                    logger.info(
                        "stop requested: stopping at this checkpoint",
                        extra={"event": "stopped_on_request", "iteration": step.iteration},
                    )
                    break
        complete = not step.requests or self._cv_stopped
        if complete:
            self.run_dir.clear_stop()  # a stop asked for during the last step: nothing to stop
        if self.stopping is not None and self.stopping.selected is not None:
            result = self._finish_selected(
                self.stopping.selected, converged=step.converged, complete=complete
            )
        else:
            estimate = algorithm.estimate()
            result = self._finish(
                estimate.state,
                estimate.predicted,
                estimate.member_ids,
                converged=step.converged,
                complete=complete,
                iteration=None if self.surrogate is None else self._last_accepted(),
            )
        self.progress.run_finished(status=run_status(result))
        return result

    def _restore_legacy(
        self, algorithm: EnsembleAlgorithm, saved: Mapping[str, Any], setup: RunSetup
    ) -> Step:
        """Continue a run saved by the old drivers.

        ``driver = "schedule"`` (ES-MDA) checkpointed the last forecast step; ``"iterative"``
        (GIES) the accepted iterate named by ``checkpoint``, with the members still in the
        run and the mean-model forecast in ``current_<k>`` (or, in older runs, all in one
        ``accepted.npz`` that also held the perturbed observations). The algorithm's
        ``restore_legacy`` continues from these exactly as the old driver would have.

        Raises
        ------
        WorkflowError
            If the algorithm cannot continue a run of that driver.
        """
        driver = str(saved["driver"])
        restore = getattr(algorithm, "restore_legacy", None)
        if restore is None:
            raise WorkflowError(f"{type(algorithm).__name__} cannot resume a {driver!r} run")
        legacy: dict[str, Any] = {"driver": driver}
        if driver == "schedule":
            arrays = self.run_dir.load_arrays(str(saved["checkpoint"]))
            member_ids = arrays["member_ids"]
            if "scoring_observations" in arrays:
                # Older runs kept the scoring perturbations in every step file; store them
                # once, as newer runs do, for the steps that follow.
                scoring = keep(arrays["scoring_observations"])
                self.run_dir.save_arrays(
                    "perturbed_observations",
                    {"perturbed_observations": scoring, "member_ids": member_ids},
                )
            else:
                scoring = self._perturbed_for(member_ids)
            legacy.update(
                state=keep(arrays["state"]),
                predicted=keep(arrays["predicted"]),
                member_ids=member_ids,
                scoring=scoring,
                iteration=int(saved["completed"]),
            )
        elif driver == "iterative":
            accepted = str(saved["checkpoint"])
            if "current" not in saved:
                current = self.run_dir.load_arrays(accepted)
                self.run_dir.save_arrays(
                    "perturbed_observations",
                    {
                        "perturbed_observations": current["perturbed_observations"],
                        "member_ids": current["member_ids"],
                    },
                )
            else:
                current = self.run_dir.load_arrays(str(saved["current"]))
            member_ids = current["member_ids"]
            arrays = self.run_dir.load_arrays(accepted)
            legacy.update(
                state=_members_of(arrays["state"], arrays["member_ids"], member_ids),
                predicted=_members_of(arrays["predicted"], arrays["member_ids"], member_ids),
                member_ids=member_ids,
                perturbed=self._perturbed_for(member_ids),
                reference_prediction=current.get("reference_prediction"),
                iteration=int(saved["iteration"]),
                damping=float(saved["damping"]),
                data_mismatch=_saved_data_mismatch(saved),
                converged=bool(saved["converged"]),
            )
        else:
            raise WorkflowError(f"unknown driver {driver!r} in state.json")
        method_state = self.run_dir.checkpoints / "method_state.npz"
        legacy["method_state"] = (
            self.run_dir.load_arrays("method_state") if method_state.exists() else {}
        )
        step: Step = restore(setup, legacy)
        return step

    def _setup(self, prior: NDArray[np.float64]) -> RunSetup:
        return RunSetup(
            prior=prior,
            observations=self.observations,
            data_error=self.data_error,
            streams=self.streams,
            diagnostics=self._diagnostics or NoDiagnostics(),
        )

    def _announce_diagnostics(self) -> None:
        """Log the diagnostics plan, warn about overridden [output] entries, save the plan.

        A plugin's own ``diagnostics`` table overrides ``[output] diagnostics``; the plan
        records, per item, which setting decided it.
        """
        plan = self.diagnostics_plan
        for warning in plan.warnings:
            logger.warning(warning, extra={"event": "diagnostics_override"})
        if not plan.items:
            return
        rows = plan.describe()
        listing = ", ".join(f"{row['item']} ({row['source']})" for row in rows)
        logger.info(
            "saving diagnostics: %s; plan in checkpoints/diagnostics_plan.json",
            listing,
            extra={"event": "diagnostics_plan", "items": rows},
        )
        (self.run_dir.checkpoints / "diagnostics_plan.json").write_text(
            json.dumps({"items": rows, "overrides": list(plan.warnings)}, indent=2) + "\n"
        )

    def _save_diagnostics(self) -> None:
        """Save the items collected during the last step (``diagnostics_<k>.npz``).

        ``k`` is the iteration the step's update produced. The file's ``index`` records,
        for every planned item offered, whether it was saved or skipped and why.
        """
        if self._diagnostics is None:
            return
        items, index = self._diagnostics.take()
        if items or index:
            self.run_dir.save_arrays(
                f"diagnostics_{self._diagnostics.iteration}",
                {**items, "index": np.array(json.dumps(index))},
            )

    def _forecast_request(self, request: ForecastRequest) -> Forecast:
        """Forecast one request; a ``required`` one (the mean model) must fully succeed."""
        if request.required and request.with_sensitivities:
            raise ValueError("a required (mean-model) forecast does not return sensitivities")
        if self.surrogate is not None and request.with_sensitivities:
            raise WorkflowError(
                "the adjoint update needs the simulator's gradients; it is refused with a surrogate"
            )
        if not request.required:
            ok, predicted, raw, sensitivity = self._forecast(
                request.state,
                request.member_ids,
                request.iteration,
                sensitivities=request.with_sensitivities,
                outputs=None if self.surrogate is None else self._surrogate_outputs(request),
            )
            if self._held_out:
                self._pending_held_out[request.iteration] = (
                    request.member_ids[ok],
                    {role: keep(predicted[rows]) for role, (rows, _) in self._held_out.items()},
                )
            if self._last_auxiliary is not None:
                self._pending_auxiliary[request.iteration] = (
                    request.member_ids[ok],
                    self._last_auxiliary,
                )
            pending = self._pending_sources.get(request.iteration)
            return Forecast(
                ok,
                self._assimilated(predicted),
                None if raw is None else keep(raw),
                None if sensitivity is None else self._assimilated(sensitivity),
                # the surrogate's variance, with variance_in_errors only.
                error_variance=None if pending is None else pending[3],
            )
        outputs = (
            self._run_members(
                request.state, request.member_ids, request.iteration, "mean model", auxiliary=False
            )
            if self.surrogate is None
            else self._surrogate_outputs(request)
        )
        values = [out for out in outputs if not isinstance(out, MemberFailure)]
        failed = [out for out in outputs if isinstance(out, MemberFailure)]
        if failed:
            what = "mean-model" if request.member_ids[0] == MEAN_MODEL_ID else request.name
            raise WorkflowError(
                f"iteration {request.iteration}: {what} forecast failed: {failed[0].message}"
            )
        predicted = np.column_stack(values)
        if not self._transformed_remotely:
            predicted = self._transform(predicted)
        return Forecast(np.ones(len(outputs), dtype=bool), self._assimilated(predicted))

    def _assimilated(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the history-matching rows of a forecast in the analysis data space."""
        return keep(predicted if self._hm_rows is None else predicted[self._hm_rows])

    def _save_artifacts(self, step: Step) -> None:
        for name, arrays in step.artifacts.items():
            self.run_dir.save_arrays(name, arrays)

    def _save_record(self, record: Record) -> None:
        """Save a record's ensemble (``iter_<k>`` or, if kept, ``trial_<k>``) and score it.

        With cross validation, its CV and test predictions are saved with it
        (``cv_predicted``, ``test_predicted``) and scored as the metrics
        ``cv_normalized_data_mismatch`` and ``test_normalized_data_mismatch``; accepted
        iterates feed the
        stopping rule.
        """
        held_out = self._held_out_of(record)
        auxiliary = self._pending_auxiliary.pop(record.iteration, None)
        sources = self._sources_of(record)
        if record.kind == "iterate" or self.save_rejected_trials:
            arrays: dict[str, NDArray[Any]] = {
                "state": record.state,
                "predicted": record.predicted,
                "member_ids": record.member_ids,
            }
            if record.raw is not None:
                arrays["raw_predicted"] = record.raw
            arrays.update(record.extras)
            arrays.update({f"{role}_predicted": values for role, values in held_out.items()})
            if auxiliary is not None:
                ids, values = auxiliary
                arrays["auxiliary_predicted"] = keep(values[:, _columns_of(ids, record.member_ids)])
                names = self.auxiliary_keys or self._surrogate_auxiliary_keys
                arrays["auxiliary_keys"] = np.array([str(k) for k in names])
            if sources is not None:
                arrays.update(sources)
            prefix = "iter" if record.kind == "iterate" else "trial"
            self.run_dir.save_arrays(f"{prefix}_{record.iteration}", arrays)
        scores = {
            f"{role}_normalized_data_mismatch": _mean_square_by_type(
                self._held_out[role][1], values
            )
            for role, values in held_out.items()
        }
        self._record(
            record.iteration,
            record.state,
            record.predicted,
            record.scoring,
            record.member_ids,
            accepted=record.accepted,
            damping=record.damping,
            held_out_scores=scores,
            forecast_source=None if sources is None else _source_of(sources["forecast_source"]),
        )
        stops = (
            self.stopping is not None
            and record.kind == "iterate"
            and record.accepted
            and self._replay_stopping(record.member_ids)
        )
        if stops and not self._cv_stop_requested:
            assert self.stopping is not None
            self._cv_stop_requested = True
            logger.info(
                "cross validation stops the run",
                extra={
                    "event": "cv_stop",
                    "iteration": record.iteration,
                    "selected_iteration": self.stopping.selected,
                },
            )

    def _replay_stopping(self, member_ids: NDArray[np.int64]) -> bool:
        """Feed the stopping rule every accepted iterate so far, scored on ``member_ids``.

        Members only ever leave a run (failed forecasts). A mean over the survivors of
        each iterate compares like with like, as the methods' own acceptance tests do. A
        mean over each iterate's own members would let a dropped member with a large (or
        small) CV mismatch fake (or hide) a minimum. The per-member values are in
        ``data_mismatch_<k>.npz``, so replaying from them is deterministic and resume-safe.
        Returns whether the rule stops after the latest iterate.
        """
        assert self.stopping is not None
        self.stopping.restore(self._rule_initial)
        stop = False
        # after the forecasts' source changed (a surrogate run's switch or
        # refit), the rule compares like with like: it starts again from the accepted
        # iterate forecast again by the new source, and takes the iterates after it.
        latest = self.surrogate.rescores[-1] if self.surrogate and self.surrogate.rescores else None
        since = -1
        if latest is not None:
            since = int(latest["accepted"])
            saved = self.run_dir.load_arrays(f"rescored_{latest['iteration']}")
            columns = _columns_of(saved["member_ids"], member_ids)
            value = float(saved["metric:cv_normalized_data_mismatch:all"][columns].mean())
            stop = self.stopping.observe(since, value)
        for entry in self.run_dir.history_dicts():
            if not entry["accepted"] or int(entry["iteration"]) <= since:
                continue
            iteration = int(entry["iteration"])
            saved = self.run_dir.load_arrays(f"data_mismatch_{iteration}")
            columns = _columns_of(saved["member_ids"], member_ids)
            value = float(saved["metric:cv_normalized_data_mismatch:all"][columns].mean())
            stop = self.stopping.observe(iteration, value)
        return stop

    def _held_out_of(self, record: Record) -> dict[str, NDArray[np.float64]]:
        """Return the CV and test predictions of a record's members, from its forecast."""
        if not self._held_out:
            return {}
        try:
            ids, values = self._pending_held_out.pop(record.iteration)
        except KeyError:
            raise WorkflowError(
                f"no forecast of iteration {record.iteration} for its record (the algorithm "
                "must emit a record after the forecast of its own iteration)"
            ) from None
        columns = _columns_of(ids, record.member_ids)
        return {role: keep(v[:, columns]) for role, v in values.items()}

    def _checkpoint_generic(self, algorithm: EnsembleAlgorithm, step: Step, sequence: int) -> None:
        """Checkpoint the algorithm after a step: its arrays, then ``state.json`` last.

        The arrays go to ``algorithm_state_<sequence>.npz`` under a new name each time,
        and ``state.json`` names the one that belongs to it, so a crash between the two
        writes leaves the previous consistent pair; superseded files are then deleted.
        """
        state = algorithm.export_state()
        name = f"algorithm_state_{sequence}"
        self.run_dir.save_arrays(name, state.arrays)
        self._save_state(
            {
                "driver": "generic",
                "sequence": sequence,
                "iteration": step.iteration,
                "algorithm_file": name,
                "algorithm": dict(state.values),
                "converged": step.converged,
                **self._validation_state(),
                **({} if self.surrogate is None else {"surrogate": self.surrogate.state()}),
            }
        )
        for path in self.run_dir.checkpoints.glob("algorithm_state_*.npz"):
            if path.stem != name:
                path.unlink()

    # ------------------------------------------------------------------ surrogate

    def _surrogate_inputs(self, state: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Return the surrogate's inputs of an ensemble: its properties, C-contiguous.

        A model-error model's parameters are not inputs: the surrogate stands for the deck
        alone.
        """
        blocks = self.layout.split(state)
        surrogate = self.surrogate
        if surrogate is not None and surrogate.model_error is not None:
            blocks = {k: v for k, v in blocks.items() if k != surrogate.model_error.state_variable}
        properties = blocks if self.property_map.identity else self.property_map.ensemble(blocks)
        return {
            name: np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            for name, value in properties.items()
        }

    def _surrogate_outputs(
        self, request: ForecastRequest
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Forecast a request with the surrogate; check it at the check iterations.

        Returns each member's raw forecast (or its failure, only for members the simulator
        ran), in member order. The sources of an ensemble forecast wait for its record.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        label = "mean model" if request.required else "ensemble"
        state, ids, iteration = request.state, request.member_ids, request.iteration
        if not surrogate.dates_verified and surrogate.switched_at is None:
            self._verify_dates(state, ids, iteration)
        # No residuals left from an earlier forecast's runs.
        self._last_auxiliary = None
        if surrogate.switched_at is not None:  # on_check_failure = "simulator" happened
            # Every member on the deck now: its model error's residuals are saved again. (With
            # them, an executor transforming remotely sends the raw rows back too: auxiliary
            # skips without_raw_rows.)
            residuals = self._surrogate_auxiliary_keys if not request.required else ()
            outputs = self._run_members(
                state, ids, iteration, label, auxiliary=bool(residuals),
                auxiliary_keys=residuals or None,
            )  # fmt: skip
            surrogate.simulator_runs += len(ids)
            sources = [SIMULATOR] * len(ids)
        else:
            outputs, sources = self._surrogate_members(state, ids, iteration, label)
            # This forecast's coarse runs, by member, for a refit of its check.
            made = dict(self._last_coarse)
            if not request.required and surrogate.checks_at(iteration):
                outputs, sources = self._check_surrogate(
                    state, ids, iteration, outputs, sources, label, coarse=made
                )
        if not request.required:
            variance = None
            if surrogate.section.variance_in_errors and SURROGATE in sources:
                variance = self._surrogate_variance(state, ids, sources)
            self._pending_sources[iteration] = (
                np.array(ids, dtype=np.int64),
                np.array(sources),
                surrogate.active.identity if SURROGATE in sources else "",
                variance,
            )
        return outputs

    def _verify_dates(
        self, state: NDArray[np.float64], member_ids: NDArray[np.int64], iteration: int
    ) -> None:
        """Run a member on the deck to check its report dates.

        The model error of the surrogate's members is read at the deck's report dates taken
        from its text; before any is forecast, a deck run's own dates are checked against
        them (its model-error simulator raises ``ConfigError`` when they differ, or when
        the run reports fewer steps). The run's forecast is discarded: the run's results are
        those of a run without it. A member whose run fails is followed by the next, up to
        :data:`DATE_CHECK_RUNS`; none succeeding refuses the run. The members are the first
        forecast's: one that asks for fewer (a method whose first forecast is a single
        mean model) gets that many tries.

        Raises
        ------
        ConfigError
            If the dates differ, or no deck run succeeds to check them.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        failures: list[str] = []
        for column in range(min(DATE_CHECK_RUNS, len(member_ids))):
            out = self._run_members(
                keep(state[:, [column]]), member_ids[[column]], iteration,
                "date check (simulator)", auxiliary=False, folder="check_dates",
            )[0]  # fmt: skip
            surrogate.simulator_runs += 1
            if not isinstance(out, MemberFailure):
                surrogate.dates_verified = True
                logger.info(
                    "the deck's report dates checked",
                    extra={"event": "surrogate_dates_checked", "iteration": iteration,
                           "member_id": int(member_ids[column])},
                )  # fmt: skip
                return
            failures.append(f"member {out.member_id}: {out.message}")
        raise ConfigError(
            "[surrogate] with [model_error]: no deck run succeeded to check the deck's report "
            f"dates before the surrogate forecasts ({'; '.join(failures)})"
        )

    def _surrogate_members(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        label: str,
        *,
        folder: str | None = None,
        simulated: Mapping[int, NDArray[np.float64] | MemberFailure] | None = None,
        coarse: Mapping[int, NDArray[np.float64] | MemberFailure] | None = None,
    ) -> tuple[list[NDArray[np.float64] | MemberFailure], list[str]]:
        """Predict the members inside the domain; the others stop the run or are simulated.

        ``folder``: where the simulator's runs go when not ``iter_<iteration>`` (an accepted
        iterate's forecast again). ``simulated``: the members' simulator outputs made
        already in this forecast, by column (a refit repeating it): a member outside reuses
        its own instead of being run again. ``coarse``: for a coarse-corrected model, the
        members' coarse runs made already in this forecast, by member id (a refit repeating
        it); the others are run first (:meth:`_coarse_members`). A member whose coarse run
        fails is a failed member, as a simulator's failure is (counted against
        ``min_success_fraction``).

        Raises
        ------
        SurrogateDomainError
            With ``outside = "stop"``, if a member lies outside the surrogate's domain; and
            always, if a key or time does.
        WorkflowError
            If the surrogate fails a member (never a dropped member), or its evaluations
            would exceed ``max_evaluations``.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        active = surrogate.active
        outputs: list[NDArray[np.float64] | MemberFailure | None] = [None] * len(member_ids)
        coarse_runs: dict[int, NDArray[np.float64]] = {}
        lost: list[int] = []  # columns whose coarse or early run failed
        # A coarse-corrected model's coarse runs, or an early-start model's early runs (ADR
        # 0073 phase 5): the companion data each prediction reads, made first.
        companion = surrogate.coarse is not None or surrogate.early_until is not None
        if companion:
            made = self._coarse_members(state, member_ids, iteration, label, folder, coarse)
            self._last_coarse = made
            for column, member in enumerate(member_ids):
                out = made[int(member)]
                if isinstance(out, MemberFailure):
                    outputs[column] = out
                    lost.append(column)
                else:
                    coarse_runs[int(member)] = out
        kept = [j for j in range(len(member_ids)) if j not in set(lost)]
        found = check_members(
            active.domain,
            active.model,
            self._surrogate_inputs(keep(state[:, kept])),
            [int(member_ids[j]) for j in kept],
            self.raw_observations.keys,
            coarse=None
            if not companion
            else np.ascontiguousarray(
                np.column_stack([coarse_runs[int(member_ids[j])] for j in kept])
            )
            if kept
            else np.zeros((len(self._companion_keys()), 0)),
        )
        stops = found.unknown_keys or found.times_outside or surrogate.section.outside == "stop"
        if not found.inside:
            logger.log(
                logging.ERROR if stops else logging.WARNING,
                "members outside the surrogate's domain",
                extra={"event": "surrogate_outside", "iteration": iteration, "label": label,
                       "members": [int(m) for m in found.members],
                       "policy": "stop" if stops else "simulator", "details": found.message()},
            )  # fmt: skip
            if stops:
                found.raise_if_outside()
        outside = {int(m) for m in found.members}
        inside = [j for j in kept if int(member_ids[j]) not in outside]
        rest = [j for j in kept if int(member_ids[j]) in outside]
        if inside:
            surrogate.charge(len(inside), iteration)
            predicted = self._run_surrogate(
                keep(state[:, inside]), member_ids[inside], iteration, label,
                coarse={int(member_ids[j]): coarse_runs[int(member_ids[j])] for j in inside}
                if companion
                else None,
            )  # fmt: skip
            for column, out in zip(inside, predicted, strict=True):
                if isinstance(out, MemberFailure):
                    # Loud: dropping members the surrogate cannot predict would bias the
                    # posterior silently.
                    logger.error(
                        "the surrogate failed a member",
                        extra={"event": "surrogate_failed", "iteration": iteration,
                               "member_id": out.member_id, "reason": out.message},
                    )  # fmt: skip
                    raise WorkflowError(
                        f"iteration {iteration}: the surrogate failed member {out.member_id} "
                        f"({out.message}); the run stops at its last checkpoint"
                    )
                outputs[column] = out
        known = dict(simulated or {})
        reused = [j for j in rest if j in known]
        for column in reused:
            outputs[column] = known[column]
        to_run = [j for j in rest if j not in known]
        if to_run:
            outcome = self._run_members(
                keep(state[:, to_run]), member_ids[to_run], iteration, f"{label} (outside)",
                auxiliary=False, folder=folder,
            )  # fmt: skip
            surrogate.simulator_runs += len(to_run)
            for column, out in zip(to_run, outcome, strict=True):
                outputs[column] = out
        logger.info(
            "surrogate forecast",
            extra={"event": "surrogate_forecast", "iteration": iteration, "label": label,
                   "surrogate": len(inside), "simulator": len(rest), "reused": len(reused),
                   "coarse_failed": len(lost), "identity": active.identity,
                   "evaluations": surrogate.evaluations},
        )  # fmt: skip
        # A member whose coarse or early run failed is the simulator's failure, never the
        # surrogate's.
        simulated_columns = set(rest) | set(lost)
        sources = [
            SIMULATOR if j in simulated_columns else SURROGATE for j in range(len(member_ids))
        ]
        done = [out for out in outputs if out is not None]
        assert len(done) == len(member_ids)  # every member is predicted or simulated
        return done, sources

    def _coarse_members(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        label: str,
        folder: str | None,
        known: Mapping[int, NDArray[np.float64] | MemberFailure] | None,
    ) -> dict[int, NDArray[np.float64] | MemberFailure]:
        """Run the members' companion runs: coarse runs, or early runs up to a date.

        A coarse-corrected model's coarse stack, or the simulator up to an
        early-start model's early date. Through the run's executor, as
        the simulator's runs (its pools, licences, retries and ``keep_simulations``), into
        ``simulations/coarse_<iteration>/`` (or ``<folder>_coarse``) at the model's keys, or
        ``early_<iteration>/`` (``<folder>_early``) at its early keys, each member stopped
        at the early date (``MemberInput.stop_at``). ``known``: runs made already in this
        forecast, by member id. Returns every member's run or failure by member id.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        early = surrogate.early_until
        assert (surrogate.coarse is None) != (early is None)
        kind = "coarse" if early is None else "early"
        made = {int(m): known[int(m)] for m in member_ids if known and int(m) in known}
        todo = [j for j, m in enumerate(member_ids) if int(m) not in made]
        if todo:
            outcome = self._run_members(
                keep(state[:, todo]), member_ids[todo], iteration,
                f"{label} {COARSE_LABEL if early is None else EARLY_LABEL}",
                surrogate.coarse, auxiliary=False, keys=self._companion_keys(),
                folder=f"{kind}_{iteration}" if folder is None else f"{folder}_{kind}",
                stop_at=early,
            )  # fmt: skip
            if early is None:
                surrogate.coarse_runs += len(todo)
            else:
                surrogate.early_runs += len(todo)
            for column, out in zip(todo, outcome, strict=True):
                made[int(member_ids[column])] = out
            failed = [int(o.member_id) for o in outcome if isinstance(o, MemberFailure)]
            if failed:
                logger.warning(
                    f"{kind} runs failed",
                    extra={"event": f"surrogate_{kind}_failed", "iteration": iteration,
                           "label": label, "members": failed},
                )  # fmt: skip
        return made

    def _companion_keys(self) -> tuple[DataKey, ...]:
        """Return the companion runs' keys: an early-start model's early keys, else its keys."""
        surrogate = self.surrogate
        assert surrogate is not None
        model = surrogate.active.model
        return tuple(model.early_keys() if isinstance(model, EarlyStartSurrogate) else model.keys)

    def _run_surrogate(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        label: str,
        coarse: Mapping[int, NDArray[np.float64]] | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run the active surrogate on the members, on its local executor.

        ``coarse``: a coarse-corrected model's coarse forecasts, or an early-start model's
        early runs, by member id.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        model: Simulator = surrogate.active.simulator
        if coarse is not None:
            model = (
                surrogate.active.simulator.with_coarse(coarse)
                if surrogate.early_until is None
                else surrogate.active.simulator.with_early(coarse)
            )
        if surrogate.model_error is not None:
            # P(m) + r(m, theta), r from the surrogate's block states of
            # the same member and prediction (never a deck run's files).
            model = ModelErrorSimulator(
                model, surrogate.model_error, step_dates=surrogate.step_dates
            )
        blocks = self.layout.split(state)
        members = [
            MemberInput(
                member_id=int(member_id),
                state={name: block[:, column].copy() for name, block in blocks.items()},
            )
            for column, member_id in enumerate(member_ids)
        ]
        self.progress.forecast_started(
            iteration=iteration, label=f"surrogate {label}", n_members=len(members)
        )
        began = time.monotonic()
        outputs = surrogate.executor.run_all(
            parameterized(model, self.property_map),
            members,
            self.raw_observations.keys,
            on_result=lambda member_id, ok: self.progress.member_finished(
                member_id=member_id, ok=ok
            ),
        )
        seconds = time.monotonic() - began
        n_failed = sum(isinstance(out, MemberFailure) for out in outputs)
        self.progress.forecast_finished(
            n_ok=len(outputs) - n_failed, n_failed=n_failed, seconds=seconds
        )
        logger.info(
            "forecast done",
            extra={"event": "forecast_done", "iteration": iteration, "label": SURROGATE_LABEL,
                   "n_ok": len(outputs) - n_failed, "n_failed": n_failed,
                   # Unrounded: a fast surrogate's forecast takes well under a millisecond.
                   "seconds": seconds,
                   # Members at once (local executors: serial is one), for its cost (§14).
                   "workers": min(int(getattr(surrogate.executor, "workers", 1)), len(members))},
        )  # fmt: skip
        return outputs

    def _check_keys(self) -> tuple[DataKey, ...]:
        """Keys a check runs: the observations', and the model's own others for a refit."""
        surrogate = self.surrogate
        assert surrogate is not None
        keys = tuple(self.raw_observations.keys)
        if surrogate.section.on_check_failure != "refit":
            return keys
        known = set(keys)
        return keys + tuple(k for k in surrogate.active.model.keys if k not in known)

    def _gate_roles(
        self,
        surrogate_raw: NDArray[np.float64],
        simulator_raw: NDArray[np.float64],
        member_ids: Sequence[int],
    ) -> dict[str, GateResult]:
        """Gate the surrogate against the simulator per role, in the analysis space (§3).

        ``*_raw`` are ``(n_raw, n)``; both are transformed as every forecast is, and gated
        against the observation errors there: the assimilated data, and with
        ``[validation]`` the cv data apart.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        transformed = self._transform(keep(surrogate_raw))
        reference = self._transform(keep(simulator_raw))
        analysis = self.analysis_observations
        rows = self.role_rows()
        verdicts: dict[str, GateResult] = {}
        model = surrogate.active.model
        # An early-start model's early keys are the members' own runs: exact, so they would
        # dilute the error of its continuation. Keys a data transform
        # made are not the raw keys, and are all gated.
        early = set(model.early_keys()) if isinstance(model, EarlyStartSurrogate) else set()
        for role in ("history_matching", "cv"):
            if role not in rows or not len(rows[role]):
                continue
            chosen = rows[role]
            if early:
                chosen = np.array(
                    [i for i in chosen if analysis.keys[int(i)] not in early], dtype=np.int64
                )
                if not len(chosen):
                    continue
            verdicts[role] = gate(
                keep(transformed[chosen]),
                keep(reference[chosen]),
                [analysis.keys[int(i)] for i in chosen],
                [int(m) for m in member_ids],
                tolerances=surrogate.tolerances,
                variances=np.ascontiguousarray(analysis.variances[chosen]),
                kinds=surrogate.kinds,
            )
        return verdicts

    def _check_surrogate(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        outputs: list[NDArray[np.float64] | MemberFailure],
        sources: list[str],
        label: str,
        coarse: Mapping[int, NDArray[np.float64] | MemberFailure] | None = None,
    ) -> tuple[list[NDArray[np.float64] | MemberFailure], list[str]]:
        """Run some members by the simulator too and gate the surrogate on them (§5).

        ``coarse``: the forecast's coarse runs by member, for a coarse-corrected model.

        Returns the forecast the update uses: the surrogate's when the check passes (every
        member's, so that the members' errors stay equal), a refit's, or the simulator's.

        Raises
        ------
        SurrogateCheckFailed
            If the check fails and the run stops (``stop``, or no refit left).
        """
        surrogate = self.surrogate
        assert surrogate is not None
        candidates = [j for j, source in enumerate(sources) if source == SURROGATE]
        if not candidates:
            logger.info(
                "nothing to check: no member was forecast by the surrogate",
                extra={"event": "surrogate_check", "iteration": iteration, "skipped": True},
            )
            return outputs, sources
        n = surrogate.check_count(len(candidates))
        if n < len(candidates):
            drawn = self.streams["surrogate"].choice(len(candidates), size=n, replace=False)
            columns = [candidates[int(i)] for i in np.sort(drawn)]
        else:
            columns = candidates
        keys = self._check_keys()
        n_raw = self.raw_observations.n_data
        simulated = self._run_members(
            keep(state[:, columns]), member_ids[columns], iteration, "check", auxiliary=False,
            keys=keys, folder=f"check_{iteration}",
        )  # fmt: skip
        surrogate.simulator_runs += len(columns)
        good = [
            (j, out)
            for j, out in zip(columns, simulated, strict=True)
            if not isinstance(out, MemberFailure)
        ]
        failed = [int(out.member_id) for out in simulated if isinstance(out, MemberFailure)]
        verdicts: dict[str, GateResult] = {}
        checked_ids = [int(member_ids[j]) for j, _ in good]
        if good:
            full = np.ascontiguousarray(np.column_stack([out for _, out in good]))
            simulator_raw = keep(full[:n_raw])
            surrogate_raw = np.ascontiguousarray(
                np.column_stack([_values(outputs[j]) for j, _ in good])
            )
            verdicts = self._gate_roles(surrogate_raw, simulator_raw, checked_ids)
            self.run_dir.save_arrays(
                f"check_{iteration}",
                {
                    "member_ids": np.array(checked_ids, dtype=np.int64),
                    "state": keep(state[:, [j for j, _ in good]]),
                    "simulator_predicted": self._transform(simulator_raw),
                    "surrogate_predicted": self._transform(surrogate_raw),
                    "failed": np.array(failed, dtype=np.int64),
                    "surrogate_id": np.array(surrogate.active.identity),
                    "gate": np.array(
                        json.dumps({r: v.model_dump(mode="json") for r, v in verdicts.items()})
                    ),
                },
            )
        passed = bool(verdicts) and all(v.passed for v in verdicts.values())
        problems = [f"{role}: {line}" for role, v in verdicts.items() for line in v.failures()]
        if not good:
            problems.append("no checked run of the simulator succeeded")
        elif not verdicts:
            # Refused at composition for an early-start model; named, never a silent failure.
            problems.append("no assimilated or cv key to gate")
        errors = {role: {d.datatype: d.error for d in v.datatypes} for role, v in verdicts.items()}
        surrogate.checks.append({
            "iteration": iteration, "passed": passed, "members": checked_ids,
            "failed": failed, "errors": errors, "identity": surrogate.active.identity,
        })  # fmt: skip
        logger.info(
            "surrogate checked",
            extra={"event": "surrogate_check", "iteration": iteration, "passed": passed,
                   "members": checked_ids, "failed": failed, "errors": errors},
        )  # fmt: skip
        if passed:
            return outputs, sources
        policy = surrogate.section.on_check_failure
        exhausted = policy == "refit" and len(surrogate.refits) >= surrogate.section.max_refits
        if policy == "stop" or exhausted or (policy == "refit" and not good):
            reason = (
                f"after {len(surrogate.refits)} refit(s), [surrogate] max_refits"
                if exhausted
                else f'on_check_failure = "{policy}"'
            )
            logger.error(
                "surrogate check failed",
                extra={"event": "surrogate_check_failed", "iteration": iteration,
                       "errors": errors, "problems": problems, "policy": policy},
            )  # fmt: skip
            raise SurrogateCheckFailed(
                f"iteration {iteration}: the surrogate fails its check against the simulator "
                f"({'; '.join(problems)}); the run stops before the update ({reason}), at "
                "its last checkpoint. Resume it once [surrogate] in its config.json says what "
                'to do (on_check_failure = "refit" or "simulator"), or rebuild the surrogate'
            )
        if policy == "refit":
            model_keys = surrogate.active.model.keys
            index = {key: row for row, key in enumerate(keys)}
            rows = [index[key] for key in model_keys]
            coarse_checked = None
            if surrogate.coarse is not None:
                assert coarse is not None
                coarse_checked = np.ascontiguousarray(
                    np.column_stack([_values(coarse[m]) for m in checked_ids])
                )
            surrogate.refit(
                iteration,
                self._surrogate_inputs(keep(state[:, [j for j, _ in good]])),
                keep(full[rows]),
                checked_ids,
                self.streams["surrogate"],
                self.run_dir.path,
                coarse_outputs=coarse_checked,
            )
            # The iteration's forecast again, with the refit (not checked again: the checked
            # runs are now its training and held-out runs). Members the simulator ran
            # already (outside the old domain) keep their runs.
            self._source_changed = iteration
            done = {j: out for j, (out, s) in enumerate(zip(outputs, sources, strict=True))
                    if s == SIMULATOR}  # fmt: skip
            return self._surrogate_members(
                state, member_ids, iteration, label, simulated=done, coarse=coarse
            )
        surrogate.switched_at = iteration
        self._source_changed = iteration
        logger.warning(
            "the surrogate failed its check: the rest of the run uses the simulator",
            extra={"event": "surrogate_switched", "iteration": iteration, "errors": errors},
        )  # fmt: skip
        checked = {j: out[:n_raw] for j, out in good}
        rest = [j for j in range(len(member_ids)) if j not in checked]
        merged: list[NDArray[np.float64] | MemberFailure] = list(outputs)
        if rest:
            simulated_rest = self._run_members(
                keep(state[:, rest]), member_ids[rest], iteration, label, auxiliary=False
            )
            surrogate.simulator_runs += len(rest)
            for j, out in zip(rest, simulated_rest, strict=True):
                merged[j] = out
        for j, out in checked.items():
            merged[j] = out
        return merged, [SIMULATOR] * len(member_ids)

    def _rescore_accepted(self, algorithm: EnsembleAlgorithm, iteration: int) -> None:
        """Forecast the accepted iterate again by the new source; let the method rescore it.

        An iterative method judges a trial against its accepted iterate: after
        a switch to the simulator or a refit, the trial is the new source's and so must the
        accepted iterate (and its mean model, with RLM-MAC) be. Methods without acceptance
        (ES-MDA) have nothing to rescore.

        Raises
        ------
        WorkflowError
            If an iterative method cannot rescore its accepted iterate (refused at
            validation), or a member of the accepted iterate fails the new forecast.
        """
        accepted = getattr(algorithm, "accepted_ensemble", None)
        rescore = getattr(algorithm, "rescore_accepted", None)
        if accepted is None or rescore is None:
            if getattr(algorithm, "iterative", False):
                raise WorkflowError(
                    f"{type(algorithm).__name__} cannot judge its trials after the forecasts' "
                    "source changed (no rescore_accepted)"
                )
            return
        state, member_ids, needs_reference = accepted()
        folder = f"accepted_{iteration}"
        forecast, sources = self._accepted_forecast(
            state, member_ids, iteration, folder, residuals=True
        )
        residuals = self._last_auxiliary
        predicted = self._assimilated(forecast)
        reference = None
        if needs_reference:
            mean = keep(state.mean(axis=-1, keepdims=True))
            ids = np.array([MEAN_MODEL_ID], dtype=np.int64)
            mean_forecast, _ = self._accepted_forecast(mean, ids, iteration, folder)
            reference = self._assimilated(mean_forecast)[:, 0].copy()
        rescore(predicted, reference)
        surrogate = self.surrogate
        assert surrogate is not None
        source = SIMULATOR if surrogate.switched_at is not None else SURROGATE
        # Saved, so that cross validation compares like with like (_replay_stopping) and the
        # posterior's forecast can reuse the simulator's runs (_forecast_full).
        accepted_iteration = self._last_accepted()
        held = {role: keep(forecast[rows]) for role, (rows, _) in self._held_out.items()}
        arrays: dict[str, NDArray[Any]] = {
            "state": keep(state),
            "predicted": predicted,
            "member_ids": member_ids,
            "forecast_source": np.array(sources),
            "surrogate_id": np.array(surrogate.active.identity if source == SURROGATE else ""),
            "accepted_iteration": np.array(accepted_iteration),
            **{f"{role}_predicted": values for role, values in held.items()},
        }
        if residuals is not None:
            # The deck's forecast with its model error's residuals:
            # the posterior's forecast reuses the members with them (_forecast_full).
            arrays["auxiliary_predicted"] = keep(residuals)
            arrays["auxiliary_keys"] = np.array([str(k) for k in self._surrogate_auxiliary_keys])
        for role, values in held.items():
            for label, per_member in _mean_square_by_type(self._held_out[role][1], values).items():
                arrays[f"metric:{role}_normalized_data_mismatch:{label}"] = per_member
        self.run_dir.save_arrays(f"rescored_{iteration}", arrays)
        surrogate.rescores.append(
            {"iteration": iteration, "accepted": accepted_iteration, "source": source}
        )
        logger.info(
            "accepted iterate forecast again by the new source",
            extra={"event": "surrogate_rescored", "iteration": iteration,
                   "accepted": accepted_iteration, "source": source,
                   "identity": surrogate.active.identity, "members": len(member_ids)},
        )  # fmt: skip

    def _accepted_forecast(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        folder: str,
        *,
        residuals: bool = False,
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Forecast members by the current source; every one must succeed.

        Returns their forecast in the analysis space (every row, held-out ones included)
        and each member's source. ``residuals``: on the deck (after a switch), also its
        model error's residuals, left in ``_last_auxiliary``.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        self._last_auxiliary = None
        if surrogate.switched_at is not None:
            keys = self._surrogate_auxiliary_keys if residuals else ()
            outputs = self._run_members(
                state, member_ids, iteration, "accepted iterate", auxiliary=bool(keys),
                folder=folder, auxiliary_keys=keys or None,
            )  # fmt: skip
            surrogate.simulator_runs += len(member_ids)
            sources = [SIMULATOR] * len(member_ids)
        else:
            outputs, sources = self._surrogate_members(
                state, member_ids, iteration, "accepted iterate", folder=folder
            )
        failed = [out for out in outputs if isinstance(out, MemberFailure)]
        if failed:
            raise WorkflowError(
                f"iteration {iteration}: the accepted iterate's member {failed[0].member_id} "
                f"failed its forecast by the new source ({failed[0].message}); the run stops at "
                "its last checkpoint"
            )
        raw = np.ascontiguousarray(np.column_stack([_values(out) for out in outputs]))
        return keep(self._transform(raw)), sources

    def _surrogate_variance(
        self, state: NDArray[np.float64], member_ids: NDArray[np.int64], sources: Sequence[str]
    ) -> NDArray[np.float64] | None:
        """Return the members' mean predictive variance per datum, analysis rows (§2).

        Only the members the surrogate forecast count. ``None`` when nothing applies.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        predict_std = getattr(surrogate.active.model, "predict_std", None)
        if predict_std is None:
            return None
        columns = [j for j, source in enumerate(sources) if source == SURROGATE]
        std = np.asarray(
            predict_std(
                self._surrogate_inputs(keep(state[:, columns])), self.raw_observations.keys
            ),
            dtype=np.float64,
        )
        # Raw rows are the analysis rows here: composition refuses variance_in_errors with
        # data transforms.
        variance = np.mean(np.square(std), axis=-1)
        return np.ascontiguousarray(variance[self.role_rows()["history_matching"]])

    def _sources_of(self, record: Record) -> dict[str, NDArray[Any]] | None:
        """Return a surrogate run's source arrays for a record's members (``None`` otherwise)."""
        if self.surrogate is None:
            return None
        pending = self._pending_sources.get(record.iteration)
        if pending is None:
            # Never saved unlabelled: it would read as the simulator's.
            raise WorkflowError(
                f"no forecast of iteration {record.iteration} for its record (the algorithm "
                "must emit a record after the forecast of its own iteration)"
            )
        ids, sources, identity, variance = pending
        # A record is the last use of its iteration's forecast (one record per forecast).
        self._pending_sources.pop(record.iteration, None)
        arrays: dict[str, NDArray[Any]] = {
            "forecast_source": sources[_columns_of(ids, record.member_ids)],
            "surrogate_id": np.array(identity),
        }
        if variance is not None:
            arrays["surrogate_variance"] = variance
        return arrays

    def _restore_surrogate(self, saved: Mapping[str, Any]) -> None:
        if self.surrogate is None:
            if "surrogate" in saved:
                raise WorkflowError(
                    "the run was started with [surrogate]; it cannot be resumed without it"
                )
            return
        self.surrogate.restore(saved.get("surrogate"), self.run_dir.path)

    def _last_accepted(self) -> int:
        """Return the iteration of the last kept ensemble (the posterior without CV)."""
        accepted = [int(h["iteration"]) for h in self.run_dir.history_dicts() if h["accepted"]]
        return accepted[-1]

    def _full_posterior(
        self, iteration: int, state: NDArray[np.float64], member_ids: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
        """Forecast the posterior iterate by the simulator (``full_<k>.npz``; §5).

        Mandatory (decision 3): the run's reported posterior is the simulator's. An iterate
        the simulator forecast already (after a switch to it) is not run again, nor is one
        whose ``full_<k>.npz`` a finished run saved; members the simulator forecast already
        (outside the domain, or the accepted iterate forecast again after a switch) keep
        those runs (``reused_ids``). Returns the members that succeeded: their state,
        forecast (history-matching rows) and ids.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        saved = upgrade_iterate(self.run_dir.load_arrays(f"iter_{iteration}"))
        columns = _columns_of(saved["member_ids"], member_ids)
        sources = saved["forecast_source"][columns]
        name = f"full_{iteration}"
        if not np.any(sources == SURROGATE):
            self._posterior_report = {"iteration": iteration, "forecast": "simulator already",
                                      "members": len(member_ids)}  # fmt: skip
            return state, keep(saved["predicted"][:, columns]), member_ids
        reused = self.run_dir.has_arrays(name) and np.array_equal(
            self.run_dir.load_arrays(name)["requested_ids"], member_ids
        )
        if not reused:
            self._forecast_full(iteration, state, member_ids, self._simulated_of(iteration))
        full = self.run_dir.load_arrays(name)
        self._posterior_report = self._compare_posterior(iteration, saved, full)
        logger.info(
            "posterior forecast by the simulator",
            extra={"event": "surrogate_posterior", "iteration": iteration, "reused": reused,
                   **{k: v for k, v in self._posterior_report.items() if k != "gates"}},
        )  # fmt: skip
        return keep(full["state"]), keep(full["predicted"]), full["member_ids"]

    def _simulated_residuals_of(self, iteration: int) -> dict[int, NDArray[np.float64]]:
        """Return the model error's residuals the simulator's forecasts of ``iteration`` saved.

        By member, from the same sources as :meth:`_simulated_of` (its ``iter_<k>`` and the
        accepted iterate's forecasts again, ``rescored_<j>``), where they hold them (the
        deck's forecasts after a switch to the simulator).
        """
        found: dict[int, NDArray[np.float64]] = {}
        sources = [upgrade_iterate(self.run_dir.load_arrays(f"iter_{iteration}"))]
        for record in self.surrogate.rescores if self.surrogate else []:
            if int(record["accepted"]) == iteration and record["source"] == SIMULATOR:
                sources.append(self.run_dir.load_arrays(f"rescored_{record['iteration']}"))
        for saved in sources:
            has = "auxiliary_predicted" in saved
            for column, (member, source) in enumerate(
                zip(saved["member_ids"], saved["forecast_source"], strict=True)
            ):
                if str(source) != SIMULATOR:
                    continue
                # As _simulated_of, a later source wins: a member's residuals come from the
                # file its forecast does, or the member is run again.
                if has:
                    found[int(member)] = np.asarray(saved["auxiliary_predicted"][:, column])
                else:
                    found.pop(int(member), None)
        return found

    def _simulated_of(self, iteration: int) -> dict[int, NDArray[np.float64]]:
        """Return the simulator's forecasts of iterate ``iteration`` made already, by member.

        Each in the analysis space, every row: the members its ``iter_<k>`` records as the
        simulator's, and an accepted iterate forecast again by the simulator after a switch
        (``rescored_<j>``).
        """
        found: dict[int, NDArray[np.float64]] = {}
        sources = [upgrade_iterate(self.run_dir.load_arrays(f"iter_{iteration}"))]
        for record in self.surrogate.rescores if self.surrogate else []:
            if int(record["accepted"]) == iteration and record["source"] == SIMULATOR:
                sources.append(self.run_dir.load_arrays(f"rescored_{record['iteration']}"))
        rows = self.role_rows()
        for saved in sources:
            for column, (member, source) in enumerate(
                zip(saved["member_ids"], saved["forecast_source"], strict=True)
            ):
                if str(source) != SIMULATOR:
                    continue
                values = np.full(self.analysis_observations.n_data, np.nan)
                values[rows["history_matching"]] = saved["predicted"][:, column]
                for role in self._held_out:
                    values[rows[role]] = saved[f"{role}_predicted"][:, column]
                found[int(member)] = values
        return found

    def _forecast_full(
        self,
        iteration: int,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        simulated: Mapping[int, NDArray[np.float64]],
    ) -> None:
        """Run the simulator on the posterior's members; score and save ``full_<k>.npz``.

        ``simulated``: the simulator's forecasts made already (analysis space, by member),
        kept rather than run again.

        With a model error the residuals are saved with the forecast
        (``auxiliary_predicted``) for every member: a member is reused only with its
        residuals (the deck's forecasts after a switch to the simulator save them), the
        others run.
        """
        residuals = self._surrogate_auxiliary_keys
        known: dict[int, NDArray[np.float64]] = {}
        if residuals:
            known = self._simulated_residuals_of(iteration)
            simulated = {m: v for m, v in simulated.items() if m in known}
        to_run = [j for j, m in enumerate(member_ids) if int(m) not in simulated]
        ran = (
            self._run_members(
                keep(state[:, to_run]), member_ids[to_run], iteration, "posterior (simulator)",
                auxiliary=bool(residuals), folder=f"full_{iteration}",
                auxiliary_keys=residuals or None,
            )
            if to_run
            else []
        )  # fmt: skip
        ran_residuals = self._last_auxiliary if residuals else None
        succeeded = [
            j for j, o in zip(to_run, ran, strict=True) if not isinstance(o, MemberFailure)
        ]
        columns: dict[int, NDArray[np.float64]] = {}
        if succeeded:
            raw = np.column_stack([o for o in ran if not isinstance(o, MemberFailure)])
            transformed = self._transform(raw)
            columns = {j: transformed[:, c] for c, j in enumerate(succeeded)}
        by_column = dict(zip(to_run, ran, strict=True))
        outputs: list[NDArray[np.float64] | MemberFailure] = []
        for j, member in enumerate(member_ids):
            if j not in by_column:
                outputs.append(simulated[int(member)])
            elif j in columns:
                outputs.append(columns[j])
            else:
                outputs.append(by_column[j])
        failures = [out for out in outputs if isinstance(out, MemberFailure)]
        for failure in failures:
            logger.warning(
                "member failed",
                extra={"event": "member_failed", "iteration": iteration,
                       "member_id": failure.member_id, "reason": failure.message,
                       "label": "posterior (simulator)"},
            )  # fmt: skip
        ok = np.array([not isinstance(out, MemberFailure) for out in outputs])
        if int(ok.sum()) < max(2, self.min_success_fraction * len(outputs)):
            raise WorkflowError(
                f"the posterior's forecast by the simulator: only {int(ok.sum())} of "
                f"{len(outputs)} members succeeded (minimum fraction "
                f"{self.min_success_fraction}); first failure: "
                f"{failures[0].message if failures else 'n/a'}"
            )
        # Already in the analysis space: run members transformed above, kept ones as saved.
        analysis = np.ascontiguousarray(
            np.column_stack([out for out in outputs if not isinstance(out, MemberFailure)])
        )
        predicted = self._assimilated(analysis)
        kept, ids = keep(state, ok), member_ids[ok]
        arrays: dict[str, NDArray[Any]] = {
            "state": kept,
            "predicted": predicted,
            "member_ids": ids,
            "requested_ids": member_ids,
            "reused_ids": np.array(
                [int(m) for m in member_ids if int(m) in simulated], dtype=np.int64
            ),
            "forecast_source": np.array([SIMULATOR] * len(ids)),
            "surrogate_id": np.array(""),
        }
        if residuals:
            # Each kept member's residuals: a reused one's as saved, a run one's from this
            # forecast (its successful members, in their order).
            ran_columns = dict(zip(succeeded, range(len(succeeded)), strict=True))
            per_member = []
            for j, member in enumerate(member_ids):
                if not ok[j]:
                    continue
                if j in ran_columns:
                    assert ran_residuals is not None
                    per_member.append(ran_residuals[:, ran_columns[j]])
                else:
                    per_member.append(known[int(member)])
            arrays["auxiliary_predicted"] = keep(np.column_stack(per_member))
            arrays["auxiliary_keys"] = np.array([str(k) for k in residuals])
        held = {role: keep(analysis[rows]) for role, (rows, _) in self._held_out.items()}
        arrays.update({f"{role}_predicted": values for role, values in held.items()})
        if self.run_dir.has_arrays("perturbed_observations"):
            # Scored as _record scores every forecast: overall and per data type.
            perturbed = self._perturbed_for(ids)
            arrays["data_mismatch"] = self.data_error.data_mismatch(perturbed - predicted)
            arrays["metric:data_mismatch:all"] = arrays["data_mismatch"]
            sigma = np.sqrt(self.observations.variances)[:, None]
            weighted = ((perturbed - predicted) / sigma) ** 2
            for name, rows in self._type_rows.items():
                arrays[f"metric:data_mismatch:{name}"] = weighted[rows].sum(axis=0)
        values = {
            **self._evaluate_metrics(kept, predicted, self._previous_state(iteration, ids)),
            **{
                f"{role}_normalized_data_mismatch": _mean_square_by_type(self._held_out[role][1], v)
                for role, v in held.items()
            },
        }
        arrays.update({
            f"metric:{key}:{label}": per_member
            for key, result in values.items()
            for label, per_member in result.items()
        })  # fmt: skip
        self.run_dir.save_arrays(f"full_{iteration}", arrays)

    def _compare_posterior(
        self, iteration: int, saved: Mapping[str, NDArray[Any]], full: Mapping[str, NDArray[Any]]
    ) -> dict[str, Any]:
        """Compare the surrogate's and the simulator's forecasts of the posterior (§5).

        Both scores on the members the simulator forecast, their difference, and the
        gate of the surrogate's forecast against the simulator's (the report flags a
        difference beyond it).
        """
        ids = full["member_ids"]
        columns = _columns_of(saved["member_ids"], ids)
        scores = self.run_dir.load_arrays(f"data_mismatch_{iteration}")
        score_columns = _columns_of(scores["member_ids"], ids)
        surrogate_side: dict[str, float] = {}
        simulator_side: dict[str, float] = {}
        for metric in ("data_mismatch", "normalized_data_mismatch", "cv_normalized_data_mismatch"):
            key = f"metric:{metric}:all"
            if key in scores and key in full:
                surrogate_side[metric] = float(np.mean(scores[key][score_columns]))
                simulator_side[metric] = float(np.mean(full[key]))
        # Both in the analysis space already (the saved forecasts are transformed).
        forecast = {"history_matching": keep(saved["predicted"][:, columns])}
        reference = {"history_matching": full["predicted"]}
        if "cv_predicted" in saved and "cv_predicted" in full:
            forecast["cv"] = keep(saved["cv_predicted"][:, columns])
            reference["cv"] = full["cv_predicted"]
        gates = self._gate_analysis(forecast, reference, [int(m) for m in ids])
        metrics: dict[str, dict[str, float]] = {}
        for key in full:
            if key.startswith("metric:"):
                _, metric, label = key.split(":", 2)
                metrics.setdefault(metric, {})[label] = float(np.mean(full[key]))
        return {
            "iteration": iteration,
            "forecast": "simulator",
            "members": len(ids),
            "failed": [int(m) for m in full["requested_ids"] if int(m) not in set(ids.tolist())],
            "reused": [int(m) for m in full.get("reused_ids", ())],
            "surrogate": surrogate_side,
            "simulator": simulator_side,
            "difference": {k: simulator_side[k] - surrogate_side[k] for k in simulator_side},
            "simulator_metrics": metrics,
            "gates": {role: g.model_dump(mode="json") for role, g in gates.items()},
            "beyond_gate": not all(g.passed for g in gates.values()),
        }

    def _gate_analysis(
        self,
        surrogate: Mapping[str, NDArray[np.float64]],
        reference: Mapping[str, NDArray[np.float64]],
        member_ids: Sequence[int],
    ) -> dict[str, GateResult]:
        """Gate forecasts already in the analysis space, by role (history_matching, cv).

        An early-start model's early keys are left out, as in the checks (:meth:`_gate_roles`).
        """
        assert self.surrogate is not None
        model = self.surrogate.active.model
        early = set(model.early_keys()) if isinstance(model, EarlyStartSurrogate) else set()
        verdicts = {}
        for role, values in surrogate.items():
            observations = (
                self.observations if role == "history_matching" else self._held_out[role][1]
            )
            rows = [i for i, key in enumerate(observations.keys) if key not in early]
            if not rows:
                continue
            verdicts[role] = gate(
                keep(values[rows]),
                keep(reference[role][rows]),
                [observations.keys[i] for i in rows],
                member_ids,
                tolerances=self.surrogate.tolerances,
                variances=np.ascontiguousarray(observations.variances[rows]),
                kinds=self.surrogate.kinds,
            )
        return verdicts

    def surrogate_report(self) -> dict[str, Any] | None:
        """Return a surrogate run's facts for ``summary.json`` (``None`` without a surrogate).

        The model, its checks and refits, the evaluations and simulator runs, each scored
        forecast's source, and the posterior: both forecasts' scores and their difference.
        """
        if self.surrogate is None:
            return None
        report = self.surrogate.describe()
        posterior = self._posterior_report
        full_runs = 0
        if posterior is not None and posterior.get("forecast") == "simulator":
            full_runs = (
                int(posterior["members"])
                + len(posterior.get("failed", []))
                - len(posterior.get("reused", []))
            )
        report["simulator_runs"] = self.surrogate.simulator_runs + full_runs
        report["sources"] = {
            str(h["iteration"]): h.get("forecast_source", SIMULATOR)
            for h in self.run_dir.history_dicts()
        }
        report["posterior"] = posterior
        report["timing"] = forecast_timing(self.run_dir.events(), self.surrogate.cost)
        return report

    # ------------------------------------------------------------------ shared steps

    def _check_prior(self, prior: NDArray[np.float64]) -> NDArray[np.float64]:
        prior = np.asarray(prior, dtype=np.float64)
        if prior.ndim != 2 or prior.shape[0] != self.layout.n_state or prior.shape[1] < 2:
            raise WorkflowError(
                f"prior must be ({self.layout.n_state}, n_ensemble >= 2), got {prior.shape}"
            )
        if not np.all(np.isfinite(prior)):
            raise WorkflowError("prior ensemble contains non-finite values")
        return prior

    def _perturbed_for(self, member_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return the saved perturbed observations of ``member_ids``, in that order."""
        saved = self.run_dir.load_arrays("perturbed_observations")
        return _members_of(saved["perturbed_observations"], saved["member_ids"], member_ids)

    def _members(
        self,
        state: NDArray[np.float64],
        member_ids: Sequence[int],
        iteration: int,
        folder: str | None = None,
        simulator: Simulator | None = None,
        stop_at: str | None = None,
    ) -> list[MemberInput]:
        needs_work_dir = bool(getattr(simulator or self.simulator, "needs_work_dir", False))
        blocks = self.layout.split(state)
        return [
            MemberInput(
                member_id=int(member_id),
                state={name: block[:, column].copy() for name, block in blocks.items()},
                work_dir=self.run_dir.simulation_dir(iteration, int(member_id), folder)
                if needs_work_dir
                else None,
                stop_at=stop_at,
            )
            for column, member_id in enumerate(member_ids)
        ]

    def _run_members(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        label: str,
        simulator: Simulator | None = None,
        *,
        auxiliary: bool = True,
        keys: Sequence[DataKey] | None = None,
        folder: str | None = None,
        stop_at: str | None = None,
        auxiliary_keys: Sequence[DataKey] | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run the members through the executor, reporting progress as they finish.

        ``auxiliary``: also request the simulator's auxiliary keys (saved with the record;
        a required forecast such as the mean model's has none). ``keys``: other keys than
        the observations' (a surrogate's check, which may need the model's too); ``folder``:
        the work directories' folder when not ``iter_<iteration>``; ``stop_at``:
        the report date the members run to (an early-start surrogate's early runs).
        """
        self.progress.forecast_started(iteration=iteration, label=label, n_members=len(member_ids))
        began = time.monotonic()
        keys = self.raw_observations.keys if keys is None else tuple(keys)
        extra = (
            tuple(auxiliary_keys)
            if auxiliary_keys is not None
            else self.auxiliary_keys
            if simulator is None and auxiliary
            else ()
        )
        # A required forecast (the mean model's) keeps no raw rows: an executor applying the
        # transforms elsewhere sends none back for it.
        without_raw = getattr(self.executor, "without_raw_rows", None)
        with nullcontext() if auxiliary or without_raw is None else without_raw():
            outputs = self.executor.run_all(
                self.simulator if simulator is None else simulator,
                self._members(state, list(member_ids), iteration, folder, simulator, stop_at),
                [*keys, *extra] if extra else keys,
                on_result=lambda member_id, ok: self.progress.member_finished(
                    member_id=member_id, ok=ok
                ),
            )
        self._last_auxiliary = None
        self._last_raw = None
        n_keys = len(keys)
        if self._transformed_remotely:
            # [transformed rows, auxiliary rows, raw rows if saved].
            n_keys = self.analysis_observations.n_data
            n_head = n_keys + len(extra)
            raws = [out[n_head:] for out in outputs if not isinstance(out, MemberFailure)]
            if self.save_raw_forecasts and auxiliary and raws:
                self._last_raw = np.column_stack(raws)
            outputs = [out if isinstance(out, MemberFailure) else out[:n_head] for out in outputs]
        if extra:
            values = [out[n_keys:] for out in outputs if not isinstance(out, MemberFailure)]
            self._last_auxiliary = np.column_stack(values) if values else None
            outputs = [out if isinstance(out, MemberFailure) else out[:n_keys] for out in outputs]
        seconds = time.monotonic() - began
        if bool(getattr(simulator or self.simulator, "needs_work_dir", False)):
            # Retention ([output] keep_simulations): failed members are marked first, so a
            # policy that keeps them can tell them apart later.
            for out in outputs:
                if isinstance(out, MemberFailure):
                    self.run_dir.mark_failed(iteration, out.member_id, out.message, folder)
            # an executor that keeps members' files on other hosts (a failed
            # one's marked there by its worker) removes them there, before their markers
            # here go.
            remove = getattr(self.executor, "remove_member_files", None)
            if remove is not None:
                remove(self.run_dir.simulations_to_prune(iteration, self.keep_simulations, folder))
            self.run_dir.prune_simulations(iteration, self.keep_simulations, folder)
        n_failed = sum(isinstance(out, MemberFailure) for out in outputs)
        self.progress.forecast_finished(
            n_ok=len(outputs) - n_failed, n_failed=n_failed, seconds=seconds
        )
        logger.info(
            "forecast done",
            extra={
                "event": "forecast_done",
                "iteration": iteration,
                "label": label,
                "n_ok": len(outputs) - n_failed,
                "n_failed": n_failed,
                "seconds": round(seconds, 3),
                **self._concurrency(len(member_ids), simulator),
            },
        )
        return outputs

    def _concurrency(self, n_members: int, simulator: Simulator | None = None) -> dict[str, int]:
        """Return ``{"workers": members run at once}`` when the executor states its workers.

        ``simulator``: the one the forecast ran (a coarse stack's), when not the run's.

        A surrogate run's timing report (decision 14) turns wall time into
        worker-seconds with it; executors that place members elsewhere state none.
        """
        workers = getattr(self.executor, "workers", None)
        if not isinstance(workers, int):
            return {}
        # The process executor runs no more than the simulator's licences at once.
        cap = getattr(simulator or self.simulator, "max_parallel_runs", None)
        cap = cap if isinstance(cap, int) and cap > 0 else workers
        return {"workers": max(1, min(workers, cap, n_members))}

    def _forecast(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        *,
        sensitivities: bool = False,
        outputs: list[NDArray[np.float64] | MemberFailure] | None = None,
    ) -> tuple[
        NDArray[np.bool_],
        NDArray[np.float64],
        NDArray[np.float64] | None,
        NDArray[np.float64] | None,
    ]:
        """Forecast every member.

        Returns the success mask, the survivors' predictions in the analysis data space,
        their raw (untransformed) predictions when those are to be saved, else ``None``,
        and with ``sensitivities`` their mean sensitivity ``(n_data, n_state)``.
        ``outputs``: the members' raw forecasts made already (a surrogate's), in
        member order; ``None`` runs them here.
        """
        simulator: Simulator | None = None
        n_keys = len(self.raw_observations.keys)
        if sensitivities:
            if not hasattr(self.simulator, "run_with_sensitivities"):
                raise WorkflowError("the update needs sensitivities, which the simulator lacks")
            if self.transforms:
                raise WorkflowError("sensitivities with data transforms are not supported")
            # The executors run it unchanged: each output is [values, jacobian.ravel()].
            simulator = SensitivityRun(self.simulator, state.shape[0])
        if outputs is None:
            outputs = self._run_members(state, member_ids, iteration, "ensemble", simulator)
        sensitivity = None
        if sensitivities:
            jacobians = [out[n_keys:] for out in outputs if not isinstance(out, MemberFailure)]
            outputs = [out if isinstance(out, MemberFailure) else out[:n_keys] for out in outputs]
            if jacobians:
                sensitivity = np.mean(jacobians, axis=0).reshape(n_keys, state.shape[0])
        failures = [out for out in outputs if isinstance(out, MemberFailure)]
        for failure in failures:
            logger.warning(
                "member failed",
                extra={
                    "event": "member_failed",
                    "iteration": iteration,
                    "member_id": failure.member_id,
                    "reason": failure.message,
                    "transient": failure.transient,
                    "attempts": failure.attempts,
                },
            )
        ok = np.array([not isinstance(out, MemberFailure) for out in outputs])
        n_ok, n_total = int(ok.sum()), len(outputs)
        if n_ok < max(2, self.min_success_fraction * n_total):
            raise WorkflowError(
                f"iteration {iteration}: only {n_ok} of {n_total} members simulated successfully "
                f"(minimum fraction {self.min_success_fraction}); first failure: "
                f"{failures[0].message if failures else 'n/a'}"
            )
        predicted = np.column_stack([out for out in outputs if not isinstance(out, MemberFailure)])
        if self._transformed_remotely:
            return ok, predicted, self._last_raw, sensitivity
        raw = predicted if self.save_raw_forecasts else None
        return ok, self._transform(predicted), raw, sensitivity

    def transform(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw predictions ``(n_raw, n)`` into the analysis data space ``(n_data, n)``.

        The transforms are those fitted to the observations (ERT's ``APEX_TRANSFORM`` step
        applies them to ERT's runs).
        """
        return self._transform(predicted)

    def _transform(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map raw predictions ``(n_raw, n)`` into the analysis data space ``(n_data, n)``."""
        for transform in self.transforms:
            predicted = transform.apply(predicted)
        return predicted

    def _record(
        self,
        iteration: int,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        perturbed: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        *,
        accepted: bool = True,
        damping: float | None = None,
        held_out_scores: Mapping[str, dict[str, NDArray[np.float64]]] | None = None,
        forecast_source: str | None = None,
    ) -> None:
        data_mismatches = self.data_error.data_mismatch(perturbed - predicted)
        # C_d is diagonal here, so the per-type shares add up to the data mismatch.
        sigma = np.sqrt(self.observations.variances)[:, None]
        weighted = ((perturbed - predicted) / sigma) ** 2
        names = list(self._type_rows)
        if "all" in names:
            # "all" labels the total; a data type of that name would overwrite it.
            raise WorkflowError("a data type may not be named 'all' (it labels the total)")
        data_mismatch_by_type = np.array(
            [weighted[r].sum(axis=0) for r in self._type_rows.values()]
        )
        previous = self._previous_state(iteration, member_ids)
        values = {
            # The objective itself is a metric too, so that studies, summaries
            # and reports find it as "data_mismatch", overall and per data type.
            DATA_MISMATCH: {
                "all": data_mismatches,
                **{n: data_mismatch_by_type[t] for t, n in enumerate(names)},
            },
            **self._evaluate_metrics(state, predicted, previous),
            **(held_out_scores or {}),
        }
        # Per-member statistics of every forecast (including rejected trials), for reports.
        self.run_dir.save_arrays(
            f"data_mismatch_{iteration}",
            {
                "data_mismatch": data_mismatches,
                "member_ids": member_ids,
                "datatypes": np.array(names),
                "data_mismatch_by_type": data_mismatch_by_type,
                **{
                    f"metric:{key}:{label}": per_member
                    for key, result in values.items()
                    for label, per_member in result.items()
                },
            },
        )
        record = IterationRecord(
            iteration=iteration,
            data_mismatch_mean=float(data_mismatches.mean()),
            data_mismatch_std=float(data_mismatches.std()),
            n_members=int(predicted.shape[-1]),
            accepted=accepted,
            damping=damping,
            data_mismatch_by_type={
                n: float(data_mismatch_by_type[t].mean()) for t, n in enumerate(names)
            },
            metrics={
                key: {label: float(v.mean()) for label, v in result.items()}
                for key, result in values.items()
            },
            forecast_source=forecast_source,
        )
        self.run_dir.append_history(record)
        logger.info("forecast scored", extra={"event": "scored", **dataclasses.asdict(record)})
        self.progress.record_scored(record)

    def _check_metrics(self) -> None:
        """Evaluate every metric once on a stand-in ensemble, before anything is simulated.

        Options that do not fit the state (a wrong grid, actnum size or variable name)
        would otherwise only fail after the prior forecast.

        Raises
        ------
        ValueError
            If a metric cannot be evaluated.
        """
        if not self.metrics:
            return
        base = np.zeros(self.layout.n_state) if self._truth is None else self._truth_vector()
        state = base[:, None] + np.array([0.0, 1.0])  # two distinct members
        predicted = np.tile(self.observations.values[:, None], (1, 2))
        try:
            # The stand-in is its own previous ensemble, so previous-based metrics are checked.
            self._evaluate_metrics(state, predicted, self.layout.split(state))
        except WorkflowError as error:
            raise ValueError(str(error)) from None

    def _save_split(self) -> None:
        """Write ``checkpoints/data_split.json`` and log the roles (runs with a split)."""
        if self.split is None:
            return
        description = {
            **self.split.describe(),
            "stopping": None if self.stopping is None else type(self.stopping).__name__,
        }
        path = self.run_dir.checkpoints / "data_split.json"
        path.write_text(json.dumps(description, indent=2) + "\n")
        counts = {
            "history_matching": len(self.split.history_matching),
            "cv": len(self.split.cv),
            "test": len(self.split.test),
            "unused": len(self.split.unused),
        }
        logger.info(
            "data split: %s",
            ", ".join(f"{role} {n}" for role, n in counts.items() if n),
            extra={"event": "data_split", "counts": counts},
        )
        if self.split.unused:
            logger.warning(
                "%d data are in no [validation] role and are not used",
                len(self.split.unused),
                extra={"event": "data_unused"},
            )

    def _validation_state(self) -> dict[str, Any]:
        """Return the stopping rule's state and whether it stopped the run (``state.json``)."""
        if self.stopping is None:
            return {}
        return {"validation": {"rule": self.stopping.state(), "stopped": self._cv_stop_requested}}

    def _restore_validation(self, saved: Mapping[str, Any]) -> None:
        if self.stopping is None:
            return
        validation = saved.get("validation")
        if validation is None:
            raise WorkflowError(
                "the run was started without cross validation; it cannot be resumed with it"
            )
        self.stopping.restore(validation["rule"])
        self._cv_stop_requested = self._cv_stopped = bool(validation["stopped"])

    def _finish_selected(
        self, iteration: int, *, converged: bool | None, complete: bool
    ) -> HistoryMatchResult:
        """Finish with the minimum-CV iterate ``iteration`` (from its ``iter_<k>``)."""
        saved = self.run_dir.load_arrays(f"iter_{iteration}")
        result = self._finish(
            keep(saved["state"]),
            keep(saved["predicted"]),
            saved["member_ids"],
            converged=converged,
            complete=complete,
            iteration=iteration,
        )
        return dataclasses.replace(
            result, selected_iteration=iteration, stopped_by_cv=self._cv_stopped
        )

    def _save_truth(self) -> None:
        # The run keeps its own copy, so resume and reports do not depend on the input file.
        if self._truth is not None:
            self.run_dir.save_arrays("truth", {"truth": self._truth_vector()})

    def _truth_vector(self) -> NDArray[np.float64]:
        assert self._truth is not None
        return self.layout.stack(self._truth)[:, 0]

    def _previous_state(
        self, iteration: int, member_ids: NDArray[np.int64]
    ) -> dict[str, NDArray[np.float64]] | None:
        """Return the ensemble iteration ``iteration`` was updated from, on ``member_ids``.

        That is the last accepted iterate before it, read back from its checkpoint (so a
        resumed run needs no extra state). ``None`` for the prior, and when no metric asks
        for it (``requires_previous``).
        """
        if not any(getattr(m, "requires_previous", False) for m in self.metrics.values()):
            return None
        earlier = [
            int(h["iteration"])
            for h in self.run_dir.history_dicts()
            if h.get("accepted", True) and int(h["iteration"]) < iteration
        ]
        if not earlier:
            return None
        saved = self.run_dir.load_arrays(f"iter_{earlier[-1]}")
        return self.layout.split(_members_of(saved["state"], saved["member_ids"], member_ids))

    def _evaluate_metrics(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        previous: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> dict[str, dict[str, NDArray[np.float64]]]:
        """Evaluate every metric on one forecast: ``{key: {label: (n_members,)}}``.

        A metric that is not defined for this forecast (``{}``, e.g. the model change of
        the prior) is left out.
        """
        if not self.metrics:
            return {}
        truth = None if self._truth is None else {k: v[:, 0] for k, v in self._truth.items()}
        grids = {v.name: v.grid for v in self.layout.variables if v.grid is not None}
        current = self.layout.split(state)
        if not self.property_map.identity:
            # Metrics compare properties (facies, porosity, ...), never latent
            # variables, which have no physical truth; the truth is mapped the same way.
            current = self.property_map.ensemble(current)
            previous = None if previous is None else self.property_map.ensemble(previous)
            truth = None if truth is None else self.property_map.member(truth)
            grids = self.property_map.grids
        context = MetricContext(
            state=current,
            predicted=predicted,
            observations=self.observations,
            truth=truth,
            grids=grids,
            previous_state=previous,
            categorical=self.property_map.categorical,
            auxiliary=self.auxiliary_variables,
        )
        results = {}
        for key, metric in self.metrics.items():
            try:
                result = metric.evaluate(context)
            except ValueError as error:
                raise WorkflowError(f"metric {key!r}: {error}") from error
            if not result:
                continue
            n = predicted.shape[-1]
            bad = [label for label, v in result.items() if np.shape(v) != (n,)]
            if bad:
                raise WorkflowError(f"metric {key!r} returned no ({n},) values for {bad}")
            results[key] = {label: np.asarray(v, dtype=np.float64) for label, v in result.items()}
        return results

    def _save_state(self, fields: Mapping[str, Any]) -> None:
        # Written last in each step: it marks the step as complete for resume.
        self.run_dir.write_state(
            {
                **fields,
                "data_fingerprint": self._data_fingerprint(),
                "history_length": len(self.run_dir.history_dicts()),
                "random_streams": self.streams.state(),
            }
        )

    def _data_fingerprint(self) -> str:
        """SHA-256 of the analysis data space: keys, values and variances."""
        digest = hashlib.sha256()
        digest.update("\n".join(map(str, self.observations.keys)).encode())
        digest.update(np.ascontiguousarray(self.observations.values).tobytes())
        digest.update(np.ascontiguousarray(self.observations.variances).tobytes())
        if self._held_out:
            # Runs without held-out data keep the fingerprint they always had.
            for role, (_, held) in self._held_out.items():
                digest.update(role.encode())
                digest.update("\n".join(map(str, held.keys)).encode())
                digest.update(np.ascontiguousarray(held.values).tobytes())
                digest.update(np.ascontiguousarray(held.variances).tobytes())
        return digest.hexdigest()

    def _load(self) -> dict[str, Any]:
        """Read ``state.json``, restore the random streams and drop unfinished history."""
        saved = self.run_dir.read_state()
        # Checkpointed forecasts and perturbations are rows of the data space the run
        # started in. Transforms are refitted here, so a changed input file or library
        # could silently realign them with other coefficients (runs saved before this
        # check have no fingerprint).
        saved_fingerprint = saved.get("data_fingerprint")
        if saved_fingerprint is not None and saved_fingerprint != self._data_fingerprint():
            raise WorkflowError(
                "the observations, after data transforms, differ from those the run was "
                "started with (changed observation, field or mask files, or a library "
                "version that changes the transform); start a new run"
            )
        # Drop history written after the last complete checkpoint (a crash mid-step).
        self.run_dir.truncate_history(int(saved["history_length"]))
        self.streams.restore(saved["random_streams"])
        logger.info(
            "resuming",
            extra={
                "event": "resume",
                **{k: saved[k] for k in saved if k not in ("random_streams", "algorithm")},
            },
        )
        return saved

    def _finish(
        self,
        state: NDArray[np.float64],
        predicted: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        *,
        converged: bool | None,
        complete: bool,
        iteration: int | None = None,
    ) -> HistoryMatchResult:
        """Save the posterior and return the result.

        With a surrogate, ``iteration`` is the posterior iterate: once the run
        is complete it is forecast by the simulator, whose forecast ``posterior.npz`` holds
        (members failing that forecast are left out, as failed members always are).
        """
        extra: dict[str, NDArray[Any]] = {}
        if self.surrogate is not None and iteration is not None:
            if complete:
                state, predicted, member_ids = self._full_posterior(iteration, state, member_ids)
                extra = {
                    "forecast_source": np.array([SIMULATOR] * len(member_ids)),
                    "surrogate_id": np.array(""),
                }
            else:  # stopped before the end: the iterate's own (surrogate) forecast, labelled
                saved = upgrade_iterate(self.run_dir.load_arrays(f"iter_{iteration}"))
                columns = _columns_of(saved["member_ids"], member_ids)
                extra = {
                    "forecast_source": saved["forecast_source"][columns],
                    "surrogate_id": saved["surrogate_id"],
                }
        blocks = self.layout.split(state)
        self.run_dir.save_arrays(
            "posterior", {**blocks, "predicted": predicted, "member_ids": member_ids, **extra}
        )
        return HistoryMatchResult(
            posterior=state,
            member_ids=member_ids,
            history=tuple(self.run_dir.history()),
            converged=converged,
            complete=complete,
        )


def run_status(result: HistoryMatchResult) -> str:
    """Describe how a run ended, e.g. ``"converged"`` or ``"stopped (resume to continue)"``."""
    if result.stopped_by_cv:
        return "stopped by cross validation"
    if result.converged:
        return "converged"
    return "complete" if result.complete else "stopped (resume to continue)"


def _progress_of(algorithm: EnsembleAlgorithm) -> IterationProgress | None:
    # Optional: algorithms without ``progress`` still run, without a budget.
    progress = getattr(algorithm, "progress", None)
    return None if progress is None else progress()


def _values(output: NDArray[np.float64] | MemberFailure) -> NDArray[np.float64]:
    """Return a member's forecast; a failure here is a bug (a surrogate member never fails)."""
    if isinstance(output, MemberFailure):
        raise WorkflowError(f"member {output.member_id} has no forecast: {output.message}")
    return output


def _source_of(sources: NDArray[np.str_]) -> str:
    """Return a forecast's source from its members': theirs if all agree, else ``"mixed"``."""
    found = {str(s) for s in sources}
    return found.pop() if len(found) == 1 else "mixed"


def _columns_of(saved_ids: NDArray[np.int64], member_ids: NDArray[np.int64]) -> NDArray[np.intp]:
    """Column of each of ``member_ids`` among ``saved_ids``.

    Raises
    ------
    WorkflowError
        If a member is not among the saved ones (an inconsistent run directory).
    """
    position = {int(m): i for i, m in enumerate(saved_ids)}
    try:
        return np.array([position[int(m)] for m in member_ids], dtype=np.intp)
    except KeyError as error:
        raise WorkflowError(f"member {error.args[0]} is missing from the checkpoint") from None


def _members_of(
    array: NDArray[np.float64], saved_ids: NDArray[np.int64], member_ids: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the columns of ``array`` (saved with ``saved_ids``) for ``member_ids``.

    C-contiguous, like every ensemble array of the loop (see :func:`_keep`).
    """
    selected: NDArray[np.float64] = np.ascontiguousarray(
        array[:, _columns_of(saved_ids, member_ids)]
    )
    return selected


def _keep(array: NDArray[np.float64], ok: NDArray[np.bool_] | None = None) -> NDArray[np.float64]:
    """Return the members ``ok`` of an ensemble array (all without ``ok``), C-contiguous.

    Every ensemble array the loop holds goes through here (after dropping failed members
    and after loading a checkpoint), so uninterrupted and resumed runs compute on the same
    memory layout. ``array[:, ok]`` returns Fortran order, and ``np.savez`` keeps it; the
    values are the same, but on AVX-512 machines NumPy and OpenBLAS sum the two layouts
    in different orders, and a resumed run drifted from an uninterrupted one by rounding
    (2e-16, seen on CI). Resume must stay bit-identical.
    """
    selected: NDArray[np.float64] = np.ascontiguousarray(array if ok is None else array[:, ok])
    return selected


def _roles(
    raw: ObservationSet, analysis: ObservationSet, split: DataSplit | None
) -> tuple[NDArray[np.intp] | None, dict[str, tuple[NDArray[np.intp], ObservationSet]]]:
    """Rows of the history-matching data, and of each held-out role, in the analysis space.

    Returns ``None`` for the history-matching rows when every datum is assimilated (no
    slicing, so runs without held-out data are unchanged).

    Raises
    ------
    ValueError
        If a data transform removed or changed a CV or test datum: held-out data must
        reach the analysis space as observed.
    """
    if split is None:
        return None, {}
    held: dict[str, tuple[NDArray[np.intp], ObservationSet]] = {}
    for role, keys in (("cv", split.cv), ("test", split.test)):
        if not keys:
            continue
        rows = []
        for key in keys:
            try:
                row = analysis.index_of(key)
            except KeyError:
                raise ValueError(f"a data transform removes {role} datum {key}") from None
            before = raw.index_of(key)
            if (
                analysis.values[row] != raw.values[before]
                or analysis.variances[row] != raw.variances[before]
            ):
                raise ValueError(f"a data transform changes {role} datum {key}")
            rows.append(row)
        indices = np.array(rows, dtype=np.intp)
        held[role] = (indices, analysis.subset(indices))
    excluded: set[DataKey] = {*split.cv, *split.test, *split.unused}
    if not excluded:
        return None, held
    hm_rows = np.array(
        [i for i, key in enumerate(analysis.keys) if key not in excluded], dtype=np.intp
    )
    return hm_rows, held


def _mean_square_by_type(
    observations: ObservationSet, predicted: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    """Per-member mean-square normalized data mismatch ``mean(((d_obs - y) / sigma)^2)``.

    Over all rows (``"all"``) and per data type; ``predicted`` is ``(n_data, n_members)``.
    """
    sigma = np.sqrt(observations.variances)[:, None]
    squared = ((observations.values[:, None] - predicted) / sigma) ** 2
    result = {"all": squared.mean(axis=0)}
    rows: dict[str, list[int]] = {}
    for i, key in enumerate(observations.keys):
        rows.setdefault(key.datatype, []).append(i)
    for datatype, r in rows.items():
        result[datatype] = squared[r].mean(axis=0)
    return result


def _saved_data_mismatch(saved: Mapping[str, Any]) -> float:
    """Return a legacy state's accepted data mismatch (named "misfit").

    Raises
    ------
    WorkflowError
        If the state holds neither name.
    """
    for key in ("data_mismatch", "misfit"):
        if key in saved:
            return float(saved[key])
    raise WorkflowError("state.json holds no accepted data mismatch (data_mismatch or misfit)")
