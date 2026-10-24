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
from apex.core.diagnostics import DiagnosticsPlan, NoDiagnostics, RecordingDiagnostics
from apex.core.errors import SimulationError, WorkflowError
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
from apex.core.validation import StoppingRule
from apex.domain.data_split import DataSplit
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.results import HistoryMatchResult, IterationRecord
from apex.domain.state import StateLayout
from apex.storage.run_directory import RunDirectory
from apex.workflows.parameterization import PropertyMap, parameterized

logger = logging.getLogger(__name__)

#: The name of the objective the methods minimise, ``(d_pert - y)^T C_d^-1 (d_pert - y)``
#: per member, as a metric (it was "misfit" before 2026-09-30).
DATA_MISMATCH = "data_mismatch"

#: Names of the independent random streams spawned from the run seed. New names are only
#: ever appended: spawned children depend on their position, so existing streams (and
#: results) do not change.
STREAM_NAMES = ("prior", "perturbations", "analysis", "localization", "chop")

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

    A simulator with ``auxiliary_keys(keys)`` (a model-error simulator's residuals) is
    asked for those keys after the observations'; their values are saved with each saved
    ensemble (``auxiliary_predicted``, ``auxiliary_keys``) and never assimilated or scored.
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
                break
            forecasts = {request.name: self._forecast_request(request) for request in step.requests}
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
        if not request.required:
            ok, predicted, raw, sensitivity = self._forecast(
                request.state,
                request.member_ids,
                request.iteration,
                sensitivities=request.with_sensitivities,
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
            return Forecast(
                ok,
                self._assimilated(predicted),
                None if raw is None else keep(raw),
                None if sensitivity is None else self._assimilated(sensitivity),
            )
        outputs = self._run_members(
            request.state, request.member_ids, request.iteration, "mean model", auxiliary=False
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
                arrays["auxiliary_keys"] = np.array([str(k) for k in self.auxiliary_keys])
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
        for entry in self.run_dir.history_dicts():
            if not entry["accepted"]:
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
            }
        )
        for path in self.run_dir.checkpoints.glob("algorithm_state_*.npz"):
            if path.stem != name:
                path.unlink()

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
        self, state: NDArray[np.float64], member_ids: Sequence[int], iteration: int
    ) -> list[MemberInput]:
        blocks = self.layout.split(state)
        return [
            MemberInput(
                member_id=int(member_id),
                state={name: block[:, column].copy() for name, block in blocks.items()},
                work_dir=self.run_dir.simulation_dir(iteration, int(member_id))
                if bool(getattr(self.simulator, "needs_work_dir", False))
                else None,
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
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run the members through the executor, reporting progress as they finish.

        ``auxiliary``: also request the simulator's auxiliary keys (saved with the record;
        a required forecast such as the mean model's has none).
        """
        self.progress.forecast_started(iteration=iteration, label=label, n_members=len(member_ids))
        began = time.monotonic()
        keys = self.raw_observations.keys
        extra = self.auxiliary_keys if simulator is None and auxiliary else ()
        # A required forecast (the mean model's) keeps no raw rows: an executor applying the
        # transforms elsewhere sends none back for it.
        without_raw = getattr(self.executor, "without_raw_rows", None)
        with nullcontext() if auxiliary or without_raw is None else without_raw():
            outputs = self.executor.run_all(
                self.simulator if simulator is None else simulator,
                self._members(state, list(member_ids), iteration),
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
        if bool(getattr(self.simulator, "needs_work_dir", False)):
            # Retention ([output] keep_simulations): failed members are marked first, so a
            # policy that keeps them can tell them apart later.
            for out in outputs:
                if isinstance(out, MemberFailure):
                    self.run_dir.mark_failed(iteration, out.member_id, out.message)
            # an executor that keeps members' files on other hosts (a failed
            # one's marked there by its worker) removes them there, before their markers
            # here go.
            remove = getattr(self.executor, "remove_member_files", None)
            if remove is not None:
                remove(self.run_dir.simulations_to_prune(iteration, self.keep_simulations))
            self.run_dir.prune_simulations(iteration, self.keep_simulations)
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
            },
        )
        return outputs

    def _forecast(
        self,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        iteration: int,
        *,
        sensitivities: bool = False,
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
    ) -> HistoryMatchResult:
        blocks = self.layout.split(state)
        self.run_dir.save_arrays(
            "posterior", {**blocks, "predicted": predicted, "member_ids": member_ids}
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


class SensitivityRun:
    """A simulator whose ``run`` returns ``[values, jacobian.ravel()]``.

    It lets every executor (serial, process pools, retries, resource pools) run a
    sensitivity forecast unchanged; capabilities are read through to the simulator.
    """

    def __init__(self, simulator: Simulator, n_state: int) -> None:
        self.simulator = simulator
        self.n_state = n_state

    def output_size(self, n_keys: int) -> int:
        """Values per member: ``n_keys`` values and their ``n_keys x n_state`` Jacobian."""
        return n_keys * (1 + self.n_state)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (the simulator's capabilities)
        """Read capabilities (``needs_work_dir``, ``resources``, ...) from the simulator."""
        if name in ("simulator", "n_state"):  # during unpickling, before __init__
            raise AttributeError(name)
        return getattr(self.simulator, name)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Run the member; return its values and its flattened Jacobian, concatenated."""
        values, jacobian = self.simulator.run_with_sensitivities(member, keys)  # type: ignore[attr-defined]
        values, jacobian = np.asarray(values, float), np.asarray(jacobian, float)
        # A transposed or flattened Jacobian would have the right size and scramble dg/dm.
        if values.shape != (len(keys),) or jacobian.shape != (len(keys), self.n_state):
            raise SimulationError(
                f"member {member.member_id}: sensitivities must be values ({len(keys)},) and "
                f"a Jacobian ({len(keys)}, {self.n_state}), got {values.shape} and "
                f"{jacobian.shape}"
            )
        return np.concatenate([values, jacobian.ravel()])


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
