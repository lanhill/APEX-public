"""Prediction workflow: forecast ensembles of a history-matching run.

:class:`PredictionWorkflow` runs every member of each requested ensemble (the posterior,
the prior, saved iterates) and, in synthetic cases, the true model over the prediction
deck; it assembles their time series, checks that they reproduce the run's stored
forecasts over the history period, and writes per-time statistics and forecast scores.

- **Axis.** The vectors and times are fixed by the request when it is explicit (listed
  vectors and times). Otherwise one *pilot* member runs first, in this process, and its
  resolved axis (patterns matched, every report time) is kept in ``state.json``; every
  other member must return the same axis, or it fails.
- **Resume.** Each finished member is saved at once (``members/``). There is no
  randomness and no iteration, so a resumed prediction forecasts only the missing members.
- **History check.** Scalar data of the run are compared, member by member, with the
  forecast at the same (vector, time). A difference beyond ``rtol = 1e-6`` fails the
  prediction: its deck does not reproduce the history deck. Data transforms
  pass scalar data through unchanged, so the run's stored scalar forecasts are raw values.
"""

import dataclasses
import datetime as dt
import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.metrics.forecast import nominal_coverage
from apex.algorithms.surrogates.domain import check_members
from apex.algorithms.surrogates.gate import gate
from apex.core.errors import SimulationError, WorkflowError
from apex.core.forecast import ForecastScore
from apex.core.progress import GuardedProgress, IterationProgress, NullProgress, ProgressReporter
from apex.core.resources import ResourcePools
from apex.core.simulation import (
    Executor,
    MemberFailure,
    MemberInput,
    MemberSeries,
    SeriesRequest,
    Simulator,
    refuse_unsupported_stop,
)
from apex.core.surrogates import EarlyStartSurrogate, SurrogateModel
from apex.domain.observations import DataKey
from apex.domain.prediction import PERCENTILE_LABELS, EnsembleForecast, PredictionResult
from apex.domain.state import StateLayout
from apex.domain.surrogate import GateTolerances, QuantityKind, TrainingDomain
from apex.simulation.surrogate import SurrogateSimulator
from apex.storage.prediction_directory import MEAN_MODEL_ID, TRUTH_ID, PredictionDirectory

logger = logging.getLogger(__name__)

#: Relative tolerance of the history check: summary files store float32 (about 6e-8), and
#: the same deck must reproduce the same values, so anything larger is a different deck.
HISTORY_RTOL = 1e-6

#: Pilot members tried before giving up on resolving the axis.
MAX_PILOTS = 3


@dataclass(frozen=True, slots=True)
class StoredForecast:
    """A run's stored forecast of one ensemble at its scalar data, for the history check.

    Attributes
    ----------
    keys
        Scalar data keys ``(n_keys,)``: ``(ISO date, "<VECTOR>:<WELL>")``.
    values
        ``(n_keys, n_members)``; NaN where a datum was not forecast (unused data).
    member_ids
        ``(n_members,)``.
    """

    keys: tuple[DataKey, ...]
    values: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    #: ``(n_keys,)``: the observation errors' variances, for the gate of a surrogate's
    #: members; ``None``: such members are not checked.
    variances: NDArray[np.float64] | None = None


#: A member block's surrogate inputs from its state blocks.
InputsOf = Callable[[Mapping[str, NDArray[np.float64]]], dict[str, NDArray[np.float64]]]
#: Wraps a simulator as the prediction's stack does (its parameterizations).
Wrap = Callable[[Simulator], Simulator]

#: The sources of a prediction's members (``state.json`` ``member_sources``).
SIMULATOR, SURROGATE = "simulator", "surrogate"


@dataclass(frozen=True)
class PredictionSurrogate:
    """A prediction's surrogate, composed by the services.

    Attributes
    ----------
    simulator
        The model as a simulator of a member's properties (``SurrogateSimulator``).
    model, domain, identity
        The fitted model, its validated domain and its files' identity.
    executor
        Where the surrogate runs (local: serial, or process workers).
    inputs
        A member block's surrogate inputs, ``{variable: (size, n)}`` from the run's state
        blocks (its parameterizations applied).
    simulator_of
        Wraps a surrogate simulator as the prediction's stack wraps its simulator (the
        run's parameterizations).
    outside
        ``"stop"`` or ``"simulator"``, for members outside the domain.
    tolerances, kinds
        The model's gate, for the history check of its members.
    """

    simulator: SurrogateSimulator
    model: SurrogateModel
    domain: TrainingDomain
    identity: str
    executor: Executor
    inputs: InputsOf
    simulator_of: Wrap
    outside: Literal["stop", "simulator"]
    tolerances: GateTolerances
    kinds: Mapping[str, QuantityKind]

    @property
    def early_until(self) -> str | None:
        """The report date members run to first, for an early-start model."""
        return self.model.early_until if isinstance(self.model, EarlyStartSurrogate) else None


@dataclass(frozen=True, slots=True)
class EnsembleInput:
    """One ensemble to forecast.

    Attributes
    ----------
    name
        ``"posterior"``, ``"prior"`` or ``"iteration:<k>"``.
    state
        ``(n_state, n_members)``.
    member_ids
        The run's member ids, ``(n_members,)``.
    stored
        The run's forecast of this ensemble, for the history check; ``None`` to skip it.
    """

    name: str
    state: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    stored: StoredForecast | None = None


class _Forwarding:
    """What an executor reads of a simulator, forwarded from the one a series run wraps."""

    simulator: Simulator

    @property
    def max_parallel_runs(self) -> int | None:
        """The wrapped simulator's cap (e.g. Eclipse licences)."""
        cap: int | None = getattr(self.simulator, "max_parallel_runs", None)
        return cap

    @property
    def resources(self) -> dict[str, int]:
        """The wrapped simulator's machine-wide tokens per run."""
        needs: dict[str, int] = dict(getattr(self.simulator, "resources", None) or {})
        return needs

    @property
    def accepts_runner(self) -> bool:
        """Whether the wrapped simulator can run its program elsewhere."""
        return bool(getattr(self.simulator, "accepts_runner", False))

    @property
    def timeout_s(self) -> float | None:
        """The wrapped simulator's time limit per run (an executor's guard)."""
        limit: float | None = getattr(self.simulator, "timeout_s", None)
        return limit

    @property
    def executable(self) -> str | None:
        """The wrapped simulator's program, which an executor checks on its hosts."""
        return getattr(self.simulator, "executable", None)


@dataclass(frozen=True)
class _SeriesRun(_Forwarding):
    """A picklable ``Simulator`` view of one ensemble's series runs.

    Executors call ``run(member, keys)``; this runs the member's series, checks its axis,
    saves it to the prediction directory at once (for resume) and returns it flattened
    in ``keys`` order (vector-major).
    """

    simulator: Simulator
    request: SeriesRequest
    directory: PredictionDirectory
    ensemble: str
    #: The axis is known: members return it (a pilot's resolves it, ``_PilotSeries``).
    resolves_axis = False

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Forecast one member's series and save it."""
        values = series_values(self.simulator, member, self.request)
        self.directory.save_member(self.ensemble, member.member_id, values)
        return values.ravel()

    def save_series(self, member_id: int, series: MemberSeries) -> None:
        """Save a member's series run elsewhere, checked against the axis.

        Raises
        ------
        SimulationError
            If the series has another axis or non-finite values.
        """
        if series.vectors != self.request.vectors or series.times != self.request.times:
            raise SimulationError(f"member {member_id} returned another axis")
        values = np.asarray(series.values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise SimulationError(f"member {member_id} returned non-finite values")
        self.directory.save_member(self.ensemble, member_id, values)


@dataclass
class _PilotSeries(_Forwarding):
    """The pilot member's series run elsewhere: kept here once it comes back."""

    simulator: Simulator
    request: SeriesRequest
    series: MemberSeries | None = None
    #: Its series resolves the request's patterns and times (the run's own ``run_series``).
    resolves_axis = True

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Run the pilot here (an executor that does not run members elsewhere)."""
        self.series = self.simulator.run_series(member, self.request)  # type: ignore[attr-defined]
        return np.asarray(self.series.values, dtype=np.float64).ravel()

    def save_series(self, member_id: int, series: MemberSeries) -> None:
        """Keep the pilot's series (its axis is the prediction's)."""
        self.series = series


def _held(executor: Executor, simulator: Simulator) -> AbstractContextManager[None]:
    """Hold the simulator's tokens in the executor's pools, if it has both."""
    pools: ResourcePools | None = getattr(executor, "pools", None)
    needs = getattr(simulator, "resources", None) or {}
    return pools.hold(needs) if pools is not None and needs else nullcontext()


def is_explicit(request: SeriesRequest) -> bool:
    """Whether the request fixes the axis by itself (listed vectors and times)."""
    patterns = any(any(c in v for c in "*?[") for v in request.vectors)
    return request.times is not None and not patterns


def axis_keys(request: SeriesRequest) -> list[DataKey]:
    """Return the keys of a resolved request, vector-major: ``(time, vector)`` per vector."""
    assert request.times is not None
    return [DataKey(t, v) for v in request.vectors for t in request.times]


def series_values(
    simulator: Simulator, member: MemberInput, request: SeriesRequest
) -> NDArray[np.float64]:
    """Run one member at a resolved request and return ``(n_vectors, n_times)``.

    Raises
    ------
    SimulationError
        If the run fails or returns another axis.
    """
    assert request.times is not None
    run_series = getattr(simulator, "run_series", None)
    if run_series is None:
        values = simulator.run(member, axis_keys(request))
        result = np.asarray(values, dtype=np.float64).reshape(len(request.vectors), -1)
    else:
        series: MemberSeries = run_series(member, request)
        if series.vectors != request.vectors or series.times != request.times:
            raise SimulationError(
                f"member {member.member_id} returned another axis: "
                f"{len(series.vectors)} vectors x {len(series.times)} times, "
                f"expected {len(request.vectors)} x {len(request.times)}"
            )
        result = np.asarray(series.values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise SimulationError(f"member {member.member_id} returned non-finite values")
    return result


def time_label(label: str) -> str:
    """Normalise a time label: ISO dates in any ISO form become ``YYYY-MM-DD``.

    The run parses observation times with :meth:`datetime.date.fromisoformat` (which also
    reads ``20240311``), and summaries give ``date.isoformat()``; both must meet.
    """
    try:
        return dt.date.fromisoformat(label).isoformat()
    except ValueError:
        return label


def first_prediction_time(times: Sequence[str], observation_times: Sequence[str]) -> int:
    """Return the index of the first time after the history period (the last observation).

    When every time is an ISO date (as :meth:`datetime.date.fromisoformat` reads it), they
    are compared as dates. Otherwise the labels are matched by position: the history ends
    at the last axis time that is an observation time. 0 without observation times. The
    observation times include every data role, so a test hold-out in time is
    history, not prediction, here.
    """
    if not observation_times:
        return 0
    try:
        end = max(dt.date.fromisoformat(t) for t in observation_times)
        axis = [dt.date.fromisoformat(t) for t in times]
    except ValueError:
        observed = set(observation_times)
        positions = [i for i, t in enumerate(times) if t in observed]
        return positions[-1] + 1 if positions else 0
    return next((i for i, t in enumerate(axis) if t > end), len(axis))


def stream_key(ensemble: str, member_id: int) -> tuple[int, int]:
    """Return a prediction member's ``stream_key``.

    The first 31 bits of the SHA-256 of the ensemble's name (``posterior``, ``prior``,
    ``iteration:3``, ``posterior:mean``, ...), so a member's draws do not depend on which
    other ensembles the prediction forecasts, and ``member_id - TRUTH_ID`` (non-negative:
    the truth 0, a mean model 1, the run's members their original index + 2): member 3 of
    the prior and of the posterior draw differently.
    """
    name = int.from_bytes(hashlib.sha256(ensemble.encode()).digest()[:4], "big") >> 1
    return (name, int(member_id) - TRUTH_ID)


def ensemble_digest(state: NDArray[np.float64], member_ids: NDArray[np.int64]) -> str:
    """SHA-256 of an ensemble (its state and member ids), to detect a changed run."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(state, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(member_ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


class PredictionWorkflow:
    """Forecasts ensembles of a finished run over a prediction deck.

    Parameters
    ----------
    layout
        The run's state layout.
    simulator, executor
        The prediction's simulator and where its members run.
    directory
        The opened prediction directory.
    request
        Vectors (names or patterns) and times (``None``: every report time).
    observation_times
        Times of the run's observations: the history period ends at the last of them.
    scores
        Forecast scores by name, computed against the truth over the prediction period.
    include_mean_model
        Also forecast each ensemble's mean model (it must succeed).
    check_history
        Fail unless the forecast reproduces the run's stored forecasts.
    percentile_convention
        ``"percentile"`` (``p10`` is the 10th percentile) or ``"exceedance"``.
    min_success_fraction
        Fail unless at least this fraction of each ensemble's members succeeds.
    progress
        Receives the prediction's progress.
    surrogate
        Forecast the ensembles (not the truth) with a fitted model; the
        request must then be explicit (the services resolve it against the model's keys).
    controls
        A control scenario (decision 6): blocks ``{name: (n,)}`` every
        member is given besides its state, the truth too, for the deck template or a
        control-aware model; never part of the state.
    """

    def __init__(
        self,
        *,
        layout: StateLayout,
        simulator: Simulator,
        executor: Executor,
        directory: PredictionDirectory,
        request: SeriesRequest,
        observation_times: Sequence[str],
        scores: Mapping[str, ForecastScore],
        include_mean_model: bool = False,
        check_history: bool = True,
        percentile_convention: Literal["percentile", "exceedance"] = "percentile",
        min_success_fraction: float = 0.9,
        progress: ProgressReporter | None = None,
        surrogate: PredictionSurrogate | None = None,
        controls: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        if surrogate is not None and not is_explicit(request):
            raise ValueError("a prediction with [surrogate] needs its vectors and times listed")
        if not 0 < min_success_fraction <= 1:
            raise ValueError(f"min_success_fraction must be in (0, 1], got {min_success_fraction}")
        # A batch simulator (``ert``) runs members only through its executor and
        # reads their series from the run paths afterwards.
        self.batch = bool(getattr(simulator, "ert_batches", False)) and hasattr(
            simulator, "read_series"
        )
        if (
            not is_explicit(request)
            and getattr(simulator, "run_series", None) is None
            and not self.batch
        ):
            raise ValueError(
                "this simulator cannot return whole series: list every vector (no patterns) "
                "and the times explicitly"
            )
        self.layout = layout
        self.simulator = simulator
        self.executor = executor
        self.directory = directory
        self.request = request
        self.observation_times = tuple(observation_times)
        self.scores = dict(scores)
        self.include_mean_model = include_mean_model
        self.check_history = check_history
        self.percentiles = PERCENTILE_LABELS[percentile_convention]
        self.min_success_fraction = min_success_fraction
        self.progress: ProgressReporter = GuardedProgress(
            NullProgress() if progress is None else progress
        )
        self.surrogate = surrogate
        clashing = sorted(set(controls or {}) & set(layout.names))
        if clashing:
            raise ValueError(f"controls named as state variables: {clashing}")
        self.controls = {name: np.ascontiguousarray(v, dtype=np.float64)
                         for name, v in (controls or {}).items()}  # fmt: skip

    def run(
        self, ensembles: Sequence[EnsembleInput], truth: NDArray[np.float64] | None = None
    ) -> PredictionResult:
        """Forecast ``ensembles`` (and the true model ``(n_state,)``, if given).

        Raises
        ------
        WorkflowError
            If too many members fail, the mean model or the truth fails, no pilot member
            resolves the axis, or the history check fails.
        """
        groups: list[tuple[str, NDArray[np.float64], NDArray[np.int64], bool]] = []
        for ensemble in ensembles:
            groups.append((ensemble.name, ensemble.state, ensemble.member_ids, False))
            if self.include_mean_model:
                mean = ensemble.state.mean(axis=-1, keepdims=True)
                groups.append((f"{ensemble.name}:mean", mean, np.array([MEAN_MODEL_ID]), True))
        if truth is not None:
            groups.append(("truth", truth[:, None], np.array([TRUTH_ID]), True))
        self.progress.run_started(
            algorithm="prediction",
            progress=IterationProgress(
                0, len(groups), groups[0][0], unit="ensembles", upper_bound=False
            ),
        )
        self._check_sources(ensembles, truth)
        request = self._axis(groups)
        failures: dict[str, dict[int, str]] = {}
        for index, (name, state, member_ids, required) in enumerate(groups):
            # The truth is the scores' reference: always the simulator's.
            if self.surrogate is not None and name != "truth":
                failures[name] = self._forecast_surrogate_group(
                    name, state, member_ids, request, required
                )
            else:
                failures[name] = self._forecast_group(name, state, member_ids, request, required)
            following = groups[index + 1][0] if index + 1 < len(groups) else "done"
            self.progress.iteration_progress(
                IterationProgress(
                    index + 1, len(groups), following, unit="ensembles", upper_bound=False
                )
            )
        assert request.times is not None
        forecasts = {e.name: self._assemble(e, request, failures[e.name]) for e in ensembles}
        if self.check_history:
            for ensemble in ensembles:
                if ensemble.stored is not None:
                    try:
                        self._check_history(
                            forecasts[ensemble.name], ensemble.stored,
                            self._sources(ensemble.name),
                        )  # fmt: skip
                    except WorkflowError:
                        # Its members came from a deck that does not reproduce the run: a
                        # corrected configuration may replace them (PredictionDirectory.open).
                        self.directory.write_state(history_check="failed")
                        raise
        for forecast in forecasts.values():
            self.directory.save_forecast(forecast)
        truth_values = None
        if truth is not None:
            truth_values = self._load_required("truth", TRUTH_ID)
            self.directory.save_truth(truth_values, request.times, request.vectors)
        first = first_prediction_time(request.times, self.observation_times)
        self.directory.write_csv("statistics", self._statistics(forecasts))
        scores = self._scores(forecasts, truth_values, first)
        self.directory.write_json("scores", scores)
        self.directory.write_csv(
            "scores",
            [
                {"ensemble": e, "vector": v, "score": s, "value": value}
                for e, by_vector in scores.items()
                for v, by_score in by_vector.items()
                for s, value in by_score.items()
            ],
        )
        self.directory.write_state(complete=True)
        self.progress.run_finished(status="complete")
        logger.info("prediction complete", extra={"event": "prediction_done"})
        return PredictionResult(forecasts, truth_values, first, scores)

    def _check_sources(
        self, ensembles: Sequence[EnsembleInput], truth: NDArray[np.float64] | None
    ) -> None:
        """Refuse to resume from members of other ensembles than these (e.g. a continued run)."""
        sources = {e.name: ensemble_digest(e.state, e.member_ids) for e in ensembles}
        if truth is not None:
            sources["truth"] = ensemble_digest(truth[:, None], np.array([TRUTH_ID]))
        saved = self.directory.read_state().get("sources")
        if saved is None:
            self.directory.write_state(sources=sources)
            return
        changed = sorted(name for name in sources if saved.get(name) != sources[name])
        if changed:
            raise WorkflowError(
                f"the run's {', '.join(changed)} ensemble(s) changed since this prediction "
                f"started (e.g. the run continued), so its saved members are of other models; "
                f"delete {self.directory.path} or give the prediction another name"
            )

    # ------------------------------------------------------------------ the axis

    def _axis(
        self, groups: Sequence[tuple[str, NDArray[np.float64], NDArray[np.int64], bool]]
    ) -> SeriesRequest:
        """Resolve the vectors and times: from the request, ``state.json``, or a pilot run."""
        state = self.directory.read_state()
        if "vectors" in state:
            return SeriesRequest(tuple(state["vectors"]), tuple(state["times"]))
        if is_explicit(self.request):
            assert self.request.times is not None
            request = SeriesRequest(tuple(sorted(set(self.request.vectors))), self.request.times)
        else:
            request = self._pilot(groups)
        assert request.times is not None
        self.directory.write_state(vectors=list(request.vectors), times=list(request.times))
        logger.info(
            "prediction axis resolved",
            extra={
                "event": "prediction_axis",
                "n_vectors": len(request.vectors),
                "n_times": len(request.times),
            },
        )
        return request

    def _pilot(
        self, groups: Sequence[tuple[str, NDArray[np.float64], NDArray[np.int64], bool]]
    ) -> SeriesRequest:
        """Run members, in this process, until one resolves the axis; keep its series."""
        name, state, member_ids, _ = groups[0]
        errors = []
        for column in range(min(MAX_PILOTS, len(member_ids))):
            member = self._member(name, state, member_ids, column)
            if member.work_dir is not None:
                member.work_dir.mkdir(parents=True, exist_ok=True)
            self.progress.forecast_started(iteration=None, label=f"{name} (pilot)", n_members=1)
            began = time.monotonic()
            try:
                if self.batch:
                    series = self._batch_pilot(member)
                elif getattr(self.executor, "remote", False):
                    series = self._remote_pilot(member)
                else:
                    # The pilot runs here, not in the executor: it holds the same
                    # machine-wide tokens through the executor's pools.
                    refuse_unsupported_stop(self.simulator, member)  # never here, as the executors
                    with _held(self.executor, self.simulator):
                        series = self.simulator.run_series(member, self.request)  # type: ignore[attr-defined]
                if not np.all(np.isfinite(np.asarray(series.values, dtype=np.float64))):
                    raise SimulationError("the pilot's series has non-finite values")
            except SimulationError as error:
                self.progress.member_finished(member_id=member.member_id, ok=False)
                errors.append(f"member {member.member_id}: {error}")
                continue
            self.progress.member_finished(member_id=member.member_id, ok=True)
            self.progress.forecast_finished(n_ok=1, n_failed=0, seconds=time.monotonic() - began)
            self.directory.save_member(name, member.member_id, series.values)
            return SeriesRequest(series.vectors, series.times)
        raise WorkflowError(
            "no pilot member resolved the prediction's vectors and times: " + "; ".join(errors)
        )

    def _remote_pilot(self, member: MemberInput) -> MemberSeries:
        """Run the pilot through an executor that runs members on other hosts."""
        pilot = _PilotSeries(self.simulator, self.request)
        (outcome,) = self.executor.run_all(pilot, [member], [])
        if isinstance(outcome, MemberFailure):
            raise SimulationError(outcome.message)
        if pilot.series is None:
            raise SimulationError("the pilot's series did not come back")
        return pilot.series

    def _batch_pilot(self, member: MemberInput) -> MemberSeries:
        """Run the pilot as a one-member batch, and read its series from the run path."""
        (outcome,) = self.executor.run_all(self.simulator, [member], [])
        if isinstance(outcome, MemberFailure):
            raise SimulationError(outcome.message)
        assert member.work_dir is not None
        series: MemberSeries = self.simulator.read_series(member.work_dir, 0, self.request)  # type: ignore[attr-defined]
        return series

    def _save_batch_member(
        self,
        name: str,
        member: MemberInput,
        outcome: NDArray[np.float64] | MemberFailure,
        request: SeriesRequest,
    ) -> NDArray[np.float64] | MemberFailure:
        """Save a batch member's series (as ``_SeriesRun`` does for a member run alone)."""
        if isinstance(outcome, MemberFailure):
            return outcome
        values = np.asarray(outcome, dtype=np.float64).reshape(len(request.vectors), -1)
        if not np.all(np.isfinite(values)):
            return MemberFailure(member.member_id, "the series has non-finite values")
        self.directory.save_member(name, member.member_id, values)
        return outcome

    # ------------------------------------------------------------------ forecasting

    def _member(
        self,
        ensemble: str,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        column: int,
    ) -> MemberInput:
        blocks = self.layout.split(state)
        member_id = int(member_ids[column])
        return MemberInput(
            member_id=member_id,
            state={name: block[:, column].copy() for name, block in blocks.items()}
            | {name: values.copy() for name, values in self.controls.items()},
            work_dir=self.directory.simulation_dir(ensemble, member_id)
            if getattr(self.simulator, "needs_work_dir", False)
            else None,
            stream_key=stream_key(ensemble, member_id),
        )

    def _forecast_group(
        self,
        name: str,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        request: SeriesRequest,
        required: bool,
    ) -> dict[int, str]:
        """Forecast the members without a saved result; return the failures by member id."""
        pending = [
            column
            for column, member_id in enumerate(member_ids)
            if self.directory.load_member(name, int(member_id)) is None
        ]
        if not pending:
            return {}
        members = [self._member(name, state, member_ids, column) for column in pending]
        self.progress.forecast_started(iteration=None, label=name, n_members=len(members))
        began = time.monotonic()
        if self.batch:
            # One batch; each member's values at the axis keys (vector-major), saved after.
            outputs = self.executor.run_all(
                self.simulator,
                members,
                axis_keys(request),
                on_result=lambda member_id, ok: self.progress.member_finished(
                    member_id=member_id, ok=ok
                ),
            )
            outputs = [
                self._save_batch_member(name, member, out, request)
                for member, out in zip(members, outputs, strict=True)
            ]
        else:
            outputs = self.executor.run_all(
                _SeriesRun(self.simulator, request, self.directory, name),
                members,
                axis_keys(request),
                on_result=lambda member_id, ok: self.progress.member_finished(
                    member_id=member_id, ok=ok
                ),
            )
        seconds = time.monotonic() - began
        failures = {out.member_id: out.message for out in outputs if isinstance(out, MemberFailure)}
        self.progress.forecast_finished(
            n_ok=len(outputs) - len(failures), n_failed=len(failures), seconds=seconds
        )
        for member_id, message in failures.items():
            logger.warning(
                "member failed",
                extra={
                    "event": "member_failed",
                    "ensemble": name,
                    "member_id": member_id,
                    "reason": message,
                },
            )
        logger.info(
            "forecast done",
            extra={
                "event": "forecast_done",
                "ensemble": name,
                "n_ok": len(outputs) - len(failures),
                "n_failed": len(failures),
                "seconds": round(seconds, 3),
            },
        )
        if required and failures:
            raise WorkflowError(f"{name} forecast failed: {next(iter(failures.values()))}")
        return failures

    def _sources(self, name: str) -> dict[int, str]:
        """Return the members' sources of an ensemble (empty: all the simulator's)."""
        saved = self.directory.read_state().get("member_sources", {}).get(name, {})
        return {int(member): str(source) for member, source in saved.items()}

    def _record_sources(self, name: str, sources: Mapping[int, str]) -> None:
        state = self.directory.read_state().get("member_sources", {})
        merged = {**state.get(name, {}), **{str(m): s for m, s in sources.items()}}
        self.directory.write_state(member_sources={**state, name: merged})

    def _forecast_surrogate_group(
        self,
        name: str,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        request: SeriesRequest,
        required: bool,
    ) -> dict[int, str]:
        """Forecast the members without a saved result by the surrogate.

        An early-start model's members first run on the simulator up to its early date
        (through the prediction's executor, work folders ``<ensemble>:early``), asked only
        for its early keys. Members outside the model's domain stop the prediction
        (``outside = "stop"``) or are run whole by the simulator; a member whose early run
        fails is a failed member. The surrogate failing a member stops the prediction (a
        dropped member would bias the forecast silently). Returns the failures by member id.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        assert request.times is not None
        pending = [
            column
            for column, member_id in enumerate(member_ids)
            if self.directory.load_member(name, int(member_id)) is None
        ]
        if not pending:
            return {}
        failures: dict[int, str] = {}
        sources: dict[int, str] = {}
        early_failed: list[int] = []
        early_until = surrogate.early_until
        early: dict[int, NDArray[np.float64]] = {}
        if early_until is not None:
            assert isinstance(surrogate.model, EarlyStartSurrogate)
            members = []
            for column in pending:
                member = self._member(name, state, member_ids, column)
                work_dir = (
                    self.directory.simulation_dir(f"{name}:early", member.member_id)
                    if member.work_dir is not None
                    else None
                )
                members.append(dataclasses.replace(member, stop_at=early_until, work_dir=work_dir))
            self.progress.forecast_started(
                iteration=None, label=f"{name} (early)", n_members=len(members)
            )
            began = time.monotonic()
            outputs = self.executor.run_all(
                self.simulator, members, list(surrogate.model.early_keys()),
                on_result=lambda member_id, ok: self.progress.member_finished(
                    member_id=member_id, ok=ok
                ),
            )  # fmt: skip
            for member, out in zip(members, outputs, strict=True):
                if isinstance(out, MemberFailure):
                    failures[member.member_id] = out.message
                else:
                    early[member.member_id] = np.ascontiguousarray(out, dtype=np.float64)
            self.progress.forecast_finished(
                n_ok=len(early), n_failed=len(failures), seconds=time.monotonic() - began
            )
            logger.info(
                "early runs done",
                extra={"event": "forecast_done", "ensemble": f"{name} (early)",
                       "n_ok": len(early), "n_failed": len(failures),
                       "seconds": round(time.monotonic() - began, 3)},
            )  # fmt: skip
            pending = [c for c in pending if int(member_ids[c]) not in failures]
        early_failed = sorted(failures)
        if pending:
            ids = [int(member_ids[c]) for c in pending]
            blocks = self.layout.split(np.ascontiguousarray(state[:, pending]))
            # The scenario's controls too, as each member is given them (_member).
            blocks |= {name: np.ascontiguousarray(np.tile(v[:, None], (1, len(pending))))
                       for name, v in self.controls.items()}  # fmt: skip
            inputs = surrogate.inputs(blocks)
            found = check_members(
                surrogate.domain, surrogate.model, inputs, ids, axis_keys(request),
                coarse=None if early_until is None
                else np.ascontiguousarray(np.column_stack([early[m] for m in ids])),
            )  # fmt: skip
            if (
                found.unknown_keys
                or found.times_outside
                or (found.members and surrogate.outside == "stop")
            ):
                logger.error(
                    "members outside the surrogate's domain",
                    extra={"event": "surrogate_outside", "ensemble": name,
                           "members": [int(m) for m in found.members],
                           "details": found.message()},
                )  # fmt: skip
                found.raise_if_outside()
            outside = {int(m) for m in found.members}
            inside = [c for c in pending if int(member_ids[c]) not in outside]
            rest = [c for c in pending if int(member_ids[c]) in outside]
            if inside:
                self._surrogate_members(name, state, member_ids, inside, request, early)
            if rest:
                logger.warning(
                    "members outside the surrogate's domain run by the simulator",
                    extra={"event": "surrogate_outside", "ensemble": name,
                           "members": sorted(outside)},
                )  # fmt: skip
                failures.update(self._forecast_columns(name, state, member_ids, rest, request))
                sources.update({int(member_ids[c]): SIMULATOR for c in rest})
        self._record_sources(name, sources)
        for member_id in early_failed:
            logger.warning(
                "member failed",
                extra={"event": "member_failed", "ensemble": name, "member_id": member_id,
                       "reason": failures[member_id]},
            )  # fmt: skip
        if required and failures:
            raise WorkflowError(f"{name} forecast failed: {next(iter(failures.values()))}")
        return failures

    def _surrogate_members(
        self,
        name: str,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        columns: Sequence[int],
        request: SeriesRequest,
        early: Mapping[int, NDArray[np.float64]],
    ) -> None:
        """Predict and save the members in ``columns`` by the surrogate, on its executor."""
        surrogate = self.surrogate
        assert surrogate is not None
        assert request.times is not None
        model = surrogate.simulator
        if surrogate.early_until is not None:
            model = model.with_early({int(member_ids[c]): early[int(member_ids[c])]
                                      for c in columns})  # fmt: skip
        members = [
            dataclasses.replace(self._member(name, state, member_ids, c), work_dir=None)
            for c in columns
        ]
        self.progress.forecast_started(
            iteration=None, label=f"{name} (surrogate)", n_members=len(members)
        )
        began = time.monotonic()
        outputs = surrogate.executor.run_all(
            surrogate.simulator_of(model), members, axis_keys(request),
            on_result=lambda member_id, ok: self.progress.member_finished(
                member_id=member_id, ok=ok
            ),
        )  # fmt: skip
        failed = [out for out in outputs if isinstance(out, MemberFailure)]
        self.progress.forecast_finished(
            n_ok=len(outputs) - len(failed), n_failed=len(failed),
            seconds=time.monotonic() - began,
        )  # fmt: skip
        if failed:
            # Loud: dropping members the surrogate cannot predict would bias the forecast.
            raise WorkflowError(
                f"{name}: the surrogate failed member {failed[0].member_id} "
                f"({failed[0].message}); the prediction stops, its saved members kept"
            )
        arrays = [np.asarray(out, dtype=np.float64) for out in outputs]
        unstable = [m.member_id for m, a in zip(members, arrays, strict=True)
                    if not np.all(np.isfinite(a))]  # fmt: skip
        if unstable:
            # As the simulator's series with non-finite values fail; loud, never saved.
            raise WorkflowError(
                f"{name}: the surrogate gave non-finite values for member {unstable[0]} "
                f"({len(unstable)} member(s)); the prediction stops, its saved members kept"
            )
        # The sources before any member is saved: a prediction killed while it saves resumes
        # knowing every saved member's (a recorded member not yet saved is forecast again).
        self._record_sources(name, {m.member_id: SURROGATE for m in members})
        for member, values in zip(members, arrays, strict=True):
            shaped = values.reshape(len(request.vectors), len(request.times))
            self.directory.save_member(name, member.member_id, np.ascontiguousarray(shaped))
        logger.info(
            "surrogate forecast",
            extra={"event": "forecast_done", "ensemble": f"{name} (surrogate)",
                   "n_ok": len(outputs), "n_failed": 0, "identity": surrogate.identity,
                   "seconds": round(time.monotonic() - began, 3)},
        )  # fmt: skip

    def _forecast_columns(
        self,
        name: str,
        state: NDArray[np.float64],
        member_ids: NDArray[np.int64],
        columns: Sequence[int],
        request: SeriesRequest,
    ) -> dict[int, str]:
        """Forecast some members whole by the simulator (:meth:`_forecast_group`'s path)."""
        selected = list(columns)
        return self._forecast_group(
            name, np.ascontiguousarray(state[:, selected]), member_ids[selected], request,
            required=False,
        )  # fmt: skip

    def _load_required(self, name: str, member_id: int) -> NDArray[np.float64]:
        values = self.directory.load_member(name, member_id)
        if values is None:
            raise WorkflowError(f"{name} has no forecast")
        return values

    def _assemble(
        self, ensemble: EnsembleInput, request: SeriesRequest, failures: Mapping[int, str]
    ) -> EnsembleForecast:
        """Stack the saved members; apply the failure policy."""
        assert request.times is not None
        values, ok_ids, failed = [], [], []
        for member_id in ensemble.member_ids:
            saved = self.directory.load_member(ensemble.name, int(member_id))
            if saved is None:
                failed.append(int(member_id))
            else:
                values.append(saved)
                ok_ids.append(int(member_id))
        n_total = len(ensemble.member_ids)
        if len(ok_ids) < max(2, self.min_success_fraction * n_total):
            first = next(iter(failures.values()), "n/a")
            raise WorkflowError(
                f"{ensemble.name}: only {len(ok_ids)} of {n_total} members forecast "
                f"successfully (minimum fraction {self.min_success_fraction}); first failure: "
                f"{first}"
            )
        mean_model = None
        if self.include_mean_model:
            mean_model = self._load_required(f"{ensemble.name}:mean", MEAN_MODEL_ID)
        return EnsembleForecast(
            name=ensemble.name,
            times=request.times,
            vectors=request.vectors,
            # (n_vectors, n_times, n_members): the ensemble axis last.
            values=np.ascontiguousarray(np.stack(values, axis=-1)),
            member_ids=np.array(ok_ids, dtype=np.int64),
            failed_member_ids=np.array(failed, dtype=np.int64),
            mean_model=mean_model,
        )

    # ------------------------------------------------------------------ checks and results

    def _check_history(
        self,
        forecast: EnsembleForecast,
        stored: StoredForecast,
        sources: Mapping[int, str] | None = None,
    ) -> None:
        """Fail unless the forecast reproduces the run's stored scalar forecasts.

        The simulator's members exactly. The surrogate's (``sources``) within
        the model's gate, against the run's observation errors; an early-start model's early
        keys, which the simulator gave, exactly.
        """
        sources = sources or {}
        surrogate_ids = {m for m, source in sources.items() if source == SURROGATE}
        time_index = {time_label(t): i for i, t in enumerate(forecast.times)}
        vector_index = {v: i for i, v in enumerate(forecast.vectors)}
        column = {int(m): j for j, m in enumerate(forecast.member_ids)}
        members = [
            (j, column[int(m)])
            for j, m in enumerate(stored.member_ids)
            if int(m) in column and int(m) not in surrogate_ids
        ]
        compared, mismatches = 0, []
        if surrogate_ids:
            compared += self._check_history_gate(forecast, stored, surrogate_ids)
        for k, key in enumerate(stored.keys):
            when = time_label(key.time)
            if when not in time_index or key.datatype not in vector_index:
                continue
            row = stored.values[k]
            if not np.isfinite(row).any():
                continue  # a datum the run did not forecast
            scale = float(np.nanmax(np.abs(row)))
            for j_stored, j_forecast in members:
                expected = row[j_stored]
                if not np.isfinite(expected):
                    continue
                value = forecast.values[vector_index[key.datatype], time_index[when], j_forecast]
                compared += 1
                if abs(value - expected) > HISTORY_RTOL * max(abs(expected), scale, 1e-300):
                    mismatches.append(
                        f"member {int(stored.member_ids[j_stored])} {key}: forecast {value:.6g}, "
                        f"run {expected:.6g}"
                    )
        logger.info(
            "history check",
            extra={
                "event": "history_check",
                "ensemble": forecast.name,
                "compared": compared,
                "mismatches": len(mismatches),
            },
        )
        if compared == 0 and np.isfinite(stored.values).any():
            raise WorkflowError(
                f"{forecast.name}: the history check found nothing to compare: the forecast "
                "shares no (vector, time) with the run's data. Add the observed vectors "
                "(e.g. WOPR:*) to [prediction.outputs] vectors and keep the history report "
                "times, or set check_history = false"
            )
        if mismatches:
            raise WorkflowError(
                f"{forecast.name}: the prediction deck does not reproduce the history match "
                f"({len(mismatches)} of {compared} values differ by more than rtol "
                f"{HISTORY_RTOL}), e.g. " + "; ".join(mismatches[:5]) + ". Check that the "
                "prediction deck has the same history schedule and template variables (the "
                "check also assumes the simulator reproduces its own results, e.g. the same "
                "version and threads), or set check_history = false (e.g. for a restart deck)."
            )

    def _check_history_gate(
        self, forecast: EnsembleForecast, stored: StoredForecast, surrogate_ids: set[int]
    ) -> int:
        """Gate the surrogate's members' history against the run's stored forecasts.

        Returns the number of values compared.

        Keys the simulator gave (an early-start model's early keys) must match exactly;
        the others pass the model's gate (``sigma_ratio`` of the run's observation errors),
        else the prediction fails, as a deck that does not reproduce the run.
        """
        surrogate = self.surrogate
        assert surrogate is not None
        time_index = {time_label(t): i for i, t in enumerate(forecast.times)}
        vector_index = {v: i for i, v in enumerate(forecast.vectors)}
        column = {int(m): j for j, m in enumerate(forecast.member_ids)}
        pairs = [
            (j, column[int(m)])
            for j, m in enumerate(stored.member_ids)
            if int(m) in surrogate_ids and int(m) in column
        ]
        if not pairs:
            return 0
        model = surrogate.model
        exact = set(model.early_keys()) if isinstance(model, EarlyStartSurrogate) else set()
        rows, keys, mismatches = [], [], []
        compared = 0
        for k, key in enumerate(stored.keys):
            when = time_label(key.time)
            if when not in time_index or key.datatype not in vector_index:
                continue
            values = [
                forecast.values[vector_index[key.datatype], time_index[when], j_forecast]
                for _, j_forecast in pairs
            ]
            expected = [stored.values[k, j_stored] for j_stored, _ in pairs]
            if not np.isfinite(expected).all():
                continue
            compared += len(pairs)
            if DataKey(key.time, key.datatype) in exact:
                scale = max(float(np.max(np.abs(expected))), 1e-300)
                mismatches += [
                    f"{key}: {v:.6g} against {e:.6g}"
                    for v, e in zip(values, expected, strict=True)
                    if abs(v - e) > HISTORY_RTOL * max(abs(e), scale)
                ]
                continue
            rows.append((values, expected, k))
            keys.append(key)
        if mismatches:
            raise WorkflowError(
                f"{forecast.name}: the simulator's early runs do not reproduce the history "
                f"match ({len(mismatches)} values), e.g. " + "; ".join(mismatches[:5])
            )
        if not rows:
            return compared
        if stored.variances is None:
            raise WorkflowError(
                f"{forecast.name}: the surrogate's members cannot be checked within its gate "
                "without the run's observation errors; set check_history = false"
            )
        try:
            verdict = gate(
                np.ascontiguousarray([r[0] for r in rows], dtype=np.float64),
                np.ascontiguousarray([r[1] for r in rows], dtype=np.float64),
                keys,
                [int(stored.member_ids[j_stored]) for j_stored, _ in pairs],
                tolerances=surrogate.tolerances,
                variances=np.ascontiguousarray(stored.variances[[r[2] for r in rows]]),
                kinds=surrogate.kinds,
            )
        except ValueError as error:
            raise WorkflowError(
                f"{forecast.name}: the surrogate's members cannot be gated: {error}"
            ) from error
        logger.info(
            "history check within the surrogate's gate",
            extra={"event": "history_check_surrogate", "ensemble": forecast.name,
                   "members": len(pairs), "keys": len(keys), "passed": verdict.passed,
                   "failures": verdict.failures()},
        )  # fmt: skip
        if not verdict.passed:
            raise WorkflowError(
                f"{forecast.name}: the surrogate's members do not reproduce the history match "
                f"within its gate ({'; '.join(verdict.failures())}): the surrogate does not "
                "hold for this run's posterior; rebuild it, or set check_history = false"
            )
        return compared

    def _statistics(self, forecasts: Mapping[str, EnsembleForecast]) -> list[dict[str, object]]:
        """Per ensemble, vector and time: members, mean, std and percentiles."""
        rows: list[dict[str, object]] = []
        levels = [q for _, q in self.percentiles]
        for forecast in forecasts.values():
            n = forecast.values.shape[-1]
            if n < 10:
                logger.warning(
                    "%s has %d members: its 10th and 90th percentiles are poorly determined",
                    forecast.name,
                    n,
                    extra={"event": "few_members", "ensemble": forecast.name, "n": n},
                )
            quantiles = np.percentile(forecast.values, levels, axis=-1)
            mean = forecast.values.mean(axis=-1)
            std = forecast.values.std(axis=-1, ddof=1)
            for v, vector in enumerate(forecast.vectors):
                for t, when in enumerate(forecast.times):
                    row: dict[str, object] = {
                        "ensemble": forecast.name,
                        "vector": vector,
                        "time": when,
                        "n_members": n,
                        "mean": float(mean[v, t]),
                        "std": float(std[v, t]),
                    }
                    for (label, _), q in zip(self.percentiles, quantiles[:, v, t], strict=True):
                        row[label] = float(q)
                    rows.append(row)
        return rows

    def _scores(
        self,
        forecasts: Mapping[str, EnsembleForecast],
        truth: NDArray[np.float64] | None,
        first: int,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Forecast scores against the truth over the prediction period."""
        if truth is None or not self.scores:
            return {}
        some = next(iter(forecasts.values()))
        if first >= len(some.times):
            logger.warning(
                "no forecast scores: the prediction has no time after the history period",
                extra={"event": "no_prediction_period"},
            )
            return {}
        result: dict[str, dict[str, dict[str, float]]] = {}
        for name, forecast in forecasts.items():
            result[name] = {}
            for v, vector in enumerate(forecast.vectors):
                values = {
                    score_name: score.score(forecast.values[v, first:, :], truth[v, first:])
                    for score_name, score in self.scores.items()
                }
                if "coverage_p10_p90" in values:
                    # A calibrated ensemble of n members covers about 0.8 (n-1)/(n+1), not
                    # 0.8: reported alongside, to read the coverage against.
                    values["coverage_p10_p90_nominal"] = nominal_coverage(forecast.values.shape[-1])
                result[name][vector] = values
        return result
