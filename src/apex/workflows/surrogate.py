"""The surrogate workflow: full runs of a design, then fit, domain and gate.

Two steps, each resumable:

1. :func:`run_members` runs a design's members through the configured simulator stack and
   executor, as a history-matching forecast does. Each finished member is saved at once
   (``members/member_<id>.npz``, written by the worker that ran it where it runs here) and
   recorded in ``snapshots.jsonl`` with its outcome, so a resume runs only the members
   without either. Failed members are recorded, never silently dropped; they are journalled
   when the batch ends, so a failure in a batch interrupted before then runs again. A run budget
   (``max_runs``) stops the step early, resumably.
2. :func:`fit_and_gate` fits the surrogate on the training runs with the build's stream
   ``surrogate`` (the only randomness of a fit), builds its validated domain from its
   coordinates of the training and held-out inputs (core's, ``algorithms/surrogates``), and
   gates it on the held-out runs, which it never saw.

A fit draws only from its stream and every input is read back from the build's files, so a
resumed build writes model files identical, byte for byte, to an uninterrupted one's.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.surrogates.domain import training_domain
from apex.algorithms.surrogates.gate import gate
from apex.core.errors import SimulationError, WorkflowError
from apex.core.progress import GuardedProgress, NullProgress, ProgressReporter
from apex.core.simulation import Executor, MemberFailure, MemberInput, Simulator
from apex.core.surrogates import SurrogateModel
from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    DomainExtent,
    GateResult,
    GateTolerances,
    QuantityKind,
    SnapshotSet,
    TrainingDomain,
)
from apex.storage.surrogate_directory import MemberRecord, SurrogateDirectory

logger = logging.getLogger(__name__)


class _Journalled:
    """A simulator whose finished members are saved at once in the build's directory.

    Picklable (a process executor ships it to its workers); every capability an executor
    reads (``needs_work_dir``, ``max_parallel_runs``, ``resources``, ...) is the wrapped
    simulator's.
    """

    def __init__(self, simulator: Simulator, directory: SurrogateDirectory) -> None:
        self.simulator = simulator
        self.directory = directory

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (the simulator's capabilities)
        """Read capabilities from the wrapped simulator."""
        if name in ("simulator", "directory"):
            raise AttributeError(name)  # while unpickling
        return getattr(self.simulator, name)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Run the member, save its outputs, and return them."""
        began = time.monotonic()
        values = np.asarray(self.simulator.run(member, keys), dtype=np.float64).reshape(-1)
        if values.shape != (len(keys),) or not np.all(np.isfinite(values)):
            raise SimulationError(
                f"member {member.member_id} returned {values.shape[0]} values for "
                f"{len(keys)} keys, or values that are not finite"
            )
        self.directory.save_member(member.member_id, values, time.monotonic() - began)
        return values


@dataclass(frozen=True)
class MembersOutcome:
    """What a design's members gave.

    Attributes
    ----------
    values
        ``{member: (n_keys,)}`` of every finished member.
    seconds
        ``{member: run time}`` where known.
    failed
        ``{member: message}`` of every failed member.
    pending
        Members not run yet (a run budget stopped the step).
    """

    values: Mapping[int, NDArray[np.float64]]
    seconds: Mapping[int, float]
    failed: Mapping[int, str]
    pending: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Whether every member ran."""
        return not self.pending


def run_members(
    directory: SurrogateDirectory,
    simulator: Simulator,
    executor: Executor,
    members: Sequence[tuple[int, Mapping[str, NDArray[np.float64]]]],
    keys: Sequence[DataKey],
    *,
    label: str,
    keep_simulations: str = "failed",
    max_runs: int | None = None,
    progress: ProgressReporter | None = None,
) -> MembersOutcome:
    """Run the members not finished yet; return every member's outcome.

    Parameters
    ----------
    members
        ``(member_id, {variable: (size,)})`` in state space, in the order they run.
    max_runs
        Run at most this many members now (``None``: all); the others stay pending.
    """
    reporter: ProgressReporter = GuardedProgress(NullProgress() if progress is None else progress)
    journal = directory.journal()
    values: dict[int, NDArray[np.float64]] = {}
    seconds: dict[int, float] = {}
    failed: dict[int, str] = {}
    missing: list[tuple[int, Mapping[str, NDArray[np.float64]]]] = []
    for member_id, state in members:
        record = journal.get(member_id)
        saved = directory.load_member(member_id)
        if saved is not None:
            values[member_id], seconds[member_id] = saved
            if record is None:  # saved by its worker, not journalled before the stop
                directory.record([MemberRecord(member_id, True, saved[1])])
        elif record is not None and not record.ok:
            failed[member_id] = record.message or "failed"
        else:
            missing.append((member_id, state))
    now = missing if max_runs is None else missing[: max(max_runs, 0)]
    pending = tuple(m for m, _ in missing[len(now) :])
    if now:
        wrapped = _Journalled(simulator, directory)
        needs_dir = bool(getattr(simulator, "needs_work_dir", False))
        inputs = [
            MemberInput(
                member_id=member_id,
                state={k: np.ascontiguousarray(v, dtype=np.float64) for k, v in state.items()},
                work_dir=directory.simulation_dir(member_id) if needs_dir else None,
            )
            for member_id, state in now
        ]
        for member in inputs:
            if member.work_dir is not None:
                member.work_dir.mkdir(parents=True, exist_ok=True)
        reporter.forecast_started(iteration=None, label=label, n_members=len(inputs))
        began = time.monotonic()
        outputs = executor.run_all(
            wrapped,
            inputs,
            list(keys),
            on_result=lambda member_id, ok: reporter.member_finished(member_id=member_id, ok=ok),
        )
        records = []
        for member, output in zip(inputs, outputs, strict=True):
            member_id = member.member_id
            if isinstance(output, MemberFailure):
                failed[member_id] = output.message
                directory.mark_failed(member_id, output.message)
                records.append(MemberRecord(member_id, False, None, output.message))
                continue
            saved = directory.load_member(member_id)
            if saved is None:
                # Run elsewhere (an executor running members whole on other hosts): the
                # values came back here, so they are saved here.
                result = np.asarray(output, dtype=np.float64).reshape(-1)
                directory.save_member(member_id, result, float("nan"))
                saved = (result, float("nan"))
            values[member_id], seconds[member_id] = saved
            records.append(MemberRecord(member_id, True, saved[1]))
        directory.record(records)
        directory.prune_simulations(keep_simulations)
        n_failed = sum(isinstance(o, MemberFailure) for o in outputs)
        reporter.forecast_finished(
            n_ok=len(outputs) - n_failed, n_failed=n_failed, seconds=time.monotonic() - began
        )
        logger.info(
            "design members run",
            extra={"event": "surrogate_members", "label": label, "ran": len(inputs),
                   "failed": n_failed, "pending": len(pending)},
        )  # fmt: skip
    return MembersOutcome(values, seconds, failed, pending)


def check_success(label: str, total: int, failed: int, min_success_fraction: float) -> None:
    """Refuse a set with too few successful runs.

    Raises
    ------
    WorkflowError
        If fewer than ``min_success_fraction`` of the set's ``total`` runs succeeded.
    """
    if total and (total - failed) / total < min_success_fraction:
        raise WorkflowError(
            f"{failed} of the {total} {label} runs failed: fewer than "
            f"{min_success_fraction:.0%} succeeded (see snapshots.jsonl and the work "
            "directories marked APEX_FAILED)"
        )


@dataclass(frozen=True)
class FitOutcome:
    """A fitted and gated surrogate.

    Attributes
    ----------
    model
        The fitted plugin.
    domain
        Its validated domain.
    gate
        Its verdict on the held-out runs.
    predicted
        Its prediction of the held-out runs ``(n_keys, n_held_out)``, C-contiguous.
    stream_state
        The fit's generator state before the fit.
    fit_seconds, prediction_seconds
        The fit's time, and one member's prediction time (averaged over the held-out runs).
    """

    model: SurrogateModel
    domain: TrainingDomain
    gate: GateResult
    predicted: NDArray[np.float64]
    stream_state: dict[str, Any]
    fit_seconds: float
    prediction_seconds: float


def fit_and_gate(
    model: SurrogateModel,
    training: SnapshotSet,
    held_out: SnapshotSet,
    rng: np.random.Generator,
    *,
    tolerances: GateTolerances,
    variances: NDArray[np.float64] | None,
    kinds: Mapping[str, QuantityKind] | None = None,
    extent: DomainExtent = "box",
    failed: Sequence[int] = (),
    clock: Callable[[], float] = time.perf_counter,
) -> FitOutcome:
    """Fit ``model`` on ``training``, bound its domain, and gate it on ``held_out``.

    ``variances`` ``(n_keys,)`` gate against observation errors (``sigma_ratio``); ``None``
    gates by relative errors.

    Raises
    ------
    WorkflowError
        If the model does not answer the snapshots' keys after its fit.
    ValueError
        If the plugin cannot fit the snapshots, or the domain cannot be built.
    """
    state = dict(rng.bit_generator.state)
    began = clock()
    model.fit(training, rng)
    fit_seconds = clock() - began
    if tuple(model.keys) != training.keys:
        raise WorkflowError("the surrogate answers other keys than it was fitted on")
    domain = training_domain(model, training.inputs, held_out.inputs, extent=extent)
    began = clock()
    predicted = np.ascontiguousarray(model.predict(held_out.inputs, training.keys))
    prediction_seconds = (clock() - began) / max(held_out.n_runs, 1)
    verdict = gate(
        predicted,
        held_out.outputs,
        training.keys,
        held_out.run_ids,
        tolerances=tolerances,
        variances=variances,
        kinds=kinds,
        failed=failed,
    )
    logger.info(
        "surrogate gated",
        extra={"event": "surrogate_gate", "passed": verdict.passed,
               "failures": verdict.failures()},
    )  # fmt: skip
    return FitOutcome(model, domain, verdict, predicted, state, fit_seconds, prediction_seconds)
