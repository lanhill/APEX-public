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
from apex.core.surrogates import (
    CoarseCorrectedSurrogate,
    EarlyStartSurrogate,
    FieldSurrogate,
    SurrogateModel,
)
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


def column_objectives(
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    outputs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the objective of each column of ``outputs`` ``(n_keys, n)``: ``(n_obj, n)``.

    One column at a time, as an optimization's journal evaluates them: a reduction over a
    batch may round in another order than over one column.
    """
    columns = [
        np.asarray(
            objective(np.ascontiguousarray(outputs[:, j : j + 1])), dtype=np.float64
        ).reshape(-1)
        for j in range(outputs.shape[1])
    ]
    if not columns:
        return np.zeros((1, 0))
    return np.ascontiguousarray(np.stack(columns, axis=-1))


def early_data(model: EarlyStartSurrogate, snapshots: SnapshotSet) -> NDArray[np.float64]:
    """Return runs' early data: their values at the model's early keys, ``(n_early, n)``.

    Raises
    ------
    WorkflowError
        If an early key is not among the snapshots' keys.
    """
    index = {key: row for row, key in enumerate(snapshots.keys)}
    missing = [k for k in model.early_keys() if k not in index]
    if missing:
        raise WorkflowError(
            f"the early-start surrogate's early key {missing[0]} is not among the runs' keys"
        )
    rows = [index[k] for k in model.early_keys()]
    return np.ascontiguousarray(snapshots.outputs[rows])


def gated_rows(model: SurrogateModel, keys: Sequence[DataKey]) -> list[int] | None:
    """Return the rows of ``keys`` the gate scores, or ``None`` for every row.

    An early-start model returns a member's early data unchanged, so its error there is zero
    by construction; scoring those keys would dilute the error it makes after its date, and
    the tolerance would depend on ``early_until`` (found by the numerics review). Only the
    later keys are gated.

    Raises
    ------
    WorkflowError
        If no key lies after the early date (the gate would pass on nothing).
    """
    if not isinstance(model, EarlyStartSurrogate):
        return None
    early = set(model.early_keys())
    rows = [row for row, key in enumerate(keys) if key not in early]
    if not rows:
        # Every key is the members' own: a gate of no rows would pass vacuously.
        raise WorkflowError(
            f"the early-start surrogate answers no key after its early date "
            f"{model.early_until}: there is nothing to gate it on"
        )
    return rows


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
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> FitOutcome:
    """Fit ``model`` on ``training``, bound its domain, and gate it on ``held_out``.

    ``variances`` ``(n_keys,)`` gate against observation errors (``sigma_ratio``); ``None``
    gates by relative errors. ``objective`` (an optimization's) maps one
    column of outputs ``(n_keys, 1)`` to its objective value(s) ``(1,)`` or ``(n_obj, 1)``:
    the held-out runs' objectives, surrogate against simulator, are gated by rank too.

    Raises
    ------
    WorkflowError
        If the model does not answer the snapshots' keys after its fit.
    ValueError
        If the plugin cannot fit the snapshots, or the domain cannot be built.
    """
    state = dict(rng.bit_generator.state)
    if isinstance(model, CoarseCorrectedSurrogate) and (
        training.coarse_outputs is None or held_out.coarse_outputs is None
    ):
        raise WorkflowError(
            "a coarse-corrected surrogate needs the runs' coarse outputs: give the build a "
            "[coarse] stack"
        )
    began = clock()
    model.fit(training, rng)
    fit_seconds = clock() - began
    if tuple(model.keys) != training.keys:
        raise WorkflowError("the surrogate answers other keys than it was fitted on")
    coarse: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if (
        isinstance(model, CoarseCorrectedSurrogate)
        and training.coarse_outputs is not None
        and held_out.coarse_outputs is not None
    ):
        coarse = (training.coarse_outputs, held_out.coarse_outputs)
    elif isinstance(model, EarlyStartSurrogate):
        # A run's early data are its own values up to the early date.
        coarse = (early_data(model, training), early_data(model, held_out))
    names = tuple(sorted(training.fields))
    if names and not isinstance(model, FieldSurrogate):
        raise WorkflowError(
            "the snapshots hold state fields, but the surrogate predicts none (predict_fields)"
        )
    domain = training_domain(
        model, training.inputs, held_out.inputs, coarse=coarse, extent=extent,
        fields=names, field_times=training.field_times if names else (),
    )  # fmt: skip
    began = clock()
    if isinstance(model, CoarseCorrectedSurrogate) and coarse is not None:
        predicted = np.ascontiguousarray(
            model.predict_corrected(held_out.inputs, coarse[1], training.keys)
        )
    elif isinstance(model, EarlyStartSurrogate) and coarse is not None:
        predicted = np.ascontiguousarray(
            model.predict_from_early(held_out.inputs, coarse[1], training.keys)
        )
    else:
        predicted = np.ascontiguousarray(model.predict(held_out.inputs, training.keys))
    fields = None
    if names:
        # The state fields gated by relative errors (pressure on its change).
        assert isinstance(model, FieldSurrogate)
        predicted_fields = model.predict_fields(held_out.inputs, names, training.field_times)
        fields = {n: (np.asarray(predicted_fields[n]), held_out.fields[n]) for n in names}
    prediction_seconds = (clock() - began) / max(held_out.n_runs, 1)
    objectives = None
    if objective is not None:
        objectives = (
            column_objectives(objective, predicted),
            column_objectives(objective, held_out.outputs),
        )
    rows = gated_rows(model, training.keys)
    verdict = gate(
        predicted if rows is None else np.ascontiguousarray(predicted[rows]),
        held_out.outputs if rows is None else np.ascontiguousarray(held_out.outputs[rows]),
        training.keys if rows is None else tuple(training.keys[r] for r in rows),
        held_out.run_ids,
        tolerances=tolerances,
        variances=variances if variances is None or rows is None else variances[rows],
        kinds=kinds,
        fields=fields,
        failed=failed,
        objectives=objectives,
    )
    logger.info(
        "surrogate gated",
        extra={"event": "surrogate_gate", "passed": verdict.passed,
               "failures": verdict.failures()},
    )  # fmt: skip
    return FitOutcome(model, domain, verdict, predicted, state, fit_seconds, prediction_seconds)
