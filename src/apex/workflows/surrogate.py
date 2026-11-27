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

A build validated by folds (``[design] validation = "folds"``) takes
:func:`fit_folds_and_gate` instead of step 2: K models, each fitted on all folds but one and
predicting that fold, give every run an out-of-fold prediction; the gate is evaluated once
over all of them, and the shipped model is fitted on every run. Each fold's predictions are
saved as it ends, so a resume fits only the folds not done.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.surrogates.domain import model_coordinates, training_domain
from apex.algorithms.surrogates.gate import gate
from apex.core.errors import SimulationError, WorkflowError
from apex.core.progress import GuardedProgress, NullProgress, ProgressReporter
from apex.core.simulation import Executor, MemberFailure, MemberInput, Simulator
from apex.core.surrogates import (
    CoarseCorrectedSurrogate,
    EarlyStartSurrogate,
    FieldSurrogate,
    FoldAwareSurrogate,
    SurrogateModel,
)
from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    FOLDS_NOTE,
    DomainExtent,
    FoldGate,
    FoldValidation,
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
        # A simulator returning more than one number per key says how many (output_size, as
        # the executors read it): a sensitivity run's values and gradients.
        output_size = getattr(self.simulator, "output_size", None)
        expected = len(keys) if output_size is None else int(output_size(len(keys)))
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise SimulationError(
                f"member {member.member_id} returned {values.shape[0]} values for "
                f"{len(keys)} keys ({expected} expected), or values that are not finite"
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
    stopped
        Whether a stop ended the step before what follows it (an adaptive design's next
        fit and batch), even with no member pending.
    """

    values: Mapping[int, NDArray[np.float64]]
    seconds: Mapping[int, float]
    failed: Mapping[int, str]
    pending: tuple[int, ...]
    stopped: bool = False

    @property
    def complete(self) -> bool:
        """Whether every member ran, and the step was not stopped."""
        return not self.pending and not self.stopped


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
    stopped: Callable[[], bool] | None = None,
) -> MembersOutcome:
    """Run the members not finished yet; return every member's outcome.

    Parameters
    ----------
    members
        ``(member_id, {variable: (size,)})`` in state space, in the order they run.
    max_runs
        Run at most this many members now (``None``: all); the others stay pending.
    stopped
        Whether the build was asked to stop (a job's stop decision 4):
        given, the members run in waves (:data:`WAVE_WIDTHS` times the executor's parallel
        width, :func:`parallel_width`; one batch when it is unknown) and the check is made
        before each; once true, the rest stay pending. Members
        are independent and journalled by id, so waves change nothing a member computes.
        ``None`` (the CLI): one batch.
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
    width = None if stopped is None else parallel_width(executor, simulator)
    if width is None:
        # The CLI, or an executor that does not say its width (as the
        # user approved): one batch, the stop checked before it only. Narrow waves there
        # could leave its capacity idle.
        waves = [now] if now else []
    else:
        size = WAVE_WIDTHS * width
        waves = [now[i : i + size] for i in range(0, len(now), size)]
    ran = 0
    for wave in waves:
        if stopped is not None and stopped():
            logger.info(
                "design members stopped",
                extra={"event": "surrogate_members_stopped", "label": label,
                       "pending": len(missing) - ran},
            )  # fmt: skip
            break
        _run_wave(directory, simulator, executor, wave, keys, label=label,
                  keep_simulations=keep_simulations, reporter=reporter, values=values,
                  seconds=seconds, failed=failed,
                  pending=len(missing) - ran - len(wave))  # fmt: skip
        ran += len(wave)
    pending = tuple(m for m, _ in missing[ran:])
    return MembersOutcome(values, seconds, failed, pending)


def parallel_width(executor: Executor, simulator: object = None) -> int | None:
    """Return how many members an executor runs at once, if it says (``None``: unknown).

    Its ``parallel_width`` (an optional capability: a number, or a method of the simulator,
    as the ssh executor's, whose width is its hosts' slots over a member's threads), else
    its ``workers`` (the local executors'), capped by the simulator's ``max_parallel_runs``
    (an Eclipse licence count), as the executor caps its pool.
    """
    width = None
    for name in ("parallel_width", "workers"):
        value = getattr(executor, name, None)
        if callable(value):
            value = value(simulator)
        if isinstance(value, int) and value > 0:
            width = value
            break
    cap = getattr(simulator, "max_parallel_runs", None)
    if width is not None and isinstance(cap, int) and cap > 0:
        width = min(width, cap)
    return width


#: A stoppable set of runs goes in waves of this many times the executor's parallel width:
#: wider waves keep the executor busy past slow members, narrower ones stop sooner. Each
#: wave is one ``run_all`` (a process pool started per wave, waiting for its slowest
#: member): a modest cost against one batch, the price of stopping within a design.
WAVE_WIDTHS = 4


def _run_wave(
    directory: SurrogateDirectory,
    simulator: Simulator,
    executor: Executor,
    now: Sequence[tuple[int, Mapping[str, NDArray[np.float64]]]],
    keys: Sequence[DataKey],
    *,
    label: str,
    keep_simulations: str,
    reporter: ProgressReporter,
    values: dict[int, NDArray[np.float64]],
    seconds: dict[int, float],
    failed: dict[int, str],
    pending: int,
) -> None:
    """Run ``now``'s members as one batch, recording each outcome in the three maps."""
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
                   "failed": n_failed, "pending": pending},
        )  # fmt: skip


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
    validation
        A build validated by folds: its folds and each fold's measures (``gate`` is then
        the pooled out-of-fold verdict, ``predicted`` every run's out-of-fold prediction);
        ``None`` for a gate on held-out runs.
    """

    model: SurrogateModel
    domain: TrainingDomain
    gate: GateResult
    predicted: NDArray[np.float64]
    stream_state: dict[str, Any]
    fit_seconds: float
    prediction_seconds: float
    validation: FoldValidation | None = None


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
    coarse = _coarse_pair(model, training, held_out)
    names = _field_names(model, training)
    domain = training_domain(
        model, training.inputs, held_out.inputs, coarse=coarse, extent=extent,
        fields=names, field_times=training.field_times if names else (),
    )  # fmt: skip
    began = clock()
    predicted, predicted_fields = _predict_runs(model, training, held_out, coarse)
    fields = (
        None if not names
        else {n: (predicted_fields[n], held_out.fields[n]) for n in names}
    )  # fmt: skip
    prediction_seconds = (clock() - began) / max(held_out.n_runs, 1)
    verdict = _gate_runs(
        model, predicted, held_out.outputs, training.keys, held_out.run_ids,
        tolerances=tolerances, variances=variances, kinds=kinds, fields=fields,
        failed=failed, objective=objective,
    )  # fmt: skip
    logger.info(
        "surrogate gated",
        extra={"event": "surrogate_gate", "passed": verdict.passed,
               "failures": verdict.failures()},
    )  # fmt: skip
    return FitOutcome(model, domain, verdict, predicted, state, fit_seconds, prediction_seconds)


def _coarse_pair(
    model: SurrogateModel, training: SnapshotSet, held_out: SnapshotSet
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return the two sets' coarse forecasts a model's coordinates and prediction take.

    A coarse-corrected model's coarse runs; an early-start model's early data (a run's
    own values up to the early date); ``None`` for any other model.
    """
    if (
        isinstance(model, CoarseCorrectedSurrogate)
        and training.coarse_outputs is not None
        and held_out.coarse_outputs is not None
    ):
        return training.coarse_outputs, held_out.coarse_outputs
    if isinstance(model, EarlyStartSurrogate):
        return early_data(model, training), early_data(model, held_out)
    return None


def _field_names(model: SurrogateModel, training: SnapshotSet) -> tuple[str, ...]:
    """Return the state fields the snapshots hold, which the model must predict.

    Raises
    ------
    WorkflowError
        If the snapshots hold fields and the model predicts none.
    """
    names = tuple(sorted(training.fields))
    if names and not isinstance(model, FieldSurrogate):
        raise WorkflowError(
            "the snapshots hold state fields, but the surrogate predicts none (predict_fields)"
        )
    return names


def _predict_runs(
    model: SurrogateModel,
    training: SnapshotSet,
    runs: SnapshotSet,
    coarse: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Return a fitted model's prediction of ``runs``: ``(n_keys, n)`` and its fields.

    The fields are ``{name: (n_cells, n_times, n)}`` for the snapshots' fields, none
    without them. ``coarse``: :func:`_coarse_pair`'s, its second half ``runs``'.
    """
    if isinstance(model, CoarseCorrectedSurrogate) and coarse is not None:
        predicted = np.ascontiguousarray(
            model.predict_corrected(runs.inputs, coarse[1], training.keys)
        )
    elif isinstance(model, EarlyStartSurrogate) and coarse is not None:
        predicted = np.ascontiguousarray(
            model.predict_from_early(runs.inputs, coarse[1], training.keys)
        )
    else:
        predicted = np.ascontiguousarray(model.predict(runs.inputs, training.keys))
    names = _field_names(model, training)
    fields: dict[str, NDArray[np.float64]] = {}
    if names:
        # The state fields gated by relative errors (pressure on its change).
        assert isinstance(model, FieldSurrogate)
        found = model.predict_fields(runs.inputs, names, training.field_times)
        fields = {n: np.asarray(found[n]) for n in names}
    return predicted, fields


def _gate_runs(
    model: SurrogateModel,
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    keys: tuple[DataKey, ...],
    run_ids: Sequence[int],
    *,
    tolerances: GateTolerances,
    variances: NDArray[np.float64] | None,
    kinds: Mapping[str, QuantityKind] | None,
    fields: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None,
    failed: Sequence[int],
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None,
) -> GateResult:
    """Gate predictions ``(n_keys, n)`` of runs against their simulator values ``reference``.

    The rows :func:`gated_rows` names (an early-start model's later keys), and the runs'
    objectives by rank when ``objective`` is given.
    """
    objectives = None
    if objective is not None:
        objectives = (
            column_objectives(objective, predicted),
            column_objectives(objective, reference),
        )
    rows = gated_rows(model, keys)
    return gate(
        predicted if rows is None else np.ascontiguousarray(predicted[rows]),
        reference if rows is None else np.ascontiguousarray(reference[rows]),
        keys if rows is None else tuple(keys[r] for r in rows),
        run_ids,
        tolerances=tolerances,
        variances=variances if variances is None or rows is None else variances[rows],
        kinds=kinds,
        fields=fields,
        failed=failed,
        objectives=objectives,
    )


# =========================================================== validation by folds (rev. 6)


def fold_split(run_ids: Sequence[int], folds: int, seed: int) -> list[tuple[int, ...]]:
    """Return the K folds of a build validated by folds: run ids, each fold in id order.

    The runs, in id order, are permuted by ``numpy.random.default_rng(seed)`` and the
    permutation cut by ``numpy.array_split`` into ``folds`` near-equal parts (the first
    ``n mod K`` one run larger): the usual K-fold split, fixed by a seed of its own, so a
    split made elsewhere as ``array_split(default_rng(seed).permutation(n), K)`` over runs
    ``0..n-1`` is reproduced when the runs are those.

    Raises
    ------
    WorkflowError
        If there are fewer than two folds, or fewer runs than folds.
    """
    ordered = sorted(int(r) for r in run_ids)
    if folds < 2:
        raise WorkflowError(f"a validation by folds needs at least 2 folds, got {folds}")
    if len(ordered) < folds:
        raise WorkflowError(
            f"{folds} folds need at least {folds} successful runs, got {len(ordered)}: "
            "lower [design] folds, or add training runs"
        )
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return [
        tuple(sorted(ordered[int(i)] for i in part)) for part in np.array_split(permutation, folds)
    ]


#: A fold's own rank check needs at least this many runs (its recorded measure only; the
#: pooled gate's rank check is over every run).
MIN_FOLD_RANK_RUNS = 3


@dataclass(frozen=True)
class FoldPrediction:
    """One fold's out-of-fold predictions, saved as the fold ends (a resume reads them).

    Attributes
    ----------
    predicted
        ``(n_keys, n_fold)``: the fold's runs, in the snapshot set's order, predicted by the
        model fitted on the other folds' runs.
    fields
        ``{name: (n_cells, n_times, n_fold)}``: its predicted state fields (none without).
    fit_seconds, prediction_seconds
        The fold's fit, and its whole prediction.
    residuals
        ``(n_fold,)``: the fold's runs' projection residuals under the fold's model (out of
        sample), which floor the shipped model's residual bound (its own residuals are all
        in sample).
    """

    predicted: NDArray[np.float64]
    fields: Mapping[str, NDArray[np.float64]]
    fit_seconds: float
    prediction_seconds: float
    residuals: NDArray[np.float64]


def fit_folds_and_gate(
    factory: Callable[[], SurrogateModel],
    snapshots: SnapshotSet,
    split: Sequence[Sequence[int]],
    rng: np.random.Generator,
    fold_rng: Callable[[int], np.random.Generator],
    *,
    fold_seed: int,
    tolerances: GateTolerances,
    variances: NDArray[np.float64] | None,
    kinds: Mapping[str, QuantityKind] | None = None,
    extent: DomainExtent = "box",
    failed: Sequence[int] = (),
    objective: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    saved: Callable[[int], FoldPrediction | None] | None = None,
    save: Callable[[int, FoldPrediction], None] | None = None,
    stopped: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> FitOutcome | None:
    """Gate a plugin's fitting procedure by K-fold cross-validation; fit the shipped model.

    For each fold ``k`` of ``split`` (run ids, :func:`fold_split`), a fresh model from
    ``factory`` is fitted, through its own ``fit``, on the other folds' runs (in the
    snapshot set's order) with the generator ``fold_rng(k)``, independent of the other
    folds' (so a fold fitted on resume is the one an uninterrupted build fits), and predicts
    the fold's runs. The gate is then evaluated **once over every run's out-of-fold
    prediction**: the same measures and tolerances as a gate on held-out runs, over all
    ``n`` runs (the pooled verdict); each fold's own measures are recorded beside it. The
    shipped model is fitted on every run with ``rng`` (the build's stream ``surrogate``, as
    a model gated on held-out runs is), and its domain is built from every run's inputs: it
    was not itself gated, only its procedure was (:data:`~apex.domain.surrogate.FOLDS_NOTE`).

    ``saved(k)`` returns fold ``k``'s predictions if an earlier attempt saved them (the fold
    is then not fitted again), and ``save(k, prediction)`` saves them as the fold ends.
    ``stopped`` is checked before each fold is fitted: once true, ``None`` is returned, the
    folds done staying saved. The other parameters are :func:`fit_and_gate`'s.

    Raises
    ------
    WorkflowError
        If the split is not a partition of the snapshots' runs, a model answers other keys
        than it was fitted on, or a coarse-corrected model has no coarse runs.
    ValueError
        If the plugin cannot fit the snapshots, or the domain cannot be built.
    """
    positions = {int(r): p for p, r in enumerate(snapshots.run_ids)}
    chosen = [int(r) for fold in split for r in fold]
    if sorted(chosen) != sorted(positions) or len(chosen) != len(set(chosen)):
        raise WorkflowError("the folds are not a partition of the snapshots' runs")
    n = snapshots.n_runs
    done: list[tuple[list[int], FoldPrediction]] = []
    for k, fold in enumerate(split):
        held = sorted(positions[int(r)] for r in fold)
        found = None if saved is None else saved(k)
        if found is None:
            if stopped is not None and stopped():
                logger.info(
                    "folds stopped",
                    extra={"event": "surrogate_folds_stopped", "done": k, "folds": len(split)},
                )  # fmt: skip
                return None
            found = _fit_fold(factory, snapshots, held, fold_rng(k), clock, fold=k)
            if save is not None:
                save(k, found)
            logger.info(
                "fold fitted", extra={"event": "surrogate_fold", "fold": k, "runs": len(held)}
            )
        if found.predicted.shape != (len(snapshots.keys), len(held)) or found.residuals.shape != (
            len(held),
        ):
            raise WorkflowError(f"fold {k}'s saved predictions do not match its runs")
        done.append((held, found))
    # The shipped model: fitted on every run; its domain every run's inputs (no held-out).
    model = factory()
    _refuse_without_coarse(model, snapshots)
    state = dict(rng.bit_generator.state)
    began = clock()
    model.fit(snapshots, rng)
    fit_seconds = clock() - began
    if tuple(model.keys) != snapshots.keys:
        raise WorkflowError("the surrogate answers other keys than it was fitted on")
    names = _field_names(model, snapshots)
    pair = None
    if isinstance(model, CoarseCorrectedSurrogate) and snapshots.coarse_outputs is not None:
        pair = (snapshots.coarse_outputs, snapshots.coarse_outputs[:, :0])
    elif isinstance(model, EarlyStartSurrogate):
        early = early_data(model, snapshots)
        pair = (early, early[:, :0])  # no held-out runs: the second half is not read
    # Every input is in sample for the shipped model, so its own residuals would bound the
    # domain too tightly (a reduced-input model's are only its truncation error): the runs'
    # out-of-fold residuals floor the bound, as a held-out run's residual does otherwise.
    floor = max(float(np.max(f.residuals, initial=0.0)) for _, f in done)
    domain = training_domain(
        model, snapshots.inputs, {}, coarse=pair, extent=extent,
        fields=names, field_times=snapshots.field_times if names else (),
        residual_floor=floor,
    )  # fmt: skip
    # Every run's out-of-fold prediction, in the snapshot set's order.
    predicted = np.zeros((len(snapshots.keys), n))
    pooled = {name: np.zeros((*snapshots.fields[name].shape[:2], n)) for name in names}
    per_fold = []
    for k, (held, found) in enumerate(done):
        predicted[:, held] = found.predicted
        for name in names:
            pooled[name][:, :, held] = found.fields[name]
        runs = snapshots.columns(held)
        fold_fields = {name: (np.asarray(found.fields[name]), runs.fields[name]) for name in names}
        fold_gate = _gate_runs(
            model, np.ascontiguousarray(found.predicted), runs.outputs, snapshots.keys,
            runs.run_ids, tolerances=tolerances, variances=variances, kinds=kinds,
            fields=fold_fields or None, failed=(),
            # A rank correlation over fewer than 3 runs says nothing (undefined, or +-1).
            objective=objective if len(held) >= MIN_FOLD_RANK_RUNS else None,
        )  # fmt: skip
        per_fold.append(FoldGate(fold=k, runs=sorted(runs.run_ids), gate=fold_gate))
    predicted = np.ascontiguousarray(predicted)
    pooled_fields = {name: (pooled[name], snapshots.fields[name]) for name in names}
    verdict = _gate_runs(
        model, predicted, snapshots.outputs, snapshots.keys, snapshots.run_ids,
        tolerances=tolerances, variances=variances, kinds=kinds,
        fields=pooled_fields or None, failed=failed, objective=objective,
    )  # fmt: skip
    validation = FoldValidation(
        folds=len(split), fold_seed=fold_seed, per_fold=per_fold, note=FOLDS_NOTE
    )
    logger.info(
        "surrogate gated by folds",
        extra={"event": "surrogate_gate", "passed": verdict.passed, "folds": len(split),
               "failures": verdict.failures()},
    )  # fmt: skip
    prediction_seconds = sum(f.prediction_seconds for _, f in done) / max(n, 1)
    return FitOutcome(model, domain, verdict, predicted, state, fit_seconds,
                      prediction_seconds, validation)  # fmt: skip


def _fit_fold(
    factory: Callable[[], SurrogateModel],
    snapshots: SnapshotSet,
    held: Sequence[int],
    rng: np.random.Generator,
    clock: Callable[[], float],
    *,
    fold: int,
) -> FoldPrediction:
    """Fit a fresh model on the runs not at positions ``held``; predict those runs.

    A model that is :class:`~apex.core.surrogates.FoldAwareSurrogate` is told ``fold``.
    """
    left_out = set(held)
    training = snapshots.columns([p for p in range(snapshots.n_runs) if p not in left_out])
    runs = snapshots.columns(list(held))
    model = factory()
    _refuse_without_coarse(model, snapshots)
    if isinstance(model, FoldAwareSurrogate):
        model.set_validation_fold(fold)
    began = clock()
    model.fit(training, rng)
    fit_seconds = clock() - began
    if tuple(model.keys) != training.keys:
        raise WorkflowError("the surrogate answers other keys than it was fitted on")
    began = clock()
    pair = _coarse_pair(model, training, runs)
    predicted, fields = _predict_runs(model, training, runs, pair)
    seconds = clock() - began
    _, residuals = model_coordinates(model, runs.inputs, None if pair is None else pair[1])
    return FoldPrediction(
        predicted, fields, fit_seconds, seconds, np.ascontiguousarray(residuals, dtype=np.float64)
    )


def _refuse_without_coarse(model: SurrogateModel, snapshots: SnapshotSet) -> None:
    if isinstance(model, CoarseCorrectedSurrogate) and snapshots.coarse_outputs is None:
        raise WorkflowError(
            "a coarse-corrected surrogate needs the runs' coarse outputs: give the build a "
            "[coarse] stack"
        )
