"""Objectives evaluated by simulation, with a journal of every evaluation.

:class:`SimulatedObjective` is an :class:`~apex.algorithms.optimization.Objective` the
optimizers call as any other. Each column of controls becomes one simulation: the member's
state is the controls by block name (a deck template reads ``${injbhp[0]}``), plus the
column's model realization when the optimization is robust. The executor runs the batch
(with its retries and the machine-wide resource pools), and a
:class:`~apex.algorithms.optimization.ResponseObjective` (e.g. ``npv``) turns the responses
into values.

**The journal.** Every evaluation is appended to ``evaluations.jsonl`` after its batch, keyed
by the bytes of its controls and its realization (a failed one without a value, so that
it is simulated again on resume, under a new id). A known key is never simulated again: a
resumed optimization restarts the iteration it was in from its checkpoint, draws the same
controls (the random stream is restored), and finds them in the journal. The same holds for
repeated controls within a run, such as the iterate the loop evaluates again. A crash mid-batch
loses only that batch's simulations. Member ids count the journal's evaluations, so they, and
the simulations' folders, are the same in a resumed run as in an uninterrupted one.

**Two sources, one journal**. An optimization with a surrogate evaluates by
the surrogate and checks by the simulator: two objectives sharing one
:class:`EvaluationJournal`, each with its own ``context`` (the surrogate's includes its
identity, so its values are never reused by a run without it; the simulator's is the plain
run's, so a later run without a surrogate reuses them) and a ``source`` recorded in each row.
"""

import hashlib
import logging
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.optimization.protocols import FloatArray, IntArray, ResponseObjective
from apex.core.errors import WorkflowError
from apex.core.progress import ProgressReporter
from apex.core.simulation import Executor, MemberFailure, MemberInput, Simulator
from apex.storage import optimization_directory as layout

logger = logging.getLogger(__name__)


def evaluation_key(controls: FloatArray, realization: int, context: str = "") -> str:
    """Return the journal key of one evaluation.

    Parameters
    ----------
    controls
        One column of controls ``(n_x,)`` (physical); its float64 bytes are hashed.
    realization
        Its model realization.
    context
        What else decides the value (a hash of the objective, the simulator, the model file
        and the deck template): a journal entry made under another context is not reused.
    """
    digest = hashlib.sha256(context.encode())
    digest.update(np.ascontiguousarray(controls, dtype=np.float64).tobytes())
    digest.update(np.int64(realization).tobytes())
    return digest.hexdigest()


class EvaluationJournal:
    """The evaluations of a run directory's ``evaluations.jsonl``, shared by its objectives.

    Read once, at the first evaluation (under the run's lock); ``next_member`` counts every
    row, failed ones too, so that no member id is used twice whichever objective adds one.

    Parameters
    ----------
    run_dir
        The optimization's run directory.
    vector
        Whether values are tuples (several objectives).
    """

    def __init__(self, run_dir: Path, *, vector: bool = False) -> None:
        self.run_dir = run_dir
        self.vector = vector
        self._values: dict[str, float | tuple[float, ...]] | None = None
        #: Each successful row's responses by key (for checks against another source).
        self.responses: dict[str, list[float]] = {}
        #: Successful rows by source (rows without one: ``"simulator"``).
        self.by_source: dict[str, int] = {}
        self.next_member = 0

    @property
    def values(self) -> dict[str, float | tuple[float, ...]]:
        """Each successful evaluation's value(s) by key."""
        if self._values is None:
            rows = layout.read_evaluations(self.run_dir, repair=True)
            values: dict[str, float | tuple[float, ...]] = {}
            for row in rows:
                if self.vector and row.get("values") is not None:
                    values[row["key"]] = tuple(float(v) for v in row["values"])
                elif not self.vector and row.get("value") is not None:
                    values[row["key"]] = float(row["value"])
                else:
                    continue
                if row.get("responses") is not None:
                    self.responses[row["key"]] = [float(v) for v in row["responses"]]
                source = str(row.get("source", "simulator"))
                self.by_source[source] = self.by_source.get(source, 0) + 1
            self._values = values
            # Failed evaluations are journaled too (without a value): no id is used twice.
            self.next_member = 1 + max((int(row["member_id"]) for row in rows), default=-1)
        return self._values

    def record(self, rows: Sequence[Mapping[str, object]], n_members: int) -> None:
        """Append ``rows`` (``n_members`` new ids) to the file and to the journal in memory."""
        values = self.values
        for row in rows:
            key = str(row["key"])
            value = row.get("values" if self.vector else "value")
            if value is None:
                continue
            values[key] = (
                tuple(float(v) for v in value)  # type: ignore[attr-defined]
                if self.vector
                else float(value)  # type: ignore[arg-type]
            )
            responses = row.get("responses")
            if responses is not None:
                self.responses[key] = [float(v) for v in responses]  # type: ignore[attr-defined]
            source = str(row.get("source", "simulator"))
            self.by_source[source] = self.by_source.get(source, 0) + 1
        self.next_member += n_members
        layout.append_evaluations(self.run_dir, list(rows))


class SimulatedObjective:
    """An objective of simulated responses (see the module docstring).

    Parameters
    ----------
    objective
        Turns the responses at its ``response_keys()`` into values.
    simulator, executor
        Run the simulations.
    run_dir
        The optimization's run directory (journal and simulations).
    blocks
        The controls' blocks: name and slice of the control vector, in order.
    model
        Model arrays ``(n_values, n_realizations)`` by name, given to the simulations of each
        realization (robust optimization); empty for one model.
    keep_simulations
        Simulation folders kept: ``all``, the ``failed`` ones, or ``none``.
    context
        Part of every journal key (see :func:`evaluation_key`).
    progress
        Receives each batch's progress.
    journal
        The run's journal, when another objective shares it; none: its own.
    source
        Recorded in each row this objective adds (``surrogate``, ``simulator``); none: no
        source (a plain run's rows are unchanged).
    """

    def __init__(
        self,
        objective: ResponseObjective,
        simulator: Simulator,
        executor: Executor,
        run_dir: Path,
        blocks: Sequence[tuple[str, slice]],
        model: Mapping[str, NDArray[np.float64]] | None = None,
        *,
        keep_simulations: Literal["all", "failed", "none"] = "failed",
        progress: ProgressReporter | None = None,
        context: str = "",
        journal: EvaluationJournal | None = None,
        source: str | None = None,
    ) -> None:
        self.context = context
        self.objective = objective
        self.simulator = simulator
        self.executor = executor
        self.run_dir = run_dir
        self.blocks = tuple(blocks)
        self.model = dict(model or {})
        clash = sorted({name for name, _ in self.blocks} & set(self.model))
        if clash:
            raise ValueError(f"controls and model arrays share names: {clash}")
        widths = {a.shape[1] for a in self.model.values()}
        if len(widths) > 1:
            raise ValueError(f"the model arrays have different realization counts: {widths}")
        self.n_realizations = widths.pop() if widths else 1
        self.keep_simulations = keep_simulations
        self.progress = progress
        self.keys = list(objective.response_keys())
        # Several objectives (StackedResponses): this is then a vector objective,
        # and the journal keeps a tuple of values per evaluation.
        names = getattr(objective, "objective_names", None)
        self.vector = names is not None
        if names is not None:
            self.objective_names = tuple(names)
            # each objective's statistic over the realizations, for the optimizer.
            self.objective_statistics = tuple(
                getattr(objective, "objective_statistics", ("mean",) * len(names))
            )
        # The journal is read at the first evaluation, under the run's lock.
        if journal is not None and journal.vector != self.vector:
            raise ValueError("a shared journal must hold the objective's kind of values")
        self.journal = journal or EvaluationJournal(run_dir, vector=self.vector)
        self.source = source
        #: Simulations run by this process (the journal's replays are not counted).
        self.simulated = 0

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the objective of each column, simulating only those not in the journal.

        One objective: ``(n,)``; several (``StackedResponses``): ``(n_obj, n)``.

        Parameters
        ----------
        controls
            Control vectors ``(n_x, n)`` (physical).
        realizations
            Model realization of each column ``(n,)``.

        Raises
        ------
        WorkflowError
            If a simulation fails (after the executor's retries) or gives a non-finite value;
            the evaluations that succeeded are in the journal, so a resume keeps them.
        """
        journal = self.journal.values
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim != 2:
            raise ValueError(f"controls must be (n_x, n), got shape {controls.shape}")
        n = controls.shape[1]
        reals = np.broadcast_to(np.asarray(realizations, dtype=np.int64), (n,))
        if np.any(reals < 0) or np.any(reals >= self.n_realizations):
            raise WorkflowError(
                f"realizations {sorted(set(reals.tolist()))} are outside the model's "
                f"{self.n_realizations}"
            )
        keys = [evaluation_key(controls[:, j], int(reals[j]), self.context) for j in range(n)]
        pending: dict[str, int] = {}
        for j, key in enumerate(keys):
            if key not in journal and key not in pending:
                pending[key] = j
        if pending:
            self._simulate(controls, reals, pending)
        values = np.array([journal[key] for key in keys], dtype=np.float64)
        return np.ascontiguousarray(values.T) if self.vector else values

    def keys_of(self, controls: FloatArray, realizations: IntArray) -> list[str]:
        """Return the journal keys of the columns of ``controls`` ``(n_x, n)``."""
        controls = np.asarray(controls, dtype=np.float64)
        n = controls.shape[1]
        reals = np.broadcast_to(np.asarray(realizations, dtype=np.int64), (n,))
        return [evaluation_key(controls[:, j], int(reals[j]), self.context) for j in range(n)]

    def responses(self, controls: FloatArray, realizations: IntArray) -> NDArray[np.float64]:
        """Return the responses of each column ``(n_keys, n)``, evaluating those not known.

        Raises
        ------
        WorkflowError
            As :meth:`__call__`, or if a journal row holds no responses.
        """
        self(controls, realizations)
        keys = self.keys_of(controls, realizations)
        missing = [k for k in keys if k not in self.journal.responses]
        if missing:
            raise WorkflowError(f"{len(missing)} journaled evaluations hold no responses")
        return np.ascontiguousarray(
            np.array([self.journal.responses[k] for k in keys], dtype=np.float64).T
        )

    def _simulate(self, controls: FloatArray, reals: IntArray, pending: dict[str, int]) -> None:
        members: list[MemberInput] = []
        needs_dir = bool(getattr(self.simulator, "needs_work_dir", False))
        _ = self.journal.values  # read before the first id is given
        for offset, j in enumerate(pending.values()):
            member_id = self.journal.next_member + offset
            state = {name: np.array(controls[part, j]) for name, part in self.blocks}
            state.update({name: np.array(a[:, reals[j]]) for name, a in self.model.items()})
            members.append(
                MemberInput(
                    member_id,
                    state,
                    layout.simulation_dir(self.run_dir, member_id) if needs_dir else None,
                )
            )
        started = time.monotonic()
        if self.progress is not None:
            self.progress.forecast_started(
                iteration=None, label="evaluations", n_members=len(members)
            )
        callback = None
        if self.progress is not None:
            reporter = self.progress

            def callback(member_id: int, ok: bool) -> None:
                reporter.member_finished(member_id=member_id, ok=ok)

        results = self.executor.run_all(self.simulator, members, self.keys, on_result=callback)
        self.simulated += len(members)
        failures = [r for r in results if isinstance(r, MemberFailure)]
        if self.progress is not None:
            self.progress.forecast_finished(
                n_ok=len(results) - len(failures),
                n_failed=len(failures),
                seconds=time.monotonic() - started,
            )
        rows: list[dict[str, object]] = []
        bad: list[str] = []
        for key, member, result in zip(pending, members, results, strict=True):
            if isinstance(result, MemberFailure):
                bad.append(f"member {member.member_id}: {result.message}")
                rows.append({"key": key, "member_id": member.member_id, "value": None,
                             "error": result.message})  # fmt: skip
                continue
            # One column at a time: a reduction over a batch may round in another order
            # than over one column, and a value must not depend on its batch (a resume
            # evaluates other batches than an uninterrupted run).
            evaluated = np.asarray(self.objective.evaluate(np.asarray(result)[:, None]))
            values = [float(v) for v in evaluated[:, 0]] if self.vector else [float(evaluated[0])]
            if not all(np.isfinite(values)):
                bad.append(f"member {member.member_id}: the objective is {values}")
                rows.append({"key": key, "member_id": member.member_id, "value": None,
                             "error": f"objective {values}"})  # fmt: skip
                continue
            rows.append(
                {
                    "key": key,
                    "member_id": member.member_id,
                    "realization": int(reals[pending[key]]),
                    # One objective: "value" (as before); several: "values".
                    **({"values": values} if self.vector else {"value": values[0]}),
                    "responses": [float(v) for v in result],
                    # which source made it (absent in a plain run's rows).
                    **({} if self.source is None else {"source": self.source}),
                }
            )
        if self.source is not None:
            for row in rows:
                row.setdefault("source", self.source)
        self.journal.record(rows, len(members))
        failed_ids = {f.member_id for f in failures}
        self._prune(members, failed_ids)
        logger.info(
            "evaluations_simulated",
            extra={
                "simulated": len(members),
                "failed": len(bad),
                "journal": len(self.journal.values),
            },
        )
        if bad:
            raise WorkflowError(
                f"{len(bad)} of {len(members)} simulations failed; the run can be resumed "
                "once the cause is fixed:\n  " + "\n  ".join(bad[:10])
            )

    def _prune(self, members: Sequence[MemberInput], failed: set[int]) -> None:
        for member in members:
            if member.work_dir is None or not member.work_dir.exists():
                continue
            keep = self.keep_simulations == "all" or (
                self.keep_simulations == "failed" and member.member_id in failed
            )
            if not keep:
                shutil.rmtree(member.work_dir, ignore_errors=True)
