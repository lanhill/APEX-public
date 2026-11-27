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
the simulations' folders, are the same in a resumed run as in an uninterrupted one. Each row
also records its ``controls``, the column as the optimizer proposed it (before any
parameterization), so that the journal's simulator rows are pairs of controls and responses
a surrogate can be fitted on; rows written before have none and replay
as before (the key alone decides a replay).

**Two sources, one journal**. An optimization with a surrogate evaluates by
the surrogate and checks by the simulator: two objectives sharing one
:class:`EvaluationJournal`, each with its own ``context`` (the surrogate's includes its
identity, so its values are never reused by a run without it; the simulator's is the plain
run's, so a later run without a surrogate reuses them) and a ``source`` recorded in each row.
A coarse-corrected surrogate adds a third, its coarse stack (``source = "coarse"``), whose
failures are journaled without stopping the run
(``tolerate_failures``): the simulator stands in for those columns, and the journal remembers
the failure (:attr:`EvaluationJournal.failed`), so a resume makes the same choice.
"""

import hashlib
import logging
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.optimization.protocols import FloatArray, IntArray, ResponseObjective
from apex.core.errors import WorkflowError
from apex.core.progress import ProgressReporter
from apex.core.simulation import Executor, MemberFailure, MemberInput, Simulator
from apex.domain.observations import DataKey
from apex.storage import optimization_directory as layout

logger = logging.getLogger(__name__)

#: Given a batch's new members (``{member_id: column}`` of the controls the objective was
#: called with), returns the simulator that runs them (e.g. a surrogate given each member's
#: coarse forecast).
SimulatorFor = Callable[[Mapping[int, int]], Simulator]


class BrokenStackError(WorkflowError):
    """A stack whose failures are tolerated failed every new simulation of a batch."""


def fit_only_digest(keys: Sequence[DataKey]) -> str:
    """Return the digest a journal row's fit-only values are recorded under.

    So that a row's values are read only against the keys they were simulated at.
    """
    digest = hashlib.sha256()
    for key in keys:
        digest.update(f"{key.time}\x00{key.datatype}\x00{key.index}\x01".encode())
    return digest.hexdigest()[:16]


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
        #: Every row by source, failed ones too, except those marked ``retry`` (a stopped
        #: batch's, run again on resume: they do not count against a bound twice).
        self.rows_by_source: dict[str, int] = {}
        #: Keys whose every row failed (an objective with ``tolerate_failures`` does not
        #: run them again; any other runs them again under a new id, as before). Rows marked
        #: ``retry`` (a stack that failed a whole batch) are left out: run again on resume.
        self.failed: set[str] = set()
        #: Each successful row's controls ``(n_x,)`` and realization, by key (rows written
        #: have no controls and are absent).
        self.inputs: dict[str, tuple[list[float], int]] = {}
        #: Each successful row's fit-only values by key, with their keys' digest (rows
        #: written before, or without fit-only keys, are absent).
        self.fit_only: dict[str, tuple[str, list[float]]] = {}
        self.next_member = 0

    @property
    def values(self) -> dict[str, float | tuple[float, ...]]:
        """Each successful evaluation's value(s) by key."""
        if self._values is None:
            rows = layout.read_evaluations(self.run_dir, repair=True)
            values: dict[str, float | tuple[float, ...]] = {}
            for row in rows:
                source = str(row.get("source", "simulator"))
                if not row.get("retry"):
                    self.rows_by_source[source] = self.rows_by_source.get(source, 0) + 1
                if self.vector and row.get("values") is not None:
                    values[row["key"]] = tuple(float(v) for v in row["values"])
                elif not self.vector and row.get("value") is not None:
                    values[row["key"]] = float(row["value"])
                else:
                    if not row.get("retry"):
                        self.failed.add(str(row["key"]))
                    continue
                if row.get("responses") is not None:
                    self.responses[row["key"]] = [float(v) for v in row["responses"]]
                self._remember(row)
                self.by_source[source] = self.by_source.get(source, 0) + 1
            self.failed -= set(values)
            self._values = values
            # Failed evaluations are journaled too (without a value): no id is used twice.
            self.next_member = 1 + max((int(row["member_id"]) for row in rows), default=-1)
        return self._values

    def record(self, rows: Sequence[Mapping[str, object]], n_members: int) -> None:
        """Append ``rows`` (``n_members`` new ids) to the file and to the journal in memory."""
        values = self.values
        for row in rows:
            key = str(row["key"])
            source = str(row.get("source", "simulator"))
            if not row.get("retry"):
                self.rows_by_source[source] = self.rows_by_source.get(source, 0) + 1
            value = row.get("values" if self.vector else "value")
            if value is None:
                if key not in values and not row.get("retry"):
                    self.failed.add(key)
                continue
            self.failed.discard(key)
            values[key] = (
                tuple(float(v) for v in value)  # type: ignore[attr-defined]
                if self.vector
                else float(value)  # type: ignore[arg-type]
            )
            responses = row.get("responses")
            if responses is not None:
                self.responses[key] = [float(v) for v in responses]  # type: ignore[attr-defined]
            self._remember(row)
            self.by_source[source] = self.by_source.get(source, 0) + 1
        self.next_member += n_members
        layout.append_evaluations(self.run_dir, list(rows))

    def _remember(self, row: Mapping[str, object]) -> None:
        """Keep a successful row's controls and realization (and fit-only values), if any."""
        key = str(row["key"])
        fit_only = row.get("fit_only")
        if isinstance(fit_only, Mapping) and key not in self.fit_only:
            values = fit_only.get("values")
            if isinstance(values, list):
                self.fit_only[key] = (str(fit_only.get("keys")), [float(v) for v in values])
        controls = row.get("controls")
        if isinstance(controls, list) and key not in self.inputs:
            realization = row.get("realization", 0)
            self.inputs[key] = (
                [float(v) for v in controls],
                int(realization) if isinstance(realization, int | float) else 0,
            )


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
        Recorded in each row this objective adds (``surrogate``, ``simulator``, ``coarse``);
        none: no source (a plain run's rows are unchanged).
    tolerate_failures
        Journal failed simulations without raising, and never run a key whose rows all
        failed again (a coarse stack's); call :meth:`run`, not the
        objective, which needs every value.
    broken_stack
        With ``tolerate_failures``: names the stack (e.g. its deck) when a batch's new
        simulations, of two or more distinct control vectors, all fail; the rows are then
        marked ``retry`` (run again on resume, not remembered) and
        :class:`BrokenStackError` is raised.
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
        tolerate_failures: bool = False,
        broken_stack: str | None = None,
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
        self.tolerate_failures = tolerate_failures
        self.broken_stack = broken_stack
        #: Simulations run by this process (the journal's replays are not counted).
        self.simulated = 0
        self._fit_only_keys: list[DataKey] = []

    @property
    def fit_only_keys(self) -> list[DataKey]:
        """Keys every new simulation also returns, for a surrogate's refits (revision 9).

        Journaled apart from the responses (``fit_only``), all of them, under their digest;
        never the objective's, and never part of an evaluation's key: a row's value does not
        depend on them.
        """
        return list(self._fit_only_keys)

    @fit_only_keys.setter
    def fit_only_keys(self, keys: Sequence[DataKey]) -> None:
        overlap = sorted({str(k) for k in keys if k in set(self.keys)})
        if overlap:
            raise ValueError(f"fit-only keys may not be the objective's: {overlap}")
        if len(set(keys)) != len(keys):
            raise ValueError("fit-only keys repeat")
        self._fit_only_keys = list(keys)

    def __call__(
        self,
        controls: FloatArray,
        realizations: IntArray,
        *,
        simulator_for: SimulatorFor | None = None,
        row_extra: Mapping[str, object] | None = None,
    ) -> FloatArray:
        """Return the objective of each column, simulating only those not in the journal.

        One objective: ``(n,)``; several (``StackedResponses``): ``(n_obj, n)``.

        Parameters
        ----------
        controls
            Control vectors ``(n_x, n)`` (physical).
        realizations
            Model realization of each column ``(n,)``.
        simulator_for
            Returns the simulator of the batch's new members (none: the objective's own).
        row_extra
            Fields added to each successful new row (e.g. ``stands_in_for``).

        Raises
        ------
        WorkflowError
            If a simulation fails (after the executor's retries) or gives a non-finite value;
            the evaluations that succeeded are in the journal, so a resume keeps them.
        """
        journal = self.journal.values
        keys = self.run(controls, realizations, simulator_for=simulator_for, row_extra=row_extra)
        missing = [k for k in keys if k not in journal]
        if missing:
            raise WorkflowError(f"{len(missing)} evaluations failed and have no value")
        values = np.array([journal[key] for key in keys], dtype=np.float64)
        return np.ascontiguousarray(values.T) if self.vector else values

    def run(
        self,
        controls: FloatArray,
        realizations: IntArray,
        *,
        simulator_for: SimulatorFor | None = None,
        row_extra: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Simulate the columns not in the journal; return every column's journal key.

        With ``tolerate_failures``, keys whose rows all failed are not run again, and new
        failures are journaled without raising. Arguments as :meth:`__call__`.

        Raises
        ------
        WorkflowError
            Without ``tolerate_failures``, as :meth:`__call__`.
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
        skipped = self.journal.failed if self.tolerate_failures else set()
        pending: dict[str, int] = {}
        for j, key in enumerate(keys):
            if key not in journal and key not in pending and key not in skipped:
                pending[key] = j
        if pending:
            self._simulate(controls, reals, pending, simulator_for, row_extra)
        return keys

    def keys_of(self, controls: FloatArray, realizations: IntArray) -> list[str]:
        """Return the journal keys of the columns of ``controls`` ``(n_x, n)``."""
        controls = np.asarray(controls, dtype=np.float64)
        n = controls.shape[1]
        reals = np.broadcast_to(np.asarray(realizations, dtype=np.int64), (n,))
        return [evaluation_key(controls[:, j], int(reals[j]), self.context) for j in range(n)]

    def responses(
        self,
        controls: FloatArray,
        realizations: IntArray,
        *,
        simulator_for: SimulatorFor | None = None,
    ) -> NDArray[np.float64]:
        """Return the responses of each column ``(n_keys, n)``, evaluating those not known.

        Raises
        ------
        WorkflowError
            As :meth:`__call__`, or if a journal row holds no responses.
        """
        self(controls, realizations, simulator_for=simulator_for)
        keys = self.keys_of(controls, realizations)
        missing = [k for k in keys if k not in self.journal.responses]
        if missing:
            raise WorkflowError(f"{len(missing)} journaled evaluations hold no responses")
        return np.ascontiguousarray(
            np.array([self.journal.responses[k] for k in keys], dtype=np.float64).T
        )

    def _simulate(
        self,
        controls: FloatArray,
        reals: IntArray,
        pending: dict[str, int],
        simulator_for: SimulatorFor | None = None,
        row_extra: Mapping[str, object] | None = None,
    ) -> None:
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

        # Member ids are given here, so a simulator that needs each member's companion data
        # (a coarse-corrected surrogate's coarse forecast) is made here too.
        simulator = (
            self.simulator
            if simulator_for is None
            else simulator_for(
                {m.member_id: j for m, j in zip(members, pending.values(), strict=True)}
            )
        )
        extra = list(self._fit_only_keys)
        n_keys = len(self.keys)
        results = self.executor.run_all(
            simulator, members, [*self.keys, *extra], on_result=callback
        )
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
                             "error": result.message,
                             "controls": _column(controls, pending[key])})  # fmt: skip
                continue
            full = np.asarray(result)
            result = full[:n_keys]
            # One column at a time: a reduction over a batch may round in another order
            # than over one column, and a value must not depend on its batch (a resume
            # evaluates other batches than an uninterrupted run).
            evaluated = np.asarray(self.objective.evaluate(np.asarray(result)[:, None]))
            values = [float(v) for v in evaluated[:, 0]] if self.vector else [float(evaluated[0])]
            if not all(np.isfinite(values)):
                bad.append(f"member {member.member_id}: the objective is {values}")
                rows.append({"key": key, "member_id": member.member_id, "value": None,
                             "error": f"objective {values}",
                             "controls": _column(controls, pending[key])})  # fmt: skip
                continue
            rows.append(
                {
                    "key": key,
                    "member_id": member.member_id,
                    "realization": int(reals[pending[key]]),
                    # One objective: "value" (as before); several: "values".
                    **({"values": values} if self.vector else {"value": values[0]}),
                    "responses": [float(v) for v in result],
                    # the controls as proposed, a surrogate's inputs.
                    "controls": _column(controls, pending[key]),
                    # which source made it (absent in a plain run's rows).
                    **({} if self.source is None else {"source": self.source}),
                    # values for a surrogate's refits, never the
                    # objective's; absent without fit-only keys (every row as before).
                    **(
                        {
                            "fit_only": {
                                "keys": fit_only_digest(extra),
                                "values": [float(v) for v in full[n_keys:]],
                            }
                        }
                        if extra
                        else {}
                    ),
                    **(row_extra or {}),
                }
            )
        if self.source is not None:
            for row in rows:
                row.setdefault("source", self.source)
        # A stack that fails every new simulation of a batch (two or more) is broken, not
        # unlucky: the run stops, and the failures are retried on resume once it is fixed.
        # Distinct controls, not members: one control vector failing at every realization
        # (a check step's batch) is that vector's failure, which falls back (the review).
        distinct = {np.ascontiguousarray(controls[:, j]).tobytes() for j in pending.values()}
        broken = (
            self.tolerate_failures
            and self.broken_stack is not None
            and len(distinct) >= 2
            and len(bad) == len(members)
        )
        if broken:
            for row in rows:
                row["retry"] = True
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
        if broken:
            raise BrokenStackError(
                f"every one of the {len(members)} new runs of {self.broken_stack} failed: the "
                "optimization stops at its last checkpoint; fix the stack (its deck, its "
                "includes, the simulator's environment) and resume, which runs them again:\n  "
                + "\n  ".join(bad[:10])
            )
        if bad and self.tolerate_failures:
            logger.warning(
                "evaluations_failed_tolerated",
                extra={"source": self.source, "failed": len(bad), "details": bad[:10]},
            )
        elif bad:
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


def _column(controls: FloatArray, column: int) -> list[float]:
    """Return one column of ``controls`` ``(n_x, n)`` as the journal records it.

    Python's JSON floats round-trip exactly: read back, they are the same float64 values, so
    the column's journal key can be computed again from the row.
    """
    return [float(v) for v in controls[:, column]]
