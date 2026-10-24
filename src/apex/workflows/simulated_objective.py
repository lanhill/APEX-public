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
        self.journal: dict[str, float | tuple[float, ...]] | None = None
        self.next_member = 0
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
        journal = self._journal()
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

    def _journal(self) -> dict[str, float | tuple[float, ...]]:
        if self.journal is None:
            rows = layout.read_evaluations(self.run_dir, repair=True)
            journal: dict[str, float | tuple[float, ...]] = {}
            for row in rows:
                if self.vector and row.get("values") is not None:
                    journal[row["key"]] = tuple(float(v) for v in row["values"])
                elif not self.vector and row.get("value") is not None:
                    journal[row["key"]] = float(row["value"])
            self.journal = journal
            # Failed evaluations are journaled too (without a value): no id is used twice.
            self.next_member = 1 + max((int(row["member_id"]) for row in rows), default=-1)
        return self.journal

    def _simulate(self, controls: FloatArray, reals: IntArray, pending: dict[str, int]) -> None:
        members: list[MemberInput] = []
        needs_dir = bool(getattr(self.simulator, "needs_work_dir", False))
        for offset, j in enumerate(pending.values()):
            member_id = self.next_member + offset
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
            self._journal()[key] = tuple(values) if self.vector else values[0]
            rows.append(
                {
                    "key": key,
                    "member_id": member.member_id,
                    "realization": int(reals[pending[key]]),
                    # One objective: "value" (as before); several: "values".
                    **({"values": values} if self.vector else {"value": values[0]}),
                    "responses": [float(v) for v in result],
                }
            )
        self.next_member += len(members)
        layout.append_evaluations(self.run_dir, rows)
        failed_ids = {f.member_id for f in failures}
        self._prune(members, failed_ids)
        logger.info(
            "evaluations_simulated",
            extra={"simulated": len(members), "failed": len(bad), "journal": len(self._journal())},
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
