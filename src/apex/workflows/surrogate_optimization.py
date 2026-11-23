"""Optimization with a surrogate, checked by the simulator.

:class:`SurrogateObjective` is the objective the optimizer sees: every evaluation it asks
for is the surrogate's, through a :class:`~apex.workflows.simulated_objective.SimulatedObjective`
whose simulator is the surrogate. A second one, on the optimization's own simulator, makes
the checks and the reported optimum. Both share the run's evaluation journal, each under its
own context:

- the surrogate's evaluations under the plain context *with the surrogate's identity*, so
  they are never reused by a run without it, or with another model;
- the simulator's under the plain context, so a later run without a surrogate reuses them.

Each row records its ``source``.

**Domain.** Controls outside the model's validated domain stop the optimization
(:class:`~apex.core.errors.SurrogateDomainError`, never a dropped evaluation), or with
``outside = "simulator"`` are evaluated by the simulator, which the row records.

**Checks.** Every ``check_every`` accepted steps the accepted controls are evaluated by the
simulator at every realization and gated against the surrogate's predictions of the same
evaluations, by relative errors (the model's tolerances, tightened by the section's). The
check runs before the step's checkpoint is written. A failing check either stops the
optimization (:class:`SurrogateCheckFailed`), resumable from its last checkpoint once the
section is changed, or (``on_check_failure = "simulator"``) records the switch and raises
:class:`~apex.workflows.optimization.RestartFromStart`: the optimization starts again from its
start with the simulator, which evaluates everything from then on. Not from its last
checkpoint, which holds the optimizer's surrogate values (an iterate's objective the
simulator's steps would be compared with): from the start the run is the plain run with the
same seed, evaluation for evaluation. The switch is kept in ``surrogate_state.json``, so a
resume makes the same choice, and the journal keeps every simulator evaluation already made
(the checks'), which are not repeated.

**The optimum is the simulator's.** When the optimization ends, its result's controls (and a
multi-objective run's whole front) are evaluated by the simulator at every realization: the
reported objectives are the simulator's, the surrogate's kept beside them.

**A coarse-corrected surrogate** predicts from each column's coarse
forecast. A third objective, the coarse stack on the optimization's executor, shares the
journal (``source = "coarse"``, under a context of the plain one and the coarse stack's).
Per batch its runs come first (made, or replayed from the journal), the domain is checked
with them, and the surrogate is given each new member's coarse forecast; a column whose
coarse run failed is evaluated by the simulator (its row records ``stands_in_for``), and the
failure is remembered, so a resume makes the same choice. The coarse runs are bounded by
``max_evaluations`` with the surrogate's evaluations; each accepted step's history row holds
the coarse objective of its controls (a diagnostic, from the journal). A batch whose new
coarse runs (of two or more distinct controls) all fail stops the run instead (a broken
coarse deck); those failures are retried on resume.
"""

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.optimization import OptimizationResult, OptimizerBase
from apex.algorithms.optimization.multiobjective import hypervolume, non_dominated, representative
from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.statistics import reduce_realizations
from apex.algorithms.surrogates.gate import gate
from apex.core.errors import SurrogateDomainError, WorkflowError
from apex.core.simulation import Simulator
from apex.domain.surrogate import GateResult, GateTolerances, QuantityKind
from apex.storage.optimization_directory import read_evaluations
from apex.workflows.optimization import RestartFromStart
from apex.workflows.simulated_objective import SimulatedObjective

logger = logging.getLogger(__name__)

#: The run directory's record of the surrogate's checks and of a switch to the simulator.
STATE_FILE = "surrogate_state.json"

#: ``(controls (n_x, n), realizations (n,)[, coarse (n_model_keys, n)]) -> (n,)`` whether
#: each column lies inside the model's domain (a coarse-corrected model's with the columns'
#: coarse forecasts); raises nothing.
InsideCheck = Callable[..., NDArray[np.bool_]]

#: ``{member_id: coarse forecast (n_model_keys,)} -> Simulator``: the surrogate's simulator
#: given its members' coarse forecasts.
WithCoarse = Callable[[Mapping[int, NDArray[np.float64]]], Simulator]

#: The rows the simulator evaluates in place of a column whose coarse run failed record this.
STANDS_IN = "stands_in_for"


class SurrogateCheckFailed(WorkflowError):
    """A check of the surrogate against the simulator failed, and the run stops."""


class SurrogateObjective:
    """An optimization's objective evaluated by a surrogate (see the module docstring).

    Parameters
    ----------
    surrogate, simulator
        The two objectives (sharing one journal; the surrogate's ``source`` is
        ``"surrogate"``, the simulator's ``"simulator"``).
    run_dir
        The optimization's run directory (``surrogate_state.json``).
    identity
        The model files' identity.
    inside
        Which columns lie inside the model's domain.
    outside
        ``stop`` or ``simulator`` for columns outside.
    check_every
        Accepted steps between checks (0: none).
    on_check_failure
        ``stop`` or ``simulator``.
    tolerances, kinds
        The checks' gate.
    max_evaluations
        The run's bound on surrogate evaluations (and on coarse runs).
    coarse
        A coarse-corrected model's coarse stack (``source = "coarse"``, failures tolerated),
        sharing the journal; none for any other model.
    coarse_rows
        For each of the model's keys, its row among the coarse stack's response keys.
    with_coarse
        Makes the surrogate's simulator given its members' coarse forecasts.
    """

    def __init__(
        self,
        surrogate: SimulatedObjective,
        simulator: SimulatedObjective,
        run_dir: Path,
        *,
        identity: str,
        inside: InsideCheck,
        outside: Literal["stop", "simulator"] = "stop",
        check_every: int = 3,
        on_check_failure: Literal["stop", "simulator"] = "stop",
        tolerances: GateTolerances | None = None,
        kinds: Mapping[str, QuantityKind] | None = None,
        max_evaluations: int = 50_000,
        coarse: SimulatedObjective | None = None,
        coarse_rows: Sequence[int] = (),
        with_coarse: WithCoarse | None = None,
    ) -> None:
        if surrogate.journal is not simulator.journal:
            raise ValueError("the surrogate and the simulator must share one journal")
        if coarse is not None:
            if coarse.journal is not simulator.journal:
                raise ValueError("the coarse stack must share the run's journal")
            if with_coarse is None or not coarse_rows:
                raise ValueError("a coarse stack needs coarse_rows and with_coarse")
            if not coarse.tolerate_failures:
                raise ValueError("the coarse stack must tolerate failures (a stand-in follows)")
        self.coarse = coarse
        self.coarse_rows = np.asarray(coarse_rows, dtype=np.int64)
        self.with_coarse = with_coarse
        if surrogate.n_realizations != simulator.n_realizations:
            raise ValueError("the surrogate and the simulator see different realizations")
        self.surrogate = surrogate
        self.simulator = simulator
        self.run_dir = run_dir
        self.identity = identity
        self.inside = inside
        self.outside = outside
        self.check_every = check_every
        self.on_check_failure = on_check_failure
        self.tolerances = tolerances or GateTolerances()
        self.kinds = dict(kinds or {})
        self.max_evaluations = max_evaluations
        self.n_realizations = simulator.n_realizations
        self.vector = simulator.vector
        if simulator.vector:
            self.objective_names = simulator.objective_names
            self.objective_statistics = simulator.objective_statistics
        state = self._read_state()
        #: Whether the simulator evaluates everything (after a failed check).
        self.switched = bool(state.get("switched", False))
        if state.get("identity") not in (None, identity):
            raise WorkflowError(
                f"{run_dir} was optimized with another surrogate ({state.get('identity')}); "
                "a resume needs the same model file"
            )
        #: Whether the run's state file exists (written at the first evaluation, not when
        #: the objective is built: `apex validate` builds it too, and writes nothing).
        self._recorded = bool(state)

    # ---------------------------------------------------------------- evaluation

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the objective of each column (``(n,)``, or ``(n_obj, n)``).

        Raises
        ------
        SurrogateDomainError
            If a column lies outside the model's domain and ``outside`` is ``stop``.
        WorkflowError
            If the surrogate's evaluations would exceed ``max_evaluations``, or an
            evaluation fails.
        """
        controls = np.ascontiguousarray(np.asarray(controls, dtype=np.float64))
        n = controls.shape[1]
        reals = np.ascontiguousarray(
            np.broadcast_to(np.asarray(realizations, dtype=np.int64), (n,))
        )
        if not self._recorded:
            # The record that the run used a surrogate: a resume without its section is
            # refused, its checkpoints holding the surrogate's values.
            self._write_state(checks=[])
            self._recorded = True
        if self.switched:
            return self.simulator(controls, reals)
        lost = np.zeros(n, dtype=bool)
        coarse = None
        if self.coarse is not None:
            self._charge_coarse(controls, reals)
            coarse, lost = self._coarse_forecasts(controls, reals, run=True)
        ok = ~lost
        if coarse is None:
            inside = np.asarray(self.inside(controls, reals), dtype=bool)
        else:
            inside = np.zeros(n, dtype=bool)
            if np.any(ok):
                inside[ok] = self._inside(controls[:, ok], reals[ok], coarse, ok)
        outside = ok & ~inside
        if np.any(outside) and self.outside == "stop":
            out = np.flatnonzero(outside)
            raise SurrogateDomainError(
                f"{out.size} of {n} control vectors lie outside the surrogate's validated "
                f"domain (columns {out[:10].tolist()}): the optimization stops at its last "
                'checkpoint; set [surrogate] outside = "simulator" to evaluate them by the '
                "simulator, or build the model over a wider box",
                members=tuple(int(j) for j in out),
            )
        new = sum(
            1
            for key in set(self.surrogate.keys_of(controls[:, inside], reals[inside]))
            if key not in self.surrogate.journal.values
        )
        done = self.surrogate.journal.by_source.get("surrogate", 0)
        if done + new > self.max_evaluations:
            raise WorkflowError(
                f"the surrogate's evaluations would exceed [surrogate] max_evaluations = "
                f"{self.max_evaluations} ({done} made, {new} more asked for)"
            )
        shape = (len(self.objective_names), n) if self.vector else (n,)
        values = np.empty(shape)
        if np.any(inside):
            values[..., inside] = self._surrogate_values(
                controls[:, inside], reals[inside], None if coarse is None else coarse[:, inside]
            )
        if np.any(outside):
            logger.info(
                "surrogate_outside_simulated", extra={"columns": int(np.count_nonzero(outside))}
            )
            values[..., outside] = self.simulator(controls[:, outside], reals[outside])
        if np.any(lost):
            # A failed coarse run leaves the
            # surrogate nothing to correct; the simulator evaluates the column.
            logger.warning(
                "surrogate_coarse_failed_simulated",
                extra={"columns": int(np.count_nonzero(lost))},
            )
            values[..., lost] = self.simulator(
                controls[:, lost], reals[lost], row_extra={STANDS_IN: "coarse"}
            )
        return np.ascontiguousarray(values)

    # ------------------------------------------------------------- coarse stack

    def _charge_coarse(self, controls: FloatArray, reals: IntArray) -> None:
        """Refuse a batch whose new coarse runs would exceed ``max_evaluations``.

        Raises
        ------
        WorkflowError
            If the coarse rows made (failed ones included) and the new ones exceed it.
        """
        assert self.coarse is not None
        journal = self.coarse.journal
        known = journal.values
        new = sum(
            1
            for key in set(self.coarse.keys_of(controls, reals))
            if key not in known and key not in journal.failed
        )
        made = journal.rows_by_source.get("coarse", 0)
        if made + new > self.max_evaluations:
            raise WorkflowError(
                f"the coarse stack's runs would exceed [surrogate] max_evaluations = "
                f"{self.max_evaluations} ({made} made, {new} more asked for)"
            )

    def _coarse_forecasts(
        self, controls: FloatArray, reals: IntArray, *, run: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return the columns' coarse forecasts ``(n_model_keys, n)`` and which have none.

        With ``run``, the columns not in the journal are run first (failures journaled);
        without, only the journal is read. A column without a forecast is NaN.
        """
        assert self.coarse is not None
        keys = self.coarse.run(controls, reals) if run else self.coarse.keys_of(controls, reals)
        responses = self.coarse.journal.responses
        lost = np.array([key not in responses for key in keys], dtype=bool)
        forecasts = np.full((self.coarse_rows.size, len(keys)), np.nan)
        for j, key in enumerate(keys):
            if not lost[j]:
                forecasts[:, j] = np.asarray(responses[key], dtype=np.float64)[self.coarse_rows]
        return np.ascontiguousarray(forecasts), lost

    def _inside(
        self,
        controls: FloatArray,
        reals: IntArray,
        coarse: NDArray[np.float64] | None,
        columns: NDArray[np.bool_],
    ) -> NDArray[np.bool_]:
        """Return which columns lie inside the domain (``coarse``: every column's, sliced)."""
        if coarse is None:
            return np.asarray(self.inside(controls, reals), dtype=bool)
        sliced = np.ascontiguousarray(coarse[:, columns])
        return np.asarray(self.inside(controls, reals, sliced), dtype=bool)

    def _surrogate_values(
        self, controls: FloatArray, reals: IntArray, coarse: NDArray[np.float64] | None
    ) -> FloatArray:
        """Evaluate columns by the surrogate (given their coarse forecasts, if any)."""
        if coarse is None:
            return self.surrogate(controls, reals)
        return self.surrogate(controls, reals, simulator_for=self._simulator_for(coarse))

    def _simulator_for(
        self, coarse: NDArray[np.float64]
    ) -> Callable[[Mapping[int, int]], Simulator]:
        """Return the hook giving the surrogate its new members' coarse forecasts."""
        with_coarse = self.with_coarse
        assert with_coarse is not None

        def simulator_for(members: Mapping[int, int]) -> Simulator:
            return with_coarse({
                member: np.ascontiguousarray(coarse[:, column])
                for member, column in members.items()
            })  # fmt: skip

        return simulator_for

    def _coarse_objective(self, x: FloatArray) -> dict[str, Any]:
        """Return the history row's coarse objective of ``x`` ``(n_x,)``, from the journal.

        One objective: the mean over the realizations (as the result's), ``coarse_objective``;
        several: each objective's statistic, ``coarse_objective:<name>``. ``None`` when a
        realization's coarse run failed or was not made (no run is made for a diagnostic).
        """
        assert self.coarse is not None
        reals = np.arange(self.n_realizations, dtype=np.int64)
        keys = self.coarse.keys_of(self._columns(x, reals), reals)
        known = self.coarse.journal.values
        names = (
            [f"coarse_objective:{n}" for n in self.objective_names]
            if self.vector
            else ["coarse_objective"]
        )
        if any(key not in known for key in keys):
            return dict.fromkeys(names)
        if self.vector:
            per = np.array([known[key] for key in keys], dtype=np.float64).T  # (n_obj, n_r)
            ranked = reduce_realizations(per[:, :, None], self.objective_statistics)
            return {n: float(v) for n, v in zip(names, ranked[:, 0], strict=True)}
        return {names[0]: float(np.mean([float(known[key]) for key in keys]))}  # type: ignore[arg-type]

    # -------------------------------------------------------------------- checks

    def after_step(self, optimizer: OptimizerBase, x: FloatArray) -> dict[str, Any]:
        """Check an accepted step's controls ``x`` ``(n_x,)`` if it is a check step.

        Returns the history row's ``surrogate_check`` (empty if no check was due).

        Raises
        ------
        SurrogateCheckFailed
            If the check fails and ``on_check_failure`` is ``stop``.
        RestartFromStart
            If it fails and ``on_check_failure`` is ``simulator`` (the switch is recorded).
        """
        if self.switched:
            return {}
        x = np.asarray(x, dtype=np.float64)
        due = bool(self.check_every) and optimizer.iteration % self.check_every == 0
        reals = np.arange(self.n_realizations, dtype=np.int64)
        controls = np.ascontiguousarray(np.repeat(x.reshape(-1, 1), reals.size, axis=1))
        coarse, lost = None, np.zeros(reals.size, dtype=bool)
        if self.coarse is not None and due:
            # The check's coarse runs first, then the diagnostic below: a resume replaying
            # this step after a kill during the check finds the same rows (the review).
            coarse, lost = self._coarse_forecasts(controls, reals, run=True)
        # the coarse objective of every accepted step, a diagnostic.
        extra = {} if self.coarse is None else self._coarse_objective(x)
        if not due:
            return extra
        reason = None
        every = np.ones(reals.size, dtype=bool)
        if np.any(lost):
            reason = "a coarse run of the accepted controls failed: not checked"
        if reason is None and not np.all(self._inside(controls, reals, coarse, every)):
            # Evaluated by the simulator (outside = "simulator"): nothing of the surrogate's
            # to check there, and the model refuses to predict it.
            reason = "the accepted controls lie outside the model's domain"
        if reason is not None:
            skipped = {
                "epf_iteration": optimizer.epf_iteration,
                "iteration": optimizer.iteration,
                "passed": None,
                "skipped": reason,
            }
            self._record(skipped)
            return {**extra, "surrogate_check": skipped}
        result = self.check(x)
        record = {
            "epf_iteration": optimizer.epf_iteration,
            "iteration": optimizer.iteration,
            "passed": result.passed,
            "failures": result.failures(),
        }
        self._record(record)
        logger.info(
            "surrogate_check",
            extra={"iteration": optimizer.iteration, "passed": result.passed},
        )
        if result.passed:
            return {**extra, "surrogate_check": record}
        if self.on_check_failure == "stop":
            raise SurrogateCheckFailed(
                f"the surrogate's check at step {optimizer.iteration} failed ("
                + "; ".join(result.failures())
                + "): the optimization stopped at its last checkpoint; resume it after "
                'setting on_check_failure = "simulator" in its config.json (a rebuilt model '
                "is another model: start a new optimization with it)"
            )
        self.switched = True
        # Pending until the workflow has removed the old source's checkpoint: a process that
        # ends in between restarts on resume, never continues that checkpoint.
        self._write_state(switched=True, switched_at=optimizer.iteration, restart_pending=True)
        logger.warning("surrogate_switched_to_simulator", extra={"iteration": optimizer.iteration})
        raise RestartFromStart(
            f"the surrogate's check at step {optimizer.iteration} failed; the optimization "
            "starts again with the simulator"
        )

    def check(self, x: FloatArray) -> GateResult:
        """Gate the surrogate on ``x`` ``(n_x,)`` at every realization against the simulator."""
        reals = np.arange(self.n_realizations, dtype=np.int64)
        controls = np.ascontiguousarray(np.repeat(x.reshape(-1, 1), reals.size, axis=1))
        simulated = self.simulator.responses(controls, reals)
        hook = None
        if self.coarse is not None:
            coarse, lost = self._coarse_forecasts(controls, reals, run=True)
            if np.any(lost):
                raise WorkflowError("a check needs the coarse forecasts of its controls")
            hook = self._simulator_for(coarse)
        predicted = self.surrogate.responses(controls, reals, simulator_for=hook)
        return gate(
            predicted,
            simulated,
            self.surrogate.keys,
            [int(r) for r in reals],
            tolerances=self.tolerances,
            kinds=self.kinds,
        )

    # -------------------------------------------------------------------- result

    def finish(
        self, result: OptimizationResult, summary: Mapping[str, Any], optimizer: OptimizerBase
    ) -> dict[str, Any]:
        """Return ``result.json`` with the optimum evaluated by the simulator.

        The result's controls, at every realization. A multi-objective run's front is
        evaluated member by member (``pareto.json`` rewritten, :meth:`_front`), and its
        representative, chosen again on the simulator's values, is the result's ``x``. The
        surrogate's values are kept under ``surrogate_*``.
        """
        reals = np.arange(self.n_realizations, dtype=np.int64)
        updated = dict(summary)
        # After a switch every value the optimizer saw was the simulator's: no surrogate_*.
        surrogate = not self.switched
        if surrogate:
            updated["surrogate_objective"] = summary["objective"]
            updated["surrogate_objective_per_realization"] = summary["objective_per_realization"]
        x = np.asarray(summary["x"], dtype=np.float64)
        front = getattr(optimizer, "front", None) if self.vector else None
        if self.vector:
            n_obj = len(self.objective_names)
            if callable(front):
                xs, per_member, chosen = self._front(front(), reals)
                x = xs[chosen]
                per_realization = per_member[:, :, chosen]
            else:
                values = self.simulator(self._columns(x, reals), reals)
                per_realization = np.asarray(values, dtype=np.float64).reshape(n_obj, reals.size)
            ranked = reduce_realizations(per_realization[:, :, None], self.objective_statistics)
            updated["x"] = [float(v) for v in x]
            if surrogate:
                updated["surrogate_x"] = summary["x"]
                updated["surrogate_objectives"] = summary.get("objectives")
                updated["surrogate_objectives_per_realization"] = summary.get(
                    "objectives_per_realization"
                )
            updated["objectives"] = [float(v) for v in ranked[:, 0]]
            updated["objectives_per_realization"] = per_realization.tolist()
            updated["objective"] = float(ranked[0, 0])
            updated["objective_per_realization"] = [float(v) for v in per_realization[0]]
        else:
            simulated = np.asarray(self.simulator(self._columns(x, reals), reals), np.float64)
            updated["objective"] = float(np.mean(simulated))
            updated["objective_per_realization"] = [float(v) for v in simulated]
        state = self._read_state()
        updated["surrogate"] = {
            "identity": self.identity,
            "evaluations": self.surrogate.journal.by_source.get("surrogate", 0),
            "simulator_evaluations": self.simulator.journal.by_source.get("simulator", 0),
            "checks": len(state.get("checks", [])),
            "failed_checks": sum(c.get("passed") is False for c in state.get("checks", [])),
            "switched_to_simulator": self.switched,
            "objective_source": "simulator",
        }
        if self.coarse is not None:
            journal = self.coarse.journal
            made = journal.by_source.get("coarse", 0)
            updated["surrogate"] |= {
                "coarse_evaluations": made,
                "coarse_failed": journal.rows_by_source.get("coarse", 0) - made,
                "coarse_stand_ins": sum(
                    1 for row in read_evaluations(self.run_dir) if row.get(STANDS_IN) is not None
                ),
            }
        return updated

    def restart_pending(self) -> bool:
        """Whether the switch's restart from the start is still to be carried out."""
        return bool(self._read_state().get("restart_pending", False))

    def restarted(self) -> None:
        """Record that the switch's restart was carried out."""
        if self.restart_pending():
            self._write_state(restart_pending=False)

    @staticmethod
    def _columns(x: FloatArray, reals: IntArray) -> FloatArray:
        """Return ``x`` ``(n_x,)`` once per realization, ``(n_x, n_r)``."""
        return np.ascontiguousarray(np.repeat(np.asarray(x).reshape(-1, 1), reals.size, axis=1))

    def _front(
        self, front: Mapping[str, Any], reals: IntArray
    ) -> tuple[FloatArray, FloatArray, int]:
        """Rewrite ``pareto.json`` with the front's members evaluated by the simulator.

        Returns the members' controls ``(m, n_x)``, their values ``(n_obj, n_r, m)`` and the
        representative's position. Every member the surrogate's front held stays listed, its
        objectives the simulator's; those the simulator's values dominate are named in
        ``dominated_by_simulator`` (the surrogate's ranking was off there), and the
        representative is chosen again among the others (the rule). The hypervolume
        is recomputed from the same reference point (unpenalised: under EPF the surrogate's,
        kept as ``surrogate_hypervolume``, included the penalty). ``violation`` is the
        surrogate's (``violation_source``).
        """
        from apex.storage import optimization_directory as layout

        xs = np.asarray(front["x"], dtype=np.float64).reshape(len(front["x"]), -1)
        n_obj = len(self.objective_names)
        per = [
            np.asarray(self.simulator(self._columns(x, reals), reals), np.float64).reshape(
                n_obj, reals.size
            )
            for x in xs
        ]
        stacked = np.stack(per, axis=-1) if per else np.zeros((n_obj, reals.size, 0))
        ranked = reduce_realizations(stacked, self.objective_statistics)
        kept = non_dominated(ranked)
        chosen = int(kept[representative(ranked[:, kept])]) if kept.size else 0
        rewritten = dict(front)
        rewritten["surrogate_objectives"] = front.get("objectives")
        rewritten["surrogate_hypervolume"] = front.get("hypervolume")
        rewritten["surrogate_representative"] = front.get("representative")
        rewritten["objectives"] = ranked.T.tolist()
        rewritten["objectives_per_realization"] = stacked.transpose(2, 0, 1).tolist()
        rewritten["dominated_by_simulator"] = sorted(set(range(xs.shape[0])) - set(kept.tolist()))
        rewritten["representative"] = chosen
        reference = front.get("reference_point")
        if reference is not None and ranked.size:
            rewritten["hypervolume"] = float(hypervolume(ranked, reference))
        rewritten["objective_source"] = "simulator"
        rewritten["violation_source"] = "surrogate"
        layout.write_pareto(self.run_dir, rewritten)
        return xs, stacked, chosen

    # --------------------------------------------------------------------- state

    def _record(self, record: Mapping[str, Any]) -> None:
        """Record a check (once per step: a repeated step replaces its record)."""
        state = self._read_state()

        # Keyed by the EPF loop and the step: iterations start again in each EPF loop.
        def key(check: Mapping[str, Any]) -> tuple[object, object]:
            return check.get("epf_iteration"), check.get("iteration")

        checks = [c for c in state.get("checks", []) if key(c) != key(record)]
        self._write_state(checks=[*checks, dict(record)])

    def _read_state(self) -> dict[str, Any]:
        path = self.run_dir / STATE_FILE
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise WorkflowError(f"cannot read {path}: {error}") from error
        return data if isinstance(data, dict) else {}

    def _write_state(self, **changes: object) -> None:
        state = self._read_state() | {"identity": self.identity, **changes}
        path = self.run_dir / STATE_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=1), encoding="utf-8")
        os.replace(temporary, path)


def checks_of(run_dir: Path) -> Sequence[Mapping[str, Any]]:
    """Return a run's recorded checks of its surrogate (none without one)."""
    path = run_dir / STATE_FILE
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    checks = data.get("checks", []) if isinstance(data, dict) else []
    return [c for c in checks if isinstance(c, dict)]
