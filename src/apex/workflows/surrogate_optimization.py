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
from apex.domain.surrogate import GateResult, GateTolerances, QuantityKind
from apex.workflows.optimization import RestartFromStart
from apex.workflows.simulated_objective import SimulatedObjective

logger = logging.getLogger(__name__)

#: The run directory's record of the surrogate's checks and of a switch to the simulator.
STATE_FILE = "surrogate_state.json"

#: ``(controls (n_x, n), realizations (n,)) -> (n,)`` whether each column lies inside the
#: model's domain; raises nothing.
InsideCheck = Callable[[FloatArray, IntArray], NDArray[np.bool_]]


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
        The run's bound on surrogate evaluations.
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
    ) -> None:
        if surrogate.journal is not simulator.journal:
            raise ValueError("the surrogate and the simulator must share one journal")
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
        inside = np.asarray(self.inside(controls, reals), dtype=bool)
        if not np.all(inside) and self.outside == "stop":
            out = np.flatnonzero(~inside)
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
            values[..., inside] = self.surrogate(controls[:, inside], reals[inside])
        if not np.all(inside):
            logger.info(
                "surrogate_outside_simulated", extra={"columns": int(np.count_nonzero(~inside))}
            )
            values[..., ~inside] = self.simulator(controls[:, ~inside], reals[~inside])
        return np.ascontiguousarray(values)

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
        if self.switched or not self.check_every or optimizer.iteration % self.check_every:
            return {}
        x = np.asarray(x, dtype=np.float64)
        reals = np.arange(self.n_realizations, dtype=np.int64)
        controls = np.ascontiguousarray(np.repeat(x.reshape(-1, 1), reals.size, axis=1))
        if not np.all(self.inside(controls, reals)):
            # Evaluated by the simulator (outside = "simulator"): nothing of the surrogate's
            # to check there, and the model refuses to predict it.
            skipped = {
                "epf_iteration": optimizer.epf_iteration,
                "iteration": optimizer.iteration,
                "passed": None,
                "skipped": "the accepted controls lie outside the model's domain",
            }
            self._record(skipped)
            return {"surrogate_check": skipped}
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
            return {"surrogate_check": record}
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
        predicted = self.surrogate.responses(controls, reals)
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
