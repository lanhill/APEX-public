"""The optimization workflow: run an optimizer with checkpoints and a history.

The optimizer calls back after its start, after every accepted step and after every EPF
update; each call appends a row to ``history.jsonl`` and checkpoints the optimizer's state
(with its random stream). A resumed run reloads the checkpoint, keeps the history rows up to
it, and continues bit-identically. ``max_iterations`` stops after that many accepted steps
in this invocation, leaving a resumable run; so does a stop request (``stop_requested`` in
the run directory, written by ``apex stop`` or the job service), at the next checkpoint. A
resume of a finished run returns its result without running.
"""

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from apex.algorithms.optimization import Draws, OptimizationResult, OptimizerBase, Problem
from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.statistics import reduce_realizations
from apex.core.errors import WorkflowError
from apex.storage import optimization_directory as layout
from apex.storage.locks import exclusive

logger = logging.getLogger(__name__)


class OptimizationHooks(Protocol):
    """What a run with a surrogate adds to the loop.

    ``after_step`` is called on each accepted step, before its history row and checkpoint
    are written, with the step's physical controls; it returns what the row records besides
    and may raise :class:`RestartFromStart` (the run then starts again from its start, its
    journal kept) or another error (the run stops at its last checkpoint). ``finish``
    returns the finished run's ``result.json``.

    Optional, read with ``getattr``: ``bind(optimizer)``, called
    before the loop, for hooks that follow the optimizer's iterations;
    ``checkpoint_state() -> str | None``, loop state saved in every checkpoint under
    :data:`HOOK_STATE` (``None``: nothing saved, the checkpoint's bytes unchanged); and
    ``restore_checkpoint(saved: str | None)``, called with it before the loop (``None``
    on a start, and on a restart from the start); and ``final_columns``, an attribute set
    before ``finish`` to the run's final ensemble (:meth:`OptimizerBase.final_ensemble`,
    drawn once by the workflow; ``None`` without one).
    """

    def after_step(self, optimizer: OptimizerBase, x: FloatArray) -> Mapping[str, Any]:
        """Check the accepted step ``x``; return the history row's additions."""
        ...

    def finish(
        self, result: OptimizationResult, summary: Mapping[str, Any], optimizer: OptimizerBase
    ) -> dict[str, Any]:
        """Return the finished run's ``result.json``."""
        ...

    def restart_pending(self) -> bool:
        """Whether a restart from the start was decided and not yet carried out."""
        ...

    def restarted(self) -> None:
        """Record that the restart was carried out (the checkpoint and history are gone)."""
        ...


class RestartFromStart(WorkflowError):
    """The optimization must start again from its start (its journal kept).

    A checkpoint holds the optimizer's objective values, which a switch of the objective's
    source would leave stale: only a start from scratch is the new source's run.
    """


#: The checkpoint key holding how many history rows belong to it.
HISTORY_ROWS = "history_rows"
#: The checkpoint key holding the hooks' own loop state.
HOOK_STATE = "hooks"
#: The file a stop request writes in the run directory (as for history-matching runs).
STOP_FILE = "stop_requested"


@dataclass(frozen=True)
class OptimizationRun:
    """What an invocation of the workflow produced.

    Attributes
    ----------
    run_dir
        The run directory.
    result
        The optimizer's result (physical controls).
    finished
        Whether the run ended (else it stopped at ``max_iterations`` and can be resumed).
    """

    run_dir: Path
    result: OptimizationResult
    finished: bool


def run_optimization(
    run_dir: Path,
    optimizer: OptimizerBase,
    problem: Problem,
    draws: Draws | None,
    *,
    cov: FloatArray | None = None,
    config: Mapping[str, Any] | None = None,
    resume: bool = False,
    max_iterations: int | None = None,
    stopped: Callable[[], bool] | None = None,
    hooks: OptimizationHooks | None = None,
    fresh_draws: Callable[[], Draws | None] | None = None,
    final_draws: Callable[[], Draws] | None = None,
    objective_scaling: float | None = None,
) -> OptimizationRun:
    """Run, or resume, an optimization in ``run_dir``.

    Parameters
    ----------
    run_dir
        The run directory (created).
    optimizer, problem, draws, cov
        What :meth:`OptimizerBase.minimize` takes.
    config
        The resolved configuration, written to ``config.json`` on a new run.
    resume
        Continue from the last checkpoint.
    max_iterations
        Stop after this many accepted steps in this invocation.
    stopped
        Whether the job running it was asked to stop: requested again once the lock is
        held, after the last invocation's request is withdrawn.
    hooks
        A surrogate's checks and final evaluation.
    fresh_draws
        The run's random stream at its start, for :class:`RestartFromStart` (default: a
        copy of ``draws`` taken here, which a resume from a checkpoint cannot give).
    final_draws
        The final ensemble's own stream, drawn once when the run
        finishes, so a resume draws the same; ``None``: no final ensemble.
    objective_scaling
        The objective's ``obj_scaling`` (its value in its own units is objective * it), for
        the final ensemble's record; ``None``: 1.

    Raises
    ------
    WorkflowError
        A new run in a directory that has one, or a resume without a run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive(run_dir, "optimization"):
        state: dict[str, object] | None = None
        # A request made before this invocation is the one that stopped the last.
        (run_dir / STOP_FILE).unlink(missing_ok=True)
        if stopped is not None and stopped():
            (run_dir / STOP_FILE).write_text("stop at the next checkpoint\n")
        if resume:
            if not (run_dir / layout.CONFIG).exists():
                raise WorkflowError(f"no optimization run to resume in {run_dir}")
            if (run_dir / layout.RESULT).exists():
                return OptimizationRun(run_dir, finished_result(run_dir), True)
            state = layout.load_checkpoint(run_dir)
            layout.truncate_history(run_dir, int(state[HISTORY_ROWS]) if state else 0)  # type: ignore[call-overload]
            if hooks is not None and hooks.restart_pending():
                # A restart decided before the process ended (the switch is recorded first):
                # carried out now, never a resume of the old source's checkpoint.
                _reset(run_dir)
                hooks.restarted()
                state = None
        else:
            if (run_dir / layout.CONFIG).exists():
                raise WorkflowError(
                    f"{run_dir} already holds an optimization run; resume it or remove it"
                )
            layout.write_config(run_dir, dict(config or {}))
        rows = len(layout.read_history(run_dir))
        accepted = 0
        # Only a run that may restart needs its stream's start (plain runs copy nothing).
        initial_draws = copy.deepcopy(draws) if state is None and hooks is not None else None
        # This invocation's accepted steps up to its last checkpoint (a restart's start).
        saved_accepted = 0

        def on_iteration(opt: OptimizerBase) -> bool:
            nonlocal rows, accepted, saved_accepted
            kind = "step" if opt.iteration > 0 else ("start" if rows == 0 else "epf_update")
            x = opt.bounds.to_physical(opt.bounds.project(opt.xk))
            extra: Mapping[str, Any] = {}
            if kind == "step" and hooks is not None:
                # Before the row and the checkpoint: a check that fails leaves the last
                # checkpoint as the one to resume (or restart) from.
                extra = hooks.after_step(opt, x)
            if kind == "step":
                accepted += 1
            layout.append_history(
                run_dir,
                {
                    "kind": kind,
                    "iteration": opt.iteration,
                    "epf_iteration": opt.epf_iteration,
                    "x": [float(v) for v in x],
                    "nfev": opt.nfev,
                    "njev": opt.njev,
                    "member_evaluations": opt.member_evaluations,
                    **opt.log_columns(),
                    **extra,
                },
            )
            rows += 1
            checkpoint: dict[str, object] = dict(opt.export_state())
            checkpoint[HISTORY_ROWS] = rows
            hook_state = _hook_state(hooks)
            if hook_state is not None:
                checkpoint[HOOK_STATE] = hook_state
            layout.save_checkpoint(run_dir, checkpoint)
            saved_accepted = accepted
            front = getattr(opt, "front", None)
            if callable(front):  # a multi-objective run: the checkpoint's front
                layout.write_pareto(run_dir, front())
            if (run_dir / STOP_FILE).exists():
                logger.info("optimization_stop_requested", extra={"run_dir": str(run_dir)})
                return False
            return not (max_iterations is not None and accepted >= max_iterations)

        optimizer.stop_check = lambda: (run_dir / STOP_FILE).exists()
        bind = getattr(hooks, "bind", None)
        if callable(bind):
            bind(optimizer)
        _restore_hooks(hooks, state)
        while True:
            try:
                result = optimizer.minimize(
                    problem,
                    draws,
                    cov=cov,
                    on_iteration=on_iteration,
                    state=state,  # type: ignore[arg-type]
                )
                break
            except RestartFromStart as restart:
                # As a new run with the same seed: no checkpoint, no history, the stream at
                # its start; the journal stays, so what was evaluated is not evaluated again.
                logger.info("optimization_restart", extra={"reason": str(restart)})
                if fresh_draws is not None:
                    draws = fresh_draws()
                elif initial_draws is not None:
                    draws = copy.deepcopy(initial_draws)
                else:
                    raise WorkflowError(
                        "a restart from the start needs the run's initial random stream"
                    ) from restart
                _reset(run_dir)
                if hooks is not None:
                    hooks.restarted()
                state, rows, accepted, saved_accepted = None, 0, 0, 0
                _restore_hooks(hooks, None)
        finished = not optimizer.stopped
        summary = result_summary(result, problem)
        # the ensemble around the final iterate, drawn once here (a
        # stream of its own: the optimizer's are untouched, and a resume draws the same).
        # Without final_draws (the option off, or a caller that asks for none) there is none.
        columns = (
            optimizer.final_ensemble(final_draws())
            if finished and final_draws is not None else None
        )  # fmt: skip
        if finished and hooks is None and columns is not None:
            # A plain run: the objective over it, as the run evaluates it (a simulator's
            # journal, else new runs of the members not in it; an analytic objective).
            try:
                ensemble = final_ensemble(problem, columns, objective_scaling)
            except WorkflowError as error:
                # A failed member must not lose the finished run's result: recorded instead.
                logger.warning("final_ensemble_failed", extra={"reason": str(error)})
                ensemble = {"error": str(error)}
            if ensemble is not None:
                summary["final_ensemble"] = ensemble
        if finished and hooks is not None:
            hooks.final_columns = columns  # type: ignore[attr-defined]
            summary = hooks.finish(result, summary, optimizer)
        if finished:
            layout.write_result(run_dir, summary)
            if hooks is not None:
                # What result.json says (the simulator's optimum), not the optimizer's own.
                result = finished_result(run_dir)
        logger.info(
            "optimization_stopped" if not finished else "optimization_done",
            extra={
                "run_dir": str(run_dir),
                "reason": result.message,
                # As result.json: the first objective's ranked statistic.
                "objective": summary["objective"],
            },
        )
        return OptimizationRun(run_dir, result, finished)


def _hook_state(hooks: OptimizationHooks | None) -> str | None:
    """Return the hooks' loop state for a checkpoint, if they keep one."""
    state_of = getattr(hooks, "checkpoint_state", None)
    if not callable(state_of):
        return None
    state = state_of()
    return None if state is None else str(state)


def _restore_hooks(hooks: OptimizationHooks | None, state: Mapping[str, object] | None) -> None:
    """Give the hooks their loop state from a checkpoint (``None``: a start)."""
    restore = getattr(hooks, "restore_checkpoint", None)
    if not callable(restore):
        return
    saved = None if state is None else state.get(HOOK_STATE)
    restore(None if saved is None else str(saved))


def _reset(run_dir: Path) -> None:
    """Remove a run's checkpoint and history (a restart from its start; the journal stays)."""
    for name in (layout.CHECKPOINT_VALUES, layout.CHECKPOINT_ARRAYS):
        (run_dir / name).unlink(missing_ok=True)
    layout.truncate_history(run_dir, 0)


def finished_result(run_dir: Path) -> OptimizationResult:
    """Return a finished run's result from ``result.json``."""
    data = layout.read_result(run_dir)
    known = {"x", "names", "objective", "objective_per_realization", "iterations",
             "epf_iterations", "nfev", "njev", "nhev", "member_evaluations", "message",
             "objectives", "objectives_per_realization", "objective_names",
             "objective_statistics",
             "representative"}  # fmt: skip
    # Several objectives: the live result's ``(n_obj, n_r)``, not only the first's ``(n_r,)``.
    fun = data.get("objectives_per_realization", data["objective_per_realization"])
    return OptimizationResult(
        x=np.asarray(data["x"], dtype=float),
        fun=np.asarray(fun, dtype=float),
        jac=None,
        hess=None,
        nit=int(data["iterations"]),
        nfev=int(data["nfev"]),
        njev=int(data["njev"]),
        nhev=int(data["nhev"]),
        member_evaluations=int(data["member_evaluations"]),
        message=str(data["message"]),
        epf_iteration=int(data["epf_iterations"]),
        # Numbers only: a surrogate's record (its objectives, the source) is
        # result.json's, not the optimizer's.
        extra={
            k: float(v)
            for k, v in data.items()
            if k not in known
            and not k.startswith("surrogate")
            and isinstance(v, int | float)
            and not isinstance(v, bool)
        },
    )


#: How a multi-objective run's single answer is chosen from its front.
REPRESENTATIVE = "closest to the ideal point in normalised objectives"


def ensemble_statistics(values: FloatArray) -> dict[str, float | int]:
    """Return the statistics of an ensemble's objective values ``(n,)``.

    ``n``, ``mean``, ``std`` (the sample std, ddof = 1; 0 for one member), ``min``, ``max``,
    ``median``, ``p10`` and ``p90``.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


#: The spawn key of the final ensemble's stream (``SeedSequence(seed, spawn_key=(2,))``):
#: apart from the optimizer's (the seed's first child) and the refits' (``(1, n)``).
FINAL_ENSEMBLE_STREAM = 2


def ensemble_record(values: FloatArray, scaling: float | None, source: str) -> dict[str, Any]:
    """Return an ensemble's objective values as ``result.json`` records them.

    ``value``: the statistics (:func:`ensemble_statistics`) of the objective in its own
    units and sign, ``objective * obj_scaling`` (an NPV objective is ``NPV / obj_scaling``:
    this is the NPV; its ``p10`` is the 10th percentile of the NPV); ``minimised``: those of
    the objective the optimizer minimises. External penalties (EPF) are not included.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    factor = 1.0 if scaling is None else float(scaling)
    return {
        "source": source,
        "objective_scaling": scaling,
        "value": ensemble_statistics(v * factor),
        "minimised": ensemble_statistics(v),
        "penalty": "excluded",
    }


def final_ensemble(
    problem: Problem, columns: tuple[FloatArray, IntArray], scaling: float | None = None
) -> dict[str, Any] | None:
    """Return the objective over a final ensemble, or ``None``.

    ``columns``: physical controls ``(n_x, n)`` and realizations ``(n,)``, each column
    evaluated by the run's objective (raw: no external penalty); recorded by
    :func:`ensemble_record`. ``None`` when the objective is a vector or not finite.
    """
    controls, realizations = columns
    values = np.asarray(problem.objective(controls, realizations), dtype=float)
    if values.shape != (controls.shape[1],) or not np.all(np.isfinite(values)):
        return None
    return ensemble_record(values, scaling, "objective") | {"members": int(controls.shape[1])}


def result_summary(result: OptimizationResult, problem: Problem) -> dict[str, Any]:
    """Return the result as JSON values (``result.json``).

    With several objectives ``x`` is the front's representative, ``objective``
    and ``objective_per_realization`` its first objective's (so that single-value views
    keep working), ``objectives`` (with ``objective_names`` and ``objective_statistics``) the
    statistics it was ranked by over the realizations (the means by default) and
    ``objectives_per_realization`` all its values; the front is in ``pareto.json``.
    """
    fun = np.atleast_2d(result.fun)
    several = np.ndim(result.fun) == 2
    names = getattr(problem.objective, "objective_names", None)
    statistics = tuple(getattr(problem.objective, "objective_statistics", ()))
    ranked = (reduce_realizations(fun[:, :, None], statistics)[:, 0]
              if several and len(statistics) == fun.shape[0] else np.mean(fun, axis=1))  # fmt: skip
    return {
        "x": [float(v) for v in result.x],
        "names": list(problem.names),
        "objective": float(ranked[0]),
        "objective_per_realization": [float(v) for v in fun[0]],
        **(
            {
                "objectives": [float(v) for v in ranked],
                **({"objective_statistics": list(statistics)} if statistics else {}),
                "objectives_per_realization": fun.tolist(),
                "objective_names": list(names or ()),
                "representative": REPRESENTATIVE,
            }
            if several
            else {}
        ),
        "iterations": result.nit,
        "epf_iterations": result.epf_iteration,
        "nfev": result.nfev,
        "njev": result.njev,
        "nhev": result.nhev,
        "member_evaluations": result.member_evaluations,
        "message": result.message,
        **result.extra,
    }
