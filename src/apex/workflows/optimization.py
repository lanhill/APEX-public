"""The optimization workflow: run an optimizer with checkpoints and a history.

The optimizer calls back after its start, after every accepted step and after every EPF
update; each call appends a row to ``history.jsonl`` and checkpoints the optimizer's state
(with its random stream). A resumed run reloads the checkpoint, keeps the history rows up to
it, and continues bit-identically. ``max_iterations`` stops after that many accepted steps
in this invocation, leaving a resumable run; so does a stop request (``stop_requested`` in
the run directory, written by ``apex stop`` or the job service), at the next checkpoint. A
resume of a finished run returns its result without running.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apex.algorithms.optimization import Draws, OptimizationResult, OptimizerBase, Problem
from apex.algorithms.optimization.protocols import FloatArray
from apex.algorithms.optimization.statistics import reduce_realizations
from apex.core.errors import WorkflowError
from apex.storage import optimization_directory as layout
from apex.storage.locks import exclusive

logger = logging.getLogger(__name__)

#: The checkpoint key holding how many history rows belong to it.
HISTORY_ROWS = "history_rows"
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
        else:
            if (run_dir / layout.CONFIG).exists():
                raise WorkflowError(
                    f"{run_dir} already holds an optimization run; resume it or remove it"
                )
            layout.write_config(run_dir, dict(config or {}))
        rows = len(layout.read_history(run_dir))
        accepted = 0

        def on_iteration(opt: OptimizerBase) -> bool:
            nonlocal rows, accepted
            kind = "step" if opt.iteration > 0 else ("start" if rows == 0 else "epf_update")
            if kind == "step":
                accepted += 1
            x = opt.bounds.to_physical(opt.bounds.project(opt.xk))
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
                },
            )
            rows += 1
            checkpoint: dict[str, object] = dict(opt.export_state())
            checkpoint[HISTORY_ROWS] = rows
            layout.save_checkpoint(run_dir, checkpoint)
            front = getattr(opt, "front", None)
            if callable(front):  # a multi-objective run: the checkpoint's front
                layout.write_pareto(run_dir, front())
            if (run_dir / STOP_FILE).exists():
                logger.info("optimization_stop_requested", extra={"run_dir": str(run_dir)})
                return False
            return not (max_iterations is not None and accepted >= max_iterations)

        optimizer.stop_check = lambda: (run_dir / STOP_FILE).exists()
        result = optimizer.minimize(
            problem,
            draws,
            cov=cov,
            on_iteration=on_iteration,
            state=state,  # type: ignore[arg-type]
        )
        finished = not optimizer.stopped
        summary = result_summary(result, problem)
        if finished:
            layout.write_result(run_dir, summary)
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
        extra={k: float(v) for k, v in data.items() if k not in known},
    )


#: How a multi-objective run's single answer is chosen from its front.
REPRESENTATIVE = "closest to the ideal point in normalised objectives"


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
