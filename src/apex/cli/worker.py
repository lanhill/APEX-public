"""A job's worker process: ``python -m apex.cli.worker ROOT JOB_ID``.

It runs one job (a run, on APEX or another engine, a resume, a prediction, or
an optimization or its resume),
writes its progress as timestamped lines to ``worker.log`` and every event to the job's
stream in the store, records how the job ended and notifies, and then (without a service)
starts the next queued jobs.
A job pinned to a commit runs this module from its frozen copy, and refuses a
store whose job protocol is not its own. A failure it records is final; a worker that
dies without recording one is resumed by the scheduler.
"""

import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

from apex.cli.progress import LineProgress
from apex.core.engines import EngineContext, EngineMethod
from apex.core.errors import RunStopped
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.domain.prediction import PredictionConfig
from apex.services import composition, policy, prediction
from apex.services.api import engine_plugin
from apex.services.jobs import LocalJobRunner, now
from apex.storage.locks import DirectoryLocked, in_use
from apex.storage.run_directory import RunDirectory
from apex.storage.sqlite_store import JOB_PROTOCOL

#: How long a job waits for its run's lock (``apex gc hosts`` holds it for seconds while
#: it cleans the run's output on the hosts) before it fails.
LOCK_WAIT_S = 60.0


def _when_unlocked[T](
    action: Callable[[], T], stopped: Callable[[], bool] = lambda: False,
    wait_s: float = LOCK_WAIT_S,
) -> T | None:  # fmt: skip
    """Run ``action``, trying again while the run's lock is held, up to ``wait_s``.

    A run another process runs stays refused: its lock is still held after the wait. A
    stop of this job before or while it waits (``stopped``) ends it, and None is returned:
    otherwise a resume would withdraw the stop itself and run to the end. The job's own
    request decides, not the run's stop file, which an earlier stop may have left or
    another process's stop written; the file stays, as after any stop.
    """
    deadline = time.monotonic() + wait_s
    while True:
        # Before each attempt, the first too: a stop asked for before the worker started
        # counts whether the lock is free or not (a resume would withdraw it).
        if stopped():
            return None
        try:
            return action()
        except DirectoryLocked:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5.0)


def _withdraw_unstarted_stop(run_dirs: Sequence[str], job_id: str) -> None:
    """Withdraw the stop a run job got before it started (the review).

    ``JobRunner.stop`` writes the run's stop file while the lock is free; a run that never
    started leaves it in its folder, where the same configuration submitted again would
    find it and stop at its first checkpoint. Only this job's request (``apex stop`` on
    the folder stays), in a folder without a run (no ``state.json``) that no process holds.
    """
    for name in run_dirs:
        run_dir = RunDirectory(Path(name))
        if (
            run_dir.stop_requested_by(f"job {job_id}")
            and not (run_dir.path / "state.json").exists()
            and not in_use(run_dir.path)
        ):
            run_dir.clear_stop()


def _stop_requested(runner: LocalJobRunner, job_id: str) -> Callable[[], bool]:
    """Whether ``JobRunner.stop`` was asked to stop this job (its ``job_stop_requested``)."""
    return lambda: any(e.event == "job_stop_requested" for e in runner.events(job_id))


def run_job(runner: LocalJobRunner, job_id: str) -> None:
    """Run one job to its end and record the outcome."""
    record = runner.record(job_id)
    spec = record["spec"]
    os.environ[composition.JOB_ROOT_ENV] = str(runner.root)
    os.environ[composition.JOB_ID_ENV] = job_id
    progress = LineProgress(sys.stdout)
    try:
        # A worker of other code than the store's newest user (a job pinned to an older
        # commit, after the service moved on to records it would misread) refuses.
        protocol = runner.store.job_protocol()
        if protocol != JOB_PROTOCOL:
            raise RuntimeError(
                f"this worker's code speaks job protocol {JOB_PROTOCOL} and the job store "
                f"{protocol}: the job cannot run on its pinned code; submit it again"
            )
        # The check at the time of use: a path swapped for a symlink, or a run's
        # snapshot changed, since the job was submitted.
        policy.check_job(record["kind"], spec, runner.settings.policy, runner.root)
        if record["kind"] == "run" and spec.get("engine"):
            # Another engine's run: it writes an APEX run directory.
            config = HistoryMatchConfig.model_validate_json(spec["config"])
            method = EngineMethod(**spec["engine"])
            assert config.run.output_dir is not None  # the service placed the run
            result = engine_plugin(method.engine).run(
                config, method, EngineContext(config.run.output_dir, progress)
            )
            state = "complete" if result.complete else "stopped"
        elif record["kind"] == "run":
            config = HistoryMatchConfig.model_validate_json(spec["config"])
            stopped = _stop_requested(runner, job_id)
            waited = _when_unlocked(
                lambda: composition.start_config(
                    config, spec.get("max_steps"), progress, stopped=stopped
                ),
                stopped,
            )
            if waited is None:
                _withdraw_unstarted_stop(record["run_dirs"], job_id)
            state = "complete" if waited is not None and waited.complete else "stopped"
        elif record["kind"] == "resume":
            stopped = _stop_requested(runner, job_id)
            waited = _when_unlocked(
                lambda: composition.resume(
                    Path(spec["run_dir"]), spec.get("max_steps"), progress, stopped=stopped
                ),
                stopped,
            )
            state = "complete" if waited is not None and waited.complete else "stopped"
        elif record["kind"] == "optimization":
            from apex.services import optimization

            run = optimization.optimize_config(
                OptimizationConfig.model_validate_json(spec["config"]),
                spec.get("max_iterations"),
                progress,
                stopped=_stop_requested(runner, job_id),
            )
            state = "complete" if run.finished else "stopped"
        elif record["kind"] == "optimization_resume":
            from apex.services import optimization

            run = optimization.resume(
                Path(spec["run_dir"]),
                spec.get("max_iterations"),
                progress,
                stopped=_stop_requested(runner, job_id),
            )
            state = "complete" if run.finished else "stopped"
        elif record["kind"] == "prediction":
            prediction.predict_config(
                PredictionConfig.model_validate_json(spec["config"]),
                progress,
                stopped=_stop_requested(runner, job_id),
            )
            state = "complete"
        else:
            raise ValueError(f"a worker cannot run a {record['kind']!r} job")
    except RunStopped as stopped:
        # Asked to stop where the run waited outside its steps: resumable, or,
        # with no checkpoint yet, started again (composition withdrew its stop).
        print(stopped, file=sys.stderr)
        state = "stopped"
        ended = runner.update(job_id, state=state, finished=now())
        runner.log_event(job_id, f"job_{state}", reason=str(stopped))
    except Exception as error:  # noqa: BLE001  (any failure ends the job as failed, recorded)
        traceback.print_exc()
        ended = runner.update(
            job_id, state="failed", finished=now(), error=f"{type(error).__name__}: {error}"
        )
        runner.log_event(job_id, "job_failed", error=str(error))
    else:
        ended = runner.update(job_id, state=state, finished=now())
        runner.log_event(job_id, f"job_{state}")
    runner.notify(ended)
    runner.schedule()


def main() -> None:
    """Entry point of the worker process."""
    root, job_id = Path(sys.argv[1]), sys.argv[2]
    try:
        runner = LocalJobRunner(root, role="worker")
    except Exception as error:
        # Without a runner the failure is recorded in the store directly: a worker that only
        # died would be resumed, and die again, with no reason given.
        traceback.print_exc()
        from apex.storage.sqlite_store import SQLiteStore

        reason = f"the worker could not start: {type(error).__name__}: {error}"
        store = SQLiteStore(root)
        store.update_job(job_id, state="failed", finished=now(), error=reason)
        store.add_event(job_id, {"time": now(), "level": "ERROR", "message": "job_failed",
                                 "event": "job_failed", "error": reason})  # fmt: skip
        # No notification: its settings are what could not be read.
        raise SystemExit(1) from error
    run_job(runner, job_id)


if __name__ == "__main__":
    main()
