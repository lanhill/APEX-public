"""Jobs: submitted work that runs in the background.

Job records, their events, approvals and quota use are kept in the root's SQLite store
(``<root>/apex.db``, :mod:`apex.storage.sqlite_store`); each job's worker writes its log
to ``<root>/jobs/<id>/worker.log``. A job runs in a detached worker process
(``python -m apex.cli.worker``, an entry point), so it outlives whoever submitted it.

:meth:`LocalJobRunner.schedule` brings the queue up to date and starts what may start:

- **recovery**: a running job whose worker died without recording an end is queued again
  (resumed from its last checkpoint, which is bit-identical) up to ``max_retries`` times;
  failures a worker records (configuration errors, too many failed members) are final;
- **admission** by reservation, **priorities** and per-client **quotas**
  (:mod:`apex.services.scheduling`), up to ``max_concurrent_jobs``.

Who calls it: with a service running (``apex service start``), only the service does, in
a loop; without one (**embedded mode**), the submitting process and each worker as it
ends do, as in phase 1. Every change is one store transaction, so the two never start a
job twice.
"""

import datetime as dt
import logging
import os
import sqlite3
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from apex.core.jobs import JobRecord
from apex.domain.services import Annotations, Event, JobState, JobStatus
from apex.execution.resources import FilePools
from apex.services.code import CodeError, worker_spawn
from apex.services.notify import notify
from apex.services.scheduling import Quota, select_starts
from apex.services.settings import ServiceSettings, load_settings
from apex.storage.locks import held_by_another
from apex.storage.run_directory import RunDirectory
from apex.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

#: Job states after which nothing more happens.
FINAL: frozenset[str] = frozenset({"complete", "stopped", "failed", "cancelled"})

Role = Literal["client", "worker", "service"]


def now() -> str:
    """Return the current time, ISO 8601 in UTC, to the second."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def alive(pid: int | None) -> bool:
    """Return whether process ``pid`` runs on this machine (a zombie does not)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return True
    return state != "Z"


def service_pid(root: Path) -> int | None:
    """Return the pid of the root's running service (``<root>/service.pid``), or ``None``."""
    path = Path(root) / "service.pid"
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


class LocalJobRunner:
    """Runs jobs in detached worker processes on this machine.

    Parameters
    ----------
    root
        The services root.
    max_concurrent_jobs
        Jobs running at once; default ``[service] max_concurrent_jobs`` of ``service.toml``.
    role
        ``"client"`` or ``"worker"`` schedule only while no service runs (embedded mode);
        ``"service"`` always does.
    """

    def __init__(
        self, root: Path, max_concurrent_jobs: int | None = None, *, role: Role = "client"
    ) -> None:
        if max_concurrent_jobs is not None and max_concurrent_jobs < 1:
            raise ValueError(f"max_concurrent_jobs must be at least 1, got {max_concurrent_jobs}")
        self.root = Path(root)
        self.jobs = self.root / "jobs"
        self.store = SQLiteStore(self.root)
        # A worker reads it leniently: it may run older (pinned) code than wrote it.
        self.settings: ServiceSettings = load_settings(self.root, lenient=role == "worker")
        self.max_concurrent_jobs = max_concurrent_jobs or self.settings.service.max_concurrent_jobs
        self.role = role

    # ------------------------------------------------------------------ records

    def record(self, job_id: str) -> JobRecord:
        """Return a job's record (``KeyError`` if there is none)."""
        return self.store.job(job_id)

    def update(self, job_id: str, **fields: Any) -> JobRecord:  # noqa: ANN401
        """Change a job's record; return it."""
        return self.store.update_job(job_id, **fields)

    def log_event(self, job_id: str, event: str, **fields: Any) -> None:  # noqa: ANN401
        """Append a job event to its stream."""
        self.store.add_event(
            job_id, {"time": now(), "level": "INFO", "message": event, "event": event, **fields}
        )

    def all_records(self) -> list[JobRecord]:
        """Return every job, oldest first."""
        return self.store.jobs()

    # ------------------------------------------------------------------ submitting

    def submit(
        self,
        kind: str,
        spec: dict[str, Any],
        *,
        run_ids: Sequence[str] = (),
        run_dirs: Sequence[str] = (),
        annotations: Annotations | None = None,
        after: Sequence[str] = (),
        parent: str | None = None,
        state: JobState = "queued",
        start: bool = True,
        priority: str = "normal",
        client: str = "user",
        reservation: dict[str, int] | None = None,
        planned_runs: int | None = None,
        code: dict[str, str] | None = None,
    ) -> str:
        """Queue a job and start what can start; return its id.

        ``code`` pins the job to a frozen copy (:class:`apex.services.code.CodePin`
        as JSON); it is a field of the record, not of the spec, so that the automatic resume
        of a dead worker, which rewrites the spec, keeps it.
        """
        job_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        record = {
            "id": job_id,
            "kind": kind,
            "spec": spec,
            "state": state,
            "created": now(),
            "started": None,
            "finished": now() if state in FINAL else None,
            "error": None,
            "pid": None,
            "run_ids": list(run_ids),
            "run_dirs": list(run_dirs),
            "annotations": dict(annotations or {}),
            "after": list(after),
            "parent": parent,
            "children": [],
            "priority": priority,
            "client": client,
            "reservation": dict(reservation or {}),
            "planned_runs": planned_runs or 0,
            "retries": 0,
            "code": code,
        }
        with self.store.locked():
            self.store.insert_job(record)
            if parent is not None:
                owner = self.store.job(parent)
                self.store.update_job(parent, children=[*owner["children"], job_id])
        self.log_event(job_id, "job_" + ("queued" if state == "queued" else state))
        if start:
            self.schedule()
        return job_id

    # ------------------------------------------------------------------ scheduling

    def may_schedule(self) -> bool:
        """Return whether this process schedules: always the service; others without one."""
        return self.role == "service" or service_pid(self.root) is None

    def schedule(self) -> list[str]:
        """Recover dead jobs and start queued ones while they fit; return the ids started."""
        if not self.may_schedule():
            return []
        started: list[str] = []
        ended: list[JobRecord] = []
        failed_at_start: list[tuple[str, str]] = []
        pools = FilePools().sizes()
        quotas = {
            client: Quota(q.max_concurrent_jobs, q.max_simulator_runs_per_day)
            for client, q in self.settings.quotas.items()
        }
        today = dt.datetime.now(dt.UTC).date().isoformat()
        with self.store.locked():
            ended += self._recover()
            ended += self._cancel_orphans()
            queued = [r for r in self.store.jobs({"queued"}) if r["kind"] != "study"]
            running = [r for r in self.store.jobs({"running"}) if r["kind"] != "study"]
            states = {r["id"]: r["state"] for r in self.store.jobs()}
            for job_id in select_starts(
                queued,
                running,
                pools=pools,
                max_concurrent_jobs=self.max_concurrent_jobs,
                quotas=quotas,
                usage_today=lambda client: self.store.usage(client, today),
                finished=states.get,
            ):
                record = self.store.job(job_id)
                try:
                    pid = self._spawn(record)
                except (CodeError, OSError) as error:
                    # Its interpreter is gone, or cannot run: this job fails, the others go on.
                    message = f"{type(error).__name__}: {error}"
                    ended.append(
                        self.store.update_job(job_id, state="failed", finished=now(), error=message)
                    )
                    failed_at_start.append((job_id, message))
                    continue
                self.store.update_job(job_id, state="running", started=now(), pid=pid)
                client = str(record.get("client", "user"))
                self.store.add_usage(client, today, int(record.get("planned_runs") or 0))
                started.append(job_id)
            ended += self._finish_studies()
        for job_id in started:
            self.log_event(job_id, "job_started")
        for job_id, message in failed_at_start:
            self.log_event(job_id, "job_failed", error=message)
        for record in ended:
            self.notify(record)
        return started

    def _recover(self) -> list[JobRecord]:
        """Queue again (or fail, after the retries) running jobs whose worker died."""
        ended = []
        for record in self.store.jobs({"running"}):
            if record["kind"] == "study" or alive(record.get("pid")):
                continue
            retries = int(record.get("retries", 0))
            if retries >= self.settings.service.max_retries:
                error = f"the worker process died ({retries} automatic resumes used)"
                ended.append(
                    self.store.update_job(record["id"], state="failed", finished=now(), error=error)
                )
                continue
            kind, spec = record["kind"], dict(record["spec"])
            run_dirs = record.get("run_dirs") or []
            checkpointed = bool(run_dirs) and (Path(run_dirs[0]) / "state.json").exists()
            if kind in ("run", "resume") and checkpointed:
                # Continue from the last checkpoint (bit-identical to an uninterrupted run).
                kind, spec = "resume", {
                    "run_dir": run_dirs[0], "max_steps": spec.get("max_steps"),
                    # The owner's web UI still shows a reuse's job by it (OwnerView).
                    **({"requested_dir": spec["requested_dir"]} if "requested_dir" in spec else {}),
                }  # fmt: skip
            self.store.update_job(
                record["id"],
                state="queued",
                kind=kind,
                spec=spec,
                pid=None,
                started=None,
                retries=retries + 1,
            )
            self.log_event(record["id"], "job_recovered", retry=retries + 1, as_kind=kind)
            logger.warning(
                "job %s: its worker died; queued again (retry %d)",
                record["id"],
                retries + 1,
                extra={"event": "job_recovered", "job": record["id"]},
            )
        return ended

    def _cancel_orphans(self) -> list[JobRecord]:
        """Cancel queued jobs whose dependencies ended without completing."""
        ended = []
        states = {r["id"]: r["state"] for r in self.store.jobs()}
        for record in self.store.jobs({"queued"}):
            if any(states.get(d) in FINAL - {"complete"} for d in record.get("after", [])):
                error = "a job it waited for did not complete"
                ended.append(
                    self.store.update_job(
                        record["id"], state="cancelled", finished=now(), error=error
                    )
                )
        return ended

    def _finish_studies(self) -> list[JobRecord]:
        ended = []
        states = {r["id"]: r["state"] for r in self.store.jobs()}
        for record in self.store.jobs({"queued", "running"}):
            # A study still being submitted may look finished after its first cells.
            if record["kind"] != "study" or not record.get("sealed"):
                continue
            children = [states[c] for c in record["children"] if c in states]
            if children and all(s in FINAL for s in children):
                final = (
                    "complete"
                    if all(s == "complete" for s in children)
                    else ("stopped" if "stopped" in children else "failed")
                )
                ended.append(self.store.update_job(record["id"], state=final, finished=now()))
            elif record["state"] == "queued" and any(s != "queued" for s in children):
                self.store.update_job(record["id"], state="running", started=now())
        return ended

    def _spawn(self, record: JobRecord) -> int:
        # A pinned job runs its frozen copy's interpreter, so the worker and everything it
        # starts import the copy's code, never the checkout's.
        job_id = record["id"]
        command, env = worker_spawn(record.get("code"), self.root, job_id)
        directory = self.jobs / job_id
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "worker.log").open("a") as log:
            process = subprocess.Popen(
                command,
                env=env,
                # A pinned worker starts in its job's folder: `python -m` puts the working
                # directory first on sys.path, and the checkout's must not be there.
                cwd=directory if record.get("code") else None,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # detached: it outlives the submitting process
            )
        return process.pid

    def notify(self, record: JobRecord) -> None:
        """Send the configured notification for an ended job, once."""
        if record.get("notified") or record["state"] not in FINAL:
            return
        self.store.update_job(record["id"], notified=True)
        # A job a chat launched: the chat hears of it, and so does the
        # notification.
        from apex.services.chat_followup import chat_followup

        extra = chat_followup(self.root, record) if not record.get("parent") else None
        notify(self.settings.notifications, record, extra)

    # ------------------------------------------------------------------ status and control

    def status(self, job_id: str) -> JobStatus:
        """Return a job's status (after bringing the queue up to date).

        Scheduling needs the store's write lock; reading the status does not (WAL). A pass
        that finds the lock held past its busy timeout (a worker's long transaction) is
        skipped, and the status read as it is: a poll (``apex study run --wait``, REST,
        the tools) failed with "database is locked" (the CI flake of 2026-10-10). The next
        poll, or the service, schedules.
        """
        try:
            self.schedule()
        except sqlite3.OperationalError as error:
            if "locked" not in str(error):
                raise
            logger.warning("job store busy: not scheduling on this poll: %s", error,
                           extra={"event": "store_busy", "job": job_id})  # fmt: skip
        record = self.record(job_id)
        children = {c: self.record(c)["state"] for c in record["children"]}
        reused = (record.get("spec") or {}).get("reused")  # an identical run, reused
        return JobStatus(
            id=record["id"],
            kind=record["kind"],
            state=record["state"],
            created=record["created"],
            started=record["started"],
            finished=record["finished"],
            error=record["error"],
            run_dirs=record["run_dirs"],
            run_ids=record["run_ids"],
            children=children,
            annotations=record["annotations"],
            reused=reused if isinstance(reused, str) else None,
            # Whoever may see the job may see its run, outside the web UI (the API narrows it).
            reused_visible=None if reused is None else True,
        )

    def events(self, job_id: str, since: int = 0) -> list[Event]:
        """Return a job's events from number ``since`` on."""
        standard = {"time", "level", "message", "event"}
        return [
            Event(
                index=seq,
                time=str(raw.get("time", "")),
                event=raw.get("event"),
                level=str(raw.get("level", "INFO")),
                message=str(raw.get("message", "")),
                fields={k: v for k, v in raw.items() if k not in standard},
            )
            for seq, raw in self.store.events(job_id, since)
        ]

    def stop(self, job_id: str) -> None:
        """Stop a job: a queued one is cancelled; a running run stops at its next checkpoint."""
        record = self.record(job_id)
        for child in record["children"]:
            self.stop(child)
        if record["state"] == "queued":
            self.notify(self.update(job_id, state="cancelled", finished=now()))
            self.log_event(job_id, "job_cancelled")
        elif record["state"] == "running":
            # The event is the job's own stop (its worker reads it while it waits for the
            # run's lock, and again once it holds it). The run's stop file is read by
            # whichever process runs the run: not written while another process holds it,
            # which a job waiting for that lock would otherwise stop.
            self.log_event(job_id, "job_stop_requested")
            for run_dir in record["run_dirs"]:
                if not held_by_another(Path(run_dir), record.get("pid")):
                    RunDirectory(Path(run_dir)).request_stop(by=f"job {job_id}")
        self.schedule()
