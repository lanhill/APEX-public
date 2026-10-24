"""The run monitor inside the job service: ``[agents.monitor] auto = true``.

:func:`service_hook` also adds the routines installed packages give the service
(:mod:`apex.services.routines`).

:class:`ServiceMonitor` is the hook the service calls after each scheduling pass
(:func:`apex.services.daemon.serve_jobs`). For every running history-matching job whose run
has scored a new forecast since the last look, it runs the run monitor
(:class:`~apex.agents.monitor.RunMonitor`) in a background thread, so model calls never
hold up scheduling. Each report goes into the job's event stream (``monitor_report``, seen
by ``apex``'s CLI, the REST API and the tools) and the run's ``agents/`` traces. A report
whose status is in ``notify_on`` calls the notification command or webhook
(``[notifications]``), once per job and change of status and findings.

It is advisory: it never stops a job. Stopping stays with a person (``apex agent monitor
--act``, which asks first). Without a usable model (``[agents.llm]``), the rules decide.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from apex.agents.monitor import RunMonitor
from apex.core.errors import ApexError
from apex.core.llm import LLMProvider
from apex.services.api import ApexService
from apex.services.jobs import LocalJobRunner
from apex.services.notify import notify
from apex.services.routines import ServiceHooks, routines
from apex.storage.run_directory import RunDirectory

logger = logging.getLogger(__name__)


class ServiceMonitor:
    """The service's run monitor (see the module docstring).

    Parameters
    ----------
    root
        The services root.
    monitor
        The monitor to use (default: one on the root's ``[agents.llm]`` model, or rules
        only when that model cannot be set up).
    """

    def __init__(self, root: Path, monitor: RunMonitor | None = None) -> None:
        self.service = ApexService(root)
        self.settings = self.service.jobs.settings
        self.monitor = monitor or RunMonitor(self.service, self._provider())
        self._seen: dict[str, int] = {}
        self._notified: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._busy: set[str] = set()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apex-monitor")
        self._futures: list[Future[None]] = []

    def _provider(self) -> LLMProvider | None:
        from apex.agents.providers import create_provider

        try:
            return create_provider(self.settings.agents.llm)
        except ApexError as error:
            logger.warning(
                "the service monitor uses rules only: %s", error,
                extra={"event": "monitor_rules_only", "reason": str(error)},
            )  # fmt: skip
            return None

    def __call__(self, runner: LocalJobRunner) -> None:
        """Queue a check of each running job with a newly scored forecast."""
        for record in runner.all_records():
            if record["state"] != "running" or record["kind"] not in ("run", "resume"):
                continue
            job, run_dirs = str(record["id"]), record.get("run_dirs") or []
            if not run_dirs:
                continue
            scored = len(RunDirectory(Path(run_dirs[0])).history_dicts())
            with self._lock:
                if scored == 0 or scored == self._seen.get(job) or job in self._busy:
                    continue
                self._seen[job] = scored
                self._busy.add(job)
            self._futures.append(self._pool.submit(self._check, runner, job, record))

    def _check(self, runner: LocalJobRunner, job: str, record: dict[str, Any]) -> None:
        try:
            result = self.monitor.check(job)  # advisory: never act=True here
        except Exception:  # (one failed check must not stop the monitor)
            logger.exception("monitor check of %s failed", job, extra={"event": "monitor_failed"})
            with self._lock:
                self._busy.discard(job)
            return
        report = result.report
        codes = sorted(f.code for f in report.findings)
        runner.log_event(
            job,
            "monitor_report",
            status=report.status,
            recommendation=report.recommendation,
            findings=codes,
            summary=report.summary,
            decided_by=result.decided_by,
            trace=str(result.trace),
        )
        key = (report.status, tuple(codes))
        with self._lock:
            self._busy.discard(job)
            send = report.status in self.settings.agents.monitor.notify_on
            send = send and self._notified.get(job) != key
            if send:
                self._notified[job] = key
        if send:
            notify(
                self.settings.notifications,
                record,
                {
                    "event": "monitor_report",
                    "status": report.status,
                    "recommendation": report.recommendation,
                    "findings": ",".join(codes) or "none",
                    "summary": report.summary,
                },
            )

    def drain(self) -> None:
        """Wait for the checks queued so far (for tests and a clean shutdown)."""
        for future in list(self._futures):
            future.result()
        self._futures = [f for f in self._futures if not f.done()]

    def close(self) -> None:
        """Stop taking checks; a running one finishes in the background."""
        self._pool.shutdown(wait=False, cancel_futures=True)


def service_hook(root: Path) -> ServiceHooks | None:
    """Return the service's hooks, or ``None`` when there are none.

    The run monitor (``[agents.monitor] auto``), and the routines installed packages add
    (:mod:`apex.services.routines`), each switched on by its own settings.
    """
    service = ApexService(root)
    hooks: list[Callable[[LocalJobRunner], None]] = []
    if service.jobs.settings.agents.monitor.auto:
        logger.info("the service monitors running jobs", extra={"event": "monitor_enabled"})
        hooks.append(ServiceMonitor(root))
    hooks += routines(root)
    return ServiceHooks(hooks) if hooks else None
