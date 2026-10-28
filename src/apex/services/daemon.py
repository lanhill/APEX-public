"""The job service's loop, shared by the plain service and the REST server.

:func:`serve_jobs` writes ``<root>/service.pid``, then schedules the root's queue every
``[service] poll_s`` seconds (recovering dead jobs, starting what fits, finishing studies,
notifying) and reaps its finished workers, until ``stop`` is set. At start and daily it
applies the chat history's limits (:func:`prune_chats`), and at start and
hourly the guests' retention and storage limits (:func:`prune_guest_work`), and
daily, counted across restarts, it removes the frozen copies of APEX nothing uses
(:func:`prune_code`, in a thread). It declares its job protocol in the store, and
schedules nothing while newer code uses the store.
Running jobs go on after it stops; their workers record their end.
"""

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from apex.core.maintenance import MaintenanceContext
from apex.core.plugins import maintenance_tasks
from apex.services.jobs import LocalJobRunner
from apex.services.settings import ServiceSettings
from apex.storage.chat_store import ChatStore
from apex.storage.sqlite_store import JOB_PROTOCOL

logger = logging.getLogger("apex.service")


def serve_jobs(
    root: Path, stop: threading.Event, on_tick: Callable[[LocalJobRunner], None] | None = None
) -> None:
    """Run the scheduling loop of ``root`` until ``stop`` is set.

    ``on_tick`` is called after each scheduling pass, e.g. the run monitor
    (the service entry points pass it: services do not import agents). It must return
    quickly; its failures are logged and never stop the service. Its ``close()``, if it
    has one, is called when the service stops.
    """
    runner = LocalJobRunner(root, role="service")
    pid_file = root / "service.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    runner.store.declare_service(os.getpid())
    pruning = CodePruning(runner)
    newer_store = False
    logger.info("service started", extra={"event": "service_started", "root": str(root)})
    for missing in stale_entry_points():
        logger.warning(
            "the environment lacks the entry point %s that the checkout declares: run "
            "uv sync --inexact --all-packages, then restart the service", missing,
            extra={"event": "stale_environment", "entry_point": missing},
        )  # fmt: skip
    last_prune = last_guests = 0.0
    try:
        while not stop.is_set():
            if time.monotonic() - last_prune >= CHAT_PRUNE_INTERVAL_S or last_prune == 0.0:
                prune_chats(root, runner.settings, first=last_prune == 0.0)
                last_prune = time.monotonic()
            if time.monotonic() - last_guests >= GUEST_PRUNE_INTERVAL_S or last_guests == 0.0:
                prune_guest_work(root)
                last_guests = time.monotonic()
            pruning.maybe_start()
            protocol = runner.store.job_protocol()
            if protocol > JOB_PROTOCOL:
                # Newer code uses this store: its records may mean what this
                # service would misread. Schedule nothing until it is restarted on that code.
                if not newer_store:
                    logger.error(
                        "the job store speaks job protocol %d, this service %d: not scheduling; "
                        "restart the service on current code", protocol, JOB_PROTOCOL,
                        extra={"event": "store_newer"},
                    )  # fmt: skip
                    newer_store = True
                stop.wait(runner.settings.service.poll_s)
                continue
            try:
                for job in runner.schedule():
                    logger.info("started %s", job, extra={"event": "job_started", "job": job})
            except Exception:  # (the service must outlive one bad pass)
                logger.exception("scheduling pass failed", extra={"event": "schedule_failed"})
            if on_tick is not None:
                try:
                    on_tick(runner)
                except Exception:  # (the service must outlive a failing hook)
                    logger.exception("service hook failed", extra={"event": "hook_failed"})
            reap()
            stop.wait(runner.settings.service.poll_s)
    except BaseException:
        # The log is where the owner looks: a thread's traceback goes only to stderr, and a
        # SystemExit (a library's sys.exit) not even there.
        logger.exception("the service loop failed", extra={"event": "service_failed"})
        raise
    finally:
        close = getattr(on_tick, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # (the service stops whatever its hooks do)
                logger.exception("closing the service hooks failed",
                                 extra={"event": "hook_close_failed"})  # fmt: skip
        pid_file.unlink(missing_ok=True)
        logger.info("service stopped", extra={"event": "service_stopped"})


def reap() -> None:
    """Collect this process's finished child processes (its job workers: no zombies)."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


#: How often the service applies the chat history's limits: daily, and at start.
CHAT_PRUNE_INTERVAL_S = 24 * 3600.0


def prune_chats(root: Path, settings: ServiceSettings, *, first: bool = False) -> list[str]:
    """Apply ``[chat]`` retention and size limits to ``<root>/chats.db`` (if it exists).

    At service start (``first``), chats a stopped service left mid-turn are freed too.
    Failures are logged: the history's limits must never stop the service.
    """
    if not (Path(root) / "chats.db").exists():
        return []
    try:
        store = ChatStore(root)
        if first:
            store.release_all()
        chat = settings.chat
        deleted = store.prune(
            retention_days=chat.retention_days, max_bytes=int(chat.max_history_mb * 1e6)
        )
    except (OSError, sqlite3.Error):
        logger.exception("chat history pruning failed", extra={"event": "chat_prune_failed"})
        return []
    if deleted:
        logger.info(
            "chat history pruned: %d chats deleted", len(deleted),
            extra={"event": "chat_pruned", "chats": deleted},
        )  # fmt: skip
    return deleted


#: How often the service applies ``[guests]``: hourly, and at start, so that a
#: guest's storage never stays over its limit for long.
GUEST_PRUNE_INTERVAL_S = 3600.0


def prune_guest_work(root: Path) -> int:
    """Apply ``[guests]`` retention and storage limits; return the submissions deleted.

    Failures are logged: the limits must never stop the service.
    """
    from apex.services.api import ApexService
    from apex.services.guest_retention import prune_guests

    try:
        return len(prune_guests(ApexService(root)))
    except (OSError, sqlite3.Error, ValueError):
        logger.exception("guest retention failed", extra={"event": "guest_prune_failed"})
        return 0


#: How often the service removes unused frozen copies of APEX: daily, counted
#: across restarts (``<code_dir>/.pruned``), in a thread of its own.
CODE_PRUNE_INTERVAL_S = 24 * 3600.0


class CodePruning:
    """Runs :func:`prune_code` daily, in the background: removals must not hold up the queue."""

    def __init__(self, runner: LocalJobRunner) -> None:
        self.runner = runner
        self.thread: threading.Thread | None = None

    def maybe_start(self) -> None:
        """Start a clean-up if one is due and none is running."""
        if self.thread is not None and self.thread.is_alive():
            return
        from apex.services.code import CodeCopies

        copies = CodeCopies.from_settings(self.runner.settings.jobs, self.runner.root)
        stamp = copies.directory / ".pruned"
        try:
            due = time.time() - stamp.stat().st_mtime >= CODE_PRUNE_INTERVAL_S
        except FileNotFoundError:
            due = copies.directory.is_dir()
        except OSError:
            return
        if not due:
            return
        try:
            stamp.touch()
        except OSError:
            return
        self.thread = threading.Thread(
            target=prune_code, args=(self.runner.root,), name="code-prune", daemon=True
        )
        self.thread.start()


def stale_entry_points(repository: Path | None = None) -> list[str]:
    """Return the entry points the checkout declares that are not installed.

    Plugins and commands register by entry point, which only an install writes: an
    environment made before a checkout gained one (``uv sync`` not run since) silently lacks
    the plugin. Empty for an installed APEX (no checkout) or an unreadable ``pyproject.toml``.
    """
    import tomllib
    from importlib.metadata import entry_points

    from apex.services.code import default_repository, workspace_member

    repository = repository or default_repository()
    if repository is None:
        return []
    try:
        root = tomllib.loads((repository / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    # The workspace's members too: a plain `uv sync` leaves them out.
    workspace = root.get("tool", {}).get("uv", {}).get("workspace", {})
    members, excluded = workspace.get("members", []), workspace.get("exclude", [])
    projects = [root]
    # A set: overlapping patterns, and `**` (which also yields the folders above), must not
    # report an entry point twice.
    folders: set[Path] = set()
    for pattern in members:
        try:
            folders.update(repository.glob(pattern))
        except (ValueError, NotImplementedError):  # an empty or absolute glob: uv refuses it
            continue
    for folder in sorted(folders):
        relative = folder.relative_to(repository).as_posix()
        if folder == repository or not workspace_member(relative, members, excluded):
            continue
        try:
            projects.append(tomllib.loads((folder / "pyproject.toml").read_text()))
        except (OSError, tomllib.TOMLDecodeError):
            continue  # a folder above a member (`**`), or unreadable
    missing = []
    for project in projects:
        groups = project.get("project", {}).get("entry-points", {})
        for group, names in groups.items():
            installed: dict[str, set[str]] = {}
            for e in entry_points(group=group):
                installed.setdefault(e.name, set()).add(e.value.strip())
            # Missing, or installed with another target, by any distribution (an old one's
            # metadata may sit next to the new one's): a plugin moved since the install,
            # whose old module is gone.
            missing += [f"{group}:{name}" for name, value in names.items()
                        if installed.get(name, set()) != {str(value).strip()}]  # fmt: skip
    return missing


def maintenance_context(
    root: Path, settings: ServiceSettings, runner: LocalJobRunner | None = None
) -> MaintenanceContext:
    """Return the context the maintenance tasks run."""
    from apex.services.code import CodeCopies

    runner = runner or LocalJobRunner(root, role="client")
    return MaintenanceContext(
        root=root,
        code=CodeCopies.from_settings(settings.jobs, root),
        jobs=runner.all_records,
        keep_days=float(settings.jobs.keep_code_days),
    )


def run_maintenance(context: MaintenanceContext, dry_run: bool = False) -> dict[str, list[str]]:
    """Run every installed maintenance task; return each one's report.

    A task that fails is logged and reported; the others still run.
    """
    reports: dict[str, list[str]] = {}
    try:
        names = maintenance_tasks.names()
    except Exception as error:
        logger.exception("maintenance tasks cannot be listed",
                         extra={"event": "maintenance_failed"})  # fmt: skip
        return {"(tasks)": [f"failed: {error}"]}
    for name in names:
        try:
            reports[name] = maintenance_tasks.create(name).run(context, dry_run)
        except Exception as error:
            logger.exception("maintenance task %s failed", name,
                             extra={"event": "maintenance_failed", "task": name})  # fmt: skip
            reports[name] = [f"failed: {error}"]
    return reports


def prune_code(root: Path) -> int:
    """Remove the frozen copies no job or process uses, unused for ``[jobs] keep_code_days``.

    Failures are logged: the clean-up must never stop the service.
    """
    from apex.services.code import CodeCopies

    try:
        runner = LocalJobRunner(root, role="client")
        settings = runner.settings.jobs
        copies = CodeCopies.from_settings(settings, root)
        removed = copies.collect(runner.all_records, settings.keep_code_days)
        # Installed packages' tasks, e.g. the hosts' copies.
        run_maintenance(maintenance_context(root, runner.settings, runner))
    except Exception:  # (the service must outlive its clean-ups)
        logger.exception("frozen code clean-up failed", extra={"event": "code_prune_failed"})
        return 0
    if removed:
        logger.info(
            "removed %d unused frozen copies of APEX", len(removed),
            extra={"event": "code_pruned", "commits": [c.commit for c in removed]},
        )  # fmt: skip
    return len(removed)
