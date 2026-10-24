"""An exclusive lock on a run or prediction directory, while it is being written.

Two processes continuing the same run would both write checkpoints and interleave their
histories, and resume would no longer be bit-identical. The job service
resumes dead workers, and a run can also be resumed from the CLI, the REST API or the
tools, so this can happen by accident. :func:`exclusive` takes an ``flock`` on
``<directory>/run.lock`` for as long as the run executes. The operating system releases it
when the process ends, however it ends (``kill -9`` included), so a crashed run never
stays locked. The file names the holder (pid, host, time) for the error message.

``flock`` needs a local disk: on a network file system it may be emulated or not shared
across machines, as for the resource pools (``apex.execution.resources``).
"""

import datetime as dt
import fcntl
import json
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from apex.core.errors import WorkflowError

LOCK_FILE = "run.lock"


class DirectoryLocked(WorkflowError):
    """Another process is running in the directory."""


@contextmanager
def exclusive(directory: Path, what: str = "run") -> Iterator[None]:
    """Hold the directory's lock for the ``with`` block.

    Raises
    ------
    DirectoryLocked
        If another process holds it; the message names that process.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_FILE
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = _holder(handle.read())
            raise DirectoryLocked(
                f"the {what} in {directory} is in use by {holder}; wait for it to end, or "
                "stop it (apex stop) before resuming it"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "since": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }))  # fmt: skip
        handle.flush()
        yield
    finally:
        handle.close()  # closing releases the lock


def locked(directory: Path) -> bool:
    """Return whether a process holds the directory's lock now (e.g. a running run)."""
    path = Path(directory) / LOCK_FILE
    if not path.exists():
        return False
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False  # closing releases the probe's lock


def held_by_another(directory: Path, pid: int | None) -> bool:
    """Return whether a process other than ``pid`` (or one it started) holds the lock.

    For a job's stop: a job waiting for its run's lock must not write the stop
    file that the process holding it reads. When the holder cannot be told (no ``pid``, an
    unreadable lock file, a holder on another machine, as when the stop is asked for on
    another machine than the worker's over a shared ``APEX_HOME``), the answer is False, so
    a job's own run is always stopped.
    """
    if pid is None:
        return False
    holder = _live_holder(directory)
    if holder is None or holder < 0:
        return False
    return not _descends(holder, pid)


def in_use(directory: Path) -> bool:
    """Return whether a process holds the directory's lock now.

    Probed with ``flock`` only when the lock file names a live holder: a process that has
    released the lock may still be running (a job's worker after its run), and a probe with
    no holder named would be needless.
    """
    return _live_holder(directory) is not None and locked(directory)


def _live_holder(directory: Path) -> int | None:
    """Return the lock's live holder here (its pid), -1 for another machine's, or None.

    Read from the lock file; probed with ``flock`` only for another machine's holder (a
    probe holds the lock for an instant, in which a process starting the run would be
    refused). A holder recorded there that has ended holds it no longer: the lock goes with
    the process. A holder writes the file in place after taking the lock, so an empty or
    partial file is read again once.
    """
    info: object = None
    for attempt in range(2):
        try:
            info = json.loads((Path(directory) / LOCK_FILE).read_text())
            break
        except OSError:
            return None
        except ValueError:
            if attempt:
                return None
            time.sleep(0.05)
    if not isinstance(info, dict):
        return None
    try:
        holder, host = int(info["pid"]), str(info["host"])
    except (KeyError, TypeError, ValueError):
        return None
    if host != socket.gethostname():
        return -1 if locked(directory) else None  # only flock can tell for another machine
    return holder if Path(f"/proc/{holder}").exists() else None


def _descends(child: int, ancestor: int) -> bool:
    """Whether process ``child`` is ``ancestor`` or one of its descendants (Linux ``/proc``)."""
    seen: set[int] = set()
    while child > 1 and child not in seen:
        if child == ancestor:
            return True
        seen.add(child)
        try:
            # The fields after the command's closing parenthesis: state, then the parent.
            stat = Path(f"/proc/{child}/stat").read_text()
            child = int(stat[stat.rindex(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return child == ancestor


def _holder(text: str) -> str:
    try:
        info = json.loads(text)
        return f"process {info['pid']} on {info['host']} (since {info['since']})"
    except (ValueError, KeyError, TypeError):
        return "another process"
