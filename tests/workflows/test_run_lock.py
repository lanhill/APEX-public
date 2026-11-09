"""The run lock: one process at a time writes a run or a prediction.

Regression for the architecture review's item E4 (2026-09-28). Nothing stopped two
processes from resuming the same run directory, and both would have written checkpoints.
"""

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import WorkflowError
from apex.services import composition
from apex.storage.locks import LOCK_FILE, DirectoryLocked, exclusive

#: Holds a directory's lock in another process until it is killed.
HOLDER = """
import sys, time
from pathlib import Path
from apex.storage.locks import exclusive
with exclusive(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(600)
"""


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _hold(directory: Path) -> subprocess.Popen[str]:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(directory)], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "locked"
    return holder


def test_a_run_held_by_another_process_is_not_resumed(linear_case: CaseFactory) -> None:
    whole = linear_case(name="whole", ensemble_size=20)
    uninterrupted = composition.start(whole.config)
    case = linear_case(ensemble_size=20)
    run_dir = case.directory / "runs" / "case"
    composition.start(case.config, max_steps=2)
    history = (run_dir / "history.json").read_text()
    holder = _hold(run_dir)
    try:
        with pytest.raises(DirectoryLocked, match=f"in use by process {holder.pid} on "):
            composition.resume(run_dir)
        # Refused before anything was written.
        assert (run_dir / "history.json").read_text() == history
    finally:
        holder.send_signal(signal.SIGKILL)  # the way a crashed worker ends
        holder.wait()

    resumed = composition.resume(run_dir)  # the lock went with the process

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    holder_info = json.loads((run_dir / LOCK_FILE).read_text())
    assert holder_info["pid"] == os.getpid()  # the last holder: this process


def test_a_new_run_in_a_directory_in_use_is_refused(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=10)
    run_dir = case.directory / "runs" / "case"
    holder = _hold(run_dir)
    try:
        with pytest.raises(DirectoryLocked):
            composition.start(case.config)
        assert not (run_dir / "config.json").exists()
    finally:
        holder.kill()
        holder.wait()


def test_the_lock_is_exclusive_within_one_process_too(tmp_path: Path) -> None:
    in_use = pytest.raises(DirectoryLocked, match=rf"the prediction in {tmp_path} is in use")
    with exclusive(tmp_path, "prediction"), in_use, exclusive(tmp_path, "prediction"):
        pass
    with exclusive(tmp_path):  # released on leaving
        pass


def test_the_lock_says_whether_it_made_the_directory(tmp_path: Path) -> None:
    """Only the call that made the directory is told so: a prediction removes a refused
    folder only if it made it, so a process that lost a race to start in the same folder
    never removes the winner's files."""
    folder = tmp_path / "a" / "b"
    with exclusive(folder) as made:
        assert made is True
    with exclusive(folder) as made:
        assert made is False
    with exclusive(tmp_path) as made:  # an existing directory
        assert made is False
    (tmp_path / "file").write_text("")
    with (
        pytest.raises(WorkflowError, match="exists and is not a directory"),
        exclusive(tmp_path / "file"),
    ):
        pass


def _flock_after(
    monkeypatch: pytest.MonkeyPatch, removals: int, remove: Callable[[], None]
) -> list[int]:
    """Make the lock's ``flock`` run ``remove`` first, the first ``removals`` times."""
    import fcntl

    from apex.storage import locks

    calls: list[int] = []

    def flock(descriptor: int, operation: int) -> None:
        calls.append(descriptor)
        if len(calls) <= removals:
            remove()
        fcntl.flock(descriptor, operation)

    monkeypatch.setattr(locks, "fcntl", SimpleNamespace(
        flock=flock, LOCK_EX=fcntl.LOCK_EX, LOCK_NB=fcntl.LOCK_NB))  # fmt: skip
    return calls


def test_a_lock_file_removed_while_being_taken_is_taken_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder removing its folder (a refused prediction's own) unlinks the lock file while
    another process has it open: that process must not hold the unlinked file, which a
    third would not see, but take the folder's lock afresh, making the folder again."""
    folder = tmp_path / "p"
    folder.mkdir()

    def holder_cleans_up() -> None:
        (folder / LOCK_FILE).unlink()
        folder.rmdir()

    calls = _flock_after(monkeypatch, 1, holder_cleans_up)
    with exclusive(folder, "prediction") as made:
        assert len(calls) == 2  # the first lock was on the removed file
        assert made is True  # the folder had gone: this call made it again
        assert json.loads((folder / LOCK_FILE).read_text())["pid"] == os.getpid()
        monkeypatch.undo()
        with pytest.raises(DirectoryLocked), exclusive(folder, "prediction"):
            pass  # the file the folder names is the one held


def test_a_lock_file_removed_every_time_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "p"
    _flock_after(monkeypatch, 99, lambda: (folder / LOCK_FILE).unlink())
    with pytest.raises(WorkflowError, match="removed 5 times"), exclusive(folder):
        pass
