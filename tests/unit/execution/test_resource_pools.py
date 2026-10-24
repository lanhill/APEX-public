"""Machine-wide resource pools: file locks shared by every process."""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.simulation import MemberInput
from apex.domain import DataKey
from apex.execution.local import SerialExecutor
from apex.execution.resources import FilePools, default_sizes
from apex.simulation.eclipse import EclipseSimulator


def _pools(root: Path, **sizes: int) -> FilePools:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pools.toml").write_text(
        "[pools]\n" + "".join(f"{k} = {v}\n" for k, v in sizes.items())
    )
    return FilePools(root, poll_s=0.02)


class _Licensed:
    """Needs one ``lic`` token per run; records when each run starts and ends."""

    needs_work_dir = True
    resources = {"lic": 1}  # noqa: RUF012 (a capability, read with getattr)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        assert member.work_dir is not None
        start = time.monotonic()
        time.sleep(0.25)
        (member.work_dir / "span").write_text(f"{start} {time.monotonic()}")
        return np.zeros(len(keys))


#: A job in its own interpreter: 4 workers, each run holding one "lic" token for 0.25 s.
#: Run with ``python -c`` (not multiprocessing), so nothing depends on how pytest imports
#: this module.
JOB = """
import sys, time
from pathlib import Path
import numpy as np
from apex.core.simulation import MemberFailure, MemberInput
from apex.execution.local import LocalProcessExecutor
from apex.execution.resources import FilePools

class Licensed:
    needs_work_dir = True
    resources = {"lic": 1}
    def run(self, member, keys):
        start = time.monotonic()
        time.sleep(0.25)
        (member.work_dir / "span").write_text(f"{start} {time.monotonic()}")
        return np.zeros(len(keys))

if __name__ == "__main__":
    root, work, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    members = [MemberInput(i, {}, work / f"member_{i}") for i in range(n)]
    results = LocalProcessExecutor(workers=4, pools=FilePools(root, poll_s=0.02)).run_all(
        Licensed(), members, []
    )
    assert not any(isinstance(r, MemberFailure) for r in results)
"""

HOLD = """
import sys, time
from pathlib import Path
from apex.execution.resources import FilePools
with FilePools(Path(sys.argv[1])).hold({"lic": 1}):
    time.sleep(60)
"""


def _python(tmp_path: Path, code: str, *args: str) -> "subprocess.Popen[bytes]":
    script = tmp_path / f"script_{abs(hash(code))}.py"
    script.write_text(code)
    return subprocess.Popen([sys.executable, str(script), *args])


def _most_at_once(work_dirs: Sequence[Path]) -> int:
    spans = [tuple(map(float, (d / "span").read_text().split())) for d in work_dirs]
    return max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)


def test_concurrent_jobs_never_hold_more_tokens_than_the_pool(tmp_path: Path) -> None:
    # Two jobs in separate processes, 4 workers each, share a pool of 2 "licences".
    _pools(tmp_path / "pools", lic=2)
    jobs = [
        _python(tmp_path, JOB, str(tmp_path / "pools"), str(tmp_path / f"job_{j}"), "4")
        for j in range(2)
    ]
    for job in jobs:
        assert job.wait(timeout=120) == 0

    work_dirs = [tmp_path / f"job_{j}" / f"member_{i}" for j in range(2) for i in range(4)]
    assert _most_at_once(work_dirs) == 2  # the pool's size: shared, and used in full


def test_a_killed_process_releases_its_tokens(tmp_path: Path) -> None:
    pools = _pools(tmp_path / "pools", lic=1)
    holder = _python(tmp_path, HOLD, str(pools.root))
    deadline = time.monotonic() + 60
    while not any((pools.root / "lic").glob("*.lock")) and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)  # let it take the lock
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait()

    with pools.hold({"lic": 1}):  # would wait forever if the lock were kept
        pass


def test_all_tokens_are_taken_at_once_or_none(tmp_path: Path) -> None:
    import threading

    pools = _pools(tmp_path / "pools", a=1, b=1)
    done = threading.Event()

    def wants_both() -> None:
        with FilePools(pools.root, poll_s=0.02).hold({"a": 1, "b": 1}):
            done.set()

    with pools.hold({"b": 1}):
        waiting = threading.Thread(target=wants_both)
        waiting.start()
        time.sleep(0.2)
        # The waiting run does not keep "a" while it waits for "b": "a" is free.
        began = time.monotonic()
        with pools.hold({"a": 1}):
            # Held, "a" would block until the waiting run gets "b" (never): any bound tells,
            # and a loose one does not fail on a loaded machine (parallel tests).
            assert time.monotonic() - began < 5.0
        assert not done.is_set()
    waiting.join(timeout=10)
    assert done.is_set()  # it got both once "b" was released


def test_impossible_needs_fail_loudly(tmp_path: Path) -> None:
    pools = _pools(tmp_path / "pools", lic=2)
    with pytest.raises(ValueError, match="unknown resource pool 'gpu'"), pools.hold({"gpu": 1}):
        pass
    with (
        pytest.raises(ValueError, match="needs 3 lic tokens, but the pool holds 2"),
        pools.hold({"lic": 3}),
    ):
        pass


def test_sizes_default_and_then_stay_fixed(tmp_path: Path) -> None:
    pools = FilePools(tmp_path / "fresh")

    assert pools.sizes() == default_sizes()
    assert default_sizes()["eclipse"] == 3  # the site's licences
    (pools.root / "pools.toml").write_text("[pools]\neclipse = 2\ncpu = 4\n")
    assert pools.sizes() == {"eclipse": 2, "cpu": 4}  # the file is authoritative


def test_the_serial_executor_holds_tokens_and_simulators_declare_them(tmp_path: Path) -> None:
    pools = _pools(tmp_path / "pools", lic=1)
    members = [MemberInput(i, {}, tmp_path / f"m{i}") for i in range(2)]

    SerialExecutor(pools=pools).run_all(_Licensed(), members, [])

    template = tmp_path / "deck.mako"
    template.write_text("")
    assert EclipseSimulator(template).resources == {"eclipse": 1, "cpu": 1}
