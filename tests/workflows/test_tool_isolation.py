"""Tools in worker processes, killed when abandoned."""

import multiprocessing
import operator
import os
import threading
import time

import pytest

from apex.agents import loop
from apex.agents.isolation import ToolCancelled, run_in_worker


def test_a_worker_returns_the_result_and_raises_the_error() -> None:
    assert run_in_worker(operator.add, (2, 3)) == 5
    with pytest.raises(ValueError, match="invalid literal"):
        run_in_worker(int, ("not a number",))


def test_an_abandoned_tool_is_killed_at_once() -> None:
    """A thread could not be stopped: Brugge's deck read ran on for minutes after its
    tool was abandoned (2026-10-06)."""
    cancel = threading.Event()
    failure: list[BaseException] = []

    def tool() -> None:
        loop._CANCEL.set(cancel)
        try:
            run_in_worker(time.sleep, (60,))
        except BaseException as error:  # noqa: BLE001
            failure.append(error)

    thread = threading.Thread(target=tool)
    started = time.monotonic()
    thread.start()
    time.sleep(1.0)
    cancel.set()
    thread.join(10)

    assert not thread.is_alive()
    assert time.monotonic() - started < 10
    assert [type(e) for e in failure] == [ToolCancelled]
    assert not [p for p in multiprocessing.active_children() if p.name == "apex-tool"]


def test_a_worker_that_dies_fails_the_tool_not_the_turn() -> None:
    """An ApexError: the loop gives the model {"ok": false} (review: a RuntimeError
    escaped and failed the whole turn)."""
    from apex.agents.isolation import WorkerFailed
    from apex.core.errors import ApexError

    with pytest.raises(WorkerFailed, match="without a result") as caught:
        run_in_worker(os._exit, (3,))
    assert isinstance(caught.value, ApexError)
