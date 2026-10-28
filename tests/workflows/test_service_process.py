"""The service process with the REST API ends when its scheduling loop fails.

On 2026-10-25 the scheduling loop ended while uvicorn served on in the same process, so
systemd's ``Restart=on-failure`` saw nothing wrong and nothing was scheduled for two days.
"""

import importlib
import logging
import socket
import threading
from pathlib import Path

import pytest

from apex.services import daemon

pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_loop_ending_unasked_is_reported_and_one_asked_to_stop_is_not() -> None:
    from apex.api.server import watched

    stop, ended = threading.Event(), []

    def failing() -> None:
        raise RuntimeError("a broken pass")

    with pytest.raises(RuntimeError, match="a broken pass"):
        watched(failing, stop, lambda: ended.append("failed"))()
    watched(lambda: None, stop, lambda: ended.append("returned"))()
    assert ended == ["failed", "returned"]

    stop.set()
    watched(lambda: None, stop, lambda: ended.append("asked"))()
    assert ended == ["failed", "returned"]


def test_the_loop_logs_its_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The traceback goes to the service log, not only to the thread's stderr."""
    root = tmp_path / "service"
    root.mkdir()

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(daemon, "prune_chats", broken)
    with (
        caplog.at_level(logging.ERROR, logger="apex.service"),
        pytest.raises(OSError, match="disk full"),
    ):
        daemon.serve_jobs(root, threading.Event())
    assert "the service loop failed" in caplog.text
    assert "disk full" in caplog.text
    assert not (root / "service.pid").exists()


def test_the_process_exits_with_a_failure_when_its_loop_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` stops uvicorn and exits with status 1, for its process manager to restart it."""
    import signal

    from apex.agents import watch
    from apex.api import server

    root = tmp_path / "service"

    def failing(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a broken pass")

    import uvicorn

    servers: list[uvicorn.Server] = []

    class Recorded(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            servers.append(self)

    # The bug kept uvicorn serving: a watchdog stops it, so the test fails instead of hanging.
    watchdog = threading.Timer(30.0, lambda: setattr(servers[0], "should_exit", True))
    monkeypatch.setattr(uvicorn, "Server", Recorded)
    # uvicorn.run's own (``uvicorn.main`` the attribute is its command, not its module)
    monkeypatch.setattr(importlib.import_module("uvicorn.main"), "Server", Recorded)
    monkeypatch.setattr(server, "serve_jobs", failing)
    monkeypatch.setattr(watch, "service_hook", lambda root: None)
    monkeypatch.setattr(signal, "signal", lambda *args: None)  # leave pytest's handlers
    monkeypatch.setattr(threading, "excepthook", lambda args: None)  # the expected traceback
    watchdog.start()
    try:
        with pytest.raises(SystemExit) as exit_:
            server.main([str(root), "--http", f"127.0.0.1:{_free_port()}"])
    finally:
        watchdog.cancel()
    assert exit_.value.code == 1
    assert not (root / "service.url").exists()
