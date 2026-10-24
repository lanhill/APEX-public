"""A job worker's events in its job's stream, logged inside the scheduler's transaction."""

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from apex.services.composition import _JobEvents
from apex.storage.sqlite_store import SQLiteStore


def _job(job_id: str) -> dict[str, object]:
    return {"id": job_id, "kind": "run", "state": "running", "created": "2026-01-01", "spec": {}}


@pytest.fixture
def logged(tmp_path: Path) -> Iterator[tuple[SQLiteStore, logging.Logger]]:
    """A store with a job, and a logger whose records go to its stream as a worker's do."""
    store = SQLiteStore(tmp_path)
    store.insert_job(_job("j1"))
    handler = _JobEvents(tmp_path, "j1")
    logger = logging.getLogger(f"apex.test.job_events.{tmp_path.name}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield store, logger
    logger.removeHandler(handler)
    handler.store.close()
    store.close()


def _messages(store: SQLiteStore) -> list[str]:
    return [str(data["message"]) for _, data in store.events("j1")]


def test_an_event_logged_inside_a_transaction_neither_waits_nor_is_lost(
    logged: tuple[SQLiteStore, logging.Logger],
) -> None:
    """The handler is a second store of the root in the same process. Its own connection
    waited for the transaction its own thread held, for the whole busy timeout (60 s),
    holding the write lock meanwhile, so another process waiting for the lock failed with
    "database is locked" (the CI flake of 2026-10-10). The event now follows the commit."""
    store, logger = logged
    began = time.monotonic()

    with store.locked():
        store.update_job("j1", state="queued")
        logger.warning("the scheduler logs while it holds the lock")
        assert _messages(store) == []  # written once the transaction ends

    assert time.monotonic() - began < 30  # the busy timeout is 60 s
    assert _messages(store) == ["the scheduler logs while it holds the lock"]
    assert store.job("j1")["state"] == "queued"


def test_an_event_survives_the_transaction_it_was_logged_in_rolling_back(
    logged: tuple[SQLiteStore, logging.Logger],
) -> None:
    store, logger = logged

    def fail() -> None:
        with store.locked():
            store.update_job("j1", state="queued")
            logger.error("about to fail")
            raise RuntimeError("the transaction fails")

    with pytest.raises(RuntimeError):
        fail()

    assert store.job("j1")["state"] == "running"  # rolled back
    assert _messages(store) == ["about to fail"]  # the log is kept


def test_another_threads_event_does_not_wait_for_the_transaction_either(
    logged: tuple[SQLiteStore, logging.Logger],
) -> None:
    store, logger = logged
    done = threading.Event()

    def log() -> None:
        logger.info("from a thread")
        done.set()

    with store.locked():
        thread = threading.Thread(target=log)
        thread.start()
        assert done.wait(10)  # it did not wait for the transaction
    thread.join()

    assert _messages(store) == ["from a thread"]


def test_events_that_cannot_be_stored_never_fail_the_transaction(
    logged: tuple[SQLiteStore, logging.Logger],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, logger = logged

    def broken(self: SQLiteStore, job_id: str, event: dict[str, object]) -> int:
        raise OSError("disk full")

    with store.locked():
        store.update_job("j1", state="queued")
        logger.warning("one")
        monkeypatch.setattr(SQLiteStore, "add_event", broken)

    assert store.job("j1")["state"] == "queued"  # committed, and nothing raised
    assert "1 job events not stored: disk full" in capsys.readouterr().err


def test_events_wait_for_the_outermost_writer_and_keep_their_order(
    logged: tuple[SQLiteStore, logging.Logger], tmp_path: Path
) -> None:
    store, logger = logged
    other = SQLiteStore(tmp_path)  # another store of the root in this process

    with store.locked():
        logger.info("first")
        with store.locked():  # nested: still one writer
            logger.info("second")
        logger.info("third")
    with other.locked():  # another store's transaction: the events wait for it too
        logger.info("fourth")
    other.close()

    assert _messages(store) == ["first", "second", "third", "fourth"]


def test_a_forked_child_starts_with_no_writers() -> None:
    from apex.storage import sqlite_store

    sqlite_store._WRITING[(0, Path("/x"))] = sqlite_store._Writing()
    before = sqlite_store._WRITING_LOCK

    sqlite_store._after_fork()

    assert sqlite_store._WRITING == {}
    assert sqlite_store._WRITING_LOCK is not before


def test_a_writer_whose_begin_fails_leaves_no_writer_behind(
    logged: tuple[SQLiteStore, logging.Logger], tmp_path: Path
) -> None:
    import sqlite3

    from apex.storage import sqlite_store

    store, logger = logged
    other = sqlite3.connect(tmp_path / "apex.db", isolation_level=None)
    other.execute("BEGIN IMMEDIATE")  # another process's lock
    store._execute("PRAGMA busy_timeout=100")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.update_job("j1", state="queued")
    finally:
        other.execute("ROLLBACK")
        other.close()
    store._execute("PRAGMA busy_timeout=60000")

    assert sqlite_store._WRITING[store._key].active == 0
    logger.info("after")  # written at once: no writer is left to wait for
    assert _messages(store) == ["after"]


def test_a_status_poll_reads_the_job_when_scheduling_finds_the_store_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduling needs the write lock, reading a status does not: a poll locked out past
    the busy timeout skips scheduling (`apex study run --wait` failed with "database is
    locked": the CI flake of 2026-10-10). Other errors still raise."""
    import sqlite3

    from apex.services.jobs import LocalJobRunner

    runner = LocalJobRunner(tmp_path / "root")
    job = runner.submit("run", {}, start=False)

    def busy() -> list[str]:
        raise sqlite3.OperationalError("database is locked: BEGIN IMMEDIATE on apex.db")

    monkeypatch.setattr(runner, "schedule", busy)
    assert runner.status(job).state == "queued"

    def broken() -> list[str]:
        raise sqlite3.OperationalError("no such table: jobs")

    monkeypatch.setattr(runner, "schedule", broken)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        runner.status(job)
