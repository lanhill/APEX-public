"""The SQLite store of a services root."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from apex.core.storage import CatalogueEntry
from apex.storage.sqlite_store import SQLiteStore


def _job(
    job_id: str, created: str = "2026-01-01T00:00:00+00:00", state: str = "queued"
) -> dict[str, object]:
    return {"id": job_id, "kind": "run", "state": state, "created": created, "spec": {}}


def test_jobs_events_approvals_and_quotas_round_trip(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path)
    store.insert_job(_job("b", "2026-01-02T00:00:00+00:00"))
    store.insert_job(_job("a"))

    store.update_job("a", state="running", pid=12)

    assert [j["id"] for j in store.jobs()] == ["a", "b"]  # oldest first
    assert [j["id"] for j in store.jobs({"queued"})] == ["b"]
    assert store.job("a")["pid"] == 12
    with pytest.raises(KeyError, match="no job 'x'"):
        store.job("x")
    assert [store.add_event("a", {"event": e}) for e in ("x", "y", "z")] == [0, 1, 2]
    assert [n for n, _ in store.events("a", since=1)] == [1, 2]
    store.add_event("b", {"event": "w"})
    assert [(job, e["event"]) for _, job, e in store.feed(since=2)] == [("a", "z"), ("b", "w")]
    store.put_approval("t", {"approved": False})
    assert store.approval("t") == {"approved": False}
    assert store.approval("nope") is None
    store.add_usage("planner", "2026-09-28", 100)
    store.add_usage("planner", "2026-09-28", 50)
    assert store.usage("planner", "2026-09-28") == 150
    assert store.usage("planner", "2026-09-29") == 0


def test_locked_is_all_or_nothing_and_re_entrant(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path)
    store.insert_job(_job("a"))

    def abort_after_an_update() -> None:
        with store.locked():
            store.update_job("a", state="running")  # re-entrant: inside the outer transaction
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError):
        abort_after_an_update()

    assert store.job("a")["state"] == "queued"  # rolled back


WRITER = """
import sys
from pathlib import Path
from apex.storage.sqlite_store import SQLiteStore
store = SQLiteStore(Path(sys.argv[1]))
for i in range(50):
    store.add_event("shared", {"writer": sys.argv[2], "i": i})
"""


def test_concurrent_writers_never_lose_or_repeat_events(tmp_path: Path) -> None:
    SQLiteStore(tmp_path)  # create the database first
    script = tmp_path / "writer.py"
    script.write_text(WRITER)
    writers = [
        subprocess.Popen([sys.executable, str(script), str(tmp_path), str(w)]) for w in range(4)
    ]
    for writer in writers:
        assert writer.wait(timeout=120) == 0

    events = SQLiteStore(tmp_path).events("shared")
    assert [n for n, _ in events] == list(range(200))  # numbered without gaps or repeats
    assert sorted((e["writer"], e["i"]) for _, e in events) == [
        (str(w), i) for w in range(4) for i in range(50)
    ]


def test_the_run_catalogue_and_its_clean_up(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path)
    kept, gone = tmp_path / "kept", tmp_path / "gone"
    kept.mkdir()
    store.register(CatalogueEntry("r1", kept, "a", "2026-01-01", {"case": "x"}))
    store.register(CatalogueEntry("r2", gone, "b", "2026-01-02", {}))

    assert store.find("r1") is not None
    assert store.find("r2") is None  # its directory is gone
    assert [e.run_id for e in store.entries()] == ["r1"]
    assert [e.run_id for e in store.forget_missing()] == ["r2"]


def test_phase_one_files_are_imported_once(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps(_job("j1", state="complete")))
    (job_dir / "events.jsonl").write_text('{"event": "job_queued"}\n{"event": "job_complete"}\n')
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "catalogue.jsonl").write_text(
        json.dumps(
            {"run_id": "r", "path": str(run), "name": "n", "created": "c", "annotations": {}}
        )
        + "\n"
    )

    store = SQLiteStore(tmp_path)

    assert store.job("j1")["state"] == "complete"
    assert store.job("j1")["priority"] == "normal"
    assert [e["event"] for _, e in store.events("j1")] == ["job_queued", "job_complete"]
    assert store.find("r") is not None
    store.update_job("j1", state="failed")
    assert SQLiteStore(tmp_path).job("j1")["state"] == "failed"  # not imported again


def test_stores_opened_at_once_on_a_new_root_all_open(tmp_path: Path) -> None:
    """Regression: the REST server's app and scheduler opened a new database together, and
    one failed with "database is locked" while switching it to WAL (no busy wait there).
    """
    import threading

    for attempt in range(20):
        root = tmp_path / f"root{attempt}"
        errors: list[BaseException] = []
        barrier = threading.Barrier(6)

        def open_store(
            root: Path = root,
            errors: list[BaseException] = errors,
            barrier: threading.Barrier = barrier,
        ) -> None:
            barrier.wait()
            try:
                SQLiteStore(root).close()
            except Exception as error:  # noqa: BLE001 (collected for the assertion)
                errors.append(error)

        threads = [threading.Thread(target=open_store) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], (attempt, errors)


def test_a_lock_error_names_the_statement_and_the_database(tmp_path: Path) -> None:
    import sqlite3

    store = SQLiteStore(tmp_path)
    other = sqlite3.connect(tmp_path / "apex.db", isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    store._execute("PRAGMA busy_timeout=100")  # not a minute
    try:
        with pytest.raises(sqlite3.OperationalError) as raised:
            store.insert_job(_job("a"))
    finally:
        other.execute("ROLLBACK")

    assert "database is locked: BEGIN IMMEDIATE on" in str(raised.value)
    assert str(tmp_path / "apex.db") in str(raised.value)
    assert raised.value.sqlite_errorname == "SQLITE_BUSY"


def test_the_job_store_is_wal_with_synchronous_normal(tmp_path: Path) -> None:
    """WAL with synchronous=NORMAL: a worker's commit per event does not hold the write lock
    through an fsync (the CI flake of 2026-10-10, on a loaded disk)."""
    store = SQLiteStore(tmp_path)

    assert store._execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    assert store._execute("PRAGMA journal_mode").fetchone()[0] == "wal"
