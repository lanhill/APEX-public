"""A job service's root inside a model folder is never read by its hash (release tester H34).

The 3well notebook kept its ``apex_home`` beside the deck it optimizes. Hashing the deck's
folder for the optimization's id opened and closed ``apex.db`` and its ``-wal``/``-shm``
files in the notebook's process, which dropped that process's SQLite (POSIX) locks; the
first job's worker, closing its connection, then found no other user and deleted the
write-ahead log the notebook still wrote to, so the next job's worker never saw its job.
"""

import subprocess
import sys
from pathlib import Path

from apex.services.identity import tree_hash
from apex.storage.sqlite_store import SQLiteStore

HOLD = """
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute("select count(*) from jobs").fetchall()
print("ready", flush=True)
sys.stdin.readline()
db.close()
"""

READ = """
import sqlite3, sys
rows = sqlite3.connect(sys.argv[1]).execute("select id from jobs").fetchall()
print(sorted(r[0] for r in rows))
"""


def _job(job_id: str) -> dict[str, object]:
    return {"id": job_id, "kind": "run", "state": "queued", "created": "2026-11-08", "spec": {}}


def test_hashing_a_folder_holding_a_live_service_loses_none_of_its_jobs(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "deck.data").write_text("RUNSPEC\n")
    store = SQLiteStore(model / "apex_home")  # this process's connection, in WAL mode
    store.insert_job(_job("first"))
    other = subprocess.Popen(  # another process's connection: the first job's worker
        [sys.executable, "-c", HOLD, str(store.path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert other.stdout is not None
    assert other.stdin is not None
    assert other.stdout.readline().strip() == "ready"

    before = tree_hash(model)
    other.stdin.write("\n")  # its connection closes, as the worker's does when it ends
    other.stdin.flush()
    assert other.wait(timeout=30) == 0
    store.insert_job(_job("second"))

    seen = subprocess.run([sys.executable, "-c", READ, str(store.path)],
                          capture_output=True, text=True, check=True)  # fmt: skip
    assert seen.stdout.strip() == "['first', 'second']"  # a new worker sees the new job
    assert (store.root / "apex.db-wal").exists()  # this process's log was not deleted
    store.insert_job(_job("third"))
    assert tree_hash(model) == before  # the service's records are not the model's inputs
    store.close()


def test_service_state_is_never_an_input_wherever_it_lies(tmp_path: Path) -> None:
    """The rule the hash, the example copy and the ssh executor's digest share."""
    from apex.core.service_files import is_service_file, service_roots
    from apex.services.examples import example_files

    folder = tmp_path / "example"
    (folder / "apex_home" / "jobs" / "j1").mkdir(parents=True)
    (folder / "deck.data").write_text("RUNSPEC\n")
    (folder / "chats.db").write_text("db")  # a loose store in WAL mode
    (folder / "chats.db-wal").write_text("wal")
    (folder / "table.db").write_text("an input that happens to be named .db")
    for name in ("apex.db", "apex.db-wal", "apex.db-shm", "jobs/j1/worker.log"):
        (folder / "apex_home" / name).write_text("state")

    roots = service_roots(folder)
    assert roots == {folder / "apex_home"}
    kept = sorted(p.relative_to(folder).as_posix() for p in example_files(folder))
    assert kept == ["deck.data", "table.db"]
    # A folder that is itself a service's root: its other files are still inputs, so two
    # such decks never share one digest (the review).
    (folder / "apex.db").write_text("db")
    assert not is_service_file(folder / "deck.data", service_roots(folder))
    assert is_service_file(folder / "apex.db", service_roots(folder))
    assert tree_hash(folder) != tree_hash(folder / "apex_home")
