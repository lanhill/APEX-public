"""The SQLite store of a services root: jobs, events, the run catalogue.

``<root>/apex.db``, in WAL mode (one writer, many readers, across processes). Records
are JSON documents in indexed tables. :meth:`SQLiteStore.locked` is an exclusive
transaction (``BEGIN IMMEDIATE``), re-entrant within a process, for read-modify-write
sequences such as starting a queued job. Results stay in the run directories; this store
indexes them.

On first use, the phase-1 files of the root (``jobs/<id>/job.json`` with their
``events.jsonl``, and ``catalogue.jsonl``) are imported.
"""

import json
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from apex.core.jobs import JobRecord
from apex.core.storage import CatalogueEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, state TEXT NOT NULL, created TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state, created);
CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job TEXT NOT NULL, seq INTEGER NOT NULL,
    data TEXT NOT NULL, UNIQUE (job, seq)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL, created TEXT NOT NULL,
    annotations TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (name TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals (token TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS grants (id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quotas_used (
    client TEXT NOT NULL, day TEXT NOT NULL, simulator_runs INTEGER NOT NULL,
    PRIMARY KEY (client, day)
);
"""

#: What the job records and events mean. A job pinned to a commit runs a worker of
#: that commit's code against this store: raise the number whenever their meaning changes, so
#: that a worker of other code is refused at once rather than misreading or miswriting them.
#: Commits without it cannot be pinned (``apex.services.code`` reads it from a commit).
JOB_PROTOCOL = 1

#: Add or replace a catalogue entry.
INSERT_RUN = (
    "INSERT OR REPLACE INTO runs (run_id, path, name, created, annotations) VALUES (?, ?, ?, ?, ?)"
)


class _Writing:
    """Who in this process is in a write transaction on one database, and what waits."""

    def __init__(self) -> None:
        self.active = 0
        self.events: list[tuple[str, dict[str, Any]]] = []


#: (process id, database path) -> its write transactions in this process. A second store
#: of the database in the same process (a job worker's event handler) must not wait for
#: them: its own connection would wait for a transaction its own thread may hold, for the
#: whole busy timeout (60 s), holding the write lock meanwhile, and other processes waiting
#: for that lock failed with "database is locked" (the CI flake of 2026-10-10).
_WRITING: dict[tuple[int, Path], _Writing] = {}
_WRITING_LOCK = threading.Lock()


def _after_fork() -> None:
    # A forked child starts with no writers, and a lock another thread may have held.
    global _WRITING_LOCK
    _WRITING_LOCK = threading.Lock()
    _WRITING.clear()


os.register_at_fork(after_in_child=_after_fork)


class SQLiteStore:
    """Jobs (:class:`~apex.core.jobs.JobStore`) and runs (:class:`~apex.core.storage.RunStore`).

    Parameters
    ----------
    root
        The services root; the database is ``<root>/apex.db``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "apex.db"
        self._depth = 0
        self._key = (os.getpid(), self.path.resolve())
        self.root.mkdir(parents=True, exist_ok=True)
        # Autocommit; transactions are explicit (BEGIN IMMEDIATE in locked()).
        # One connection per store, usable from any thread of the process (the MCP server
        # runs tool calls in worker threads); _mutex serialises its use.
        self._db = sqlite3.connect(
            self.path, timeout=60.0, isolation_level=None, check_same_thread=False
        )
        self._mutex = threading.RLock()
        self._execute("PRAGMA busy_timeout=60000")
        self._set_wal()
        # In WAL mode NORMAL is still safe against corruption, and a commit does not wait
        # for an fsync (only checkpoints do): on a loaded disk, FULL's fsync per commit (a
        # worker commits every event it logs) held the write lock for tens of seconds, and
        # a poller waiting for it past the busy timeout failed (the CI flake of 2026-10-10).
        self._execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._note_protocol()
        self._import_phase1()

    def _note_protocol(self) -> None:
        # The store keeps the highest protocol any of its users speaks. Read first: opening a
        # store is frequent, and a write transaction on every opening would contend for the
        # lock for nothing.
        if self.job_protocol() >= JOB_PROTOCOL:
            return
        with self.locked():
            if self.job_protocol() < JOB_PROTOCOL:
                self._execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('job_protocol', ?)",
                    (str(JOB_PROTOCOL),),
                )

    def declare_service(self, pid: int) -> None:
        """Record that the service of this root is process ``pid``, speaking ``JOB_PROTOCOL``.

        Submissions pinned to a commit check it: a service of older code would
        start their workers with its own interpreter, ignoring the pin.
        """
        with self.locked():
            self._execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('service', ?)",
                (json.dumps({"pid": pid, "protocol": JOB_PROTOCOL}),),
            )

    def service_protocol(self, pid: int) -> int | None:
        """Return the job protocol service ``pid`` declared, or ``None`` (it declared none)."""
        row = self._execute("SELECT value FROM meta WHERE key = 'service'").fetchone()
        try:
            declared = json.loads(row[0]) if row is not None else {}
        except ValueError:
            return None
        if declared.get("pid") != pid:
            return None
        return int(declared["protocol"])

    def job_protocol(self) -> int:
        """Return the job protocol of the store's newest user (0)."""
        row = self._execute("SELECT value FROM meta WHERE key = 'job_protocol'").fetchone()
        return 0 if row is None else int(row[0])

    def _set_wal(self, attempts: int = 100) -> None:
        # Switching a new database to WAL can fail at once with "database is locked" while
        # another connection sets it up (SQLite does not busy-wait there), e.g. when the
        # REST server's app and scheduler, or several workers, open a new root together.
        for attempt in range(attempts):
            try:
                self._execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as error:
                if "locked" not in str(error) or attempt == attempts - 1:
                    raise
                time.sleep(0.05)

    def _execute(self, query: str, arguments: tuple[Any, ...] = ()) -> "_Rows":
        # Rows are fetched under the lock: another thread may use the connection next.
        with self._mutex:
            try:
                return _Rows(self._db.execute(query, arguments).fetchall())
            except sqlite3.OperationalError as error:
                if "locked" not in str(error):
                    raise
                # Name the statement and the database, so that a lock held too long can be
                # traced from a log or a test's output alone.
                statement = " ".join(query.split()[:4])
                named = sqlite3.OperationalError(f"{error}: {statement} on {self.path}")
                named.sqlite_errorcode = error.sqlite_errorcode
                named.sqlite_errorname = error.sqlite_errorname
                raise named from error

    def close(self) -> None:
        """Close the connection."""
        self._db.close()

    # ------------------------------------------------------------------ transactions

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive write transaction, re-entrant within this store."""
        with self._mutex:
            yield from self._transaction()

    def _transaction(self) -> Iterator[None]:
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        # A writer from before its BEGIN: an event logged meanwhile waits rather than wait
        # for the lock this transaction is about to hold.
        with _WRITING_LOCK:
            _WRITING.setdefault(self._key, _Writing()).active += 1
        try:
            self._execute("BEGIN IMMEDIATE")
        except BaseException:
            self._drain(self._end_writing())
            raise
        self._depth = 1
        try:
            yield
        except BaseException:
            self._execute("ROLLBACK")
            raise
        else:
            self._execute("COMMIT")
        finally:
            self._depth = 0
            self._drain(self._end_writing())

    def _end_writing(self) -> list[tuple[str, dict[str, Any]]]:
        """Leave this process's writers; the last one takes the events that waited."""
        with _WRITING_LOCK:
            writing = _WRITING[self._key]
            writing.active -= 1
            if writing.active:
                return []
            waiting, writing.events = writing.events, []
            return waiting

    def _drain(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        """Write the events that waited; a failure never reaches the transaction's caller."""
        if not events:
            return
        try:
            self._write_events(events)
        except sqlite3.OperationalError as error:
            if "locked" not in str(error):
                print(f"apex: {len(events)} job events not stored: {error}", file=sys.stderr)
                return
            # Another process holds the lock (a handler's write waits a second at most): the
            # events wait again, for the next writer's end or the handler's next attempt.
            with _WRITING_LOCK:
                writing = _WRITING.setdefault(self._key, _Writing())
                writing.events[:0] = events
        except Exception as error:  # noqa: BLE001 (logging must never stop the work)
            # Not to a logger: its handler is what wrote these events.
            print(f"apex: {len(events)} job events not stored: {error}", file=sys.stderr)

    def _write_events(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        # In a transaction of their own (whose end writes what waited meanwhile, in turn).
        with self.locked():
            for job_id, event in events:
                self.add_event(job_id, event)

    # ------------------------------------------------------------------ jobs

    def insert_job(self, record: JobRecord) -> None:
        """Add a job record."""
        with self.locked():
            self._execute(
                "INSERT INTO jobs (id, state, created, data) VALUES (?, ?, ?, ?)",
                (record["id"], record["state"], record["created"], json.dumps(record)),
            )

    def job(self, job_id: str) -> JobRecord:
        """Return a job's record (``KeyError`` if there is none)."""
        row = self._execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no job {job_id!r}")
        record: JobRecord = json.loads(row[0])
        return record

    def update_job(self, job_id: str, **fields: Any) -> JobRecord:  # noqa: ANN401
        """Change fields of a job's record and return it."""
        with self.locked():
            record = self.job(job_id)
            record.update(fields)
            self._execute(
                "UPDATE jobs SET state = ?, data = ? WHERE id = ?",
                (record["state"], json.dumps(record), job_id),
            )
        return record

    def jobs(self, states: Collection[str] | None = None) -> list[JobRecord]:
        """Return the jobs (in ``states``), oldest first."""
        query = "SELECT data FROM jobs"
        arguments: tuple[str, ...] = ()
        if states is not None:
            arguments = tuple(states)
            query += f" WHERE state IN ({', '.join('?' for _ in arguments)})"
        # Insertion order breaks ties: created has one-second resolution, ids are random.
        rows = self._execute(query + " ORDER BY created, rowid", arguments).fetchall()
        return [json.loads(row[0]) for row in rows]

    # ------------------------------------------------------------------ events

    def add_event(self, job_id: str, event: dict[str, Any]) -> int:
        """Append an event to a job's stream; return its per-job number."""
        with self.locked():
            row = self._execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) FROM job_events WHERE job = ?", (job_id,)
            ).fetchone()
            seq = int(row[0])
            self._execute(
                "INSERT INTO job_events (job, seq, data) VALUES (?, ?, ?)",
                (job_id, seq, json.dumps(event, default=str)),
            )
        return seq

    def add_event_soon(self, job_id: str, event: dict[str, Any]) -> None:
        """Append an event now, or once this process's write transaction on the database ends.

        For a logging handler: a record logged inside another store's transaction (the
        scheduler's, in a job worker) is written right after it, instead of waiting for it.
        """
        deadline = time.monotonic() + 60.0  # the busy timeout's minute, as add_event
        while True:
            with _WRITING_LOCK:
                writing = _WRITING.get(self._key)
                if writing is not None and writing.active:
                    writing.events.append((job_id, event))
                    return
            # A transaction of this process may begin right after the check: a write that
            # waited a minute for it could hold up the very thread that began it (logging
            # waits for this handler). So wait a second at most, then look again.
            try:
                with self._mutex:
                    self._execute("PRAGMA busy_timeout=1000")
                    try:
                        self.add_event(job_id, event)
                    finally:
                        self._execute("PRAGMA busy_timeout=60000")
                return
            except sqlite3.OperationalError as error:
                if "locked" not in str(error) or time.monotonic() > deadline:
                    raise

    def events(self, job_id: str, since: int = 0) -> list[tuple[int, dict[str, Any]]]:
        """Return a job's events numbered ``since`` and later."""
        rows = self._execute(
            "SELECT seq, data FROM job_events WHERE job = ? AND seq >= ? ORDER BY seq",
            (job_id, since),
        ).fetchall()
        return [(int(seq), json.loads(data)) for seq, data in rows]

    def feed(self, since: int = 0) -> list[tuple[int, str, dict[str, Any]]]:
        """Return every job's events after the service-wide number ``since``."""
        rows = self._execute(
            "SELECT id, job, data FROM job_events WHERE id > ? ORDER BY id", (since,)
        ).fetchall()
        return [(int(i), str(job), json.loads(data)) for i, job, data in rows]

    # ------------------------------------------------------------------ approvals, grants, quotas

    def put_approval(self, token: str, record: dict[str, Any]) -> None:
        """Create or replace an approval record."""
        with self.locked():
            self._execute(
                "INSERT OR REPLACE INTO approvals (token, data) VALUES (?, ?)",
                (token, json.dumps(record)),
            )

    def approval(self, token: str) -> dict[str, Any] | None:
        """Return an approval record, or ``None``."""
        row = self._execute("SELECT data FROM approvals WHERE token = ?", (token,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def put_grant(self, grant_id: str, record: dict[str, Any]) -> None:
        """Create or replace a budget grant."""
        with self.locked():
            self._execute(
                "INSERT OR REPLACE INTO grants (id, data) VALUES (?, ?)",
                (grant_id, json.dumps(record)),
            )

    def grant(self, grant_id: str) -> dict[str, Any] | None:
        """Return a budget grant, or ``None``."""
        row = self._execute("SELECT data FROM grants WHERE id = ?", (grant_id,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def grants(self) -> list[dict[str, Any]]:
        """Return every budget grant, oldest first."""
        rows = self._execute("SELECT data FROM grants").fetchall()
        records = [dict(json.loads(row[0])) for row in rows]
        return sorted(records, key=lambda r: (str(r.get("created_at", "")), str(r.get("id", ""))))

    def add_usage(self, client: str, day: str, simulator_runs: int) -> None:
        """Count simulator runs a client's jobs were admitted with on ``day``."""
        with self.locked():
            self._execute(
                "INSERT INTO quotas_used (client, day, simulator_runs) VALUES (?, ?, ?) "
                "ON CONFLICT (client, day) DO UPDATE SET simulator_runs = simulator_runs + ?",
                (client, day, simulator_runs, simulator_runs),
            )

    def usage(self, client: str, day: str) -> int:
        """Return a client's admitted simulator runs on ``day``."""
        row = self._execute(
            "SELECT simulator_runs FROM quotas_used WHERE client = ? AND day = ?", (client, day)
        ).fetchone()
        return 0 if row is None else int(row[0])

    # ------------------------------------------------------------------ tool calls

    def add_tool_call(self, record: dict[str, Any]) -> None:
        """Record one tool call of an LLM client."""
        with self.locked():
            self._execute(
                "INSERT INTO tool_calls (time, data) VALUES (?, ?)",
                (str(record.get("time", "")), json.dumps(record, default=str)),
            )

    def tool_calls(self, since: int = 0) -> list[tuple[int, dict[str, Any]]]:
        """Return the recorded tool calls after number ``since``."""
        rows = self._execute(
            "SELECT id, data FROM tool_calls WHERE id > ? ORDER BY id", (since,)
        ).fetchall()
        return [(int(i), json.loads(data)) for i, data in rows]

    # ------------------------------------------------------------------ studies

    def put_study(self, name: str, record: dict[str, Any]) -> None:
        """Create or replace a study's record."""
        with self.locked():
            self._execute(
                "INSERT OR REPLACE INTO studies (name, data) VALUES (?, ?)",
                (name, json.dumps(record, default=str)),
            )

    def study(self, name: str) -> dict[str, Any] | None:
        """Return a study's record, or ``None``."""
        row = self._execute("SELECT data FROM studies WHERE name = ?", (name,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    # ------------------------------------------------------------------ runs (RunStore)

    def register(self, entry: CatalogueEntry) -> None:
        """Record a run (replacing an earlier entry for the same id)."""
        with self.locked():
            self._execute(
                INSERT_RUN,
                (
                    entry.run_id,
                    str(entry.path),
                    entry.name,
                    entry.created,
                    json.dumps(entry.annotations),
                ),
            )

    def _entries(self) -> list[CatalogueEntry]:
        rows = self._execute(
            "SELECT run_id, path, name, created, annotations FROM runs ORDER BY created, rowid"
        ).fetchall()
        return [CatalogueEntry(r, Path(p), n, c, json.loads(a)) for r, p, n, c, a in rows]

    def find(self, run_id: str) -> CatalogueEntry | None:
        """Return the run with this id whose directory still exists, or ``None``."""
        row = self._execute(
            "SELECT run_id, path, name, created, annotations FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None or not Path(row[1]).exists():
            return None
        return CatalogueEntry(row[0], Path(row[1]), row[2], row[3], json.loads(row[4]))

    def entries(self) -> list[CatalogueEntry]:
        """Return every run whose directory still exists, oldest first."""
        return [e for e in self._entries() if e.path.exists()]

    def forget_missing(self) -> list[CatalogueEntry]:
        """Remove entries whose directory is gone (``apex gc``); return them."""
        with self.locked():
            gone = [e for e in self._entries() if not e.path.exists()]
            for entry in gone:
                self._execute("DELETE FROM runs WHERE run_id = ?", (entry.run_id,))
        return gone

    # ------------------------------------------------------------------ deleting

    def delete_job(self, job_id: str) -> None:
        """Delete a job's record and its events (the guests' retention)."""
        with self.locked():
            self._execute("DELETE FROM job_events WHERE job = ?", (job_id,))
            self._execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def forget_run(self, run_id: str) -> None:
        """Remove a run's catalogue entry (its directory is the caller's to delete)."""
        with self.locked():
            self._execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def delete_study(self, name: str) -> None:
        """Delete a study's record."""
        with self.locked():
            self._execute("DELETE FROM studies WHERE name = ?", (name,))

    # ------------------------------------------------------------------ phase 1

    def _import_phase1(self) -> None:
        with self.locked():
            done = self._execute("SELECT 1 FROM meta WHERE key = 'phase1_imported'").fetchone()
            if done:
                return
            for path in sorted((self.root / "jobs").glob("*/job.json")):
                record = json.loads(path.read_text())
                record.setdefault("priority", "normal")
                record.setdefault("client", "user")
                record.setdefault("retries", 0)
                self._execute(
                    "INSERT OR IGNORE INTO jobs (id, state, created, data) VALUES (?, ?, ?, ?)",
                    (record["id"], record["state"], record["created"], json.dumps(record)),
                )
                events = path.parent / "events.jsonl"
                if events.exists():
                    for seq, line in enumerate(
                        text for text in events.read_text().splitlines() if text.strip()
                    ):
                        self._execute(
                            "INSERT OR IGNORE INTO job_events (job, seq, data) VALUES (?, ?, ?)",
                            (record["id"], seq, line),
                        )
            catalogue = self.root / "catalogue.jsonl"
            if catalogue.exists():
                for line in catalogue.read_text().splitlines():
                    if line.strip():
                        raw = json.loads(line)
                        self._execute(
                            INSERT_RUN,
                            (
                                raw["run_id"],
                                raw["path"],
                                raw["name"],
                                raw["created"],
                                json.dumps(raw["annotations"]),
                            ),
                        )
            self._execute("INSERT INTO meta (key, value) VALUES ('phase1_imported', '1')")


class _Rows:
    """The rows of one statement, fetched (the cursor's interface, without the cursor)."""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def fetchone(self) -> Any:  # noqa: ANN401 (a row)
        """Return the first row, or ``None``."""
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[Any]:
        """Return every row."""
        return self.rows
