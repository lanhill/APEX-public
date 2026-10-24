"""The chat assistant's history: ``<root>/chats.db``.

A SQLite file of its own, not ``apex.db``: chat text can grow large, and a separate file is
capped, pruned and deleted without touching jobs, approvals or the run catalogue.

- ``chats``: one row per chat (title, created, last activity, the page it was started from).
- ``turns``: the conversation, numbered from 0 (role ``user`` or ``assistant``, text, and
  the tools the assistant used).
- ``events``: a turn's progress (model calls, tool calls, the answer, errors), numbered
  per chat, for the UI's event stream.
- ``proposals``: actions the assistant proposes and only a person carries out (phase 2),
  e.g. launching a study draft: ``pending``, then ``launched`` (with the job) or
  ``discarded``.

:meth:`ChatStore.prune` enforces the limits: chats inactive for more than
``retention_days`` are deleted, and while the file is larger than ``max_bytes`` the chats
with the oldest activity go too, until it is below 90 % of the limit. Deleting a chat also
deletes its traces (``<root>/chats/<id>/``), so no copy of the conversation outlives it.
"""

import datetime as dt
import json
import secrets
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '{}', busy INTEGER NOT NULL DEFAULT 0,
    closed TEXT, owner TEXT NOT NULL DEFAULT 'owner'
);
CREATE TABLE IF NOT EXISTS turns (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, n INTEGER NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}',
    created TEXT NOT NULL, PRIMARY KEY (chat_id, n)
);
CREATE TABLE IF NOT EXISTS events (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, n INTEGER NOT NULL,
    created TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (chat_id, n)
);
CREATE TABLE IF NOT EXISTS proposals (
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, n INTEGER NOT NULL,
    kind TEXT NOT NULL, draft TEXT NOT NULL, summary TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL DEFAULT 'pending',
    job TEXT, created TEXT NOT NULL, updated TEXT NOT NULL, PRIMARY KEY (chat_id, n)
);
CREATE INDEX IF NOT EXISTS chats_updated ON chats(updated);
"""

#: After pruning for size, the file is brought below this share of the limit.
PRUNE_TARGET = 0.9


def now() -> str:
    """Return the current UTC time, ISO 8601."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Chat:
    """A chat's summary."""

    id: str
    title: str
    created: str
    updated: str
    context: dict[str, Any]
    busy: bool
    closed: str | None
    turns: int
    owner: str = "owner"


@dataclass(frozen=True, slots=True)
class Turn:
    """One message of a chat."""

    n: int
    role: str
    content: str
    data: dict[str, Any]
    created: str


@dataclass(frozen=True, slots=True)
class ChatEvent:
    """One step of a chat's progress."""

    n: int
    created: str
    kind: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Proposal:
    """An action the assistant proposes; a person launches or discards it."""

    n: int
    kind: str
    draft: str
    summary: str
    data: dict[str, Any]
    state: str
    job: str | None
    created: str
    updated: str


class ChatBusy(Exception):
    """A chat already has a turn running."""


class ChatStore:
    """The chats of a services root (see the module docstring).

    Parameters
    ----------
    root
        The services root; the file is ``<root>/chats.db``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "chats.db"
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            columns = {r[1] for r in db.execute("PRAGMA table_info(chats)")}
            if "owner" not in columns:  # a file
                db.execute("ALTER TABLE chats ADD COLUMN owner TEXT NOT NULL DEFAULT 'owner'")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        # A connection per operation: the store is used from the API's threads and the
        # service loop, and each operation is short.
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    def traces(self, chat_id: str) -> Path:
        """Return the folder of a chat's traces (the agent loop's calls and decisions)."""
        return self.root / "chats" / chat_id

    # ------------------------------------------------------------------ chats

    def create(
        self, title: str, context: Mapping[str, Any] | None = None, *, owner: str = "owner"
    ) -> Chat:
        """Start a chat, for its owner: ``owner`` (the API token) or ``ui``."""
        chat_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
        stamp = now()
        with self._db() as db:
            db.execute(
                "INSERT INTO chats (id, title, created, updated, context, owner) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    title[:200] or "Chat",
                    stamp,
                    stamp,
                    json.dumps(dict(context or {})),
                    owner,
                ),
            )
        return self.get(chat_id)

    def get(self, chat_id: str) -> Chat:
        """Return a chat.

        Raises
        ------
        KeyError
            If there is no such chat.
        """
        with self._db() as db:
            row = db.execute(
                "SELECT id, title, created, updated, context, busy, closed, "
                "(SELECT COUNT(*) FROM turns WHERE chat_id = chats.id), owner FROM chats "
                "WHERE id = ?",
                (chat_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"no chat {chat_id!r}")
        return _chat(row)

    def chats(self) -> list[Chat]:
        """Return every chat, the most recently active first."""
        with self._db() as db:
            rows = db.execute(
                "SELECT id, title, created, updated, context, busy, closed, "
                "(SELECT COUNT(*) FROM turns WHERE chat_id = chats.id), owner FROM chats "
                "ORDER BY updated DESC"
            ).fetchall()
        return [_chat(r) for r in rows]

    def delete(self, chat_id: str) -> None:
        """Delete a chat, its turns, its events and its traces.

        Raises
        ------
        KeyError
            If there is no such chat.
        """
        with self._db() as db:
            deleted = db.execute("DELETE FROM chats WHERE id = ?", (chat_id,)).rowcount
        if not deleted:
            raise KeyError(f"no chat {chat_id!r}")
        shutil.rmtree(self.traces(chat_id), ignore_errors=True)

    def close(self, chat_id: str, reason: str) -> None:
        """Close a chat to new turns (e.g. at its turn limit), with the reason."""
        with self._db() as db:
            db.execute("UPDATE chats SET closed = ? WHERE id = ?", (reason, chat_id))

    def acquire(self, chat_id: str) -> None:
        """Take a chat's one running turn.

        Raises
        ------
        ChatBusy
            If a turn is already running.
        KeyError
            If there is no such chat.
        """
        with self._db() as db:
            taken = db.execute(
                "UPDATE chats SET busy = 1 WHERE id = ? AND busy = 0", (chat_id,)
            ).rowcount
        if not taken:
            self.get(chat_id)  # KeyError for an unknown chat
            raise ChatBusy(f"chat {chat_id} is already answering")

    def release(self, chat_id: str) -> None:
        """End a chat's running turn."""
        with self._db() as db:
            db.execute("UPDATE chats SET busy = 0 WHERE id = ?", (chat_id,))

    def release_all(self) -> None:
        """Free chats left busy by a service that stopped mid-turn (at service start)."""
        with self._db() as db:
            db.execute("UPDATE chats SET busy = 0 WHERE busy = 1")

    # ------------------------------------------------------------------ turns and events

    def add_turn(
        self, chat_id: str, role: str, content: str, data: Mapping[str, Any] | None = None
    ) -> Turn:
        """Append a message; the chat's last activity is now."""
        stamp = now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            (n,) = db.execute(
                "SELECT COALESCE(MAX(n) + 1, 0) FROM turns WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO turns (chat_id, n, role, content, data, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chat_id, n, role, content, json.dumps(dict(data or {}), default=str), stamp),
            )
            db.execute("UPDATE chats SET updated = ? WHERE id = ?", (stamp, chat_id))
            db.execute("COMMIT")
        return Turn(int(n), role, content, dict(data or {}), stamp)

    def turns(self, chat_id: str) -> list[Turn]:
        """Return a chat's messages, in order."""
        with self._db() as db:
            rows = db.execute(
                "SELECT n, role, content, data, created FROM turns WHERE chat_id = ? ORDER BY n",
                (chat_id,),
            ).fetchall()
        return [Turn(int(n), r, c, json.loads(d), t) for n, r, c, d, t in rows]

    def add_event(self, chat_id: str, kind: str, data: Mapping[str, Any] | None = None) -> None:
        """Record a step of a chat's progress."""
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            (n,) = db.execute(
                "SELECT COALESCE(MAX(n) + 1, 0) FROM events WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO events (chat_id, n, created, kind, data) VALUES (?, ?, ?, ?, ?)",
                (chat_id, n, now(), kind, json.dumps(dict(data or {}), default=str)),
            )
            db.execute("COMMIT")

    def tell_job_ended(
        self, chat_id: str, job: str, content: str, data: Mapping[str, Any], state: str
    ) -> bool:
        """Tell a chat its job ended, once: the turn and the ``job_ended`` event; or nothing.

        The check and both writes are one transaction: the job's notification and the
        launch's late follow-up may both tell, from two processes, and
        a check outside the writes let both through. Returns whether this call told it.
        """
        stamp = now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            told = db.execute(
                "SELECT data FROM events WHERE chat_id = ? AND kind = 'job_ended'", (chat_id,)
            ).fetchall()
            if any(_job_of(d) == job for (d,) in told):
                db.execute("ROLLBACK")
                return False
            (turn,) = db.execute(
                "SELECT COALESCE(MAX(n) + 1, 0) FROM turns WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO turns (chat_id, n, role, content, data, created) "
                "VALUES (?, ?, 'assistant', ?, ?, ?)",
                (chat_id, turn, content, json.dumps(dict(data), default=str), stamp),
            )
            db.execute("UPDATE chats SET updated = ? WHERE id = ?", (stamp, chat_id))
            (event,) = db.execute(
                "SELECT COALESCE(MAX(n) + 1, 0) FROM events WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO events (chat_id, n, created, kind, data) VALUES (?, ?, ?, ?, ?)",
                (chat_id, event, stamp, "job_ended",
                 json.dumps({"job": job, "state": state})),
            )  # fmt: skip
            db.execute("COMMIT")
        return True

    def events(self, chat_id: str, since: int = 0) -> list[ChatEvent]:
        """Return a chat's events from number ``since`` on."""
        with self._db() as db:
            rows = db.execute(
                "SELECT n, created, kind, data FROM events WHERE chat_id = ? AND n >= ? ORDER BY n",
                (chat_id, since),
            ).fetchall()
        return [ChatEvent(int(n), c, k, json.loads(d)) for n, c, k, d in rows]

    # ------------------------------------------------------------------ proposals

    def add_proposal(
        self, chat_id: str, kind: str, draft: str, summary: str, data: Mapping[str, Any]
    ) -> Proposal:
        """Record a proposal (pending); earlier pending ones of the same draft are replaced."""
        stamp = now()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE proposals SET state = 'replaced', updated = ? "
                "WHERE chat_id = ? AND draft = ? AND state = 'pending'",
                (stamp, chat_id, draft),
            )
            (n,) = db.execute(
                "SELECT COALESCE(MAX(n) + 1, 0) FROM proposals WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO proposals (chat_id, n, kind, draft, summary, data, created, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (chat_id, n, kind, draft, summary, json.dumps(dict(data), default=str), stamp,
                 stamp),
            )  # fmt: skip
            db.execute("UPDATE chats SET updated = ? WHERE id = ?", (stamp, chat_id))
            db.execute("COMMIT")
        return self.proposal(chat_id, int(n))

    def proposal(self, chat_id: str, n: int) -> Proposal:
        """Return a proposal.

        Raises
        ------
        KeyError
            If there is no such proposal.
        """
        with self._db() as db:
            row = db.execute(
                "SELECT n, kind, draft, summary, data, state, job, created, updated "
                "FROM proposals WHERE chat_id = ? AND n = ?",
                (chat_id, n),
            ).fetchone()
        if row is None:
            raise KeyError(f"no proposal {n} in chat {chat_id!r}")
        return _proposal(row)

    def proposals(self, chat_id: str) -> list[Proposal]:
        """Return a chat's proposals, in order."""
        with self._db() as db:
            rows = db.execute(
                "SELECT n, kind, draft, summary, data, state, job, created, updated "
                "FROM proposals WHERE chat_id = ? ORDER BY n",
                (chat_id,),
            ).fetchall()
        return [_proposal(r) for r in rows]

    def launched(self, job: str) -> list[tuple[str, Proposal]]:
        """Return the (chat, proposal) pairs whose launch made ``job``."""
        with self._db() as db:
            rows = db.execute(
                "SELECT chat_id, n, kind, draft, summary, data, state, job, created, updated "
                "FROM proposals WHERE job = ? AND state = 'launched'",
                (job,),
            ).fetchall()
        return [(str(r[0]), _proposal(r[1:])) for r in rows]

    def settle(self, chat_id: str, n: int, state: str, job: str | None = None) -> Proposal:
        """Mark a pending proposal ``launched`` (with its job) or ``discarded``.

        Raises
        ------
        KeyError
            If there is no such proposal.
        ValueError
            If it is not pending any more, or ``state`` is neither.
        """
        if state not in ("launched", "discarded"):
            raise ValueError(f"a proposal is launched or discarded, not {state!r}")
        with self._db() as db:
            changed = db.execute(
                "UPDATE proposals SET state = ?, job = ?, updated = ? "
                "WHERE chat_id = ? AND n = ? AND state = 'pending'",
                (state, job, now(), chat_id, n),
            ).rowcount
        if not changed:
            current = self.proposal(chat_id, n)  # KeyError if there is none
            raise ValueError(f"proposal {n} is {current.state}, not pending")
        return self.proposal(chat_id, n)

    # ------------------------------------------------------------------ limits

    def size(self) -> int:
        """Return the bytes the history takes on disk (the file and its write-ahead log).

        SQLite deletes the write-ahead log when the last connection closes, which may
        happen between listing a file and reading its size: a file that is gone counts as
        empty.
        """
        total = 0
        for path in (self.path, self.path.with_name(self.path.name + "-wal")):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def prune(self, *, retention_days: float, max_bytes: int) -> list[str]:
        """Enforce the limits (see the module docstring); return the deleted chats.

        Busy chats are never deleted for size (a turn is running); they are for age only
        if inactive past the retention, which a running turn is not.
        """
        cutoff = (dt.datetime.now(dt.UTC) - dt.timedelta(days=retention_days)).isoformat(
            timespec="seconds"
        )
        deleted: list[str] = []
        with self._db() as db:
            old = [r[0] for r in db.execute("SELECT id FROM chats WHERE updated < ?", (cutoff,))]
        for chat_id in old:
            self.delete(chat_id)
            deleted.append(chat_id)
        if self.size() > max_bytes:
            deleted.extend(self._shrink(int(max_bytes * PRUNE_TARGET)))
        elif deleted:
            self._compact()
        return deleted

    def _shrink(self, target: int) -> list[str]:
        """Delete the chats of oldest activity until the file is below ``target`` bytes."""
        deleted: list[str] = []
        self._compact()
        while self.size() > target:
            with self._db() as db:
                rows = db.execute(
                    "SELECT id, (SELECT COALESCE(SUM(LENGTH(content) + LENGTH(data)), 0) "
                    "FROM turns WHERE chat_id = chats.id) + (SELECT COALESCE(SUM(LENGTH(data)"
                    "), 0) FROM events WHERE chat_id = chats.id) FROM chats WHERE busy = 0 "
                    "ORDER BY updated"
                ).fetchall()
            if not rows:
                break
            # Delete about as much as the excess (estimated from the stored text), then
            # measure again after compacting.
            excess, batch, freed = self.size() - target, [], 0
            for chat_id, stored in rows:
                batch.append(chat_id)
                freed += int(stored)
                if freed >= excess:
                    break
            for chat_id in batch:
                self.delete(chat_id)
            deleted.extend(batch)
            self._compact()
        return deleted

    def _compact(self) -> None:
        """Return freed space to the file system (checkpoint the log, then vacuum)."""
        with self._db() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _chat(row: tuple[Any, ...]) -> Chat:
    chat_id, title, created, updated, context, busy, closed, turns, owner = row
    return Chat(
        chat_id, title, created, updated, json.loads(context), bool(busy), closed, turns, owner
    )


def _proposal(row: tuple[Any, ...]) -> Proposal:
    n, kind, draft, summary, data, state, job, created, updated = row
    return Proposal(int(n), kind, draft, summary, json.loads(data), state, job, created, updated)


def _job_of(data: str) -> object:
    """Return the job an event's data names, or None (an old row of another shape)."""
    try:
        value = json.loads(data)
    except ValueError:
        return None
    return value.get("job") if isinstance(value, dict) else None
