"""Which files in a model, case or example folder are a job service's state, not inputs.

A job service's root (``<root>/apex.db``, SQLite in WAL mode) may sit inside a folder that
APEX hashes or copies: a notebook's ``apex_home`` beside the deck it runs (release tester
H34). Such files are never inputs, and they must never be opened outside SQLite by a process
that also uses the database: closing any descriptor of a file drops all of that process's
POSIX locks on it, after which another process's last connection deletes the write-ahead log
this one still writes to, and its later commits are lost to every other process.
"""

from collections.abc import Iterable
from pathlib import Path

#: The database of a job service's root (``storage/sqlite_store.py``).
SERVICE_DATABASE = "apex.db"
#: SQLite's side files: a write-ahead log, its shared-memory index and a rollback journal.
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


def service_roots(folder: Path) -> frozenset[Path]:
    """Return the job services' roots strictly below ``folder`` (folders holding ``apex.db``)."""
    return frozenset(p.parent for p in folder.rglob(SERVICE_DATABASE) if p.parent != folder)


def is_service_file(path: Path, roots: Iterable[Path] = ()) -> bool:
    """Whether ``path`` is a job service's state rather than an input.

    It is when it lies in one of ``roots`` (:func:`service_roots`), is a SQLite side file,
    is a service's database, or is a database with a live side file beside it (a store in
    WAL mode: ``chats.db``, ``docs.db``, ...). A database in use in WAL mode is therefore
    never an input: a model folder's own one is hashed only while no connection holds it,
    so its use can change a run's id, but it is never opened while live.
    """
    if any(root in path.parents for root in roots):
        return True
    name = path.name
    if name.endswith(SQLITE_SIDE_SUFFIXES) or name == SERVICE_DATABASE:
        return True
    return any(path.with_name(name + suffix).exists() for suffix in SQLITE_SIDE_SUFFIXES[:2])
