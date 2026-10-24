"""Where jobs, their events, approvals and quota use are kept.

A job is a JSON-able record (``id``, ``kind``, ``state``, ``spec``, ...). A
:class:`JobStore` keeps them for the job runner, the service process and every client,
across processes: ``locked`` gives exclusive, all-or-nothing updates (e.g. "start this
queued job" never happens twice).
"""

from collections.abc import Collection
from contextlib import AbstractContextManager
from typing import Any, Protocol

#: A job's record, as stored (JSON values).
JobRecord = dict[str, Any]


class JobStore(Protocol):
    """Jobs, job events, approvals and quota use, shared by the processes of one root."""

    def locked(self) -> AbstractContextManager[None]:
        """Hold exclusive write access for a read-modify-write sequence (re-entrant)."""
        ...

    def insert_job(self, record: JobRecord) -> None:
        """Add a job record."""
        ...

    def job(self, job_id: str) -> JobRecord:
        """Return a job's record.

        Raises
        ------
        KeyError
            If there is no such job.
        """
        ...

    def update_job(self, job_id: str, **fields: Any) -> JobRecord:  # noqa: ANN401
        """Change fields of a job's record and return it."""
        ...

    def jobs(self, states: Collection[str] | None = None) -> list[JobRecord]:
        """Return the jobs (in these states), oldest first."""
        ...

    def add_event(self, job_id: str, event: dict[str, Any]) -> int:
        """Append an event to a job's stream; return its number in the job (from 0)."""
        ...

    def events(self, job_id: str, since: int = 0) -> list[tuple[int, dict[str, Any]]]:
        """Return a job's events numbered ``since`` and later."""
        ...

    def feed(self, since: int = 0) -> list[tuple[int, str, dict[str, Any]]]:
        """Return every job's events after the service-wide number ``since``."""
        ...

    def put_approval(self, token: str, record: dict[str, Any]) -> None:
        """Create or replace an approval record."""
        ...

    def approval(self, token: str) -> dict[str, Any] | None:
        """Return an approval record, or ``None``."""
        ...

    def add_usage(self, client: str, day: str, simulator_runs: int) -> None:
        """Count simulator runs a client's jobs were admitted with on ``day``."""
        ...

    def usage(self, client: str, day: str) -> int:
        """Return a client's admitted simulator runs on ``day``."""
        ...
