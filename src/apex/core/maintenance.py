"""Maintenance tasks other packages contribute to the service.

The service's daily clean-up and ``apex gc --code`` run every task registered in
:data:`apex.core.plugins.maintenance_tasks` (entry-point group ``apex.maintenance``), after
removing unused frozen copies of APEX: e.g. the hosts' copies, contributed by
the package that makes them. Core names no task.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class FrozenCopies(Protocol):
    """What a task may use of the frozen copies of APEX (``services.code.CodeCopies``)."""

    @property
    def repository(self) -> Path | None:
        """The git checkout copies are made from (``None``: none)."""
        ...

    @property
    def directory(self) -> Path:
        """Where the copies live."""
        ...

    def heads(self) -> frozenset[str]:
        """Return the commits always kept (the newest of ``master``)."""
        ...

    def in_use(self, commit: str, jobs: Iterable[dict[str, Any]]) -> list[str]:
        """Return what uses ``commit``'s copy."""
        ...

    def last_used(self, commit: str) -> str | None:
        """Return when ``commit``'s copy was last pinned."""
        ...


@dataclass(frozen=True)
class MaintenanceContext:
    """What a maintenance task works on.

    Attributes
    ----------
    root
        The services root.
    code
        The frozen copies of APEX (``services.code.CodeCopies``).
    jobs
        Returns the job records (read again by a task when it decides).
    keep_days
        ``[jobs] keep_code_days``.
    options
        The task's own options, if any.
    """

    root: Path
    code: FrozenCopies
    jobs: Callable[[], Sequence[dict[str, Any]]]
    keep_days: float
    options: Mapping[str, object] = field(default_factory=dict)


class MaintenanceTask(Protocol):
    """One maintenance task (a class registered by name; made with no arguments)."""

    def run(self, context: MaintenanceContext, dry_run: bool) -> list[str]:
        """Do (or, with ``dry_run``, only say) the task's work; return lines to report."""
        ...
