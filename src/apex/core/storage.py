"""Finding runs by what they compute.

A :class:`RunStore` maps run ids (``services.identity.run_id``) to run directories, so a
client can submit the same configuration twice and get the same run, and services can
list and open runs without knowing where they were written.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CatalogueEntry:
    """One run in a store.

    Attributes
    ----------
    run_id
        What the run computes (a SHA-256 hex digest).
    path
        Its run directory.
    name
        Its ``[run] name``.
    created
        When it was registered, ISO 8601.
    annotations
        The client's labels (hypothesis, study, variant, ...).
    """

    run_id: str
    path: Path
    name: str
    created: str
    annotations: dict[str, str] = field(default_factory=dict)


class RunStore(Protocol):
    """A catalogue of runs, shared by the processes of one installation."""

    def register(self, entry: CatalogueEntry) -> None:
        """Record a run (a later entry for the same id replaces an earlier one)."""
        ...

    def find(self, run_id: str) -> CatalogueEntry | None:
        """Return the run with this id whose directory still exists, or ``None``."""
        ...

    def entries(self) -> list[CatalogueEntry]:
        """Return every run whose directory still exists, oldest first."""
        ...
