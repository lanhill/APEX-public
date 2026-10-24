"""Run summarizers: one per workflow, chosen by what a run directory holds.

A :class:`Summarizer` turns a run directory into a :class:`~apex.domain.summary.RunSummary`.
They are plugins in :data:`apex.core.plugins.summaries`, keyed by workflow
(``"history_matching"``; optimization and prediction summaries are added the same way).
"""

from pathlib import Path
from typing import Protocol

from apex.domain.summary import RunSummary


class Summarizer(Protocol):
    """Summarises the runs of one workflow."""

    def applies(self, path: Path) -> bool:
        """Whether ``path`` is a run directory of this workflow."""
        ...

    def summarize(self, path: Path) -> RunSummary:
        """Summarise the run in ``path``.

        Raises
        ------
        ApexError
            If the run cannot be read (no configuration, no scored step).
        """
        ...
