"""Stopping rules driven by cross-validation (CV) data.

The workflow holds the CV data out of the assimilation, scores each accepted iterate on
them (mean-square data mismatch, member mean) and hands the values to a
:class:`StoppingRule`, in order, the prior first. The rule tracks the minimum (the
posterior of the run) and says when to stop.
"""

from collections.abc import Mapping
from typing import Any, Protocol


class StoppingRule(Protocol):
    """Decides, from the CV mismatch of accepted iterates, which to keep and when to stop."""

    def observe(self, iteration: int, value: float) -> bool:
        """Take the CV mismatch of accepted iterate ``iteration``; return whether to stop."""
        ...

    @property
    def selected(self) -> int | None:
        """The iterate with the lowest CV mismatch so far (the earliest on ties)."""
        ...

    def state(self) -> dict[str, Any]:
        """Everything needed to continue (JSON-serialisable), for checkpoints."""
        ...

    def restore(self, state: Mapping[str, Any]) -> None:
        """Continue from :meth:`state`'s output."""
        ...
