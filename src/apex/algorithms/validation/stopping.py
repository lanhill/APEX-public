"""Stopping rules on the cross-validation (CV) data mismatch.

Reference: X. Luo and P. Ranazzi (2024), *Better stopping through cross validation in an
iterative ensemble smoother: A perspective from supervised machine learning*. Geoenergy
Science and Engineering, 243, 213298. doi:10.1016/j.geoen.2024.213298. Sections as in the
manuscript's revision R1: §3.1. The paper keeps the iterate with the lowest average CV
data mismatch along the iteration path (``select``) and suggests, for practice, stopping
once the path has passed a minimum by a few iterations (``patience``). No PET equivalent.
"""

import math
from collections.abc import Mapping
from typing import Any

from apex.core.plugins import stopping_rules


class _Minimum:
    """Tracks the lowest CV mismatch; a later iterate must be strictly lower to replace it."""

    def __init__(self) -> None:
        self._best: tuple[int, float] | None = None
        self._since_best = 0

    def _track(self, iteration: int, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"the CV data mismatch of iterate {iteration} is {value}")
        # Strictly lower: on ties the earlier iterate (fewer iterations) is kept.
        if self._best is None or value < self._best[1]:
            self._best, self._since_best = (iteration, value), 0
        else:
            self._since_best += 1

    @property
    def selected(self) -> int | None:
        """The iterate with the lowest CV mismatch so far (the earliest on ties)."""
        return None if self._best is None else self._best[0]

    def state(self) -> dict[str, Any]:
        """Return the minimum and the iterates since it."""
        return {"best": None if self._best is None else list(self._best), "since": self._since_best}

    def restore(self, state: Mapping[str, Any]) -> None:
        """Continue from :meth:`state`'s output."""
        best = state["best"]
        self._best = None if best is None else (int(best[0]), float(best[1]))
        self._since_best = int(state["since"])


@stopping_rules.register("patience")
class PatienceStopping(_Minimum):
    """Stop after ``patience`` accepted iterates without a new CV minimum.

    Parameters
    ----------
    patience
        Accepted iterates to wait after the last minimum (at least 1), so that a local
        minimum is recognised before stopping.

    Raises
    ------
    ValueError
        If ``patience < 1``.
    """

    def __init__(self, patience: int = 2) -> None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be an integer >= 1, got {patience!r}")
        super().__init__()
        self.patience = patience

    def observe(self, iteration: int, value: float) -> bool:
        """Track ``value``; return whether ``patience`` iterates have passed the minimum."""
        self._track(iteration, value)
        return self._since_best >= self.patience


@stopping_rules.register("select")
class SelectStopping(_Minimum):
    """Never stop early; keep the minimum-CV iterate of the whole path (the paper's protocol)."""

    def observe(self, iteration: int, value: float) -> bool:
        """Track ``value``; never stop."""
        self._track(iteration, value)
        return False
