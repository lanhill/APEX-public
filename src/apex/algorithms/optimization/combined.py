"""Several scalar objectives as one vector objective (``[[objectives]]``).

:func:`stack_objectives` stacks analytic objectives: each column is evaluated by each, and the
rows are the objectives. :class:`StackedResponses` does the same for response objectives
(simulated): the union of their response keys is simulated once per member, and each
objective evaluates its own keys, so that NPV and, say, the water injected come from one
simulation. Constraints of the stacked objectives are collected, in order.

Each objective may carry a statistic over the realizations (``mean``, ``std``,
``worst``, ``p<q>``; :mod:`.statistics`), as ``objective_statistics``: the multi-objective
optimizers rank that statistic of its values.
"""

from collections.abc import Sequence

import numpy as np

from apex.algorithms.optimization.protocols import (
    ConstrainedObjective,
    FloatArray,
    IntArray,
    Objective,
    ResponseObjective,
)
from apex.algorithms.optimization.statistics import MEAN, check_statistic
from apex.domain.observations import DataKey


class StackedObjectives:
    """Analytic objectives stacked into rows ``(n_obj, n)``.

    Parameters
    ----------
    names
        The objectives' names (labels), one per objective.
    objectives
        Scalar objectives, each returning ``(n,)``.
    statistics
        Each objective's statistic over the realizations (default: all ``mean``).
    """

    def __init__(
        self,
        names: Sequence[str],
        objectives: Sequence[Objective],
        statistics: Sequence[str] | None = None,
    ) -> None:
        if len(names) != len(objectives) or len(objectives) < 2:
            raise ValueError("stack at least two objectives, each with a name")
        self.objective_names = tuple(names)
        self.objective_statistics = _statistics(statistics, len(names))
        self.objectives = list(objectives)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(n_obj, n)`` for ``controls`` ``(n_x, n)``."""
        rows = [np.asarray(f(controls, realizations), dtype=float) for f in self.objectives]
        for name, row in zip(self.objective_names, rows, strict=True):
            if row.shape != (controls.shape[1],):
                raise ValueError(f"objective {name!r} returned shape {row.shape}, not one value "
                                 "per column (a stacked objective must be scalar)")  # fmt: skip
        return np.vstack(rows)


class ConstrainedStackedObjectives(StackedObjectives):
    """Stacked objectives of which some have constraints (collected in order)."""

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return every constrained objective's ``c_eq`` and ``c_iq``, stacked by rows."""
        n = controls.shape[1]
        eq, iq = [np.zeros((0, n))], [np.zeros((0, n))]
        for objective in self.objectives:
            if isinstance(objective, ConstrainedObjective):
                c_eq, c_iq = objective.constraints(controls)
                eq.append(np.asarray(c_eq, dtype=float).reshape(-1, n))
                iq.append(np.asarray(c_iq, dtype=float).reshape(-1, n))
        return np.vstack(eq), np.vstack(iq)


def _statistics(statistics: Sequence[str] | None, n: int) -> tuple[str, ...]:
    if statistics is None:
        return (MEAN,) * n
    if len(statistics) != n:
        raise ValueError(f"{len(statistics)} statistics for {n} objectives")
    return tuple(check_statistic(s) for s in statistics)


def stack_objectives(
    names: Sequence[str], objectives: Sequence[Objective], statistics: Sequence[str] | None = None
) -> StackedObjectives:
    """Return the objectives stacked, with constraints if any of them has some."""
    if any(isinstance(o, ConstrainedObjective) for o in objectives):
        return ConstrainedStackedObjectives(names, objectives, statistics)
    return StackedObjectives(names, objectives, statistics)


class StackedResponses:
    """Response objectives evaluated from one simulation per member: ``(n_obj, n)``.

    Parameters
    ----------
    names
        The objectives' names, one per objective.
    objectives
        Response objectives; their keys' union, in first-seen order, is what is simulated.
    statistics
        Each objective's statistic over the realizations (default: all ``mean``).
    """

    def __init__(
        self,
        names: Sequence[str],
        objectives: Sequence[ResponseObjective],
        statistics: Sequence[str] | None = None,
    ) -> None:
        if len(names) != len(objectives) or len(objectives) < 2:
            raise ValueError("stack at least two response objectives, each with a name")
        self.objective_names = tuple(names)
        self.objective_statistics = _statistics(statistics, len(names))
        self.objectives = list(objectives)
        keys: list[DataKey] = []
        self.rows: list[list[int]] = []
        for objective in self.objectives:
            rows = []
            for key in objective.response_keys():
                if key not in keys:
                    keys.append(key)
                rows.append(keys.index(key))
            self.rows.append(rows)
        self.keys = keys

    def response_keys(self) -> Sequence[DataKey]:
        """Return the union of the objectives' keys (each simulated once)."""
        return list(self.keys)

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``(n_obj, n)`` from the responses ``(n_keys, n)`` at :meth:`response_keys`."""
        return np.vstack([
            np.asarray(objective.evaluate(responses[rows]), dtype=float)
            for objective, rows in zip(self.objectives, self.rows, strict=True)
        ])  # fmt: skip
