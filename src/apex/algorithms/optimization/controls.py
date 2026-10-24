"""Control parameterizations: from the optimizer's controls to what the simulator reads.

An optimization's ``[[parameterizations]]`` (2026-10-04) are the same
plugins as a history match's (registry ``parameterizations``): each consumes
variables (control blocks, or the outputs of an earlier one) and produces the values a deck
template reads. The optimizer works on few, well-scaled controls; the simulation gets the
schedule. These two shape controls in time and range; the facies ones apply too.

``control_schedule``: a block of ``K`` values over ``n_steps`` report steps. ``knots`` are
the first steps of the values (0-based, increasing, the first 0). ``hold`` keeps each value
until the next knot (piecewise constant); ``linear`` interpolates between the knots' values
and holds the last after its knot.

``control_scaling``: ``lower + (upper - lower) * u`` for normalized controls ``u`` (``[0, 1]``
maps onto the bounds; values outside extrapolate, so bound the controls in ``[controls]``).
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray

from apex.core.options import TEXT
from apex.core.plugins import parameterizations


def _one(state: Mapping[str, NDArray[np.float64]], name: str, size: int) -> NDArray[np.float64]:
    if name not in state:
        raise ValueError(f"the control {name!r} is missing")
    values = np.asarray(state[name], dtype=np.float64)
    if values.shape != (size,):
        raise ValueError(f"the control {name!r} has shape {values.shape}, expected ({size},)")
    return values


@parameterizations.register("control_schedule")
class ControlSchedule:
    """A control block's values spread over the report steps of its periods.

    Parameters
    ----------
    variable
        The control block (or an earlier parameterization's output) of ``len(knots)`` values.
    output
        The name the simulation reads, ``(n_steps,)`` (default: ``variable``).
    knots
        The first report step of each value (0-based, increasing, starting at 0).
    n_steps
        The report steps of the schedule.
    interpolation
        ``hold`` (piecewise constant) or ``linear``.
    """

    status = "stable"

    def __init__(
        self,
        variable: Annotated[str, TEXT],
        knots: Sequence[int],
        n_steps: int,
        output: Annotated[str | None, TEXT] = None,
        interpolation: Literal["hold", "linear"] = "hold",
    ) -> None:
        points = np.asarray(knots, dtype=np.int64)
        if not np.array_equal(np.asarray(knots, dtype=np.float64), points):
            raise ValueError(f"knots must be whole report steps, got {list(knots)}")
        if points.ndim != 1 or points.size == 0 or points[0] != 0:
            raise ValueError("knots must start at step 0")
        if np.any(np.diff(points) <= 0) or points[-1] >= n_steps:
            raise ValueError(f"knots must increase and stay below n_steps = {n_steps}")
        if interpolation not in ("hold", "linear"):
            raise ValueError("interpolation must be 'hold' or 'linear'")
        self.variable = variable
        self.knots = points
        self.n_steps = int(n_steps)
        self.interpolation = interpolation
        self.variables: tuple[str, ...] = (variable,)
        self.outputs: tuple[str, ...] = (output or variable,)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{output: (n_steps,)}`` from the block's ``len(knots)`` values."""
        values = _one(state, self.variable, self.knots.size)
        steps = np.arange(self.n_steps)
        if self.interpolation == "hold":
            # The value of the last knot at or before each step.
            schedule = values[np.searchsorted(self.knots, steps, side="right") - 1]
        else:
            schedule = np.interp(steps, self.knots, values)  # holds after the last knot
        return {self.outputs[0]: np.asarray(schedule, dtype=np.float64)}


@parameterizations.register("control_scaling")
class ControlScaling:
    """Normalized controls in [0, 1] mapped linearly onto their physical bounds.

    Parameters
    ----------
    variable
        The normalized control block.
    lower, upper
        The physical value at 0 and at 1: scalars, or one per value of the block.
    output
        The name the simulation reads (default: ``variable``).
    """

    status = "stable"

    def __init__(
        self,
        variable: Annotated[str, TEXT],
        lower: float | Sequence[float],
        upper: float | Sequence[float],
        output: Annotated[str | None, TEXT] = None,
    ) -> None:
        low = np.atleast_1d(np.asarray(lower, dtype=np.float64))
        high = np.atleast_1d(np.asarray(upper, dtype=np.float64))
        if low.shape != high.shape or low.ndim != 1:
            raise ValueError("lower and upper must both be scalars or lists of one length")
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("lower and upper must be finite")
        if np.any(high <= low):
            raise ValueError("upper must exceed lower")
        self.variable = variable
        self.lower, self.upper = low, high
        self.variables: tuple[str, ...] = (variable,)
        self.outputs: tuple[str, ...] = (output or variable,)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{output: lower + (upper - lower) * u}`` for the block ``u``."""
        if self.variable not in state:
            raise ValueError(f"the control {self.variable!r} is missing")
        u = np.asarray(state[self.variable], dtype=np.float64)
        if self.lower.size not in (1, u.size):
            raise ValueError(
                f"{self.variable!r} has {u.size} values, but lower and upper have {self.lower.size}"
            )
        return {self.outputs[0]: self.lower + (self.upper - self.lower) * u}
