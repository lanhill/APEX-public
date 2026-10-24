"""Per-state bounds applied to posterior ensembles (PET ``prior_<name>.limits``)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_assimilation.kernels import clip_to_bounds


class StateBounds:
    """Optional lower and upper bounds for each state row.

    Parameters
    ----------
    lower, upper
        Bounds of shape ``(n_state,)``; ``None`` or ``-inf``/``inf`` means unbounded. Their
        length is checked against ``n_state`` when first applied, since the state size is
        not known at construction.

    Raises
    ------
    ValueError
        If a bound is not 1-D, the two differ in shape, or ``lower > upper`` anywhere.
    """

    __slots__ = ("lower", "upper")

    def __init__(self, lower: ArrayLike | None = None, upper: ArrayLike | None = None) -> None:
        self.lower = _as_bounds(lower)
        self.upper = _as_bounds(upper)
        if self.lower is not None and self.upper is not None:
            if self.lower.shape != self.upper.shape:
                raise ValueError(
                    f"lower and upper bounds differ in shape: "
                    f"{self.lower.shape} vs {self.upper.shape}"
                )
            # np.clip would silently return the upper bound where lower > upper.
            if np.any(self.lower > self.upper):
                raise ValueError("lower bounds must not exceed upper bounds")

    @property
    def is_unbounded(self) -> bool:
        """Whether neither bound is set."""
        return self.lower is None and self.upper is None

    def apply(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``state`` ``(n_state, n_ensemble)`` clipped to the bounds.

        Raises
        ------
        ValueError
            If a bound's length differs from ``n_state``.
        """
        if self.is_unbounded:
            return state
        for bound in (self.lower, self.upper):
            if bound is not None and bound.shape != (state.shape[0],):
                raise ValueError(f"bounds must have shape ({state.shape[0]},), got {bound.shape}")
        return clip_to_bounds(state, self.lower, self.upper)


def _as_bounds(bounds: ArrayLike | None) -> NDArray[np.float64] | None:
    if bounds is None:
        return None
    array = np.asarray(bounds, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"bounds must be 1-D (n_state,), got shape {array.shape}")
    return array
