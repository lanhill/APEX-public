"""Layout of the uncertain state (model parameters) being history matched."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class StateVariable:
    """One named block of the state vector, e.g. a permeability field.

    Attributes
    ----------
    name
        Unique variable name, e.g. ``"permx"``.
    size
        Number of state entries (e.g. grid cells).
    lower, upper
        Scalar bounds applied to every entry after each update; infinite means unbounded
        (PET ``prior_<name>.limits``).
    grid
        Shape of the field on the model grid, e.g. ``(nx, ny, nz)``, when the variable is
        one (cells in Eclipse order, x fastest); ``None`` for other parameters. Used by
        spatial metrics (SSIM) and parameter maps.
    """

    name: str
    size: int
    lower: float = -math.inf
    upper: float = math.inf
    grid: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the variable."""
        if not self.name:
            raise ValueError("state variable name must not be empty")
        if self.size < 1:
            raise ValueError(f"state variable {self.name!r} must have size >= 1, got {self.size}")
        if not self.lower <= self.upper:
            raise ValueError(f"state variable {self.name!r} has lower bound above upper bound")
        if self.grid is not None and (
            not 1 <= len(self.grid) <= 3 or math.prod(self.grid) != self.size
        ):
            raise ValueError(
                f"state variable {self.name!r}: grid {self.grid} must have 1-3 dimensions "
                f"whose product is its size {self.size}"
            )


class StateLayout:
    """Ordered state variables stacked into rows of a ``(n_state, n_ensemble)`` matrix.

    Raises
    ------
    ValueError
        If there are no variables or names repeat.
    """

    __slots__ = ("_offsets", "variables")

    def __init__(self, variables: Sequence[StateVariable]) -> None:
        self.variables: tuple[StateVariable, ...] = tuple(variables)
        if not self.variables:
            raise ValueError("a state layout needs at least one variable")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"state variable names must be unique, got {names}")
        offsets = np.cumsum([0] + [v.size for v in self.variables])
        self._offsets = {
            v.name: (int(offsets[i]), int(offsets[i + 1])) for i, v in enumerate(variables)
        }

    @property
    def n_state(self) -> int:
        """Total number of state rows."""
        return sum(v.size for v in self.variables)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names in stacking order."""
        return tuple(v.name for v in self.variables)

    def split(self, matrix: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Split rows of ``matrix`` ``(n_state, ...)`` into one block per variable (views)."""
        if matrix.shape[0] != self.n_state:
            raise ValueError(f"expected {self.n_state} state rows, got {matrix.shape[0]}")
        return {name: matrix[start:stop] for name, (start, stop) in self._offsets.items()}

    def stack(self, blocks: Mapping[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Inverse of :meth:`split`: stack per-variable blocks ``(size, ...)`` into rows.

        Raises
        ------
        ValueError
            If a variable is missing or has the wrong number of rows.
        """
        missing = set(self.names) - set(blocks)
        if missing:
            raise ValueError(f"missing state variables: {sorted(missing)}")
        for variable in self.variables:
            if blocks[variable.name].shape[0] != variable.size:
                raise ValueError(
                    f"variable {variable.name!r} needs {variable.size} rows, "
                    f"got {blocks[variable.name].shape[0]}"
                )
        return np.concatenate([np.asarray(blocks[name], dtype=np.float64) for name in self.names])

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-row lower and upper bounds, each ``(n_state,)``."""
        lower = np.concatenate([np.full(v.size, v.lower) for v in self.variables])
        upper = np.concatenate([np.full(v.size, v.upper) for v in self.variables])
        return lower, upper
