"""A toy parameterization for workflow tests: ``y = scale * x + shift``.

In a module of its own, importable by name, so that the process executor's workers (which
start with ``forkserver``) can unpickle simulators that carry it.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from apex.core.plugins import parameterizations


@parameterizations.register("test_affine")
class Affine:
    """Maps the state variable ``x`` to the property ``y``."""

    variables = ("x",)
    outputs = ("y",)

    def __init__(self, scale: float = 2.0, shift: float = 1.0) -> None:
        self.scale = float(scale)
        self.shift = float(shift)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{"y": scale * x + shift}``."""
        return {"y": self.scale * np.asarray(state["x"]) + self.shift}
