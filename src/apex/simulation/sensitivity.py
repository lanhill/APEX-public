"""A simulator wrapper that returns each member's sensitivities as its output.

Shared by history matching's adjoint update and the surrogate builds of plugins that learn
from gradients (``uses_gradients``): any executor runs it unchanged.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.simulation import MemberInput, Simulator
from apex.domain.observations import DataKey


class SensitivityRun:
    """A simulator whose ``run`` returns ``[values, jacobian.ravel()]``.

    It lets every executor (serial, process pools, retries, resource pools) run a
    sensitivity forecast unchanged; capabilities are read through to the simulator.
    """

    def __init__(self, simulator: Simulator, n_state: int) -> None:
        self.simulator = simulator
        self.n_state = n_state

    def output_size(self, n_keys: int) -> int:
        """Values per member: ``n_keys`` values and their ``n_keys x n_state`` Jacobian."""
        return n_keys * (1 + self.n_state)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (the simulator's capabilities)
        """Read capabilities (``needs_work_dir``, ``resources``, ...) from the simulator."""
        if name in ("simulator", "n_state"):  # during unpickling, before __init__
            raise AttributeError(name)
        return getattr(self.simulator, name)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Run the member; return its values and its flattened Jacobian, concatenated."""
        values, jacobian = self.simulator.run_with_sensitivities(member, keys)  # type: ignore[attr-defined]
        values, jacobian = np.asarray(values, float), np.asarray(jacobian, float)
        # A transposed or flattened Jacobian would have the right size and scramble dg/dm.
        if values.shape != (len(keys),) or jacobian.shape != (len(keys), self.n_state):
            raise SimulationError(
                f"member {member.member_id}: sensitivities must be values ({len(keys)},) and "
                f"a Jacobian ({len(keys)}, {self.n_state}), got {values.shape} and "
                f"{jacobian.shape}"
            )
        return np.concatenate([values, jacobian.ravel()])
