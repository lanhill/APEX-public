"""The state as the simulator sees it: parameterizations applied in order.

A :class:`~apex.core.simulation.Parameterization` maps state variables (e.g. the latent
fields of a plurigaussian simulation) to the properties a forward model reads (facies,
porosity, permeability). :class:`PropertyMap` applies a sequence of them: each consumes
its variables and adds its outputs; variables nobody consumes pass through.
:class:`ParameterizedSimulator` hands a simulator the properties instead of the state, so
forecasts, mean-model runs and predictions all go through the same map.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import (
    MemberInput,
    MemberSeries,
    Parameterization,
    SeriesRequest,
    Simulator,
)
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout


class PropertyMap:
    """Parameterizations applied in order; the identity when there are none.

    Raises
    ------
    ValueError
        If a parameterization consumes a variable that neither the state nor an earlier
        parameterization provides, or two produce the same output.
    """

    def __init__(self, parameterizations: Sequence[Parameterization], layout: StateLayout) -> None:
        self.parameterizations = tuple(parameterizations)
        available = set(layout.names)
        grids = {v.name: v.grid for v in layout.variables if v.grid is not None}
        sizes = {v.name: v.size for v in layout.variables}
        for parameterization in self.parameterizations:
            missing = sorted(set(parameterization.variables) - available)
            if missing:
                raise ValueError(
                    f"{type(parameterization).__name__} needs {', '.join(missing)}, which "
                    "neither the state nor an earlier parameterization provides"
                )
            clashing = sorted(
                set(parameterization.outputs) & (available - set(parameterization.variables))
            )
            if clashing:
                raise ValueError(
                    f"{type(parameterization).__name__} produces {', '.join(clashing)}, which "
                    "already exist(s)"
                )
            # Outputs live on the grid of the variables they come from (one cell each).
            source = next((v for v in parameterization.variables if v in grids), None)
            available -= set(parameterization.variables)
            available |= set(parameterization.outputs)
            for output in parameterization.outputs:
                if source is not None:
                    grids[output] = grids[source]
                    sizes[output] = sizes[source]
        self.names = tuple(sorted(available))
        #: Outputs that are codes (facies), which quantity metrics such as RMSE leave out.
        self.categorical = frozenset(
            name
            for p in self.parameterizations
            for name in getattr(p, "categorical", ())
            if name in available
        )
        self.grids: dict[str, tuple[int, ...]] = {k: v for k, v in grids.items() if k in available}

    @property
    def identity(self) -> bool:
        """Whether there is no parameterization (the state is passed as is)."""
        return not self.parameterizations

    def member(self, state: Mapping[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        """Return one member's properties ``{name: (n,)}`` from its state ``{name: (size,)}``."""
        properties = dict(state)
        for parameterization in self.parameterizations:
            consumed = {name: properties.pop(name) for name in parameterization.variables}
            properties.update(parameterization.properties(consumed))
        return properties

    def ensemble(self, state: Mapping[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        """Return the ensemble's properties ``{name: (n, N)}`` from ``{name: (size, N)}``."""
        if self.identity:
            return dict(state)
        n_members = next(iter(state.values())).shape[-1]
        columns = [self.member({k: v[:, j] for k, v in state.items()}) for j in range(n_members)]
        return {
            name: np.ascontiguousarray(np.stack([c[name] for c in columns], axis=-1))
            for name in columns[0]
        }


class StackedProperties:
    """A :class:`PropertyMap` on stacked arrays, as a strategy sees the state.

    Called with a state matrix ``(n_state, k)`` in ``layout``'s order, it returns the
    properties stacked in :attr:`names` order ``(n_p, k)`` and their layout
    ``[{"name", "size", "grid"}]``.
    """

    def __init__(self, properties: PropertyMap, layout: StateLayout) -> None:
        self.properties = properties
        self.layout = layout
        self.names = properties.names
        self.grids = properties.grids

    def __call__(
        self, state: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], list[dict[str, object]]]:
        """Return the stacked properties of each column of ``state`` and their layout."""
        mapped = self.properties.ensemble(self.layout.split(state))
        blocks = [mapped[name].reshape(-1, state.shape[-1]) for name in self.names]
        stacked = np.concatenate(blocks)
        variables: list[dict[str, object]] = [
            {"name": name, "size": block.shape[0], "grid": self.grids.get(name)}
            for name, block in zip(self.names, blocks, strict=True)
        ]
        return np.ascontiguousarray(stacked), variables


class ParameterizedSimulator:
    """A simulator that is given each member's properties instead of its state.

    Capabilities (``needs_work_dir``, ``max_parallel_runs``, ``resources``, ``version``,
    ...) are the wrapped simulator's; ``write_parameters`` and ``run_series`` are given the
    properties too. Sensitivities are not offered: the simulator's gradient is with respect
    to the properties, not the state (the composition refuses the adjoint update with a
    parameterization).
    """

    def __init__(self, simulator: Simulator, properties: PropertyMap) -> None:
        self.simulator = simulator
        self.property_map = properties

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (the simulator's capabilities)
        """Read capabilities from the wrapped simulator, except its gradients."""
        if name in ("simulator", "property_map", "run_with_sensitivities", "run_series"):
            # During unpickling, a capability withheld, or one the subclass provides.
            raise AttributeError(name)
        capability = getattr(self.simulator, name)
        if name == "write_parameters":
            # A simulator that writes a member's parameters for a batch (the ert simulator)
            # writes its properties, as run() simulates them.
            return lambda member: capability(self._mapped(member))
        return capability

    def _mapped(self, member: MemberInput) -> MemberInput:
        # replace(), not a new MemberInput: the member keeps where it runs.
        return replace(member, state=self.property_map.member(member.state))

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Run the member on its properties."""
        return self.simulator.run(self._mapped(member), keys)

    def __dir__(self) -> list[str]:
        """List the wrapper's and the simulator's attributes."""
        return sorted(
            set(super().__dir__()) | set(dir(self.simulator)) - {"run_with_sensitivities"}
        )


class _SeriesParameterizedSimulator(ParameterizedSimulator):
    """A :class:`ParameterizedSimulator` of a simulator that returns time series."""

    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        """Run the member's series on its properties."""
        series: MemberSeries = self.simulator.run_series(self._mapped(member), request)  # type: ignore[attr-defined]
        return series


def parameterized(simulator: Simulator, properties: PropertyMap) -> Simulator:
    """Return ``simulator`` given properties, or ``simulator`` itself for the identity map.

    The wrapper offers ``run_series`` exactly when the simulator does.
    """
    if properties.identity:
        return simulator
    if hasattr(simulator, "run_series"):
        return _SeriesParameterizedSimulator(simulator, properties)
    return ParameterizedSimulator(simulator, properties)
