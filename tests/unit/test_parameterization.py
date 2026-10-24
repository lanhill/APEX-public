"""The property map and the parameterized simulator."""

import pickle
from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.simulation import MemberInput, MemberSeries, SeriesRequest
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.workflows.parameterization import ParameterizedSimulator, PropertyMap, parameterized


class Doubling:
    """Maps ``x`` to ``twice`` (a stand-in parameterization)."""

    variables: tuple[str, ...] = ("x",)
    outputs: tuple[str, ...] = ("twice",)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        return {"twice": 2 * state["x"]}


class Recorder:
    """A simulator that returns the sum of what it was given, with a version and a cap."""

    max_parallel_runs = 3

    def version(self) -> str:
        return "1"

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.array([sum(float(v.sum()) for v in member.state.values()), len(member.state)])

    def run_with_sensitivities(self, member: MemberInput, keys: Sequence[DataKey]) -> None:
        raise AssertionError("never offered through a parameterization")


class SeriesRecorder(Recorder):
    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        return MemberSeries(("t",), tuple(sorted(member.state)), np.zeros((len(member.state), 1)))


LAYOUT = StateLayout([StateVariable("x", 3), StateVariable("z", 2)])


def test_the_simulator_sees_properties_and_keeps_its_capabilities() -> None:
    simulator = parameterized(Recorder(), PropertyMap([Doubling()], LAYOUT))
    member = MemberInput(0, {"x": np.array([1.0, 2.0, 3.0]), "z": np.array([1.0, 1.0])})

    assert isinstance(simulator, ParameterizedSimulator)
    np.testing.assert_array_equal(simulator.run(member, []), [14.0, 2])  # twice + z
    assert simulator.max_parallel_runs == 3
    assert simulator.version() == "1"
    assert not hasattr(simulator, "run_with_sensitivities")
    assert not hasattr(simulator, "run_series")
    copy = pickle.loads(pickle.dumps(simulator))  # executors ship it to workers
    np.testing.assert_array_equal(copy.run(member, []), [14.0, 2])


def test_run_series_is_offered_exactly_when_the_simulator_has_it() -> None:
    simulator = parameterized(SeriesRecorder(), PropertyMap([Doubling()], LAYOUT))
    member = MemberInput(0, {"x": np.zeros(3), "z": np.zeros(2)})

    series = simulator.run_series(member, SeriesRequest(("FOPT",)))  # type: ignore[attr-defined]

    assert series.vectors == ("twice", "z")


def test_no_parameterization_is_the_identity() -> None:
    simulator = Recorder()
    properties = PropertyMap([], LAYOUT)

    assert parameterized(simulator, properties) is simulator
    state = {"x": np.ones((3, 4)), "z": np.ones((2, 4))}
    assert properties.ensemble(state) is not state
    np.testing.assert_array_equal(properties.ensemble(state)["x"], state["x"])


def test_the_map_checks_what_each_parameterization_consumes_and_produces() -> None:
    with pytest.raises(ValueError, match="needs x"):
        PropertyMap([Doubling()], StateLayout([StateVariable("z", 2)]))
    with pytest.raises(ValueError, match="produces twice"):
        PropertyMap([Doubling()], StateLayout([StateVariable("x", 3), StateVariable("twice", 3)]))
    grid_layout = StateLayout([StateVariable("x", 6, grid=(3, 2, 1))])
    assert PropertyMap([Doubling()], grid_layout).grids == {"twice": (3, 2, 1)}
