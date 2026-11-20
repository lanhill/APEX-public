"""State fields as field keys: the reader's placement and refusals.

The run's files are stood in for by a fake case-state reader (a mesh, its states
by date, arrays over its active cells in the simulator's active order); the real reading is
the Eclipse-format reader's, tested with it (``test_case_state.py``) and the geophysics
(``test_geophysics*.py``). Phase 1 moved the snapshots from ``CaseState`` (a
corner-point grid) to the reader's ``mesh()``, ``times()`` and ``state()``; these tests
patched ``CaseState`` before and give the model a reader now, with the same assertions.
"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.case_state import CaseStateReader
from apex.core.errors import ConfigError, SimulationError
from apex.domain.case_state import CaseMesh, StateTime
from apex.domain.observations import DataKey
from apex.simulation.state_fields import StateFieldModel, field_keys, split_fields

#: A 3 x 2 x 1 grid whose cell (i=1, j=0) is inactive: global F order 0..5, active 0,2,3,4,5.
ACTNUM = np.ones((3, 2, 1), dtype=bool)
ACTNUM[1, 0, 0] = False

DATES = (dt.date(2025, 1, 1), dt.date(2025, 2, 1))


class FakeCase:
    """A reader of a structured grid (``ACTNUM``), or of an unstructured mesh of 5 cells."""

    unstructured = False
    dates: tuple[dt.date, ...] = DATES
    reached_on: dt.date | None = dt.date(2025, 12, 31)
    step_offset = 0.0

    def __init__(self, case: Path) -> None:
        self.case = case

    def regions(self) -> tuple[str, ...]:
        return ()

    def mesh(self, region: str | None = None) -> CaseMesh:
        centres, volumes = np.zeros((5, 3)), np.ones(5)
        if self.unstructured:
            return CaseMesh(5, centres, volumes)
        return CaseMesh(5, centres, volumes, ACTNUM.shape, ACTNUM.ravel(order="F"))

    def static_names(self, region: str | None = None) -> frozenset[str]:
        return frozenset()

    def static(self, name: str, region: str | None = None) -> NDArray[np.float64]:
        raise SimulationError("no static arrays")

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        return tuple(StateTime(n, None, d) for n, d in enumerate(self.dates))

    def names(self, time: StateTime, region: str | None = None) -> frozenset[str]:
        return frozenset({"PRESSURE"})

    def at_step(self, step: int) -> StateTime:
        return self.times()[step]

    def state(
        self,
        time: StateTime,
        names: Sequence[str],
        region: str | None = None,
        *,
        skip_missing: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        assert time.date is not None
        base = 100.0 * time.date.month + self.step_offset * time.index
        return {"PRESSURE": base + np.arange(5.0)} if "PRESSURE" in names else {}

    def unit(self, name: str) -> str | None:
        return None

    def reached(self) -> StateTime | None:
        return None if self.reached_on is None else StateTime(-1, None, self.reached_on)


class MeshCase(FakeCase):
    unstructured = True


def test_the_fake_is_a_reader() -> None:
    assert isinstance(FakeCase(Path("CASE")), CaseStateReader)


def test_fields_are_placed_on_the_whole_grid_in_eclipse_order() -> None:
    model = StateFieldModel(["pressure"], ["2025-01-01", "2025-02-01"], cells=6,
                           reader=FakeCase)  # fmt: skip
    keys = field_keys(model.names, model.times, 6)
    assert all(model.owns(k) for k in keys)
    assert not model.owns(DataKey("2025-01-01", "WOPR:P1"))
    values = model.forecast(Path("CASE"), keys)
    january = values[:6]
    # Active cells in F order take the restart's values in order; the inactive one is 0.
    np.testing.assert_array_equal(january, [100.0, 0.0, 101.0, 102.0, 103.0, 104.0])
    np.testing.assert_array_equal(values[6:], [200.0, 0.0, 201.0, 202.0, 203.0, 204.0])
    split = split_fields(values[:, None], model.names, model.times, 6)
    assert split["PRESSURE"].shape == (6, 2, 1)
    np.testing.assert_array_equal(split["PRESSURE"][:, 1, 0], values[6:])


def test_a_missing_array_and_another_grid_are_the_decks_errors() -> None:
    swat = StateFieldModel(["SWAT"], ["2025-01-01"], cells=6, reader=FakeCase)
    with pytest.raises(ConfigError, match="no SWAT"):
        swat.forecast(Path("CASE"), field_keys(swat.names, swat.times, 6))
    wrong = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=8, reader=FakeCase)
    with pytest.raises(ConfigError, match="6 cells, the build 8"):
        wrong.forecast(Path("CASE"), field_keys(wrong.names, wrong.times, 8))
    with pytest.raises(ConfigError, match="ISO dates"):
        StateFieldModel(["PRESSURE"], ["step 3"], cells=6)


def test_an_unstructured_mesh_gives_its_cells_in_its_own_order() -> None:
    """Decision 5: no padding, no grid; the mesh's cells as the reader gives them."""
    model = StateFieldModel(["PRESSURE"], ["2025-01-01", "2025-02-01"], cells=5, reader=MeshCase)
    keys = field_keys(model.names, model.times, 5)
    values = model.forecast(Path("CASE"), keys)
    np.testing.assert_array_equal(values, [*(100.0 + np.arange(5.0)), *(200.0 + np.arange(5.0))])
    wrong = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=MeshCase)
    with pytest.raises(ConfigError, match="mesh has 5 cells, the build 6"):
        wrong.forecast(Path("CASE"), field_keys(wrong.names, wrong.times, 6))
    grid_only = StateFieldModel(["PRESSURE"], ["2025-01-01"], 5, MeshCase, structured=True)
    with pytest.raises(ConfigError, match="unstructured, and the build's surrogate needs a"):
        grid_only.forecast(Path("CASE"), field_keys(grid_only.names, grid_only.times, 5))


class SameDay(FakeCase):
    dates = (DATES[0], DATES[0])
    step_offset = 10.0


def test_the_later_state_of_a_day_is_taken() -> None:
    """As geophysics' CaseState (the summary's report step's last)."""
    model = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=SameDay)
    values = model.forecast(Path("CASE"), field_keys(model.names, model.times, 6))
    assert values[0] == 110.0  # the second state (index 1)


class EndedEarly(FakeCase):
    reached_on = dt.date(2025, 1, 15)


class NoRecord(FakeCase):
    reached_on = None


def test_a_missing_date_is_the_members_or_the_inputs_error() -> None:
    late = ["2025-03-01"]
    for reader, error, match in (
        (FakeCase, ConfigError, "must write a state at every field time"),
        (EndedEarly, SimulationError, "the run ended on 2025-01-15"),
        (NoRecord, SimulationError, "nothing tells how far it ran"),
    ):
        model = StateFieldModel(["PRESSURE"], late, cells=6, reader=reader)
        with pytest.raises(error, match=match):
            model.forecast(Path("CASE"), field_keys(model.names, model.times, 6))
