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
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.case_state import CaseStateReader
from apex.core.errors import ConfigError, SimulationError
from apex.domain.case_state import CaseMesh, StateTime
from apex.domain.cell_graph import CellGraph, cell_graph_from_arrays
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


class InSeconds(FakeCase):
    """States with seconds and no dates (a run without a calendar); two at one millisecond."""

    seconds = (0.0, 86400.0, 86400.0004)

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        return tuple(StateTime(n, s, None) for n, s in enumerate(self.seconds))

    def state(
        self,
        time: StateTime,
        names: Sequence[str],
        region: str | None = None,
        *,
        skip_missing: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        assert time.seconds is not None
        return {"PRESSURE": 10.0 * time.index + np.arange(5.0)}

    def reached(self) -> StateTime | None:
        return StateTime(-1, 172800.0, None)


def test_field_times_in_seconds_match_to_the_millisecond() -> None:
    """Seconds since the run's start; the later of one millisecond."""
    model = StateFieldModel(["PRESSURE"], ["0", "86400.0003"], cells=6, reader=InSeconds)
    values = model.forecast(Path("CASE"), field_keys(model.names, model.times, 6))
    np.testing.assert_array_equal(values[:6], [0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    assert values[6] == 20.0  # the second of the two states at 86400 s
    gap = StateFieldModel(["PRESSURE"], ["100"], cells=6, reader=InSeconds)
    with pytest.raises(ConfigError, match=r"no state at 100 s \(written: 0, 86400, 86400.0004\)"):
        gap.forecast(Path("CASE"), field_keys(gap.names, gap.times, 6))
    late = StateFieldModel(["PRESSURE"], ["200000"], cells=6, reader=InSeconds)
    with pytest.raises(SimulationError, match="the run ended at 172800 s"):
        late.forecast(Path("CASE"), field_keys(late.names, late.times, 6))
    with pytest.raises(ConfigError, match="mix ISO dates and seconds"):
        StateFieldModel(["PRESSURE"], ["0", "2025-01-01"], cells=6)
    # A run of dates only has no seconds to match.
    dated = StateFieldModel(["PRESSURE"], ["0"], cells=6, reader=FakeCase)
    with pytest.raises(SimulationError, match=r"no state at 0 s \(written: none\), and nothing"):
        dated.forecast(Path("CASE"), field_keys(dated.names, dated.times, 6))


class Regions(FakeCase):
    asked: ClassVar[list[str | None]] = []

    def mesh(self, region: str | None = None) -> CaseMesh:
        self.asked.append(region)
        return super().mesh(region)


def test_a_region_is_passed_to_the_reader() -> None:
    """(d): ``[case] field_region`` reaches the reader's mesh, times and states."""
    model = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=Regions, region="gas")
    model.forecast(Path("CASE"), field_keys(model.names, model.times, 6))
    assert Regions.asked == ["gas"]


class Connected(FakeCase):
    """A reader that declares its faces, counting its calls."""

    capabilities: ClassVar[frozenset[str]] = frozenset({"connectivity"})
    calls: ClassVar[list[Path]] = []

    def connectivity(self, region: str | None = None) -> CellGraph:
        Connected.calls.append(self.case)
        return CellGraph(5, np.array([[0, 1], [1, 2], [2, 3]]), np.ones(3), np.ones(3))


class Undeclared(Connected):
    """The method without the capability: not taken."""

    capabilities: ClassVar[frozenset[str]] = frozenset()


def test_the_first_run_records_its_cells_and_faces(tmp_path: Path) -> None:
    target = tmp_path / "build" / "fields_connectivity.npz"
    model = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=Connected,
                            connectivity_file=target)  # fmt: skip
    keys = field_keys(model.names, model.times, 6)
    Connected.calls.clear()
    plain = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=Connected)
    np.testing.assert_array_equal(model.forecast(Path("A"), keys), plain.forecast(Path("A"), keys))
    model.forecast(Path("B"), keys)
    assert Connected.calls == [Path("A")]  # once, and only where asked
    with np.load(target, allow_pickle=False) as data:
        mesh, graph = cell_graph_from_arrays({n: data[n] for n in data.files})
    assert graph.pairs.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert mesh.dims == ACTNUM.shape
    assert list(tmp_path.joinpath("build").iterdir()) == [target]  # no temporary left
    other = tmp_path / "other.npz"
    StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=Undeclared,
                    connectivity_file=other).forecast(Path("A"), keys)  # fmt: skip
    assert not other.exists()


class Broken(Connected):
    """A reader that declares its faces and cannot give them for this case."""

    def connectivity(self, region: str | None = None) -> CellGraph:
        raise ConfigError("the mesh mixes dimensions")


def test_a_reader_that_cannot_give_its_faces_leaves_no_file(tmp_path: Path) -> None:
    """The member's fields stand; no file is written (a surrogate that needs the faces then
    refuses with its own reason)."""
    target = tmp_path / "fields_connectivity.npz"
    model = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=6, reader=Broken,
                            connectivity_file=target)  # fmt: skip
    values = model.forecast(Path("A"), field_keys(model.names, model.times, 6))
    np.testing.assert_array_equal(values, [100.0, 0.0, 101.0, 102.0, 103.0, 104.0])
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


class _CaseFiles:
    """A case-file simulator stand-in: its values are the keys' order, its files nothing."""

    stops_at_report_dates = True

    def __init__(self) -> None:
        self.cleaned: list[Path] = []

    def simulate(self, member: object) -> Path:
        return Path("A")

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.arange(float(len(keys)))

    def clean(self, case: Path) -> None:
        self.cleaned.append(case)

    def run(self, member: object, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return self.read(self.simulate(member), keys)


class Unopenable(Connected):
    """A reader whose case cannot be opened."""

    def __init__(self, case: Path) -> None:
        raise SimulationError("no such files")


def test_a_build_without_fields_records_the_first_runs_graph_and_keeps_the_values(
    tmp_path: Path,
) -> None:
    """The field-less route: every key is the simulator's, the graph is
    recorded once after the first reading, and a case the reader cannot open fails nothing."""
    from apex.simulation.state_fields import CellGraphRecorder

    target = tmp_path / "fields_connectivity.npz"
    inner = _CaseFiles()
    recorder = CellGraphRecorder(inner, Connected, target)
    assert recorder.stops_at_report_dates
    keys = [DataKey("1", "WOPR:P1"), DataKey("2", "WOPR:P1")]
    Connected.calls.clear()
    np.testing.assert_array_equal(recorder.read(Path("A"), keys), [0.0, 1.0])
    recorder.read(Path("B"), keys)
    assert Connected.calls == [Path("A")]
    with np.load(target, allow_pickle=False) as data:
        _, graph = cell_graph_from_arrays({n: data[n] for n in data.files})
    assert graph.pairs.tolist() == [[0, 1], [1, 2], [2, 3]]
    recorder.clean(Path("A"))
    assert inner.cleaned == [Path("A")]
    for reader in (Unopenable, Broken, Undeclared):
        other = tmp_path / f"{reader.__name__}.npz"
        values = CellGraphRecorder(_CaseFiles(), reader, other).read(Path("A"), keys)
        np.testing.assert_array_equal(values, [0.0, 1.0])
        assert not other.exists()
