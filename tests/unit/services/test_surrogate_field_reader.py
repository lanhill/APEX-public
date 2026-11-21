"""Field snapshots ask their reader for a mesh and states only.

A surrogate takes fields on an unstructured mesh only when it declares
``unstructured_fields``; any other needs a reader that promises a structured grid
(``structured_mesh``), refused before any run otherwise, with the reason.
"""

import datetime as dt
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apex.core.case_state import StateWriter
from apex.core.errors import ConfigError
from apex.core.plugins import case_state_readers, surrogate_models
from apex.domain.case_state import CaseMesh, StateOutput
from apex.domain.cell_graph import CellGraph, cell_graph_arrays
from apex.services.surrogates import (
    _field_stack,
    _hand_connectivity,
    _state_writer,
    _wants_connectivity,
    field_reader,
)
from apex.simulation.eclipse_files import EclipseFiles
from apex.simulation.geophysics import GeophysicsSimulator
from apex.storage.surrogate_directory import FIELDS_CONNECTIVITY, SurrogateDirectory
from apex.workflows.surrogate_forecasts import carry_connectivity

READER, GRID_PLUGIN, MESH_PLUGIN = "test_mesh_reader", "test_grid_fields", "test_mesh_fields"


class _MeshReader:
    """A reader class of the state alone: no structured grid, no corner geometry."""


class _GridFields:
    """A field surrogate that needs a structured grid (declares nothing)."""


class _MeshFields:
    """A field surrogate that takes an unstructured mesh."""

    unstructured_fields = True


@pytest.fixture(autouse=True)
def _plugins() -> Iterator[None]:
    case_state_readers.add(READER, _MeshReader)  # type: ignore[arg-type]
    surrogate_models.add(GRID_PLUGIN, _GridFields)  # type: ignore[arg-type]
    surrogate_models.add(MESH_PLUGIN, _MeshFields)  # type: ignore[arg-type]
    try:
        yield
    finally:
        case_state_readers.remove(READER)
        surrogate_models.remove(GRID_PLUGIN)
        surrogate_models.remove(MESH_PLUGIN)


def test_the_deck_readers_promise_a_structured_grid() -> None:
    assert field_reader("opm_flow", GRID_PLUGIN) == (EclipseFiles, True)
    assert field_reader("eclipse", MESH_PLUGIN) == (EclipseFiles, False)


def test_a_mesh_reader_feeds_a_surrogate_that_takes_a_mesh() -> None:
    """No corner geometry is asked for (phase 0 asked for it)."""
    reader, structured = field_reader(READER, MESH_PLUGIN)
    assert reader is _MeshReader  # type: ignore[comparison-overlap]
    assert not structured


def test_a_grid_based_surrogate_refuses_a_mesh_reader() -> None:
    with pytest.raises(ConfigError, match=r"'test_grid_fields' needs its fields on a structured"):
        field_reader(READER, GRID_PLUGIN)


def test_a_simulator_without_a_reader_is_refused() -> None:
    with pytest.raises(ConfigError, match="the 'linear' simulator has no case-state reader"):
        field_reader("linear", GRID_PLUGIN)


def test_a_simulators_answer_for_its_case_stands_for_the_readers_promise() -> None:
    """A ``StateWriter`` whose case's mesh is a grid feeds a grid-based surrogate."""
    reader, structured = field_reader(READER, GRID_PLUGIN, structured=True)
    assert reader is _MeshReader  # type: ignore[comparison-overlap]
    assert structured


class _Writer:
    """A simulator that leaves case files and states what its runs write (``StateWriter``)."""

    def __init__(self, structured: bool) -> None:
        self.structured = structured
        self.asked: list[tuple[list[str], list[object]]] = []

    def simulate(self, member: object) -> Path:
        raise NotImplementedError

    def read(self, case: Path, keys: object) -> None:
        raise NotImplementedError

    def clean(self, case: Path) -> None:
        raise NotImplementedError

    def run(self, member: object, keys: object) -> None:
        raise NotImplementedError

    def state_output(self, names: Sequence[str], dates: Sequence[dt.date | float]) -> StateOutput:
        self.asked.append((list(names), list(dates)))
        return StateOutput(cells=12, structured=self.structured)


class _Wrapper:
    def __init__(self, inner: object) -> None:
        self.inner = inner


def _case() -> Any:  # noqa: ANN401 (a stand-in for a configuration)
    return SimpleNamespace(geophysics=None, simulator=SimpleNamespace(name=READER))


def test_a_state_writer_is_asked_in_place_of_a_decks_template() -> None:
    writer = _Writer(structured=True)
    assert isinstance(writer, StateWriter)
    assert _state_writer(_Wrapper(writer)) is writer
    simulator, cells, per_value = _field_stack(_case(), writer, ["pressure"], ["2020-01-11"], None,  # type: ignore[arg-type]
                                    GRID_PLUGIN)  # fmt: skip
    assert cells == 12
    assert per_value == 8.0  # the writer states none: a 64-bit float assumed
    assert isinstance(simulator, GeophysicsSimulator)
    assert writer.asked == [(["PRESSURE"], [dt.date(2020, 1, 11)])]
    with pytest.raises(ConfigError, match=r"field_cells = 10, but the 'test_mesh_reader'"):
        _field_stack(_case(), writer, ["PRESSURE"], ["2020-01-11"], 10)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="needs its fields on a structured grid"):
        _field_stack(_case(), _Writer(structured=False), ["PRESSURE"], ["2020-01-11"],  # type: ignore[arg-type]
                     None, GRID_PLUGIN)  # fmt: skip


# -- -------------------------------------------------------------------


class _RegionWriter(_Writer):
    """A writer of multi-region cases: it takes ``region``."""

    def state_output(
        self, names: Sequence[str], times: Sequence[dt.date | float], *, region: str | None = None
    ) -> StateOutput:
        self.asked.append((list(names), [*times, region]))
        return StateOutput(cells=12, structured=self.structured)


def test_field_times_in_seconds_reach_the_writer_and_the_snapshots() -> None:
    """(a): numbers are seconds since the run's start; a mixed list is refused."""
    writer = _Writer(structured=True)
    simulator, _, _ = _field_stack(_case(), writer, ["PRESSURE"], ["86400", "1500.5"], None)  # type: ignore[arg-type]
    assert writer.asked == [(["PRESSURE"], [86400.0, 1500.5])]
    assert isinstance(simulator, GeophysicsSimulator)
    with pytest.raises(ConfigError, match="mix ISO dates and seconds"):
        _field_stack(_case(), writer, ["PRESSURE"], ["2020-01-11", "86400"], None)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="neither an ISO date nor a number"):
        _field_stack(_case(), writer, ["PRESSURE"], ["soon"], None)  # type: ignore[arg-type]


def test_a_region_reaches_a_writer_that_takes_one() -> None:
    """(d): [case] field_region goes to the writer and to the snapshots' reader."""
    writer = _RegionWriter(structured=True)
    simulator, _, _ = _field_stack(_case(), writer, ["T"], ["2020-01-11"], None, region="solid")  # type: ignore[arg-type]
    assert writer.asked == [(["T"], [dt.date(2020, 1, 11), "solid"])]
    assert simulator.model.region == "solid"  # type: ignore[attr-defined]
    # A writer of single-region cases (no region keyword) and a simulator without a writer.
    with pytest.raises(ConfigError, match="states have one region; leave field_region out"):
        _field_stack(_case(), _Writer(structured=True), ["T"], ["2020-01-11"], None,  # type: ignore[arg-type]
                     region="solid")  # fmt: skip
    plain = SimpleNamespace(run=lambda member, keys: None)
    with pytest.raises(ConfigError, match="states have one region"):
        _field_stack(_case(), plain, ["T"], ["2020-01-11"], 12, region="solid")


def test_the_case_section_takes_seconds_and_a_region(tmp_path: Path) -> None:
    from apex.domain.surrogate import SurrogateCaseSection

    case = SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"],
                                field_times=[86400, 1500.5, "3600"], field_region="solid")  # type: ignore[list-item]  # fmt: skip
    assert case.field_times == ["86400", "1500.5", "3600"]
    with pytest.raises(ValueError, match="mix ISO dates and seconds"):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"],
                             field_times=["2020-01-11", 86400])  # type: ignore[list-item]  # fmt: skip
    with pytest.raises(ValueError, match=">= 0"):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"], field_times=[-1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="not a date nor seconds"):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"], field_times=[True])  # type: ignore[list-item]
    with pytest.raises(
        ValueError, match=r"write the date as 2020-01-01, or the seconds .*20200101\.0"
    ):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"],
                             field_times=["20200101"])  # fmt: skip
    # Eight digits that are no date are seconds; with a decimal point they always are.
    for label in ("20201399", "20200101.0"):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", fields=["T"], field_times=[label])
    with pytest.raises(ValueError, match="field_region without fields"):
        SurrogateCaseSection(config_file=tmp_path / "c.toml", field_region="solid")


def test_a_writer_without_region_is_found_by_its_signature() -> None:
    """The review: by ``inspect.signature``, not by an error's text; ``**kwargs`` takes it."""
    from apex.services.surrogates import _takes_region

    class _Raising(_RegionWriter):
        def state_output(
            self, names: Sequence[str], times: Sequence[dt.date | float], *,
            region: str | None = None,
        ) -> StateOutput:  # fmt: skip
            raise TypeError("an unrelated bug mentioning region")

    class _Keywords(_Writer):
        def state_output(
            self, names: Sequence[str], dates: Sequence[dt.date | float], **options: object
        ) -> StateOutput:
            return StateOutput(cells=12, structured=True)

    assert not _takes_region(_Writer(structured=True))  # type: ignore[arg-type]
    assert _takes_region(_RegionWriter(structured=True))
    assert _takes_region(_Keywords(structured=True))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unrelated bug"):  # never turned into a refusal
        _field_stack(_case(), _Raising(structured=True), ["T"], ["2020-01-11"], None,  # type: ignore[arg-type]
                     region="solid")  # fmt: skip


# -- the state files' added size (follow-up 8) --------------------------------


class _SizedWriter(_Writer):
    def state_output(self, names: Sequence[str], dates: Sequence[dt.date | float]) -> StateOutput:
        return StateOutput(cells=12, structured=True, bytes_per_value=20.0)


def test_a_writers_bytes_per_value_reaches_the_plan() -> None:
    _, cells, per_value = _field_stack(_case(), _SizedWriter(structured=True), ["T"],  # type: ignore[arg-type]
                                       ["2020-01-11"], None)  # fmt: skip
    assert (cells, per_value) == (12, 20.0)
    # A simulator without state files answers its fields itself: no size.
    plain = SimpleNamespace(run=lambda member, keys: None)
    assert _field_stack(_case(), plain, ["T"], ["2020-01-11"], 12)[2] is None


def test_state_output_bytes_is_cells_fields_states_and_bytes() -> None:
    from apex.domain.case_state import state_output_bytes

    assert state_output_bytes(1000, 2, 3, 8.0) == 48_000
    assert state_output_bytes(3, 1, 1, 4.5) == 14  # rounded up
    assert state_output_bytes(0, 2, 3, 8.0) == 0
    with pytest.raises(ValueError, match="positive and finite"):
        state_output_bytes(10, 1, 1, 0.0)
    with pytest.raises(ValueError, match="negative"):
        state_output_bytes(-1, 1, 1, 8.0)


def test_the_plan_states_the_state_files_size_as_an_estimate() -> None:
    from apex.cli.surrogate_commands import surrogate_plan_lines
    from apex.domain.surrogate import SurrogateBuildPlan

    common: dict[str, Any] = {
        "name": "s", "model": "m", "plan": "prior", "training": 12, "held_out": 3,
        "simulator_runs": 15, "keys": 2, "datatypes": ["WBHP"], "measure": "relative",
        "simulator": "sim", "executor": "process", "inputs": 4, "output_dir": "out",
    }  # fmt: skip
    plain = surrogate_plan_lines(SurrogateBuildPlan(**common))
    assert not any("state files" in line for line in plain)
    sized = SurrogateBuildPlan(
        **common, state_bytes_per_run=1_500_000, state_bytes=22_500_000, state_bytes_per_value=8.0
    )
    (line,) = [line for line in surrogate_plan_lines(sized) if "state files" in line]
    assert "about 1.5 MB per run, about 22.5 MB in all (sim, an estimate" in line
    assert "restarts hold every array" in line
    # No new runs: the total is not given as 0 B.
    idle = SurrogateBuildPlan(**common, state_bytes_per_run=1_500_000, state_bytes=0)
    (line,) = [line for line in surrogate_plan_lines(idle) if "state files" in line]
    assert "per run, no new runs (" in line


def test_a_decks_active_cells_scale_the_restart_estimate(tmp_path: Path) -> None:
    """The review: a restart holds the active cells only; ACTNUM in the template's own text
    scales the estimate, anything it cannot read counts every cell."""
    from apex.services.surrogates import _deck_active_cells

    deck = tmp_path / "CASE.DATA.mako"
    deck.write_text("GRID\nACTNUM\n 3*1 2*0 -- two inactive\n 1 /\nPORO\n 6*0.2 /\n")
    assert _deck_active_cells(deck, 6) == 4
    assert _deck_active_cells(deck, 8) == 8  # not the grid's count: every cell
    deck.write_text("GRID\nACTNUM\n ${actnum} /\n")
    assert _deck_active_cells(deck, 6) == 6
    deck.write_text("GRID\nPORO\n 6*0.2 /\n")
    assert _deck_active_cells(deck, 6) == 6
    assert _deck_active_cells(None, 6) == 6


# -- -------------------------------------------------------------------

CONNECTED, GRAPH_PLUGIN = "test_connected_reader", "test_faces_fields"


class _ConnectedReader:
    """A reader class that declares the faces between its cells."""

    capabilities = frozenset({"connectivity"})


class _FacesFields:
    """A field surrogate that takes the runs' cell graph."""

    uses_connectivity = True

    def __init__(self) -> None:
        self.given: list[tuple[object, object]] = []

    def use_connectivity(self, mesh: object, graph: object) -> None:
        self.given.append((mesh, graph))


class _SomeFaces(_FacesFields):
    """A plugin class that takes the cell graph, its instances only with ``faces``."""

    def __init__(self, faces: bool = True) -> None:
        super().__init__()
        self.uses_connectivity = faces


def test_the_connectivity_file_goes_to_the_snapshots_of_a_reader_that_gives_faces(
    tmp_path: Path,
) -> None:
    case_state_readers.add(CONNECTED, _ConnectedReader)  # type: ignore[arg-type]
    surrogate_models.add(GRAPH_PLUGIN, _FacesFields)  # type: ignore[arg-type]
    try:
        target = tmp_path / FIELDS_CONNECTIVITY
        connected = SimpleNamespace(geophysics=None, simulator=SimpleNamespace(name=CONNECTED))
        simulator, _, _ = _field_stack(connected, _Writer(structured=True), ["P"],  # type: ignore[arg-type]
                                       ["2020-01-11"], None, GRAPH_PLUGIN,
                                       connectivity_file=target)  # fmt: skip
        assert isinstance(simulator, GeophysicsSimulator)
        assert simulator.model.connectivity_file == target  # type: ignore[attr-defined]
        # A reader without the capability: the snapshots write none.
        plain, _, _ = _field_stack(_case(), _Writer(structured=True), ["P"], ["2020-01-11"],  # type: ignore[arg-type]
                                   None, GRAPH_PLUGIN, connectivity_file=target)  # fmt: skip
        assert plain.model.connectivity_file is None  # type: ignore[attr-defined]
        # With or without fields (a field-less build records it too); the instance decides.
        assert _wants_connectivity(GRAPH_PLUGIN, {})
        assert not _wants_connectivity(GRID_PLUGIN, {})
        assert not _wants_connectivity("no_such_plugin", {})
        surrogate_models.add("test_some_faces", _SomeFaces)  # type: ignore[arg-type]
        try:
            assert _wants_connectivity("test_some_faces", {"faces": True})
            assert not _wants_connectivity("test_some_faces", {"faces": False})
            # Options the plugin refuses: the class's word (the build refuses them itself).
            assert _wants_connectivity("test_some_faces", {"unknown": 1})
        finally:
            surrogate_models.remove("test_some_faces")
    finally:
        case_state_readers.remove(CONNECTED)
        surrogate_models.remove(GRAPH_PLUGIN)


def test_a_plugin_that_takes_the_cell_graph_is_handed_the_recorded_one(tmp_path: Path) -> None:
    directory = SurrogateDirectory(tmp_path)
    model = _FacesFields()
    _hand_connectivity(model, directory)
    assert model.given == []  # nothing recorded: the plugin keeps to its own geometry
    mesh = CaseMesh(3, np.zeros((3, 3)), np.ones(3))
    graph = CellGraph(3, np.array([[0, 1], [1, 2]]))
    np.savez(tmp_path / FIELDS_CONNECTIVITY, allow_pickle=False, **cell_graph_arrays(mesh, graph))
    _hand_connectivity(model, directory)
    ((given_mesh, given_graph),) = model.given
    assert isinstance(given_graph, CellGraph)
    assert given_graph.same_faces(graph)
    assert isinstance(given_mesh, CaseMesh)
    assert given_mesh.cells == 3
    _hand_connectivity(_GridFields(), directory)  # takes none: not called
    (tmp_path / FIELDS_CONNECTIVITY).write_bytes(b"not a file of arrays")
    with pytest.raises(ConfigError, match="cannot read"):
        _hand_connectivity(_FacesFields(), directory)


class _Fitted:
    """A fitted model that keeps the cell graph it was fitted on."""

    def __init__(self, found: tuple[object, object] | None) -> None:
        self.found = found

    def fitted_connectivity(self) -> tuple[object, object] | None:
        return self.found


def test_a_refit_is_handed_the_cell_graph_its_model_was_fitted_on() -> None:
    """A refit has no build directory, so a model on a mesh
    refits on the mesh its active model keeps (``fitted_connectivity``)."""
    mesh = CaseMesh(3, np.zeros((3, 3)), np.ones(3))
    graph = CellGraph(3, np.array([[0, 1], [1, 2]]))
    refit = _FacesFields()
    carry_connectivity(_Fitted((mesh, graph)), refit)
    assert refit.given == [(mesh, graph)]
    nothing = _FacesFields()
    carry_connectivity(_Fitted(None), nothing)
    carry_connectivity(object(), nothing)
    assert nothing.given == []
    carry_connectivity(_Fitted((mesh, graph)), _GridFields())  # takes none: not called


def test_a_field_less_build_wraps_a_case_file_simulator_only_when_it_can_record(
    tmp_path: Path,
) -> None:
    """The field-less route: a reader with ``connectivity`` and a
    simulator leaving case files, read here; otherwise the simulator itself."""
    from apex.services.surrogates import _cell_graph_stack
    from apex.simulation.state_fields import CellGraphRecorder

    case_state_readers.add(CONNECTED, _ConnectedReader)  # type: ignore[arg-type]
    try:
        target = tmp_path / FIELDS_CONNECTIVITY
        connected: Any = SimpleNamespace(simulator=SimpleNamespace(name=CONNECTED))
        writer: Any = _Writer(structured=False)
        wrapped = _cell_graph_stack(connected, writer, target)
        assert isinstance(wrapped, CellGraphRecorder)
        assert wrapped.target == target
        assert not wrapped.needs_work_dir  # the wrapped simulator's (it states none)
        remote = SimpleNamespace(remote=True)
        assert _cell_graph_stack(connected, writer, target, remote) is writer
        assert _cell_graph_stack(_case(), writer, target) is writer  # no connectivity
        linear: Any = SimpleNamespace(simulator=SimpleNamespace(name="linear"))
        assert _cell_graph_stack(linear, writer, target) is writer
        plain: Any = object()
        assert _cell_graph_stack(connected, plain, target) is plain
        assert not target.exists()
    finally:
        case_state_readers.remove(CONNECTED)
