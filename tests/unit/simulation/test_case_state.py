"""The case-state reader protocol and the Eclipse-format reader.

The case files are written with resdata (the ``write_case`` fixture of ``tests/conftest.py``):
INIT and restarts over the active cells. The geophysics, model-error and state-field
readings through the reader are tested with them (``test_geophysics*.py``,
``test_model_error.py``, ``test_state_fields.py``), unchanged.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from resdata import ResDataType
from resdata.resfile import FortIO, ResdataFile, ResdataKW, openFortIO

from apex.core.case_state import (
    CaseStateReader,
    ConnectivityReader,
    CornerGeometryReader,
    MissingState,
    check_region,
)
from apex.core.errors import ConfigError, SimulationError
from apex.core.plugins import case_state_readers
from apex.domain.case_state import SI_UNITS, StateTime, to_si
from apex.domain.cell_graph import grid_faces
from apex.services import composition
from apex.simulation.eclipse_files import UNIT_SYSTEMS, EclipseFiles
from apex.simulation.geophysics import CaseState
from apex.simulation.model_error import CaseFiles

SHAPE = (3, 2, 2)
ACTNUM = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
JAN, FEB = dt.date(2020, 1, 1), dt.date(2020, 2, 1)


def _two(write_case: Callable[..., Path], tmp_path: Path, **options: object) -> Path:
    restarts = {
        JAN: {"PRESSURE": 200.0, "SWAT": 0.5, "SOMAX": 0.1},
        FEB: {"PRESSURE": 210.0, "SWAT": 0.6, "SOMAX": 0.2},
    }
    return write_case(tmp_path / "C", SHAPE, restarts, actnum=ACTNUM, **options)


@pytest.mark.parametrize("unified", [False, True])
def test_the_eclipse_reader_gives_the_protocol(
    write_case: Callable[..., Path], tmp_path: Path, unified: bool
) -> None:
    reader = EclipseFiles(_two(write_case, tmp_path, unified=unified))
    assert isinstance(reader, CaseStateReader)
    assert isinstance(reader, CornerGeometryReader)
    assert reader.regions() == ()
    times = reader.times()
    # Report steps from the files (SEQNUM, or the suffix); INIT states no start here.
    assert [(t.index, t.date) for t in times] == [(0, JAN), (1, FEB)]
    assert all(t.seconds is None for t in times)
    mesh = reader.mesh()
    assert mesh.cells == 11
    assert mesh.dims == SHAPE
    assert mesh.structured
    assert mesh.active is not None
    assert int(mesh.active.sum()) == 11
    assert mesh.centres.shape == (11, 3)
    assert mesh.volumes.shape == (11,)
    assert {"PORO", "DEPTH", "INTEHEAD"} <= reader.static_names()
    np.testing.assert_array_equal(reader.static("PORO"), np.full(11, 0.25, dtype=np.float32))
    # Any array the run wrote passes through, as written; its unit is not stated.
    assert "SOMAX" in reader.names(times[1])
    arrays = reader.state(times[1], ["PRESSURE", "SOMAX"])
    assert list(arrays) == ["PRESSURE", "SOMAX"]
    np.testing.assert_array_equal(arrays["SOMAX"], np.full(11, np.float32(0.2)))
    assert reader.unit("SOMAX") is None
    with pytest.raises(SimulationError, match="no SGAS"):
        reader.state(times[0], ["PRESSURE", "SGAS"])
    assert list(reader.state(times[0], ["SGAS", "PRESSURE"], skip_missing=True)) == ["PRESSURE"]
    with pytest.raises(MissingState, match="wrote no restart at output 7"):
        reader.state(StateTime(7, None, None), ["PRESSURE"])


def test_units_follow_the_decks_unit_system(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """The deck's own units, stated; SI by one conversion, the readers' factors."""
    for code, system in UNIT_SYSTEMS.items():
        if code == 4:  # the test grid writer has no PVT-M length: its INIT is enough here
            continue
        reader = EclipseFiles(
            write_case(tmp_path / f"U{code}" / "C", SHAPE, {JAN: {"PRESSURE": 1.0}}, units=code)
        )
        assert reader.unit_system == system
        factors = {
            name: SI_UNITS[str(reader.unit(name))][0]
            for name in ("PRESSURE", "WAT_DEN", "DEPTH", "RPORV")
        }
        assert factors == {
            "PRESSURE": system.pressure,
            "WAT_DEN": system.density,
            "DEPTH": system.length,
            "RPORV": system.volume,
        }
        assert reader.unit("TEMP") == ("F" if system.name == "FIELD" else "C")
        assert reader.unit("SWAT") == "1"
    # PVT-M (its factors, from the table the reader uses).
    pvtm = UNIT_SYSTEMS[4]
    assert (SI_UNITS["atm"][0], SI_UNITS["kg/m3"][0]) == (pvtm.pressure, pvtm.density)
    values = np.array([-0.0, 1.5, 250.0])
    # A pure factor never adds 0.0, so -0.0 stays -0.0 (bit-identical to values * factor).
    assert np.array_equal(to_si(values, "bar"), values * 1.0e5)
    assert np.signbit(to_si(values, "bar")[0])
    np.testing.assert_allclose(to_si(np.array([0.0, 100.0]), "C"), [273.15, 373.15])
    np.testing.assert_allclose(to_si(np.array([32.0, 212.0]), "F"), [273.15, 373.15])
    with pytest.raises(ValueError, match="no conversion to SI from None"):
        to_si(values, None)


def test_two_restarts_on_one_day_are_two_states(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """The reader gives both; geophysics' view takes the later, as the summary does."""
    case = write_case(
        tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.5}}, actnum=ACTNUM,
        unified=True,
    )  # fmt: skip
    path = case.with_suffix(".UNRST")
    first = ResdataFile(str(path))
    blocks = [first.iget_kw(n) for n in range(len(first))]
    later = [kw.copy() for kw in blocks]
    later[0][0] = 1  # SEQNUM 1, the same date
    pressure = next(kw for kw in later if kw.get_name() == "PRESSURE")
    for i in range(len(pressure)):
        pressure[i] = 300.0
    # DOUBHEAD: elapsed days, a quarter and three quarters of a day (sub-daily steps).
    fortio = openFortIO(str(path), mode=FortIO.WRITE_MODE)
    with fortio as out:
        for kw in [*blocks[:2], _doubhead(0.25), *blocks[2:], *later[:2], _doubhead(0.75),
                   *later[2:]]:  # fmt: skip
            kw.fwrite(out)
    del first
    reader = EclipseFiles(case)
    times = reader.times()
    assert [(t.index, t.date) for t in times] == [(0, JAN), (1, JAN)]
    # Exact elapsed times from DOUBHEAD: two restarts on one day differ.
    assert [t.seconds for t in times] == [0.25 * 86400.0, 0.75 * 86400.0]
    assert reader.state(times[0], ["PRESSURE"])["PRESSURE"][0] == 200.0
    assert reader.state(times[1], ["PRESSURE"])["PRESSURE"][0] == 300.0
    assert CaseState(reader).restart(JAN, ["PRESSURE"])["PRESSURE"][0] == 300.0


def _doubhead(days: float) -> ResdataKW:
    kw = ResdataKW("DOUBHEAD", 1, ResDataType.RD_DOUBLE)
    kw[0] = days
    return kw


def _rewrite(path: Path, edit: Callable[[int, list[ResdataKW]], None]) -> None:
    """Rewrite a unified restart file, each SEQNUM block passed to ``edit`` (in place)."""
    source = ResdataFile(str(path))
    keywords = [source.iget_kw(n).copy() for n in range(len(source))]
    del source
    starts = [n for n, kw in enumerate(keywords) if kw.get_name() == "SEQNUM"]
    blocks = [keywords[a:b] for a, b in zip(starts, [*starts[1:], len(keywords)], strict=True)]
    for n, block in enumerate(blocks):
        edit(n, block)
    fortio = openFortIO(str(path), mode=FortIO.WRITE_MODE)
    with fortio as out:
        for block in blocks:
            for kw in block:
                kw.fwrite(out)


def _steps_0_3_6(write_case: Callable[..., Path], tmp_path: Path, *, bad_last: bool) -> Path:
    """A unified file with restarts at report steps 0, 3 and 6 (blocks 0, 1, 2)."""
    dates = [JAN, FEB, dt.date(2020, 3, 1)]
    restarts = {d: {"PRESSURE": 200.0 + 10 * n, "SWAT": 0.5} for n, d in enumerate(dates)}
    case = write_case(
        tmp_path / "C", SHAPE, restarts, actnum=ACTNUM, unified=True, summary_end=dates[-1]
    )

    def edit(n: int, block: list[ResdataKW]) -> None:
        block[0][0] = 3 * n  # SEQNUM: the report step
        if bad_last and n == 2:
            header = next(kw for kw in block if kw.get_name() == "INTEHEAD")
            for i in (64, 65, 66):
                header[i] = 0  # no date

    _rewrite(case.with_suffix(".UNRST"), edit)
    return case


def test_model_error_reads_a_report_step_directly(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """A report step is found by its SEQNUM, not its block, and no date is read."""
    case = _steps_0_3_6(write_case, tmp_path, bad_last=True)
    files = CaseFiles(case, EclipseFiles(case))  # constructed as the composition does
    assert files.restart(3)["PRESSURE"][0] == 210.0
    assert files.restart(0)["PRESSURE"][0] == 200.0
    np.testing.assert_array_equal(files.static("PORO"), np.full(11, np.float32(0.25)))
    # A step the run did not write: the member's error, as before.
    with pytest.raises(SimulationError, match="no restart at report step 4"):
        files.restart(4)
    with pytest.raises(MissingState, match="report steps written: 0, 3, 6"):
        files.files.at_step(4)
    # Step 6's restart has no date: model error never asks for it, so it never fails;
    # a consumer that lists the times does.
    with pytest.raises(SimulationError, match="holds no date"):
        files.files.times()


def test_case_files_default_to_the_eclipse_reader(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    case = _steps_0_3_6(write_case, tmp_path, bad_last=False)
    files = CaseFiles(case)
    assert isinstance(files.files, EclipseFiles)
    arrays = files.restart(6)
    assert arrays["PRESSURE"][0] == 220.0
    # Model error's own rule: the unwritten SOIL is 1 minus the others, unclipped, last.
    assert list(arrays) == ["PRESSURE", "SWAT", "SOIL"]
    np.testing.assert_allclose(arrays["SOIL"], 1.0 - np.float32(0.5))


def test_regions_are_checked() -> None:
    check_region(Path("C"), (), None)
    with pytest.raises(ConfigError, match="has one region, not 'gas'"):
        check_region(Path("C"), (), "gas")
    with pytest.raises(ConfigError, match="has regions gas, solid: name one"):
        check_region(Path("C"), ("gas", "solid"), None)
    with pytest.raises(ConfigError, match="no region 'rock'"):
        check_region(Path("C"), ("gas", "solid"), "rock")
    check_region(Path("C"), ("gas", "solid"), "solid")


def test_the_services_choose_readers_by_simulator_and_capability() -> None:
    assert {"opm_flow", "eclipse"} <= set(case_state_readers.names())
    assert composition.case_state_reader("opm_flow", "[case] fields") is EclipseFiles
    assert composition.case_state_reader("eclipse", "[geophysics]", "corner_geometry") is (
        EclipseFiles
    )
    with pytest.raises(ConfigError, match="the 'linear' simulator has no case-state reader"):
        composition.case_state_reader("linear", "[case] fields")

    class _Plain:
        """A reader with no capabilities beyond the state."""

    case_state_readers.add("test_plain_reader", _Plain)  # type: ignore[arg-type]
    try:
        with pytest.raises(ConfigError, match="needs a reader with corner_geometry"):
            composition.case_state_reader("test_plain_reader", "[geophysics]", "corner_geometry")
    finally:
        case_state_readers.remove("test_plain_reader")


class _StateOnly:
    """A reader of the state alone (no corner geometry)."""

    def __init__(self, case: Path) -> None:
        self.case = case

    def regions(self) -> tuple[str, ...]:
        return ()

    def mesh(self, region: str | None = None) -> object:
        raise NotImplementedError

    def static_names(self, region: str | None = None) -> frozenset[str]:
        return frozenset()

    def static(self, name: str, region: str | None = None) -> object:
        raise NotImplementedError

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        return ()

    def at_step(self, step: int) -> StateTime:
        raise NotImplementedError

    def names(self, time: StateTime, region: str | None = None) -> frozenset[str]:
        return frozenset()

    def state(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {}

    def unit(self, name: str) -> str | None:
        return None

    def reached(self) -> StateTime | None:
        return None


def test_case_state_takes_paths_and_readers_only(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    case = _two(write_case, tmp_path)
    assert CaseState(str(case)).restart_dates == [JAN, FEB]  # a str is a path
    assert isinstance(_StateOnly(case), CaseStateReader)
    with pytest.raises(ConfigError, match="geophysics needs a corner-point grid"):
        CaseState(_StateOnly(case))  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="a case's path or a case-state reader, not a int"):
        CaseState(7)  # type: ignore[arg-type]


def test_a_repeated_seqnum_keeps_its_first_block(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    """As resdata's lookup by report step (``restart_view(report_step=)``): the first block."""
    case = _steps_0_3_6(write_case, tmp_path, bad_last=False)

    def edit(n: int, block: list[ResdataKW]) -> None:
        block[0][0] = min(n, 1)  # SEQNUMs 0, 1, 1: pressures 200, 210, 220

    _rewrite(case.with_suffix(".UNRST"), edit)
    reader = EclipseFiles(case)
    assert [(t.index, t.date) for t in reader.times()] == [(0, JAN), (1, FEB)]
    assert reader.state(reader.at_step(1), ["PRESSURE"])["PRESSURE"][0] == 210.0
    resdata_view = ResdataFile(str(case.with_suffix(".UNRST"))).restart_view(report_step=1)
    assert resdata_view["PRESSURE"][0][0] == 210.0  # resdata's own choice, the same block


def test_a_unified_file_without_seqnum(write_case: Callable[..., Path], tmp_path: Path) -> None:
    """No restarts by date (as geophysics found before); a lookup by step names the file."""
    case = _two(write_case, tmp_path, unified=True)
    path = case.with_suffix(".UNRST")
    source = ResdataFile(str(path))
    keywords = [source.iget_kw(n).copy() for n in range(len(source))]
    del source
    fortio = openFortIO(str(path), mode=FortIO.WRITE_MODE)
    with fortio as out:
        for kw in keywords:
            if kw.get_name() != "SEQNUM":
                kw.fwrite(out)
    reader = EclipseFiles(case)
    assert reader.times() == ()
    with pytest.raises(SimulationError, match=r"C\.UNRST has no SEQNUM keyword") as caught:
        reader.at_step(0)
    assert not isinstance(caught.value, MissingState)
    assert CaseState(reader).restart_dates == []


def test_a_truncated_doubhead_falls_back_to_whole_days(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    case = _two(write_case, tmp_path, unified=True)
    init = case.with_suffix(".INIT")
    source = ResdataFile(str(init))
    keywords = [source.iget_kw(n).copy() for n in range(len(source))]
    del source
    for i, value in zip((64, 65, 66), (JAN.day, JAN.month, JAN.year), strict=True):
        keywords[0][i] = value  # INIT's INTEHEAD: the run's start
    fortio = openFortIO(str(init), mode=FortIO.WRITE_MODE)
    with fortio as out:
        for kw in keywords:
            kw.fwrite(out)

    def edit(n: int, block: list[ResdataKW]) -> None:
        block.insert(2, ResdataKW("DOUBHEAD", 0, ResDataType.RD_DOUBLE))  # no item 0

    _rewrite(case.with_suffix(".UNRST"), edit)
    times = EclipseFiles(case).times()
    assert [t.seconds for t in times] == [0.0, 31 * 86400.0]


def test_case_files_tell_an_unreadable_init_as_before(
    write_case: Callable[..., Path], tmp_path: Path
) -> None:
    case = _two(write_case, tmp_path)
    case.with_suffix(".INIT").write_bytes(b"not a Fortran file")
    with pytest.raises(SimulationError, match="cannot read the grid, INIT or summary of") as caught:
        CaseFiles(case)
    assert isinstance(caught.value.__cause__, SimulationError)


@pytest.mark.parametrize("unified", [False, True])
def test_field_snapshots_place_each_cell_as_the_files_hold_it(
    write_case: Callable[..., Path], tmp_path: Path, unified: bool
) -> None:
    """``mesh()``/``times()``/``state()`` place every cell as written.

    Cell-varying values per field and restart, and two restarts on 1 February (report
    steps 1 and 2: the later one must win, as geophysics' rule). The expected arrays are read
    with resdata's ``ResdataFile`` alone, placed on the grid by ACTNUM in F order, inactive
    cells 0: independent of the reader.
    """
    from apex.simulation.state_fields import StateFieldModel, field_keys

    march = dt.date(2020, 3, 1)
    cells = np.arange(11.0)
    restarts = {
        date: {"PRESSURE": 200.0 + 10.0 * n + 1.5 * cells, "SOMAX": 0.1 * (n + 1) + 0.01 * cells}
        for n, date in enumerate((JAN, FEB, march))
    }
    case = write_case(tmp_path / "C", SHAPE, restarts, actnum=ACTNUM, unified=unified)

    def to_february(header: ResdataKW) -> None:
        header[64], header[65], header[66] = FEB.day, FEB.month, FEB.year

    if unified:

        def edit(n: int, block: list[ResdataKW]) -> None:
            if n == 2:
                to_february(next(kw for kw in block if kw.get_name() == "INTEHEAD"))

        _rewrite(case.with_suffix(".UNRST"), edit)
    else:
        third = case.parent / "C.X0002"
        source = ResdataFile(str(third))
        keywords = [source.iget_kw(n).copy() for n in range(len(source))]
        del source
        to_february(keywords[0])
        with openFortIO(str(third), mode=FortIO.WRITE_MODE) as out:
            for kw in keywords:
                kw.fwrite(out)

    # Expected, from the files alone: each restart's date and arrays; the later of a day.
    blocks: list[dict[str, np.ndarray]] = []
    if unified:
        whole = ResdataFile(str(case.with_suffix(".UNRST")))
        names = [whole.iget_kw(n).get_name() for n in range(len(whole))]
        for n, name in enumerate(names):
            if name == "SEQNUM":
                blocks.append({})
            else:
                blocks[-1][name] = np.array(whole.iget_kw(n).numpy_view(), dtype=np.float64)
    else:
        for path in sorted(case.parent.glob("C.X[0-9][0-9][0-9][0-9]")):
            single = ResdataFile(str(path))
            blocks.append({
                single.iget_kw(n).get_name(): np.array(single.iget_kw(n).numpy_view(),
                                                       dtype=np.float64)
                for n in range(len(single))
            })  # fmt: skip
    by_date: dict[dt.date, dict[str, np.ndarray]] = {}
    for block in blocks:
        header = block["INTEHEAD"]
        by_date[dt.date(int(header[66]), int(header[65]), int(header[64]))] = block
    assert sorted(by_date) == [JAN, FEB]
    active = np.asarray(ACTNUM, dtype=bool)  # already F order (i fastest)
    expected = []
    for name in ("PRESSURE", "SOMAX"):
        for date in (JAN, FEB):
            full = np.zeros(12)
            full[active] = by_date[date][name]
            expected.append(full)
    # The later February restart (report step 2) is the one taken.
    np.testing.assert_array_equal(by_date[FEB]["PRESSURE"], np.float32(220.0 + 1.5 * cells))

    model = StateFieldModel(["PRESSURE", "SOMAX"], [JAN.isoformat(), FEB.isoformat()], 12,
                            EclipseFiles)  # fmt: skip
    values = model.forecast(case, field_keys(model.names, model.times, 12))
    assert np.array_equal(values, np.concatenate(expected))


def test_case_files_pass_a_transient_error_on_unchanged(
    write_case: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient failure stays transient, so the executors retry it (not wrapped)."""
    from apex.core.errors import TransientSimulationError
    from apex.simulation import eclipse_files

    case = _two(write_case, tmp_path)
    error = TransientSimulationError("stale file handle")

    def fail(case: Path) -> EclipseFiles:
        raise error

    monkeypatch.setattr(eclipse_files, "EclipseFiles", fail)
    with pytest.raises(TransientSimulationError) as caught:
        CaseFiles(case)
    assert caught.value is error


@pytest.mark.parametrize("units", [1, 2])
def test_the_eclipse_reader_gives_the_grids_faces(
    write_case: Callable[..., Path], tmp_path: Path, units: int
) -> None:
    """The active cells' logical neighbours (the grid's definition, as
    ``grid_faces`` gives it), each face's area from the corners (i+ ``dy dz``, j+ ``dx dz``,
    k+ ``dx dy``, in m² whatever the deck's length) and the centres' distance."""
    case = write_case(tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0}}, actnum=ACTNUM,
                      cell=(100.0, 50.0, 5.0), units=units)  # fmt: skip
    reader = EclipseFiles(case)
    assert "connectivity" in EclipseFiles.capabilities
    assert isinstance(reader, ConnectivityReader)
    graph = reader.connectivity()
    mesh = reader.mesh()
    assert mesh.active is not None
    pairs, axis = grid_faces(SHAPE, mesh.active)
    np.testing.assert_array_equal(graph.pairs, pairs)
    assert graph.cells == mesh.cells == 11
    metres = 1.0 if units == 1 else 0.3048
    # The EGRID's float32 depths (about 2000 ft) round dz to about 1e-4 of itself.
    assert graph.areas is not None
    assert graph.distances is not None
    np.testing.assert_allclose(
        graph.areas, np.choose(axis, [250.0, 500.0, 5000.0]) * metres**2, rtol=1e-4
    )
    np.testing.assert_allclose(
        graph.distances, np.choose(axis, [100.0, 50.0, 5.0]) * metres, rtol=1e-4
    )
    with pytest.raises(ConfigError, match="has one region"):
        reader.connectivity("gas")
