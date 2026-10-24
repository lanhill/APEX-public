"""The geophysics simulator, its case reading and the AVO forward model.

The case files are written with resdata (the ``write_case`` fixture of ``tests/conftest.py``);
no simulator runs. The forward model's values are checked against the kernels called directly on the
same columns, so these tests pin the reading, the keys and the layout, not the physics
(the kernels have their own parity and reference tests).
"""

import datetime as dt
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from apex.algorithms.geophysics import (
    Columns,
    ConstantOverburden,
    FlowState,
    PetroElasticModel,
    RickerWavelet,
    SeismicModel,
    SoftSandModel,
    StandardModel,
    TimeAxis,
    Zoeppritz,
    avo_attributes,
)
from apex.core.errors import ConfigError, SimulationError
from apex.core.simulation import MemberInput
from apex.domain.config import GeophysicsSection, avo_datatype
from apex.domain.observations import DataKey
from apex.simulation.geophysics import CaseState, GeophysicsSimulator, Grid
from apex.workflows.geophysics import AVOSurvey, GeophysicsModel, build_geophysics_model


class _Writers:
    """The conftest's case writers, for helpers that are not tests (set per test)."""

    write_case: Callable[..., Path]
    write_grid: Callable[..., Path]


_cases = _Writers()


@pytest.fixture(autouse=True)
def _case_writers(write_case: Callable[..., Path], write_grid: Callable[..., Path]) -> None:
    _cases.write_case, _cases.write_grid = write_case, write_grid


SHAPE = (3, 2, 2)
JAN, JUL = dt.date(2020, 1, 1), dt.date(2020, 7, 1)
#: Cell 5 (I=3, J=2, K=1) inactive: column (3, 2) keeps its lower cell.
ACTNUM = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
N_ACTIVE = 11


def _restarts(swat: float | None = None) -> dict[dt.date, dict[str, object]]:
    sw = np.linspace(0.2, 0.8, N_ACTIVE) if swat is None else swat
    return {JAN: {"PRESSURE": 200.0, "SWAT": 0.3}, JUL: {"PRESSURE": 180.0, "SWAT": sw}}


def _section(tmp_path: Path, **options: object) -> GeophysicsSection:
    grid = _cases.write_grid(tmp_path / "REF.EGRID", SHAPE, actnum=ACTNUM)
    avo = {
        "angles": [10, 12.5],
        "wavelet": {"frequency": 45.0, "length": 0.05},
        "overburden": {"vp": 2450.0, "vs": 849.0, "density": 2300.0},
        "times": {"start": 1.62, "end": 1.66, "step": 0.002},
        **options,
    }
    return GeophysicsSection.model_validate(
        {
            "pem": {"name": "standard"},
            "overburden_pressure": {"gradient": 22600.0, "surface": 101325.0},
            "grid_file": grid,
            "data": [
                {"kind": "avo", "datatype": "avo", "vintages": ["2020-07-01"], "options": avo}
            ],
        }
    )


def _model(section: GeophysicsSection, pem: PetroElasticModel | None = None) -> GeophysicsModel:
    d = section.data[0].avo()
    attribute = avo_attributes.create("trace" if d.form == "traces" else d.attribute)
    return build_geophysics_model(section, pem or StandardModel(), [Zoeppritz()], [attribute])


def _keys(n: int, angle: float = 10, time: str = "2020-07-01") -> list[DataKey]:
    return [DataKey(time, avo_datatype("avo", angle), i) for i in range(n)]


# --- reading a case ---------------------------------------------------------------------


def test_restarts_are_found_by_date_in_separate_and_unified_files(tmp_path: Path) -> None:
    for unified in (False, True):
        case = _cases.write_case(
            tmp_path / f"u{unified}" / "C", SHAPE, _restarts(), unified=unified, actnum=ACTNUM
        )
        state = CaseState(case)
        assert state.restart_dates == [JAN, JUL]
        pressure, saturations, densities = state.flow_state(JUL, densities=False)
        np.testing.assert_allclose(pressure, 180e5)
        np.testing.assert_allclose(saturations["water"], np.linspace(0.2, 0.8, N_ACTIVE), rtol=1e-6)
        np.testing.assert_allclose(saturations["oil"] + saturations["water"], 1.0)
        assert list(saturations) == ["oil", "water"]
        assert densities == {}


def test_a_missing_restart_stops_the_run_naming_the_written_ones(tmp_path: Path) -> None:
    case = _cases.write_case(
        tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM, summary_end=dt.date(2021, 1, 1)
    )
    state = CaseState(case)
    with pytest.raises(ConfigError, match=r"no restart at 2020-03-01 \(written: 2020-01-01, "):
        state.restart(dt.date(2020, 3, 1), ["PRESSURE"])


def test_field_units_are_converted_to_si(tmp_path: Path) -> None:
    restarts = {JAN: {"PRESSURE": 3000.0, "SWAT": 0.5, "OIL_DEN": 50.0, "WAT_DEN": 62.4}}
    state = CaseState(_cases.write_case(tmp_path / "C", SHAPE, restarts, units=2, top=6000.0))
    pressure, _, densities = state.flow_state(JAN, densities=True)
    np.testing.assert_allclose(pressure, 3000.0 * 6894.757293168361)
    np.testing.assert_allclose(densities["water"], 62.4 * 16.018463373960138, rtol=1e-6)
    np.testing.assert_allclose(state.depth[0], 6002.5 * 0.3048)
    assert state.grid.top[0, 0, 0] == pytest.approx(6000.0 * 0.3048)


def test_what_the_restarts_lack_is_named(tmp_path: Path) -> None:
    case = _cases.write_case(
        tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.5}}, phases=7
    )
    # Every restart holds them: a member's broken file fails only that member.
    with pytest.raises(SimulationError, match="no saturation of oil, gas"):
        CaseState(case).flow_state(JAN, densities=False)
    case = _cases.write_case(tmp_path / "D", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.5}})
    with pytest.raises(ConfigError, match=r"no OIL_DEN, WAT_DEN.*add DEN to the deck's RPTRST"):
        CaseState(case).flow_state(JAN, densities=True)


# --- the simulator --------------------------------------------------------------------


class _Deck:
    """A deck simulator stand-in: its 'summary' is the key's index, its case a folder."""

    accepts_runner = True
    executable = "flow"
    resources: ClassVar[dict[str, int]] = {"cpu": 1}

    def __init__(self, case: Path) -> None:
        self.case = case
        self.read_keys: list[DataKey] = []
        self.cleaned = False

    def simulate(self, member: MemberInput) -> Path:
        return self.case

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        self.read_keys += keys
        return np.array([100.0 + len(k.datatype) for k in keys])

    def clean(self, case: Path) -> None:
        self.cleaned = True

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise AssertionError("the wrapper runs simulate, read and clean")


class _Field:
    def owns(self, key: DataKey) -> bool:
        return key.index is not None

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        assert all(k.index is not None for k in keys)
        return np.array([float(k.index or 0) for k in keys])


def test_the_simulator_splits_the_keys_and_keeps_their_order(tmp_path: Path) -> None:
    deck = _Deck(tmp_path)
    simulator = GeophysicsSimulator(deck, _Field())
    keys = [DataKey("t", "WOPR:P1"), DataKey("t", "avo_10", 4), DataKey("t", "WBHP"),
            DataKey("t", "avo_10", 2)]  # fmt: skip
    values = simulator.run(MemberInput(0, {}, tmp_path), keys)
    np.testing.assert_array_equal(values, [107.0, 4.0, 104.0, 2.0])
    assert deck.read_keys == [keys[0], keys[2]]
    assert deck.cleaned
    assert simulator.accepts_runner
    assert simulator.executable == "flow"
    assert simulator.resources == {"cpu": 1}


def test_a_simulator_without_files_is_refused() -> None:
    class Analytic:
        def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
            return np.zeros(len(keys))

    with pytest.raises(ValueError, match="runs leave files"):
        GeophysicsSimulator(Analytic(), _Field())


# --- the AVO forward model ------------------------------------------------------------


def _expected(
    case: Path, section: GeophysicsSection, columns: list[tuple[int, int]]
) -> NDArray[np.float64]:
    """The traces of ``columns`` from the kernels directly, ``(n_trace, n_angle, n_time)``."""
    state = CaseState(case)
    grid = Grid.read(case.with_suffix(".EGRID"))
    pressure, saturations, _ = state.flow_state(JUL, densities=False)
    props = StandardModel().elastic(
        FlowState(state.static("PORO"), pressure, 101325.0 + 22600.0 * state.depth, saturations)
    )
    active = grid.actnum.ravel(order="F")
    cube = []
    for values in (props.vp, props.vs, props.density):
        full = np.full(active.size, 1.0)
        full[active] = values
        cube.append(full.reshape(SHAPE, order="F"))
    i, j = (np.array(c) for c in zip(*columns, strict=True))
    cols = Columns.of(
        grid.top[i, j, 0], grid.bottom[i, j] - grid.top[i, j], cube[0][i, j], cube[1][i, j],
        cube[2][i, j], grid.actnum[i, j],
    )  # fmt: skip
    model = SeismicModel(
        Zoeppritz(), RickerWavelet(45.0, 0.05), ConstantOverburden(2450.0, 849.0, 2300.0),
        (10.0, 12.5), TimeAxis(1.62, 1.66, 0.002),
    )  # fmt: skip
    return model.traces(cols)


#: Every column has an active cell; F order, I fastest.
COLUMNS = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_maps_are_the_kernels_rms_per_column_in_f_order(tmp_path: Path) -> None:
    section = _section(tmp_path)
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    model = _model(section)
    traces = _expected(case, section, COLUMNS)
    rms = np.sqrt(np.mean(traces**2, axis=2))  # (n_trace, n_angle)
    np.testing.assert_allclose(model.forecast(case, _keys(6)), rms[:, 0], rtol=1e-12)
    np.testing.assert_allclose(model.forecast(case, _keys(6, 12.5)), rms[:, 1], rtol=1e-12)
    assert model.field_sizes() == {"avo_10": 6, "avo_12p5": 6}
    assert rms.min() > 0  # the saturation contrasts reflect


def test_traces_are_trace_major_over_the_window_and_steps_thin_the_columns(
    tmp_path: Path,
) -> None:
    section = _section(tmp_path, form="traces", time_window=[1.63, 1.65], step_x=2)
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    model = _model(section)
    kept = [(0, 0), (2, 0), (0, 1), (2, 1)]  # I = 1 and 3 (0-based 0 and 2)
    traces = _expected(case, section, kept)
    times = TimeAxis(1.62, 1.66, 0.002).times
    window = (times >= 1.63) & (times <= 1.65)
    n = int(window.sum())
    assert model.field_sizes()["avo_10"] == 4 * n
    values = model.forecast(case, _keys(4 * n))
    np.testing.assert_allclose(values, traces[:, 0, window].ravel(), rtol=1e-12)


def test_a_member_with_other_active_columns_is_refused(tmp_path: Path) -> None:
    model = _model(_section(tmp_path))
    column_off = list(ACTNUM)
    column_off[0] = column_off[6] = 0  # column (1, 1) wholly inactive
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(0.5), actnum=column_off)
    with pytest.raises(SimulationError, match=r"1 column\(s\).*\(I, J\) = \(1, 1\)"):
        model.forecast(case, _keys(6))


def test_keys_beyond_the_field_or_off_the_vintages_stop_the_run(tmp_path: Path) -> None:
    model = _model(_section(tmp_path))
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    with pytest.raises(ConfigError, match="has 6 values"):
        model.forecast(case, _keys(7))
    with pytest.raises(ConfigError, match="not a vintage of avo"):
        model.forecast(case, _keys(1, time="2020-01-01"))
    assert not model.owns(DataKey("2020-07-01", "avo_10"))  # a scalar is the deck's
    assert not model.owns(DataKey("2020-07-01", "avo_30", 0))


def test_an_evanescent_wave_fails_the_member(tmp_path: Path) -> None:
    """A slow overburden over a fast reservoir: past the critical angle at 60 degrees."""
    section = _section(
        tmp_path, angles=[60], overburden={"vp": 1500.0, "vs": 400.0, "density": 2000.0}
    )
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    with pytest.raises(SimulationError, match="C at 2020-07-01"):
        _model(section).forecast(case, _keys(6, 60))


def test_an_overburden_file_may_hold_active_or_all_cells(tmp_path: Path) -> None:
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    gradient = _model(_section(tmp_path)).forecast(case, _keys(6))
    depth = CaseState(case).depth
    np.savez(tmp_path / "active.npz", obvalues=(101325.0 + 22600.0 * depth) / 1e5)
    every = np.full(12, 1.0)
    every[np.flatnonzero(ACTNUM)] = 101325.0 + 22600.0 * depth
    np.save(tmp_path / "all.npy", every)
    for file, key, unit in (("active.npz", "obvalues", "bar"), ("all.npy", None, "Pa")):
        pressure = {"file": tmp_path / file, "key": key, "unit": unit}
        section = GeophysicsSection.model_validate(
            {**_section(tmp_path).model_dump(), "overburden_pressure": pressure}
        )
        np.testing.assert_allclose(_model(section).forecast(case, _keys(6)), gradient, rtol=1e-6)
    np.save(tmp_path / "short.npy", np.ones(5))
    section = GeophysicsSection.model_validate(
        {**_section(tmp_path).model_dump(), "overburden_pressure": {"file": tmp_path / "short.npy"}}
    )
    with pytest.raises(ConfigError, match="has 5 values; grid_file has 12 cells, 11 active"):
        _model(section)


def test_soft_sand_needs_the_densities_and_refuses_co2_outside_batzle_wang(
    tmp_path: Path,
) -> None:
    section = _section(tmp_path)
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    with pytest.raises(ConfigError, match="add DEN"):
        _model(section, SoftSandModel()).forecast(case, _keys(6))
    gas = _cases.write_case(
        tmp_path / "G", SHAPE, {JUL: {"PRESSURE": 100.0, "SGAS": 0.1, "GAS_DEN": 600.0,
                                     "WAT_DEN": 1030.0}},
        actnum=ACTNUM, phases=6,
    )  # fmt: skip
    with pytest.raises(ConfigError, match="extrapolate_gas_correlation"):
        _model(section, SoftSandModel(gas_model="batzle_wang")).forecast(gas, _keys(6))
    values = _model(section, SoftSandModel(extrapolate_gas_correlation=True)).forecast(
        gas, _keys(6)
    )
    assert np.all(np.isfinite(values))


# --- review fixes -----------------------------------------------------------------------


class _Releasing:
    """A ResdataFile whose arrays are zeroed once it is released.

    resdata's ``numpy_view()`` points into the file's buffer without keeping it: a view
    used after its file is gone reads freed memory. Here that memory is zeroed at once, so
    such a read gives day 0, month 0 instead of passing by luck.
    """

    def __init__(self, path: str, owner: "_Releasing | None" = None, view: object = None) -> None:
        from resdata.resfile import ResdataFile

        self._file: Any = ResdataFile(path) if view is None else view
        self._owner = owner  # None for the file itself: no cycle, so it is freed at once
        self._arrays: list[NDArray[np.float64]] = []

    def __contains__(self, name: str) -> bool:
        return name in self._file

    def __getitem__(self, name: str) -> list["_Keyword"]:
        return [_Keyword(kw, self._owner or self) for kw in self._file[name]]

    def num_named_kw(self, name: str) -> int:
        return int(self._file.num_named_kw(name))

    def restart_view(self, seqnum_index: int) -> "_Releasing":
        view = self._file.restart_view(seqnum_index=seqnum_index)
        return _Releasing("", owner=self._owner or self, view=view)

    def __del__(self) -> None:
        if self._owner is None:
            for array in self._arrays:
                array[:] = 0


class _Keyword:
    """A keyword whose arrays its file zeroes when released (no class closure: a class
    object lives until the garbage collector runs and would keep the file alive)."""

    def __init__(self, kw: Any, owner: _Releasing) -> None:  # noqa: ANN401  (a resdata keyword)
        self.kw, self.owner = kw, owner

    def numpy_view(self) -> NDArray[np.float64]:
        array = np.array(self.kw.numpy_view(), dtype=np.float64)
        self.owner._arrays.append(array)
        return array


def test_restart_headers_are_copied_not_viewed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression: a header read through numpy_view() of a file already released read
    freed memory, so dates came out as garbage once the buffer was reused."""
    dates = [dt.date(2020, m, 1) for m in range(1, 7)]
    restarts = {d: {"PRESSURE": 200.0 + d.month, "SWAT": 0.5} for d in dates}
    for unified in (False, True):
        case = _cases.write_case(
            tmp_path / f"u{unified}" / "C", SHAPE, restarts, actnum=ACTNUM, unified=unified
        )
        state = CaseState(case)
        monkeypatch.setattr(state, "_file", _Releasing)
        assert state.restart_dates == dates
        assert state.restart(dates[2], ["PRESSURE"])["PRESSURE"][0] == 203.0


def test_the_later_of_two_restarts_on_one_day_is_read(tmp_path: Path) -> None:
    """Two report steps on one day: the summary takes the later one, and so do restarts."""
    from resdata.resfile import FortIO, ResdataFile, openFortIO

    case = _cases.write_case(
        tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.5}}, actnum=ACTNUM,
        unified=True,
    )  # fmt: skip
    first = ResdataFile(str(case.with_suffix(".UNRST")))
    blocks = [first.iget_kw(n) for n in range(len(first))]
    later = [kw.copy() for kw in blocks]
    later[0][0] = 1
    later[2][0] = 300.0  # PRESSURE's first cell
    with openFortIO(str(case.with_suffix(".UNRST")), FortIO.WRITE_MODE) as f:
        for kw in [*blocks, *later]:
            kw.fwrite(f)
    pressure = CaseState(case).restart(JAN, ["PRESSURE"])["PRESSURE"]
    assert pressure[0] == 300.0


def test_a_run_that_ended_before_a_vintage_fails_only_its_member(tmp_path: Path) -> None:
    """Judged by the summary's end: restarts written yearly end before the last report time
    in every member, which is the deck's fault, not one member's."""
    early = _cases.write_case(tmp_path / "E", SHAPE, _restarts(), actnum=ACTNUM, summary_end=JUL)
    with pytest.raises(SimulationError, match="the run ended on 2020-07-01"):
        CaseState(early).restart(dt.date(2021, 1, 1), ["PRESSURE"])
    sparse = _cases.write_case(
        tmp_path / "S", SHAPE, _restarts(), actnum=ACTNUM, summary_end=dt.date(2022, 1, 1)
    )
    with pytest.raises(ConfigError, match="RPTRST must write one"):
        CaseState(sparse).restart(dt.date(2021, 1, 1), ["PRESSURE"])
    # No summary tells how far the run went: this member's files, not the deck.
    broken = _cases.write_case(tmp_path / "N", SHAPE, _restarts(), actnum=ACTNUM)
    with pytest.raises(SimulationError, match="no summary tells how far it ran"):
        CaseState(broken).restart(dt.date(2021, 1, 1), ["PRESSURE"])


def test_a_result_file_without_a_header_fails_the_member(tmp_path: Path) -> None:
    from resdata import ResDataType
    from resdata.resfile import FortIO, ResdataKW, openFortIO

    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    with openFortIO(str(tmp_path / "C.X0001"), FortIO.WRITE_MODE) as f:
        ResdataKW("PRESSURE", N_ACTIVE, ResDataType.RD_FLOAT).fwrite(f)
    with pytest.raises(SimulationError, match="lacks INTEHEAD"):
        _ = CaseState(case).restart_dates


def test_collapsed_inactive_cells_take_no_time_and_active_ones_are_refused(
    tmp_path: Path,
) -> None:
    """Corner-point grids may cross inactive cells (base above top); an active one is not
    physical: refused in grid_file before the run, and in a member as its failure."""
    layer_off = [1] * 6 + [0] * 6  # layer 2 inactive and crossed
    thickness = [5.0, -1.0]
    _cases.write_grid(tmp_path / "REF.EGRID", SHAPE, actnum=layer_off, thickness=thickness)
    section = _section(tmp_path)  # writes REF.EGRID again: ACTNUM's layer 2 is active there
    grid_file = section.grid_file
    _cases.write_grid(grid_file, SHAPE, actnum=layer_off, thickness=thickness)
    case = _cases.write_case(
        tmp_path / "C", SHAPE, {JUL: {"PRESSURE": 180.0, "SWAT": 0.4}}, actnum=layer_off
    )
    _cases.write_grid(case.with_suffix(".EGRID"), SHAPE, actnum=layer_off, thickness=thickness)
    assert np.all(np.isfinite(_model(section).forecast(case, _keys(6))))

    _cases.write_grid(grid_file, SHAPE, thickness=thickness)  # layer 2 active
    with pytest.raises(
        ConfigError, match=r"base above their top, e\.g\. \(I, J, K\) = \(1, 1, 2\)"
    ):
        _model(section)

    _cases.write_grid(grid_file, SHAPE, actnum=layer_off)
    member = _cases.write_case(tmp_path / "M", SHAPE, {JUL: {"PRESSURE": 180.0, "SWAT": 0.4}})
    _cases.write_grid(member.with_suffix(".EGRID"), SHAPE, thickness=thickness)
    with pytest.raises(SimulationError, match="thickness must be finite and non-negative"):
        _model(section).forecast(member, _keys(6))


def test_an_active_cells_overburden_file_follows_grid_file(tmp_path: Path) -> None:
    """A file over grid_file's active cells is laid on the grid once; a member with a cell
    grid_file has inactive has no pressure there and fails, not the run."""
    section = _section(tmp_path)
    np.save(tmp_path / "ob.npy", np.full(N_ACTIVE, 40e6))
    filed = GeophysicsSection.model_validate(
        {**section.model_dump(), "overburden_pressure": {"file": tmp_path / "ob.npy"}}
    )
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(0.5), actnum=ACTNUM)
    assert np.all(np.isfinite(_model(filed).forecast(case, _keys(6))))
    every = _cases.write_case(tmp_path / "E", SHAPE, _restarts(0.5))  # cell 6 active too
    with pytest.raises(SimulationError, match="1 active cell"):
        _model(filed).forecast(every, _keys(6))
    np.save(tmp_path / "ob.npy", np.full(7, 40e6))
    with pytest.raises(ConfigError, match="has 7 values; grid_file has 12 cells, 11 active"):
        _model(filed)


def test_observed_fields_are_checked_against_the_configuration(tmp_path: Path) -> None:
    model = _model(_section(tmp_path))
    model.check_observations([DataKey("2020-01-01", "WOPR:P1"), *_keys(6), *_keys(6, 12.5)])
    with pytest.raises(ConfigError, match=r"AVO_10@2020-07-01 is not a field .*avo_10, avo_12p5"):
        model.check_observations([DataKey("2020-07-01", "AVO_10", i) for i in range(6)])
    with pytest.raises(ConfigError, match="not a vintage of avo"):
        model.check_observations(_keys(6, time="2021-01-01"))
    with pytest.raises(ConfigError, match="avo_10@2020-07-01 has 5 values; the field has 6"):
        model.check_observations(_keys(5))
    repeated = [*_keys(5), DataKey("2020-07-01", "avo_10", 7)]
    with pytest.raises(ConfigError, match="indices 0 to 5"):
        model.check_observations(repeated)


def test_saturations_within_round_off_of_0_and_1_are_clipped(tmp_path: Path) -> None:
    """A regression: OPM Flow wrote SWAT = -4.8e-5 in a 3dBox member, which the
    petro-elastic model refused, failing the member. Round-off is clipped; a value well
    outside [0, 1] is left for the model to refuse."""
    swat = np.full(N_ACTIVE, 0.4)
    swat[:3] = (-4.8e-5, 1.0004, -0.05)
    state = CaseState(
        _cases.write_case(tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": swat}},
                          actnum=ACTNUM)
    )  # fmt: skip
    _, saturations, _ = state.flow_state(JAN, densities=False)
    np.testing.assert_allclose(saturations["water"][:3], (0.0, 1.0, -0.05), atol=1e-7)
    np.testing.assert_allclose(saturations["oil"][:3], (1.0, 0.0, 1.05), atol=1e-7)
    both = CaseState(
        _cases.write_case(tmp_path / "B", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.4002,
                                                         "SOIL": 0.6}}, actnum=ACTNUM)
    )  # fmt: skip
    _, saturations, _ = both.flow_state(JAN, densities=False)
    np.testing.assert_allclose(saturations["water"] + saturations["oil"], 1.0)


def test_a_clipped_derived_phase_leaves_saturations_that_sum_to_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Three phases, oil derived: SWAT 0.7 and SGAS 0.3002 give SOIL -2e-4, clipped to 0;
    the sum, 1.0002, is normalized, so the petro-elastic model's 1e-5 check holds."""
    state = CaseState(
        _cases.write_case(
            tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 200.0, "SWAT": 0.7, "SGAS": 0.3002}},
            actnum=ACTNUM, phases=7,
        )
    )  # fmt: skip
    with caplog.at_level("DEBUG", logger="apex.simulation.geophysics"):
        _, saturations, _ = state.flow_state(JAN, densities=False)
    total = saturations["oil"] + saturations["water"] + saturations["gas"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert np.all(saturations["oil"] == 0.0)
    record = next(r for r in caplog.records if r.message == "geophysics_saturations_settled")
    assert record.levelname == "WARNING"  # moved 2e-4, above the 1e-4 drift
    assert record.__dict__["cells"] == N_ACTIVE


def test_an_overshoot_within_the_slack_is_clipped_and_beyond_it_refused(tmp_path: Path) -> None:
    """The 3dBox GIES runs: OPM Flow's derived oil saturation reached 1.009 in one cell
    (SWAT -0.009), and the petro-elastic model refused 9 of 100 members; with porosity
    uncertain, 1.021. Within 3e-2 it is clipped and the phases sum to one; a real violation
    (1.035, just past it) is still left to refuse."""
    from apex.simulation.geophysics import SATURATION_SLACK

    near = CaseState(_cases.write_case(tmp_path / "N", SHAPE, {JAN: {
        "PRESSURE": 200.0, "SWAT": -0.021}}, actnum=ACTNUM))  # fmt: skip
    _, saturations, _ = near.flow_state(JAN, densities=False)
    assert np.all(saturations["oil"] == 1.0)
    assert np.all(saturations["water"] == 0.0)
    far = CaseState(_cases.write_case(tmp_path / "F", SHAPE, {JAN: {
        "PRESSURE": 200.0, "SWAT": -0.035}}, actnum=ACTNUM))  # fmt: skip
    _, beyond, _ = far.flow_state(JAN, densities=False)
    assert np.all(beyond["oil"] > 1.0 + SATURATION_SLACK)  # left as is, for the model to refuse


def test_a_three_phase_overshoot_within_the_slack_sums_to_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The review of the slack: SWAT 1.015 and SGAS 0.005 give a derived oil of -0.02;
    clipped, the sum (1.02) is normalized for the petro-elastic model's 1e-5 sum check,
    and the change is a warning."""
    state = CaseState(_cases.write_case(tmp_path / "C", SHAPE, {JAN: {
        "PRESSURE": 200.0, "SWAT": 1.015, "SGAS": 0.005}}, actnum=ACTNUM, phases=7))  # fmt: skip
    with caplog.at_level("DEBUG", logger="apex.simulation.geophysics"):
        _, saturations, _ = state.flow_state(JAN, densities=False)
    total = saturations["oil"] + saturations["water"] + saturations["gas"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert np.all(saturations["oil"] == 0.0)
    record = next(r for r in caplog.records if r.message == "geophysics_saturations_settled")
    assert record.levelname == "WARNING"


# --- the region of interest ------------------------------------------------


def _columns(model: GeophysicsModel) -> list[tuple[int, int]]:
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    return [(int(i), int(j)) for i, j in survey.columns]


def test_a_region_file_keeps_its_columns(tmp_path: Path) -> None:
    region = np.ones((3, 2))
    region[0, 0] = region[2, 1] = 0
    np.save(tmp_path / "region.npy", region)
    model = _model(_section(tmp_path, region={"file": str(tmp_path / "region.npy")}))
    assert _columns(model) == [(1, 0), (2, 0), (0, 1), (1, 1)]
    assert model.field_sizes()["avo_10"] == 4

    np.save(tmp_path / "region.npy", np.ones((2, 3)))
    with pytest.raises(ConfigError, match=r"shape \(n_x, n_y\) = \(3, 2\)"):
        _model(_section(tmp_path, region={"file": str(tmp_path / "region.npy")}))


def test_a_region_around_wells_is_the_references_ellipse_or_rectangle(tmp_path: Path) -> None:
    # 100 m cells: column centres x = 50, 150, 250 and y = 50, 150.
    one = {"P1": [(1, 1)]}
    section = _section(tmp_path, region={"shape": "ellipse", "margin": 100.0})
    model = build_geophysics_model(
        section, StandardModel(), [Zoeppritz()], [avo_attributes.create("rms")], one
    )
    # A disc of radius 100 m around (50, 50): the boundary (150, 50) and (50, 150) count.
    assert _columns(model) == [(0, 0), (1, 0), (0, 1)]

    two = {"P1": [(1, 1)], "I1": [(2, 1), (2, 1)]}
    rectangle = _section(tmp_path, region={"shape": "rectangle", "margin": 0.0})
    model = build_geophysics_model(
        rectangle, StandardModel(), [Zoeppritz()], [avo_attributes.create("rms")], two
    )
    assert _columns(model) == [(0, 0), (1, 0)]

    ranged = _section(
        tmp_path, region={"shape": "rectangle", "margin": 0.0, "x_range": [100.0, 300.0]}
    )
    model = build_geophysics_model(
        ranged, StandardModel(), [Zoeppritz()], [avo_attributes.create("rms")], two
    )
    assert _columns(model) == [(1, 0), (2, 0)]

    only = _section(tmp_path, region={"shape": "ellipse", "margin": 100.0, "wells": ["I1"]})
    model = build_geophysics_model(
        only, StandardModel(), [Zoeppritz()], [avo_attributes.create("rms")], two
    )
    assert _columns(model) == [(0, 0), (1, 0), (2, 0), (1, 1)]


@pytest.mark.parametrize(
    ("completions", "region", "message"),
    [
        (None, {"shape": "ellipse"}, "needs the deck's completions"),
        ({"P1": [(1, 1)]}, {"shape": "ellipse", "wells": ["X"]}, r"wells \['X'\]"),
        ({}, {"shape": "ellipse"}, "no well completions"),
        ({"P1": [(4, 1)]}, {"shape": "ellipse"}, "outside the grid"),
        ({"P1": [(1, 1)]}, {"shape": "ellipse", "margin": 0.5}, None),
    ],
)
def test_regions_around_wells_need_known_wells_on_the_grid(
    tmp_path: Path,
    completions: dict[str, list[tuple[int, int]]] | None,
    region: dict[str, object],
    message: str | None,
) -> None:
    section = _section(tmp_path, region=region)

    def build() -> GeophysicsModel:
        return build_geophysics_model(
            section, StandardModel(), [Zoeppritz()], [avo_attributes.create("rms")], completions
        )

    if message is None:
        # A semi-axis of at least 1 m: one completion keeps its own column.
        assert _columns(build()) == [(0, 0)]
        return
    with pytest.raises(ConfigError, match=message):
        build()


# --- time-lapse differences ------------------------------------------------


def test_differences_are_each_vintage_minus_the_baseline(tmp_path: Path) -> None:
    case = _cases.write_case(tmp_path / "C", SHAPE, _restarts(), actnum=ACTNUM)
    plain = _section(tmp_path, form="traces")
    both = plain.model_copy(
        update={
            "data": [plain.data[0].model_copy(update={"vintages": ["2020-01-01", "2020-07-01"]})]
        }
    )
    n = _model(both).field_sizes()["avo_10"]
    vintages = _model(both)
    after = vintages.forecast(case, _keys(n))
    before = vintages.forecast(case, _keys(n, time="2020-01-01"))
    assert np.abs(after - before).max() > 0  # the saturations changed

    differences = _model(
        _section(tmp_path, form="traces", four_d="differences", baseline="2020-01-01")
    )
    np.testing.assert_allclose(differences.forecast(case, _keys(n)), after - before, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"four_d": "differences"}, "needs a baseline"),
        ({"baseline": "2020-01-01"}, "for four_d = 'differences'"),
        ({"four_d": "differences", "baseline": "1 JAN 2020"}, "not an ISO date"),
        ({"four_d": "differences", "baseline": "2020-07-01"}, "not after the baseline"),
    ],
)
def test_a_difference_needs_a_baseline_before_its_vintages(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _section(tmp_path, **options)


class _ReleasingKeyword:
    """A resdata keyword whose buffer is blanked when it is released, as freed memory is
    reused: a view taken from a temporary keyword reads NaN instead of passing by luck."""

    def __init__(self, kw: Any) -> None:  # noqa: ANN401  (a resdata keyword)
        self.array = np.array(kw.numpy_view(), dtype=np.float64)

    def numpy_view(self) -> NDArray[np.float64]:
        return self.array

    def __del__(self) -> None:
        self.array[:] = np.nan


def test_column_centres_copy_the_pillars_while_their_keyword_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regression: COORD read through numpy_view() of a temporary keyword gave garbage
    centres (9e18 m), so a region of interest kept no column, or the wrong ones."""
    from resdata.grid import Grid as ResGrid

    from apex.simulation.geophysics import column_centres
    from apex.storage.pet_geophysics import _read_grid

    grid = _cases.write_grid(tmp_path / "REF.EGRID", SHAPE, actnum=ACTNUM)
    export = ResGrid.export_coord
    monkeypatch.setattr(ResGrid, "export_coord", lambda self: _ReleasingKeyword(export(self)))
    expected = np.stack(np.meshgrid([50.0, 150.0, 250.0], [50.0, 150.0], indexing="ij"), axis=-1)
    np.testing.assert_array_equal(column_centres(grid), expected)
    np.testing.assert_array_equal(_read_grid(grid)[1], expected)


def test_column_centres_of_a_feet_grid_are_in_metres(tmp_path: Path) -> None:
    from resdata import UnitSystem
    from resdata.grid import Grid as ResGrid

    from apex.simulation.geophysics import column_centres

    metric = _cases.write_grid(tmp_path / "M.EGRID", SHAPE, actnum=ACTNUM)
    ResGrid(str(metric)).save_EGRID(str(tmp_path / "F.EGRID"), UnitSystem.FIELD)
    # The FIELD file holds feet (float32): the same centres within its rounding.
    np.testing.assert_allclose(
        column_centres(tmp_path / "F.EGRID"), column_centres(metric), rtol=1e-6
    )


def test_co2stores_oil_gas_form_reads_its_oil_as_brine(tmp_path: Path) -> None:
    """OPM Flow's CO2STORE in its oil-gas form writes the brine as oil (SOIL derived,
    OIL_DEN): read as water, its AVO data are the gas-water form's."""
    section = _section(tmp_path)
    sg = np.linspace(0.0, 0.6, N_ACTIVE)
    gas_water = _cases.write_case(
        tmp_path / "GW" / "C", SHAPE,
        {JUL: {"PRESSURE": 150.0, "SGAS": sg, "GAS_DEN": 650.0, "WAT_DEN": 1030.0}},
        actnum=ACTNUM, phases=6,
    )  # fmt: skip
    oil_gas = _cases.write_case(
        tmp_path / "OG" / "C", SHAPE,
        {JUL: {"PRESSURE": 150.0, "SGAS": sg, "GAS_DEN": 650.0, "OIL_DEN": 1030.0}},
        actnum=ACTNUM, phases=5,
    )  # fmt: skip
    pem = SoftSandModel(gas_model="batzle_wang", extrapolate_gas_correlation=True)
    d = section.data[0].avo()
    attribute = avo_attributes.create("trace" if d.form == "traces" else d.attribute)

    def forecast(case: Path, brine_as_oil: bool) -> NDArray[np.float64]:
        model = build_geophysics_model(
            section, pem, [Zoeppritz()], [attribute], brine_as_oil=brine_as_oil
        )
        return model.forecast(case, _keys(6))

    expected = forecast(gas_water, False)
    np.testing.assert_array_equal(forecast(oil_gas, True), expected)
    # Read as oil, the brine goes through the dead-oil fit: other values.
    assert not np.allclose(forecast(oil_gas, False), expected)
    # A templated RUNSPEC cannot tell: an oil-gas run with CO₂ is refused, not misread.
    unknown = build_geophysics_model(section, pem, [Zoeppritz()], [attribute], brine_as_oil=None)
    with pytest.raises(ConfigError, match="does not say whether it is CO2STORE"):
        unknown.forecast(oil_gas, _keys(6))
    assert np.array_equal(unknown.forecast(gas_water, _keys(6)), expected)


# --- uncertain petro-elastic parameters -------------------------


def test_a_member_sets_its_petro_elastic_parameter_and_the_deck_never_sees_it(
    tmp_path: Path,
) -> None:
    from apex.workflows.geophysics import PemParameter

    section = _section(tmp_path)
    restarts = {JUL: {"PRESSURE": 180.0, "SWAT": np.linspace(0.2, 0.8, N_ACTIVE),
                      "OIL_DEN": 800.0, "WAT_DEN": 1030.0}}  # fmt: skip
    case = _cases.write_case(tmp_path / "C", SHAPE, restarts, actnum=ACTNUM)
    pem = SoftSandModel(gas_model="batzle_wang", mixing="patchiness")
    d = section.data[0].avo()
    attribute = avo_attributes.create("trace" if d.form == "traces" else d.attribute)
    q = PemParameter("patchiness", "pem_q", "logit", value=0.25)
    model = build_geophysics_model(section, pem, [Zoeppritz()], [attribute], parameters=[q])
    assert model.state_variables == ("pem_q",)
    assert model.for_state({"pem_q": [0.0]}).pem.patchiness == 0.5  # type: ignore[union-attr]
    assert model.for_state({}).pem.patchiness == 0.25  # type: ignore[union-attr]
    assert model.pem is pem  # the configured model is left as it is
    with pytest.raises(SimulationError, match="2 values, not 1"):
        model.for_state({"pem_q": [0.0, 1.0]})
    no_fallback = build_geophysics_model(
        section, pem, [Zoeppritz()], [attribute],
        parameters=[PemParameter("patchiness", "pem_q", "logit")],
    )  # fmt: skip
    with pytest.raises(SimulationError, match="no state variable 'pem_q'"):
        no_fallback.for_state({})

    deck = _Deck(case)
    simulator = GeophysicsSimulator(deck, model)
    seen: list[set[str]] = []
    original = deck.simulate

    def simulate(member: MemberInput) -> Path:
        seen.append(set(member.state))
        return original(member)

    deck.simulate = simulate  # type: ignore[method-assign]
    keys = _keys(6)
    low = simulator.run(MemberInput(0, {"perm": np.ones(3), "pem_q": np.array([-4.0])}), keys)
    high = simulator.run(MemberInput(1, {"perm": np.ones(3), "pem_q": np.array([4.0])}), keys)
    assert seen == [{"perm"}, {"perm"}]  # stripped before the deck
    assert not np.allclose(low, high)  # q moves the data
    np.testing.assert_array_equal(high, model.for_state({"pem_q": [4.0]}).forecast(case, keys))


@pytest.mark.parametrize(
    ("transform", "x", "expected"),
    [("logit", 0.0, 0.5), ("logit", 2.0, 1 / (1 + np.exp(-2.0))), ("log", 1.0, np.e),
     ("none", 0.3, 0.3)],
)  # fmt: skip
def test_parameter_transforms(transform: str, x: float, expected: float) -> None:
    from apex.workflows.geophysics import PemParameter

    assert PemParameter("o", "s", transform).apply(x) == pytest.approx(expected)


def test_a_state_value_without_a_finite_option_fails_the_member(tmp_path: Path) -> None:
    from apex.workflows.geophysics import PemParameter

    section = _section(tmp_path)
    pem = SoftSandModel(gas_model="batzle_wang")
    d = section.data[0].avo()
    attribute = avo_attributes.create("trace" if d.form == "traces" else d.attribute)
    cn = PemParameter("coordination_number", "pem_cn", "log")
    model = build_geophysics_model(section, pem, [Zoeppritz()], [attribute], parameters=[cn])
    assert model.for_state({"pem_cn": [np.log(7.0)]}).pem.coordination_number == pytest.approx(7.0)  # type: ignore[union-attr]
    with pytest.raises(SimulationError, match="gives pem coordination_number = inf"):
        model.for_state({"pem_cn": [1000.0]})
    q = PemParameter("patchiness", "pem_q", "logit")
    assert q.apply(-1000.0) == 0.0  # no overflow


def test_a_failed_read_leaves_no_model_behind(tmp_path: Path) -> None:
    from apex.workflows.geophysics import PemParameter

    section = _section(tmp_path)
    case = _cases.write_case(tmp_path / "C", SHAPE, {JAN: {"PRESSURE": 180.0, "SWAT": 0.5}})
    pem = SoftSandModel(gas_model="batzle_wang", mixing="patchiness")
    d = section.data[0].avo()
    attribute = avo_attributes.create("trace" if d.form == "traces" else d.attribute)
    q = PemParameter("patchiness", "pem_q", "logit")
    model = build_geophysics_model(section, pem, [Zoeppritz()], [attribute], parameters=[q])

    class Releasing(_Deck):
        released: list[Path]

        def release(self, case: Path) -> None:
            self.released = [*getattr(self, "released", []), case]

    deck = Releasing(case)
    simulator = GeophysicsSimulator(deck, model)
    with pytest.raises(SimulationError, match="no restart at 2020-07-01"):
        simulator.run(MemberInput(0, {"pem_q": np.array([0.0])}), _keys(6))
    assert simulator._models == {}
    # A failed member's files stay for diagnosis (the coder's review of phase 2b): the
    # deck is released, not cleaned.
    assert not deck.cleaned
    assert deck.released == [case]


def test_time_shift_maps_are_the_change_of_the_base_time(tmp_path: Path) -> None:
    """Per trace, the two-way time to the column's base at the vintage
    minus at the baseline, the same at every angle; by hand, Σ 2 h / Vp over its cells."""
    section = _section(tmp_path, attribute="time_shift", baseline="2020-01-01", angles=[0])
    model = _model(section)
    restarts = {JAN: {"PRESSURE": 200.0, "SWAT": 0.3},
                JUL: {"PRESSURE": 180.0, "SWAT": np.linspace(0.2, 0.8, N_ACTIVE)}}  # fmt: skip
    case = _cases.write_case(tmp_path / "C", SHAPE, restarts, actnum=ACTNUM)
    n = model.field_sizes()["avo_0"]
    shift = model.forecast(case, _keys(n, angle=0))
    state = CaseState(case)
    grid = state.grid
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)

    def base_times(date: dt.date) -> NDArray[np.float64]:
        vp, _, _ = model._elastic(state, grid, date)
        out = []
        for i, j in survey.columns:
            cells = grid.actnum[i, j, :]
            h = (grid.bottom[i, j, :] - grid.top[i, j, :])[cells]
            out.append(np.sum(2 * h / vp[i, j, cells]))
        return np.array(out)

    # Only the cells change between the dates: the overburden's time cancels. (Column
    # (3, 2) has an inactive cell, filled with the overburden at both dates.)
    np.testing.assert_allclose(shift, base_times(JUL) - base_times(JAN), rtol=1e-10)
    assert np.any(shift != 0)


def test_a_time_shift_needs_a_baseline_and_one_angle(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="time_shift needs a baseline"):
        _section(tmp_path, attribute="time_shift", angles=[0])
    # The same at every angle: copies would count as independent data (review, step 5).
    with pytest.raises(ValidationError, match="time_shift takes one angle"):
        _section(tmp_path, attribute="time_shift", baseline="2020-01-01")
    section = _section(tmp_path, attribute="time_shift", baseline="2020-01-01", angles=[0])
    assert section.data[0].avo().four_d == "vintages"
    model = _model(section)
    assert not any("Nyquist" in w for w in model.warnings())  # no samples to alias


@pytest.mark.parametrize("attribute", ["rms", "time_shift"])
def test_a_rolling_baseline_takes_each_vintage_from_the_one_before(
    tmp_path: Path, attribute: str
) -> None:
    """SimulatorWrap's rolling mode: the first vintage minus the baseline,
    each later one minus the vintage before it; so rolling = fixed differenced in time."""
    oct_ = dt.date(2020, 10, 1)
    restarts = {**_restarts(), oct_: {"PRESSURE": 170.0, "SWAT": np.linspace(0.3, 0.9, N_ACTIVE)}}
    case = _cases.write_case(tmp_path / "C", SHAPE, restarts, actnum=ACTNUM)
    angles = [0] if attribute == "time_shift" else [10, 12.5]
    forecasts = {}
    for mode in ("fixed", "rolling"):
        section = _section(tmp_path, angles=angles, attribute=attribute, baseline="2020-01-01",
                           four_d="differences", baseline_mode=mode)  # fmt: skip
        data = section.data[0].model_copy(update={"vintages": ["2020-07-01", "2020-10-01"]})
        model = _model(section.model_copy(update={"data": [data]}))
        n = model.field_sizes()[avo_datatype("avo", angles[0])]
        forecasts[mode] = {
            t: model.forecast(case, _keys(n, angle=angles[0], time=t))
            for t in ("2020-07-01", "2020-10-01")
        }
    fixed, rolling = forecasts["fixed"], forecasts["rolling"]
    np.testing.assert_array_equal(rolling["2020-07-01"], fixed["2020-07-01"])
    np.testing.assert_allclose(
        rolling["2020-10-01"], fixed["2020-10-01"] - fixed["2020-07-01"], rtol=1e-12, atol=1e-15
    )
    assert np.abs(rolling["2020-10-01"]).max() > 0


def test_a_rolling_baseline_needs_increasing_vintages(tmp_path: Path) -> None:
    section = _section(tmp_path, baseline="2020-01-01", four_d="differences",
                       baseline_mode="rolling")  # fmt: skip
    raw = section.data[0].model_dump()
    raw["vintages"] = ["2020-10-01", "2020-07-01"]
    with pytest.raises(ValueError, match="increasing order"):
        type(section.data[0]).model_validate(raw)
    raw["vintages"] = ["2020-07-01"]
    raw["options"] = {**raw["options"], "four_d": "vintages", "baseline": None}
    with pytest.raises(ValueError, match="rolling' takes differences"):
        type(section.data[0]).model_validate(raw)


def test_a_model_taking_state_without_for_state_is_refused(tmp_path: Path) -> None:
    """Its variables would reach the deck (the review of phase 4, step A)."""

    class Half(_Field):
        state_variables = ("pem_q",)

    with pytest.raises(ValueError, match="cannot give a member's model"):
        GeophysicsSimulator(_Deck(tmp_path), Half())


def _lattice_section(tmp_path: Path, **options: object) -> GeophysicsSection:
    """A 7 x 5 x 2 grid whose column (2, 0) is inactive (a hole in the lattice)."""
    shape = (7, 5, 2)
    actnum = np.ones(shape, dtype=int)
    actnum[2, 0, :] = 0
    grid = _cases.write_grid(tmp_path / "L.EGRID", shape, actnum=actnum.ravel(order="F"))
    section = _section(tmp_path, **options)
    return section.model_copy(update={"grid_file": grid})


@pytest.mark.parametrize(("step_x", "step_y"), [(1, 1), (2, 1), (2, 2), (3, 2)])
def test_the_wavelet_layout_is_the_maps_own_order(tmp_path: Path, step_x: int, step_y: int) -> None:
    """The masked lattice, cropped to the kept columns, lists a map's
    values in F order as the map does (Eclipse order of the kept columns), with holes
    (an inactive column) and NX, NY not multiples of the steps."""
    model = _model(_lattice_section(tmp_path, step_x=step_x, step_y=step_y))
    name = avo_datatype("avo", 10)
    grid, mask = model.wavelet_layout(name)
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    columns = survey.columns
    assert grid == (-(-7 // step_x), -(-5 // step_y))  # every lattice row has a kept column
    assert int(mask.sum()) == model.field_sizes()[name] == len(columns)
    assert (2 % step_x == 0) == (not mask.all())  # the hole (i = 2) is on the lattice
    values = columns[:, 0] * 1000.0 + columns[:, 1]  # each value names its column
    flat = np.zeros(int(np.prod(grid)))
    flat[np.flatnonzero(mask.ravel(order="F"))] = values
    cube = flat.reshape(grid, order="F")
    for i, j in columns:
        assert cube[i // step_x, j // step_y] == i * 1000.0 + j


def test_the_traces_wavelet_layout_is_their_own_order(tmp_path: Path) -> None:
    """Traces: (n_sample, lattice_x, lattice_y) in F order is trace by trace (Eclipse
    order), each trace's samples in turn: one 3-D transform per angle, as the user's
    wavelet3D_sparse_representation (MATLAB's wavedec3 on the cube)."""
    model = _model(_lattice_section(tmp_path, form="traces", step_x=2, step_y=2))
    name = avo_datatype("avo", 10)
    grid, mask = model.wavelet_layout(name)
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    n_sample = int(survey.window.sum())
    assert grid == (n_sample, 4, 3)
    assert int(mask.sum()) == model.field_sizes()[name]
    columns = survey.columns
    values = (columns[:, 0, None] * 1000.0 + columns[:, 1, None]) * 100 + np.arange(n_sample)
    flat = np.zeros(int(np.prod(grid)))
    flat[np.flatnonzero(mask.ravel(order="F"))] = values.ravel()
    cube = flat.reshape(grid, order="F")
    for i, j in columns:
        expected = (i * 1000.0 + j) * 100 + np.arange(n_sample)
        np.testing.assert_array_equal(cube[:, i // 2, j // 2], expected)
    with pytest.raises(ConfigError, match="not a field"):
        model.wavelet_layout("nope")
