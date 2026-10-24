"""Gravity and displacement data of a deck run.

The case files are written with resdata (``tests/conftest.py``); no simulator runs. The
values are checked against the kernels called on hand-computed cell centres and changes,
and the gravity of one cell against Newton's law, so these tests pin the reading, the
quantities, the active order and the units; the kernels have their own reference tests.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import (
    Geertsma,
    point_mass_gravity,
    regular_stations,
)
from apex.core.errors import ConfigError, SimulationError
from apex.domain.config import GeophysicsSection
from apex.domain.observations import DataKey
from apex.simulation.geophysics import CaseState, Grid
from apex.workflows.geophysics import (
    CaseQuantities,
    DisplacementSurvey,
    GeophysicsModel,
    GravitySurvey,
    build_geophysics_model,
)

SHAPE = (3, 2, 2)
CELL = (100.0, 100.0, 5.0)
TOP = 2000.0
JAN, JUL = dt.date(2020, 1, 1), dt.date(2020, 7, 1)
#: Cell 5 (I=3, J=2, K=1) inactive.
ACTNUM = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
N_ACTIVE = 11
G = 6.67430e-11


def _centres() -> NDArray[np.float64]:
    """``(n_active, 3)``: the box grid's active cell centres, in F order."""
    i, j, k = np.unravel_index(np.arange(12), SHAPE, order="F")
    centres = np.column_stack(((i + 0.5) * CELL[0], (j + 0.5) * CELL[1], TOP + (k + 0.5) * CELL[2]))
    out: NDArray[np.float64] = centres[np.asarray(ACTNUM, dtype=bool)]
    return out


def _restarts() -> dict[dt.date, dict[str, Any]]:
    sw = np.linspace(0.2, 0.8, N_ACTIVE)
    return {
        JAN: {"PRESSURE": 200.0, "SWAT": 0.3, "OIL_DEN": 800.0, "WAT_DEN": 1000.0,
              "RPORV": 12000.0},
        JUL: {"PRESSURE": np.linspace(170.0, 190.0, N_ACTIVE), "SWAT": sw,
              "OIL_DEN": 790.0, "WAT_DEN": 1010.0, "RPORV": np.linspace(11800, 11900, N_ACTIVE)},
    }  # fmt: skip


def _model(
    tmp_path: Path,
    write_grid: Callable[..., Path],
    kind: str,
    **options: Any,  # noqa: ANN401
) -> GeophysicsModel:
    tmp_path.mkdir(parents=True, exist_ok=True)  # resdata crashes saving into no folder
    grid = write_grid(tmp_path / "REF.EGRID", SHAPE, CELL, TOP, actnum=ACTNUM)
    np.save(tmp_path / "stations.npy", STATIONS)
    stations = options.pop("stations", {"file": str(tmp_path / "stations.npy")})
    section = GeophysicsSection.model_validate(
        {
            "grid_file": grid,
            "data": [
                {
                    "kind": kind,
                    "datatype": "obs",
                    "vintages": ["2020-07-01"],
                    "options": {"baseline": "2020-01-01", "stations": stations, **options},
                }
            ],
        }
    )
    kernel = Geertsma() if kind == "seafloor_displacement" else None
    return build_geophysics_model(section, None, [None], [None], None, [kernel])


#: Stations at the sea floor (300 m), one above cell (1, 1, 1) and others around.
STATIONS = np.array(
    [[50.0, 50.0, 300.0], [150.0, 100.0, 300.0], [-400.0, 250.0, 280.0], [900.0, -100.0, 0.0]]
)


def _keys() -> list[DataKey]:
    return [DataKey("2020-07-01", "obs", i) for i in range(len(STATIONS))]


def test_the_grid_gives_centres_boxes_and_volumes(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    grid = Grid.read(write_grid(tmp_path / "G.EGRID", SHAPE, CELL, TOP, actnum=ACTNUM))
    np.testing.assert_allclose(grid.active(grid.centres), _centres())
    np.testing.assert_allclose(grid.active(grid.volume), 100.0 * 100.0 * 5.0)
    np.testing.assert_allclose(grid.boxes[2, 1, 1], [200.0, 300.0, 100.0, 200.0])
    assert grid.active(grid.top).shape == (N_ACTIVE,)


def test_gravity_is_newton_over_the_cells_fluid_mass_change(
    tmp_path: Path, write_grid: Callable[..., Path], write_case: Callable[..., Path]
) -> None:
    model = _model(tmp_path, write_grid, "gravity")
    assert model.field_sizes() == {"obs": 4}
    assert model.restart_needs() == {"DEN": "obs", "RPORV": "obs"}
    case = write_case(tmp_path / "m" / "C", SHAPE, _restarts(), actnum=ACTNUM, cell=CELL, top=TOP)
    got = model.forecast(case, _keys())
    r = _restarts()

    def mass(d: dict[str, Any]) -> NDArray[np.float64]:
        sw = np.broadcast_to(np.asarray(d["SWAT"], dtype=float), (N_ACTIVE,))
        pv = np.broadcast_to(np.asarray(d["RPORV"], dtype=float), (N_ACTIVE,))
        out: NDArray[np.float64] = pv * (sw * d["WAT_DEN"] + (1 - sw) * d["OIL_DEN"])
        return out

    dm = mass(r[JUL]) - mass(r[JAN])
    expected = point_mass_gravity(STATIONS, _centres(), dm, G) / 1e-8  # µGal
    np.testing.assert_allclose(got, expected, rtol=1e-6)  # restarts are float32
    # Newton: one cell's change alone, under a station straight above it.
    one = np.zeros(N_ACTIVE)
    one[0] = dm[0]
    above = point_mass_gravity(STATIONS[:1], _centres(), one, G)[0]
    assert above == pytest.approx(G * dm[0] / (TOP + 2.5 - 300.0) ** 2)


def test_prisms_keep_each_cells_mass(
    tmp_path: Path, write_grid: Callable[..., Path], write_case: Callable[..., Path]
) -> None:
    model = _model(tmp_path, write_grid, "gravity", model="prism")
    survey = model.surveys[0]
    assert isinstance(survey, GravitySurvey)
    assert survey.bounds is not None
    np.testing.assert_allclose(survey.bounds[0], [0.0, 100.0, 0.0, 100.0, 2000.0, 2005.0])
    case = write_case(tmp_path / "m" / "C", SHAPE, _restarts(), actnum=ACTNUM, cell=CELL, top=TOP)
    point = _model(tmp_path / "p", write_grid, "gravity").forecast(case, _keys())
    prism = model.forecast(case, _keys())
    assert model.warnings() == []  # the prisms are the cells: no skew
    # 1.7 km away a 100 m cell is nearly its point mass.
    np.testing.assert_allclose(prism, point, rtol=1e-3)
    assert not np.array_equal(prism, point)


@pytest.mark.parametrize(
    ("options", "change"),
    [
        ({"compressibility": 1e-9}, "bulk"),
        ({"compressibility": 1e-9, "volume": "pore"}, "pore"),
        ({"volume_change": "pore_volume"}, "rporv"),
    ],
)
def test_displacement_is_the_kernel_over_each_cells_volume_decrease(
    tmp_path: Path,
    write_grid: Callable[..., Path],
    write_case: Callable[..., Path],
    options: dict[str, Any],
    change: str,
) -> None:
    model = _model(tmp_path, write_grid, "seafloor_displacement", **options)
    survey = model.surveys[0]
    assert isinstance(survey, DisplacementSurvey)
    needs = {} if change == "bulk" else {"RPORV": "obs"}
    assert model.restart_needs() == needs
    case = write_case(tmp_path / "m" / "C", SHAPE, _restarts(), actnum=ACTNUM, cell=CELL, top=TOP)
    got = model.forecast(case, _keys())
    r = _restarts()
    drop = 1e5 * (200.0 - np.linspace(170.0, 190.0, N_ACTIVE))  # Pa
    rporv = np.linspace(11800, 11900, N_ACTIVE)
    decrease = {
        "bulk": 1e-9 * 50000.0 * drop,
        "pore": 1e-9 * 12000.0 * drop,
        "rporv": 12000.0 - rporv,
    }[change]
    del r
    expected = Geertsma().matrix(STATIONS, _centres()) @ decrease / 1e-2  # cm
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    assert np.all(got[:3] > 0)  # a pressure drop compacts: subsidence, positive down


def test_regular_stations_cover_the_active_cells(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    stations = {"spacing": 100.0, "padding": 200.0, "depth": 300.0}
    model = _model(tmp_path, write_grid, "gravity", stations=stations)
    survey = model.surveys[0]
    assert isinstance(survey, GravitySurvey)
    c = _centres()
    expected = regular_stations(c[:, 0], c[:, 1], spacing=100.0, padding=200.0, depth=300.0)
    np.testing.assert_array_equal(survey.stations, expected)


def test_what_holds_for_no_member_is_refused_before_any_runs(
    tmp_path: Path, write_grid: Callable[..., Path], write_case: Callable[..., Path]
) -> None:
    np.save(tmp_path / "bad.npy", np.zeros((3, 2)))
    with pytest.raises(ConfigError, match=r"must hold an \(n, 3\) array"):
        _model(tmp_path, write_grid, "gravity", stations={"file": str(tmp_path / "bad.npy")})
    # A station below a cell: Geertsma's half space has none.
    deep = {"spacing": 100.0, "depth": 2100.0}
    with pytest.raises(ConfigError, match="obs: a cell is not below a station"):
        _model(tmp_path, write_grid, "seafloor_displacement", compressibility=1e-9, stations=deep)
    # The restarts without RPORV: the deck's RPTRST, so every member's.
    model = _model(tmp_path, write_grid, "gravity")
    restarts = {d: {k: v for k, v in a.items() if k != "RPORV"} for d, a in _restarts().items()}
    case = write_case(tmp_path / "m" / "C", SHAPE, restarts, actnum=ACTNUM, cell=CELL, top=TOP)
    with pytest.raises(ConfigError, match="RPTRST BASIC=2 RPORV"):
        model.forecast(case, _keys())


def test_a_member_with_other_active_cells_is_refused(
    tmp_path: Path, write_grid: Callable[..., Path], write_case: Callable[..., Path]
) -> None:
    model = _model(tmp_path, write_grid, "gravity")
    other = list(ACTNUM)
    other[10] = 0  # its column keeps an active cell, so AVO's traces would not notice
    restarts = {
        d: {k: (np.asarray(v)[:10] if np.ndim(v) else v) for k, v in a.items()}
        for d, a in _restarts().items()
    }
    case = write_case(tmp_path / "m" / "C", SHAPE, restarts, actnum=other, cell=CELL, top=TOP)
    with pytest.raises(SimulationError, match=r"1 cell\(s\) are active .* \(2, 2, 2\)"):
        model.forecast(case, _keys())


@pytest.mark.parametrize(
    ("units", "factors"),
    [
        (2, (0.3048, 6894.757293168361, 16.018463373960138, 0.158987294928)),  # FIELD
        (3, (0.01, 101325.0, 1000.0, 1e-6)),  # LAB: cm, atm, g/cc, cc
    ],
)
def test_other_units_give_the_metric_cases_data(
    tmp_path: Path,
    write_grid: Callable[..., Path],
    write_case: Callable[..., Path],
    units: int,
    factors: tuple[float, float, float, float],
) -> None:
    """A FIELD (LAB) deck's feet (cm), psia (atm), lb/ft³ (g/cc) and rb (cc) give the
    same Δg and u_z as METRIC's."""
    ft, psi, lbft3, rb = factors
    cell_ft = tuple(c / ft for c in CELL)
    field_restarts = {
        d: {
            "PRESSURE": np.asarray(a["PRESSURE"]) * 1e5 / psi,
            "SWAT": a["SWAT"],
            "OIL_DEN": a["OIL_DEN"] / lbft3,
            "WAT_DEN": a["WAT_DEN"] / lbft3,
            "RPORV": np.asarray(a["RPORV"]) / rb,
        }
        for d, a in _restarts().items()
    }
    metric = write_case(tmp_path / "m" / "C", SHAPE, _restarts(), actnum=ACTNUM, cell=CELL, top=TOP)
    field = write_case(
        tmp_path / "f" / "C", SHAPE, field_restarts, actnum=ACTNUM, cell=cell_ft, top=TOP / ft,
        units=units,
    )  # fmt: skip
    for kind, options in (("gravity", {}), ("seafloor_displacement", {"compressibility": 1e-9})):
        feet = tmp_path / kind / "feet"
        feet.mkdir(parents=True)
        write_grid(feet / "REF.EGRID", SHAPE, cell_ft, TOP / ft, actnum=ACTNUM, units=units)
        np.save(feet / "stations.npy", STATIONS)
        section = GeophysicsSection.model_validate(
            {
                "grid_file": feet / "REF.EGRID",
                "data": [
                    {
                        "kind": kind,
                        "datatype": "obs",
                        "vintages": ["2020-07-01"],
                        "options": {
                            "baseline": "2020-01-01",
                            "stations": {"file": str(feet / "stations.npy")},
                            **options,
                        },
                    }
                ],
            }
        )
        kernel = Geertsma() if kind == "seafloor_displacement" else None
        in_feet = build_geophysics_model(section, None, [None], [None], None, [kernel])
        expected = _model(tmp_path / kind, write_grid, kind, **options).forecast(metric, _keys())
        np.testing.assert_allclose(in_feet.forecast(field, _keys()), expected, rtol=1e-5)


def test_cell_vectors_of_another_length_fail_the_member() -> None:
    """Dual porosity or local grids give more values than grid_file's active cells."""

    class _State:
        case = Path("C")

        def pore_volume(self, date: dt.date) -> NDArray[np.float64]:
            return np.ones(2 * N_ACTIVE)

    quantities = CaseQuantities(cast(CaseState, _State()), N_ACTIVE)
    with pytest.raises(SimulationError, match="RPORV have 22 values, grid_file 11 active"):
        quantities.pore_volume(JAN)


def test_a_grid_file_without_its_length_unit_is_refused(
    tmp_path: Path, write_grid: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import apex.workflows.geophysics as geophysics

    monkeypatch.setattr(geophysics, "grid_unit_stated", lambda path: False)
    with pytest.raises(ConfigError, match="states no length unit"):
        _model(tmp_path, write_grid, "gravity")


def test_stations_are_in_the_grids_own_frame_not_mapaxes(
    tmp_path: Path, write_grid: Callable[..., Path], write_case: Callable[..., Path]
) -> None:
    """A regression (the phase-3 review): with resdata's default, MAPAXES moved the cells,
    so a y-flip mirrored the stations' order, while SimulatorWrap's stations and the AVO
    regions are in the grid's own frame (its COORD)."""
    flipped = tmp_path / "flipped"
    flipped.mkdir()
    write_grid(
        flipped / "REF.EGRID", SHAPE, CELL, TOP, actnum=ACTNUM,
        mapaxes=(1000.0, 0.0, 1000.0, 2000.0, 2000.0, 2000.0),
    )  # fmt: skip
    grid = Grid.read(flipped / "REF.EGRID")
    np.testing.assert_allclose(grid.active(grid.centres), _centres())
    stations = {"spacing": 100.0, "padding": 200.0, "depth": 300.0}
    plain = _model(tmp_path / "plain", write_grid, "gravity", stations=stations)
    section = GeophysicsSection.model_validate(
        {
            "grid_file": flipped / "REF.EGRID",
            "data": [
                {
                    "kind": "gravity",
                    "datatype": "obs",
                    "vintages": ["2020-07-01"],
                    "options": {"baseline": "2020-01-01", "stations": stations},
                }
            ],
        }
    )
    mapped = build_geophysics_model(section, None, [None], [None], None, [None])
    a, b = plain.surveys[0], mapped.surveys[0]
    assert isinstance(a, GravitySurvey)
    assert isinstance(b, GravitySurvey)
    np.testing.assert_array_equal(a.stations, b.stations)


def test_only_displacement_has_a_size_cap(
    tmp_path: Path, write_grid: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gravity keeps no matrix: its size is stated, not refused (the phase-3 review)."""
    import apex.workflows.geophysics as geophysics

    monkeypatch.setattr(geophysics, "_MAX_PAIRS", 10)
    monkeypatch.setattr(geophysics, "_GRAVITY_NOTE", 10)
    gravity = _model(tmp_path / "g", write_grid, "gravity")
    assert any("station-cell pairs, computed for every member" in w for w in gravity.warnings())
    with pytest.raises(ConfigError, match="kernel matrix every member's task carries"):
        _model(tmp_path / "d", write_grid, "seafloor_displacement", compressibility=1e-9)


def test_a_station_at_or_near_a_cell_centre(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    centre = _centres()[0]
    np.save(tmp_path / "at.npy", [centre])
    with pytest.raises(ConfigError, match="at a cell's centre"):
        _model(tmp_path / "a", write_grid, "gravity", stations={"file": str(tmp_path / "at.npy")})
    np.save(tmp_path / "near.npy", [centre + np.array([0.0, 0.0, -10.0])])
    near = _model(
        tmp_path / "n", write_grid, "gravity", stations={"file": str(tmp_path / "near.npy")}
    )
    assert any("model = 'prism' is exact there" in w for w in near.warnings())
    prism = _model(
        tmp_path / "p", write_grid, "gravity", model="prism",
        stations={"file": str(tmp_path / "at.npy")},
    )  # fmt: skip
    assert not any("exact there" in w for w in prism.warnings())


def test_no_station_or_an_ambiguous_seabed_is_refused(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    column = tmp_path / "column"
    column.mkdir()
    write_grid(column / "REF.EGRID", (1, 1, 2), CELL, TOP)
    section = GeophysicsSection.model_validate(
        {
            "grid_file": column / "REF.EGRID",
            "data": [
                {
                    "kind": "gravity",
                    "datatype": "obs",
                    "vintages": ["2020-07-01"],
                    "options": {"baseline": "2020-01-01", "stations": {"spacing": 100.0}},
                }
            ],
        }
    )
    with pytest.raises(ConfigError, match="span nothing along x or y"):
        build_geophysics_model(section, None, [None], [None], None, [None])
    np.savez(tmp_path / "seabed.npz", a=np.zeros((2, 3)), b=np.ones((2, 3)))
    seabed = {"spacing": 100.0, "seabed_file": str(tmp_path / "seabed.npz")}
    with pytest.raises(ConfigError, match="several arrays"):
        _model(tmp_path / "s", write_grid, "gravity", stations=seabed)
    keyed = _model(tmp_path / "k", write_grid, "gravity", stations={**seabed, "seabed_key": "b"})
    survey = keyed.surveys[0]
    assert isinstance(survey, GravitySurvey)
    np.testing.assert_array_equal(survey.stations[:, 2], 1.0)


def test_a_feet_grid_file_is_read_in_metres_for_every_kind(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    """The reference grid is read in its GRIDUNIT, AVO's too (before phase 3 it was raw)."""
    grid = Grid.read(write_grid(tmp_path / "F.EGRID", SHAPE, (328.0, 328.0, 16.4), 6562.0,
                                units=2))  # fmt: skip
    assert grid.top[0, 0, 0] == pytest.approx(6562.0 * 0.3048, rel=1e-6)
    assert grid.centres[1, 0, 0, 0] == pytest.approx(1.5 * 328.0 * 0.3048, rel=1e-6)


def test_refinement_and_station_files_on_a_mapaxes_grid_are_in_its_own_frame(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    flipped = tmp_path / "flipped"
    flipped.mkdir()
    grid = write_grid(
        flipped / "REF.EGRID", SHAPE, CELL, TOP, actnum=ACTNUM,
        mapaxes=(1000.0, 0.0, 1000.0, 2000.0, 2000.0, 2000.0),
    )  # fmt: skip
    np.save(tmp_path / "stations.npy", STATIONS)
    refined = {"spacing": 100.0, "padding": 200.0, "depth": 300.0, "refine": [50.0, 50.0, 120.0]}
    for stations in (refined, {"file": str(tmp_path / "stations.npy")}):
        plain = _model(tmp_path / f"p{len(stations)}", write_grid, "gravity", stations=stations)
        section = GeophysicsSection.model_validate(
            {
                "grid_file": grid,
                "data": [
                    {
                        "kind": "gravity",
                        "datatype": "obs",
                        "vintages": ["2020-07-01"],
                        "options": {"baseline": "2020-01-01", "stations": stations},
                    }
                ],
            }
        )
        mapped = build_geophysics_model(section, None, [None], [None], None, [None])
        a, b = plain.surveys[0], mapped.surveys[0]
        assert isinstance(a, GravitySurvey)
        assert isinstance(b, GravitySurvey)
        np.testing.assert_array_equal(a.stations, b.stations)
        np.testing.assert_array_equal(a.centres, b.centres)
        assert mapped.warnings() == []


def test_stations_far_from_the_cells_are_warned_of(
    tmp_path: Path, write_grid: Callable[..., Path]
) -> None:
    """A station file in map coordinates for a MAPAXES grid lands far from its cells."""
    np.save(tmp_path / "utm.npy", STATIONS + np.array([5e5, 6.7e6, 0.0]))
    model = _model(
        tmp_path / "m", write_grid, "gravity", stations={"file": str(tmp_path / "utm.npy")}
    )
    assert any("no station lies over the cells" in w for w in model.warnings())


def test_a_degenerate_cell_cannot_hide_a_station_at_a_centre() -> None:
    from apex.workflows.geophysics import _closest

    centres = np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]])
    assert _closest(np.array([[0.0, 0.0, 100.0]]), centres, np.array([0.0, 10.0])) == 0.0
    assert _closest(np.array([[0.0, 0.0, 90.0]]), centres, np.array([0.0, 10.0])) == pytest.approx(
        np.hypot(50.0, 10.0) / 10.0
    )
