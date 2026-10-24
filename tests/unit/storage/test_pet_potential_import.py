"""``apex pet-import`` of PET's gravity and seafloor displacement: ``grav``, ``sea_disp``.

SimulatorWrap's ``flow_grav`` and ``flow_seafloor_disp`` (``d29dc57``) write one value per
station of ``measurement_locations`` at each vintage, a difference from the baseline: the
import keeps them as station fields, with the reference's settings.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.services.composition import build_geophysics, load_config
from apex.storage.inputs import read_observation_section
from apex.storage.pet_config import ImportResult, import_pet_case
from apex.workflows.geophysics import DisplacementSurvey, GravitySurvey

#: A 4 x 3 x 2 grid of 100 m cells: centres x = 50 ... 350, y = 50 ... 250, so stations
#: 100 m apart without padding are 3 x 2 (``ceil(span / spacing)`` per axis).
NX, NY, NZ = 4, 3, 2
N_STATION = 6
DECK = "RUNSPEC\nOIL\nWATER\nGAS\nMETRIC\nSTART\n 15 MAR 2022 /\nSCHEDULE\n"

CONFIG = """
[ensemble]
ne = 4
state = "permx"
importstate = "prior.npz"

[dataassim]
scheme = "esmda"
data = "data.csv"
datavar = "var.csv"
[dataassim.mda]
tot_assim_steps = 2

[simulator]
runfile = "RUNFILE"
startdate = "15/3/2022"
pem = [["model", "standardrp elasticproperties"], ["phases", "OIL WAT GAS"],
       ["press_conv", 0.1]]
grav = [["baseline", 0], ["vintage", [100, 465]], ["meas_spacing", 100], ["padding", 0],
        ["water_depth", 50]]
sea_disp = [["baseline", 0], ["vintage", [100, 465]], ["meas_spacing", 100], ["padding", 0],
            ["water_depth", 50], ["method", "pressure"], ["model", "geertsma"],
            ["poisson", 0.25], ["compressibility", 1e-3], ["z_base", 3000]]
"""


@pytest.fixture
def pet_case(tmp_path: Path, write_grid: Callable[..., Path]) -> Path:
    case = tmp_path / "pet"
    case.mkdir()
    (case / "RUNFILE.mako").write_text(DECK + "${permx[0]}\n")
    write_grid(case / "REF.EGRID", (NX, NY, NZ))
    rng = np.random.default_rng(3)
    for kind in ("grav", "sea_disp"):
        for days in (100, 465):
            np.savez(case / f"{kind}_{days}.npz", values=rng.normal(size=N_STATION))
    (case / "data.csv").write_text(
        "days,WOPR:P1,grav,sea_disp\n100,10.0,grav_100.npz,sea_disp_100.npz\n"
        "465,9.0,grav_465.npz,sea_disp_465.npz\n"
    )
    (case / "var.csv").write_text(
        "days,WOPR:P1,grav,sea_disp\n100,\"['abs', 1]\",\"['abs', 4]\",\"['abs', 0.01]\"\n"
        "465,\"['abs', 1]\",\"['abs', 4]\",\"['abs', 0.01]\"\n"
    )
    np.savez(case / "prior.npz", permx=np.zeros((24, 4)))
    (case / "CONFIG.toml").write_text(CONFIG)
    return case / "CONFIG.toml"


def _edit(old: str, new: str) -> Callable[[Path], None]:
    def edit(case: Path) -> None:
        text = case.read_text()
        assert old in text
        case.write_text(text.replace(old, new))

    return edit


def _result(config: Path, grid: bool = True) -> ImportResult:
    return import_pet_case(
        config,
        config.parent.parent / "apex",
        grid_file=config.parent / "REF.EGRID" if grid else None,
    )


def _data(result: ImportResult) -> dict[str, dict[str, Any]]:
    raw = tomllib.loads(result.config.read_text())
    return {d["datatype"]: d for d in raw["geophysics"]["data"]}


def test_pets_gravity_and_displacement_become_station_data(pet_case: Path) -> None:
    result = _result(pet_case)
    data = _data(result)
    assert set(data) == {"grav", "sea_disp"}
    grav, disp = data["grav"], data["sea_disp"]
    assert (grav["kind"], disp["kind"]) == ("gravity", "seafloor_displacement")
    assert grav["vintages"] == ["2022-06-23", "2023-06-23"]
    assert grav["options"]["baseline"] == "2022-03-15"
    assert grav["options"]["stations"] == {"file": "grav_stations.npy"}
    stations = np.load(result.config.parent / "grav_stations.npy")
    # SimulatorWrap's stations, 100 apart without padding over centres 50 ... 350, 50 ... 250.
    np.testing.assert_array_equal(stations, _box_stations(100.0, 0.0, 50.0))
    assert grav["options"]["gravitational_constant"] == 6.67e-11
    # 1e-3 per MPa (the deck's bar times press_conv 0.1) is 1e-9 per Pa.
    assert disp["options"]["compressibility"] == pytest.approx(1e-9)
    assert (disp["options"]["volume_change"], disp["options"]["volume"]) == ("pressure", "pore")
    assert disp["options"]["kernel"] == {
        "name": "geertsma",
        "options": {"poisson": 0.25, "truncation_factor": 3.0, "free_surface": "datum"},
    }
    raw = tomllib.loads(result.config.read_text())
    assert "pem" not in raw["geophysics"]  # AVO's only
    assert "simulator.grav" not in " ".join(result.ignored)

    config = load_config(result.config)
    model = build_geophysics(config)
    assert model is not None
    observations = read_observation_section(config.observations)
    model.check_observations(observations.keys)  # one value per APEX station
    assert isinstance(model.surveys[0], GravitySurvey)
    assert isinstance(model.surveys[1], DisplacementSurvey)
    values = dict(zip(observations.keys, observations.values, strict=True))
    pet = np.load(pet_case.parent / "grav_465.npz")["values"]
    got = [values[DataKey("2023-06-23", "grav", i)] for i in range(N_STATION)]
    np.testing.assert_array_equal(got, pet)
    assert any("filter_rporv" in n for n in result.notes)
    assert any("6.67e-11" in n for n in result.notes)


def _box_stations(
    spacing: float,
    padding: float,
    depth: float,
    length: float = 1.0,
    **kwargs: Any,  # noqa: ANN401
) -> np.ndarray:
    """APEX's regular stations over the box's cell centres, in the grid's unit, scaled."""
    from apex.algorithms.geophysics import regular_stations

    x, y = np.meshgrid(np.arange(NX) * 100.0 + 50.0, np.arange(NY) * 100.0 + 50.0)
    out: np.ndarray = (
        regular_stations(
            x.ravel(), y.ravel(), spacing=spacing, padding=padding, depth=depth, **kwargs
        )
        * length
    )
    return out


def _sized(case: Path, kind: str, n: int) -> None:
    rng = np.random.default_rng(4)
    for days in (100, 465):
        np.savez(case.parent / f"{kind}_{days}.npz", values=rng.normal(size=n))


def test_the_options_of_each_kind_are_mapped(pet_case: Path) -> None:
    np.savetxt(pet_case.parent / "seabed.txt", [[0, 0, -60, 1, 1], [400, 300, -80, 4, 3]])
    _edit('["water_depth", 50]]', '["seabed", "seabed.txt"], ["well_coord", [200, 150]]]')(pet_case)
    _edit('["method", "pressure"]', '["method", "porv"]')(pet_case)
    _edit('["z_base", 3000]', '["z_base", 3000], ["receiver_trunc_factor", 6]')(pet_case)
    seabed = (np.array([0.0, 400.0]), np.array([0.0, 300.0]), np.array([-60.0, -80.0]))
    expected = _box_stations(100.0, 0.0, 0.0, refine=(200.0, 150.0, 5000.0), seabed=seabed)
    _sized(pet_case, "grav", expected.shape[0])
    result = _result(pet_case)
    data = _data(result)
    stations = np.load(result.config.parent / "grav_stations.npy")
    np.testing.assert_array_equal(stations, expected)
    disp = data["sea_disp"]["options"]
    assert disp["volume_change"] == "pore_volume"
    assert "compressibility" not in disp
    assert disp["kernel"]["options"]["truncation_factor"] == 3.0  # G14
    assert any("G14" in n for n in result.notes)
    load_config(result.config)  # valid


def test_van_opstal_is_geertsma_as_pet_computes_and_a_baseline_vintage_is_dropped(
    pet_case: Path,
) -> None:
    """SimulatorWrap lowercases the model before comparing it with 'van_Opstal' (G16)."""
    _edit('["model", "geertsma"]', '["model", "van_Opstal"]')(pet_case)
    grav = 'grav = [["baseline", 0], ["vintage", ['
    _edit(grav + "100, 465]]", grav + "0, 100, 465]]")(pet_case)  # PET's zero difference
    result = _result(pet_case)
    data = _data(result)
    assert data["sea_disp"]["options"]["kernel"] == {
        "name": "geertsma",
        "options": {"poisson": 0.25, "truncation_factor": 3.0, "free_surface": "datum"},
    }
    assert data["grav"]["vintages"] == ["2022-06-23", "2023-06-23"]
    assert any("G16" in n for n in result.notes)
    assert any("difference from itself" in n for n in result.notes)


def test_gravity_alone_needs_no_petro_elastic_model(pet_case: Path) -> None:
    _edit(
        'sea_disp = [["baseline", 0], ["vintage", [100, 465]], ["meas_spacing", 100], '
        '["padding", 0],\n            ["water_depth", 50], ["method", "pressure"], ["model", '
        '"geertsma"],\n            ["poisson", 0.25], ["compressibility", 1e-3], ["z_base", '
        "3000]]\n",
        "",
    )(pet_case)
    _edit(",grav,sea_disp\n100,10.0,grav_100.npz,sea_disp_100.npz\n465,9.0,grav_465.npz,"
          "sea_disp_465.npz\n", ",grav\n100,10.0,grav_100.npz\n465,9.0,grav_465.npz\n")(
        pet_case.parent / "data.csv"
    )  # fmt: skip
    _edit(",grav,sea_disp\n", ",grav\n")(pet_case.parent / "var.csv")
    _edit(",\"['abs', 4]\",\"['abs', 0.01]\"\n", ",\"['abs', 4]\"\n")(pet_case.parent / "var.csv")
    _edit('["phases", "OIL WAT GAS"]', '["phases", "WAT GAS"]')(pet_case)
    result = _result(pet_case)
    assert set(_data(result)) == {"grav"}
    config = load_config(result.config)
    assert config.geophysics is not None
    assert config.geophysics.pem is None
    assert any("leave out the deck's OIL" in n for n in result.notes)
    assert any("pem model, press_conv serve its AVO only" in n for n in result.notes)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit('["z_base", 3000]', '["z_base", 3000], ["geertsma_mode", "fast"]'), "G15"),
        (
            _edit('["water_depth", 50]]', '["water_depth", 50], ["baseline_mode", "rolling"]]'),
            "baseline_mode 'rolling'",
        ),
        (_edit('grav = [["baseline", 0], ', "grav = ["), "grav needs baseline"),
        (
            _edit(
                '["padding", 0],\n        ["water_depth", 50]]',
                '["padding", 0], ["colour", 1],\n        ["water_depth", 50]]',
            ),
            "grav has keys SimulatorWrap does not read: colour",
        ),
        (
            _edit('["model", "geertsma"]', '["model", "van_Opstal"], ["geertsma_mode", "fast"]'),
            "G15",
        ),
        (_edit('["phases", "OIL WAT GAS"]', '["phases", "OIL WAT"]'), "phases 'OIL WAT'"),
        # Data at a baseline that is not a vintage: PET would never fill them.
        (
            _edit(
                'grav = [["baseline", 0], ["vintage", [100, 465]]',
                'grav = [["baseline", 100], ["vintage", [465]]',
            ),
            "only at its vintages",
        ),
        (_edit('["compressibility", 1e-3], ', ""), "sea_disp needs compressibility"),
        (
            _edit(
                'pem = [["model", "standardrp elasticproperties"], ["phases", "OIL WAT GAS"],\n'
                '       ["press_conv", 0.1]]\n',
                "",
            ),
            "need a pem section",
        ),
        (
            _edit(
                "[dataassim.mda]", '[dataassim.compress]\ncompress_data = ["grav"]\n[dataassim.mda]'
            ),
            "compresses the grav data",
        ),
    ],
)
def test_what_apex_cannot_reproduce_is_refused(
    pet_case: Path, edit: Callable[[Path], None], message: str
) -> None:
    edit(pet_case)
    with pytest.raises(ConfigError, match=message):
        _result(pet_case)


def test_station_data_need_the_grid_file(pet_case: Path) -> None:
    with pytest.raises(ConfigError, match="give --grid-file"):
        _result(pet_case, grid=False)


def test_seabed_tables_and_units_follow_the_reference(pet_case: Path) -> None:
    np.savetxt(pet_case.parent / "seabed.txt", [[0, 0, -60], [400, 300, -80]])
    _edit('["water_depth", 50]]', '["seabed", "seabed.txt"]]')(pet_case)
    with pytest.raises(ConfigError, match="rows of 5 values"):
        _result(pet_case)


def test_field_decks_and_list_phases_are_noted(pet_case: Path) -> None:
    deck = pet_case.parent / "RUNFILE.mako"
    deck.write_text(deck.read_text().replace("METRIC", "FIELD").replace("GAS\n", "", 1))
    _edit('["phases", "OIL WAT GAS"]', '["phases", ["OIL WAT", "GAS"]]')(pet_case)
    notes = " | ".join(_result(pet_case).notes)
    assert "grav on a FIELD deck" in notes
    assert "sea_disp on a FIELD deck" in notes
    assert "name GAS, which the deck's RUNSPEC lacks" in notes


def test_station_lengths_are_converted_from_the_grids_unit(
    pet_case: Path, write_grid: Callable[..., Path]
) -> None:
    """SimulatorWrap places its stations in the grid's raw coordinates (feet on a FEET
    grid), so its meas_spacing, padding, water_depth and well_coord are in that unit."""
    write_grid(pet_case.parent / "REF.EGRID", (NX, NY, NZ), units=2)  # 100 ft cells
    _edit('["water_depth", 50]]', '["water_depth", 50], ["well_coord", [100, 200]]]')(pet_case)
    expected = _box_stations(100.0, 0.0, 50.0, 0.3048, refine=(100.0, 200.0, 5000.0))
    _sized(pet_case, "grav", expected.shape[0])
    result = _result(pet_case)
    stations = np.load(result.config.parent / "grav_stations.npy")
    np.testing.assert_allclose(stations, expected, rtol=1e-12)


def test_the_data_must_be_one_value_per_station(pet_case: Path) -> None:
    _sized(pet_case, "grav", N_STATION + 1)
    with pytest.raises(ConfigError, match="7 values, but SimulatorWrap's stations"):
        _result(pet_case)


def test_the_importers_copy_of_simulatorwraps_stations_is_its_own() -> None:
    """Against SimulatorWrap's find_cell_centre and measurement_locations' own outputs on a
    slanted grid (``tests/data/reference/geophysics``, cases stations and gravity)."""
    from apex.storage.pet_geophysics import _reference_centres, _reference_stations

    reference = Path(__file__).resolve().parents[2] / "data" / "reference" / "geophysics"
    with np.load(reference / "gravity.npz") as ref:
        x, y = _reference_centres(ref["in_coord"], ref["in_zcorn"], ref["in_actnum"])
        np.testing.assert_array_equal(np.vstack((x, y)), ref["out_centres"][:2])
    with np.load(reference / "stations.npz") as ref:
        x, y = _reference_centres(ref["in_coord"], ref["in_zcorn"], ref["in_actnum"])
        np.testing.assert_array_equal(np.vstack((x, y)), ref["in_centres"][:2])
        plain = _reference_stations(x, y, 300.0, 700.0, 250.0)
        np.testing.assert_array_equal(plain, ref["out_plain"])
        refined = _reference_stations(x, y, 300.0, 700.0, 250.0, [330.0, 210.0], 400.0)
        np.testing.assert_array_equal(refined, ref["out_refined"])
        table = ref["in_seabed"]
        seabed = {"x": table[:, 0], "y": table[:, 1], "z": table[:, 2]}
        np.testing.assert_array_equal(
            _reference_stations(x, y, seabed, 500.0, 300.0), ref["out_seabed"]
        )


def test_a_deck_without_units_is_noted_once(pet_case: Path) -> None:
    deck = pet_case.parent / "RUNFILE.mako"
    deck.write_text(deck.read_text().replace("METRIC\n", ""))
    notes = _result(pet_case).notes
    assert sum("names no unit system" in n for n in notes) == 1
