"""APEX cases as PET cases: the inverse of ``apex pet-import``.

The test of both directions is the round trip: a PET case imported, exported and imported
again gives the same APEX case. What PET cannot run is refused, never approximated.
"""

import tomllib
from pathlib import Path
from typing import Any

import pytest

from apex.core.errors import ConfigError
from apex.storage.pet_config import import_pet_case
from apex.storage.pet_export import export_pet_case
from apex.testing import PET_CASES as CASES
from apex.testing import PET_STORED as STORED
from apex.testing import write_pet_case as _pet_case

VARIO = """
[ensemble]
ne = 30
state = "permx"
[ensemble.prior_permx]
vario = "exp"
mean = {mean}
var = [1.0, 2.0]
range = 4.0
aniso = {aniso}
angle = {angle}
grid = {grid}
limits = [-5.0, 9.0]
"""


def _round_trip(tmp_path: Path, pet: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    first = import_pet_case(pet, tmp_path / "apex1", name="case")
    exported = export_pet_case(first.config, tmp_path / "pet2")
    second = import_pet_case(exported.path, tmp_path / "apex2", name="case")
    return (tomllib.loads(first.config.read_text()), tomllib.loads(second.config.read_text()),
            exported.path)  # fmt: skip


@pytest.mark.parametrize("scheme", sorted(CASES))
def test_a_pet_case_comes_back_the_same(tmp_path: Path, scheme: str) -> None:
    before, after, pet = _round_trip(tmp_path, _pet_case(tmp_path / "pet", STORED, CASES[scheme]))

    assert after == before
    assert (pet.parent / "RUNFILE.mako").read_text().startswith("INCLUDE\n'../grid/Grid.grdecl' /")
    written = tomllib.loads(pet.read_text())
    assert written["dataassim"]["obsname"] == "dates"
    assert written["simulator"]["datatype"] == ["WOPR:P1", "WWCT:P1"]
    assert written["simulator"]["reportpoint"][0].year == 2024  # a TOML datetime
    assert (pet.parent / "grid" / "Grid.grdecl").is_file()
    obs1 = (tmp_path / "apex1" / "observations.csv").read_text()
    assert (tmp_path / "apex2" / "observations.csv").read_text() == obs1


@pytest.mark.parametrize(
    ("grid", "aniso", "angle", "mean"),
    [
        ("[3, 3, 2]", "2.0", "0.0", '"priormean.npz"'),  # PET's own, along j
        ("[3, 3, 2]", "1.0", "0.0", "0.5"),  # isotropic
    ],
)
def test_a_variogram_prior_comes_back_the_same(
    tmp_path: Path, grid: str, aniso: str, angle: str, mean: str
) -> None:
    ensemble = VARIO.format(grid=grid, aniso=aniso, angle=angle, mean=mean)
    pet = _pet_case(tmp_path / "pet", ensemble, CASES["esmda"])

    before, after, _ = _round_trip(tmp_path, pet)

    assert after == before


def _apex_case(tmp_path: Path, **changes: Any) -> Path:  # noqa: ANN401
    pet = _pet_case(tmp_path / "pet", STORED, CASES["lmenrml"])
    config = import_pet_case(pet, tmp_path / "apex").config
    raw = tomllib.loads(config.read_text())
    for key, value in changes.items():
        section, _, field = key.partition(".")
        if field:
            raw.setdefault(section, {})[field] = value
        else:
            raw[section] = value
    import tomli_w

    config.write_text(tomli_w.dumps(raw))
    return config


KEYED = {"ensemble_file": "prior_ensemble.npz", "key": "k"}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"data_transforms": [{"name": "wavelet"}]}, "data transforms"),
        ({"method.localization": {"name": "adaptive_prior"}}, "localization"),
        ({"method": {"name": "gies"}}, "^GIES is not implemented in PET"),
        ({"method.options": {"max_inner_iterations": 4, "trust": 1}}, "no option"),
        ({"method.schedule": {"confirm_convergence": True}}, "no schedule option"),
        ({"method.strategy": {"name": "subspace", "options": {"truncation": 3}}}, "no option"),
        ({"state": {"variables": [{"name": "permx", "size": 6, "lower": 0.0}]}}, "both bounds"),
        ({"prior": {"generator": {"name": "file", "options": KEYED}}}, "key"),
        ({"method": {"name": "chop"}}, "^CHOP is not implemented in PET"),
        ({"method.strategy": {"name": "gmac"}}, "'gmac' analysis is not implemented in PET"),
        ({"prior": {"ensemble_size": 4, "generator": {"name": "gaussian"}}}, "stored ensemble"),
        ({"simulator.options": {"deck_template_file": "deck.mako",
                                "template_values": {"cap": 6}}}, "template values"),
        ({"validation": {"wells": ["P1"]}}, "cross validation"),
    ],
)  # fmt: skip
def test_what_pet_cannot_run_is_refused(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    config = _apex_case(tmp_path, **changes)

    with pytest.raises(ConfigError, match=message):
        export_pet_case(config, tmp_path / "out")


def test_a_non_square_variogram_is_refused(tmp_path: Path) -> None:
    ensemble = VARIO.format(grid="[6, 4, 2]", aniso="2.0", angle="30.0", mean="0.0")
    config = import_pet_case(_pet_case(tmp_path / "pet", ensemble, CASES["esmda"]),
                             tmp_path / "apex").config  # fmt: skip

    with pytest.raises(ConfigError, match="scrambles"):
        export_pet_case(config, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # checked before anything is written


def test_the_energy_apex_uses_is_written(tmp_path: Path) -> None:
    config = _apex_case(tmp_path, **{"method.strategy": {"name": "approx"}})

    written = tomllib.loads(export_pet_case(config, tmp_path / "out").path.read_text())

    assert written["dataassim"]["iteration"]["energy"] == 98.0  # APEX's, not PET's 95


def test_a_rotated_field_on_a_square_grid_is_refused(tmp_path: Path) -> None:
    options = {"grid": [3, 3, 1], "major_range": 4.0, "minor_range": 2.0, "azimuth": 30.0}
    field = {"name": "gaussian_field", "options": options}
    config = _apex_case(tmp_path, prior={"ensemble_size": 5, "variables": {"permx": field}})

    with pytest.raises(ConfigError, match="square grid"):
        export_pet_case(config, tmp_path / "out")


def test_with_no_strategy_the_enrml_energy_is_apexs_own(tmp_path: Path) -> None:
    config = _apex_case(tmp_path)
    raw = tomllib.loads(config.read_text())
    del raw["method"]["strategy"]
    import tomli_w

    config.write_text(tomli_w.dumps(raw))

    written = tomllib.loads(export_pet_case(config, tmp_path / "out").path.read_text())

    assert written["dataassim"]["iteration"]["energy"] == 95.0  # LM-EnRML's ApproxUpdate(0.95)


def test_the_full_analysis_needs_pets_scaling(tmp_path: Path) -> None:
    config = _apex_case(tmp_path, **{"method.strategy": {"name": "full"}})

    with pytest.raises(ConfigError, match="configured variances"):
        export_pet_case(config, tmp_path / "out")


def test_simulator_options_pet_has_no_place_for_are_noted(tmp_path: Path) -> None:
    options = {"deck_template_file": "deck.mako", "keep_files": False, "threads": 4}
    config = _apex_case(tmp_path, **{"simulator.options": options})

    notes = export_pet_case(config, tmp_path / "out").notes

    assert any("keep_files, threads" in n for n in notes)
