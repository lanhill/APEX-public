"""``apex pet-import`` of PET's AVO simulator: ``pem`` and ``avo`` to ``[geophysics]``.

The observations are written as SimulatorWrap ``flow_avo.get_avo_result`` (``d29dc57``)
writes them: the cube ``(NY, NX, n_time, n_angle)`` under its mask (active columns,
``step_x`` thinning J and ``step_y`` thinning I, the region of interest), in C order.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.domain.config import ConstantOverburdenSection, RickerWaveletSection
from apex.domain.observations import DataKey
from apex.services.composition import build_geophysics, load_config
from apex.storage.inputs import read_observation_section
from apex.storage.pet_config import ImportResult, import_pet_case
from apex.workflows.geophysics import AVOSurvey

#: A 4 x 3 x 2 grid of 100 m cells whose column (I, J) = (3, 0) is inactive: column
#: centres x = 50, 150, 250, 350 and y = 50, 150, 250.
NX, NY, NZ = 4, 3, 2
ANGLES = (10.0, 20.0)
TIMES = np.arange(1.0, 1.0045, 0.001)  # 5 samples, as SimulatorWrap forms them
ACTIVE = np.ones((NX, NY), dtype=bool)
ACTIVE[3, 0] = False
DECK = "RUNSPEC\nMETRIC\nSCHEDULE\nCOMPDAT\n 'P1' 1 1 1 2 OPEN /\n 'I1' 2 1 1 2 OPEN /\n/\n"

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
pem = [["model", "standardrp elasticproperties"], ["vintage", [100.0, 465.0]],
       ["overburden", "overburden.npz"], ["ntg", "no"]]
avo = [["dz", "./grid/dz.grdecl"], ["tops", "./grid/tops.grdecl"], ["angle", [10.0, 20.0]],
       ["frequency", 45.0], ["wave_len", 0.05], ["vp_shale", 2450], ["vs_shale", 849],
       ["den_shale", 2300], ["t_min", 1.0], ["t_max", 1.0045], ["t_sampling", 0.001],
       ["pp_func", "approx_zoeppritz_pp"], ["step_x", 2], ["roi_mode", "none"]]
"""


def _cube(seed: int, n_angle: int = 2) -> np.ndarray:
    """SimulatorWrap's cube ``(NY, NX, n_time, n_angle)``, each angle its own."""
    out: np.ndarray = np.random.default_rng(seed).normal(size=(NY, NX, TIMES.size, n_angle))
    return out


def _mask(step_x: int = 1, step_y: int = 1, roi: np.ndarray | None = None) -> np.ndarray:
    """SimulatorWrap's mask over ``(NY, NX)``: active, thinned ``[::step_x, ::step_y]``, ROI."""
    sample = np.zeros((NY, NX), dtype=bool)
    sample[::step_x, ::step_y] = True
    out: np.ndarray = ACTIVE.T & sample & (np.ones((NY, NX), dtype=bool) if roi is None else roi)
    return out


def _write_data(case: Path, mask: np.ndarray, n_angle: int = 2) -> None:
    for seed, name in ((1, "avo_1.npz"), (2, "avo_2.npz")):
        np.savez(case / name, avo=_cube(seed, n_angle)[mask].ravel())


@pytest.fixture
def pet_case(tmp_path: Path, write_grid: Callable[..., Path]) -> Path:
    case = tmp_path / "pet"
    case.mkdir()
    (case / "RUNFILE.mako").write_text(DECK + "${permx[0]}\n")
    actnum = np.repeat(ACTIVE[:, :, np.newaxis], NZ, axis=2).astype(int)
    write_grid(case / "REF.EGRID", (NX, NY, NZ), actnum=actnum.ravel(order="F"))
    np.savez(case / "overburden.npz", obvalues=np.full(int(actnum.sum()), 320.0))
    _write_data(case, _mask(step_x=2))
    (case / "data.csv").write_text("days,WOPR:P1,avo\n100,10.0,avo_1.npz\n465,9.0,avo_2.npz\n")
    (case / "var.csv").write_text(
        "days,WOPR:P1,avo\n100,\"['abs', 1]\",\"['abs', 0.04]\"\n"
        "465,\"['abs', 1]\",\"['abs', 0.09]\"\n"
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


def _result(
    config: Path, grid_file: Path | bool | None = True, avo_angles_checked: bool = True
) -> ImportResult:
    grid = config.parent / "REF.EGRID" if grid_file is True else grid_file
    return import_pet_case(
        config,
        config.parent.parent / "apex",
        grid_file=grid if isinstance(grid, Path) else None,
        avo_angles_checked=avo_angles_checked,
    )


def _import(config: Path, **kwargs: bool) -> Path:
    return _result(config, **kwargs).config


def _check_values(
    config_path: Path, mask: np.ndarray, angles: tuple[float, ...]
) -> list[tuple[int, int]]:
    """Check the model forecasts the observed keys, each PET's value of its column."""
    config = load_config(config_path)
    model = build_geophysics(config)
    assert model is not None
    observations = read_observation_section(config.observations)
    model.check_observations(observations.keys)
    survey = model.surveys[0]
    assert isinstance(survey, AVOSurvey)
    columns = [(int(i), int(j)) for i, j in survey.columns]
    # PET's kept columns, in its C order over (J, I): APEX's traces in Eclipse order.
    assert columns == [(int(i), int(j)) for j, i in zip(*np.nonzero(mask), strict=True)]
    values = dict(zip(observations.keys, observations.values, strict=True))
    for seed, vintage in ((1, "2022-06-23"), (2, "2023-06-23")):
        cube = _cube(seed, len(angles))
        for a, angle in enumerate(angles):
            name = f"avo_{angle:g}"
            for t, (i, j) in enumerate(columns):
                for s in range(TIMES.size):
                    assert values[DataKey(vintage, name, t * TIMES.size + s)] == cube[j, i, s, a]
    return columns


def test_pets_avo_vectors_become_the_traces_apex_forecasts(pet_case: Path) -> None:
    result = _result(pet_case)
    config = load_config(result.config)

    assert config.geophysics is not None
    assert config.geophysics.pem is not None
    assert config.geophysics.pem.name == "standard"
    assert config.geophysics.grid_file.name == "REF.EGRID"
    (datum,) = config.geophysics.data
    assert datum.datatype == "avo"
    # startdate is day/month/year; the vintages are days after it.
    assert datum.vintages == ["2022-06-23", "2023-06-23"]
    options = datum.avo()
    assert options.angles == [10.0, 20.0]
    assert (options.form, options.sampling, options.region) == ("traces", "nearest", None)
    assert options.reflectivity.name == "approx_zoeppritz_pp"
    assert options.wavelet == RickerWaveletSection(frequency=45.0, length=0.05)
    assert options.overburden == ConstantOverburdenSection(vp=2450.0, vs=849.0, density=2300.0)
    # PET's step_x thins J (its cube's first axis): APEX's step_y.
    assert (options.step_x, options.step_y) == (1, 2)
    # A METRIC deck without press_conv: PET's pressures, and its overburden, are in bar.
    overburden = config.geophysics.overburden_pressure
    assert (overburden.unit, overburden.key) == ("bar", "obvalues")  # type: ignore[union-attr]
    assert any("G9" in note for note in result.notes)
    assert any("step_x = 1, step_y = 2" in note for note in result.notes)
    assert "simulator.avo.dz" in result.ignored
    assert "simulator.pem.ntg" in result.ignored

    columns = _check_values(result.config, _mask(step_x=2), ANGLES)
    assert columns == [(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2), (3, 2)]
    # PET's day labels are the scalars' dates too.
    observations = read_observation_section(config.observations)
    assert DataKey("2022-06-23", "WOPR:P1") in observations.keys


def _single_angle(case: Path) -> None:
    _edit('["angle", [10.0, 20.0]]', '["angle", 20.0]')(case)
    _edit('["step_x", 2], ', "")(case)


def test_pets_default_ellipse_around_the_decks_completions_is_a_region_file(
    pet_case: Path,
) -> None:
    _single_angle(pet_case)
    _edit('["roi_mode", "none"]', '["roi_margin_m", 100]')(pet_case)
    # Seeds (50, 50) and (150, 50): an ellipse at (100, 50) with semi-axes 150 and 100.
    roi = np.zeros((NY, NX), dtype=bool)
    roi[0, :3] = True
    _write_data(pet_case.parent, _mask(roi=roi), n_angle=1)

    result = _result(pet_case, avo_angles_checked=False)

    config = load_config(result.config)
    assert config.geophysics is not None
    region = config.geophysics.data[0].avo().region
    assert region is not None
    np.testing.assert_array_equal(np.load(region.file), roi.T)  # type: ignore[union-attr]
    assert _check_values(result.config, _mask(roi=roi), (20.0,)) == [(0, 0), (1, 0), (2, 0)]


def test_completions_in_includes_are_not_pets_seeds(pet_case: Path) -> None:
    # SimulatorWrap reads the rendered deck only: with fewer than two completions there,
    # its ellipse surrounds every active column, so it keeps them all.
    _single_angle(pet_case)
    _edit('["roi_mode", "none"]', '["use_ellipse", true]')(pet_case)
    (pet_case.parent / "RUNFILE.mako").write_text("RUNSPEC\nMETRIC\nINCLUDE\n'../wells.inc' /\n")
    (pet_case.parent / "wells.inc").write_text(DECK)
    _write_data(pet_case.parent, _mask(), n_angle=1)

    result = _result(pet_case)
    config = load_config(result.config)
    assert config.geophysics is not None
    assert config.geophysics.data[0].avo().region is None
    assert any("included files" in note for note in result.notes)
    _check_values(result.config, _mask(), (20.0,))


def test_pets_rectangle_and_file_regions(pet_case: Path) -> None:
    _single_angle(pet_case)
    rectangle = '["roi_mode", "rectangle"], ["roi_margin_m", 0], ["x_range", [120, 0]]'
    _edit('["roi_mode", "none"]', rectangle + ', ["y_range", [0, 160]]')(pet_case)
    roi = np.zeros((NY, NX), dtype=bool)
    roi[:2, 0] = True  # x = 50 in [0, 120], y = 50 and 150 in [0, 160]
    _write_data(pet_case.parent, _mask(roi=roi), n_angle=1)
    assert _check_values(_import(pet_case), _mask(roi=roi), (20.0,)) == [(0, 0), (0, 1)]

    _edit(rectangle, '["roi_mode", "file"], ["roi_mask_file", "roi.npz"]')(pet_case)
    roi = np.zeros((NY, NX), dtype=bool)
    roi[1] = True  # SimulatorWrap's (NY, NX) layout
    np.savez(pet_case.parent / "roi.npz", mask=roi)
    _write_data(pet_case.parent, _mask(roi=roi), n_angle=1)
    columns = _check_values(_import(pet_case), _mask(roi=roi), (20.0,))
    assert columns == [(0, 1), (1, 1), (2, 1), (3, 1)]


def test_multi_angle_data_need_vouching_for_and_simulatorwraps_are_refused(
    pet_case: Path,
) -> None:
    with pytest.raises(ConfigError, match="--avo-angles-checked"):
        _import(pet_case, avo_angles_checked=False)
    cube = _cube(1)
    cube[..., 0] = cube[..., 1]  # what SimulatorWrap writes (G7)
    np.savez(pet_case.parent / "avo_1.npz", avo=cube[_mask(step_x=2)].ravel())
    with pytest.raises(ConfigError, match="G7"):
        _import(pet_case)


def test_pp_funcs_simulatorwrap_cannot_run_are_imported_with_a_note(pet_case: Path) -> None:
    _edit('"approx_zoeppritz_pp"', '"akirichards"')(pet_case)
    result = _result(pet_case)
    config = load_config(result.config)
    assert config.geophysics is not None
    reflectivity = config.geophysics.data[0].avo().reflectivity
    assert (reflectivity.name, reflectivity.options) == ("aki_richards", {"form": "linear"})
    assert any("G1" in note for note in result.notes)


def test_the_overburden_files_unit_follows_the_deck_and_press_conv(pet_case: Path) -> None:
    def unit() -> str:
        config = tomllib.loads(_import(pet_case).read_text())
        return str(config["geophysics"]["overburden_pressure"]["unit"])

    _edit('["ntg", "no"]', '["press_conv", 0.1]')(pet_case)
    assert unit() == "MPa"
    _edit('["press_conv", 0.1]', '["ntg", "no"]')(pet_case)
    template = pet_case.parent / "RUNFILE.mako"
    template.write_text(template.read_text().replace("METRIC", "FIELD"))
    assert unit() == "psi"
    template.write_text(template.read_text().replace("FIELD\n", ""))
    assert unit() == "bar"
    assert any("METRIC is assumed" in note for note in _result(pet_case).notes)


def _variance(case: Path) -> None:
    text = (case.parent / "var.csv").read_text().replace("'abs', 0.04", "'rel', 5")
    (case.parent / "var.csv").write_text(text)


def _short(case: Path) -> None:
    np.savez(case.parent / "avo_2.npz", avo=np.zeros(7))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_edit("[100.0, 465.0]", '[100.0, 465.0]], ["baseline", 200.0'), "before the baseline"),
        (_edit('["ntg", "no"]', '["compaction", "yes"]'), "PORV_RC"),
        (_edit('["ntg", "no"]', '["pressure", 1]'), "keys SimulatorWrap does not read: pressure"),
        (_edit('["ntg", "no"]', '["press_conv", 0.5]'), "none of Pa"),
        (_edit('["ntg", "no"]', '["press_conv", 0]'), "press_conv must be positive"),
        (_edit('["roi_mode", "none"]', '["roi_mode", "box"]'), "roi_mode 'box'"),
        # SimulatorWrap reads the flag with bool(): "no" is true, with its default ellipse.
        (_edit('["roi_mode", "none"]', '["roi_use_signal", "no"]'), "follows each member's"),
        (_edit('["step_x", 2]', '["colour", 2]'), "keys SimulatorWrap does not read: colour"),
        (_edit("avo = [", 'grav = [["x", 1]]\navo = ['), "grav has keys SimulatorWrap does not"),
        (_edit("standardrp elasticproperties", "elasticrp x"), "is not ported"),
        (_edit('"approx_zoeppritz_pp"', '"shuey"'), "not one of PyLops'"),
        (_edit('["frequency", 45.0], ', ""), "avo needs frequency"),
        (_edit('startdate = "15/3/2022"', 'startdate = "2022-03-15"'), "day/month/year"),
        (_edit("[100.0, 465.0]", "[100.0, 466.0]"), "only at the pem vintages"),
        (_edit("[simulator]", "[simulator]\ndimension = [3, 4, 2]"), "could not align"),
        (_short, r"7 values, but PET's mask keeps 7 traces of 5 samples at 2 angle\(s\) = 70"),
        (_variance, "absolute variance"),
        (
            _edit(
                "[dataassim.mda]",
                '[dataassim.localization]\nname = "distance_loc"\n[dataassim.mda]',
            ),
            "cannot localize geophysical traces",
        ),
    ],
)
def test_what_apex_cannot_reproduce_is_refused(
    pet_case: Path, edit: Callable[[Path], None], message: str
) -> None:
    edit(pet_case)
    with pytest.raises(ConfigError, match=message):
        _import(pet_case)


def test_a_mako_compdat_leaves_pets_region_unknown(pet_case: Path) -> None:
    _edit('["roi_mode", "none"]', '["roi_mode", "auto"]')(pet_case)
    template = pet_case.parent / "RUNFILE.mako"
    template.write_text(template.read_text().replace("'I1' 2 1", "'I1' ${i} 1"))
    with pytest.raises(ConfigError, match="COMPDAT holds Mako"):
        _import(pet_case)


def test_avo_data_need_a_grid_file(pet_case: Path) -> None:
    with pytest.raises(ConfigError, match="--grid-file"):
        _import(pet_case, grid_file=False)


def test_compressed_avo_data_are_refused(pet_case: Path) -> None:
    _edit("[dataassim.mda]", '[dataassim.compress]\ncompress_data = ["avo"]\n[dataassim.mda]')(
        pet_case
    )
    with pytest.raises(ConfigError, match="compresses the avo data"):
        _import(pet_case)


def test_a_pem_section_without_avo_is_refused(pet_case: Path) -> None:
    text = pet_case.read_text()
    pet_case.write_text(text[: text.index("avo = [")])
    with pytest.raises(ConfigError, match="pem section without avo"):
        _import(pet_case)


def test_large_traces_raise_max_field_values_with_a_note(pet_case: Path) -> None:
    _edit('["t_sampling", 0.001]', '["t_sampling", 0.0000001]')(pet_case)
    n = np.arange(1.0, 1.0045, 0.0000001).size * 7 * 2
    for name in ("avo_1.npz", "avo_2.npz"):
        np.savez(pet_case.parent / name, avo=np.random.default_rng(0).normal(size=n))
    result = _result(pet_case)
    config = load_config(result.config)
    assert config.geophysics is not None
    assert config.geophysics.max_field_values == 2 * n
    assert any("max_field_values is set" in note for note in result.notes)


def test_day_labels_become_dates_and_still_name_localization_times(pet_case: Path) -> None:
    text = pet_case.read_text()
    pet_case.write_text(
        text[: text.index("pem = [")].replace(
            "[dataassim.mda]",
            '[dataassim.localization]\nname = "distance_loc"\nfield = [1, 4, 6]\n'
            'entries = ["fb 0 1 0 2 0 1.0 0.0 WOPR:P1 465 permx"]\n[dataassim.mda]',
        )
    )
    (pet_case.parent / "data.csv").write_text("days,WOPR:P1\n100,10.0\n465,9.0\n")
    (pet_case.parent / "var.csv").write_text(
        "days,WOPR:P1\n100,\"['abs', 1]\"\n465,\"['abs', 1]\"\n"
    )
    result = _result(pet_case, grid_file=False)
    config = tomllib.loads(result.config.read_text())
    assert config["method"]["localization"]["options"]["entries"][0]["time"] == "2023-06-23"
    rows = (result.config.parent / "observations.csv").read_text().splitlines()
    assert [r.split(",")[0] for r in rows[1:]] == ["2022-06-23", "2023-06-23"]


def test_the_cli_passes_the_grid_file_and_the_angle_check(pet_case: Path) -> None:
    from typer.testing import CliRunner

    from apex.cli.main import app

    out = pet_case.parent.parent / "apex"
    grid = pet_case.parent / "REF.EGRID"
    command = ["pet-import", str(pet_case), str(out), "--grid-file", str(grid)]
    refused = CliRunner().invoke(app, command)
    assert refused.exit_code == 1
    assert "--avo-angles-checked" in refused.output
    result = CliRunner().invoke(app, [*command, "--avo-angles-checked"])
    assert result.exit_code == 0, result.output
    assert "simulator.avo.dz" in result.output
    written = tomllib.loads((out / "config.toml").read_text())
    assert written["geophysics"]["grid_file"] == "REF.EGRID"


def test_a_fixed_baseline_makes_differences_and_drops_its_own_vintage(pet_case: Path) -> None:
    # SimulatorWrap's data with a baseline are each vintage minus it (G10: it cannot run
    # without one); its vintage at the baseline is zero.
    _edit("[100.0, 465.0]", '[0.0, 100.0, 465.0]], ["baseline", 0.0')(pet_case)
    np.savez(pet_case.parent / "avo_0.npz", avo=np.zeros(70))
    data = pet_case.parent / "data.csv"
    data.write_text(
        data.read_text().replace("days,WOPR:P1,avo\n", "days,WOPR:P1,avo\n0,,avo_0.npz\n")
    )
    var = pet_case.parent / "var.csv"
    var.write_text(
        var.read_text().replace("days,WOPR:P1,avo\n", "days,WOPR:P1,avo\n0,,\"['abs', 1]\"\n")
    )

    result = _result(pet_case)

    options = load_config(result.config).geophysics.data[0].avo()  # type: ignore[union-attr]
    assert (options.four_d, options.baseline, options.baseline_mode) == (
        "differences", "2022-03-15", "fixed",
    )  # fmt: skip
    assert any("zero by construction" in note for note in result.notes)
    _check_values(result.config, _mask(step_x=2), ANGLES)


def _swap_xy(egrid: Path) -> None:
    """Rewrite an EGRID with x and y swapped: x then runs along J."""
    from resdata.resfile import FortIO, ResdataFile, openFortIO

    original = ResdataFile(str(egrid))
    keywords = [original.iget_kw(n).copy() for n in range(len(original))]
    del original
    for kw in keywords:
        if kw.get_name() == "COORD":
            values = np.array(kw.numpy_view(), dtype=np.float64).reshape(-1, 6)
            values[:, [0, 1, 3, 4]] = values[:, [1, 0, 4, 3]]
            for n, v in enumerate(values.ravel()):
                kw[n] = float(v)
    with openFortIO(str(egrid), FortIO.WRITE_MODE) as f:
        for kw in keywords:
            kw.fwrite(f)


def test_on_a_grid_whose_x_runs_along_j_pets_seeds_are_transposed(pet_case: Path) -> None:
    # Centres x = 50 + 100 J, y = 50 + 100 I. SimulatorWrap's median test finds x along its
    # first axis (J), so it seeds completion (I, J) at column (J, I): (1, 1) and (2, 1)
    # become (1, 1) and (1, 2), i.e. x 50 to 150 at y = 50. A rectangle without margin
    # around them keeps (I, J) = (0, 0) and (0, 1).
    _swap_xy(pet_case.parent / "REF.EGRID")
    _single_angle(pet_case)
    _edit('["roi_mode", "none"]', '["roi_mode", "rectangle"], ["roi_margin_m", 0]')(pet_case)
    roi = np.zeros((NY, NX), dtype=bool)
    roi[:2, 0] = True  # (NY, NX): J = 0, 1 at I = 0
    _write_data(pet_case.parent, _mask(roi=roi), n_angle=1)
    result = _result(pet_case)
    assert any("x runs along J" in note for note in result.notes)
    assert _check_values(result.config, _mask(roi=roi), (20.0,)) == [(0, 0), (0, 1)]


def test_given_ranges_need_no_completions(pet_case: Path) -> None:
    _single_angle(pet_case)
    ranges = '["roi_mode", "rectangle"], ["x_range", [0, 120]], ["y_range", [0, 60]]'
    _edit('["roi_mode", "none"]', ranges + ', ["roi_margin_m", 0]')(pet_case)
    template = pet_case.parent / "RUNFILE.mako"
    template.write_text(template.read_text().replace("'I1' 2 1", "'I1' ${i} 1"))
    roi = np.zeros((NY, NX), dtype=bool)
    roi[0, 0] = True
    _write_data(pet_case.parent, _mask(roi=roi), n_angle=1)
    assert _check_values(_import(pet_case), _mask(roi=roi), (20.0,)) == [(0, 0)]


def test_region_files_and_margins_simulatorwrap_refuses_are_refused(pet_case: Path) -> None:
    _single_angle(pet_case)
    _edit('["roi_mode", "none"]', '["roi_mode", "file"], ["roi_mask_file", "roi.npy"]')(pet_case)
    np.save(pet_case.parent / "roi.npy", np.ones((NY, NX)))
    with pytest.raises(ConfigError, match=r"must be an \.npz"):
        _import(pet_case)
    _edit('["roi_mode", "file"], ["roi_mask_file", "roi.npy"]', '["roi_margin_m", -1]')(pet_case)
    with pytest.raises(ConfigError, match="roi_margin_m must be at least 0"):
        _import(pet_case)


def test_a_cell_at_a_baseline_that_is_no_vintage_is_refused(pet_case: Path) -> None:
    _edit("[100.0, 465.0]", '[100.0, 465.0]], ["baseline", 0.0')(pet_case)
    data = pet_case.parent / "data.csv"
    data.write_text(data.read_text() + "0,,avo_1.npz\n")
    var = pet_case.parent / "var.csv"
    var.write_text(var.read_text() + "0,,\"['abs', 1]\"\n")
    with pytest.raises(ConfigError, match="only at the pem vintages"):
        _import(pet_case)


def test_start_dates_apex_cannot_use(pet_case: Path) -> None:
    # Outside AVO, a startdate that is not day/month/year leaves PET's labels as they are.
    text = pet_case.read_text()
    pet_case.write_text(text[: text.index("pem = [")].replace('"15/3/2022"', '"2022-03-15"'))
    (pet_case.parent / "data.csv").write_text("days,WOPR:P1\n100,10.0\n")
    (pet_case.parent / "var.csv").write_text("days,WOPR:P1\n100,\"['abs', 1]\"\n")
    result = _result(pet_case, grid_file=False)
    assert any("labels are kept" in note for note in result.notes)
    rows = (result.config.parent / "observations.csv").read_text().splitlines()
    assert rows[1].split(",")[0] == "100"

    # Days after startdate on a deck that starts on another date would match no report.
    pet_case.write_text(pet_case.read_text().replace('"2022-03-15"', '"15/3/2022"'))
    template = pet_case.parent / "RUNFILE.mako"
    template.write_text("RUNSPEC\nSTART\n 1 'JAN' 2022 /\n" + template.read_text())
    with pytest.raises(ConfigError, match="the deck starts on 2022-01-01"):
        _result(pet_case, grid_file=False)


def test_pems_dimension_and_a_baseline_mode_without_baseline(pet_case: Path) -> None:
    _edit('["ntg", "no"]', '["dimension", [3, 4, 2]]')(pet_case)
    with pytest.raises(
        ConfigError, match=r"pem dimension \[3, 4, 2\] \(NX, NY \[3, 4\]\) is not grid_file's"
    ):
        _import(pet_case)
    _edit('["dimension", [3, 4, 2]]', '["dimension", [4, 3, 2]]')(pet_case)
    _edit('["roi_mode", "none"]', '["roi_mode", "none"], ["baseline_mode", "rolling"]')(pet_case)
    result = _result(pet_case)
    assert any("without a pem baseline" in note for note in result.notes)


def test_fwdsim_is_read_before_simulator_as_pet_reads_them(pet_case: Path) -> None:
    text = pet_case.read_text().replace("[simulator]", "[fwdsim]")
    pet_case.write_text(text + '\n[simulator]\nrunfile = "OTHER"\n')
    result = _result(pet_case)
    config = tomllib.loads(result.config.read_text())
    assert config["geophysics"]["grid_file"] == "REF.EGRID"  # the [fwdsim] AVO sections
    assert "fwdsim.avo.dz" in result.ignored


def test_a_comment_does_not_hide_the_decks_start(pet_case: Path) -> None:
    template = pet_case.parent / "RUNFILE.mako"
    template.write_text("RUNSPEC\nSTART\n-- the start\n 1 'JAN' 2022 /\n" + template.read_text())
    with pytest.raises(ConfigError, match="the deck starts on 2022-01-01"):
        _import(pet_case)


def test_soft_sand_keeps_the_references_co2_correlation(pet_case: Path) -> None:
    """softsandrp's CO₂ is Batzle & Wang's: imported as such, not APEX's default
    equation of state."""
    _edit("standardrp elasticproperties", "softsandrp elasticproperties")(pet_case)
    result = _result(pet_case)
    pem = tomllib.loads(result.config.read_text())["geophysics"]["pem"]
    assert pem == {
        "name": "soft_sand",
        "options": {"gas_model": "batzle_wang", "extrapolate_gas_correlation": True},
    }
    assert any("Span & Wagner" in n for n in result.notes)


@pytest.mark.parametrize("mode", ["fixed", "rolling"])
def test_a_baseline_imports_its_mode(pet_case: Path, mode: str) -> None:
    """PET's baseline_mode: fixed, or rolling, each vintage minus the one
    before (the first minus the baseline), as SimulatorWrap's flow_avo takes them."""
    _edit('["ntg", "no"]', '["baseline", 0.0]')(pet_case)
    _edit('["roi_mode", "none"]', f'["roi_mode", "none"], ["baseline_mode", "{mode}"]')(pet_case)
    config = tomllib.loads(_result(pet_case).config.read_text())
    (datum,) = config["geophysics"]["data"]
    assert datum["options"]["four_d"] == "differences"
    assert datum["options"]["baseline_mode"] == mode


def test_a_rolling_import_keeps_a_vintage_without_data(pet_case: Path) -> None:
    """A regression (the review of phase 4, step B): SimulatorWrap's rolling reference moves
    on at every pem vintage, data or not, so a vintage without data stays, in the listed
    order; dropping it took the later datum from the wrong survey."""
    _edit('["ntg", "no"]', '["baseline", 0.0]')(pet_case)
    _edit('["roi_mode", "none"]', '["roi_mode", "none"], ["baseline_mode", "rolling"]')(pet_case)
    _edit("[100.0, 465.0]", "[100.0, 200.0, 465.0]")(pet_case)
    result = _result(pet_case)
    config = tomllib.loads(result.config.read_text())
    (datum,) = config["geophysics"]["data"]
    assert len(datum["vintages"]) == 3
    assert datum["vintages"] == sorted(datum["vintages"])
    assert any("have no data but are kept" in note for note in result.notes)
    assert build_geophysics(load_config(result.config)) is not None  # it composes
    _edit("[100.0, 200.0, 465.0]", "[200.0, 100.0, 465.0]")(pet_case)
    with pytest.raises(ConfigError, match="list them in increasing order"):
        _import(pet_case)
