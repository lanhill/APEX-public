import tomllib
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services.composition import load_config
from apex.storage.pet_config import import_pet_case

ESMDA = """
[ensemble]
ne = 4
state = "permx"
importstate = "prior.npz"
[ensemble.prior_permx]
limits = [0.0, 8.0]

[dataassim]
scheme = "esmda"
energy = 98.0
data = "data.csv"
datavar = "var.csv"
savefolder = "Results"
[dataassim.mda]
tot_assim_steps = 3

[simulator]
runfile = "RUNFILE"
parallel = 2
sim_limit = 60.0
"""

GIES = """
[ensemble]
ne = 4
state = ["permx"]
importstate = "prior.npz"

[dataassim]
daalg = ["gies", "rlmmac"]
data = "data.csv"
datavar = "var.csv"
iteration = [["lambda", 10.0], ["lambda_factor", 4], ["data_misfit_tol", 0.05], ["max_iter", 7]]

[simulator]
runfile = "RUNFILE"
"""


def _pet_case(tmp_path: Path, config: str) -> Path:
    case = tmp_path / "pet"
    (case / "grid").mkdir(parents=True)
    (case / "grid" / "Grid.grdecl").write_text("-- grid\n")
    (case / "RUNFILE.mako").write_text("INCLUDE\n'../grid/Grid.grdecl' /\n${permx[0]}\n")
    (case / "data.csv").write_text("dates,WOPR:P1\n2024-01-01,100.0\n2025-01-01,90.0\n")
    (case / "var.csv").write_text(
        "dates,WOPR:P1\n2024-01-01,\"['rel', 10]\"\n2025-01-01,\"['abs', 4]\"\n"
    )
    np.savez(case / "prior.npz", permx=np.zeros((6, 4)))
    (case / "CONFIG.toml").write_text(config)
    return case / "CONFIG.toml"


_GIES_ITERATION = GIES.split("iteration = ", 1)[1].split("\n", 1)[0]
LMENRML = GIES.replace('daalg = ["gies", "rlmmac"]', 'scheme = "lmenrml"').replace(
    _GIES_ITERATION,
    '[["lambda", "auto"], ["lambda_max", 1e6], ["max_inner_iter", 4], ["energy", 90], '
    '["max_iter", 12]]',
)


def test_lm_enrml_config_maps_the_iteration_block(tmp_path: Path) -> None:
    result = import_pet_case(_pet_case(tmp_path, LMENRML), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    assert method == {
        "name": "lm_enrml",
        "options": {"max_inner_iterations": 4},
        # PET reads the energy from the iteration block, in percent there.
        "strategy": {"name": "approx", "options": {"energy": 0.9}},
        "schedule": {"initial": "auto", "maximum": 1e6},
        "max_iterations": 12,
    }
    assert load_config(result.config).method.name == "lm_enrml"
    full = import_pet_case(
        _pet_case(
            tmp_path / "full",
            LMENRML.replace('scheme = "lmenrml"', 'scheme = "lmenrml"\nanalysis = "full"'),
        ),
        tmp_path / "apex_full",
    )
    assert tomllib.loads(full.config.read_text())["method"]["strategy"]["name"] == "full"
    # Without a configured variance, the full update falls back to the ensemble spread.
    assert any("gives no variance" in note for note in full.notes)
    assert not (tmp_path / "apex_full" / "state_scaling.npz").exists()


def test_lm_enrml_full_analysis_gets_pets_state_scaling(tmp_path: Path) -> None:
    config = LMENRML.replace('scheme = "lmenrml"', 'scheme = "lmenrml"\nanalysis = "full"').replace(
        'importstate = "prior.npz"', 'importstate = "prior.npz"\n[ensemble.prior_permx]\nvar = 4.0'
    )
    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    options = tomllib.loads(result.config.read_text())["method"]["strategy"]["options"]
    assert options["state_scaling_file"] == "state_scaling.npz"
    with np.load(tmp_path / "apex" / "state_scaling.npz") as data:
        # PET's calc_scaling: sqrt of the configured variance, per cell.
        np.testing.assert_array_equal(data["scaling"], np.full(6, 2.0))
    # The relative path resolves against the configuration's directory.
    strategy = load_config(result.config).method.strategy
    assert strategy is not None
    assert Path(str(strategy.options["state_scaling_file"])) == (
        tmp_path / "apex" / "state_scaling.npz"
    )


def test_gn_enrml_config_maps_the_step_length_block(tmp_path: Path) -> None:
    config = LMENRML.replace(
        'scheme = "lmenrml"', 'scheme = "gnenrml"\nanalysis = "subspace"'
    ).replace(
        '[["lambda", "auto"], ["lambda_max", 1e6], ',
        '[["gamma", "auto"], ["gamma_max", 0.8], ["gamma_factor", 3], ',
    )
    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    assert method == {
        "name": "gn_enrml",
        "options": {"max_inner_iterations": 4},
        "strategy": {"name": "subspace", "options": {"energy": 0.9}},
        "schedule": {"initial": "auto", "maximum": 0.8, "factor": 3.0},
        "max_iterations": 12,
    }
    assert load_config(result.config).method.name == "gn_enrml"
    # PET's legacy gn_enrml is GN-EnRML with the subspace analysis.
    legacy = import_pet_case(
        _pet_case(
            tmp_path / "legacy", LMENRML.replace('scheme = "lmenrml"', 'scheme = "gn_enrml"')
        ),
        tmp_path / "apex_legacy",
    )
    legacy_method = tomllib.loads(legacy.config.read_text())["method"]
    assert (legacy_method["name"], legacy_method["strategy"]["name"]) == ("gn_enrml", "subspace")


def test_es_config_maps_to_es(tmp_path: Path) -> None:
    config = ESMDA.replace('scheme = "esmda"', 'scheme = "es"\nanalysis = "full"')
    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    # PET's ES runs "full" as approx (one step: no prior-increment term).
    assert method == {"name": "es", "strategy": {"name": "approx", "options": {"energy": 0.98}}}
    assert load_config(result.config).method.name == "es"


def test_esmda_config_becomes_a_valid_self_contained_apex_case(tmp_path: Path) -> None:
    result = import_pet_case(_pet_case(tmp_path, ESMDA), tmp_path / "apex", seed=5)

    config = load_config(result.config)
    assert config.method.name == "esmda"
    assert config.method.options == {"inflation": [3.0, 3.0, 3.0]}
    assert config.method.strategy is not None
    assert config.method.strategy.options == {"energy": 0.98}
    assert config.state.variables[0].size == 6
    assert (config.state.variables[0].lower, config.state.variables[0].upper) == (0.0, 8.0)
    assert config.execution.options == {"workers": 2}
    assert config.simulator.options["timeout_s"] == 60.0
    assert config.run.seed == 5
    assert "'${case_dir}/grid/Grid.grdecl'" in (tmp_path / "apex" / "deck.mako").read_text()
    assert (tmp_path / "apex" / "grid" / "Grid.grdecl").is_file()
    assert result.ignored == ["dataassim.savefolder"]


def test_legacy_gies_config_maps_the_damping_schedule(tmp_path: Path) -> None:
    result = import_pet_case(_pet_case(tmp_path, GIES), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    assert method == {
        "name": "gies",
        "strategy": {"name": "rlmmac", "options": {"energy": 0.95}},
        "schedule": {
            "initial": 10.0, "factor": 4.0, "data_mismatch_tolerance": 0.05, "reduction": 0.25,
        },
        "max_iterations": 7,
    }  # fmt: skip


def test_an_imported_gies_case_keeps_pet_forks_damping_reduction(tmp_path: Path) -> None:
    """PET-fork divides λ by lambda_factor after an accepted step; the import writes
    reduction = 1/lambda_factor, 0.5 at its default factor (the user, 2026-10-14), not
    APEX's default 0.9."""
    without_factor = GIES.replace('["lambda_factor", 4], ', "")
    assert without_factor != GIES
    result = import_pet_case(_pet_case(tmp_path, without_factor), tmp_path / "apex")

    schedule = tomllib.loads(result.config.read_text())["method"]["schedule"]
    assert "factor" not in schedule
    assert schedule["reduction"] == 0.5
    load_config(result.config)  # a valid configuration


@pytest.mark.parametrize("factor", ["0", "1", "-2", "0.5"])
def test_a_gies_lambda_factor_of_one_or_less_is_refused(tmp_path: Path, factor: str) -> None:
    case = GIES.replace('["lambda_factor", 4]', f'["lambda_factor", {factor}]')
    assert case != GIES

    with pytest.raises(ConfigError, match="lambda_factor must be greater than 1"):
        import_pet_case(_pet_case(tmp_path, case), tmp_path / "apex")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda c: c.replace('scheme = "esmda"', 'scheme = "esmda"\nlocalization = "x"'),
            "localization",
        ),
        (lambda c: c.replace('scheme = "esmda"', 'scheme = "enkf"'), "not ported"),
        (
            lambda c: c.replace('scheme = "esmda"', 'scheme = "lmenrml"\nanalysis = "margis"'),
            "approx and full and subspace and subspace2 analyses only",
        ),
        (
            lambda c: c.replace('scheme = "esmda"', 'scheme = "gn_enrml"\nanalysis = "approx"'),
            "subspace analyses only",
        ),
        (lambda c: c.replace('scheme = "esmda"', 'scheme = "esmda"\nanalysis = "full"'), "approx"),
        (lambda c: c.replace('importstate = "prior.npz"\n', ""), "needs `vario` and `grid`"),
    ],
)
def test_unsupported_configurations_are_refused(tmp_path: Path, edit: object, message: str) -> None:
    config = _pet_case(tmp_path, edit(ESMDA))  # type: ignore[operator]

    with pytest.raises(ConfigError, match=message):
        import_pet_case(config, tmp_path / "apex")


VARIO = """
[ensemble]
ne = 30
state = "permx"
[ensemble.prior_permx]
vario = "exp"
mean = "priormean.npz"
var = [1.0, 2.0]
range = 4.0
aniso = 2.0
angle = 30.0
grid = [6, 4, 2]
limits = [-5.0, 9.0]

[dataassim]
scheme = "esmda"
data = "data.csv"
datavar = "var.csv"
[dataassim.mda]
tot_assim_steps = 2

[simulator]
runfile = "RUNFILE"
"""


def test_variogram_prior_becomes_a_gaussian_field(tmp_path: Path) -> None:
    config_path = _pet_case(tmp_path, VARIO)
    np.savez(config_path.parent / "priormean.npz", PriorMean=np.arange(48.0))

    result = import_pet_case(config_path, tmp_path / "apex")

    prior = tomllib.loads(result.config.read_text())["prior"]
    field = prior["variables"]["permx"]
    assert prior["ensemble_size"] == 30
    assert field["name"] == "gaussian_field"
    # A non-square grid: PET's field cannot be reproduced, so the intended geometry.
    assert field["options"] == {
        "grid": [6, 4, 2],
        "variance": [1.0, 2.0],
        "variogram": ["stable", "stable"],
        "exponent": [1.9, 1.9],  # PET's "exp" is exp(-3 u^1.9)
        "minor_range": [4.0, 4.0],
        "major_range": [8.0, 8.0],  # PET: aniso * range along the major axis
        "azimuth": [-30.0, -30.0],  # PET's angle is clockwise
        "mean_file": "permx_prior_mean.npz",
    }
    assert (tmp_path / "apex" / "permx_prior_mean.npz").is_file()
    assert any("scrambles" in note for note in result.notes)
    config = load_config(result.config)
    assert config.state.variables[0].size == 48
    assert (config.state.variables[0].lower, config.state.variables[0].upper) == (-5.0, 9.0)


SQUARE = VARIO.replace('mean = "priormean.npz"', "mean = [1.0, 2.0]").replace(
    "grid = [6, 4, 2]", "grid = [3, 3, 2]"
)


def test_square_grids_reproduce_what_pet_samples(tmp_path: Path) -> None:
    config = SQUARE.replace("var = [1.0, 2.0]", "variance = [1.0, 2.0]").replace(
        "range = 4.0", "corr_length = [4.0, 2.0]"
    )  # PET's other spellings

    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    options = tomllib.loads(result.config.read_text())["prior"]["variables"]["permx"]["options"]
    scale = np.cos(np.radians(30.0))
    np.testing.assert_allclose(options["minor_range"], [4.0 / scale, 2.0 / scale])
    np.testing.assert_allclose(options["major_range"], [8.0 / scale, 4.0 / scale])
    assert options["azimuth"] == [90.0, 90.0]  # PET's major axis runs along j
    assert options["variance"] == [1.0, 2.0]
    assert options["mean"] == [1.0] * 9 + [2.0] * 9  # one mean per layer
    assert any("does not rotate" in note for note in result.notes)
    assert any("y-fastest" in note for note in result.notes)

    intended = import_pet_case(
        _pet_case(tmp_path / "b", config), tmp_path / "b" / "apex", pet_geometry=False
    )
    options = tomllib.loads(intended.config.read_text())["prior"]["variables"]["permx"]["options"]
    assert (options["minor_range"], options["azimuth"]) == ([4.0, 2.0], [-30.0, -30.0])
    assert intended.notes == []


def test_inline_per_cell_means_are_kept(tmp_path: Path) -> None:
    means = ", ".join(str(float(m)) for m in range(18))

    result = import_pet_case(
        _pet_case(tmp_path, SQUARE.replace("mean = [1.0, 2.0]", f"mean = [{means}]")),
        tmp_path / "apex",
    )

    options = tomllib.loads(result.config.read_text())["prior"]["variables"]["permx"]["options"]
    assert options["mean"] == [float(m) for m in range(18)]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.replace("range = 4.0\n", ""), "needs `range`"),
        (lambda c: c.replace("mean = [1.0, 2.0]", "mean = [1.0, 2.0, 3.0]"), "mean has 3"),
        (lambda c: c.replace("var = [1.0, 2.0]", "var = [1.0, 2.0, 3.0]"), "`var` has 3"),
        (lambda c: c.replace("angle = 30.0", "angle = 90.0"), "degenerate"),
        (
            lambda c: c.replace("limits = [-5.0, 9.0]", "limits = [[0.0, 1.0], [0.0, 2.0]]"),
            "one \\[lower, upper\\] pair",
        ),
    ],
)
def test_bad_variogram_priors_are_refused(tmp_path: Path, edit: object, message: str) -> None:
    config = _pet_case(tmp_path, edit(SQUARE))  # type: ignore[operator]

    with pytest.raises(ConfigError, match=message):
        import_pet_case(config, tmp_path / "apex")


def test_uncompressed_fields_need_an_absolute_variance(tmp_path: Path) -> None:
    config = _pet_case(tmp_path, ESMDA)
    np.savez(config.parent / "avo.npz", np.arange(3.0))
    (config.parent / "data.csv").write_text("dates,WOPR:P1,avo\n2024-01-01,100.0,avo.npz\n")
    (config.parent / "var.csv").write_text(
        "dates,WOPR:P1,avo\n2024-01-01,\"['abs', 4]\",\"['rel', 5]\"\n"
    )
    with pytest.raises(ConfigError, match="relative variance"):
        import_pet_case(config, tmp_path / "apex")

    (config.parent / "var.csv").write_text(
        "dates,WOPR:P1,avo\n2024-01-01,\"['abs', 4]\",\"['abs', 0.5]\"\n"
    )
    result = import_pet_case(config, tmp_path / "apex2")
    observations = tomllib.loads(result.config.read_text())["observations"]
    assert observations["fields"] == [
        {
            "time": "2024-01-01",
            "datatype": "avo",
            "file": "fields/avo_2024-01-01.npz",
            "variance": 0.5,
        }
    ]
    np.testing.assert_array_equal(
        np.load(tmp_path / "apex2" / "fields" / "avo_2024-01-01.npz")["values"], [0, 1, 2]
    )
    load_config(result.config)


def test_imported_cases_are_real_unless_a_truth_is_given(tmp_path: Path) -> None:
    config = _pet_case(tmp_path, ESMDA)
    real = tomllib.loads(import_pet_case(config, tmp_path / "real").config.read_text())
    assert real["case"] == {"kind": "real"}

    np.savez(tmp_path / "truth.npz", permx=np.ones(6))
    result = import_pet_case(config, tmp_path / "synthetic", truth_file=tmp_path / "truth.npz")
    synthetic = tomllib.loads(result.config.read_text())
    assert synthetic["case"] == {"kind": "synthetic", "truth_file": "truth.npz"}
    assert (tmp_path / "synthetic" / "truth.npz").is_file()
    assert load_config(result.config).case.kind == "synthetic"
    with pytest.raises(ConfigError, match="not found"):
        import_pet_case(config, tmp_path / "x", truth_file=tmp_path / "absent.npz")


@pytest.mark.parametrize(("scheme", "analysis"), [("gnenrml", "margis"), ("lmenrml", "subspace2")])
def test_the_transform_analyses_import_without_energy(
    tmp_path: Path, scheme: str, analysis: str
) -> None:
    config = LMENRML.replace('scheme = "lmenrml"', f'scheme = "{scheme}"\nanalysis = "{analysis}"')
    if scheme == "gnenrml":
        config = config.replace('["lambda", "auto"], ["lambda_max", 1e6], ', "")
    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    strategy = tomllib.loads(result.config.read_text())["method"]["strategy"]
    assert strategy == {"name": analysis, "options": {}}
    assert load_config(result.config).method.strategy is not None


def test_multilevel_es_mda_maps_the_levels_and_their_coarsening(tmp_path: Path) -> None:
    config = (
        ESMDA.replace('scheme = "esmda"', 'scheme = "esmda"\nanalysis = "hybrid"')
        .replace(
            "[ensemble.prior_permx]",
            "multilevel = [['levels', 2], ['en_size', [2, 2]], ['cov_wgt', [0.2, 0.8]]]\n"
            "[ensemble.prior_permx]",
        )
        .replace(
            'runfile = "RUNFILE"',
            'runfile = "RUNFILE"\ntrunc_level = [0.5, 0.0]\n'
            "upscale = [['state', 'permx', [2, 3]], ['maxdiff', 0.1], ['wells', [1, 1]], "
            "['radius', [1]], ['us_type', 'unif']]",
        )
    )
    result = import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    assert method["name"] == "multilevel_esmda"
    assert method["strategy"] == {"name": "hybrid", "options": {"energy": 0.98}}
    assert method["options"]["levels"] == [
        {
            "members": 2,
            "weight": 0.2,
            "upscale": {
                "method": "uniform",
                "dim": [2, 3],
                "state": "permx",
                "max_trunc": 0.5,
                "max_diff": 0.1,
                "wells": [[1, 1]],
                "radius": [1],
            },
        },
        {"members": 2, "weight": 0.8},
    ]
    assert any("COARSEN" in note for note in result.notes)
    assert load_config(result.config).method.name == "multilevel_esmda"


def test_emp_cov_is_refused(tmp_path: Path) -> None:
    """PET's emp_cov also reads an error ensemble as the data variance: not mappable."""
    config = ESMDA.replace('scheme = "esmda"', 'scheme = "esmda"\nemp_cov = "yes"')

    with pytest.raises(ConfigError, match="emp_cov"):
        import_pet_case(_pet_case(tmp_path, config), tmp_path / "apex")


def _localized(config: str, block: str) -> str:
    return config.replace("[dataassim.mda]", f"{block}\n[dataassim.mda]")


def test_pets_autoadaloc_becomes_adaptive_projected(tmp_path: Path) -> None:
    block = (
        '[dataassim.localization]\nname = "autoadaloc"\nfield = [1, 2, 3]\n'
        'threshold = "fixed"\nnstd = 0.4\ntype = "sigm"'
    )
    result = import_pet_case(_pet_case(tmp_path, _localized(ESMDA, block)), tmp_path / "apex")

    method = tomllib.loads(result.config.read_text())["method"]
    assert method["localization"] == {
        "name": "adaptive_projected",
        "options": {"threshold": "fixed", "cutoff": 0.4, "taper": "sigmoid"},
    }
    config = load_config(result.config)
    assert config.method.localization is not None
    # The default projection, and the cutoff carried as the autoadaloc key's value.
    shorthand = "[dataassim.localization]\nautoadaloc = 2.0\nfield = [1, 2, 3]"
    other = import_pet_case(
        _pet_case(tmp_path / "b", _localized(ESMDA, shorthand)), tmp_path / "apex_b"
    )
    options = tomllib.loads(other.config.read_text())["method"]["localization"]["options"]
    assert options == {"threshold": "adaptive", "cutoff": 2.0, "taper": "hard"}


def test_pets_distance_loc_becomes_distance_entries_in_eclipse_terms(tmp_path: Path) -> None:
    # PET's field [nz, ny, nx] = [1, 2, 3]; its x is Eclipse j and its y is i, 0-based.
    gc = (
        '  { taper = "gc", x = 1, y = 2, z = 0, radius = 3, data_type = "wopr:p1", '
        'time = "*", param = "PERMX", aniso = 2.0, rotation = 30.0 },\n'
    )
    imported = (
        '  { taper = "import", file = "taper.npy", z_range = ":", data_type = "*", '
        'time = "2024-01-01", param = "*" },\n'
    )
    block = """[dataassim.localization]
name = "distance_loc"
field = [1, 2, 3]
actnum = "actnum.npz"
entries = [
{GC}  "fb 0 1 0 2 0 1.0 0.0 WOPR:P1 2025-01-01 permx",
  { taper = "region", x = 0, y = 0, data_type = "*", time = "2030-01-01", param = "*" },
{IMPORT}]
""".replace("{GC}", gc).replace("{IMPORT}", imported)
    config = _pet_case(tmp_path, _localized(ESMDA, block))
    np.savez(config.parent / "actnum.npz", actnum=np.ones(6, dtype=bool))
    np.save(config.parent / "taper.npy", np.linspace(0, 1, 6))

    result = import_pet_case(config, tmp_path / "apex")

    localization = tomllib.loads(result.config.read_text())["method"]["localization"]
    assert localization["name"] == "distance_entries"
    options = localization["options"]
    assert options["grid"] == [3, 2, 1]
    assert options["actnum_file"] == "localization_actnum.npz"
    assert options["entries"][0] == {
        "data": "WOPR:P1", "time": "*", "parameter": "permx", "taper": "gaspari_cohn",
        "z_range": "all", "position": [3, 2, 1], "radius": 3.0, "anisotropy": [2.0, 30.0],
    }  # fmt: skip
    assert options["entries"][1] == {
        "data": "WOPR:P1", "time": "2025-01-01", "parameter": "permx",
        "taper": "furrer_bengtsson", "z_range": 0, "position": [2, 1, 1], "radius": 2.0,
        "anisotropy": [1.0, 0.0],
    }  # fmt: skip
    # The region entry names a time without data: PET ignores it, so it is left out.
    assert [e["taper"] for e in options["entries"]] == [
        "gaspari_cohn",
        "furrer_bengtsson",
        "import",
    ]
    assert any("entry 2" in note for note in result.notes)
    assert options["entries"][2]["taper_file"] == "localization_taper_3.npy"
    assert (tmp_path / "apex" / "localization_taper_3.npy").is_file()
    assert (tmp_path / "apex" / "localization_actnum.npz").is_file()
    assert load_config(result.config).method.localization is not None


@pytest.mark.parametrize(
    ("config", "block", "message"),
    [
        (
            ESMDA,
            '[dataassim.localization]\nname = "autoadaloc"\nprojection = "ensemble"',
            "ensemble",
        ),
        (
            ESMDA,
            '[dataassim.localization]\nname = "autoadaloc"\nini_only = true',
            "unknown keys ini_only",
        ),
        (ESMDA, '[dataassim.localization]\nname = "localanalysis"', "not ported"),
        (
            ESMDA,
            '[dataassim.localization]\nname = "distance_loc"\nfield = [1, 2, 3]',
            "needs `entries`",
        ),
        (
            LMENRML.replace('scheme = "lmenrml"', 'scheme = "lmenrml"\nanalysis = "subspace"'),
            '[dataassim.localization]\nname = "autoadaloc"',
            "approx analysis only",
        ),
        (GIES, '[dataassim.localization]\nname = "autoadaloc"', "PET-fork"),
    ],
)
def test_localizations_apex_cannot_reproduce_are_refused(
    tmp_path: Path, config: str, block: str, message: str
) -> None:
    text = (
        _localized(config, block)
        if "[dataassim.mda]" in config
        else config.replace("[simulator]", f"{block}\n\n[simulator]")
    )

    with pytest.raises(ConfigError, match=message):
        import_pet_case(_pet_case(tmp_path, text), tmp_path / "apex")


def test_a_relative_prior_is_relative_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--prior`` is the caller's path, not the PET configuration's (release tester, 2026-10-19).

    A relative prior was looked up in the configuration's folder; the working directory's
    ensemble, with another shape, must be the one imported.
    """
    config = _pet_case(tmp_path, ESMDA)
    work = tmp_path / "work"
    work.mkdir()
    np.savez(work / "prior.npz", permx=np.ones((9, 4)))  # the PET folder's is (6, 4)
    monkeypatch.chdir(work)

    result = import_pet_case(config, tmp_path / "apex", prior_file=Path("prior.npz"))

    assert load_config(result.config).state.variables[0].size == 9
    with np.load(tmp_path / "apex" / "prior_ensemble.npz") as data:
        assert np.all(data["permx"] == 1.0)
    # The configuration's own importstate stays relative to its folder.
    other = import_pet_case(config, tmp_path / "apex2")
    assert load_config(other.config).state.variables[0].size == 6
