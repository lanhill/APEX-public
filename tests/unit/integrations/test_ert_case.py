"""An ERT case as an APEX run, without ERT: the parsed case is given."""

import csv
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError, SimulationError
from apex.core.simulation import MemberInput
from apex.integrations.ert.case import (
    ErtCaseInfo,
    ErtField,
    ErtObservation,
    ErtParameter,
    run_directory,
    write_apex_case,
)
from apex.integrations.ert.render import render
from apex.services.composition import load_config
from apex.simulation.ert import MEMBER_FILE, PARAMETERS_FILE, ErtCase
from apex.simulation.grdecl import read_grdecl, write_grdecl


def _case(tmp_path: Path, realizations: int = 4, seed: int | None = 7) -> ErtCaseInfo:
    tmp_path.mkdir(exist_ok=True)
    case_file = tmp_path / "case.ert"
    case_file.write_text("ECLBASE CASE\nFORWARD_MODEL FLOW\n")
    rng = np.random.default_rng(0)
    (tmp_path / "prior").mkdir()
    for i in range(realizations):
        write_grdecl(tmp_path / "prior" / f"PERMX_{i}.grdecl", "PERMX", rng.random(6))
    return ErtCaseInfo(
        case_file=case_file,
        realizations=realizations,
        eclbase="CASE",
        fields=[
            ErtField("PERMX", (3, 2, 1), "permx.grdecl", "prior/PERMX_<IENS>.grdecl", 0.0, 2.0)
        ],
        observations=[
            ErtObservation("WOPR:P1", "2020-01-31T00:00:00", 100.25, 3.0),
            ErtObservation("FOPR", "2020-02-29T00:00:00", 7.0, 0.1),
        ],
        max_running=3,
        seed=seed,
    )


def _apex_toml(tmp_path: Path, text: str = "") -> Path:
    path = tmp_path / "apex.toml"
    path.write_text('[method]\nname = "esmda"\noptions = { inflation = [2, 2] }\n' + text)
    return path


def test_the_case_becomes_a_complete_apex_configuration(tmp_path: Path) -> None:
    info = _case(tmp_path)

    prepared = write_apex_case(info, _apex_toml(tmp_path))

    assert prepared.run_dir == tmp_path / "apex_runs" / "case_esmda"
    assert run_directory(info, tmp_path / "apex.toml") == prepared.run_dir
    config = load_config(prepared.config)
    assert config.run.seed == 7
    assert config.run.output_dir == prepared.run_dir
    (variable,) = config.state.variables
    assert (variable.name, variable.size, variable.grid) == ("permx", 6, [3, 2, 1])
    assert (variable.lower, variable.upper) == (0.0, 2.0)
    assert config.simulator.name == "ert"
    assert config.simulator.options["fields"] == {
        "permx": {"keyword": "PERMX", "file": "permx.grdecl"}
    }
    assert config.execution.name == "ert"
    assert config.execution.options == {"workers": 3}
    assert config.case.kind == "real"
    prior = np.load(prepared.config.parent / "prior.npz")["permx"]
    for i in range(4):  # exact: the INIT_FILES are read as written
        np.testing.assert_array_equal(
            prior[:, i], read_grdecl(tmp_path / "prior" / f"PERMX_{i}.grdecl", "PERMX")
        )
    with (prepared.config.parent / "observations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {
        "time": "2020-01-31",
        "datatype": "WOPR:P1",
        "value": "100.25",
        "variance": "9.0",
    }
    assert float(rows[1]["variance"]) == 0.1**2


def test_apex_toml_chooses_the_name_seed_prior_and_executor_options(tmp_path: Path) -> None:
    apex_toml = _apex_toml(
        tmp_path,
        '[run]\nname = "mine"\nseed = 3\n[execution]\noptions = { workers = 1, retries = 2 }\n'
        '[prior]\ngenerator = { name = "gaussian", options = { mean = 1.0, std = 0.1 } }\n',
    )

    prepared = write_apex_case(_case(tmp_path), apex_toml)

    config = tomllib.loads(prepared.config.read_text())
    assert prepared.run_dir.name == "mine"
    assert config["run"]["seed"] == 3
    assert config["execution"] == {"name": "ert", "options": {"workers": 1, "retries": 2}}
    assert config["prior"]["generator"]["name"] == "gaussian"
    assert config["prior"]["ensemble_size"] == 4
    assert not (prepared.config.parent / "prior.npz").exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[simulator]\nname = "opm_flow"\n', r"\[simulator\] come from the ERT case"),
        ('[execution]\nname = "process"\n', "runs on the 'ert' executor"),
    ],
)
def test_apex_toml_may_not_redefine_the_case(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        write_apex_case(_case(tmp_path), _apex_toml(tmp_path, text))


def test_a_seed_is_needed_and_a_prior_must_be_per_realization(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no RANDOM_SEED"):
        write_apex_case(_case(tmp_path, seed=None), _apex_toml(tmp_path))
    info = _case(tmp_path / "shared")
    shared = ErtCaseInfo(
        **{**info.__dict__, "fields": [ErtField("PERMX", (3, 2, 1), "permx.grdecl", "p.grdecl")]}
    )
    with pytest.raises(ConfigError, match="per-realization INIT_FILES"):
        write_apex_case(shared, _apex_toml(tmp_path / "shared"))


def test_the_simulator_writes_exact_fields_and_the_member_id(tmp_path: Path) -> None:
    case_file = tmp_path / "case.ert"
    case_file.write_text("ECLBASE CASE_<IENS>\n")
    simulator = ErtCase(
        case_file, {"permx": {"keyword": "PERMX", "file": "permx.grdecl"}}, "CASE_<IENS>"
    )
    values = np.array([0.1, 1 / 3, np.pi * 1e-17, 2.5e300])
    run_path = tmp_path / "realization-2"
    run_path.mkdir()

    simulator.write_parameters(MemberInput(41, {"permx": values}, run_path))

    np.testing.assert_array_equal(read_grdecl(run_path / "permx.grdecl", "PERMX"), values)
    assert (run_path / MEMBER_FILE).read_text() == "41\n"
    assert simulator.case_name(2) == "CASE_2"
    with pytest.raises(SimulationError, match="use the 'ert' executor"):
        simulator.run(MemberInput(41, {"permx": values}, run_path), [])


def test_the_deck_is_rendered_for_apexs_member_not_erts_realization(tmp_path: Path) -> None:
    template = tmp_path / "deck.mako"
    template.write_text("-- member ${member_id}\n")
    export = tmp_path / "apex_export.json"
    export.write_text(json.dumps({
        "deck_template_file": str(template),
        "fields": {"PERMX": {"variable": "permx", "file": "permx.grdecl"}},
    }))  # fmt: skip
    write_grdecl(tmp_path / "permx.grdecl", "PERMX", np.ones(2))

    render(export, Path("a.DATA"), realization=2, run_path=tmp_path)
    (tmp_path / MEMBER_FILE).write_text("41\n")
    render(export, Path("b.DATA"), realization=2, run_path=tmp_path)

    assert (tmp_path / "a.DATA").read_text() == "-- member 2\n"
    assert (tmp_path / "b.DATA").read_text() == "-- member 41\n"


def test_fields_apex_cannot_write_exactly_are_refused(tmp_path: Path) -> None:
    """A FIELD in ROFF or binary GRDECL would get an ASCII GRDECL file under its name."""
    info = _case(tmp_path)
    roff = ErtCaseInfo(
        **{**info.__dict__, "fields": [ErtField("PERMX", (3, 2, 1), "permx.roff",
                                                 "prior/PERMX_<IENS>.roff", file_format="roff")]}
    )  # fmt: skip

    with pytest.raises(ConfigError, match=r"PERMX \(roff\)"):
        write_apex_case(roff, _apex_toml(tmp_path))


def _written(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "apex.toml"
    path.write_text(text)
    return path


PGS = """
[method]
name = "esmda"
options = { inflation = [2, 2] }

[[state.variables]]
name = "y1"
size = 6
grid = [3, 2, 1]

[prior]
ensemble_size = 4
generator = { name = "gaussian", options = { mean = 0.0, std = 1.0 } }

[[parameterizations]]
name = "test_affine"
"""


def test_with_a_parameterization_the_state_is_apex_tomls_and_fields_are_properties(
    tmp_path: Path,
) -> None:
    info = _case(tmp_path)

    prepared = write_apex_case(info, _written(tmp_path, '[run]\nname = "pgs"\n' + PGS))

    config = tomllib.loads(prepared.config.read_text())
    assert config["state"]["variables"][0]["name"] == "y1"  # apex.toml's latent state
    assert config["simulator"]["options"]["fields"] == {
        "permx": {"keyword": "PERMX", "file": "permx.grdecl"}  # the property it maps to
    }
    assert config["parameterizations"] == [{"name": "test_affine"}]
    assert not (prepared.config.parent / "prior.npz").exists()  # no INIT_FILES prior


def test_a_parameterization_needs_apex_tomls_state_and_prior(tmp_path: Path) -> None:
    without_prior = PGS.split("[prior]")[0] + '\n[[parameterizations]]\nname = "test_affine"\n'

    with pytest.raises(ConfigError, match=r"give the state they map \(\[state\]\) and its prior"):
        write_apex_case(_case(tmp_path / "a"), _written(tmp_path, without_prior))
    with pytest.raises(ConfigError, match=r"\[state\] come from the ERT case"):
        write_apex_case(
            _case(tmp_path / "b"), _written(tmp_path, PGS.split("[[parameterizations]]")[0])
        )


def _with(info: ErtCaseInfo, *parameters: ErtParameter) -> ErtCaseInfo:
    return ErtCaseInfo(**{**info.__dict__, "parameters": list(parameters)})


def test_gen_kw_and_surfaces_are_state_variables_written_by_the_parameter_step(
    tmp_path: Path,
) -> None:
    latent = np.random.default_rng(2).standard_normal((2, 4))
    top = np.random.default_rng(3).random((6, 4))
    info = _with(
        _case(tmp_path),
        ErtParameter("COEF", "gen_kw", 2, prior=latent),
        ErtParameter("TOP", "surface", 6, prior=top),
    )

    prepared = write_apex_case(info, _apex_toml(tmp_path))

    config = tomllib.loads(prepared.config.read_text())
    variables = {v["name"]: v for v in config["state"]["variables"]}
    assert (variables["coef"]["size"], variables["top"]["size"]) == (2, 6)
    assert "lower" not in variables["coef"]  # latent space: no bounds
    options = config["simulator"]["options"]
    assert options["step_parameters"] == {"coef": "COEF", "top": "TOP"}
    assert options["parameter_step"] == f"APEX_WRITE_PARAMETERS(<CASE>={info.case_file})"
    with np.load(prepared.config.parent / "prior.npz") as prior:
        np.testing.assert_array_equal(prior["coef"], latent)  # ERT's own prior draws
        np.testing.assert_array_equal(prior["top"], top)


def test_gen_kw_needs_a_seed_for_erts_sampler(tmp_path: Path) -> None:
    info = _with(_case(tmp_path), ErtParameter("COEF", "gen_kw", 2, prior=None))

    with pytest.raises(ConfigError, match="RANDOM_SEED"):
        write_apex_case(info, _apex_toml(tmp_path))


def test_forward_init_parameters_are_drawn_by_their_generating_steps(tmp_path: Path) -> None:
    generated = ErtParameter("SAT", "field", 6, (3, 2, 1), forward_init=True)
    info = _with(_case(tmp_path), generated)

    with pytest.raises(ConfigError, match=r"\[ert\] forward_init_steps"):
        write_apex_case(info, _apex_toml(tmp_path))
    prepared = write_apex_case(info, _apex_toml(tmp_path, "[ert]\nforward_init_steps = 2\n"))

    config = tomllib.loads(prepared.config.read_text())
    group = config["prior"]["groups"]["forward_init"]
    assert group["name"] == "ert_forward_init"
    assert group["variables"] == ["sat"]
    assert group["options"]["steps"] == 2
    assert config["simulator"]["options"]["skip_forward_steps"] == 2
    with np.load(prepared.config.parent / "prior.npz") as prior:
        assert set(prior.files) == {"permx"}  # the others come from the file


def test_the_batch_runs_the_parameter_step_first_and_skips_generating_steps(
    tmp_path: Path,
) -> None:
    case_file = tmp_path / "case.ert"
    case_file.write_text(
        "ECLBASE CASE\nFIELD X PARAMETER x.grdecl INIT_FILES:x_%d.grdecl\n"
        "FORWARD_MODEL MAKE_SAT\nFORWARD_MODEL FLOW\n"
    )
    simulator = ErtCase(
        case_file, {}, "CASE", step_parameters={"coef": "COEF"},
        parameter_step="APEX_WRITE_PARAMETERS(<CASE>=/c.ert)", skip_forward_steps=1,
    )  # fmt: skip
    run_path = tmp_path / "realization-0"
    run_path.mkdir()

    lines = simulator.forward_config()
    simulator.write_parameters(MemberInput(7, {"coef": np.array([0.25, -1 / 3])}, run_path))

    assert lines == [
        "ECLBASE CASE", "FORWARD_MODEL APEX_WRITE_PARAMETERS(<CASE>=/c.ert)", "FORWARD_MODEL FLOW"
    ]  # fmt: skip
    with np.load(run_path / PARAMETERS_FILE) as written:
        np.testing.assert_array_equal(written["COEF"], [0.25, -1 / 3])  # exact
    with pytest.raises(ValueError, match="parameter_step"):
        ErtCase(case_file, {}, "CASE", step_parameters={"coef": "COEF"})
