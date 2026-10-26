"""The geomodeler agent: its rules, the model's plans held to the skills
and the facts, the drawn sample's check, and its command, on a case the converter wrote."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.agents.converter import CaseConverter
from apex.agents.geomodeler import Geomodeler, GeomodelPlan, plan_problems
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.core.geomodeling import GeomodelFacts
from apex.core.plugins import geomodeling_skills
from apex.services.api import ApexService
from apex.services.geomodeling_skills import FaciesParameters
from apex.services.settings import CaseConverterSection, GeomodelerSection
from apex.testing import SMALL_DECK_PERMX, SMALL_DECK_PORO

SETTINGS = GeomodelerSection(ensemble_size=20, qc_members=40, max_revisions=1)


def _case(tmp_path: Path, small_deck: Path) -> tuple[ApexService, Path, Path]:
    """The small deck converted, and a reference of its values in state space."""
    service = ApexService(tmp_path / "service")
    CaseConverter(service, None, settings=CaseConverterSection(ensemble_size=20)).convert(
        small_deck, tmp_path / "case"
    )
    reference = tmp_path / "case" / "reference.npz"
    np.savez(reference, permx=np.log(SMALL_DECK_PERMX), poro=np.array(SMALL_DECK_PORO))
    return service, tmp_path / "case" / "config.toml", reference


def _plan(**field: object) -> str:
    base = {"variable": "poro", "mean": 0.15, "std": 0.02, "major_range": 2.0}
    return GeomodelPlan(skill="continuous", parameters={"fields": [{**base, **field}]},
                        reason="as asked").model_dump_json()  # fmt: skip


def test_the_rules_give_fields_around_the_reference(tmp_path: Path, small_deck: Path) -> None:
    service, config, reference = _case(tmp_path, small_deck)

    outcome = Geomodeler(service, None, settings=SETTINGS).model(config, reference=reference)

    assert outcome.valid, (outcome.validation.errors, outcome.qc)
    assert outcome.decided_by == "rules"
    assert outcome.config == config.with_name("config_prior.toml")
    prior = tomllib.loads(outcome.config.read_text())["prior"]
    permx = prior["variables"]["permx"]
    assert permx["name"] == "gaussian_field"
    assert permx["options"]["mean"] == pytest.approx(np.log(SMALL_DECK_PERMX).mean())
    assert permx["options"]["major_range"] == 1.0  # a quarter of 4 cells
    assert any("assumed range, variogram" in n for n in outcome.notes)
    assert list((tmp_path / "service" / "agents").glob("geomodeler-*-decision.json"))


def test_a_prior_the_bounds_clip_goes_back_to_the_model(tmp_path: Path, small_deck: Path) -> None:
    """The bounds clip a field whose mean sits on the lower bound: the QC says so."""
    service, config, _ = _case(tmp_path, small_deck)
    bounds = tomllib.loads(config.read_text())["state"]["variables"][1]
    clipped = _plan(mean=bounds["lower"] + 0.001, std=0.2)
    strict = SETTINGS.model_copy(update={"qc_tolerance": 0.25})

    outcome = Geomodeler(service, StubProvider([clipped, _plan()]), settings=strict).model(
        config, request="porosity around 0.15"
    )

    assert outcome.decided_by == "model"
    assert outcome.valid, outcome.qc
    assert (
        tomllib.loads(outcome.config.read_text())["prior"]["variables"]["poro"]["options"]["mean"]
        == 0.15
    )


def test_plans_are_held_to_the_skills_and_the_facts(tmp_path: Path, small_deck: Path) -> None:
    from apex.agents.geomodeler import geomodel_facts

    _, config, _ = _case(tmp_path, small_deck)
    facts = geomodel_facts(tomllib.loads(config.read_text()), config.parent, None)

    unknown = GeomodelPlan(skill="kriging", parameters={})
    missing = GeomodelPlan.model_validate_json(_plan(variable="ntg"))
    huge = GeomodelPlan.model_validate_json(_plan(major_range=500.0))

    assert "no geomodeling skill 'kriging'" in plan_problems(unknown, facts)[0]
    assert plan_problems(missing, facts) == ["ntg is not a state variable"]
    assert "outside [0.5, 16]" in plan_problems(huge, facts)[0]


def test_facies_by_proportion_with_their_values() -> None:
    skill = geomodeling_skills.create("facies_pluri")
    parameters = FaciesParameters(
        latent="y1", facies=["shale", "sand"], proportions=[0.3, 0.7], len_scale=5.0,
        values={"shale": {"poro": 0.1}, "sand": {"poro": 0.3}},
        conditioning_cells=[[1, 1, 1]], conditioning_facies=[1],
    )  # fmt: skip
    from apex.core.geomodeling import VariableFact

    facts = GeomodelFacts(variables={"y1": VariableFact(name="y1", size=100, grid=[10, 10, 1])},
                          kind="synthetic")  # fmt: skip

    draft = skill.prior(parameters, facts)
    rng = np.random.default_rng(1)
    right = skill.qc(parameters, {"y1": rng.standard_normal((100, 50))}, 0.05, facts)
    wrong = skill.qc(parameters, {"y1": rng.standard_normal((100, 50)) + 1.0}, 0.05, facts)

    assert draft.groups["facies"]["name"] == "gstools_pgs"
    assert draft.groups["facies"]["options"]["proportions"] == [0.3, 0.7]
    assert draft.parameterizations[0]["name"] == "pgs_truncation"
    assert right == []
    assert any("shale" in p for p in wrong)
    with pytest.raises(ValueError, match="sum to 1"):
        FaciesParameters(latent="y1", facies=["a", "b"], proportions=[0.5, 0.6], len_scale=1,
                         values={"a": {}, "b": {}})  # fmt: skip


def test_the_command_writes_a_prior_and_refuses_an_existing_output(
    tmp_path: Path, small_deck: Path
) -> None:
    _, config, reference = _case(tmp_path, small_deck)
    root = str(tmp_path / "svc")
    runner = CliRunner()
    arguments = ["agent", "geomodel", str(config), "--reference", str(reference), "--no-llm",
                 "--root", root]  # fmt: skip

    first = runner.invoke(app, arguments)
    again = runner.invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert "valid, and the drawn sample matches" in first.stdout
    assert again.exit_code == 1
    # The refusal names the file and how to choose another (release tester H12).
    said = " ".join(again.output.split())
    assert f"{config.parent.name}/{config.stem}_prior.toml exists" in said
    assert "--out" in said


def test_a_stored_prior_leaves_nothing_to_model(tmp_path: Path) -> None:
    from apex.agents.geomodeler import geomodel_facts, rules

    stored = {"name": "file", "options": {"ensemble_file": "p.npz"}}
    raw = {"state": {"variables": [{"name": "x", "size": 4, "grid": [2, 2, 1]}]},
           "prior": {"generator": stored}}  # fmt: skip

    with pytest.raises(ConfigError, match="no gridded state variable"):
        rules(geomodel_facts(raw, tmp_path, None), SETTINGS)


def test_a_facies_prior_validates_and_draws_its_proportions(
    tmp_path: Path, small_deck: Path
) -> None:
    """The deck reads permx as ln(mD): a facies' permx of 500 is refused, ln values pass."""
    pytest.importorskip("gstools")
    import tomli_w

    service, config, _ = _case(tmp_path, small_deck)
    raw = tomllib.loads(config.read_text())
    raw["state"]["variables"] = [{"name": "y1", "size": 24, "grid": [4, 3, 2]}]
    del raw["prior"]["variables"]
    config.write_text(tomli_w.dumps(raw))

    def facies(permx: tuple[float, float]) -> str:
        return GeomodelPlan(skill="facies_pluri", parameters={
            "latent": "y1", "facies": ["shale", "sand"], "proportions": [0.4, 0.6],
            "len_scale": 1.5, "values": {"shale": {"poro": 0.12, "permx": permx[0]},
                                          "sand": {"poro": 0.25, "permx": permx[1]}},
        }).model_dump_json()  # fmt: skip

    outcome = Geomodeler(
        service, StubProvider([facies((10.0, 500.0)), facies((2.3, 6.2))]), settings=SETTINGS
    ).model(config, request="shale and sand, 40/60")

    assert outcome.decided_by == "model"
    assert outcome.valid, (outcome.validation.errors, outcome.qc)
    written = tomllib.loads(outcome.config.read_text())
    assert written["prior"]["groups"]["facies"]["name"] == "gstools_pgs"
    assert written["parameterizations"][0]["options"]["values"]["sand"]["permx"] == 6.2


def _facts(grid: list[int], log: tuple[str, ...] = ()) -> GeomodelFacts:
    from apex.core.geomodeling import VariableFact

    size = grid[0] * grid[1] * grid[2]
    return GeomodelFacts(
        variables={n: VariableFact(name=n, size=size, grid=grid, prior="gaussian")
                   for n in ("permx", "poro")},
        kind="synthetic", log_variables=list(log),
    )  # fmt: skip


def _hard_data(tmp_path: Path, rows: list[tuple[int, int, int, str, float]]) -> Path:
    path = tmp_path / "wells.csv"
    path.write_text("i,j,k,property,value\n" + "".join(f"{i},{j},{k},{p},{v}\n"
                                                      for i, j, k, p, v in rows))  # fmt: skip
    return path


def test_hard_data_give_the_rules_their_statistics_and_variogram(tmp_path: Path) -> None:
    """A field of known range (8 cells) sampled at 600 cells: the rules' prior takes the
    data's mean and std (permx's in ln mD) and the fitted variogram, marked estimated."""
    from apex.agents.geomodeler import rules
    from apex.algorithms.geostatistics.covariance import covariance_factor, layer_covariance
    from apex.services.geomodeling_data import estimate_from_data, load_hard_data

    n = 40
    factor = covariance_factor(layer_covariance(n, n, model="spherical", sill=1.0,
                                                major_range=8.0))  # fmt: skip
    field = factor @ np.random.default_rng(3).standard_normal(n * n)
    index = np.random.default_rng(4).choice(n * n, 600, replace=False)
    j, i = np.divmod(index, n)
    rows = [(int(a) + 1, int(b) + 1, 1, "permx", float(np.exp(4.0 + 0.5 * z)))
            for a, b, z in zip(i, j, field[index], strict=True)]  # fmt: skip
    rows += [(1, 1, 1, "ntg", 0.5)]
    facts = _facts([n, n, 1], log=("permx",))

    estimates, facies, notes = estimate_from_data(load_hard_data(_hard_data(tmp_path, rows)),
                                                  facts)  # fmt: skip
    plan = rules(facts.model_copy(update={"estimates": estimates}), SETTINGS)

    assert facies is None
    assert notes == ["hard data of 'ntg' left out: no such state variable"]
    [permx] = [f for f in plan.parameters["fields"] if f["variable"] == "permx"]
    assert permx["mean"] == pytest.approx(4.0, abs=0.25)
    assert permx["std"] == pytest.approx(0.5, rel=0.25)
    # One realization (the 20-seed recovery is test_variogram.py's): the fitted range used.
    assert permx["major_range"] == pytest.approx(estimates["permx"].major_range)
    assert 4.0 < permx["major_range"] < 16.0
    assert set(permx["estimated"]) == {"mean", "std", "variogram", "range", "azimuth"}
    assert permx["assumed"] == []
    [poro] = [f for f in plan.parameters["fields"] if f["variable"] == "poro"]
    assert "range" in poro["assumed"]  # no data: as before


def test_facies_codes_give_proportions_conditioning_and_a_length_scale(tmp_path: Path) -> None:
    from apex.algorithms.geostatistics.covariance import covariance_factor, layer_covariance
    from apex.services.geomodeling_data import (
        GAUSSIAN_RANGE_PER_LEN_SCALE,
        estimate_from_data,
        load_hard_data,
    )

    n = 30  # facies: a latent gaussian field of range 6 cut at 0
    factor = covariance_factor(layer_covariance(n, n, model="gaussian", sill=1.0,
                                                major_range=6.0))  # fmt: skip
    latent = factor @ np.random.default_rng(5).standard_normal(n * n)
    index = np.random.default_rng(6).choice(n * n, 400, replace=False)  # distinct cells
    rows = [(int(i) + 1, int(j) + 1, 1, "facies", float(latent[c] >= 0))
            for c, (j, i) in zip(index, zip(*np.divmod(index, n), strict=True),
                                 strict=True)]  # fmt: skip
    data = load_hard_data(_hard_data(tmp_path, rows))

    _, facies, _ = estimate_from_data(data, _facts([n, n, 1]))

    assert facies is not None
    assert facies.codes == [0, 1]
    assert sum(facies.proportions) == pytest.approx(1.0)
    assert facies.proportions[0] == pytest.approx(0.5, abs=0.1)
    assert len(facies.conditioning_cells) == 400
    assert facies.len_scale is not None
    # One field (the 20-seed recovery is test_variogram.py's): a range near 6 cells.
    assert 3.0 < facies.len_scale * GAUSSIAN_RANGE_PER_LEN_SCALE < 12.0


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1, 1, 1, "poro", 0.2), (9, 1, 1, "poro", 0.2)], "outside its grid"),
        ([(1, 1, 1, "permx", -3.0)], "must be positive"),
        ([(1, 1, 1, "facies", 0.5)], "integers"),
    ],
)  # fmt: skip
def test_bad_hard_data_are_refused(
    tmp_path: Path, rows: list[tuple[int, int, int, str, float]], message: str
) -> None:
    from apex.services.geomodeling_data import estimate_from_data, load_hard_data

    data = load_hard_data(_hard_data(tmp_path, rows))
    with pytest.raises(ConfigError, match=message):
        estimate_from_data(data, _facts([4, 4, 1], log=("permx",)))


def test_hard_data_without_a_column_are_refused(tmp_path: Path) -> None:
    from apex.services.geomodeling_data import load_hard_data

    path = tmp_path / "wells.csv"
    path.write_text("i,j,property,value\n1,1,poro,0.2\n")
    with pytest.raises(ConfigError, match="no column k"):
        load_hard_data(path)


def test_the_command_takes_hard_data(tmp_path: Path, small_deck: Path) -> None:
    """Too few data for a variogram: the mean and std are estimated, the range assumed."""
    _, config, _ = _case(tmp_path, small_deck)
    data = _hard_data(tmp_path, [(1, 1, 1, "poro", 0.18), (2, 1, 1, "poro", 0.22),
                                 (3, 2, 2, "poro", 0.2), (4, 3, 1, "poro", 0.24)])  # fmt: skip

    result = CliRunner().invoke(app, ["agent", "geomodel", str(config), "--data", str(data),
                                      "--no-llm", "--root", str(tmp_path / "svc")])  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "poro: estimated from the hard data: mean, std" in result.stdout
    written = tomllib.loads(config.with_name("config_prior.toml").read_text())
    assert written["prior"]["variables"]["poro"]["options"]["mean"] == pytest.approx(0.21)


def test_correlated_fields_are_prepare_py_bit_for_bit() -> None:
    """3dBox's prepare.py draws z_K, z_P from gaussian_field with one generator, then
    poro = mean + std (rho z_K + sqrt(1 - rho^2) z_P): the skill draws the same."""
    from apex.algorithms.priors import GaussianFieldPrior
    from apex.domain.state import StateVariable
    from apex.services.geomodeling_skills import (
        CorrelatedField,
        CorrelatedParameters,
        realize_correlated,
    )

    grid, members, seed, rho = [8, 6, 2], 7, 2026, 0.7
    facts = _facts(grid)
    parameters = CorrelatedParameters(
        fields=[CorrelatedField(variable="permx", mean=5.0, std=1.0),
                CorrelatedField(variable="poro", mean=0.2, std=0.05)],
        correlation=[[1.0, rho], [rho, 1.0]], variogram="gaussian", major_range=4.0, seed=seed,
    )  # fmt: skip

    drawn = realize_correlated(parameters, facts, members)

    field = GaussianFieldPrior(grid=grid, mean=0.0, variance=1.0, major_range=4.0,
                               variogram="gaussian")  # fmt: skip
    rng = np.random.default_rng(seed)
    z_k, z_p = (field.generate(StateVariable(name=n, size=96), members, rng) for n in "kp")
    np.testing.assert_array_equal(drawn["permx"], 5.0 + 1.0 * z_k)
    np.testing.assert_allclose(drawn["poro"], 0.2 + 0.05 * (rho * z_k + np.sqrt(1 - rho**2) * z_p),
                               rtol=1e-15, atol=1e-15)  # fmt: skip


def test_correlated_fields_match_3dbox_prepare_statistics() -> None:
    """The committed 3dBox prior (prepare.py: ln PERMX 5 +- 1, PORO 0.2 +- 0.05, rho 0.7,
    gaussian range 17 on 60 x 60 x 5, clipped) against the skill's on a 30 x 30 x 2 grid
    with the same parameters and range halved: means, spreads and correlation agree within
    sampling error."""
    from apex.services.geomodeling_skills import (
        CorrelatedField,
        CorrelatedParameters,
        realize_correlated,
    )

    root = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production_avo"
    with np.load(root / "prior_ensemble.npz") as stored:
        reference = {k: np.asarray(stored[k], dtype=np.float64) for k in ("permx", "poro")}
    facts = _facts([30, 30, 2]).model_copy(update={"variables": {
        n: v.model_copy(update={"lower": low, "upper": high})
        for (n, v), (low, high) in zip(_facts([30, 30, 2]).variables.items(),
                                       [(-3.0, 10.0), (0.01, 0.4)], strict=True)}})  # fmt: skip
    parameters = CorrelatedParameters(
        fields=[CorrelatedField(variable="permx", mean=5.0, std=1.0),
                CorrelatedField(variable="poro", mean=0.2, std=0.05)],
        correlation=[[1.0, 0.7], [0.7, 1.0]], variogram="gaussian", major_range=8.5,
    )  # fmt: skip

    drawn = realize_correlated(parameters, facts, 100)

    for name, spread in (("permx", 1.0), ("poro", 0.05)):
        assert drawn[name].mean() == pytest.approx(reference[name].mean(), abs=0.15 * spread)
        assert drawn[name].std() == pytest.approx(reference[name].std(), rel=0.1)

    def corr(fields: dict[str, np.ndarray]) -> float:
        return float(np.corrcoef(fields["permx"].ravel(), fields["poro"].ravel())[0, 1])

    assert corr(drawn) == pytest.approx(corr(reference), abs=0.05)
    assert drawn["poro"].min() >= 0.01


def test_the_geomodeler_writes_a_correlated_prior_that_passes(
    tmp_path: Path, small_deck: Path
) -> None:
    service, config, _ = _case(tmp_path, small_deck)
    plan = GeomodelPlan(skill="correlated_continuous", parameters={
        "fields": [{"variable": "permx", "mean": 4.5, "std": 0.5},
                   {"variable": "poro", "mean": 0.2, "std": 0.02}],
        "correlation": [[1.0, 0.8], [0.8, 1.0]], "major_range": 2.0,
        "assumed": ["correlation"]}).model_dump_json()  # fmt: skip

    outcome = Geomodeler(service, StubProvider([plan]), settings=SETTINGS).model(
        config, request="porosity correlated with permeability, 0.8"
    )

    assert outcome.decided_by == "model"
    assert outcome.valid, (outcome.validation.errors, outcome.qc)
    written = tomllib.loads(outcome.config.read_text())
    assert written["prior"]["variables"]["poro"] == {
        "name": "file",
        "options": {"ensemble_file": "config_prior_ensemble.npz", "key": "poro"},
    }
    with np.load(config.with_name("config_prior_ensemble.npz")) as stored:
        assert stored["poro"].shape == (24, 20)
    assert any("assumed correlation" in n for n in outcome.notes)


def test_a_correlation_that_is_not_one_is_refused() -> None:
    from apex.services.geomodeling_skills import CorrelatedField, CorrelatedParameters

    fields = [CorrelatedField(variable="permx", mean=5.0, std=1.0),
              CorrelatedField(variable="poro", mean=0.2, std=0.05)]  # fmt: skip
    with pytest.raises(ValueError, match="unit diagonal"):
        CorrelatedParameters(fields=fields, correlation=[[1.0, 0.5], [0.4, 1.0]], major_range=2)
    with pytest.raises(ValueError, match="in \\[-1, 1\\]"):
        CorrelatedParameters(fields=fields, correlation=[[1.0, 1.5], [1.5, 1.0]], major_range=2)


def _training_image(folder: Path) -> Path:
    i = np.arange(20)
    image = ((i[:, None] // 4) % 2) * np.ones((1, 20), dtype=np.int64)  # stripes along j
    path = folder / "ti.npy"
    np.save(path, image.astype(np.int64))
    return path


def test_mps_facies_sections_and_checks(tmp_path: Path) -> None:
    from apex.services.geomodeling_skills import MpsParameters

    _training_image(tmp_path)
    skill = geomodeling_skills.create("facies_mps")
    facts = _facts([4, 4, 1]).model_copy(update={"folder": str(tmp_path), "variables": {
        n: v.model_copy(update={"name": n}) for n, v in
        zip(("d0", "d1"), _facts([4, 4, 1]).variables.values(), strict=True)}})  # fmt: skip
    parameters = MpsParameters(fields=["d0", "d1"], facies=["shale", "sand"],
                               training_image_file="ti.npy",
                               values={"shale": {"poro": 0.1}, "sand": {"poro": 0.3}},
                               conditioning_cells=[[1, 1, 1]], conditioning_facies=[1])  # fmt: skip

    draft = skill.prior(parameters, facts)
    three = parameters.model_copy(update={"facies": ["a", "b", "c"], "fields": ["d0", "d1", "d2"]})
    right = np.where(np.arange(16)[:, None] % 2 == 0, 1.0, -1.0) * np.ones((1, 30))
    sample = {"d0": right, "d1": -right}  # half shale, half sand, as the image

    assert skill.check(parameters, facts) == []
    assert draft.groups["facies"]["name"] == "mpslib_snesim"
    assert draft.groups["facies"]["options"]["conditioning"] == {"cells": [[1, 1, 1]],
                                                                  "facies": [1]}  # fmt: skip
    assert draft.parameterizations[0]["name"] == "facies_distance"
    assert any("not 0...2" in p for p in skill.check(three, facts))
    assert skill.qc(parameters, sample, 0.05, facts) == []
    assert any("sand" in p for p in skill.qc(parameters, {"d0": right * 0 + 1, "d1": -right * 0},
                                             0.05, facts))  # fmt: skip


def test_a_cell_given_twice_is_refused(tmp_path: Path) -> None:
    from apex.services.geomodeling_data import load_hard_data

    with pytest.raises(ConfigError, match="second poro value"):
        load_hard_data(_hard_data(tmp_path, [(1, 1, 1, "poro", 0.2), (1, 1, 1, "poro", 0.3)]))


@pytest.mark.parametrize("escape", ["../secret.npy", "ABSOLUTE", "link.npy", "none.npy"])
def test_a_training_image_outside_the_case_is_never_read(tmp_path: Path, escape: str) -> None:
    """The review's blocker (2026-10-25): the model names the training image; a path out of
    the case's folder (.., absolute, a link out) is refused as a missing one is, the same
    words whatever is there, and nothing of it reaches the problems."""
    from apex.services.geomodeling_skills import MpsParameters

    case = tmp_path / "case"
    case.mkdir()
    secret = _training_image(tmp_path)  # a valid image, outside the case
    secret.rename(tmp_path / "secret.npy")
    (case / "link.npy").symlink_to(tmp_path / "secret.npy")
    name = str(tmp_path / "secret.npy") if escape == "ABSOLUTE" else escape
    skill = geomodeling_skills.create("facies_mps")
    base = _facts([4, 4, 1])
    facts = base.model_copy(update={"folder": str(case), "variables": {
        n: v.model_copy(update={"name": n})
        for n, v in zip(("d0", "d1"), base.variables.values(), strict=True)}})  # fmt: skip
    parameters = MpsParameters(fields=["d0", "d1"], facies=["a", "b"], training_image_file=name,
                               values={"a": {"poro": 0.1}, "b": {"poro": 0.3}})  # fmt: skip

    problems = skill.check(parameters, facts)

    assert problems == [f"no training image {name!r} in the case's folder (an .npy or GSLIB "
                        "grid of codes, at most 200 MB)"]  # fmt: skip
    assert skill.qc(parameters, {"d0": np.ones((16, 2)), "d1": np.zeros((16, 2))}, 0.05,
                    facts) == []  # fmt: skip


def test_two_correlated_priors_beside_one_configuration(tmp_path: Path, small_deck: Path) -> None:
    """The review (2026-10-25): the stored ensemble is named after the new configuration, so
    a second one beside the same configuration does not collide; an existing one is refused
    before the model is asked."""
    service, config, _ = _case(tmp_path, small_deck)
    plan = GeomodelPlan(skill="correlated_continuous", parameters={
        "fields": [{"variable": "permx", "mean": 4.5, "std": 0.5},
                   {"variable": "poro", "mean": 0.2, "std": 0.02}],
        "correlation": [[1.0, 0.6], [0.6, 1.0]], "major_range": 2.0}).model_dump_json()  # fmt: skip
    geomodeler = Geomodeler(service, StubProvider([plan, plan]), settings=SETTINGS)

    first = geomodeler.model(config, config.with_name("a.toml"))
    second = geomodeler.model(config, config.with_name("b.toml"))
    config.with_name("c_ensemble.npz").write_bytes(b"")

    assert first.valid
    assert second.valid
    assert config.with_name("a_ensemble.npz").is_file()
    assert config.with_name("b_ensemble.npz").is_file()
    with pytest.raises(ConfigError, match=r"c_ensemble\.npz exists"):
        Geomodeler(service, StubProvider([]), settings=SETTINGS).model(
            config, config.with_name("c.toml")
        )


def test_a_mean_outside_the_bounds_is_clamped_and_assumed() -> None:
    from apex.agents.geomodeler import rules
    from apex.core.geomodeling import PropertyEstimate

    facts = _facts([10, 10, 1])
    poro = facts.variables["poro"].model_copy(update={"lower": 0.01, "upper": 0.4})
    facts = facts.model_copy(update={
        "variables": {**facts.variables, "poro": poro},
        "estimates": {"poro": PropertyEstimate(variable="poro", n_data=50, mean=21.0, std=3.0)},
    })  # fmt: skip

    [field] = [f for f in rules(facts, SETTINGS).parameters["fields"] if f["variable"] == "poro"]

    assert field["mean"] == 0.4
    assert field["estimated"] == ["std"]
    assert "21" in field["assumed"][0]


def _field(n: int, seed: int, major_range: float = 8.0) -> np.ndarray:
    """A unit spherical field on an n x n layer (Eclipse order)."""
    from apex.algorithms.geostatistics.covariance import covariance_factor, layer_covariance

    factor = covariance_factor(layer_covariance(n, n, model="spherical", sill=1.0,
                                                major_range=major_range))  # fmt: skip
    return factor @ np.random.default_rng(seed).standard_normal(n * n)


def test_a_reference_field_gives_its_variogram_and_is_not_conditioned_to(tmp_path: Path) -> None:
    """A whole field of range 8 as the reference: its mean, std (every cell) and variogram
    (a fixed sample of cells) are the facts, and the prior shares them without being
    conditioned to the field."""
    from apex.agents.geomodeler import rules
    from apex.services.geomodeling_data import estimate_from_reference

    n = 50
    permx = 4.0 + 0.5 * _field(n, 11)
    reference = tmp_path / "reference.npz"
    np.savez(reference, permx=permx, poro=np.zeros(n * n))  # poro all inactive: left out
    facts = _facts([n, n, 1], log=("permx",))

    estimates, notes = estimate_from_reference(reference, facts)
    again, _ = estimate_from_reference(reference, facts)

    assert set(estimates) == {"permx"}
    assert notes == []
    found = estimates["permx"]
    assert found.source == "reference"
    assert found.cells == []
    assert found.n_data == n * n
    assert found.mean == pytest.approx(permx.mean())
    assert found.std == pytest.approx(permx.std())
    assert 4.0 < (found.major_range or 0) < 16.0
    assert again == estimates  # a fixed sample: the same reference, the same estimate
    facts = facts.model_copy(update={"estimates": estimates})
    plan = rules(facts, SETTINGS)
    skill = geomodeling_skills.create("continuous")
    parameters = skill.parameters.model_validate(plan.parameters)
    draft = skill.prior(parameters, facts)
    assert "conditioning" not in draft.variables["permx"]["options"]
    assert any("permx: estimated from the reference field" in note for note in draft.notes)


def test_a_reference_and_hard_data_give_its_statistics_conditioned_to_them(
    tmp_path: Path,
) -> None:
    from apex.services.geomodeling_data import (
        estimate_from_data,
        estimate_from_reference,
        load_hard_data,
        merge_estimates,
    )

    n = 30
    reference = tmp_path / "reference.npz"
    np.savez(reference, permx=4.0 + 0.5 * _field(n, 12, 6.0))
    facts = _facts([n, n, 1], log=("permx",))
    wells = _hard_data(tmp_path, [(3, 4, 1, "permx", 120.0), (20, 25, 1, "permx", 15.0)])

    data, _, _ = estimate_from_data(load_hard_data(wells), facts)
    from_reference, _ = estimate_from_reference(reference, facts)
    merged = merge_estimates(from_reference, data)["permx"]

    assert merged.source == "reference+data"
    assert merged.mean == from_reference["permx"].mean
    assert merged.major_range == from_reference["permx"].major_range
    assert merged.cells == [[3, 4, 1], [20, 25, 1]]
    assert merged.values == pytest.approx([np.log(120.0), np.log(15.0)])  # state space
    assert (
        "cells"
        not in facts.model_copy(update={"estimates": {"permx": merged}}).model_dump()["estimates"][
            "permx"
        ]
    )  # fmt: skip  (the model never sees the data)


def test_the_geomodeler_conditions_a_prior_to_hard_data(tmp_path: Path, small_deck: Path) -> None:
    """The rules' prior from four porosity data: the generator takes them as conditioning,
    and the drawn sample's check (every member takes the data) passes."""
    service, config, _ = _case(tmp_path, small_deck)
    rows = [(1, 1, 1, "poro", 0.18), (2, 1, 1, "poro", 0.22), (3, 2, 2, "poro", 0.2),
            (4, 3, 1, "poro", 0.24)]  # fmt: skip

    outcome = Geomodeler(service, None, settings=SETTINGS).model(
        config, data=_hard_data(tmp_path, rows)
    )

    assert outcome.valid, (outcome.validation.errors, outcome.qc)
    options = tomllib.loads(outcome.config.read_text())["prior"]["variables"]["poro"]["options"]
    assert options["conditioning"] == {"cells": [list(r[:3]) for r in rows],
                                       "values": [r[4] for r in rows]}  # fmt: skip
    assert any("poro: conditioned to 4 hard data, taken exactly" in n for n in outcome.notes)


def test_hard_data_the_bounds_clip_are_not_conditioned_to() -> None:
    """Data outside the bounds cannot be honoured: the rules leave the conditioning out
    (assumed), and a plan that keeps it is refused with the reason."""
    from apex.agents.geomodeler import rules
    from apex.core.geomodeling import PropertyEstimate

    facts = _facts([4, 4, 1])
    poro = facts.variables["poro"].model_copy(update={"lower": 0.05, "upper": 0.3})
    estimate = PropertyEstimate(variable="poro", n_data=2, mean=0.2, std=0.05,
                                cells=[[1, 1, 1], [2, 2, 1]], values=[0.2, 0.45])  # fmt: skip
    facts = facts.model_copy(update={"variables": {**facts.variables, "poro": poro},
                                     "estimates": {"poro": estimate}})  # fmt: skip

    plan = rules(facts, SETTINGS)
    [field] = [f for f in plan.parameters["fields"] if f["variable"] == "poro"]
    asked = GeomodelPlan(skill="continuous", parameters={"fields": [{**field, "condition": True}]},
                         reason="as asked")  # fmt: skip

    assert field["condition"] is False
    assert any("no conditioning" in a and "outside its bounds" in a for a in field["assumed"])
    assert plan_problems(plan, facts) == []
    assert any("outside its bounds [0.05, 0.3]" in p for p in plan_problems(asked, facts))


def test_correlated_fields_are_cokriged_to_their_hard_data() -> None:
    """ln PERMX known at one cell: every member takes it, and PORO there follows it through
    the correlation, E[poro | permx = d] = mu_p + rho s_p (d - mu_k) / s_k, with the
    conditional std s_p sqrt(1 - rho^2) (the bivariate normal's closed form)."""
    from apex.core.geomodeling import PropertyEstimate
    from apex.services.geomodeling_skills import (
        CorrelatedField,
        CorrelatedParameters,
        realize_correlated,
    )

    rho, datum = 0.8, 6.5
    known = PropertyEstimate(variable="permx", n_data=1, mean=5.0, std=1.0, cells=[[3, 2, 1]],
                             values=[datum])  # fmt: skip
    facts = _facts([6, 5, 1]).model_copy(update={"estimates": {"permx": known}})
    parameters = CorrelatedParameters(
        fields=[CorrelatedField(variable="permx", mean=5.0, std=1.0),
                CorrelatedField(variable="poro", mean=0.2, std=0.05)],
        correlation=[[1.0, rho], [rho, 1.0]], major_range=3.0,
    )  # fmt: skip

    drawn = realize_correlated(parameters, facts, 20_000)

    row = 2 + 6 * 1
    np.testing.assert_allclose(drawn["permx"][row], datum, atol=1e-12)
    poro = drawn["poro"][row]
    assert poro.mean() == pytest.approx(0.2 + rho * 0.05 * (datum - 5.0), abs=4 * 0.03 / 141)
    assert poro.std() == pytest.approx(0.05 * np.sqrt(1 - rho**2), rel=0.03)
    unconditioned = realize_correlated(parameters.model_copy(update={"condition": False}),
                                       facts, 50)  # fmt: skip
    assert not np.allclose(unconditioned["permx"][row], datum)


def test_a_large_reference_is_sampled_for_its_variogram(tmp_path: Path) -> None:
    from apex.services.geomodeling_data import REFERENCE_SAMPLE, estimate_from_reference

    n = 60  # 3600 cells: more than the sample
    reference = tmp_path / "reference.npz"
    np.savez(reference, poro=0.2 + 0.03 * _field(n, 13, 10.0))

    estimates, _ = estimate_from_reference(reference, _facts([n, n, 1]))

    assert estimates["poro"].n_data == n * n
    assert f"variogram from {REFERENCE_SAMPLE} of {n * n} cells" in estimates["poro"].notes


def test_one_datum_outside_the_bounds_leaves_the_rules_unconditioned() -> None:
    """One datum gives no std, so the rules' field comes from the bounds; its data still
    cannot be honoured, and the rules' plan, which must pass, says so."""
    from apex.agents.geomodeler import rules
    from apex.core.geomodeling import PropertyEstimate

    facts = _facts([4, 4, 1])
    poro = facts.variables["poro"].model_copy(update={"lower": 0.0, "upper": 0.4})
    one = PropertyEstimate(variable="poro", n_data=1, mean=0.45, std=0.0, cells=[[1, 1, 1]],
                           values=[0.45])  # fmt: skip
    facts = facts.model_copy(update={"variables": {**facts.variables, "poro": poro},
                                     "estimates": {"poro": one}})  # fmt: skip

    plan = rules(facts, SETTINGS)

    [field] = [f for f in plan.parameters["fields"] if f["variable"] == "poro"]
    assert field["condition"] is False
    assert plan_problems(plan, facts) == []


def test_data_improbable_under_the_variogram_are_refused_and_left_out_by_the_rules() -> None:
    from apex.agents.geomodeler import rules
    from apex.core.geomodeling import PropertyEstimate

    facts = _facts([60, 1, 1])
    rough = PropertyEstimate(variable="poro", n_data=4, mean=0.5, std=0.5, variogram="gaussian",
                             major_range=40.0, cells=[[i, 1, 1] for i in (10, 13, 16, 19)],
                             values=[0.0, 1.0, 0.0, 1.0])  # fmt: skip
    facts = facts.model_copy(update={"estimates": {"poro": rough}})
    asked = GeomodelPlan(skill="continuous", reason="as asked", parameters={"fields": [
        {"variable": "poro", "mean": 0.5, "std": 0.5, "variogram": "gaussian",
         "major_range": 40.0}]})  # fmt: skip

    [problem] = plan_problems(asked, facts)
    plan = rules(facts, SETTINGS)

    assert "improbable under this variogram" in problem
    assert "prior stds" in problem
    [field] = [f for f in plan.parameters["fields"] if f["variable"] == "poro"]
    assert field["condition"] is False
    assert plan_problems(plan, facts) == []


def test_a_log_variables_zeros_are_values_and_its_sample_its_own(tmp_path: Path) -> None:
    """ln 1 mD = 0 is a permeability, not an inactive cell; and a variable's sample does
    not change when another variable joins the reference."""
    from apex.services.geomodeling_data import estimate_from_reference

    n = 30
    permx = np.round(_field(n, 14, 6.0), 1)  # many exact zeros
    alone, both = tmp_path / "alone.npz", tmp_path / "both.npz"
    np.savez(alone, permx=permx)
    np.savez(both, permx=permx, poro=0.2 + 0.03 * _field(n, 15, 6.0))
    facts = _facts([n, n, 1], log=("permx",))

    first, _ = estimate_from_reference(alone, facts)
    second, _ = estimate_from_reference(both, facts)

    assert np.count_nonzero(permx == 0) > 10
    assert first["permx"].n_data == n * n
    assert first["permx"] == second["permx"]


def test_cokriging_matches_the_joint_conditional_gaussian_off_the_data() -> None:
    """Both variables with data, in two layers: the members' mean and covariance at cells
    off the data equal the joint Gaussian's conditional ones in closed form (covariance
    s_a s_b R_ab C between all variables' cells, layers independent)."""
    from apex.algorithms.geostatistics import layer_covariance
    from apex.core.geomodeling import PropertyEstimate
    from apex.services.geomodeling_skills import (
        CorrelatedField,
        CorrelatedParameters,
        realize_correlated,
    )

    grid, rho, members = [5, 4, 2], -0.6, 40_000
    means, stds = np.array([5.0, 0.2]), np.array([1.0, 0.05])
    data = {"permx": ([[2, 2, 1], [4, 3, 2]], [6.0, 3.5]),
            "poro": ([[3, 2, 1], [1, 4, 2]], [0.25, 0.12])}  # fmt: skip
    facts = _facts(grid).model_copy(update={"estimates": {
        name: PropertyEstimate(variable=name, n_data=2, mean=0.0, std=1.0, cells=cells,
                               values=values)
        for name, (cells, values) in data.items()}})  # fmt: skip
    parameters = CorrelatedParameters(
        fields=[CorrelatedField(variable="permx", mean=5.0, std=1.0),
                CorrelatedField(variable="poro", mean=0.2, std=0.05)],
        correlation=[[1.0, rho], [rho, 1.0]], variogram="exponential", major_range=4.0,
    )  # fmt: skip

    drawn = realize_correlated(parameters, facts, members)

    layer = 20
    unit = layer_covariance(5, 4, model="exponential", sill=1.0, major_range=4.0)
    correlation = np.array([[1.0, rho], [rho, 1.0]])
    joint = np.kron(np.outer(stds, stds) * correlation, unit)  # (permx, poro) x cells
    for k in range(2):
        rows = []
        observed = []
        for a, name in enumerate(("permx", "poro")):
            for (i, j, kk), value in zip(*data[name], strict=True):
                if kk == k + 1:
                    rows.append(a * layer + (i - 1) + 5 * (j - 1))
                    observed.append(value)
        prior_mean = np.repeat(means, layer)
        gain = np.linalg.solve(joint[np.ix_(rows, rows)], joint[rows]).T
        mean = prior_mean + gain @ (np.array(observed) - prior_mean[rows])
        cov = joint - gain @ joint[rows]
        stacked = np.vstack([drawn["permx"][k * layer : (k + 1) * layer],
                             drawn["poro"][k * layer : (k + 1) * layer]])  # fmt: skip
        scale = np.repeat(stds, layer)
        np.testing.assert_allclose((stacked.mean(axis=1) - mean) / scale, 0.0,
                                   atol=5 / np.sqrt(members))  # fmt: skip
        np.testing.assert_allclose(np.cov(stacked) / np.outer(scale, scale),
                                   cov / np.outer(scale, scale),
                                   atol=6 * np.sqrt(2 / members))  # fmt: skip


def test_cokriged_porosity_meets_its_data_beside_ln_k_on_a_smooth_model() -> None:
    """ln k (std 1.5) and porosity (std 0.03) with data at close cells of a long Gaussian
    model, drawn from the joint prior: both are met to far below a std (the nugget was
    relative to the largest variance, which left porosity 0.05 std off)."""
    from apex.algorithms.priors import GaussianFieldPrior
    from apex.core.geomodeling import PropertyEstimate
    from apex.domain.state import StateVariable
    from apex.services.geomodeling_skills import (
        CorrelatedField,
        CorrelatedParameters,
        realize_correlated,
    )

    grid, rho = [30, 1, 1], 0.7
    unit = GaussianFieldPrior(grid=grid, variogram="gaussian", major_range=25.0)
    rng = np.random.default_rng(9)
    z = [unit.generate(StateVariable(name=n, size=30), 2, rng)[:, 0] for n in "kp"]
    lnk, poro = 5.0 + 1.5 * z[0], 0.2 + 0.03 * (rho * z[0] + np.sqrt(1 - rho**2) * z[1])
    cells = [[i, 1, 1] for i in (10, 12, 14, 16)]
    facts = _facts(grid).model_copy(update={"estimates": {
        "permx": PropertyEstimate(variable="permx", n_data=4, mean=5.0, std=1.5, cells=cells,
                                  values=[float(lnk[c[0] - 1]) for c in cells]),
        "poro": PropertyEstimate(variable="poro", n_data=4, mean=0.2, std=0.03, cells=cells,
                                 values=[float(poro[c[0] - 1]) for c in cells])}})  # fmt: skip
    parameters = CorrelatedParameters(
        fields=[CorrelatedField(variable="permx", mean=5.0, std=1.5),
                CorrelatedField(variable="poro", mean=0.2, std=0.03)],
        correlation=[[1.0, rho], [rho, 1.0]], variogram="gaussian", major_range=25.0,
    )  # fmt: skip

    drawn = realize_correlated(parameters, facts, 20)

    rows = [c[0] - 1 for c in cells]
    assert np.abs(drawn["permx"][rows] - lnk[rows, None]).max() < 1e-2 * 1.5
    assert np.abs(drawn["poro"][rows] - poro[rows, None]).max() < 1e-2 * 0.03
