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
    from apex.core.geomodeling import GeomodelFacts, VariableFact

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
    assert "exists" in again.output


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
