"""The case-converter agent: the rules, the model's choices held to the deck's
facts, revisions on validation errors, and its command, on a small deck with a history."""

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apex.agents.catalogue import AGENTS
from apex.agents.converter import CONFIG, NOTES, CaseConverter, choice_problems
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.core.conversion import ConversionChoices, ObservationChoice, VariableChoice
from apex.core.errors import ConfigError
from apex.core.llm import LLMUnavailable
from apex.domain.services import Validation
from apex.services.api import ApexService
from apex.services.settings import CaseConverterSection

SMALL = CaseConverterSection(ensemble_size=20, iterations=2, max_revisions=1)


def _choices(**changes: object) -> ConversionChoices:
    base: dict[str, object] = {
        "name": "small",
        "variables": [VariableChoice(property="PORO")],
        "observations": [ObservationChoice(vector="WOPR:PROD", relative_error=0.05, floor=2.0)],
        "method": "esmda",
        "iterations": 3,
        "ensemble_size": 30,
        "reason": "the user asked for porosity and ES-MDA",
    }
    return ConversionChoices.model_validate({**base, **changes})


def test_the_rules_convert_a_deck_that_validates(tmp_path: Path, small_deck: Path) -> None:
    service = ApexService(tmp_path / "service")

    outcome = CaseConverter(service, None, settings=SMALL).convert(small_deck, tmp_path / "out")

    assert outcome.valid, outcome.validation.errors
    assert outcome.decided_by == "rules"
    assert [(v.property, v.transform) for v in outcome.choices.variables] == [
        ("PERMX", "log"), ("PORO", "none")]  # fmt: skip
    # Every rate and pressure with data; WGPR is 0 throughout and is left out.
    assert {o.vector for o in outcome.choices.observations} == {
        "WOPR:PROD", "WWPR:PROD", "WBHP:PROD", "WWIR:INJ", "WBHP:INJ"}  # fmt: skip
    assert outcome.validation.plan is not None
    assert outcome.validation.plan.members == 20
    assert (tmp_path / "out" / CONFIG).is_file()
    assert (tmp_path / "out" / "deck.mako").is_file()
    assert "assumed" in (tmp_path / "out" / NOTES).read_text()
    assert list((tmp_path / "service" / "agents").glob("case-converter-*-decision.json"))


def test_the_models_choices_are_used_when_they_hold(tmp_path: Path, small_deck: Path) -> None:
    service = ApexService(tmp_path / "service")
    wrong = _choices(variables=[VariableChoice(property="NTG")]).model_dump_json()
    right = _choices().model_dump_json()

    outcome = CaseConverter(service, StubProvider([wrong, right]), settings=SMALL).convert(
        small_deck, tmp_path / "out", request="match the porosity with ES-MDA"
    )

    assert outcome.decided_by == "model"
    assert outcome.valid
    assert outcome.choices.method == "esmda"
    method = tomllib.loads((tmp_path / "out" / CONFIG).read_text())["method"]
    assert method["name"] == "esmda"
    assert method["options"]["inflation"] == [3, 3, 3]


def test_choices_are_held_to_the_facts(small_deck: Path) -> None:
    from apex.core.plugins import converter_skills

    facts = converter_skills.create("opm_flow").read(small_deck)
    choices = _choices(
        variables=[VariableChoice(property="NTG"), VariableChoice(property="SWAT"),
                   VariableChoice(property="DX")],
        observations=[ObservationChoice(vector="WOPR:NOWELL", relative_error=0.1, floor=1.0)],
    )  # fmt: skip

    problems = choice_problems(choices, facts)

    assert any("NTG cannot carry the state" in p for p in problems)
    assert any("SWAT is not a property" in p for p in problems)
    assert any("DX cannot carry the state" in p for p in problems)
    assert "WOPR:NOWELL has no observed data" in problems


def test_an_unavailable_model_leaves_it_to_the_rules(tmp_path: Path, small_deck: Path) -> None:
    service = ApexService(tmp_path / "service")

    outcome = CaseConverter(
        service, StubProvider([LLMUnavailable("down")]), settings=SMALL
    ).convert(small_deck, tmp_path / "out")

    assert outcome.decided_by == "fallback"
    assert outcome.valid


def test_a_request_the_rules_did_not_apply_is_named(tmp_path: Path, small_deck: Path) -> None:
    """A reported issue (2026-10-24): a failed model's request was dropped without a word."""
    service = ApexService(tmp_path / "service")

    down = CaseConverter(service, StubProvider([LLMUnavailable("down")]), settings=SMALL).convert(
        small_deck, tmp_path / "a", request="porosity only, with ES-MDA"
    )
    unasked = CaseConverter(service, None, settings=SMALL).convert(
        small_deck, tmp_path / "b", request="porosity only, with ES-MDA"
    )

    assert down.notes[0].startswith("the request was not applied (")
    assert down.notes[0].endswith(": 'porosity only, with ES-MDA'; the rules' choices are used")
    assert unasked.notes[0] == ("the request was not applied (no model was asked): 'porosity "
                                "only, with ES-MDA'; the rules' choices are used")  # fmt: skip
    assert "the request was not applied" in (tmp_path / "b" / NOTES).read_text()
    plain = CaseConverter(service, None, settings=SMALL).convert(small_deck, tmp_path / "c")
    assert not any("request" in note for note in plain.notes)


def test_the_prior_is_noted_with_its_mean_and_variance(tmp_path: Path, small_deck: Path) -> None:
    """A reported issue (2026-10-24): the notes said "a Gaussian field around the deck's
    values"; the prior is stationary, with the deck's mean and variance, which were not said."""
    service = ApexService(tmp_path / "service")

    outcome = CaseConverter(service, None, settings=SMALL).convert(small_deck, tmp_path / "out")

    raw = tomllib.loads((tmp_path / "out" / CONFIG).read_text())
    for name, space in (("permx", " of its logarithm"), ("poro", "")):
        options = raw["prior"]["variables"][name]["options"]
        note = next(n for n in outcome.notes if n.startswith(f"{name}: a stationary"))
        assert (f"with the deck's mean {options['mean']:.4g} and variance "
                f"{options['variance']:.4g}{space}, range") in note  # fmt: skip
    assert "around the deck's values" not in (tmp_path / "out" / NOTES).read_text()


def test_a_missing_deck_is_named_by_the_path_given(tmp_path: Path) -> None:
    given = tmp_path / "decks" / "MODEL.DATA"

    with pytest.raises(ConfigError, match=f"^no case at {given}$"):
        CaseConverter(ApexService(tmp_path / "s"), None, settings=SMALL).convert(
            given, tmp_path / "out"
        )


def test_an_invalid_draft_goes_back_to_the_model_then_to_the_rules(
    tmp_path: Path, small_deck: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ApexService(tmp_path / "service")
    validate = service.validate
    calls: list[Path] = []

    def first_invalid(config: Path, *args: object, **kwargs: object) -> Validation:
        calls.append(config)
        if len(calls) <= 2:  # the model's draft and its revision
            return Validation(valid=False, errors=["too few observations for this test"])
        return validate(config)

    monkeypatch.setattr(service, "validate", first_invalid)
    provider = StubProvider([_choices().model_dump_json(), _choices().model_dump_json()])

    outcome = CaseConverter(service, provider, settings=SMALL).convert(small_deck, tmp_path / "out")

    assert len(calls) == 3
    assert outcome.decided_by == "fallback"
    assert outcome.valid
    assert "did not validate" in (outcome.reason or "")
    assert [v.property for v in outcome.choices.variables] == ["PERMX", "PORO"]


def test_an_output_folder_with_files_is_refused(tmp_path: Path, small_deck: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "mine.txt").write_text("keep me")

    with pytest.raises(ConfigError, match="not an empty folder"):
        CaseConverter(ApexService(tmp_path / "s"), None, settings=SMALL).convert(
            small_deck, tmp_path / "out"
        )
    assert (tmp_path / "out" / "mine.txt").read_text() == "keep me"


def test_the_command_converts_and_lists_the_skills(tmp_path: Path, small_deck: Path) -> None:
    root = tmp_path / "svc"
    root.mkdir()
    (root / "service.toml").write_text("[agents.case_converter]\nensemble_size = 20\n")

    result = CliRunner().invoke(app, ["agent", "convert", "import", str(small_deck), "--out",
                                      str(tmp_path / "out"), "--no-llm", "--root", str(root),
                                      "--json"])  # fmt: skip
    skills = CliRunner().invoke(app, ["agent", "convert", "skills"])

    assert result.exit_code == 0, result.output
    shown = json.loads(result.stdout)
    assert shown["valid"]
    assert shown["skill"] == "opm_flow"
    assert shown["decided_by"] == "rules"
    assert shown["choices"]["ensemble_size"] == 20
    assert "opm_flow" in skills.stdout
    assert "eclipse" in skills.stdout


def test_agent_commands_have_one_no_llm_flag() -> None:
    """A reported issue (2026-10-24): `--help` showed a flag named `--no-no-llm`."""
    import re

    shown = CliRunner().invoke(app, ["agent", "convert", "create", "--help"], terminal_width=200)

    plain = re.sub(r"\x1b\[[0-9;]*m", "", shown.stdout)
    assert "--no-llm" in plain
    assert "--no-no-" not in plain


def test_the_converter_is_in_the_catalogue() -> None:
    assert "case-converter" in {a.name for a in AGENTS}


def test_bounds_must_hold_the_decks_values_in_state_space(small_deck: Path) -> None:
    """Bounds of a log variable are of ln(value): physical ones would clip every member."""
    from apex.core.plugins import converter_skills

    facts = converter_skills.create("opm_flow").read(small_deck)
    physical = _choices(variables=[VariableChoice(property="PERMX", transform="log",
                                                  lower=1.0, upper=1000.0)])  # fmt: skip
    in_ln = _choices(variables=[VariableChoice(property="PERMX", transform="log",
                                               lower=0.0, upper=8.0)])  # fmt: skip

    assert any("are of ln(value)" in p for p in choice_problems(physical, facts))
    assert choice_problems(in_ln, facts) == []


def test_a_draft_that_fails_part_way_leaves_no_files(tmp_path: Path) -> None:
    from apex.agents.converter import write_draft
    from apex.core.conversion import ConversionDraft

    written: list[Path] = []
    draft = ConversionDraft({}, {"a.txt": "a", "../outside.txt": "b"})

    with pytest.raises(ConfigError, match="outside"):
        write_draft(tmp_path / "out", draft, written)

    assert written == [tmp_path / "out" / "a.txt"]
    assert not (tmp_path / "outside.txt").exists()


# ------------------------------------------------------------------ PET (phase 2)

PET = """
[ensemble]
ne = 4
state = ["permx"]
importstate = "prior.npz"

[simulator]
runfile = "RUNFILE"

[dataassim]
scheme = "esmda"
data = "data.csv"
datavar = "var.csv"
[dataassim.mda]
tot_assim_steps = 2
"""


def _pet(folder: Path) -> Path:
    import numpy as np

    folder.mkdir(parents=True)
    (folder / "RUNFILE.mako").write_text("${permx[0]}\n")
    (folder / "data.csv").write_text("dates,WOPR:P1\n2024-01-01,100.0\n")
    (folder / "var.csv").write_text("dates,WOPR:P1\n2024-01-01,\"['abs', 4]\"\n")
    np.savez(folder / "prior.npz", permx=np.zeros((6, 4)))
    (folder / "CONFIG.toml").write_text(PET)
    return folder / "CONFIG.toml"


def test_pet_cases_go_both_ways_from_the_command_line(tmp_path: Path) -> None:
    root = str(tmp_path / "svc")
    runner = CliRunner()

    imported = runner.invoke(app, ["agent", "convert", "from-pet", str(_pet(tmp_path / "pet")),
                                   str(tmp_path / "apex"), "--root", root])  # fmt: skip
    config = str(tmp_path / "apex" / "config.toml")
    exported = runner.invoke(app, ["agent", "convert", "to-pet", config, str(tmp_path / "back"),
                                   "--root", root])  # fmt: skip

    assert imported.exit_code == 0, imported.output
    assert exported.exit_code == 0, exported.output
    pet = tomllib.loads((tmp_path / "back" / "pet.toml").read_text())
    assert pet["dataassim"]["scheme"] == "esmda"
    assert pet["dataassim"]["mda"]["inflation_param"] == [2.0, 2.0]


def test_the_ert_skill_is_registered_by_its_entry_point() -> None:
    """It translates through apex ert convert/export, tested with ERT in tests/integrations."""
    from apex.core.plugins import converter_skills

    skill = converter_skills.create("ert")

    assert skill.capabilities == {"convert_to", "convert_from"}
    assert skill.accepts(Path("case.ert"))
    assert not skill.accepts(Path("CASE.DATA"))


def test_a_deck_goes_to_opm_flow_and_private_skills_stay_private(small_deck: Path) -> None:
    from apex.agents.converter import visible_skills

    converter = CaseConverter(ApexService(small_deck.parent / "s"), None, settings=SMALL)

    assert converter.skill_for(small_deck).name == "opm_flow"
    public = visible_skills(public_only=True)
    assert {"opm_flow", "eclipse", "pet", "ert"} <= set(public)
    assert not {"intersect", "geos"} & set(public)
    with pytest.raises(ConfigError, match="convert from ert"):
        converter.skill_for(small_deck.parent / "case.ert")


def test_installed_advisors_advise_a_skill_and_hear_its_events(
    tmp_path: Path, small_deck: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The converter asks the installed advisors for its skill's advice (given to its
    model, recorded by id) and reports what its code saw; with none installed, nothing."""
    import apex.core.advice as advice

    heard: list[tuple[str, str, str]] = []

    class Fake:
        def prompt(self, agent: str, system: str, root: Path) -> tuple[str, list[int]]:
            return system, []

        def thresholds(
            self, section_name: str, section: object, root: Path
        ) -> tuple[object, list[int]]:
            return section, []

        def skill(self, agent: str, skill: str, root: Path) -> tuple[list[str], list[int]]:
            return ([f"advice for {skill}"], [7]) if agent == "case-converter" else ([], [])

        def observe(self, agent: str, event: advice.AgentEvent, root: Path) -> None:
            heard.append((agent, event.skill, event.key))

    monkeypatch.setattr(advice, "advisors", lambda: (("fake", Fake()),))
    service = ApexService(tmp_path / "service")
    provider = StubProvider([_choices().model_dump_json()])

    outcome = CaseConverter(service, provider, settings=SMALL).convert(small_deck, tmp_path / "o")

    config = tmp_path / "o" / CONFIG
    config.write_text(config.read_text() + '\n[validation]\nwells = ["PROD"]\n')
    with pytest.raises(ConfigError):  # PET has no cross validation: a refusal, reported
        CaseConverter(service, None, settings=SMALL).translate("pet", "to", config, tmp_path / "p")

    task = json.loads(provider.requests[0].messages[-1].content)
    assert task["skill"]["advice"] == ["advice for opm_flow"]
    decision = json.loads(outcome.traces[0].read_text())
    assert "fake:7" in decision["advice"]
    assert [(a, s) for a, s, _ in heard] == [("case-converter", "pet")]
    assert heard[0][2].startswith("refused:to:")


def test_the_example_deck_is_its_template_rendered_and_converts(tmp_path: Path) -> None:
    """The 3dBox example ships a rendered deck to try `convert import` on (a reported issue,
    2026-10-24: the examples had only templates). It stays the template rendered with the
    truth, and converts with the example's observations as its history."""
    import numpy as np

    from apex.core.simulation import MemberInput
    from apex.simulation.deck import render_deck

    example = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production"
    with np.load(example / "truth.npz") as truth:
        state = {"permx": truth["permx"], "poro": truth["poro"]}
    rendered = render_deck(example / "deck.mako", MemberInput(0, state), {"permx_cap": 8.0},
                           case_dir=Path("."))  # fmt: skip
    shipped = (example / "BOX.DATA").read_text()
    assert shipped.startswith("-- BOX.DATA: ")
    assert shipped.endswith(rendered), "re-render BOX.DATA: see the example's README"

    outcome = CaseConverter(ApexService(tmp_path / "service"), None, settings=SMALL).convert(
        example / "BOX.DATA", tmp_path / "box", history=example / "observations.csv"
    )

    assert outcome.valid, outcome.validation.errors
    assert [(v.property, v.transform) for v in outcome.choices.variables] == [
        ("PERMX", "log"), ("PORO", "none")]  # fmt: skip
    assert {f.name for f in (tmp_path / "box").iterdir()} >= {"pvt.txt", "Schdl.sch"}
