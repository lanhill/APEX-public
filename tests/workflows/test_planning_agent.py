"""The planning agent, on linear cases and scripted models (no live model).

Independent reference: a run of the linear case with ``N`` members and ES-MDA's four
steps takes ``5 N`` simulator runs (the prior forecast and four updates), so a study's cost
is ``variants x replicates x 5 N``.
"""

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest
from typer.testing import CliRunner

from apex.agents.planner import (
    Planner,
    StudyProposal,
    VariantProposal,
    case_steps,
    level_problems,
    method_differences,
    plan_facts,
    submit,
)
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.services.api import ApexService
from apex.services.study import load_study

TIMEOUT = 180.0
MEMBERS = 20
PER_RUN = 5 * MEMBERS  # ES-MDA with four steps: the prior forecast and four updates


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _service(tmp_path: Path, limit: int = 20_000) -> ApexService:
    root = tmp_path / "service"
    root.mkdir(exist_ok=True)
    (root / "service.toml").write_text(f"[approval]\nmax_simulator_runs = {limit}\n")
    return ApexService(root)


def _proposal(replicates: int = 2, **patches: dict[str, Any]) -> str:
    variants = [VariantProposal(name="baseline", patch={}, rationale="as configured")]
    variants += [VariantProposal(name=n, patch=p, rationale="try it") for n, p in patches.items()]
    return StudyProposal(
        name="loc_test", hypothesis="localization lowers the RMSE", variants=variants,
        replicates=replicates, metrics=["rmse:all"], reasoning="paired replicates",
    ).model_dump_json()  # fmt: skip


LOCALIZED = {"method": {"localization": {"name": "adaptive"}}}


def test_the_facts_cost_one_run_of_the_case(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=MEMBERS)

    facts = plan_facts(_service(tmp_path), case.config, budget=1000)

    assert (facts.method, facts.iterative, facts.localization) == ("esmda", False, None)
    assert facts.runs_per_cell == PER_RUN
    assert facts.max_cells == 1000 // PER_RUN
    assert facts.synthetic
    assert "adaptive" in facts.plugins["localizations"]
    assert facts.budget_simulator_runs == 1000
    assert plan_facts(_service(tmp_path, 777), case.config).budget_simulator_runs == 777


@pytest.mark.parametrize(
    ("budget", "variants", "replicates", "within"),
    [(1000, 2, 5, True), (400, 2, 2, True), (150, 1, 1, True), (50, 1, 1, False)],
)
def test_the_rules_fit_the_budget(
    tmp_path: Path, linear_case: CaseFactory, budget: int, variants: int, replicates: int,
    within: bool,
) -> None:  # fmt: skip
    case = linear_case(ensemble_size=MEMBERS)

    result = Planner(_service(tmp_path), None).plan(case.config, "compare", budget=budget)

    assert result.decided_by == "rules"
    assert len(result.proposal.variants) == variants
    assert result.proposal.replicates == replicates
    assert result.cost == variants * replicates * PER_RUN
    assert result.within_budget is within


def test_a_valid_proposal_is_accepted(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    stub = StubProvider([_proposal(2, localized=LOCALIZED)])

    result = Planner(_service(tmp_path), stub).plan(case.config, "does it help?", budget=1000)

    assert result.decided_by == "model"
    assert result.cost == 2 * 2 * PER_RUN
    assert result.revisions == ()
    [request] = stub.requests
    assert "submit_study" not in [t.name for t in request.tools]
    assert '"runs_per_cell": 100' in request.messages[1].content
    written = tmp_path / "study.toml"
    written.write_text(result.toml())
    spec = load_study(written)
    assert [v.name for v in spec.study.variants] == ["baseline", "localized"]
    assert spec.study.budget.max_simulator_runs == 1000


def test_an_over_budget_or_invalid_proposal_is_revised(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    over = _proposal(5, localized=LOCALIZED)  # 2 x 5 x 100 = 1000 > 600
    seeded = _proposal(1, seeded={"run": {"seed": 3}})  # the study owns run.seed
    stub = StubProvider([over, seeded, _proposal(3)])

    result = Planner(_service(tmp_path), stub).plan(case.config, "x", budget=600)

    assert result.decided_by == "model"
    assert result.proposal.replicates == 3
    assert len(result.revisions) == 2
    assert "over its budget of 600. Runs per replicate" in result.revisions[0]
    assert "over the budget" not in result.revisions[0]  # said once, by the study
    assert "run.seed" in result.revisions[1]
    assert "refused" in stub.requests[1].messages[1].content
    assert result.cost == 3 * PER_RUN


def test_after_the_revisions_the_rules_plan(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    over = _proposal(5, localized=LOCALIZED)

    result = Planner(_service(tmp_path), StubProvider([over] * 3)).plan(
        case.config, "x", budget=600
    )

    assert result.decided_by == "fallback"
    assert len(result.revisions) == 3  # the first proposal and max_revisions = 2 more
    assert result.proposal.reasoning.startswith("rules")
    assert result.within_budget
    assert json.loads(result.trace.read_text())["decided_by"] == "fallback"


def test_submission_needs_a_person_and_the_approval(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    service = _service(tmp_path, limit=150)  # the study (400 runs) needs approval
    result = Planner(service, None).plan(case.config, "x", budget=400)
    study = case.directory / "study.toml"
    study.write_text(result.toml())

    pending = submit(service, study, confirm=lambda q: True)
    assert pending["outcome"] == "pending"
    assert service.jobs.all_records() == []
    token = pending["approval_token"]
    service.approvals.approve(token, by="a person")
    declined = submit(service, study, confirm=lambda q: False, approval_token=token)
    assert declined["outcome"] == "declined"
    assert service.jobs.all_records() == []

    submitted = submit(service, study, confirm=lambda q: True, approval_token=token)

    assert submitted["outcome"] == "submitted"
    assert service.wait(submitted["job_id"], timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    again = submit(service, study, confirm=lambda q: True, approval_token=token)
    assert again["outcome"] == "refused"  # a token works once


def test_below_the_limits_a_confirmed_plan_is_submitted(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    service = _service(tmp_path)
    result = Planner(service, None).plan(case.config, "x", budget=200)
    study = case.directory / "study.toml"
    study.write_text(result.toml())

    assert submit(service, study, confirm=None)["outcome"] == "declined"
    submitted = submit(service, study, confirm=lambda q: True)

    assert submitted["outcome"] == "submitted"
    assert service.wait(submitted["job_id"], timeout_s=TIMEOUT, poll_s=0.2).state == "complete"


def test_the_command_writes_the_study(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    service = _service(tmp_path)
    out = tmp_path / "planned.toml"

    result = CliRunner().invoke(
        app,
        ["agent", "plan", str(case.config), "--goal", "does localization help", "--budget",
         "400", "--no-llm", "--json", "--out", str(out), "--root", str(service.root)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert document["within_budget"]
    assert document["submission"] is None
    assert load_study(out).study.replicates == 2
    assert service.jobs.all_records() == []
    # No server path in the file: outside the workspace, relative to the file.
    assert tomllib.loads(out.read_text())["study"]["cases"] == {"case": "case/config.toml"}


def test_the_command_names_workspace_cases_as_the_ui_shows_them(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=MEMBERS)
    service = _service(tmp_path)
    (service.root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    out = tmp_path / "plans" / "deep" / "planned.toml"
    out.parent.mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        ["agent", "plan", str(case.config), "--goal", "does localization help", "--budget",
         "400", "--no-llm", "--out", str(out), "--root", str(service.root)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert tomllib.loads(out.read_text())["study"]["cases"] == {"case": "case/config.toml"}
    assert ApexService(service.root).validate_study(out).valid  # found in the workspace


def test_a_draft_that_would_lose_a_null_is_refused() -> None:
    """TOML has no null: writing a patch's null would silently keep the key, so the file
    would differ from the study validated (2026-09-30)."""
    from apex.agents.planner import study_toml
    from apex.core.errors import ConfigError
    from apex.domain.study import StudySpec

    def study(x: int | None) -> StudySpec:
        return StudySpec.model_validate({"study": {
            "name": "s", "cases": {"c": "/c.toml"},
            "variants": [{"name": "v", "patch": {"method": {"options": {"x": x}}}}],
        }})  # fmt: skip

    proposal = StudyProposal(
        name="s", hypothesis="h h", replicates=1, metrics=["rmse:all"], reasoning="r r",
        variants=[VariantProposal(name="v", rationale="r r")],
    )  # fmt: skip
    with pytest.raises(ConfigError, match="differ from the validated study"):
        study_toml(study(None), proposal, 10, 100, "rules")
    assert "x = 1" in study_toml(study(1), proposal, 10, 100, "rules")


def test_planned_studies_run_decks_on_opm_flow_unless_asked() -> None:
    """The user's choice (2026-10-01): OPM Flow by default, Eclipse on request."""
    from apex.agents.planner import study_simulator

    assert study_simulator(None, "eclipse") == "opm_flow"
    assert study_simulator("eclipse", "opm_flow") == "eclipse"
    assert study_simulator("case", "eclipse") is None  # each case's own
    assert study_simulator(None, "linear") is None  # no deck: its own


def test_the_draft_names_the_simulators_to_choose_from() -> None:
    from apex.agents.planner import study_toml
    from apex.domain.study import StudySpec

    spec = StudySpec.model_validate({"study": {
        "name": "s", "cases": {"c": "c/config.toml"}, "simulator": "opm_flow",
        "variants": [{"name": "v"}],
    }})  # fmt: skip
    proposal = StudyProposal(
        name="s", hypothesis="h h", replicates=1, metrics=["rmse:all"], reasoning="r r",
        variants=[VariantProposal(name="v", rationale="r r")],
    )  # fmt: skip

    text = study_toml(spec, proposal, 10, 100, "rules", ["eclipse", "linear", "opm_flow"])

    assert '# Simulator: [study] simulator = "opm_flow"; the deck simulators are eclipse, ' in text
    assert 'simulator = "opm_flow"' in text.split("[study]", 1)[1]
    assert 'c = "c/config.toml"' in text


@pytest.mark.parametrize(
    ("run_name", "case"),
    [
        ("m\u00e9thode 1", "m_thode_1"),
        ("..", "case"),
        (".", "case"),
        ("", "case"),
        ("...", "..."),  # the study's rule refuses only "." and ".."
        ("ok.name-2", "ok.name-2"),
    ],
)
def test_the_planners_case_name_is_one_the_study_accepts(run_name: str, case: str) -> None:
    # Case names are ASCII and never dots alone (domain/study.py): the planner's must be too.
    from types import SimpleNamespace

    from apex.agents.planner import _case_name
    from apex.domain.study import StudySpec

    config = SimpleNamespace(run=SimpleNamespace(name=run_name))
    name = _case_name(config)  # type: ignore[arg-type]
    assert name == case
    StudySpec.model_validate(
        {"study": {"name": "s", "cases": {name: "c.toml"}, "variants": [{"name": "v"}]}}
    )


class _FakeErt:
    """An ``ert`` engine that can run every case (the real one needs a deck simulator)."""

    name = "ert"

    def check(self, config: object, method: object) -> list[str]:
        return []

    def simulator_runs(self, method: object, members: int | None) -> int | None:
        return None if members is None else 5 * members

    def run(self, config: object, method: object, context: object) -> object:
        raise AssertionError("the planner never runs anything")


def test_a_goal_about_ert_plans_apex_against_erts_es_mda(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow-up: with the ert engine, the rules compare the two engines."""
    from apex.core.plugins import engines

    # The installed engines first: loaded after the fake, ERT's entry point would collide
    # with it (DuplicatePluginError). An earlier test in the process used to load them, so
    # the test failed alone, and in a pytest-xdist worker (CI, 2026-10-12).
    engines.names()
    monkeypatch.setitem(engines._factories, "ert", _FakeErt)
    case = linear_case(ensemble_size=MEMBERS)
    service = _service(tmp_path)

    facts = plan_facts(service, case.config, budget=1000)
    result = Planner(service, None).plan(
        case.config, "How does APEX compare with ERT?", budget=1000
    )

    assert facts.engines == {"ert": ["es_mda", "ensemble_smoother"]}
    methods = result.proposal.methods
    assert set(methods) == {"esmda", "ert_es_mda"}
    assert methods["ert_es_mda"].engine == "ert"
    assert methods["ert_es_mda"].options == {"weights": [4, 4, 4, 4]}  # the case's inflation
    assert result.decided_by == "rules"
    assert result.validation.valid, result.validation.errors
    # Without the word ERT in the goal, the rules plan as before.
    plain = Planner(service, None).plan(case.config, "compare", budget=1000)
    assert plain.proposal.methods == {}


def _methods_proposal(methods: dict[str, Any]) -> StudyProposal:
    return StudyProposal.model_validate({
        "name": "algorithms", "hypothesis": "compare", "replicates": 1,
        "metrics": ["rmse:all"], "methods": methods, "reasoning": "compare the algorithms",
        "variants": [{"name": "baseline", "patch": {}, "rationale": "as configured"}],
    })  # fmt: skip


ESMDA4 = {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]}}


def test_a_misspelt_method_is_revised_not_a_crash(tmp_path: Path, linear_case: CaseFactory) -> None:
    """The review of #173: matching looked the name up and raised on an unknown one."""
    case = linear_case(ensemble_size=MEMBERS)
    bad = _methods_proposal({"esmda": ESMDA4, "gies": {"name": "gise"}})
    good = _methods_proposal({"esmda": ESMDA4, "gies": {"name": "gies"}})
    stub = StubProvider([bad.model_dump_json(exclude_none=True),
                         good.model_dump_json(exclude_none=True)])  # fmt: skip

    result = Planner(_service(tmp_path), stub).plan(case.config, "compare", budget=5000)

    assert result.decided_by == "model"
    assert len(result.revisions) == 1
    assert "gise" in result.revisions[0]


def test_the_cases_budget_in_updates(tmp_path: Path, linear_case: CaseFactory) -> None:
    """ES makes one update; ES-MDA one per inflation; an iterative case max_iterations
    (the review of #173: ES counted the method section's default of 10)."""
    service = _service(tmp_path)
    es = linear_case(name="es", ensemble_size=MEMBERS, method='name = "es"')
    esmda = linear_case(name="esmda", ensemble_size=MEMBERS)
    gies = linear_case(name="gies", ensemble_size=MEMBERS,
                       method='name = "gies"\nmax_iterations = 3')  # fmt: skip

    assert case_steps(plan_facts(service, es.config, budget=5000)) == 1
    assert case_steps(plan_facts(service, esmda.config, budget=5000)) == 4
    assert case_steps(plan_facts(service, gies.config, budget=5000)) == 3


def test_the_models_compared_algorithms_are_left_as_given_and_their_differences_recorded(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue, with the user's rule that each algorithm is its own base:
    a GIES arm against the case's ES-MDA keeps its own defaults; how it differs (its
    iteration limit, its strategy) is a recorded action of the decision and comes back
    in the result, for the user to decide."""
    case = linear_case(ensemble_size=MEMBERS)
    proposal = _methods_proposal(
        {"esmda": {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]},
                   "strategy": {"name": "approx", "options": {"energy": 1.0}}},
         "gies": {"name": "gies"}, "gies3": {"name": "gies", "max_iterations": 3}},
    ).model_dump_json(exclude_none=True)  # fmt: skip

    result = Planner(_service(tmp_path), StubProvider([proposal])).plan(
        case.config, "compare GIES", budget=5000
    )

    assert result.decided_by == "model"
    assert result.proposal.methods["gies"].max_iterations is None  # its own default
    assert result.proposal.methods["gies"].strategy is None
    assert "gies runs to its own iteration limit, the case 4 steps" in result.differences
    assert "gies3 runs up to 3 iterations, the case 4 steps" in result.differences
    assert any(d.startswith("gies's strategy is rlmmac (its default); the case's is approx")
               for d in result.differences)  # fmt: skip
    # The draft names the arms' strategies (#225, #231): the default, never the case's.
    service = _service(tmp_path)
    facts = plan_facts(service, case.config, budget=5000)
    methods = Planner(service, None).spec(facts, result.proposal).study.methods
    names = {k: None if m.strategy is None else m.strategy.name for k, m in methods.items()}
    assert names == {"esmda": "approx", "gies": "rlmmac", "gies3": "rlmmac"}
    assert not any(d.startswith("esmda") for d in result.differences)
    actions = [json.loads(p.read_text()) for p in (tmp_path / "service" / "agents").rglob("*.json")]
    [recorded] = [a for a in actions if "differ from the case" in str(a.get("action"))]
    assert recorded["differences"] == list(result.differences)


def test_the_differences_name_an_engines_schedule_and_es_mdas_steps(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    service = _service(tmp_path)
    case = linear_case(name="gies", ensemble_size=MEMBERS,
                       method='name = "gies"\nmax_iterations = 2')  # fmt: skip
    facts = plan_facts(service, case.config, budget=5000)
    proposal = _methods_proposal({
        "esmda": {"name": "esmda", "options": {"inflation": [3, 3, 3]}},
        "ert": {"engine": "ert", "name": "es_mda", "options": {"weights": [4, 4, 4, 4]}},
    })  # fmt: skip
    differences = method_differences(facts, proposal)
    assert "esmda runs 3 steps, the case 2 iterations" in differences
    assert "ert runs 4 steps, the case 2 iterations" in differences


def test_an_es_arm_counts_its_one_update(tmp_path: Path, linear_case: CaseFactory) -> None:
    """The review of #173: ES takes no inflation; it runs one update, against an ES case
    no difference, against an ES-MDA case of four steps the numbers."""
    service = _service(tmp_path)
    es_case = linear_case(name="es", ensemble_size=MEMBERS, method='name = "es"')
    esmda_case = linear_case(name="esmda", ensemble_size=MEMBERS)
    proposal = _methods_proposal({"es": {"name": "es"}})
    on_es = method_differences(plan_facts(service, es_case.config, budget=5000), proposal)
    on_esmda = method_differences(plan_facts(service, esmda_case.config, budget=5000), proposal)
    assert not any(d.startswith("es runs") for d in on_es)
    assert "es runs 1 step, the case 4 steps" in on_esmda


def test_a_localization_goal_compares_every_adaptive_localization_the_method_takes(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#201): a localization study compared one default variant."""
    case = linear_case(ensemble_size=MEMBERS)

    result = Planner(_service(tmp_path), None).plan(
        case.config, "which localization helps", budget=50_000
    )

    names = [v.patch["method"]["localization"]["name"] for v in result.proposal.variants[1:]
             if "prior" not in v.patch]  # fmt: skip
    assert names[0] == "adaptive"  # the method's own first (adaptive_prior's alias)
    assert {"adaptive_per_update", "adaptive_projected"} <= set(names)  # ES-MDA takes them
    assert result.within_budget
    plain = Planner(_service(tmp_path), None).plan(case.config, "compare", budget=50_000)
    assert len(plain.proposal.variants) == 2  # another goal: the method's one localization


def test_an_unset_arm_strategy_is_named_with_its_methods_own_options(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #225/#231: LM-EnRML's default approx keeps 95 % of the energy, approx's
    own default 98 %; naming it bare would change the run. And an unset arm whose default
    differs from the case's options (energy 0.99) is said to."""
    service = _service(tmp_path)
    case = linear_case(ensemble_size=MEMBERS, method=(
        'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }'))  # fmt: skip
    facts = plan_facts(service, case.config, budget=5000)
    proposal = _methods_proposal({"lm": {"name": "lm_enrml"}, "esmda": ESMDA4})

    methods = Planner(service, None).spec(facts, proposal).study.methods
    differences = method_differences(facts, proposal)

    assert methods["lm"].strategy is not None
    assert methods["lm"].strategy.name == "approx"
    assert methods["lm"].strategy.options == {"energy": 0.95}
    assert any(d.startswith("esmda's approx (its default) options differ from the case's")
               for d in differences)  # fmt: skip
    assert any(d.startswith("lm's approx (its default) options differ") for d in differences)


def test_a_localization_variant_keeps_the_methods_own_strategy_options(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #201: on an LM-EnRML case without a strategy, the localization patch
    names approx with LM-EnRML's energy 0.95, as the baseline runs it."""
    case = linear_case(ensemble_size=MEMBERS, method='name = "lm_enrml"\nmax_iterations = 2')

    result = Planner(_service(tmp_path), None).plan(case.config, "localization", budget=50_000)

    for variant in [v for v in result.proposal.variants[1:] if "method" in v.patch]:
        assert variant.patch["method"]["strategy"] == {"name": "approx",
                                                      "options": {"energy": 0.95}}  # fmt: skip


def test_a_localization_study_runs_every_arm_at_both_sizes_or_at_the_cases_alone(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (2026-10-24): a localization study at the case's size alone
    cannot show where localization helps, so the arms run again at half the size; and the
    reported issue #246: only some arms at the smaller size answered neither question, so every
    arm runs at both sizes when they all fit, else every localization at the case's."""
    from apex.agents.planner import rules

    service = _service(tmp_path)
    case = linear_case(ensemble_size=20)
    facts = plan_facts(service, case.config, budget=50_000)
    goal = "does localization help"

    def plan(cells: int, variants: int) -> dict[str, dict[str, Any]]:
        update = {"max_cells": cells, "max_variants": variants}
        return {v.name: v.patch for v in rules(facts.model_copy(update=update), goal).variants}

    arms = [n for n in plan(1000, 1000) if "small" not in n]
    assert len(arms) >= 3  # the baseline and every adaptive localization the method takes
    both = plan(1000, 1000)
    assert list(both) == arms + [f"{a}_small_ensemble" for a in arms]
    for arm in arms:
        small = both[f"{arm}_small_ensemble"]
        assert small["prior"] == {"ensemble_size": 10}
        assert small.get("method") == both[arm].get("method")
    # Too few variants or runs for both sizes: every arm at the case's size, none smaller.
    assert list(plan(1000, 2 * len(arms) - 1)) == arms
    assert list(plan(4 * len(arms) - 1, 1000)) == arms
    # An unknown cost has no budget to fit both sizes in: every arm once, none trimmed.
    unknown = rules(facts.model_copy(update={"max_cells": None, "max_variants": 1000}), goal)
    assert [v.name for v in unknown.variants] == arms
    # A case with its own localization gets no smaller arms.
    localized = rules(
        facts.model_copy(
            update={"localization": "adaptive_prior", "max_cells": 1000, "max_variants": 1000}
        ),
        goal,
    )
    assert not any("small" in v.name for v in localized.variants)
    # Not for another goal, nor below ten members.
    other = Planner(service, None).plan(case.config, "compare", budget=50_000)
    assert not any("small" in v.name for v in other.proposal.variants)
    tiny = linear_case(name="tiny", ensemble_size=18)
    short = Planner(service, None).plan(tiny.config, goal, budget=50_000)
    assert not any("small" in v.name for v in short.proposal.variants)


def test_the_smaller_ensemble_is_left_out_where_it_cannot_run(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Its review: a stored prior or multilevel levels fix the ensemble size."""
    from apex.agents.planner import _smaller_ensemble, rules

    facts = plan_facts(_service(tmp_path), linear_case(ensemble_size=20).config, budget=50_000)
    goal = "does localization help"
    arms = rules(facts, goal).variants[:2]
    assert len(_smaller_ensemble(facts, goal, arms)) == 2
    config = facts.configuration
    stored = {**config, "prior": {**config["prior"], "generator": {"name": "file"}}}
    assert _smaller_ensemble(facts.model_copy(update={"configuration": stored}), goal,
                             arms) == []  # fmt: skip
    per_variable = {
        **config,
        "prior": {**config["prior"], "variables": {"x": {"generator": {"name": "file"}}}},
    }
    assert _smaller_ensemble(facts.model_copy(update={"configuration": per_variable}), goal,
                             arms) == []  # fmt: skip
    multilevel = {**facts.case_method, "options": {"levels": [{"members": 10}]}}
    assert _smaller_ensemble(facts.model_copy(update={"case_method": multilevel}), goal,
                             arms) == []  # fmt: skip


def test_a_multilevel_arm_is_refused_without_two_levels_in_one_message(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#232): the study's validation named a level's keys one at a time
    (four refusals), and the design that passed had one level of all 20 members, which is
    ES-MDA on the hybrid update. The planner states the whole format at once and refuses
    one level; two levels of the fine model are allowed, and said to be no cheaper."""
    service = _service(tmp_path)
    case = linear_case(ensemble_size=MEMBERS)
    facts = plan_facts(service, case.config, budget=5000)
    planner = Planner(service, None)

    for levels in (None, [{"members": MEMBERS, "weight": 1.0}]):
        options: dict[str, Any] = {"inflation": [4, 4, 4, 4]}
        if levels is not None:
            options["levels"] = levels
        proposal = _methods_proposal(
            {"esmda": ESMDA4, "ml": {"name": "multilevel_esmda", "options": options}}
        )
        [problem] = level_problems(facts, proposal)
        assert "at least two" in problem
        assert f"adding up to the case's {MEMBERS}" in problem
        assert "{ members = 10, weight = 0.5 }" in problem  # an example that fits the case
        assert "on linear: forward_file or matrix" in problem  # what makes a level coarse
        assert ("one level" in problem) == (levels is not None)
        spec, _, _, errors = planner.check(facts, proposal)
        assert spec is None
        assert errors == [problem]

    two = [{"members": 10, "weight": 0.5}, {"members": 10, "weight": 0.5}]
    proposal = _methods_proposal(
        {
            "esmda": ESMDA4,
            "ml": {
                "name": "multilevel_esmda",
                "options": {"inflation": [4, 4, 4, 4], "levels": two},
            },
        }
    )
    assert level_problems(facts, proposal) == []
    spec, _, _, errors = planner.check(facts, proposal)
    assert spec is not None
    assert errors == []
    assert any(d.startswith("every level of ml runs the fine model")
               for d in method_differences(facts, proposal))  # fmt: skip
    coarse = [{**two[0], "matrix": [[1.0]] * 1}, two[1]]
    proposal = _methods_proposal(
        {
            "ml": {
                "name": "multilevel_esmda",
                "options": {"inflation": [4, 4, 4, 4], "levels": coarse},
            }
        }
    )
    assert not any(d.startswith("every level") for d in method_differences(facts, proposal))
    # Other errors stay with it (its review: one refusal, not one per round).
    narrow = facts.model_copy(update={"max_variants": 1})
    proposal = _methods_proposal({"esmda": ESMDA4, "ml": {"name": "multilevel_esmda"}})
    _, _, _, errors = planner.check(narrow, proposal)
    assert len(errors) == 2
    assert "at most 1" in errors[0]
    assert "has no levels" in errors[1]
    # The case's own multilevel design is the case's: accepted as it is, with no note.
    own = {"name": "multilevel_esmda", "options": {"inflation": [4, 4, 4, 4],
           "levels": [{"members": MEMBERS, "weight": 1.0}]}}  # fmt: skip
    mine = facts.model_copy(update={"method": "multilevel_esmda", "case_method": own})
    proposal = _methods_proposal({"ml": own, "esmda": ESMDA4})
    assert level_problems(mine, proposal) == []
    assert not any(d.startswith("every level") for d in method_differences(mine, proposal))
