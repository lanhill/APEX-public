"""Optimizations drafted from a history-matching case, on a 5Spots-shaped deck.

The deck has 5Spots' shape: four producers on ``WCONHIST`` (``PROD4`` never producing), one
water injector on ``WCONINJE``, monthly ``DATES``, and a SUMMARY without ``FWIT``. The
history match's posterior is a stored ``checkpoints/posterior.npz``; nothing is simulated.

Covered: the schedule read from the deck, the draft (controls, bounds, the forecast's
schedule, the model, the objective counted from the history's end, validation and cost),
its refusals, a guest's allowlist, and the chat's guard: "plan an optimization study in
the 5spots case" never gives a history-matching study (the 2026-10-04 fault).
"""

import datetime as dt
import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pytest

pytest.importorskip("mako")

from apex.agents.chat import ChatAssistant
from apex.agents.providers.stub import StubProvider
from apex.core.llm import Completion, ToolCall
from apex.services.api import ApexService
from apex.services.optimization_draft import DraftRefused, DraftRequest, draft_optimization
from apex.simulation.deck import schedule_summary
from apex.storage.chat_store import ChatStore

WELLS = ("PROD1", "PROD2", "PROD3", "PROD4")
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN")

DECK = """RUNSPEC
DIMENS
 4 3 1 /
OIL
WATER
START
  1 'JAN' 1970 /
GRID
PERMX
% for v in permx:
${"%.6e" % v}
% endfor
/
SUMMARY
FOPT
FWPT
WOPR
 'PROD1' 'PROD2' 'PROD3' 'PROD4'
/
SCHEDULE
WELSPECS
 PROD1 G1 1 1 2000 OIL /
 PROD2 G1 4 1 2000 OIL /
 PROD3 G1 1 3 2000 OIL /
 PROD4 G1 4 3 2000 OIL /
 INJ G1 2 2 2000 WATER /
/
INCLUDE
 '${case_dir}/HISTORY.SCH'/

END
"""


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


def _history() -> str:
    blocks = []
    for k, month in enumerate(MONTHS[1:]):
        rows = [f"{w} OPEN ORAT 0 0 0 /" if w == "PROD4" else
                f"{w} OPEN ORAT {200.0 + 10 * k} 5.0 1000 /" for w in WELLS]  # fmt: skip
        blocks.append("WCONHIST\n" + "\n".join(rows) + "\n/\n"
                      f"WCONINJE\nINJ WATER 1* RATE {800.0 + 50 * k} /\n/\n"
                      f"DATES\n 1 {month} 1970 /\n/\n")  # fmt: skip
    return "".join(blocks)


def five_spots(root: Path, *, posterior: bool = True) -> Path:
    """Write a 5Spots-shaped case under ``root/five`` and return its configuration."""
    case = root / "five"
    (case / "model").mkdir(parents=True)
    (case / "configs").mkdir()
    (case / "model" / "deck.mako").write_text(DECK)
    (case / "model" / "HISTORY.SCH").write_text(_history())
    rows = ["time,datatype,value,variance"]
    rows += [f"1970-0{m}-01,WOPR:PROD1,{200.0 + m},4.0" for m in range(2, 7)]
    (case / "configs" / "observations.csv").write_text("\n".join(rows) + "\n")
    (case / "configs" / "hm.toml").write_text("""
[run]
name = "hm"
seed = 7
output_dir = "../runs/hm"

[[state.variables]]
name = "permx"
size = 12
grid = [4, 3, 1]

[prior]
ensemble_size = 6
generator = { name = "gaussian", options = { mean = 100.0, std = 10.0 } }

[observations]
file = "observations.csv"

[case]
kind = "real"

[simulator]
name = "opm_flow"
options = { deck_template_file = "../model/deck.mako" }

[execution]
name = "process"
options = { workers = 2 }

[method]
name = "esmda"
options = { inflation = [2, 2] }
strategy = { name = "approx" }
""")
    if posterior:
        checkpoints = case / "runs" / "hm" / "checkpoints"
        checkpoints.mkdir(parents=True)
        members = np.arange(12 * 6, dtype=float).reshape(12, 6)
        np.savez(checkpoints / "posterior.npz", permx=members)
    return case / "configs" / "hm.toml"


def _service(root: Path, max_simulator_runs: int = 100000) -> ApexService:
    home = root / "service"
    home.mkdir(exist_ok=True)
    (home / "service.toml").write_text(
        f'[workspace]\nroots = ["{root}"]\n[approval]\nmax_simulator_runs = {max_simulator_runs}\n'
    )
    return ApexService(home)


ECONOMICS: dict[str, Any] = {"wop": 283.0, "wwp": 37.7, "wwi": 12.6, "disc": 0.08,
                             "producer_bhp": (50.0, 300.0)}  # fmt: skip


def test_the_schedule_is_read_from_the_deck_and_its_includes(tmp_path: Path) -> None:
    config = five_spots(tmp_path)

    summary = schedule_summary(config.parent.parent / "model" / "deck.mako")

    assert summary["start"] == dt.date(1970, 1, 1)
    assert summary["dates"] == [dt.date(1970, m, 1) for m in range(2, 7)]
    producers = cast(dict[str, dict[str, float]], summary["producers"])
    assert set(producers) == set(WELLS)
    assert producers["PROD1"]["max_oil_rate"] == 240.0
    assert summary["injectors"] == {"INJ": {"type": "WATER", "max_rate": 1000.0}}
    assert summary["shut"] == ["PROD4"]  # never producing
    assert summary["tstep"] is False


def test_a_draft_continues_the_history_with_controls_and_validates(tmp_path: Path) -> None:
    config = five_spots(tmp_path)
    service = _service(tmp_path)
    out = tmp_path / "drafts"

    made = draft_optimization(service, DraftRequest(
        case=str(config), forecast_end="1971-06-01", control_periods=3, members=4, **ECONOMICS
    ), out)  # fmt: skip

    assert made.wells == {"PROD1": "bhp_PROD1", "PROD2": "bhp_PROD2", "PROD3": "bhp_PROD3",
                          "INJ": "rate_INJ"}  # PROD4 is shut  # fmt: skip
    document = tomllib.loads(made.config.read_text())
    controls = {v["name"]: v for v in document["controls"]["variables"]}
    assert controls["bhp_PROD1"]["initial"] == [175.0] * 3
    assert (controls["bhp_PROD1"]["lower"], controls["bhp_PROD1"]["upper"]) == (50.0, 300.0)
    assert (controls["rate_INJ"]["lower"], controls["rate_INJ"]["upper"]) == (0.0, 2000.0)
    assert controls["rate_INJ"]["initial"] == [1000.0] * 3
    npv = document["objective"]["options"]
    assert document["objective"]["name"] == "npv"
    assert npv["start"] == "1970-01-01"
    assert npv["times"][0] == npv["from_time"] == "1970-06-01"  # NPV counts after history
    assert npv["times"][-1] == "1971-06-01"
    assert len(npv["times"]) == 4
    assert (npv["wop"], npv["wgp"], npv["disc"], npv["obj_scaling"]) == (283.0, 0.0, 0.08, -1e6)
    assert document["optimizer"]["name"] == "enopt"
    assert any("wgp = 0" in a for a in made.assumptions)
    assert "# Assumed: gas is not priced" in made.config.read_text()

    deck = made.deck.read_text()
    # The case's includes, reached from the copy's folder by a relative path.
    assert "'${case_dir}/../five/model/HISTORY.SCH'" in deck
    assert str(tmp_path) not in deck
    assert deck.count("WCONPROD") == 3
    assert deck.count("WCONINJE") == 3
    assert "'PROD1' 'OPEN' BHP 5* ${\"%.6f\" % bhp_PROD1[2]} /" in deck
    assert "'INJ' WATER 'OPEN' RATE ${\"%.6f\" % rate_INJ[0]} /" in deck
    assert "PROD4" not in deck.split("drafted by APEX")[1]
    assert deck.index("drafted by APEX") < deck.rindex("END")
    summary_section = deck.split("SUMMARY")[1].split("SCHEDULE")[0]
    assert "FWIT" in summary_section  # the objective's totals
    assert "FGPT" in summary_section
    assert summary_section.count("FOPT") == 1  # not added twice
    assert (config.parent.parent / "model" / "deck.mako").read_text() == DECK  # unchanged

    with np.load(made.model) as model:
        np.testing.assert_array_equal(model["permx"], np.arange(72.0).reshape(12, 6)[:, :4])
    assert document["controls"]["realizations"] == 4
    assert made.validation.valid
    assert made.validation.optimization is not None
    # The start and EPF's refresh on 4 members, then 5 EnOpt iterations of 10 perturbations
    # and up to 6 trial points on every member (EnOpt.evaluations_per_iteration).
    assert made.validation.optimization.max_simulator_runs == 4 + 4 + 5 * (10 + 6 * 4)


def test_oil_rate_controls_are_bounded_by_the_history(tmp_path: Path) -> None:
    config = five_spots(tmp_path)
    economics = {k: v for k, v in ECONOMICS.items() if k != "producer_bhp"}

    made = draft_optimization(_service(tmp_path), DraftRequest(
        case=str(config), forecast_end="1971-01-01", producer_control="oil_rate", **economics
    ), tmp_path / "drafts")  # fmt: skip

    controls = {v["name"]: v for v in tomllib.loads(made.config.read_text())["controls"][
        "variables"]}  # fmt: skip
    assert (controls["orat_PROD2"]["lower"], controls["orat_PROD2"]["upper"]) == (0.0, 480.0)
    assert "ORAT ${" in made.deck.read_text()


def test_economics_are_never_invented(tmp_path: Path) -> None:
    config = five_spots(tmp_path)

    with pytest.raises(DraftRefused, match="do not invent") as refused:
        draft_optimization(_service(tmp_path), DraftRequest(
            case=str(config), forecast_end="1971-01-01", wop=283.0
        ), tmp_path / "drafts")  # fmt: skip

    assert refused.value.missing[:3] == ["wwp", "wwi", "disc"]
    assert refused.value.missing[3].startswith("producer_bhp")
    assert not (tmp_path / "drafts").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"forecast_end": "1970-03-01"}, "after the history's end, 1970-06-01"),
        ({"producer_bhp": (300.0, 50.0)}, "lower < upper"),
        ({"run": "missing"}, "no finished run"),
    ],
)
def test_impossible_drafts_are_refused_with_the_reason(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    config = five_spots(tmp_path)
    request = {"case": str(config), "forecast_end": "1971-01-01", **ECONOMICS, **change}

    with pytest.raises(DraftRefused, match=message):
        draft_optimization(_service(tmp_path), DraftRequest(**request), tmp_path / "drafts")


def test_without_a_finished_run_the_draft_is_refused(tmp_path: Path) -> None:
    config = five_spots(tmp_path, posterior=False)

    with pytest.raises(DraftRefused, match="run the history match first"):
        draft_optimization(_service(tmp_path), DraftRequest(
            case=str(config), forecast_end="1971-01-01", **ECONOMICS
        ), tmp_path / "drafts")  # fmt: skip


def test_an_analytic_case_has_nothing_to_draft_from(
    tmp_path: Path, linear_case: Callable[..., Case]
) -> None:
    case = linear_case(name="alpha")

    with pytest.raises(DraftRefused, match="no deck simulator"):
        draft_optimization(_service(tmp_path), DraftRequest(
            case=str(case.config), forecast_end="1971-01-01", **ECONOMICS
        ), tmp_path / "drafts")  # fmt: skip


# ---------------------------------------------------------------------- the assistant


def _answer(text: str) -> Completion:
    return Completion(text=json.dumps({"answer": text}))


def test_an_optimization_request_never_gets_a_history_matching_study(tmp_path: Path) -> None:
    """The 2026-10-04 fault: "plan an optimization study in the 5spots case" gave a
    localization study. The guard is in plan_study, whatever the model asks for; with
    the economics given, the assistant drafts an optimization, plans a study of
    optimizations on it, and proposes it as launch_optimization/launch_study."""
    five_spots(tmp_path)
    service = _service(tmp_path)
    store = ChatStore(service.root)
    chat = store.create("optimize")
    economics = {"wop": 283.0, "wwp": 37.7, "wwi": 12.6, "disc": 0.08,
                 "producer_bhp": [50.0, 300.0]}  # fmt: skip
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "five/configs/hm.toml",
            "goal": "plan an optimization study in the 5spots case"}),)),
        Completion(tool_calls=(ToolCall("c2", "draft_optimization", {
            "case": "five/configs/hm.toml", "forecast_end": "1971-06-01"}),)),
        Completion(tool_calls=(ToolCall("c3", "draft_optimization", {
            "case": "five/configs/hm.toml", "forecast_end": "1971-06-01", "members": 2,
            **economics}),)),
        _answer("I drafted the optimization; please confirm the assumptions."),
    ])  # fmt: skip
    assistant = ChatAssistant(service, store, lambda: stub)

    answer = assistant.run(chat.id, "plan an optimization study in the 5spots case")

    # The draft's card was added (the model did not propose it), and the answer says so.
    assert answer.startswith("I drafted the optimization; please confirm the assumptions.\n\n")
    assert answer.endswith("launched from the card.")
    results = [e.data for e in store.events(chat.id) if e.kind == "tool_result"]
    assert results[0]["ok"] is False
    assert "call draft_optimization" in results[0]["error"]
    assert results[1]["ok"] is False
    assert "wwp, wwi, disc, producer_bhp" in results[1]["error"]
    drafted = json.loads(stub.requests[3].messages[-1].content)["result"]
    assert drafted["kind"] == "optimization"
    assert drafted["cost"] == 2 + 2 + 5 * (10 + 6 * 2)
    shown, folder = assistant.drafts(chat.id)
    assert drafted["draft"] == f"{shown}/hm_npv.toml"
    assert sorted(p.name for p in folder.iterdir()) == [
        "hm_npv.toml",
        "hm_npv_deck.mako",
        "hm_npv_model.npz",
    ]  # no study of the case
    [draft_event] = [e.data for e in store.events(chat.id) if e.kind == "draft"]
    assert draft_event == {"draft": drafted["draft"], "cost": 114, "kind": "optimization"}

    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    propose = tools["propose_action"]
    # The turn ended with the draft and no propose_action: its card was shown anyway
    # (a reported issue, 2026-10-24), before the study below is planned and proposed.
    [card] = store.proposals(chat.id)
    assert (card.kind, card.draft, card.state) == (
        "launch_optimization",
        drafted["draft"],
        "pending",
    )
    plan = tools["plan_study"]
    study = plan.handler(plan.arguments(case=drafted["draft"], goal="compare optimizers",
                                        budget=100000))  # fmt: skip
    assert study["kind"] == "optimization"
    propose.handler(propose.arguments(draft=study["draft"], summary="compare"))
    kinds = [(p.kind, p.data["cost"]) for p in store.proposals(chat.id)]
    assert kinds[0] == ("launch_optimization", 114)
    assert kinds[1][0] == "launch_study"
    proposals = [e.data for e in store.events(chat.id) if e.kind == "proposal"]
    assert proposals[0] == {"n": 0, "draft": drafted["draft"], "kind": "launch_optimization"}
    assert service.jobs.all_records() == []  # only a person launches


def _run(root: Path, folder: str, first: float) -> Path:
    """Write a finished run's posterior (12 values, 6 members) under ``root/folder``."""
    checkpoints = root / folder / "checkpoints"
    checkpoints.mkdir(parents=True)
    np.savez(checkpoints / "posterior.npz", permx=first + np.arange(72.0).reshape(12, 6))
    return root / folder


def test_a_guest_drafts_from_their_own_run_and_plugins_only(tmp_path: Path) -> None:
    """For drafts: a guest's model is a run of theirs, never the owner's (the case's
    output_dir, a catalogued run of the owner's, or a path), and their plugin list holds."""
    from apex.core.storage import CatalogueEntry
    from apex.services.access import SUBMITTED_BY, Access, load_access, submitter

    five_spots(tmp_path)
    service = _service(tmp_path)
    store = ChatStore(service.root)
    chat = store.create("guest")
    owners = _run(tmp_path, "owners_run", 1000.0)
    service.store.register(CatalogueEntry("owner1", owners, "hm", "2026-10-04T10:00:00"))
    allow = (
        '[workspace]\nfolders = ["five"]\n[agents]\nnames = ["planner"]\n'
        '[plugins]\nsimulators = ["*"]\nexecutors = ["*"]\nresponse_objectives = ["*"]\n'
    )

    def tool(text: str) -> Any:  # noqa: ANN401
        (service.root / "ui_access.toml").write_text(text)
        access = Access(load_access(service.root), service.jobs.all_records())
        tools = ChatAssistant(service, store, None)._planning(chat.id, access)
        return {t.name: t for t in tools}["draft_optimization"]

    economics = {**ECONOMICS, "producer_bhp": [50.0, 300.0]}
    draft = tool(allow + 'optimizers = ["enopt"]\n')
    arguments = draft.arguments(
        case="five/configs/hm.toml", forecast_end="1971-01-01", members=2, **economics
    )
    with pytest.raises(ValueError, match="no finished run") as refused:
        draft.handler(arguments)  # the owner's runs (output_dir, catalogue) are not theirs
    assert str(tmp_path) not in str(refused.value)
    with pytest.raises(ValueError, match="no finished run"):
        draft.handler(draft.arguments(**{**arguments.model_dump(), "run": str(owners)}))

    theirs = _run(tmp_path, "UI/guests/guest/runs/hm", 0.0)
    service.store.register(CatalogueEntry("guest1", theirs, "hm", "2026-10-04T09:00:00",
                                          {SUBMITTED_BY: submitter("guest")}))  # fmt: skip
    with pytest.raises(ValueError, match="optimizers:enopt"):
        tool(allow).handler(arguments)
    _, folder = ChatAssistant(service, store, None).drafts(chat.id)
    assert not folder.exists() or list(folder.iterdir()) == []  # nothing left behind

    made = draft.handler(arguments)
    assert made["kind"] == "optimization"
    with np.load(folder / "hm_npv_model.npz") as model:
        assert model["permx"][0, 0] == 0.0  # their run's posterior, not the owner's newer one
    written = (folder / "hm_npv.toml").read_text() + (folder / "hm_npv_deck.mako").read_text()
    assert str(tmp_path) not in written  # no server path in a guest's draft
    with pytest.raises(KeyError):  # a folder not on their list
        draft.handler(draft.arguments(case="elsewhere/hm.toml", forecast_end="1971-01-01"))
    # Review of 2026-10-05: the draft copies the case's template into a drafted deck, so a
    # case whose template is text they write (a .toml) is refused before anything is copied.
    case = tmp_path / "five" / "configs"
    (case / "bad.toml").write_text((case / "hm.toml").read_text().replace(
        "../model/deck.mako", "hm.toml").replace('name = "hm"', 'name = "bad"'))  # fmt: skip
    with pytest.raises(ValueError, match="a template must be"):
        draft.handler(draft.arguments(**{**arguments.model_dump(),
                                         "case": "five/configs/bad.toml"}))  # fmt: skip
    assert not (folder / "bad_npv_deck.mako").exists()


def test_a_second_draft_does_not_overwrite_the_first(tmp_path: Path) -> None:
    config = five_spots(tmp_path)
    service = _service(tmp_path)
    request = DraftRequest(case=str(config), forecast_end="1971-01-01", members=2, **ECONOMICS)

    first = draft_optimization(service, request, tmp_path / "drafts")
    second = draft_optimization(service, request, tmp_path / "drafts")

    assert (first.config.name, second.config.name) == ("hm_npv.toml", "hm_npv_2.toml")
    assert second.deck.name == "hm_npv_2_deck.mako"
    with pytest.raises(DraftRefused, match="exists already"):
        draft_optimization(service, request.model_copy(update={"name": "hm_npv"}),
                           tmp_path / "drafts")  # fmt: skip


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ("TSTEP\n 30 /\n", "uses TSTEP"),
        ("WCONINJE\nINJ WATER 1* RATE ${inj_rate} /\n/\n", "INJ are template expressions"),
    ],
)
def test_a_schedule_that_cannot_be_read_is_refused(
    tmp_path: Path, schedule: str, message: str
) -> None:
    config = five_spots(tmp_path)
    history = config.parent.parent / "model" / "HISTORY.SCH"
    history.write_text(history.read_text() + schedule)

    with pytest.raises(DraftRefused, match=message):
        draft_optimization(_service(tmp_path), DraftRequest(
            case=str(config), forecast_end="1971-01-01", **ECONOMICS
        ), tmp_path / "drafts")  # fmt: skip


def test_the_guard_tells_optimizing_from_history_matching_goals() -> None:
    from apex.agents.chat import OPTIMIZING

    for goal in ("plan an optimization study in the 5spots case", "maximise NPV",
                 "a production strategy for 5Spots", "maximize the oil recovery"):  # fmt: skip
        assert OPTIMIZING.search(goal), goal
    for goal in ("maximise the data match", "does localization help", "compare GIES"):
        assert not OPTIMIZING.search(goal), goal


def test_the_command_line_drafts_and_refuses_as_the_tool(tmp_path: Path) -> None:
    """``apex agent draft-optimization``: the same draft, no model; refusals name what is
    missing and exit 1."""
    from typer.testing import CliRunner

    from apex.cli.main import app

    config = five_spots(tmp_path)
    service = _service(tmp_path)
    base = ["agent", "draft-optimization", str(config), "--forecast-end", "1971-01-01",
            "--root", str(service.root), "--out", str(tmp_path / "out")]  # fmt: skip

    refused = CliRunner().invoke(app, [*base, "--wop", "283"])
    assert refused.exit_code == 1
    assert "give wwp, wwi, disc, producer_bhp" in refused.output

    economics = ["--wop", "283", "--wwp", "37.7", "--wwi", "12.6", "--disc", "0.08",
                 "--producer-bhp", "50,300", "--members", "2", "--json"]  # fmt: skip
    made = CliRunner().invoke(app, [*base, *economics])
    assert made.exit_code == 0, made.output
    result = json.loads(made.output)
    assert Path(result["config"]) == tmp_path / "out" / "hm_npv.toml"
    assert result["wells"]["INJ"] == "rate_INJ"
    assert result["cost"] == 2 + 2 + 5 * (10 + 6 * 2)


def test_field_totals_go_into_a_summary_written_with_a_banner() -> None:
    """A deck that writes ``SUMMARY ====...``: the draft refused it as having no SUMMARY."""
    from apex.services.optimization_draft import _with_field_totals

    deck = "RUNSPEC\nSUMMARY =====\nFOPT\nWOPR\n/\nSCHEDULE =====\nFWIT\nEND\n"

    added = _with_field_totals(deck)

    summary = added.split("SUMMARY =====")[1].split("SCHEDULE")[0]
    assert [v for v in ("FOPT", "FGPT", "FWPT", "FWIT") if v in summary] == [
        "FOPT", "FGPT", "FWPT", "FWIT"]  # FWIT in SCHEDULE does not count  # fmt: skip
    assert summary.count("FOPT") == 1
