"""Optimizations in the web UI follow history matching's rules.

A guest submits a study of optimizations from the workspace: its cells run in the guest's
own folder (``UI/guests/<name>/studies/...``), carry the guest's annotation, are checked
against the allowlist cell by cell, are invisible to other guests, can be diagnosed by
their owner only, and are deleted after confirming. The planner writes a guest's draft in
their folder and refuses plugins the guest may not use.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.services.api import ApexService
from apex.services.auth import ui_token

TIMEOUT = 180.0


class Service:
    """A service root whose workspace is ``tmp_path``, with the guests' tokens."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "service"
        self.root.mkdir()
        (tmp_path / "alpha").mkdir()
        (self.root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
        self.service = ApexService(self.root)
        self.http = TestClient(create_app(self.root))

    def guest(self, name: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {ui_token(self.root, create=True, guest=name)}"}

    def file(self, name: str, text: str) -> str:
        path = self.tmp / "alpha" / name
        path.write_text(text)
        return str(path)

    def wait(self, job: str) -> None:
        assert self.service.wait(job, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"


@pytest.fixture
def svc(tmp_path: Path) -> Service:
    return Service(tmp_path)


ACCESS = """
[workspace]
folders = ["alpha"]

[plugins]
optimizers = [{optimizers}]
response_objectives = ["npv"]
simulators = ["toy_reservoir"]
executors = ["*"]

[agents]
names = ["diagnosis", "planner"]

[actions]
submit = true
"""

CASE = """
[optimization]
name = "toy"
seed = 3

[[controls.variables]]
name = "inj"
initial = [300.0, 250.0]
lower = 10.0
upper = 400.0
variance = 100.0

[objective]
name = "npv"
options = {{ start = "2000-01-01", times = ["2001-01-01", "2002-01-01"], wop = 1.0, \
wwi = 1.0, obj_scaling = -1.0 }}

[optimizer]
name = "enopt"
options = {{ ne = 4, maxiter = 2, alpha = 0.1 }}

[simulator]
name = "{simulator}"
"""

STUDY = """
[study]
kind = "optimization"
name = "{name}"

[study.cases]
toy = "opt.toml"

[[study.variants]]
name = "base"
{variant}
"""


def _setup(svc: Service, optimizers: str = '"enopt", "smcopt"') -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS.format(optimizers=optimizers))
    svc.file("opt.toml", CASE.format(simulator="toy_reservoir"))


def test_a_guests_optimization_study_runs_in_their_folder_and_is_theirs(
    svc: Service,
) -> None:
    _setup(svc)
    http, alice, bob = svc.http, svc.guest("alice"), svc.guest("bob")
    study = svc.file("opt_study.toml", STUDY.format(name="optst", variant=""))
    handle = http.post("/v1/studies", headers=alice, json={"study_file": study})
    assert handle.status_code == 202, handle.text
    svc.wait(handle.json()["id"])
    cell = svc.tmp / "UI/guests/alice/studies/optst/toy/base/r0"
    assert (cell / "history.jsonl").exists()
    assert not (svc.root / "studies" / "optst" / "toy").exists()  # not the shared folder
    records = [r for r in svc.service.jobs.all_records() if r["kind"] == "optimization"]
    assert all(r["annotations"]["submitted_by"] == "guest:alice" for r in records)
    # Bob sees nothing of it, and may not diagnose it; Alice may.
    seen = {j["id"] for j in http.get("/v1/jobs", headers=bob).json()["items"]}
    assert handle.json()["id"] not in seen
    body = {"optimization": str(cell), "model": False}
    assert http.post("/v1/agents/diagnose", headers=bob, json=body).status_code == 404
    mine = http.post("/v1/agents/diagnose", headers=alice, json=body)
    assert mine.status_code == 200, mine.text
    assert mine.json()["decided_by"] == "rules"
    # Deleted after confirming, folders and records.
    preview = http.get("/v1/deletions", headers=alice, params={"study": "optst"}).json()
    assert preview["blocked"] == []
    assert any(d.endswith("studies/optst/toy/base/r0") for d in preview["dirs"])
    done = http.post("/v1/deletions", headers=alice,
                     json={"job": preview["job"], "token": preview["token"]})  # fmt: skip
    assert done.status_code == 200, done.text
    assert not cell.exists()
    assert not (svc.root / "studies" / "optst").exists()


def test_a_cell_with_a_plugin_the_guest_may_not_use_is_refused(svc: Service) -> None:
    _setup(svc)
    study = svc.file(
        "opt_study.toml",
        STUDY.format(name="sneaky", variant='patch = { simulator = { name = "linear" } }'),
    )
    response = svc.http.post("/v1/studies", headers=svc.guest("alice"), json={"study_file": study})
    assert response.status_code == 403
    assert "simulators:linear" in response.text


def test_the_planner_writes_a_guests_draft_and_keeps_to_their_plugins(
    svc: Service,
) -> None:
    _setup(svc)
    http, alice = svc.http, svc.guest("alice")
    body = {"case": "alpha/opt.toml", "goal": "which optimizer", "budget": 400, "model": False}
    planned = http.post("/v1/agents/plan", headers=alice, json=body)
    assert planned.status_code == 200, planned.text
    draft = planned.json()["draft"]
    assert draft.startswith("UI/guests/alice/drafts/planner/")
    text = (svc.tmp / draft).read_text()
    assert 'toy = "alpha/opt.toml"' in text  # as the workspace shows it: no server path
    assert str(svc.tmp) not in text
    # Without SmcOpt in her list, the rules' comparison uses a plugin she may not.
    _setup(svc, optimizers='"enopt"')
    refused = http.post("/v1/agents/plan", headers=alice, json=body)
    assert refused.status_code == 403
    assert "optimizers:smcopt" in refused.text


def test_the_owners_web_ui_study_goes_to_their_folder_and_a_scripts_stays(svc: Service) -> None:
    from apex.services.auth import owner_ui_token

    _setup(svc)
    owner_ui = {"Authorization": f"Bearer {owner_ui_token(svc.root, create=True)}",
                "X-APEX-Client": "ui"}  # fmt: skip
    study = svc.file("opt_study.toml", STUDY.format(name="mine", variant=""))
    body = {"study_file": study, "client": "ui"}
    handle = svc.http.post("/v1/studies", headers=owner_ui, json=body)
    assert handle.status_code == 202, handle.text
    svc.wait(handle.json()["id"])
    assert (svc.tmp / "UI/owner/studies/mine/toy/base/r0/history.jsonl").exists()
    script = svc.file("opt_study2.toml", STUDY.format(name="script", variant=""))
    done = svc.service.submit_study(Path(script))
    svc.wait(done.id)
    assert (svc.root / "studies/script/toy/base/r0/history.jsonl").exists()


def test_a_study_name_is_one_persons_and_cases_lie_in_the_listed_folders(
    svc: Service,
) -> None:
    """Review of 2026-10-04: a study's record is keyed by its name, so a second person may
    not reuse it; and a study's cases must lie in the guest's listed folders."""
    _setup(svc)
    http, alice, bob = svc.http, svc.guest("alice"), svc.guest("bob")
    study = svc.file("opt_study.toml", STUDY.format(name="taken", variant=""))
    first = http.post("/v1/studies", headers=alice, json={"study_file": study})
    assert first.status_code == 202, first.text
    svc.wait(first.json()["id"])
    clash = http.post("/v1/studies", headers=bob, json={"study_file": study})
    assert clash.status_code in (400, 409, 422), clash.text
    assert "another name" in clash.text
    # A case outside alpha (the listed folder), named from a study inside it.
    outside = svc.tmp / "beta"
    outside.mkdir()
    (outside / "opt.toml").write_text(CASE.format(simulator="toy_reservoir"))
    sneaky = svc.file(
        "opt_study_out.toml",
        STUDY.format(name="out", variant="").replace(
            'toy = "opt.toml"', 'toy = "../beta/opt.toml"'
        ),
    )
    refused = http.post("/v1/studies", headers=alice, json={"study_file": sneaky})
    assert refused.status_code in (403, 404), refused.text


def test_a_history_matching_studys_case_must_lie_in_the_listed_folders(svc: Service) -> None:
    """The same check for a history-matching study (review of 2026-10-04)."""
    _setup(svc)
    outside = svc.tmp / "beta"
    outside.mkdir()
    (outside / "run.toml").write_text("[run]\nname = 'x'\n")
    study = svc.file(
        "hm_study.toml",
        '[study]\nname = "hm"\n[study.cases]\ncase = "../beta/run.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    refused = svc.http.post("/v1/studies", headers=svc.guest("alice"), json={"study_file": study})
    assert refused.status_code in (403, 404, 422), refused.text
    assert refused.status_code != 422 or "beta" not in refused.text  # no server path


def test_the_editor_validates_an_optimization_studys_text(svc: Service) -> None:
    """POST /v1/configs:validate with config_text (the Editor's buffer): an optimization
    study validates with its notes, and a cell with a plugin the guest may not use is an
    error, as for history-matching studies (a review of 2026-10-04)."""
    _setup(svc)
    alice = svc.guest("alice")
    text = STUDY.format(name="checked", variant="")
    body = {"config_text": text, "base_dir": str(svc.tmp / "alpha")}
    checked = svc.http.post("/v1/configs:validate", headers=alice, json=body)
    assert checked.status_code == 200, checked.text
    assert checked.json()["valid"], checked.json()
    assert any("simulator runs" in note for note in checked.json()["notes"])
    # A valid cell with an objective the guest's list does not have (ren_npv).
    patch = 'patch = { objective = { name = "ren_npv", options = { wwi = 0.0 } } }'
    sneaky = STUDY.format(name="sneaky", variant=patch)
    refused = svc.http.post("/v1/configs:validate", headers=alice,
                            json={**body, "config_text": sneaky})  # fmt: skip
    assert refused.status_code == 200, refused.text
    assert not refused.json()["valid"]
    assert "response_objectives:ren_npv" in " ".join(refused.json()["errors"])
    by_file = svc.http.post("/v1/configs:validate", headers=alice,
                            json={"config_file": svc.file("s.toml", text)})  # fmt: skip
    assert by_file.json()["valid"], by_file.json()


def test_the_control_maps_are_optimizations_plugins(svc: Service) -> None:
    """Their registry is history matching's, their workflow optimization's (per plugin)."""
    workflow = {(p.registry, p.name): p.workflow for p in svc.service.plugins()}
    assert workflow["parameterizations", "control_schedule"] == "optimization"
    assert workflow["parameterizations", "control_scaling"] == "optimization"
    assert workflow["parameterizations", "pgs_truncation"] == "history_matching"


def test_validation_is_safe_for_guests(svc: Service) -> None:
    """Review of 2026-10-04: malformed text is a validation error (not a 500), a case
    outside the listed folders is refused before it is read (by text and by file, for both
    kinds of study), and no server path is shown."""
    _setup(svc)
    http, alice = svc.http, svc.guest("alice")
    base = str(svc.tmp / "alpha")
    for text in ('[study]\nkind = "optimization"\nname = "x"\nbogus = 1\n',
                 'title = "x"\n' + STUDY.format(name="y", variant="")):  # fmt: skip
        bad = http.post("/v1/configs:validate", headers=alice,
                        json={"config_text": text, "base_dir": base})  # fmt: skip
        assert bad.status_code == 200, bad.text
        assert not bad.json()["valid"]
    outside = svc.tmp / "beta"
    outside.mkdir()
    (outside / "opt.toml").write_text(CASE.format(simulator="toy_reservoir"))
    for study in (
        STUDY.format(name="o", variant="").replace('toy = "opt.toml"', 'toy = "../beta/opt.toml"'),
        '[study]\nname = "h"\n[study.cases]\ncase = "../beta/opt.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    ):
        by_text = http.post("/v1/configs:validate", headers=alice,
                            json={"config_text": study, "base_dir": base})  # fmt: skip
        assert by_text.status_code in (403, 404), by_text.text
        by_file = http.post("/v1/configs:validate", headers=alice,
                            json={"config_file": svc.file("probe.toml", study)})  # fmt: skip
        assert by_file.status_code in (403, 404), by_file.text
        for response in (by_text, by_file):
            assert str(svc.tmp) not in response.text  # the case as written, no server path
        submitted = http.post("/v1/studies", headers=alice,
                              json={"study_file": svc.file("probe.toml", study)})  # fmt: skip
        assert submitted.status_code in (403, 404), submitted.text
        assert str(svc.tmp) not in submitted.text
    missing = STUDY.format(name="m", variant="").replace('toy = "opt.toml"', 'toy = "gone.toml"')
    shown = http.post("/v1/configs:validate", headers=alice,
                      json={"config_text": missing, "base_dir": base})  # fmt: skip
    assert shown.status_code in (200, 404), shown.text
    assert str(svc.tmp) not in shown.text


def test_the_owners_json_config_of_an_optimization_study_validates(svc: Service) -> None:
    import tomllib

    _setup(svc)
    from apex.services.auth import api_token

    owner = {"Authorization": f"Bearer {api_token(svc.root)}"}
    document = tomllib.loads(STUDY.format(name="j", variant=""))
    ok = svc.http.post("/v1/configs:validate", headers=owner,
                       json={"config": document, "base_dir": str(svc.tmp / "alpha")})  # fmt: skip
    assert ok.status_code == 200, ok.text
    assert ok.json()["valid"], ok.json()
    bad = svc.http.post("/v1/configs:validate", headers=owner,
                        json={"config": {"study": {"kind": "optimization", "bogus": 1}},
                              "base_dir": str(svc.tmp / "alpha")})  # fmt: skip
    assert bad.status_code == 422


def test_the_assistants_design_of_an_optimization_study_is_kept_and_revisable(
    svc: Service,
) -> None:
    """A reported issue: plan_study kept only the rules' enopt-vs-smcopt plan,
    and revise_draft refused a study of optimizations as a history-matching study."""
    from apex.agents.chat import ChatAssistant
    from apex.storage.chat_store import ChatStore

    _setup(svc)
    store = ChatStore(svc.root)
    assistant = ChatAssistant(svc.service, store, None)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan, revise = tools["plan_study"], tools["revise_draft"]
    smcopt = {"name": "smcopt", "options": {"ne": 4, "maxiter": 2}}
    result = plan.handler(plan.arguments(
        case="alpha/opt.toml", goal="how much does smcopt vary", budget=400,
        methods={"smc": smcopt}, replicates=2, name="smc_only",
    ))  # fmt: skip
    assert result["methods"] == {"smc": smcopt}  # not the rules' enopt against smcopt
    assert result["replicates"] == 2
    # A reported issue: the comparison's metrics, and those it could use.
    assert result["metrics"] == ["objective:final", "objective:best", "evaluations:all"]
    assert {"objective:final", "iterations:all", "hypervolume:final"} <= set(
        result["metrics_available"]
    )  # full keys: the comparison finds only these
    assert result["held_fixed"]["optimizer.name"] == "smcopt"  # one arm: all of it is fixed
    assert result["differs"] == {}
    assert len(result["seeds"]) == 2  # r0 and r1
    # Its approval is reported at planning too (a reported issue): two runs.
    assert result["approval"]["runs"] == 2
    assert {"needed", "limits", "simulator_runs"} <= set(result["approval"])
    text = (svc.tmp / result["draft"]).read_text()
    assert 'kind = "optimization"' in text
    revised = revise.handler(revise.arguments(draft=result["draft"], text=text + "\n# ok\n"))
    assert "+# ok" in revised["diff"]
    assert revised["held_fixed"] == result["held_fixed"]
    with pytest.raises(ValueError, match="belong to history matching"):
        plan.handler(plan.arguments(
            case="alpha/opt.toml", goal="compare", budget=400,
            methods={"smc": {**smcopt, "max_iterations": 3}},
        ))  # fmt: skip
    with pytest.raises(ValueError, match=r"over its budget of 1\. Runs per replicate"):
        plan.handler(plan.arguments(case="alpha/opt.toml", goal="compare", budget=1,
                                    methods={"smc": smcopt}))  # fmt: skip


def test_validating_an_optimization_file_applies_the_guests_list(svc: Service) -> None:
    """Review of 2026-10-05: validation said "valid" and the submission then answered 403."""
    _setup(svc, optimizers='"smcopt"')
    config = str(svc.tmp / "alpha" / "opt.toml")
    checked = svc.http.post("/v1/configs:validate", headers=svc.guest("alice"),
                            json={"config_file": config}).json()  # fmt: skip
    assert not checked["valid"]
    assert "optimizers:enopt" in checked["errors"][0]
    text = (svc.tmp / "alpha" / "opt.toml").read_text()
    body = {"config_text": text, "base_dir": str(svc.tmp / "alpha")}
    as_text = svc.http.post("/v1/configs:validate", headers=svc.guest("alice"), json=body)
    assert not as_text.json()["valid"]


def test_nsga2_on_a_one_objective_case_is_explained_and_case_info_reads_it(svc: Service) -> None:
    """A reported issue: asked for NSGA-II against EnOpt on a one-objective case, the
    assistant's plan was refused without saying what fits, and case_info failed on the
    optimization case, so the assistant could not know it had one objective."""
    from apex.agents.chat import ChatAssistant
    from apex.storage.chat_store import ChatStore

    _setup(svc)
    store = ChatStore(svc.root)
    assistant = ChatAssistant(svc.service, store, None)
    chat = store.create("plan")
    info_tool = {t.name: t for t in assistant.local_tools()}["case_info"]
    info = info_tool.handler(info_tool.arguments(case="alpha/opt.toml"))
    assert info["kind"] == "optimization"
    assert info["several_objectives"] is False
    assert "enopt" in info["optimizers_that_fit"]
    assert not {"nsga2", "nsga3", "moead"} & set(info["optimizers_that_fit"])
    assert "second objective" in info["note"]
    assert info["plan"]["max_simulator_runs"] is not None
    assert "pop_size" in info["bounds"]
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    with pytest.raises(ValueError, match="the study is refused") as refused:
        plan.handler(plan.arguments(
            case="alpha/opt.toml", goal="NSGA-II against EnOpt", budget=400,
            methods={"nsga2": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}},
                     "enopt": {"name": "enopt", "options": {"ne": 4, "maxiter": 2}}},
        ))  # fmt: skip
    message = str(refused.value)
    assert "needs several objectives" in message
    assert "optimizers that fit it" in message
    assert "nsga2" not in message.split("optimizers that fit it:")[1].split(".")[0]
    assert "pop_size" in message


def test_a_guests_case_info_and_refusals_show_no_server_paths_nor_hidden_optimizers(
    svc: Service,
) -> None:
    """Review of the #136 fix: case_info dumped the resolved (absolute) file options, and
    listed every optimizer; a guest sees workspace paths and only their list's optimizers."""
    from apex.agents.chat import ChatAssistant
    from apex.services.access import Access, load_access
    from apex.storage.chat_store import ChatStore

    _setup(svc)  # the guest may use enopt and smcopt only
    # An option naming a file: validation resolves it to an absolute (server) path.
    case = (svc.tmp / "alpha" / "opt.toml").read_text()
    svc.file("opt.toml", case.replace("[simulator]", "[model]\nrealizations_file = "
                                      '"models.npz"\n\n[simulator]'))  # fmt: skip
    import numpy as np

    np.savez(svc.tmp / "alpha" / "models.npz", perm=np.array([[1.0]]))
    access = Access(load_access(svc.root), svc.service.jobs.all_records())
    store = ChatStore(svc.root)
    assistant = ChatAssistant(svc.service, store, None)
    chat = store.create("plan")
    tool = {t.name: t for t in assistant.local_tools(access, chat.id)}["case_info"]
    info = tool.handler(tool.arguments(case="alpha/opt.toml"))
    text = repr(info)
    assert str(svc.tmp) not in text  # no server path anywhere
    assert set(info["optimizers_that_fit"]) == {"enopt", "smcopt"}  # their list only
    plan = {t.name: t for t in assistant.planning_tools(chat.id, access)}["plan_study"]
    with pytest.raises(ValueError, match="the study is refused") as refused:
        plan.handler(plan.arguments(
            case="alpha/opt.toml", goal="NSGA-II against EnOpt", budget=400,
            methods={"nsga2": {"name": "nsga2", "options": {"pop_size": 8, "maxiter": 2}},
                     "enopt": {"name": "enopt", "options": {"ne": 4, "maxiter": 2}}},
        ))  # fmt: skip
    assert str(svc.tmp) not in str(refused.value)
    assert "optimizers that fit it: enopt, smcopt." in str(refused.value)


def test_server_paths_are_hidden_in_values_and_messages(tmp_path: Path) -> None:
    from apex.agents.chat import _no_server_paths
    from apex.services.workspace import Workspace

    class Space:
        def relative(self, path: Path) -> str:
            if tmp_path in path.parents:
                return "ws/" + path.relative_to(tmp_path).as_posix()
            from apex.core.errors import ApexError

            raise ApexError("outside")

    space: Workspace | Space = Space()
    message = (f"invalid optimization configuration {tmp_path}/a/opt.toml: see "
               "/etc/secret/x.yaml and/or a/b")  # fmt: skip
    shown = _no_server_paths({"m": message, "l": [f"{tmp_path}/m.yaml", 3]}, space)
    assert shown["m"] == "invalid optimization configuration ws/a/opt.toml: see x.yaml and/or a/b"
    assert shown["l"] == ["ws/m.yaml", 3]


def test_a_planned_optimization_study_reports_unequal_budgets_through_plan_study(
    svc: Service,
) -> None:
    """A reported issue (2026-10-20): optimizers compared on unequal evaluation budgets. The
    real plan_study gives each arm's cost and flags the mismatch (no costs given by hand)."""
    from apex.agents.chat import ChatAssistant
    from apex.storage.chat_store import ChatStore

    _setup(svc)
    store = ChatStore(svc.root)
    assistant = ChatAssistant(svc.service, store, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    small = {"name": "smcopt", "options": {"ne": 2, "maxiter": 1}}
    large = {"name": "smcopt", "options": {"ne": 6, "maxiter": 3}}
    result = plan.handler(plan.arguments(
        case="alpha/opt.toml", goal="small against large", budget=4000,
        methods={"small": small, "large": large}, name="unequal",
    ))  # fmt: skip
    costs = result["arm_costs"]
    assert all(isinstance(c["evaluations"], int) for c in costs.values()), costs
    [budget] = [u for u in result["unmatched"] if u.startswith("budget")]
    assert "evaluations differ" in budget
    assert "unmatched" in result["next"]


def test_plan_study_says_what_it_matched_between_optimizers(svc: Service) -> None:
    """A reported issue (2026-10-24): arms of one family ran their plugins' own defaults.
    The real plan_study matches them and tells the assistant what it changed."""
    from apex.agents.chat import ChatAssistant
    from apex.storage.chat_store import ChatStore

    _setup(svc)
    store = ChatStore(svc.root)
    assistant = ChatAssistant(svc.service, store, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    result = plan.handler(plan.arguments(
        case="alpha/opt.toml", goal="EnOpt against SmcOpt", budget=4000, name="family",
        methods={"en": {"name": "enopt", "options": {"ne": 2, "alpha": 0.3, "maxiter": 1}},
                 "smc": {"name": "smcopt", "options": {"maxiter": 1}}},
    ))  # fmt: skip
    assert any(m.startswith("smc: alpha 0.1 -> 0.3") for m in result["matched"])
    assert any(m.startswith("smc: ne 100 -> 2") for m in result["matched"])
    assert "'matched'" in result["next"]
    same = [u for u in result.get("unmatched", []) if u.startswith("same setting")]
    assert not [u for u in same if "alpha" in u or ".ne " in u]
