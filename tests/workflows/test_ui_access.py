"""What web-UI guests see: only their own work.

The owner runs two linear cases, ``alpha`` and ``beta``, also through the UI's client name
(``ui``). Each guest has a token of their own; a guest sees only the jobs, runs, studies,
predictions and chats they created, never the owner's or another guest's (``404``, or left
out of lists), while the owner's token sees everything. The list ``ui_access.toml`` sets the
folders, plugins, agents and actions.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.agents.chat import ChatAssistant
from apex.api.app import create_app
from apex.core.errors import ConfigError
from apex.services.access import SUBMITTED_BY, Access, UIAccess, load_access
from apex.services.api import ApexService
from apex.services.auth import api_token, guest_tokens, revoke_guest, ui_token
from apex.storage.chat_store import ChatStore

TIMEOUT = 180.0
ACCESS = """
[workspace]
folders = ["alpha"]

[plugins]
methods = ["esmda"]
update_strategies = ["approx"]
simulators = ["linear"]
executors = ["*"]
priors = ["gaussian"]
metrics = ["*"]

[agents]
names = ["diagnosis"]

[actions]
submit = true
"""


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


class Service:
    """The test's service root, its two owner runs, and the tokens."""

    def __init__(self, tmp_path: Path, linear_case: CaseFactory) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "service"
        self.root.mkdir()
        (self.root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
        self.service = ApexService(self.root)
        self.runs: dict[str, str] = {}
        self.jobs: dict[str, str] = {}
        for name in ("alpha", "beta"):
            case = linear_case(name=name, ensemble_size=10 + len(self.runs))
            # The owner submits as the UI does (client "ui"): guests must still not see it.
            handle = self.service.submit_run(case.config, client="ui")
            self.wait(handle.id)
            self.runs[name], self.jobs[name] = handle.run_ids[0], handle.id
        self.owner = {"Authorization": f"Bearer {api_token(self.root)}"}
        self.http = TestClient(create_app(self.root))

    def guest(self, name: str = "guest") -> dict[str, str]:
        return {"Authorization": f"Bearer {ui_token(self.root, create=True, guest=name)}"}

    def file(self, name: str, text: str) -> str:
        path = self.tmp / "alpha" / name
        path.write_text(text)
        return str(path)

    def variant(self, name: str, members: int) -> str:
        """A run file in alpha's folder, of another size (an identical one is a reuse)."""
        text = (self.tmp / "alpha" / "config.toml").read_text()
        text = text.replace('name = "alpha"', f'name = "{name}"').replace(
            "ensemble_size = 10", f"ensemble_size = {members}"
        )
        return self.file(f"{name}.toml", text)

    def wait(self, job: str) -> None:
        assert self.service.wait(job, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"


@pytest.fixture
def svc(tmp_path: Path, linear_case: CaseFactory) -> Service:
    return Service(tmp_path, linear_case)


def _names(http: TestClient, headers: dict[str, str]) -> set[str]:
    return {r["name"] for r in http.get("/v1/runs", headers=headers).json()["items"]}


def _jobs(http: TestClient, headers: dict[str, str]) -> set[str]:
    return {j["id"] for j in http.get("/v1/jobs", headers=headers).json()["items"]}


def test_a_guest_sees_none_of_the_owners_work(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")

    me = http.get("/v1/whoami", headers=alice).json()
    assert me.pop("expires")  # a web-UI token expires (tests/workflows/test_ui_tokens.py)
    assert me == {
        "role": "ui", "guest": "alice", "actions": ["submit", "delete"],
        "ui_folder": "UI/guests/alice",
    }  # fmt: skip
    assert _names(http, alice) == set()
    assert _jobs(http, alice) == set()
    assert http.get("/v1/events", headers=alice).json()["events"] == []
    for name in ("alpha", "beta"):
        run = svc.runs[name]
        for suffix in ("", "/steps", "/figures", "/artifacts/summary.json"):
            assert http.get(f"/v1/runs/{run}{suffix}", headers=alice).status_code == 404
        assert http.get(f"/v1/jobs/{svc.jobs[name]}", headers=alice).status_code == 404
    assert _names(http, svc.owner) == {"alpha", "beta"}  # the owner sees everything
    assert http.get("/v1/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_each_guest_sees_only_their_own_jobs_runs_and_studies(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice, bob = svc.http, svc.guest("alice"), svc.guest("bob")

    forged = {SUBMITTED_BY: "guest:bob"}
    body = {"config_file": svc.variant("gamma", 13), "annotations": forged}
    run = http.post("/v1/runs", headers=alice, json=body).json()
    svc.wait(run["id"])
    svc.variant("delta", 14)
    study = svc.file(
        "study.toml",
        '[study]\nname = "alices_study"\n[study.cases]\nalpha = "delta.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    handle = http.post("/v1/studies", headers=alice, json={"study_file": study}).json()
    svc.wait(handle["id"])

    record = svc.service.jobs.record(run["id"])
    assert record["client"] == "ui"
    assert record["annotations"][SUBMITTED_BY] == "guest:alice"  # a request cannot set it
    assert _names(http, alice) == {"gamma", "alices_study-alpha-baseline-r0"}
    jobs = _jobs(http, alice)
    assert {run["id"], handle["id"]} <= jobs
    assert not {svc.jobs["alpha"], svc.jobs["beta"]} & jobs
    assert http.get("/v1/studies/alices_study", headers=alice).status_code == 200
    comparison = http.get("/v1/studies/alices_study/comparison", headers=alice)
    assert comparison.status_code == 200
    # Bob sees none of it.
    assert _names(http, bob) == set()
    assert _jobs(http, bob) == set()
    assert http.get(f"/v1/jobs/{run['id']}", headers=bob).status_code == 404
    assert http.get("/v1/studies/alices_study", headers=bob).status_code == 404
    assert http.get(f"/v1/runs/{run['run_ids'][0]}", headers=bob).status_code == 404
    feed = http.get("/v1/events", headers=alice).json()["events"]
    assert feed
    assert {e["job"] for e in feed} <= jobs
    # Stopping is not allowed by the list.
    assert http.post(f"/v1/jobs/{run['id']}:stop", headers=alice).status_code == 403


def test_a_guests_identical_resubmission_runs_again_in_their_own_folder(
    svc: Service,
) -> None:
    """Guests never see the owner's runs, not even one
    they would have reused; their identical configuration runs again in their folder, and
    their own resubmissions reuse that run."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    alpha = str(svc.tmp / "alpha" / "config.toml")

    handle = http.post("/v1/runs", headers=alice, json={"config_file": alpha}).json()

    assert not handle["reused"]
    assert handle["run_ids"] != [svc.runs["alpha"]]
    svc.wait(handle["id"])
    assert svc.service.jobs.record(handle["id"])["client"] == "ui"
    mine = svc.service.store.find(handle["run_ids"][0])
    assert mine is not None
    assert Path(mine.path) == svc.tmp / "UI/guests/alice/runs/alpha"
    assert http.get(f"/v1/runs/{svc.runs['alpha']}", headers=alice).status_code == 404
    assert _names(http, alice) == {"alpha"}  # their own run of it
    assert svc.service.store.find(svc.runs["alpha"]) is not None  # the owner's is untouched
    again = http.post("/v1/runs", headers=alice, json={"config_file": alpha}).json()
    assert again["reused"]
    assert again["run_ids"] == handle["run_ids"]
    # Another guest gets a run of their own too, and sees neither.
    bobs = http.post("/v1/runs", headers=svc.guest("bob"), json={"config_file": alpha}).json()
    assert bobs["run_ids"] not in (handle["run_ids"], [svc.runs["alpha"]])


def test_the_list_sets_folders_plugins_agents_and_actions(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")

    plugins = {(p["registry"], p["name"]) for p in http.get("/v1/plugins", headers=alice).json()}
    assert ("methods", "esmda") in plugins
    assert ("methods", "gies") not in plugins
    assert [a["name"] for a in http.get("/v1/agents", headers=alice).json()] == ["diagnosis"]
    assert [e["path"] for e in http.get("/v1/files", headers=alice).json()] == ["alpha"]
    assert http.get("/v1/files", headers=alice, params={"path": "beta"}).status_code == 404
    read = http.get("/v1/files/content", headers=alice, params={"path": "alpha/config.toml"})
    assert read.status_code == 200
    hidden = http.get("/v1/files/content", headers=alice, params={"path": "beta/config.toml"})
    assert hidden.status_code == 404
    body = {"path": "alpha/config.toml", "text": "", "sha256": None}
    assert http.put("/v1/files/content", headers=alice, json=body).status_code == 403
    alpha_file = str(svc.tmp / "alpha" / "config.toml")
    ok = http.post("/v1/configs:validate", headers=alice, json={"config_file": alpha_file})
    assert ok.json()["valid"]
    beta_file = str(svc.tmp / "beta" / "config.toml")
    other = http.post("/v1/configs:validate", headers=alice, json={"config_file": beta_file})
    assert other.status_code == 404
    localized = (
        (svc.tmp / "alpha" / "config.toml")
        .read_text()
        .replace('name = "esmda"', 'name = "esmda"\nlocalization = { name = "adaptive" }')
    )
    localized_file = svc.file("localized.toml", localized)
    refused = http.post(
        "/v1/configs:validate", headers=alice, json={"config_file": localized_file}
    ).json()
    assert not refused["valid"]
    assert "localizations:adaptive" in refused["errors"][0]


def test_a_private_plugin_is_hidden_from_guests_whatever_the_list(
    svc: Service, private_executor: str
) -> None:
    # The user, 2026-10-13: a private executor (the ssh executor, built for NORCE's hosts;
    # here a test one) is never seen by guests, even with every executor listed;
    # the owner still sees it.
    (svc.root / "ui_access.toml").write_text(ACCESS)  # executors = ["*"]
    http, alice = svc.http, svc.guest("alice")

    seen = {(p["registry"], p["name"]) for p in http.get("/v1/plugins", headers=alice).json()}

    assert ("executors", "process") in seen
    assert ("executors", private_executor) not in seen
    owner = {(p["registry"], p["name"]) for p in http.get("/v1/plugins", headers=svc.owner).json()}
    assert ("executors", private_executor) in owner


def test_without_a_list_a_guest_can_open_and_use_nothing(svc: Service) -> None:
    alice = svc.guest("alice")

    assert svc.http.get("/v1/whoami", headers=alice).json()["actions"] == []
    assert svc.http.get("/v1/plugins", headers=alice).json() == []
    assert svc.http.get("/v1/agents", headers=alice).json() == []
    assert svc.http.get("/v1/files", headers=alice).json() == []
    refused = svc.http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("g", 12)})
    assert refused.status_code == 403


def test_an_old_list_that_shares_runs_is_refused_with_a_hint(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text('[runs]\nnames = ["alpha"]\n' + ACCESS)

    with pytest.raises(ConfigError, match="only the jobs, runs and studies they created"):
        load_access(svc.root)
    response = svc.http.get("/v1/runs", headers=svc.guest())
    assert response.status_code == 422
    assert "delete these sections" in response.json()["detail"]


def test_guest_tokens_are_named_listed_and_revoked(svc: Service) -> None:
    alice, bob = svc.guest("alice"), svc.guest("bob")

    assert set(guest_tokens(svc.root)) == {"alice", "bob"}
    assert revoke_guest(svc.root, "bob")
    assert set(guest_tokens(svc.root)) == {"alice"}
    assert svc.http.get("/v1/whoami", headers=alice).status_code == 200
    assert svc.http.get("/v1/whoami", headers=bob).status_code == 401
    with pytest.raises(ValueError, match="guest names"):
        ui_token(svc.root, create=True, guest="../escape")


def test_chats_are_each_guests_own_and_the_assistant_is_limited(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice, bob = svc.http, svc.guest("alice"), svc.guest("bob")

    mine = http.post("/v1/chats", headers=svc.owner, json={"title": "owner's"}).json()
    alices = http.post("/v1/chats", headers=alice, json={"title": "alice's"}).json()

    assert [c["id"] for c in http.get("/v1/chats", headers=alice).json()] == [alices["id"]]
    assert http.get("/v1/chats", headers=bob).json() == []
    assert {c["id"] for c in http.get("/v1/chats", headers=svc.owner).json()} == {
        mine["id"], alices["id"],
    }  # fmt: skip
    assert http.get(f"/v1/chats/{alices['id']}", headers=bob).status_code == 404
    assert http.delete(f"/v1/chats/{mine['id']}", headers=alice).status_code == 404
    # The Stop: one's own chat only; nothing running is not an error.
    assert http.post(f"/v1/chats/{alices['id']}:stop", headers=bob).status_code == 404
    assert http.post(f"/v1/chats/{mine['id']}:stop", headers=alice).status_code == 404
    stopped = http.post(f"/v1/chats/{alices['id']}:stop", headers=alice)
    assert stopped.status_code == 200
    assert stopped.json() == {"chat": alices["id"], "stopped": False}

    rules = UIAccess.model_validate(tomllib.loads(ACCESS))
    access = Access(rules, svc.service.jobs.all_records(), "alice")
    assistant = ChatAssistant(svc.service, ChatStore(svc.root), None)
    tools = {t.name: t for t in assistant.local_tools(access)}
    # monitor and report are not listed; case_info reads only the visible folders
    assert set(tools) == {"list_agents", "find_cases", "case_info", "diagnose_run"}
    assert tools["case_info"].handler(tools["case_info"].arguments(case="alpha/config.toml"))
    with pytest.raises(KeyError):
        tools["case_info"].handler(tools["case_info"].arguments(case="beta/config.toml"))
    with pytest.raises(KeyError):  # the owner's run is not alice's
        tools["diagnose_run"].handler(tools["diagnose_run"].arguments(run=svc.runs["alpha"]))


def test_the_matching_rules() -> None:
    rules = UIAccess.model_validate({
        "workspace": {"folders": ["5Spots", "Brugge/*/configs"]},
        "agents": {"names": ["*"]},
    })  # fmt: skip
    spec = {"study": {"study": {"name": "mine"}}}
    records = [
        {"id": "j1", "kind": "study", "spec": spec, "run_ids": [], "run_dirs": [],
         "annotations": {SUBMITTED_BY: "guest:alice"}},
        {"id": "j2", "kind": "run", "run_ids": ["r9"], "run_dirs": ["/r9"],
         "annotations": {SUBMITTED_BY: "guest:bob"}},
        {"id": "j3", "kind": "run", "client": "ui", "run_ids": ["r8"], "run_dirs": ["/r8"],
         "annotations": {}},
    ]  # fmt: skip
    access = Access(rules, records, "alice")

    assert access.study("mine")
    assert not access.study("other")
    assert access.job(records[0])
    assert not access.job(records[1])  # bob's
    assert not access.job(records[2])  # the owner's, even under the client "ui"
    assert access.run("r1", "x", {SUBMITTED_BY: "guest:alice"}, Path("/r1"))  # a study cell
    assert not access.run("r9", "x", {}, Path("/r9"))
    assert access.folder("5Spots/configs/a.toml")
    assert not access.folder("5SpotsX/a.toml")
    assert access.folder("Brugge/v1/configs/a.toml")
    assert not access.folder("Brugge/v1/model/a.toml")
    assert access.on_the_way("Brugge")
    assert not access.on_the_way("Norne")
    assert access.agent("report")
    assert not access.plugin("methods", "esmda")  # no [plugins]: none


def test_the_web_uis_work_goes_to_its_users_ui_folders(svc: Service) -> None:
    """What the web UI submits writes under <first root>/UI/<user>/, the owner's
    in UI/owner and each guest's in UI/guests/<name>; other clients' work stays put."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    ui = svc.tmp / "UI"

    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("gamma", 13)})
    svc.wait(run.json()["id"])
    svc.variant("delta", 14)
    study = svc.file(
        "study.toml",
        '[study]\nname = "placed"\n[study.cases]\nalpha = "delta.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    cells = http.post("/v1/studies", headers=alice, json={"study_file": study}).json()
    svc.wait(cells["id"])
    owners = http.post(
        "/v1/runs", headers=svc.owner, json={"config_file": svc.variant("eps", 15), "client": "ui"}
    ).json()
    svc.wait(owners["id"])
    script = http.post(
        "/v1/runs", headers=svc.owner, json={"config_file": svc.variant("zeta", 16)}
    ).json()
    svc.wait(script["id"])

    def run_dir(handle: dict[str, Any]) -> Path:
        entry = svc.service.store.find(handle["run_ids"][0])
        assert entry is not None
        return Path(entry.path)

    assert run_dir(run.json()) == ui / "guests/alice/runs/gamma"
    assert run_dir(cells) == ui / "guests/alice/studies/placed/alpha/baseline/r0"
    assert (run_dir(cells) / "summary.json").exists()
    assert run_dir(owners) == ui / "owner/runs/eps"
    assert run_dir(script) == svc.tmp / "alpha/runs/zeta"  # not the web UI: where it says
    # Alice may browse her own folder, but not the owner's.
    listing = http.get("/v1/files", headers=alice, params={"path": "UI/guests"}).json()
    assert [e["path"] for e in listing] == ["UI/guests/alice"]
    assert http.get("/v1/files", headers=alice, params={"path": "UI/owner"}).status_code == 404


def test_the_ui_folder_is_one_folder_name() -> None:
    from pydantic import ValidationError

    from apex.services.settings import WorkspaceSection

    assert WorkspaceSection().ui_folder == "UI"
    assert WorkspaceSection(ui_folder="").ui_folder is None  # off
    for bad in ("..", "a/b", "."):
        with pytest.raises(ValidationError, match="ui_folder"):
            WorkspaceSection(ui_folder=bad)


# ---------------------------------------------------------------------- deletion


def _preview(http: TestClient, headers: dict[str, str], **target: str) -> dict[str, Any]:
    response = http.get("/v1/deletions", headers=headers, params=target)
    assert response.status_code == 200, response.text
    document: dict[str, Any] = response.json()
    return document


def _confirm(
    http: TestClient, headers: dict[str, str], job: str, token: str
) -> tuple[int, dict[str, Any]]:
    response = http.post("/v1/deletions", headers=headers, json={"job": job, "token": token})
    return response.status_code, response.json()


def test_a_guest_deletes_their_own_run_and_study_after_confirming(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("gamma", 13)})
    svc.wait(run.json()["id"])
    svc.variant("delta", 14)
    study_file = svc.file(
        "study.toml",
        '[study]\nname = "gone"\n[study.cases]\nalpha = "delta.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    study = http.post("/v1/studies", headers=alice, json={"study_file": study_file}).json()
    svc.wait(study["id"])
    run_id = run.json()["run_ids"][0]

    preview = _preview(http, alice, run=run_id)

    assert preview["job"] == run.json()["id"]
    assert preview["dirs"] == ["UI/guests/alice/runs/gamma"]  # as the UI shows it
    assert preview["blocked"] == []
    assert preview["size"] > 0
    assert (svc.tmp / "UI/guests/alice/runs/gamma").exists()  # nothing deleted yet
    assert _confirm(http, alice, preview["job"], "0" * 32)[0] == 409  # not what was shown
    status, done = _confirm(http, alice, preview["job"], preview["token"])
    assert status == 200, done
    assert not (svc.tmp / "UI/guests/alice/runs/gamma").exists()
    assert svc.service.store.find(run_id) is None
    assert http.get(f"/v1/jobs/{preview['job']}", headers=alice).status_code == 404

    whole = _preview(http, alice, study="gone")
    assert len(whole["jobs"]) == 2  # the study and its cell
    assert whole["studies"] == ["gone"]
    assert _confirm(http, alice, whole["job"], whole["token"])[0] == 200
    assert not (svc.root / "studies" / "gone").exists()
    # Emptied folders go too, up to the guest's own folder (item 6), which stays.
    assert not (svc.tmp / "UI/guests/alice/studies").exists()
    assert not (svc.tmp / "UI/guests/alice/runs").exists()
    assert (svc.tmp / "UI/guests/alice").is_dir()
    assert http.get("/v1/studies/gone", headers=alice).status_code == 404


def test_only_ones_own_finished_work_inside_the_workspace_is_deleted(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice, bob = svc.http, svc.guest("alice"), svc.guest("bob")
    owner_dir = str(svc.tmp / "alpha" / "runs" / "alpha")

    def legacy_reuse() -> dict[str, str]:
        # A guest's job that reused the owner's run.
        job = svc.service.jobs.submit(
            "run", {"reused": owner_dir}, run_ids=[svc.runs["alpha"]], run_dirs=[owner_dir],
            annotations={SUBMITTED_BY: "guest:alice"}, state="complete", client="ui",
        )  # fmt: skip
        return {"id": job}

    reuse = legacy_reuse()

    # A guest's submission is hidden from other guests, and refused to the owner here.
    assert http.get("/v1/deletions", headers=bob, params={"job": reuse["id"]}).status_code == 404
    assert (
        http.get("/v1/deletions", headers=svc.owner, params={"job": reuse["id"]}).status_code == 403
    )
    assert (
        http.get("/v1/deletions", headers=alice, params={"job": svc.jobs["alpha"]}).status_code
        == 404
    )
    # Deleting the reuse deletes only alice's job: the run is the owner's.
    preview = _preview(http, alice, job=reuse["id"])
    assert (preview["runs"], preview["dirs"]) == ([], [])
    assert _confirm(http, alice, reuse["id"], preview["token"])[0] == 200
    assert svc.service.store.find(svc.runs["alpha"]) is not None

    # A run another submission still uses is kept.
    legacy_reuse()
    owners = _preview(http, svc.owner, job=svc.jobs["alpha"])
    assert owners["kept"] == [svc.runs["alpha"]]
    assert owners["dirs"] == []

    # Running work is refused, and so is a folder outside the workspace.
    # (a study job not started yet: queued, and never scheduled while unsealed)
    pending = svc.service.jobs.submit("study", {"study": {}}, start=False, client="ui")
    running = _preview(http, svc.owner, job=pending)
    assert "stop it first" in running["blocked"][0]
    status, error = _confirm(http, svc.owner, pending, running["token"])
    assert status == 409
    assert "stop it first" in error["detail"]
    outside = svc.tmp.parent / f"{svc.tmp.name}-outside"
    text = (
        (svc.tmp / "alpha" / "config.toml")
        .read_text()
        .replace('name = "alpha"', f'name = "far"\noutput_dir = "{outside}"')
        .replace("ensemble_size = 10", "ensemble_size = 17")  # not a reuse of alpha
    )
    far = svc.service.submit_run(Path(svc.file("far.toml", text)), client="api")
    svc.wait(far.id)
    blocked = _preview(http, svc.owner, job=far.id)
    assert "outside the workspace" in blocked["blocked"][0]
    assert blocked["dirs"] == [outside.name]  # its name only: no server path
    assert str(svc.tmp.parent) not in blocked["blocked"][0]
    # A run written into a folder that also holds configurations is not deleted: its
    # output_dir was the user's choice, and the folder is not only APEX output.
    mixed = svc.tmp / "alpha" / "mixed"
    mixed.mkdir()
    (mixed / "notes.toml").write_text("# a person's file\n")
    text = (svc.tmp / "alpha" / "config.toml").read_text()
    text = text.replace('name = "alpha"', f'name = "mixed"\noutput_dir = "{mixed}"')
    text = text.replace("ensemble_size = 10", "ensemble_size = 18")
    shared = svc.service.submit_run(Path(svc.file("mixed.toml", text)), client="api")
    svc.wait(shared.id)
    refused = _preview(http, svc.owner, job=shared.id)
    assert "not only APEX output" in refused["blocked"][0]
    assert _confirm(http, svc.owner, shared.id, refused["token"])[0] == 409
    assert (mixed / "notes.toml").exists()


def test_guests_may_be_refused_deletion_by_the_list(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS + "delete = false\n")
    alice = svc.guest("alice")
    run = svc.http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("eta", 15)})
    svc.wait(run.json()["id"])

    response = svc.http.get("/v1/deletions", headers=alice, params={"job": run.json()["id"]})

    assert response.status_code == 403


def test_a_runs_predictions_go_with_it_and_guests_predict_only_into_their_folder(
    svc: Service,
) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("theta", 13)})
    svc.wait(run.json()["id"])
    run_dir = svc.tmp / "UI/guests/alice/runs/theta"
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(8))
    prediction = (
        f'[prediction]\nname = "p"\nrun_dir = "{run_dir}"\nensembles = ["posterior"]\n'
        '{extra}\n[prediction.outputs]\nvectors = ["y"]\n\n'
        f'[simulator]\noptions = {{ forward_file = "forward.npz", outputs = [{outputs}] }}\n'
    )
    elsewhere = svc.file("elsewhere.toml", prediction.replace(
        "{extra}", f'output_dir = "{svc.tmp / "alpha" / "theirs"}"'))  # fmt: skip
    refused = http.post("/v1/predictions", headers=alice, json={"prediction_file": elsewhere})
    assert refused.status_code == 403
    file = svc.file("predict.toml", prediction.replace("{extra}", ""))
    predicted = http.post("/v1/predictions", headers=alice, json={"prediction_file": file})
    assert predicted.status_code == 202, predicted.text
    svc.wait(predicted.json()["id"])
    assert (run_dir / "predictions" / "p" / "state.json").exists()

    preview = _preview(http, alice, run=run.json()["run_ids"][0])

    assert predicted.json()["id"] in preview["jobs"]  # the prediction goes with its run
    assert preview["dirs"] == ["UI/guests/alice/runs/theta"]
    assert preview["blocked"] == []
    assert _confirm(http, alice, preview["job"], preview["token"])[0] == 200
    assert not run_dir.exists()
    assert http.get(f"/v1/jobs/{predicted.json()['id']}", headers=alice).status_code == 404


def test_run_names_are_one_folder() -> None:
    from pydantic import ValidationError

    from apex.domain.config import RunSection

    for bad in ("../../cases/x", "a/b", "..", "."):
        with pytest.raises(ValidationError, match="one folder name"):
            RunSection(name=bad, seed=1)
    assert RunSection(name="gies.v2-r0", seed=1).name == "gies.v2-r0"


def test_a_recovered_or_resumed_run_is_still_deleted_with_its_submission(svc: Service) -> None:
    """A worker crash turns the creating job into a resume in place, and a stopped run is
    resumed by a job of its own: neither may leave the run undeletable (review)."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("iota", 13)})
    svc.wait(run.json()["id"])
    job, run_id = run.json()["id"], run.json()["run_ids"][0]
    run_dir = svc.tmp / "UI/guests/alice/runs/iota"
    svc.service.jobs.update(job, kind="resume", spec={"run_dir": str(run_dir)})  # recovered
    resumed = svc.service.jobs.submit(
        "resume", {"run_dir": str(run_dir)}, run_ids=[run_id], run_dirs=[str(run_dir)],
        annotations={SUBMITTED_BY: "guest:alice"}, state="complete", client="ui",
    )  # fmt: skip

    preview = _preview(http, alice, run=run_id)

    assert preview["job"] == job
    assert set(preview["jobs"]) == {job, resumed}
    assert preview["dirs"] == ["UI/guests/alice/runs/iota"]
    assert preview["kept"] == []
    assert _confirm(http, alice, job, preview["token"])[0] == 200
    assert not run_dir.exists()


def test_the_editor_deletes_files_of_ones_own_ui_folder_only(svc: Service) -> None:
    """A draft in UI/<user>/ is deleted, if unchanged; nothing elsewhere."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    drafts = svc.tmp / "UI/guests/alice/drafts/chat1"
    drafts.mkdir(parents=True)
    (drafts / "study_x.toml").write_text("[study]\n")
    (svc.tmp / "UI/owner/drafts").mkdir(parents=True)
    (svc.tmp / "UI/owner/drafts/mine.toml").write_text("# the owner's\n")
    path = "UI/guests/alice/drafts/chat1/study_x.toml"
    current = http.get("/v1/files/content", headers=alice, params={"path": path}).json()

    stale = http.delete("/v1/files/content", headers=alice, params={"path": path, "sha256": "0"})
    assert stale.status_code == 409  # not what the editor read
    done = http.delete(
        "/v1/files/content", headers=alice, params={"path": path, "sha256": current["sha256"]}
    )
    assert done.status_code == 200, done.text
    assert done.json() == {"deleted": path}
    assert not (svc.tmp / "UI/guests/alice/drafts").exists()  # emptied folders go too
    assert (svc.tmp / "UI/guests/alice").exists()  # not the user's folder itself

    # Another user's folder, and the visible case folders, are not the guest's to delete.
    alpha = http.get("/v1/files/content", headers=alice, params={"path": "alpha/config.toml"})
    refused = http.delete(
        "/v1/files/content",
        headers=alice,
        params={"path": "alpha/config.toml", "sha256": alpha.json()["sha256"]},
    )
    assert refused.status_code == 403
    assert (svc.tmp / "alpha/config.toml").exists()
    hidden = http.delete(
        "/v1/files/content",
        headers=alice,
        params={"path": "UI/owner/drafts/mine.toml", "sha256": "x"},
    )
    assert hidden.status_code == 404
    # The owner deletes in UI/owner, and is refused elsewhere in the editor.
    owners = http.get(
        "/v1/files/content", headers=svc.owner, params={"path": "UI/owner/drafts/mine.toml"}
    ).json()
    assert (
        http.delete(
            "/v1/files/content",
            headers=svc.owner,
            params={"path": "UI/owner/drafts/mine.toml", "sha256": owners["sha256"]},
        ).status_code
        == 200
    )
    assert (
        http.delete(
            "/v1/files/content",
            headers=svc.owner,
            params={"path": "alpha/config.toml", "sha256": alpha.json()["sha256"]},
        ).status_code
        == 403
    )


def test_a_stopped_run_is_deleted_with_its_stop_marker(svc: Service) -> None:
    """2026-10-01: a run stopped with `apex stop` keeps a stop_requested file, which the
    deletion did not know as APEX output and refused."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("kappa", 13)})
    svc.wait(run.json()["id"])
    run_dir = svc.tmp / "UI/guests/alice/runs/kappa"
    (run_dir / "stop_requested").write_text("")
    logs = svc.root / "jobs" / run.json()["id"]
    logs.mkdir(parents=True, exist_ok=True)

    preview = _preview(http, alice, job=run.json()["id"])

    assert preview["blocked"] == []
    assert _confirm(http, alice, preview["job"], preview["token"])[0] == 200
    assert not run_dir.exists()
    assert not logs.exists()  # the worker's log folder goes with its job


def test_in_the_web_ui_the_owner_sees_only_the_owners_work(svc: Service) -> None:
    """In the web UI the owner and the guests do not see
    each other's work. The web UI names itself (X-APEX-Client: ui); other clients with the
    owner's token (the CLI, the Python client, MCP) still see everything."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    from apex.services.auth import owner_ui_token

    owner_ui = {
        "Authorization": f"Bearer {owner_ui_token(svc.root, create=True)}",
        "X-APEX-Client": "ui",
    }
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("lambda", 13)})
    svc.wait(run.json()["id"])
    svc.variant("mu", 14)
    study_file = svc.file(
        "study.toml",
        '[study]\nname = "alices"\n[study.cases]\nalpha = "mu.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    study = http.post("/v1/studies", headers=alice, json={"study_file": study_file}).json()
    svc.wait(study["id"])
    theirs = http.post("/v1/chats", headers=alice, json={}).json()
    mine = http.post("/v1/chats", headers=owner_ui, json={}).json()
    own = http.post("/v1/runs", headers=owner_ui,
                    json={"config_file": svc.variant("nu", 15), "client": "ui"}).json()  # fmt: skip
    svc.wait(own["id"])

    owners_study = svc.file(
        "owners_study.toml",
        '[study]\nname = "owners"\n[study.cases]\nalpha = "nu.toml"\n'
        '[[study.variants]]\nname = "baseline"\n',
    )
    launched = http.post("/v1/studies", headers=owner_ui,
                         json={"study_file": owners_study, "client": "ui"}).json()  # fmt: skip
    svc.wait(launched["id"])
    assert http.get("/v1/studies/owners", headers=owner_ui).status_code == 200  # in UI/owner

    # Only the owner's work in UI/owner (revision 3): not the guest's, and not the
    # owner's runs elsewhere (alpha and beta, in the case folders).
    assert _names(http, owner_ui) == {"nu"}  # the study reused nu's run: no run of its own
    assert own["id"] in _jobs(http, owner_ui)
    assert launched["id"] in _jobs(http, owner_ui)
    assert not {svc.jobs["alpha"], svc.jobs["beta"], run.json()["id"]} & _jobs(http, owner_ui)
    for path in (f"/v1/jobs/{run.json()['id']}", f"/v1/runs/{run.json()['run_ids'][0]}",
                 "/v1/studies/alices", f"/v1/chats/{theirs['id']}",
                 f"/v1/jobs/{svc.jobs['alpha']}", f"/v1/runs/{svc.runs['alpha']}"):  # fmt: skip
        assert http.get(path, headers=owner_ui).status_code == 404, path
    feed = http.get("/v1/events", headers=owner_ui).json()["events"]
    assert not {e["job"] for e in feed} & {svc.jobs["alpha"], svc.jobs["beta"], run.json()["id"]}
    assert [c["id"] for c in http.get("/v1/chats", headers=owner_ui).json()] == [mine["id"]]
    # Without the web UI's name, the owner's token sees everything, as before.
    assert {"alpha", "lambda", "alices-alpha-baseline-r0"} <= _names(http, svc.owner)
    assert http.get(f"/v1/chats/{theirs['id']}", headers=svc.owner).status_code == 200
    # And the guest still sees none of the owner's work.
    assert _names(http, alice) == {"lambda", "alices-alpha-baseline-r0"}
    assert http.get(f"/v1/chats/{mine['id']}", headers=alice).status_code == 404


def test_a_prediction_is_deleted_from_its_page_and_its_run_stays(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("xi", 13)})
    svc.wait(run.json()["id"])
    run_dir = svc.tmp / "UI/guests/alice/runs/xi"
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(8))
    simulator = f'options = {{ forward_file = "forward.npz", outputs = [{outputs}] }}'
    file = svc.file(
        "predict_xi.toml",
        f'[prediction]\nname = "p"\nrun_dir = "{run_dir}"\nensembles = ["posterior"]\n'
        f'[prediction.outputs]\nvectors = ["y"]\n\n[simulator]\n{simulator}\n',
    )
    predicted = http.post("/v1/predictions", headers=alice, json={"prediction_file": file}).json()
    svc.wait(predicted["id"])
    folder = run_dir / "predictions" / "p"

    preview = _preview(http, alice, prediction=str(folder))

    assert preview["job"] == predicted["id"]
    assert preview["dirs"] == ["UI/guests/alice/runs/xi/predictions/p"]
    assert preview["runs"] == []
    assert _confirm(http, alice, preview["job"], preview["token"])[0] == 200
    assert not folder.exists()
    assert (run_dir / "summary.json").exists()  # the run stays
    # Bob cannot even look at it.
    other = http.get("/v1/deletions", headers=svc.guest("bob"), params={"prediction": str(folder)})
    assert other.status_code == 404


def test_every_plugin_a_configuration_names_is_checked() -> None:
    """Per-variable and joint priors, parameterizations and damping are checked.

    Per-variable priors and damping schemes used to pass unchecked.
    """
    from apex.domain.config import HistoryMatchConfig

    config = HistoryMatchConfig.model_validate({
        "run": {"name": "r", "seed": 1},
        "state": {"variables": [{"name": "a", "size": 2}, {"name": "b", "size": 2},
                                {"name": "c", "size": 2}]},
        "prior": {"ensemble_size": 4, "variables": {"a": {"name": "gaussian_field"}},
                  "groups": {"g": {"name": "gstools_pgs", "variables": ["b", "c"]}}},
        "observations": {"file": "obs.csv"},
        "simulator": {"name": "linear"},
        "method": {"name": "gies", "strategy": {"name": "rlmmac"}, "damping": {"name": "lm"}},
        "parameterizations": [{"name": "pgs_truncation"}],
        "case": {"kind": "real"},
    })  # fmt: skip
    rules = UIAccess.model_validate({
        "plugins": {"methods": ["gies"], "simulators": ["linear"], "executors": ["serial"],
                    "update_strategies": ["rlmmac"], "priors": ["gaussian"]},
    })  # fmt: skip

    hidden = Access(rules, [], "alice").hidden_plugins(config)

    assert hidden == [
        "damping_schemes:lm", "parameterizations:pgs_truncation", "priors:gaussian_field",
        "priors:gstools_pgs",
    ]  # fmt: skip


def test_a_guest_names_only_files_they_may_read(svc: Service) -> None:
    """The declarations hold for web-UI users without a ``[policy]`` (item 4a).

    An input outside the visible folders is refused, by its real path: a symlink in ``alpha``
    to ``beta``'s file is ``beta``'s file. Validation says so, and a submission is refused.
    """
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    text = (svc.tmp / "alpha" / "config.toml").read_text()
    beta_forward = svc.tmp / "beta" / "forward.npz"
    (svc.tmp / "alpha" / "link.npz").symlink_to(beta_forward)
    for name, forward in (("outside", str(beta_forward)), ("linked", "link.npz")):
        config = svc.file(f"{name}.toml", text.replace(
            'forward_file = "forward.npz"', f'forward_file = "{forward}"'
        ).replace('name = "alpha"', f'name = "{name}"'))  # fmt: skip
        checked = http.post("/v1/configs:validate", headers=alice, json={"config_file": config})
        assert not checked.json()["valid"]
        assert "forward_file" in checked.json()["errors"][0]
        assert "not in a folder you may use" in checked.json()["errors"][0]
        submitted = http.post("/v1/runs", headers=alice, json={"config_file": config})
        assert submitted.status_code == 403
    # The owner is not a web-UI user: the same file validates.
    owner = http.post("/v1/configs:validate", headers=svc.owner,
                      json={"config_file": str(svc.tmp / "alpha" / "linked.toml")})  # fmt: skip
    assert owner.json()["valid"]


def test_web_ui_users_choose_no_programs_arguments_or_undeclared_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """What a configuration names, by kind (``policy.named``), as a web-UI user may use it."""
    from apex.services import policy

    named = [
        ("executable", "simulators.opm_flow: executable", "flow"),
        ("executable", "simulators.opm_flow: executable", "/opt/flow"),
        ("executable", "simulators.eclipse: executable", "eclrun"),
        ("arguments", "simulators.opm_flow: flow_args", "--threads=8"),
        ("undeclared", "simulators.x: mystery", "?"),
        ("input", "model.realizations_file", str(tmp_path / "ok.npz")),
        ("input", "observations.file", "relative.csv"),
        ("text", "run.name", "anything"),
        ("output", "output_dir", "/anywhere"),  # the server places a UI user's runs
        ("output", "priors.ert_forward_init: work_dir", "/anywhere"),
    ]
    monkeypatch.setattr(policy, "named", lambda config: named)
    rules = UIAccess.model_validate({"programs": {"executables": ["flow"]}})
    places = _Places({str(tmp_path / "ok.npz"): "folder"})
    refused = Access(rules, []).refused_names(object(), places)  # type: ignore[arg-type]
    assert refused == [
        "simulators.opm_flow: executable = '/opt/flow': web-UI users may not choose this program",
        "simulators.eclipse: executable = 'eclrun': web-UI users may not choose this program",
        "simulators.opm_flow: flow_args: web-UI users may not set a program's arguments",
        "simulators.x: mystery: the plugin does not declare what this option names",
        "observations.file = 'relative.csv' is not in a folder you may use: use a file next "
        "to the case, in a listed folder or your UI folder",
        "priors.ert_forward_init: work_dir: web-UI users may not choose where a plugin writes",
    ]


class _Places:
    """A stand-in for ``PathRules``: where each test path lies."""

    def __init__(self, places: dict[str, str]) -> None:
        self.places = places

    def where(self, path: Path) -> str | None:
        return self.places.get(str(path))

    def shown(self, value: str) -> str:
        return value


def test_a_template_is_never_text_web_ui_users_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of 2026-10-05: a Mako template is Python. A guest writes ``.toml`` files (the
    editor, drafts) and APEX writes ``.md``/``.json`` from what they type, so none of those,
    nothing in a run and nothing in their own folder but a drafted deck is a template."""
    from apex.services import policy

    places = {
        "/w/alpha/deck.mako": "folder", "/w/alpha/x.toml": "folder",
        "/w/UI/g/drafts/c/opt_deck.mako": "draft", "/w/UI/g/drafts/c/report.md": "draft",
        "/w/UI/g/notes.mako": "own", "/w/UI/g/runs/r/config.json": "run",
    }  # fmt: skip
    named = [("code", "simulators.opm_flow: deck_template_file", p) for p in places]
    named.append(("input", "observations.file", "/w/UI/g/runs/r/config.json"))
    monkeypatch.setattr(policy, "named", lambda config: named)
    refused = Access(UIAccess(), []).refused_names(object(), _Places(places))  # type: ignore[arg-type]
    assert [r.split(" = ")[1].split("'")[1] for r in refused] == [
        "/w/alpha/x.toml", "/w/UI/g/drafts/c/report.md", "/w/UI/g/notes.mako",
        "/w/UI/g/runs/r/config.json",
    ]  # fmt: skip
    assert all("a template must be" in r for r in refused)


def test_a_symlink_never_opens_what_it_leads_to(svc: Service) -> None:
    """Review of 2026-10-05: a link in a visible folder (or in a guest's own) into a hidden
    folder, or into another guest's, is refused as that folder is: listing, reading,
    validating; and with every folder listed (``"*"``), others' UI folders stay hidden."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    alpha, beta = svc.tmp / "alpha", svc.tmp / "beta"
    (alpha / "beta_link").symlink_to(beta, target_is_directory=True)
    (alpha / "linked.toml").symlink_to(beta / "config.toml")
    for path in ("alpha/beta_link", "alpha/beta_link/config.toml", "alpha/linked.toml"):
        listed = http.get("/v1/files", headers=alice, params={"path": path})
        read = http.get("/v1/files/content", headers=alice, params={"path": path})
        assert listed.status_code == 404, path
        assert read.status_code == 404, path
    checked = http.post("/v1/configs:validate", headers=alice,
                        json={"config_file": str(alpha / "linked.toml")})  # fmt: skip
    assert checked.status_code == 404
    # Into another guest's folder, from one's own.
    guests = svc.tmp / "UI" / "guests"
    (guests / "bob").mkdir(parents=True)
    (guests / "bob" / "secret.toml").write_text('[run]\nname = "b"\n')
    (guests / "alice").mkdir(parents=True)
    (guests / "alice" / "bobs").symlink_to(guests / "bob", target_is_directory=True)
    own = http.get("/v1/files", headers=alice, params={"path": "UI/guests/alice"})
    assert own.status_code == 200
    for path in ("UI/guests/alice/bobs", "UI/guests/alice/bobs/secret.toml"):
        assert http.get("/v1/files/content", headers=alice,
                        params={"path": path}).status_code == 404, path  # fmt: skip
    # The owner (no list applies) still reads through a link.
    owner = http.get("/v1/files/content", headers=svc.owner, params={"path": "alpha/linked.toml"})
    assert owner.status_code == 200


def test_listing_every_folder_never_shows_other_web_ui_users_folders(svc: Service) -> None:
    """Review of 2026-10-05: ``folders = ["*"]`` covered ``UI/`` too, so a guest read the
    owner's and other guests' web-UI folders; they stay hidden, the guest's own stays open,
    and a link in a listed folder is left out of its listing when it leads somewhere hidden."""
    (svc.root / "ui_access.toml").write_text(
        ACCESS.replace('folders = ["alpha"]', 'folders = ["*"]')
    )
    http, alice = svc.http, svc.guest("alice")
    ui = svc.tmp / "UI"
    for folder in ("owner", "guests/bob", "guests/alice"):
        (ui / folder).mkdir(parents=True, exist_ok=True)
        (ui / folder / "x.toml").write_text('[run]\nname = "x"\n')
    for path in ("UI/owner/x.toml", "UI/guests/bob/x.toml"):
        read = http.get("/v1/files/content", headers=alice, params={"path": path})
        assert read.status_code == 404, path
    own = http.get("/v1/files/content", headers=alice, params={"path": "UI/guests/alice/x.toml"})
    assert own.status_code == 200
    guests = http.get("/v1/files", headers=alice, params={"path": "UI/guests"}).json()
    assert [e["path"] for e in guests] == ["UI/guests/alice"]
    (svc.tmp / "alpha" / "to_bob.toml").symlink_to(ui / "guests" / "bob" / "x.toml")
    alpha = http.get("/v1/files", headers=alice, params={"path": "alpha"}).json()
    assert "alpha/to_bob.toml" not in [e["path"] for e in alpha]
    assert "alpha/config.toml" in [e["path"] for e in alpha]


def test_the_owners_web_ui_view_names_any_file_or_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of 2026-10-06: the owner's view in the web UI (``OwnerView``) inherited the
    guests' rules for files and programs, so the owner's own assistant refused a deck case
    with ``executable = "flow"``; the owner's are theirs."""
    from apex.services import policy
    from apex.services.access import OwnerView

    named = [("executable", "simulators.opm_flow: executable", "/opt/flow"),
             ("arguments", "simulators.opm_flow: flow_args", "--threads=8"),
             ("input", "observations.file", "/anywhere/obs.csv")]  # fmt: skip
    monkeypatch.setattr(policy, "named", lambda config: named)
    assert OwnerView([]).refused_names(object(), _Places({})) == []  # type: ignore[arg-type]


def test_a_link_swapped_after_the_check_is_not_followed(
    svc: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of 2026-10-05: between the folder check and the read, a link could be pointed
    elsewhere; the read asks the opened file where it really is, and refuses it there."""
    from apex.api import guard

    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    link = svc.tmp / "alpha" / "swap.toml"
    link.symlink_to(svc.tmp / "alpha" / "config.toml")
    folder = guard.Guard.folder

    def check_then_swap(self: guard.Guard, path: str | Path) -> None:
        folder(self, path)  # passes: the link leads into alpha
        if Path(str(path)).name == "swap.toml":
            link.unlink()
            link.symlink_to(svc.tmp / "beta" / "config.toml")  # now into hidden beta

    monkeypatch.setattr(guard.Guard, "folder", check_then_swap)
    read = http.get("/v1/files/content", headers=alice, params={"path": "alpha/swap.toml"})
    assert read.status_code == 404


def test_resuming_or_predicting_from_older_work_holds_it_to_todays_rules(svc: Service) -> None:
    """Review of 2026-10-05: a resume (and a prediction) runs the configuration saved with
    the work, not a file, so work from before the file checks is checked as it is saved."""
    import json

    (svc.root / "ui_access.toml").write_text(
        ACCESS.replace("submit = true", "submit = true\nresume = true")
    )
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("old", 13)})
    svc.wait(run.json()["id"])
    run_dir = svc.service.jobs.record(run.json()["id"])["run_dirs"][0]
    saved = Path(run_dir) / "config.json"
    document = json.loads(saved.read_text())
    document["simulator"]["options"]["forward_file"] = str(svc.tmp / "beta" / "forward.npz")
    saved.write_text(json.dumps(document))  # as if submitted before the checks

    resumed = http.post(f"/v1/jobs/{run.json()['id']}:resume", headers=alice, json={})
    assert resumed.status_code == 403
    assert "forward_file" in resumed.text
    predicted = http.post("/v1/predictions", headers=alice, json={
        "prediction_file": svc.file("p.toml", f'[prediction]\nname = "p"\nrun_dir = "{run_dir}"'
                                    '\n[prediction.outputs]\nvectors = ["y"]\n'),
    })  # fmt: skip
    assert predicted.status_code == 403


def test_a_studys_cells_are_checked_as_they_are_submitted(svc: Service) -> None:
    """Review of 2026-10-05: a study's cases were checked from the files, then read again to
    submit; the service now hands every planned cell to the check before submitting any."""
    seen: list[str] = []

    def refuse(config: Any) -> None:  # noqa: ANN401
        seen.append(config.run.name)
        raise PermissionError("refused")

    study = svc.file("cells.toml", '[study]\nname = "cells"\n[study.cases]\nalpha = '
                     '"config.toml"\n[[study.variants]]\nname = "baseline"\n')  # fmt: skip
    before = len(svc.service.jobs.all_records())
    with pytest.raises(PermissionError):
        svc.service.submit_study(Path(study), check=refuse)
    assert seen  # the planned cell, as it would be submitted
    assert len(svc.service.jobs.all_records()) == before  # nothing submitted


def test_a_revised_drafts_cases_are_held_to_the_folders(svc: Service) -> None:
    """Review of 2026-10-05: a revised study draft could name a case outside the guest's
    folders and was refused only at launch; it is refused when revised."""
    from apex.agents.chat import _hidden_case

    (svc.root / "ui_access.toml").write_text(ACCESS)
    access = Access(load_access(svc.root), [], "alice")
    inside = svc.file("in.toml", '[study]\nname = "a"\n[study.cases]\nalpha = "config.toml"\n')
    outside = svc.file("out.toml", f'[study]\nname = "b"\n[study.cases]\nbeta = '
                       f'"{svc.tmp / "beta" / "config.toml"}"\n')  # fmt: skip
    assert _hidden_case(svc.service, access, Path(inside)) is None
    assert _hidden_case(svc.service, access, Path(outside)) == (
        f"case 'beta' ({svc.tmp / 'beta' / 'config.toml'}) is not available"
    )


def test_a_prediction_runs_only_on_listed_plugins_alone_or_in_a_study(
    svc: Service, private_executor: str
) -> None:
    """Review of 2026-10-05: a prediction could replace the run's simulator (or executor)
    with one the guest's list does not have, and a study's predictions were not checked."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("base", 13)})
    svc.wait(run.json()["id"])
    run_dir = svc.service.jobs.record(run.json()["id"])["run_dirs"][0]
    switched = svc.file("switch.toml", f'[prediction]\nname = "p"\nrun_dir = "{run_dir}"\n'
                        '[prediction.outputs]\nvectors = ["y"]\n'
                        '[simulator]\nname = "toy_reservoir"\n')  # fmt: skip
    refused = http.post("/v1/predictions", headers=alice, json={"prediction_file": switched})
    assert refused.status_code == 403
    assert "simulators:toy_reservoir" in refused.text

    svc.variant("cell", 14)
    study = svc.file("pstudy.toml", '[study]\nname = "pstudy"\n[study.cases]\nalpha = '
                     '"cell.toml"\n[[study.variants]]\nname = "baseline"\n'
                     '[[study.predictions]]\n[study.predictions.prediction]\nname = "p"\n'
                     '[study.predictions.prediction.outputs]\nvectors = ["y"]\n'
                     '[study.predictions.simulator]\nname = "toy_reservoir"\n')  # fmt: skip
    in_study = http.post("/v1/studies", headers=alice, json={"study_file": study})
    assert in_study.status_code == 403
    assert "simulators:toy_reservoir" in in_study.text

    # A private executor, although the list has every executor (the user, 2026-10-13).
    remote = svc.file("rstudy.toml", '[study]\nname = "rstudy"\n[study.cases]\nalpha = '
                      '"cell.toml"\n[[study.variants]]\nname = "baseline"\n'
                      '[[study.predictions]]\n[study.predictions.prediction]\nname = "p"\n'
                      '[study.predictions.prediction.outputs]\nvectors = ["y"]\n'
                      '[study.predictions.execution]\nname = "relay"\n')  # fmt: skip
    on_hosts = http.post("/v1/studies", headers=alice, json={"study_file": remote})
    assert on_hosts.status_code == 403
    assert "executors:relay" in on_hosts.text


def test_a_guest_resumes_their_own_work_when_it_passes(svc: Service) -> None:
    """The saved configuration's check refuses only what breaks a rule: a guest's own run,
    within their list, passes it (whatever the service then says about resuming it)."""
    (svc.root / "ui_access.toml").write_text(
        ACCESS.replace("submit = true", "submit = true\nresume = true")
    )
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("mine", 13)})
    svc.wait(run.json()["id"])
    resumed = http.post(f"/v1/jobs/{run.json()['id']}:resume", headers=alice, json={})
    assert resumed.status_code != 403, resumed.text


def test_only_the_owner_chooses_the_code_a_job_runs(svc: Service) -> None:
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http = svc.http
    config = svc.variant("coded", 12)

    frozen = {"config_file": config, "code": "frozen"}
    guest = http.post("/v1/runs", headers=svc.guest(), json=frozen)
    assert guest.status_code == 403
    assert "only the owner chooses the code" in guest.text
    assert (
        http.post(
            "/v1/runs", headers=svc.owner, json={"config_file": config, "code": "--upload-pack=x"}
        ).status_code
        == 422
    )  # never an option for git
    owner = http.post("/v1/runs", headers=svc.owner, json={"config_file": config, "code": "live"})
    assert owner.status_code == 202, owner.text
    job = owner.json()["id"]
    assert svc.service.jobs.record(job)["code"] is None  # live: no pin
    svc.wait(job)


def _no_server_path(svc: Service, response: Any) -> None:  # noqa: ANN401 - a response
    """Assert an answer names no server path (the workspace's root, the service's)."""
    assert str(svc.tmp) not in response.text, response.text


def test_a_guest_sees_and_sends_only_workspace_relative_paths(svc: Service) -> None:
    """Finding 31 of the release tester (2026-10-19): /v1/workspace gave guests the roots'
    server paths. Guests now get names only, send relative paths everywhere, and get none
    of the server's paths back; the owner's answers keep them."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")

    space = http.get("/v1/workspace", headers=alice)
    assert space.json() == {"roots": [{"name": ""}]}
    owners = http.get("/v1/workspace", headers=svc.owner).json()
    assert owners == {"roots": [{"name": "", "path": str(svc.tmp.resolve())}]}

    svc.variant("lam", 13)
    text = (svc.tmp / "alpha" / "lam.toml").read_text()
    validated = http.post(
        "/v1/configs:validate", headers=alice, json={"config_text": text, "base_dir": "alpha"}
    )
    assert validated.status_code == 200, validated.text
    assert validated.json()["valid"], validated.text
    _no_server_path(svc, validated)
    content = http.get("/v1/files/content", headers=alice, params={"path": "alpha/lam.toml"})
    assert content.json()["path"] == "alpha/lam.toml"

    run = http.post("/v1/runs", headers=alice, json={"config_file": "alpha/lam.toml"})
    assert run.status_code == 202, run.text
    svc.wait(run.json()["id"])
    status = http.get(f"/v1/jobs/{run.json()['id']}", headers=alice)
    assert status.json()["run_dirs"] == ["UI/guests/alice/runs/lam"]
    rows = http.get("/v1/runs", headers=alice)
    assert [r["path"] for r in rows.json()["items"]] == ["UI/guests/alice/runs/lam"]
    events = http.get(f"/v1/jobs/{run.json()['id']}/events", headers=alice)
    summary = http.get(f"/v1/runs/{run.json()['run_ids'][0]}", headers=alice)
    assert summary.json()["path"] == "UI/guests/alice/runs/lam"
    for response in (status, rows, events, summary, http.get("/v1/events", headers=alice)):
        _no_server_path(svc, response)
    # The owner's API answers keep the server's paths.
    owner_rows = http.get("/v1/runs", headers=svc.owner).json()["items"]
    assert str(svc.tmp / "UI/guests/alice/runs/lam") in {r["path"] for r in owner_rows}

    # A study and a prediction from relative paths; the prediction's page by its relative path.
    svc.variant("mu", 14)
    svc.file("rel_study.toml", '[study]\nname = "rel"\n[study.cases]\nalpha = "mu.toml"\n'
             '[[study.variants]]\nname = "baseline"\n')  # fmt: skip
    study = http.post("/v1/studies", headers=alice, json={"study_file": "alpha/rel_study.toml"})
    assert study.status_code == 202, study.text
    svc.wait(study.json()["id"])
    for answer in (
        http.get("/v1/studies/rel", headers=alice),
        http.get(f"/v1/runs/{run.json()['run_ids'][0]}", headers=alice),
        http.get(f"/v1/runs/{run.json()['run_ids'][0]}/steps", headers=alice),
        http.get("/v1/plugins", headers=alice),
    ):
        assert answer.status_code == 200, answer.text
        for prefix in _prefixes(svc):
            assert prefix not in answer.text, (prefix, answer.url)
    run_dir = svc.tmp / "UI/guests/alice/runs/lam"
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(8))
    simulator = f'options = {{ forward_file = "forward.npz", outputs = [{outputs}] }}'
    svc.file(
        "rel_predict.toml",
        f'[prediction]\nname = "p"\nrun_dir = "{run_dir}"\nensembles = ["posterior"]\n'
        f'[prediction.outputs]\nvectors = ["y"]\n\n[simulator]\n{simulator}\n',
    )
    predicted = http.post(
        "/v1/predictions", headers=alice, json={"prediction_file": "alpha/rel_predict.toml"}
    )
    assert predicted.status_code == 202, predicted.text
    svc.wait(predicted.json()["id"])
    where = http.get(f"/v1/jobs/{predicted.json()['id']}", headers=alice).json()["run_dirs"]
    assert where == ["UI/guests/alice/runs/lam/predictions/p"]
    page = http.get("/v1/predictions/summary", headers=alice, params={"path": where[0]})
    assert page.status_code == 200, page.text
    _no_server_path(svc, page)
    preview = http.get("/v1/deletions", headers=alice, params={"prediction": where[0]})
    assert preview.status_code == 200, preview.text
    _no_server_path(svc, preview)
    # The guest's absolute path inside their folders still works; the answer is relative.
    absolute = http.get(
        "/v1/files/content", headers=alice, params={"path": str(svc.tmp / "alpha/lam.toml")}
    )
    assert absolute.status_code == 200
    assert absolute.json()["path"] == "alpha/lam.toml"


def test_relative_paths_stay_confined(svc: Service) -> None:
    """Finding 31: relative paths are resolved through the workspace, with its confinement
    and the guest's list: no '..' escapes, no other user's folder, nothing unlisted."""
    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    (svc.tmp / "UI/owner/drafts").mkdir(parents=True)
    (svc.tmp / "UI/owner/drafts/mine.toml").write_text("# the owner's\n")
    (svc.tmp / "UI/guests/bob").mkdir(parents=True)
    (svc.tmp / "UI/guests/bob/b.toml").write_text("# bob's\n")
    (svc.tmp / "beta" / "beta.toml").write_text((svc.tmp / "alpha" / "config.toml").read_text())

    escapes = ["alpha/../beta/beta.toml", "../outside.toml", "alpha/../../x.toml"]
    hidden = ["UI/owner/drafts/mine.toml", "UI/guests/bob/b.toml", "beta/beta.toml"]
    for path in escapes + hidden:
        read = http.get("/v1/files/content", headers=alice, params={"path": path})
        assert read.status_code in (403, 404), (path, read.text)
        for route, field in (("/v1/runs", "config_file"), ("/v1/studies", "study_file"),
                             ("/v1/predictions", "prediction_file"),
                             ("/v1/optimizations", "optimization_file")):  # fmt: skip
            sent = http.post(route, headers=alice, json={field: path})
            assert sent.status_code in (403, 404), (route, path, sent.text)
            _no_server_path(svc, sent)
    for path in ("alpha/..", "UI/owner", "UI/guests/bob"):
        listed = http.get("/v1/files", headers=alice, params={"path": path})
        assert listed.status_code in (403, 404), (path, listed.text)
    for base in ("alpha/..", "UI/owner", "beta"):
        checked = http.post("/v1/configs:validate", headers=alice,
                            json={"config_text": "[run]\n", "base_dir": base})  # fmt: skip
        assert checked.status_code in (403, 404), (base, checked.text)
    for target in ("UI/owner/runs/x/predictions/p", "alpha/../beta"):
        assert http.get("/v1/predictions/summary", headers=alice,
                        params={"path": target}).status_code == 404  # fmt: skip
        assert http.get("/v1/optimizations/summary", headers=alice,
                        params={"path": target}).status_code == 404  # fmt: skip
    # The owner's absolute and relative paths both work.
    for given in (str(svc.tmp / "alpha/config.toml"), "alpha/config.toml"):
        owner = http.post("/v1/configs:validate", headers=svc.owner, json={"config_file": given})
        assert owner.status_code == 200, owner.text
        assert owner.json()["valid"], owner.text


def test_the_scrubber_rewrites_known_server_paths_only() -> None:
    """Finding 31: roots become their names (none for the only root), other folders their
    placeholders; a folder that merely begins like a root is left alone, a sentence's
    final period is not part of the path, keys and a file's text are never rewritten."""
    from apex.api.scrub import PathScrubber, _scrubbed_json

    one = PathScrubber([("/srv/w", ""), ("/srv/w/svc", "<service>")])
    assert one.text("no file /srv/w/a/b.toml here") == "no file a/b.toml here"
    assert one.text("/srv/w/svc/jobs/j1/log") == "<service>/jobs/j1/log"
    assert one.text("/srv/w") == "(workspace)"
    assert one.text("see /srv/w. And /srv/w/svc.") == "see (workspace). And <service>."
    assert one.text("see /srv/w/x.") == "see x."
    assert one.text("/srv/w.bak /srv/w-2/a /srv/wx/a and /x/srv/w/a") == (
        "/srv/w.bak /srv/w-2/a /srv/wx/a and /x/srv/w/a"
    )
    two = PathScrubber([("/d/cases", "cases"), ("/d/cases/deep", "deep")])
    assert two.text("/d/cases/deep/a and /d/cases/b") == "deep/a and cases/b"
    assert two.value({"/d/cases/k": ["/d/cases/deep", 3, None]}) == {
        "/d/cases/k": ["deep", 3, None]
    }
    # Events: each data line parsed, so a path after an escape is found too.
    event = 'id: 3\nevent: x\ndata: {"message": "Traceback:\\n/d/cases/a.toml failed"}\n\n'
    assert two.event_stream(event) == (
        'id: 3\nevent: x\ndata: {"message": "Traceback:\\ncases/a.toml failed"}\n\n'
    )
    assert _scrubbed_json(two, b"") is None
    assert _scrubbed_json(two, b"\xff\xfe") is None  # not UTF-8: as it is


def test_a_guest_saves_a_file_unchanged(svc: Service) -> None:
    """Review of finding 31: a file's text must reach the editor as it is on disk, or a
    guest who opens and saves it would write its absolute paths as relative ones."""
    (svc.root / "ui_access.toml").write_text(
        ACCESS.replace("submit = true", "submit = true\nedit = true")
    )
    http, alice = svc.http, svc.guest("alice")
    target = svc.tmp / "alpha" / "abs.toml"
    text = (svc.tmp / "alpha" / "config.toml").read_text()
    text += f'\n# observations at "{svc.tmp / "alpha" / "obs.csv"}" (see {svc.tmp}.)\n'
    target.write_bytes(text.encode())
    before = target.read_bytes()

    read = http.get("/v1/files/content", headers=alice, params={"path": "alpha/abs.toml"}).json()
    assert read["text"] == text
    body = {"path": "alpha/abs.toml", "text": read["text"], "sha256": read["sha256"]}
    saved = http.put("/v1/files/content", headers=alice, json=body)
    assert saved.status_code == 200, saved.text
    assert target.read_bytes() == before


def _prefixes(svc: Service) -> list[str]:
    """Every server folder a guest must not see: the workspace's, the service's and the
    machine's (home, APEX's source, Python, the temporary folder, the resources)."""
    import sys
    import tempfile

    import apex
    from apex.execution.resources import default_root

    folders = [svc.tmp, svc.root, Path.home(), Path(apex.__file__).resolve().parent.parent,
               Path(sys.prefix), Path(tempfile.gettempdir()), default_root(),
               svc.root / "code"]  # fmt: skip
    return sorted({str(f) for f in folders} | {str(f.resolve()) for f in folders})


@pytest.mark.parametrize("temporary", ["host", "/tmp"])
def test_a_failed_jobs_traceback_reaches_a_guest_without_server_paths(
    svc: Service, temporary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of finding 31: errors and events name folders outside the workspace too (the
    frozen code, the resources, APEX's source in a traceback, a simulator's scratch). The
    traceback is written into a guest's job as a failing worker records one.

    Release tester N1 (2026-10-20): with ``TMPDIR=/tmp`` the scratch path reached guests,
    since ``/tmp`` is a system folder; it runs with the host's temporary folder and /tmp.
    """
    import sys
    import tempfile

    import apex
    from apex.execution.resources import default_root

    (svc.root / "ui_access.toml").write_text(ACCESS)
    http, alice = svc.http, svc.guest("alice")
    run = http.post("/v1/runs", headers=alice, json={"config_file": svc.variant("nu", 13)})
    job = run.json()["id"]
    svc.wait(job)
    if temporary != "host":  # after the run, which keeps its own scratch
        monkeypatch.setattr(tempfile, "tempdir", temporary)
    source = Path(apex.__file__).resolve().parent / "workflows" / "history_matching.py"
    trace = "\n".join([
        "Traceback (most recent call last):",
        f'  File "{source}", line 10, in run',
        f'  File "{Path(sys.prefix) / "lib" / "numpy.py"}", line 3',
        f"OSError: cannot write {Path(tempfile.gettempdir()) / "sim" / "m0"}.",
        f"pool {default_root() / "pools.toml"}; code {svc.root / "code" / "abc"}",
        f"case {svc.tmp / "alpha" / "nu.toml"}; home {Path.home() / ".config" / "x"}",
    ])  # fmt: skip
    store = svc.service.jobs.store
    store.update_job(job, state="failed", error=f"RuntimeError: {trace}")
    store.add_event(job, {"time": "2026-10-19T00:00:00", "level": "ERROR",
                          "message": trace, "event": "job_failed", "error": trace})  # fmt: skip

    answers = [
        http.get(f"/v1/jobs/{job}", headers=alice),
        http.get(f"/v1/jobs/{job}/events", headers=alice),
        http.get(f"/v1/jobs/{job}/events", headers={**alice, "Accept": "text/event-stream"}),
        http.get("/v1/events", headers=alice),
    ]
    for answer in answers:
        assert answer.status_code == 200, answer.text
        for prefix in _prefixes(svc):
            assert prefix not in answer.text, (prefix, answer.text)
    assert "<apex>/apex/workflows/history_matching.py" in answers[0].json()["error"]
    assert "case alpha/nu.toml;" in answers[0].json()["error"]
    streamed = answers[2].text
    assert "alpha/nu.toml" in streamed
    assert "<tmp>/sim/m0." in streamed
    owner = http.get(f"/v1/jobs/{job}", headers=svc.owner).json()
    assert str(source) in owner["error"]  # the owner's answers keep the paths


def test_service_paths_shown_to_guests_go_back_to_the_service() -> None:
    """Review of finding 31: ``<service>/...``, as guests' answers show the service's root,
    is taken back as the service's root (still confined by the routes' checks)."""
    from types import SimpleNamespace

    from apex.api.guard import Denied, Guard

    root = Path("/srv/apex")
    service = SimpleNamespace(root=root)
    who = Guard(service, "ui", "alice")  # type: ignore[arg-type]
    assert who.server_path("<service>/runs/a") == root / "runs" / "a"
    assert who.server_path("/srv/elsewhere/a") == Path("/srv/elsewhere/a")
    with pytest.raises(Denied):
        who.server_path("<service>/../etc/passwd")


@pytest.mark.parametrize("temporary", ["/tmp", None])
def test_placeholders_never_stand_for_system_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temporary: str | None
) -> None:
    """Re-review of finding 31: ``sys.base_prefix`` is ``/usr`` on most machines, so a
    ``<python>`` for it turned ``/usr/bin/flow`` into ``<python>/bin/flow``. The temporary
    folder is the one exception (``<tmp>`` even as ``/tmp``): the test failed where it is
    ``/tmp`` (release tester C3, 2026-10-24), so it runs with ``/tmp`` too."""
    import tempfile

    from apex.api.guard import Guard, _placeable

    if temporary is not None:
        monkeypatch.setattr(tempfile, "tempdir", temporary)

    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path / "w"}"]\n')
    (tmp_path / "w").mkdir()
    who = Guard(ApexService(root), "ui", "alice")
    scrub = who.scrubber()
    for system in ("/usr/bin/flow", "/opt/eclipse/bin/eclrun", "/usr/local/lib/x", "/var/log/y"):
        assert scrub.text(f"ran {system}.") == f"ran {system}."
    assert all(_placeable(Path(p)) for p, shown in scrub.prefixes.items() if shown != "<tmp>")
    assert not any(_placeable(Path(p)) for p in ("/", "/usr", "/opt", "/home", "/data"))
    assert scrub.text(f"see {tmp_path / 'w'}.") == "see (workspace)."
    assert who.server_path("(workspace)") == (tmp_path / "w").resolve()
    assert who.scrubber() is scrub  # built once per set of folders


def _owner_ui(svc: Service) -> dict[str, str]:
    from apex.services.auth import owner_ui_token

    return {"Authorization": f"Bearer {owner_ui_token(svc.root, create=True)}",
            "X-APEX-Client": "ui"}  # fmt: skip


def test_a_reused_run_is_said_and_linked_on_its_job_page(svc: Service) -> None:
    """Release tester U9: the owner's web UI submits a configuration a study (or the CLI)
    has already run. The job was complete but hidden from the owner's web UI, and its page
    said nothing of the run. Now it records the run it reused and says so; whose runs the
    owner's web UI lists is unchanged (the study's run lies outside UI/owner)."""
    http, owner_ui = svc.http, _owner_ui(svc)
    config = svc.variant("nu", 15)
    study = svc.file("cli.toml", '[study]\nname = "cli"\n[study.cases]\nalpha = "nu.toml"\n'
                                 '[[study.variants]]\nname = "base"\n')  # fmt: skip
    svc.wait(svc.service.submit_study(Path(study)).id)

    handle = http.post("/v1/runs", headers=owner_ui,
                       json={"config_file": config, "client": "ui"}).json()  # fmt: skip
    assert handle["reused"] is True
    record = svc.service.jobs.record(handle["id"])
    assert record["run_ids"] == handle["run_ids"]
    assert record["run_dirs"]
    assert record["spec"]["requested_dir"] == str(svc.tmp / "UI" / "owner" / "runs" / "nu")
    assert handle["id"] in _jobs(http, owner_ui)  # the owner's own submission
    status = http.get(f"/v1/jobs/{handle['id']}", headers=owner_ui).json()
    assert status["state"] == "complete"
    assert status["run_ids"] == []  # a run page the owner's web UI may not open: no id
    assert status["reused"] == record["run_dirs"][0]  # where it is (shown workspace-relative)
    assert status["reused_visible"] is False  # outside the web-UI folder: no run page
    assert "nu" not in _names(http, owner_ui)  # the owner's web UI lists no more runs
    assert http.get(f"/v1/runs/{handle['run_ids'][0]}", headers=owner_ui).status_code == 404
    # Outside the web UI the owner's token sees the run, and may open it.
    assert http.get(f"/v1/jobs/{handle['id']}", headers=svc.owner).json()["reused_visible"]

    # A run of the owner's web UI, reused by the web UI again: linked.
    own = http.post("/v1/runs", headers=owner_ui,
                    json={"config_file": svc.variant("xi", 16), "client": "ui"}).json()  # fmt: skip
    svc.wait(own["id"])
    again = http.post(
        "/v1/runs",
        headers=owner_ui,
        json={"config_file": str(svc.tmp / "alpha" / "xi.toml"), "client": "ui"},
    )
    assert again.json()["reused"] is True
    seen = http.get(f"/v1/jobs/{again.json()['id']}", headers=owner_ui).json()
    assert (seen["reused_visible"], seen["run_ids"]) == (True, own["run_ids"])
    # A job that ran its run says nothing of reuse.
    ran = http.get(f"/v1/jobs/{own['id']}", headers=owner_ui).json()
    assert (ran["reused"], ran["reused_visible"]) == (None, None)


def test_a_reuse_that_resumes_a_run_elsewhere_is_the_owners_job(svc: Service) -> None:
    """The review of U9: a reuse that resumes (or restarts) a run outside UI/owner made a job
    the owner's web UI could not see, and it then said "nothing new was submitted"; so did
    resuming that job again by hand."""
    http, owner_ui = svc.http, _owner_ui(svc)
    config = svc.variant("rho", 15)
    stopped = svc.service.submit_run(Path(config), max_steps=1)  # the CLI's, stopped early
    assert svc.service.wait(stopped.id, timeout_s=300, poll_s=0.2).state == "stopped"

    handle = http.post("/v1/runs", headers=owner_ui, json={
        "config_file": config, "client": "ui", "max_steps": 1}).json()  # fmt: skip
    assert handle["kind"] == "resume"
    assert handle["job_visible"] is True  # the owner's own job, not "nothing new"
    assert handle["id"] in _jobs(http, owner_ui)
    asked = str(svc.tmp / "UI" / "owner" / "runs" / "rho")
    assert svc.service.jobs.record(handle["id"])["spec"]["requested_dir"] == asked
    assert svc.service.wait(handle["id"], timeout_s=300, poll_s=0.2).state == "stopped"

    again = svc.service.resume(handle["id"])  # the job page's Resume
    assert svc.service.jobs.record(again.id)["spec"]["requested_dir"] == asked
    assert again.id in _jobs(http, owner_ui)
    svc.wait(again.id)


def test_validate_says_what_submit_refuses_a_run_folder_holding_another_run(
    svc: Service,
) -> None:
    """Release tester U10: Validate said "Valid", then Submit refused the file because its
    run folder (UI/owner/runs/<name>) held another run. Validation now says so too."""
    http, owner_ui = svc.http, _owner_ui(svc)
    first = svc.variant("omicron", 17)
    done = http.post("/v1/runs", headers=owner_ui, json={"config_file": first, "client": "ui"})
    svc.wait(done.json()["id"])
    # The same name, another configuration: the same folder, holding another run.
    other = svc.file(
        "other.toml", Path(first).read_text().replace("ensemble_size = 17", "ensemble_size = 18")
    )
    text = Path(other).read_text()
    for body in ({"config_file": other},
                 {"config_text": text, "base_dir": str(svc.tmp / "alpha")}):  # fmt: skip
        validation = http.post("/v1/configs:validate", headers=owner_ui, json=body).json()
        assert not validation["valid"], body
        assert any("already holds another run" in e for e in validation["errors"])
    refused = http.post("/v1/runs", headers=owner_ui, json={"config_file": other, "client": "ui"})
    assert refused.status_code == 422
    assert "already holds another run" in refused.text
    # The identical configuration is reused, not refused: valid.
    same = http.post("/v1/configs:validate", headers=owner_ui, json={"config_file": first})
    assert same.json()["valid"], same.json()
    # A guest's validation creates no folder of theirs (it only names where a run would go).
    (svc.root / "ui_access.toml").write_text(ACCESS)
    zed = svc.guest("zed")
    checked = http.post("/v1/configs:validate", headers=zed, json={"config_file": other})
    assert checked.status_code == 200, checked.json()
    assert not (svc.tmp / "UI" / "guests" / "zed").exists()
    # Outside the web UI the file's own folder counts (alpha/runs/omicron: empty).
    assert http.post("/v1/configs:validate", headers=svc.owner,
                     json={"config_file": other}).json()["valid"]  # fmt: skip


def test_a_submission_answered_with_a_job_the_user_may_not_see_says_so(
    svc: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release tester U9: an identical run's job still queued or running elsewhere answers a
    web-UI submission; the web UI must not open its page (404) but say so. Here the answer
    is the owner's CLI job of alpha (outside UI/owner), as ``submit_run`` gives it."""
    from typing import cast

    from fastapi import FastAPI

    from apex.domain.services import JobHandle

    http, owner_ui, alice = svc.http, _owner_ui(svc), svc.guest("alice")
    (svc.root / "ui_access.toml").write_text(ACCESS)
    other = JobHandle(id=svc.jobs["alpha"], kind="run", run_ids=[svc.runs["alpha"]], reused=True)
    # The API's own service answers (create_app builds it).
    app = cast("FastAPI", http.app)
    monkeypatch.setattr(app.state.service, "submit_run", lambda *a, **k: other)
    config = str(svc.tmp / "alpha" / "config.toml")
    for headers in (owner_ui, alice):
        answer = http.post(
            "/v1/runs", headers=headers, json={"config_file": config, "client": "ui"}
        )
        assert answer.status_code == 202, answer.json()
        assert answer.json()["job_visible"] is False
        assert answer.json()["job_state"] == "complete"
        assert http.get(f"/v1/jobs/{other.id}", headers=headers).status_code == 404
    # Other clients see every job: nothing is added.
    plain = http.post("/v1/runs", headers=svc.owner, json={"config_file": config}).json()
    assert plain["job_visible"] is None


def test_an_identical_queued_job_is_no_output_conflict(
    svc: Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of U10: submit returns a queued identical job (its folder not made yet),
    so validate must not report the folder as holding another run."""
    from apex.services import composition

    first = svc.variant("tau", 17)
    svc.wait(svc.service.submit_run(Path(first)).id)
    other = svc.file("tau2.toml", Path(first).read_text().replace(
        "ensemble_size = 17", "ensemble_size = 18"))  # fmt: skip
    config = composition.load_config(Path(other))
    assert svc.service.output_conflict(config) is not None  # another run there
    monkeypatch.setattr(svc.service, "_live_job", lambda identity: {"id": "queued"})
    assert svc.service.output_conflict(config) is None  # submit would return that job
