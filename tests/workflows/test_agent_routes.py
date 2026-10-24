"""The agents as REST routes (the follow-up; 2026-10-01): monitor, diagnose,
report and plan, with their rules or a model, within what each token may see."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.core.errors import ConfigError
from apex.services.api import ApexService
from apex.services.auth import ui_token

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _world(tmp_path: Path, linear_case: CaseFactory) -> tuple[ApexService, str, TestClient]:
    case = linear_case(name="alpha", ensemble_size=20)
    root = tmp_path / "root"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    service = ApexService(root)
    handle = service.submit_run(case.config)
    assert service.wait(handle.id, timeout_s=300, poll_s=0.2).state == "complete"

    def unavailable() -> object:
        raise ConfigError("no key for the model")

    http = TestClient(create_app(root, token=TOKEN, agent_provider=unavailable))  # type: ignore[arg-type]
    return service, handle.run_ids[0], http


def test_the_agents_answer_over_rest(tmp_path: Path, linear_case: CaseFactory) -> None:
    _, run, http = _world(tmp_path, linear_case)

    monitor = http.post("/v1/agents/monitor", headers=AUTH, json={"run": run}).json()
    assert monitor["report"]["status"] in ("healthy", "warning", "critical")
    assert monitor["decided_by"] == "rules"
    diagnosis = http.post("/v1/agents/diagnose", headers=AUTH, json={"run": run}).json()
    assert diagnosis["diagnosis"]["verdict"] in ("healthy", "problems", "failed")
    report = http.post("/v1/agents/report", headers=AUTH, json={"run": run}).json()
    assert report["markdown"].startswith("#")
    # Asked for the model, without one: the rules answer, and say so.
    asked = http.post("/v1/agents/monitor", headers=AUTH, json={"run": run, "model": True})
    assert asked.status_code == 200
    assert "no model" in asked.json()["reason"]
    assert http.post("/v1/agents/monitor", headers=AUTH, json={"run": "nope"}).status_code == 404


def test_the_planner_writes_its_draft_in_the_ui_folder(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    service, _, http = _world(tmp_path, linear_case)

    from apex.services.auth import owner_ui_token

    owner_ui = {"Authorization": f"Bearer {owner_ui_token(service.root, create=True)}",
                "X-APEX-Client": "ui"}  # fmt: skip
    planned = http.post("/v1/agents/plan", headers=owner_ui, json={
        "case": "alpha/config.toml", "goal": "does localization help", "budget": 400,
    })  # fmt: skip

    assert planned.status_code == 200, planned.text
    body = planned.json()
    assert body["draft"].startswith("UI/owner/drafts/planner/study_")
    assert body["within_budget"]
    assert len(body["variants"]) == 2  # the rules: baseline and adaptive localization
    draft = tmp_path / body["draft"]
    assert service.validate_study(draft).valid  # a study ready to launch from the editor
    assert str(tmp_path) not in draft.read_text()  # no server paths in the draft


def test_guests_use_only_listed_agents_on_their_own_runs(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    service, owners, http = _world(tmp_path, linear_case)
    (service.root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[agents]\nnames = ["diagnosis"]\n'
        '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nlocalizations = ["*"]\n'
        'simulators = ["*"]\nexecutors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
        "[actions]\nsubmit = true\n"
    )
    guest = {"Authorization": f"Bearer {ui_token(service.root, create=True, guest='alice')}"}
    text = (
        (tmp_path / "alpha" / "config.toml")
        .read_text()
        .replace("ensemble_size = 20", "ensemble_size = 13")
    )
    (tmp_path / "alpha" / "mine.toml").write_text(text)
    submitted = {"config_file": str(tmp_path / "alpha" / "mine.toml")}
    handle = http.post("/v1/runs", headers=guest, json=submitted).json()
    assert service.wait(handle["id"], timeout_s=300, poll_s=0.2).state == "complete"
    mine = handle["run_ids"][0]

    ok = http.post("/v1/agents/diagnose", headers=guest, json={"run": mine})
    assert ok.status_code == 200, ok.text
    assert http.post("/v1/agents/diagnose", headers=guest, json={"run": owners}).status_code == 404
    assert http.post("/v1/agents/monitor", headers=guest, json={"run": mine}).status_code == 404
    planned = http.post("/v1/agents/plan", headers=guest,
                        json={"case": "alpha/config.toml", "goal": "compare"})  # fmt: skip
    assert planned.status_code == 404  # the planner is not in alice's list
