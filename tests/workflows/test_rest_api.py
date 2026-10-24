"""The REST API and the remote client.

The contract tests run the same calls through the in-process ``Client`` and through the
``RemoteClient`` over HTTP (Starlette's ``TestClient`` drives the app in this process), so
the two cannot drift. Jobs run in real worker processes on the linear-Gaussian cases.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import pytest
from typer.testing import CliRunner

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.cli.main import app as cli
from apex.client import Client, RemoteClient, RemoteError
from apex.core.errors import ConfigError
from apex.services.approval import ApprovalRequired
from apex.services.auth import api_token

TIMEOUT = 180.0
TOKEN = "test-token"


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]
AnyClient = Client | RemoteClient


def _remote(root: Path) -> RemoteClient:
    return RemoteClient("http://testserver", TOKEN, http=TestClient(create_app(root, token=TOKEN)))


@pytest.fixture(params=["in_process", "http"])
def client(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AnyClient]:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[approval]\nmax_simulator_runs = 500\n")
    if request.param == "in_process":
        yield Client(root)
    else:
        with _remote(root) as remote:
            yield remote


def _study(case: Case) -> Path:
    path = case.directory / "study.toml"
    path.write_text(
        """
[study]
name = "inflation"
seed = 11
metrics = ["normalized_data_mismatch:all"]

[study.cases]
linear = "config.toml"

[[study.variants]]
name = "four_steps"

[[study.variants]]
name = "two_steps"
patch = { method = { options = { inflation = [2, 2] } } }
"""
    )
    return path


# ---------------------------------------------------------------------- contract


def test_discovery_and_validation_agree(client: AnyClient, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)

    assert any(p.registry == "methods" and p.name == "gies" for p in client.plugins())
    assert "study" in client.schema("study")["properties"]
    checked = client.validate(case.config)
    assert checked.valid
    assert checked.plan is not None
    assert checked.plan.max_simulator_runs == 100
    bad = client.validate({"run": {"name": "x", "seed": 1}, "method": {"name": "esmda"}})
    assert not bad.valid
    assert bad.errors
    study = client.validate_study(_study(case))
    assert len(study.plans) == 2


def test_a_run_is_submitted_followed_and_summarised(
    client: AnyClient, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)

    handle = client.submit_run(case.config, annotations={"via": "contract"})
    status = client.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2)

    assert status.state == "complete", status
    assert status.annotations["via"] == "contract"
    events = client.events(handle.id)
    assert [e.index for e in events] == list(range(len(events)))
    assert client.events(handle.id, since=2)[0].index == 2
    again = client.submit_run(case.config)
    assert again.reused
    assert again.run_ids == handle.run_ids
    summary = client.run_summary(handle.run_ids[0])
    assert summary.iterations == [0, 1, 2, 3, 4]
    assert summary.state == "complete"
    steps = client.run_steps(handle.run_ids[0])
    assert [s.iteration for s in steps.steps] == [0, 1, 2, 3, 4]
    mismatch = [
        m for m in steps.metrics if m.metric == "normalized_data_mismatch" and m.label == "all"
    ]
    assert [m.iteration for m in mismatch] == [0, 1, 2, 3, 4]
    assert all(m.mean is not None and m.p10 is not None for m in mismatch)


def test_a_study_runs_and_compares(client: AnyClient, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)

    handle = client.submit_study(_study(case))
    status = client.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3)

    assert status.state == "complete", status
    assert client.study_status("inflation").id == handle.id
    comparison = client.compare("inflation")
    assert comparison.missing == []
    assert [v.variant for v in comparison.metrics[0].variants] == ["four_steps", "two_steps"]


def test_errors_are_the_same_exceptions(
    client: AnyClient, linear_case: CaseFactory, tmp_path: Path
) -> None:
    with pytest.raises(KeyError):
        client.status("no-such-job")
    with pytest.raises(ConfigError, match="no run"):
        client.run_summary("0" * 64)
    with pytest.raises(ConfigError):
        client.submit_run({"run": {"name": "x", "seed": 1}, "method": {"name": "nope"}})

    # 200 members x 5 forecasts = 1000 simulator runs: over this root's limit of 500.
    large = linear_case(ensemble_size=200, name="large")
    with pytest.raises(ApprovalRequired) as refused:
        client.submit_run(large.config)
    token = refused.value.token
    approved = CliRunner().invoke(cli, ["approve", token, "--root", str(tmp_path / "service")])
    assert approved.exit_code == 0, approved.output
    handle = client.submit_run(large.config, approval_token=token)
    assert client.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"


def test_the_service_policy_holds_for_both_clients(
    linear_case: CaseFactory, tmp_path: Path
) -> None:
    """A submission outside [policy] output_roots is refused, in process and over HTTP."""
    import tomllib

    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(
        f'[policy]\ninput_roots = ["{tmp_path}"]\ncode_roots = ["{tmp_path / "trusted"}"]\n'
        f'output_roots = ["{tmp_path / "allowed"}"]\n'
    )
    case = linear_case(name="policed")
    config = tomllib.loads(case.config.read_text())
    for kind, client in (("in_process", Client(root)), ("http", _remote(root))):
        config["run"]["output_dir"] = str(tmp_path.parent / f"{tmp_path.name}-forbidden" / kind)
        validation = client.validate(config, base_dir=case.directory)
        assert not validation.valid, kind
        assert "outside [policy] output_roots" in validation.errors[0]
        with pytest.raises(ConfigError, match="policy"):
            client.submit_run(config, base_dir=case.directory)
        config["run"]["output_dir"] = str(tmp_path / "allowed" / kind)
        assert client.validate(config, base_dir=case.directory).valid, kind
    assert not (tmp_path.parent / f"{tmp_path.name}-forbidden").exists()


# ---------------------------------------------------------------------- HTTP only


@pytest.fixture
def http(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(tmp_path / "service", token=TOKEN)) as test_client:
        test_client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield test_client


def test_a_config_file_outside_the_policy_is_not_opened(tmp_path: Path) -> None:
    """/v1/configs:validate peeked into config_file first."""
    root = tmp_path / "service"
    root.mkdir()
    allowed = tmp_path / "allowed"
    (root / "service.toml").write_text(
        f'[policy]\ninput_roots = ["{allowed}"]\ncode_roots = ["{allowed}"]\n'
        f'output_roots = ["{allowed}"]\n'
    )
    secret = tmp_path / "secret.toml"
    secret.write_text('api_key = "sk-TOPSECRET"\n')
    with TestClient(create_app(root, token=TOKEN)) as test_client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        for path in (str(secret), "/dev/zero"):  # /dev/zero would never end if it were read
            reply = test_client.post(
                "/v1/configs:validate", json={"config_file": path}, headers=headers
            )
            assert reply.status_code == 422, reply.text
            assert "policy" in reply.text
            assert "TOPSECRET" not in reply.text


def test_every_request_but_health_needs_the_token(tmp_path: Path) -> None:
    root = tmp_path / "service"
    test_client = TestClient(create_app(root))  # the root's own token file
    first = api_token(root)
    assert (root / "api_token").stat().st_mode & 0o777 == 0o600

    assert test_client.get("/v1/health").status_code == 200
    missing = test_client.get("/v1/plugins")
    assert missing.status_code == 401
    assert missing.headers["content-type"] == "application/problem+json"
    assert missing.json()["title"] == "Unauthorised"
    wrong = test_client.get("/v1/plugins", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert test_client.get("/v1/plugins", headers={"Authorization": f"Bearer {first}"}).is_success

    rotated = CliRunner().invoke(cli, ["service", "token", "--rotate", "--root", str(root)])
    second = rotated.stdout.strip()
    assert second != first
    assert (
        test_client.get("/v1/plugins", headers={"Authorization": f"Bearer {first}"}).status_code
        == 401
    )
    assert test_client.get("/v1/plugins", headers={"Authorization": f"Bearer {second}"}).is_success


def test_submission_is_accepted_idempotent_and_streams_its_events(
    http: TestClient, linear_case: CaseFactory
) -> None:
    body = {"config_file": str(linear_case(ensemble_size=20).config)}

    first = http.post("/v1/runs", json=body)
    assert first.status_code == 202
    job = first.json()["id"]
    assert first.headers["location"] == f"/v1/jobs/{job}"
    RemoteClient("http://testserver", TOKEN, http=http).wait(job, timeout_s=TIMEOUT, poll_s=0.2)
    second = http.post("/v1/runs", json=body)
    assert second.status_code == 202
    assert second.json()["reused"]
    assert second.json()["run_ids"] == first.json()["run_ids"]

    # Server-Sent Events, resumed after event 2: the next is event 3; the stream ends
    # with the job.
    stream = http.get(
        f"/v1/jobs/{job}/events",
        headers={"Accept": "text/event-stream", "Last-Event-ID": "2"},
    )
    assert stream.headers["content-type"].startswith("text/event-stream")
    ids = [int(line[4:]) for line in stream.text.splitlines() if line.startswith("id: ")]
    polled = http.get(f"/v1/jobs/{job}/events", params={"since": 3}).json()
    assert ids[0] == 3
    assert ids == [e["index"] for e in polled["events"]]
    remote = RemoteClient("http://testserver", TOKEN, http=http)
    assert [e.index for e in remote.follow(job, since=3)] == ids

    feed = http.get("/v1/events").json()
    assert {e["job"] for e in feed["events"]} >= {job}
    later = http.get("/v1/events", params={"since": feed["next_since"]}).json()
    assert later["events"] == []


def test_lists_are_paged_and_artifacts_are_confined_to_their_run(
    http: TestClient, linear_case: CaseFactory
) -> None:
    remote = RemoteClient("http://testserver", TOKEN, http=http)
    # Distinct ensembles: identical cases would be one content-addressed run.
    handles = [
        remote.submit_run(
            linear_case(ensemble_size=10 + i, name=f"c{i}").config, annotations={"i": str(i)}
        )
        for i in range(3)
    ]
    for handle in handles:
        assert remote.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"

    page = http.get("/v1/jobs", params={"limit": 2}).json()
    assert len(page["items"]) == 2
    rest = http.get("/v1/jobs", params={"limit": 2, "cursor": page["next_cursor"]}).json()
    assert rest["next_cursor"] is None
    assert {j["id"] for j in page["items"] + rest["items"]} == {h.id for h in handles}
    assert http.get("/v1/jobs", params={"cursor": "garbage!"}).status_code == 400
    assert [r["annotations"]["i"] for r in remote.runs({"i": "1"})] == ["1"]
    assert len(remote.runs()) == 3

    run = handles[0].run_ids[0]
    summary = json.loads(remote.artifact(run, "summary.json"))
    assert summary["complete"]
    assert http.get(f"/v1/runs/{run}/figures").json() == []
    report = Path(remote.runs({"i": "0"})[0]["path"]) / "report" / "health"
    report.mkdir(parents=True)
    (report / "spread.png").write_bytes(b"png")
    (report / "spread.pdf").write_bytes(b"pdf")
    run0 = remote.runs({"i": "0"})[0]["run_id"]
    assert http.get(f"/v1/runs/{run0}/figures").json() == ["report/health/spread.png"]
    assert http.get(f"/v1/runs/{'0' * 64}/figures").status_code == 404
    for escape in ("../../api_token", "..%2F..%2Fapi_token", "%2e%2e/%2e%2e/api_token"):
        # Refused here, or normalised by the client into another route: never served.
        response = http.get(f"/v1/runs/{run}/artifacts/{escape}")
        assert response.status_code in (404, 422), escape
        assert TOKEN not in response.text


def test_problem_documents_carry_the_errors(http: TestClient) -> None:
    unknown = http.post("/v1/runs", json={"config_file": "x.toml", "colour": "red"})
    assert unknown.status_code == 422
    problem = unknown.json()
    assert problem["title"] == "Invalid request"
    assert any(e["location"].endswith("colour") for e in problem["errors"])
    invalid = http.post("/v1/configs:validate", json={"config_file": "/no/such/file.toml"})
    assert invalid.status_code == 422
    assert invalid.headers["content-type"] == "application/problem+json"
    assert http.get("/v1/jobs/nope").status_code == 404
    assert http.post("/v1/jobs/nope:stop").status_code == 404


def test_the_openapi_document_describes_the_models(http: TestClient) -> None:
    document = http.get("/openapi.json").json()

    paths = document["paths"]
    assert {
        "/v1/runs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/events",
        "/v1/configs:validate",
    } <= set(paths)
    schemas = document["components"]["schemas"]
    assert {"JobHandle", "JobStatus", "Validation", "RunSummary", "Comparison"} <= set(schemas)
    assert set(schemas["JobStatus"]["properties"]) >= {"id", "state", "children"}


def test_a_grant_lets_a_large_submission_through_over_http(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A grant created with the CLI approves a matching REST submission."""
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text("[approval]\nmax_simulator_runs = 500\n")
    budget = ["--max-simulator-runs", "1500", "--expires", "1d", "--root", str(root)]
    created = CliRunner().invoke(cli, ["grant", "create", "planner", "X", *budget])
    assert created.exit_code == 0, created.output
    # 200 members x 5 forecasts = 1000 simulator runs: over this root's limit of 500.
    large = linear_case(ensemble_size=200, name="large")
    with _remote(root) as remote:
        with pytest.raises(ApprovalRequired, match="no experiment annotation"):
            remote.submit_run(large.config, client="planner")
        handle = remote.submit_run(large.config, client="planner", annotations={"experiment": "X"})
        assert remote.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
        # The budget has 500 left: the same cost again is refused, with the reason. Another
        # run (other inflation factors): an identical one would be reused, needing no approval.
        again = linear_case(
            ensemble_size=200,
            name="again",
            method='name = "esmda"\noptions = { inflation = [2, 4, 8, 8] }\n'
            'strategy = { name = "approx", options = { energy = 1.0 } }',
        )
        with pytest.raises(ApprovalRequired, match="a grant has 500 simulator runs left"):
            remote.submit_run(again.config, client="planner", annotations={"experiment": "X"})
    listed = CliRunner().invoke(cli, ["grant", "list", "--root", str(root)])
    assert "1,000/1,500" in listed.output


def test_no_route_creates_or_changes_grants(http: TestClient) -> None:
    """Grants and approvals are made only by the CLI on the service's machine."""
    routes = [getattr(r, "path", "") for r in http.app.routes]  # type: ignore[attr-defined]
    assert not [p for p in routes if "grant" in p or "approv" in p]
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        for path in ("/v1/grants", "/v1/grants/abc", "/v1/approvals", "/v1/approvals/abc"):
            assert http.request(method, path, json={}).status_code in (404, 405), (method, path)
    # No request body accepts a grant either: unknown keys are refused.
    reply = http.post("/v1/runs", json={"config_file": "x.toml", "grant": {"budget": 10**9}})
    assert reply.status_code == 422
    assert "grant" in reply.text  # refused for the unknown key, not for the missing file


# ---------------------------------------------------------------------- the real service


def _apex(*args: str) -> list[str]:
    import sys

    return [sys.executable, "-c", "from apex.cli.main import app; app()", *args]


def test_the_service_serves_the_api_over_http(tmp_path: Path, linear_case: CaseFactory) -> None:
    import socket
    import subprocess

    root = tmp_path / "service"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started = subprocess.run(
        _apex("service", "start", "--http", f"127.0.0.1:{port}", "--root", str(root)),
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert started.returncode == 0, started.stdout + started.stderr
    try:
        token = subprocess.run(
            _apex("service", "token", "--root", str(root)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        remote = _wait_for(f"http://127.0.0.1:{port}", token)
        handle = remote.submit_run(linear_case(ensemble_size=10).config)
        # The service (not this process) starts the worker: the queue is its own.
        assert remote.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.3).state == "complete"
        assert remote.run_summary(handle.run_ids[0]).iterations == [0, 1, 2, 3, 4]
    finally:
        stopped = subprocess.run(
            _apex("service", "stop", "--root", str(root)),
            capture_output=True,
            text=True,
            check=False,
        )
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert not (root / "service.pid").exists()


def _wait_for(url: str, token: str) -> RemoteClient:
    import time

    import httpx

    deadline = time.monotonic() + 60
    while True:
        try:
            httpx.get(f"{url}/v1/health", timeout=1).raise_for_status()
            return Client.connect(url, token)
        except httpx.HTTPError:
            assert time.monotonic() < deadline, "the REST API did not start"
            time.sleep(0.2)


def test_artifact_errors_other_than_404_are_not_missing_files(http: TestClient) -> None:
    """Every HTTP error used to read as "no file"; a wrong token is a 401, not a KeyError."""
    with pytest.raises(KeyError, match="no file"):
        RemoteClient("http://testserver", TOKEN, http=http).artifact("nope", "summary.json")
    with pytest.raises(RemoteError) as error:
        RemoteClient("http://testserver", "wrong-token", http=http).artifact("nope", "summary.json")
    assert error.value.status == 401


def test_an_optimization_is_validated_submitted_and_summarised(
    client: AnyClient, tmp_path: Path
) -> None:
    """The same calls in process and over HTTP (NPV on the toy reservoir)."""
    import tomli_w

    times = [f"{2001 + k}-01-01" for k in range(3)]
    config = {
        "optimization": {"name": "toy", "seed": 1},
        "controls": {"variables": [{"name": "inj", "initial": [300.0], "lower": 10.0,
                                    "upper": 400.0, "variance": 100.0}]},
        "objective": {"name": "npv", "options": {"start": "2000-01-01", "times": times,
                                                 "wop": 1.0, "wwi": 1.0, "obj_scaling": -1.0}},
        "optimizer": {"name": "enopt", "options": {"ne": 3, "maxiter": 2}},
        "simulator": {"name": "toy_reservoir"},
        "execution": {"name": "serial", "options": {"retries": 0}},
    }  # fmt: skip
    path = tmp_path / "toy_optimization.toml"
    path.write_text(tomli_w.dumps(config), encoding="utf-8")
    validation = client.validate_optimization(path)
    assert validation.valid, validation.errors
    assert validation.optimization is not None
    handle = client.submit_optimization(path)
    assert handle.kind == "optimization"
    assert client.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    [row] = client.optimizations()
    summary = client.optimization_summary(Path(row["path"]))
    assert summary.state == "finished"
    assert summary.evaluations > 0
