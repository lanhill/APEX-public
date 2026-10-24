"""The tool layer for LLM clients: registry, dispatcher, approval, MCP.

A scripted "model" makes the calls: no live model is used (runtime-agent rules).
"""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.services.api import ApexService
from apex.services.tools import RESULT_LIMIT, TOOLS, ToolSession, bound, call, export

TIMEOUT = 180.0
#: ES-MDA with other inflation factors than the linear case's: another run.
OTHER_ESMDA = (
    'name = "esmda"\noptions = { inflation = [2, 4, 8, 8] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _service_root(tmp_path: Path, settings: str = "") -> Path:
    """Write the service's ``service.toml``: ``settings``, and the cases' folder readable.

    Tool clients name files only under the workspace's roots and ``[tools] roots``
    (release tester N3); the cases are written to ``tmp_path``.
    """
    root = tmp_path / "service"
    root.mkdir(exist_ok=True)
    (root / "service.toml").write_text(f'{settings}\n[tools]\nroots = ["{tmp_path}"]\n')
    return root


def _session(tmp_path: Path, **kwargs: Any) -> ToolSession:  # noqa: ANN401
    root = tmp_path / "service"
    if not (root / "service.toml").exists():
        _service_root(tmp_path)
    return ToolSession(ApexService(root), client="planner", model="stub-model", **kwargs)


def _no_refs(node: Any) -> bool:  # noqa: ANN401
    if isinstance(node, dict):
        options = node.get("anyOf")
        null_union = isinstance(options, list) and {"type": "null"} in options
        return "$ref" not in node and not null_union and all(_no_refs(v) for v in node.values())
    if isinstance(node, list):
        return all(_no_refs(v) for v in node)
    return True


def test_every_tool_exports_to_each_format_with_flat_schemas() -> None:
    openai, anthropic, plain = export("openai"), export("anthropic"), export("json-schema")

    assert [t["function"]["name"] for t in openai] == list(TOOLS)
    assert [t["name"] for t in anthropic] == list(TOOLS)
    for tool in plain:
        assert tool["parameters"]["type"] == "object"
        assert _no_refs(tool["parameters"]), tool["name"]  # small-model friendly
    spenders = {t["name"] for t in plain if t["spends_compute"]}
    assert spenders == {
        "submit_run",
        "submit_study",
        "submit_prediction",
        "submit_optimization",
        "resume_job",
    }
    assert len(json.dumps(openai)) < 12_000  # the whole tool set fits a small context


def test_a_scripted_model_validates_submits_and_reads_results(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    session = _session(tmp_path)
    case = linear_case(ensemble_size=20)
    file = str(case.config)

    # A misspelt tool, then a call without its token: errors that say what to do.
    hint = call("sumbit_run", {}, session)["hint"]
    assert hint.startswith("did you mean submit_run")
    missing = call("submit_run", {"config_file": file}, session)
    assert not missing["ok"]
    assert "approval_token" in missing["error"]
    wrong = call("submit_run", {"config_file": file, "approval_token": "made-up"}, session)
    assert "call validate again" in wrong["hint"]
    # Validate: the plan, and a token approved at once (100 simulator runs, 1 run).
    checked = call("validate", {"config_file": file}, session)
    assert checked["ok"], checked
    result = checked["result"]
    assert result["plan"]["max_simulator_runs"] == 100
    assert result["approval"] == "approved"
    token = result["approval_token"]
    # The same file with another override is another submission: refused.
    other = call(
        "submit_run",
        {"config_file": file, "approval_token": token, "overrides": {"run.seed": 5}},
        session,
    )
    assert "another submission" in other["error"]
    submitted = call("submit_run", {"config_file": file, "approval_token": token}, session)
    assert submitted["ok"], submitted
    job = submitted["result"]["job_id"]
    assert session.service.wait(job, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    # The identical run, complete, is reused: nothing runs, so the used token does not
    # matter (release tester N8).
    again = call("submit_run", {"config_file": file, "approval_token": token}, session)
    assert again["result"]["reused"], again

    status = call("job_status", {"job_id": job}, session)["result"]
    assert status["state"] == "complete"
    events = call("job_events", {"job_id": job, "limit": 3}, session)["result"]
    assert len(events["events"]) == 3
    assert events["more"]
    summary = call("run_summary", {"run_id": submitted["result"]["run_id"]}, session)["result"]
    assert summary["iterations"] == [0, 1, 2, 3, 4]
    assert summary["annotations"]["model"] == "stub-model"
    listed = call("list_runs", {"annotations": {"via": "tools"}}, session)["result"]
    assert listed["total"] == 1
    # Every call is recorded, with the declared model.
    calls = session.service.jobs.store.tool_calls()
    assert [c["tool"] for _, c in calls][:3] == ["sumbit_run", "submit_run", "submit_run"]
    assert {c["model"] for _, c in calls} == {"stub-model"}


def test_large_submissions_wait_for_a_person_and_budgets_hold(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    root = _service_root(tmp_path, "[approval]\nmax_simulator_runs = 50\n")
    session = _session(tmp_path)
    file = str(linear_case(ensemble_size=20).config)  # 100 simulator runs

    checked = call("validate", {"config_file": file}, session)["result"]
    assert checked["approval"] == "pending"
    token = checked["approval_token"]
    pending = call("submit_run", {"config_file": file, "approval_token": token}, session)
    assert "still pending" in pending["error"]
    CliRunner().invoke(app, ["approve", token, "--root", str(root)])
    approved = call("submit_run", {"config_file": file, "approval_token": token}, session)
    assert approved["ok"], approved

    small_budget = _session(tmp_path, budget_runs=50)
    # Another run: the identical one is reused now, and a reuse spends nothing.
    file = str(linear_case(name="other", ensemble_size=20, method=OTHER_ESMDA).config)
    token = call("validate", {"config_file": file}, small_budget)["result"]["approval_token"]
    CliRunner().invoke(app, ["approve", token, "--root", str(root)])
    over = call("submit_run", {"config_file": file, "approval_token": token}, small_budget)
    assert "session's budget is 50" in over["error"]


def test_results_are_bounded() -> None:
    big = {"rows": [{"i": i, "text": "x" * 50} for i in range(1000)], "name": "keep"}

    bounded = bound(big, limit=2000)

    assert len(json.dumps(bounded)) <= 2000
    assert bounded["truncated"]
    assert bounded["name"] == "keep"


def test_list_runs_pages_newest_first_without_losing_a_run(tmp_path: Path) -> None:
    """A reported issue: the latest run was guessed from 20 of 39 runs, oldest first.

    The runs come newest first, every page fits a result, and next_offset walks them all.
    """
    created = [f"2026-10-{day:02d}T{hour:02d}:00:00+00:00" for day in (1, 2) for hour in range(24)]
    entries = [
        SimpleNamespace(run_id=f"{i:012d}abcd", name=f"run{i}", created=c,
                        path=tmp_path / ("deep/" * 20) / f"run{i}", annotations={})
        for i, c in enumerate(created)
    ]  # fmt: skip
    store = SimpleNamespace(entries=lambda: list(reversed(entries)))
    session = cast(ToolSession, SimpleNamespace(service=SimpleNamespace(store=store), access=None))

    seen: list[str] = []
    offset: int | None = 0
    while offset is not None:
        page = call_handler("list_runs", session, {"limit": 500, "offset": offset})
        assert page["total"] == len(entries)
        assert page["order"] == "newest first"
        assert len(json.dumps(page, default=str)) <= RESULT_LIMIT
        seen += [row["name"] for row in page["runs"]]
        offset = page.get("next_offset")

    assert seen == [f"run{i}" for i in reversed(range(len(entries)))]
    first = call_handler("list_runs", session, {"limit": 1})
    assert first["runs"][0]["name"] == f"run{len(entries) - 1}"
    assert first["next_offset"] == 1


def test_list_runs_puts_the_later_of_two_runs_of_one_second_first(tmp_path: Path) -> None:
    """Review: ``created`` has one-second resolution; the catalogue's order breaks ties."""
    entries = [
        SimpleNamespace(run_id=f"{i:016d}", name=f"run{i}", created="2026-10-24T10:00:00",
                        path=tmp_path / f"run{i}", annotations={})
        for i in range(3)
    ]  # fmt: skip
    store = SimpleNamespace(entries=lambda: entries)
    session = cast(ToolSession, SimpleNamespace(service=SimpleNamespace(store=store), access=None))
    listed = call_handler("list_runs", session, {})
    assert [row["name"] for row in listed["runs"]] == ["run2", "run1", "run0"]


def call_handler(name: str, session: ToolSession, arguments: dict[str, Any]) -> Any:  # noqa: ANN401
    tool = TOOLS[name]
    return tool.handler(session, tool.arguments.model_validate(arguments))


def _text(item: Any) -> str:  # noqa: ANN401 (an MCP content or resource item)
    text = getattr(item, "text", None)
    assert isinstance(text, str), item
    return text


def test_the_mcp_server_lists_and_calls_the_tools(tmp_path: Path, linear_case: CaseFactory) -> None:
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from apex.cli.mcp_server import build_server

    session = _session(tmp_path)
    file = str(linear_case(ensemble_size=10).config)

    async def scenario() -> dict[str, Any]:
        async with Client(build_server(session)) as client:
            listed = await client.list_tools()
            names = [t.name for t in listed.tools]
            checked = await client.call_tool("validate", {"config_file": file})
            token = json.loads(_text(checked.content[0]))["result"]["approval_token"]
            submitted = await client.call_tool(
                "submit_run", {"config_file": file, "approval_token": token}
            )
            job = json.loads(_text(submitted.content[0]))["result"]["job_id"]
            await anyio.to_thread.run_sync(lambda: session.service.wait(job, timeout_s=TIMEOUT))
            resources = await client.list_resources()
            uri = next(r.uri for r in resources.resources if "runs" in str(r.uri))
            read = await client.read_resource(uri)
            bad = await client.call_tool("no_such_tool", {})
            return {"names": names, "summary": json.loads(_text(read.contents[0])), "bad": bad}

    outcome = anyio.run(scenario)

    assert outcome["names"] == list(TOOLS)
    assert outcome["summary"]["result"]["state"] == "complete"
    assert outcome["bad"].is_error


def test_the_cli_exports_and_calls_tools(tmp_path: Path, linear_case: CaseFactory) -> None:
    runner = CliRunner()
    exported = runner.invoke(app, ["tools", "export", "--format", "anthropic"])
    assert json.loads(exported.stdout)[0]["name"] == "list_plugins"

    file = str(linear_case(ensemble_size=10).config)
    _service_root(tmp_path)
    called = runner.invoke(
        app,
        ["tools", "call", "validate", "--json", json.dumps({"config_file": file}),
         "--model", "glm-4.6", "--root", str(tmp_path / "service")],
    )  # fmt: skip
    assert called.exit_code == 0, called.output
    assert json.loads(called.stdout)["result"]["valid"]


def _apex(*args: str) -> list[str]:
    import sys

    return [sys.executable, "-c", "from apex.cli.main import app; app()", *args]


def test_the_mcp_server_runs_over_stdio(tmp_path: Path) -> None:
    """As Claude Code launches it: a subprocess speaking MCP on stdin and stdout."""
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client, StdioServerParameters

    command = _apex("mcp", "--root", str(tmp_path / "service"), "--model", "stub")
    params = StdioServerParameters(command=command[0], args=command[1:])

    async def scenario() -> tuple[int, str]:
        async with Client(params) as client:
            tools = await client.list_tools()
            plugins = await client.call_tool("list_plugins", {"registry": "localizations"})
            first = json.loads(_text(plugins.content[0]))["result"]["plugins"][0]["name"]
            return len(tools.tools), first

    assert anyio.run(scenario) == (len(TOOLS), "adaptive")


def test_the_mcp_server_runs_over_streamable_http(tmp_path: Path) -> None:
    """Behind the root's API token (release tester N2, 2026-10-20): without it, 401."""
    pytest.importorskip("mcp")
    import socket
    import subprocess
    import time

    import anyio
    import httpx
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    from apex.services.auth import api_token

    root = tmp_path / "service"
    token = api_token(root)  # the token `apex service token --root ROOT` prints
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        _apex("mcp", "--http", f"127.0.0.1:{port}", "--root", str(root)),
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                assert time.monotonic() < deadline, "the HTTP server did not start"
                time.sleep(0.1)

        url = f"http://127.0.0.1:{port}/mcp"
        body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        assert httpx.post(url, json=body).status_code == 401
        wrong = httpx.post(url, json=body, headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert "apex service token" in wrong.json()["detail"]

        async def scenario() -> int:
            headers = {"Authorization": f"Bearer {token}"}
            transport = streamable_http_client(
                url, http_client=create_mcp_http_client(headers=headers)
            )
            async with Client(transport) as client:
                return len((await client.list_tools()).tools)

        assert anyio.run(scenario) == len(TOOLS)
    finally:
        server.terminate()
        _, stderr = server.communicate(timeout=30)
    assert "apex service token" in stderr  # how to pass the token


def _agent_example() -> Any:  # noqa: ANN401 (a module loaded from examples/)
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "examples" / "llm" / "openai_compatible_agent.py"
    spec = importlib.util.spec_from_file_location("openai_compatible_agent", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_agent_loop_runs_tool_calls_through_the_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scripted OpenAI-compatible server: one tool call, then an answer."""
    agent = _agent_example()
    replies = iter(
        [
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "list_plugins", "arguments": '{"registry": "localizations"}'}}]}}]},
            {"choices": [{"message": {"role": "assistant", "content": "adaptive and distance"}}]},
        ]
    )  # fmt: skip
    seen: list[list[dict[str, Any]]] = []

    def fake_chat(base_url: str, model: str, messages: list[dict[str, Any]], key: object) -> Any:  # noqa: ANN401
        seen.append(list(messages))
        return next(replies)

    monkeypatch.setattr(agent, "chat", fake_chat)

    answer = agent.run("which localizations?", "http://stub/v1", "stub-model", None)

    assert answer == "adaptive and distance"
    tool_message = seen[1][-1]
    assert tool_message["role"] == "tool"
    assert "adaptive_per_update" in tool_message["content"]


@pytest.mark.skipif(
    not __import__("os").environ.get("APEX_TEST_OLLAMA_MODEL"),
    reason="opt-in: set APEX_TEST_OLLAMA_MODEL (and APEX_LLM_BASE_URL) to use a local model",
)
def test_a_local_model_can_use_the_tools() -> None:
    """Opt-in smoke test with a real local model (never in CI)."""
    import os

    agent = _agent_example()
    answer = agent.run(
        "List APEX's localization plugins by name, using the tools.",
        os.environ.get("APEX_LLM_BASE_URL", "http://localhost:11434/v1"),
        os.environ["APEX_TEST_OLLAMA_MODEL"],
        None,
    )
    assert "adaptive" in answer.lower()


def test_list_plugins_takes_every_registry(tmp_path: Path) -> None:
    """``list_plugins`` lists each registry, optimizers included (a reported issue)."""
    from typing import get_args

    from apex.services.plugins import REGISTRIES
    from apex.services.tools import Registry

    assert set(get_args(Registry)) == set(REGISTRIES)
    listed = call("list_plugins", {"registry": "optimizers"}, _session(tmp_path))
    assert listed["ok"], listed
    assert "enopt" in json.dumps(listed["result"])


SECRET = "sk-FAKE-1234567890"


def test_validation_errors_never_echo_the_files_values(tmp_path: Path) -> None:
    """Release tester N3 (2026-10-20): validating a file that is not a configuration (a
    ``.env``) returned pydantic's ``input_value={'API_KEY': 'sk-FAKE-...'}``. Errors name
    the keys and what is wrong, never the values."""
    session = _session(tmp_path)
    secrets = tmp_path / "fake.env"
    secrets.write_text(f'API_KEY = "{SECRET}"\n[case]\nkind = "real"\n')

    result = call("validate", {"config_file": str(secrets)}, session)

    text = json.dumps(result)
    assert SECRET not in text
    assert "sk-FAKE" not in text
    assert "input_value" not in text
    assert "API_KEY: Extra inputs are not permitted" in text  # still says what is wrong


def test_tool_clients_read_only_the_workspace_without_a_policy(tmp_path: Path) -> None:
    """Release tester N3: with no ``[policy]``, a tool client named any file the owner can
    read. Now only the workspace's roots, ``[tools] roots`` and the service's own folders,
    refused before the file is opened (so "not found" tells nothing either)."""
    root = tmp_path / "service"
    session = ToolSession(ApexService(root), client="planner")  # no service.toml
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    secrets = outside / "fake.env"
    secrets.write_text(f'API_KEY = "{SECRET}"\n')

    for name, arguments in [
        ("validate", {"config_file": str(secrets)}),
        ("validate", {"config_file": str(outside / "missing.toml")}),
        ("submit_run", {"config_file": str(secrets), "approval_token": "t"}),
        ("prediction_summary", {"path": str(outside)}),
    ]:
        result = call(name, arguments, session)
        assert not result["ok"], (name, result)
        assert "outside the folders tool clients may read" in result["error"]
        assert SECRET not in json.dumps(result)
    assert "[tools] roots" in call("validate", {"config_file": str(secrets)}, session)["hint"]

    # The workspace (by default <root>/configs) is readable; [tools] roots widen it.
    inside = ApexService(root).workspace().roots[0] / "fake.env"
    inside.write_text(f'API_KEY = "{SECRET}"\n')
    assert call("validate", {"config_file": str(inside)}, session)["ok"]
    (root / "service.toml").write_text(f'[tools]\nroots = ["{outside}"]\n')
    widened = ToolSession(ApexService(root), client="planner")
    assert call("validate", {"config_file": str(secrets)}, widened)["ok"]


def test_a_grant_approves_a_tool_clients_submission(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Release tester N4 (2026-10-20): with an active grant for its client and experiment,
    an MCP client's submission stayed "still pending". ``validate`` with the annotations
    says the grant covers it; the submission with them charges the grant, and only
    then."""
    import datetime as dt

    root = _service_root(tmp_path, "[approval]\nmax_simulator_runs = 50\n")
    session = _session(tmp_path)
    grants = session.service.grants
    grant = grants.create("planner", "exp-042", 3000, dt.timedelta(days=1), by="test")
    file = str(linear_case(ensemble_size=20).config)  # 100 simulator runs
    experiment = {"experiment": "exp-042"}

    checked = call("validate", {"config_file": file, "annotations": experiment}, session)
    result = checked["result"]
    assert result["approval"] == "granted", result
    assert result["grant"] == grant["id"]
    assert grants.records()[0]["used_simulator_runs"] == 0  # checked, not charged
    token = result["approval_token"]
    # Another client's token is not paid by planner's grant.
    other = ToolSession(ApexService(root), client="other")
    pending = call("validate", {"config_file": file, "annotations": experiment}, other)
    assert pending["result"]["approval"] == "pending"
    without = call("submit_run", {"config_file": file, "approval_token": token}, session)
    assert "still pending" in without["error"]  # no experiment: the grant is not used
    submitted = call(
        "submit_run",
        {"config_file": file, "approval_token": token, "annotations": experiment},
        session,
    )
    assert submitted["ok"], submitted
    [record] = grants.records()
    assert record["used_simulator_runs"] == 100
    assert record["uses"][0]["token"] == token
    approval = session.service.approvals.store.approval(token)
    assert approval is not None
    assert approval["approved_by"] == f"grant {grant['id']}"

    # The same token again: the identical job is reused (queued or running), charged once.
    again = call(
        "submit_run",
        {"config_file": file, "approval_token": token, "annotations": experiment},
        session,
    )
    assert again["result"]["reused"], again
    assert grants.records()[0]["used_simulator_runs"] == 100


def _granted_session(tmp_path: Path) -> tuple[ToolSession, dict[str, Any]]:
    import datetime as dt

    _service_root(tmp_path, "[approval]\nmax_simulator_runs = 50\n")
    session = _session(tmp_path)
    grant = session.service.grants.create("planner", "exp", 3000, dt.timedelta(days=1), by="t")
    return session, grant


def test_a_failed_submission_refunds_the_grant_and_releases_the_token(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: the grant was charged before the submission; if it then
    failed, the grant had paid for a job that did not exist."""
    from apex.core.errors import ConfigError

    session, _ = _granted_session(tmp_path)
    file = str(linear_case(ensemble_size=20).config)
    notes = {"experiment": "exp"}
    token = call("validate", {"config_file": file, "annotations": notes}, session)["result"][
        "approval_token"
    ]
    arguments = {"config_file": file, "approval_token": token, "annotations": notes}

    def refuse(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise ConfigError("the output folder holds another run")

    monkeypatch.setattr(session.service, "submit_run", refuse)
    failed = call("submit_run", arguments, session)
    assert "holds another run" in failed["error"]
    [grant] = session.service.grants.records()
    assert (grant["used_simulator_runs"], grant["uses"]) == (0, [])
    record = session.service.approvals.store.approval(token)
    assert record is not None
    assert (record["used"], record["approved_by"]) == (False, None)
    assert session.used_runs == 0

    monkeypatch.undo()
    submitted = call("submit_run", arguments, session)
    assert submitted["ok"], submitted
    session.service.wait(submitted["result"]["job_id"], timeout_s=TIMEOUT)
    assert session.service.grants.records()[0]["used_simulator_runs"] == 100

    # The identical run again: reused, nothing runs, so nothing is pending or charged.
    reused = call("validate", {"config_file": file, "annotations": notes}, session)["result"]
    assert (reused["approval"], reused["reused"]) == ("approved", True)
    again = call(
        "submit_run",
        {"config_file": file, "approval_token": reused["approval_token"], "annotations": notes},
        session,
    )
    assert again["result"]["reused"], again
    assert session.service.grants.records()[0]["used_simulator_runs"] == 100


def test_configurations_and_overrides_name_only_readable_files(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Review of release tester N3: confinement checked only the file a call names, so
    ``overrides`` (or a configuration inside the folders) could make validation read any
    file and show its text: ``/etc/passwd`` came back as a CSV header error."""
    case = linear_case(ensemble_size=10)
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[tools]\nroots = ["{case.directory}"]\n')
    session = ToolSession(ApexService(root), client="planner")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "obs.csv"
    secret.write_text(f"{SECRET},b\n1,2\n")
    file = str(case.config)
    assert call("validate", {"config_file": file}, session)["result"]["valid"]

    answers = {
        target: call("validate", {"config_file": file,
                                  "overrides": {"observations.file": target}}, session)
        for target in (str(secret), str(outside / "missing.csv"), "/etc/passwd")
    }  # fmt: skip
    for answer in answers.values():
        assert not answer["ok"], answer
        assert "outside the folders tool clients may read" in answer["error"]
        assert SECRET not in json.dumps(answer)
        assert "root:" not in json.dumps(answer)
        assert "not found" not in answer["error"]
    # A file that exists and one that does not are refused alike.
    shapes = {a["error"].replace(t, "X") for t, a in answers.items() if t != "/etc/passwd"}
    assert len(shapes) == 1

    # What runs may not be set by overrides at all.
    program = call("validate", {"config_file": file,
                                "overrides": {"simulator.options.executable": "/bin/sh"}},
                   session)  # fmt: skip
    assert "overrides may not set" in program["error"]

    # A configuration inside the folders that names a file outside them.
    pointing = case.directory / "pointing.toml"
    pointing.write_text(case.config.read_text().replace('"observations.csv"', f'"{secret}"'))
    refused = call("validate", {"config_file": str(pointing)}, session)
    assert "outside the folders tool clients may read" in refused["error"]
    assert SECRET not in json.dumps(refused)
    submitted = call("submit_run", {"config_file": str(pointing), "approval_token": "t"}, session)
    assert "outside the folders tool clients may read" in submitted["error"]


def test_a_web_ui_users_assistant_is_not_told_the_servers_folders(tmp_path: Path) -> None:
    """Review of release tester N3: a refusal lists the readable folders to the owner's
    clients only; a web-UI user's session (``access`` set) gets no server path."""
    root = tmp_path / "service"
    session = ToolSession(ApexService(root), client="chat", access=object())  # type: ignore[arg-type]
    answer = call("validate", {"config_file": str(tmp_path / "x.toml")}, session)
    assert "outside the folders tool clients may read" in answer["error"]
    assert str(root) not in json.dumps(answer)


def test_the_http_guard_refuses_websockets_without_the_token() -> None:
    """Review of release tester N2: every scope but lifespan needs the token."""
    import anyio

    from apex.services.mcp import BearerGuard

    reached: list[str] = []
    sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
        reached.append(scope["type"])

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {}

    guard = BearerGuard(inner, lambda: "secret")

    async def scenario() -> None:
        await guard({"type": "websocket", "headers": []}, receive, send)
        await guard({"type": "http", "headers": []}, receive, send)
        await guard({"type": "lifespan"}, receive, send)
        await guard({"type": "http", "headers": [(b"authorization", b"Bearer secret")]},
                    receive, send)  # fmt: skip

    anyio.run(scenario)
    assert reached == ["lifespan", "http"]
    assert sent[0] == {"type": "websocket.close", "code": 1008}
    assert sent[1]["status"] == 401


def test_a_failure_after_a_job_was_queued_keeps_the_charge(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second review: a study queues its jobs before it records itself,
    so a later failure refunded the grant and freed the token while its cells ran."""
    from apex.core.errors import ConfigError

    session, _ = _granted_session(tmp_path)
    file = str(linear_case(ensemble_size=20).config)
    notes = {"experiment": "exp"}
    token = call("validate", {"config_file": file, "annotations": notes}, session)["result"][
        "approval_token"
    ]
    real = session.service.submit_run

    def queue_then_fail(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        real(*args, **kwargs)
        raise ConfigError("failed after the first job")

    monkeypatch.setattr(session.service, "submit_run", queue_then_fail)
    failed = call(
        "submit_run", {"config_file": file, "approval_token": token, "annotations": notes}, session
    )
    assert "after the first job" in failed["error"]
    assert session.service.grants.records()[0]["used_simulator_runs"] == 100  # kept
    record = session.service.approvals.store.approval(token)
    assert record is not None
    assert record["used"]


def test_a_reuse_token_never_pays_for_a_run(tmp_path: Path, linear_case: CaseFactory) -> None:
    """Second review of release tester N8: the token validate approved for a free reuse
    bore the run's own fingerprint, so once the run was gone it paid for running it."""
    import shutil

    _service_root(tmp_path, "[approval]\nmax_simulator_runs = 50\n")
    session = _session(tmp_path)
    service = session.service
    file = str(linear_case(ensemble_size=20).config)  # 100 simulator runs
    first = service.submit_run(Path(file), approve=lambda cost: True)
    assert service.wait(first.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    reused = call("validate", {"config_file": file}, session)["result"]
    assert (reused["approval"], reused["reused"]) == ("approved", True)

    shutil.rmtree(service.jobs.record(first.id)["run_dirs"][0])  # pruned after validate
    jobs = len(service.jobs.all_records())
    refused = call("submit_run", {"config_file": file, "approval_token": reused["approval_token"]},
                   session)  # fmt: skip
    assert not refused["ok"]
    assert "another submission" in refused["error"]
    assert len(service.jobs.all_records()) == jobs


def test_a_reuse_token_of_a_stopped_job_never_pays_for_its_run(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Second review of release tester N8: the identical job queued (its folder not made
    yet, so this also checks that a queued job is reused), then stopped after validate."""
    _service_root(
        tmp_path, "[approval]\nmax_simulator_runs = 50\n\n[service]\nmax_concurrent_jobs = 1\n"
    )
    session = _session(tmp_path)
    service = session.service
    slow = linear_case(name="slow", ensemble_size=10, simulator_extra="delay_s = 0.05")
    busy = service.submit_run(slow.config, approve=lambda cost: True)
    file = str(linear_case(ensemble_size=20).config)
    queued = service.submit_run(Path(file), approve=lambda cost: True)
    assert service.jobs.record(queued.id)["state"] == "queued"
    again = service.submit_run(Path(file))  # queued: reused, no approval asked
    assert (again.id, again.reused) == (queued.id, True)
    reused = call("validate", {"config_file": file}, session)["result"]
    assert (reused["approval"], reused["reused"]) == ("approved", True)

    service.stop(queued.id)
    jobs = len(service.jobs.all_records())
    refused = call("submit_run", {"config_file": file, "approval_token": reused["approval_token"]},
                   session)  # fmt: skip
    assert not refused["ok"]
    assert len(service.jobs.all_records()) == jobs
    service.stop(busy.id)


def test_the_owners_programs_pass_but_unreadable_files_and_broken_files_do_not(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second review of release tester N3: programs and templates written in the owner's
    own file are the owner's (only overrides are refused them); a study or optimization
    file that does not load is refused rather than validated unchecked; a prediction's
    run outside the folders is refused before its snapshot is read."""
    from apex.services import policy, tools

    roots = [str(tmp_path / "in")]
    assert tools._readable_item("executable", "/opt/geos/bin/geosx", roots, [])
    assert tools._readable_item("code", "/elsewhere/deck.mako", roots, [])
    assert not tools._readable_item("input", "/elsewhere/obs.csv", roots, [])

    session = _session(tmp_path)
    space = session.service.workspace().roots[0]
    case = linear_case(ensemble_size=10)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "s"\n\n[study.cases]\nlinear = "config.toml"\n\n'
        '[[study.variants]]\nname = "bad"\npatch = { run = { not_a_key = 1 } }\n'
    )
    broken = call("validate", {"config_file": str(study)}, session)
    assert not broken["ok"]
    assert "fix the study" in broken["hint"]

    optimization = space / "opt.toml"
    optimization.write_text("[optimization]\nnot_a_key = 1\n\n[optimizer]\nname = 'enopt'\n")
    refused = call("validate", {"config_file": str(optimization)}, session)
    assert not refused["ok"]
    assert "fix the optimization file" in refused["hint"]

    outside = tmp_path.parent / f"{tmp_path.name}-elsewhere" / "run"  # not under the roots
    read: list[Path] = []
    monkeypatch.setattr(policy, "_run_simulator", lambda path: read.append(path))
    forecast = space / "forecast.toml"
    forecast.write_text(
        f'[prediction]\nname = "p"\nrun_dir = "{outside}"\n\n[prediction.outputs]\n'
        'vectors = ["y"]\n\n[simulator]\noptions = { forward_file = "f.npz" }\n'
    )
    answer = call("validate", {"config_file": str(forecast)}, session)
    assert not answer["ok"]
    assert read == []  # refused before the run's snapshot was read


def test_templates_and_programs_other_web_ui_users_wrote_are_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Third review of release tester N3: a guest's (or an agent's) folder holds files the
    owner did not write. An owner's tool client validating one would render its template
    or run its program as the owner; the owner's own folders stay trusted."""
    import re
    import shutil

    case = linear_case(ensemble_size=10)
    session = ToolSession(ApexService(tmp_path / "service"), client="planner")
    space = session.service.workspace()
    ui = space.roots[0] / "UI"
    opm = (
        '[simulator]\nname = "opm_flow"\n\n[simulator.options]\n'
        'deck_template_file = "deck.mako"\nexecutable = "/opt/sim/flow"\n'
    )
    text = re.sub(r"\[simulator\].*?(?=\n\[execution\])", opm, case.config.read_text(),
                  flags=re.DOTALL)  # fmt: skip
    answers = {}
    for folder in (ui / "guests" / "alice", ui / "agents" / "helper-agent", ui / "owner"):
        folder.mkdir(parents=True)
        for name in ("observations.csv", "truth.npz"):
            shutil.copy(case.directory / name, folder / name)
        (folder / "deck.mako").write_text("RUNSPEC\n")
        (folder / "config.toml").write_text(text)
        answers[folder.parent.name] = call(
            "validate", {"config_file": str(folder / "config.toml")}, session
        )
    for who in ("guests", "agents"):
        assert not answers[who]["ok"], answers[who]
        assert "another web-UI user wrote" in answers[who]["error"]
    assert "web-UI user" not in json.dumps(answers["UI"])  # the owner's: trusted
    assert space.in_other_ui_user_folder(ui / "guests" / "alice" / "x.toml")
    assert not space.in_other_ui_user_folder(ui / "owner" / "x.toml")
    assert not space.in_other_ui_user_folder(space.roots[0] / "cases" / "x.toml")


def test_a_reuse_that_runs_after_all_spends_the_sessions_budget(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Third review of release tester N8: a reuse that turned into a run (its job ended
    meanwhile) was not counted against the session's budget."""
    from apex.domain.services import JobHandle

    session = _session(tmp_path)
    file = str(linear_case(ensemble_size=20).config)  # 100 simulator runs
    monkeypatch.setattr(session.service, "free_reuse", lambda config: True)
    monkeypatch.setattr(
        session.service, "submit_run",
        lambda *a, **k: JobHandle(id="j", kind="run", run_ids=["r"], reused=False),
    )  # fmt: skip
    result = call("submit_run", {"config_file": file, "approval_token": "t"}, session)
    assert result["ok"], result
    assert session.used_runs == 100
    # The review of the combined fixes: an existing live job handed back, even a resume,
    # runs nothing new and is not charged.
    session.used_runs = 0
    monkeypatch.setattr(session.service.jobs, "all_records", lambda: [{"id": "live"}])
    monkeypatch.setattr(
        session.service, "submit_run",
        lambda *a, **k: JobHandle(id="live", kind="resume", run_ids=["r"], reused=True),
    )  # fmt: skip
    second = call("submit_run", {"config_file": file, "approval_token": "t"}, session)
    assert second["ok"], second
    assert session.used_runs == 0


def test_validate_refuses_a_run_folder_holding_another_run_everywhere(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Release tester U10: Validate said "Valid", then Submit refused the run because its
    folder held another run. ``ApexService.validate(check_output=True)`` judges it as the
    submission would, and the tools' validate (which issues approval tokens), ``apex
    validate`` and the in-process ``Client`` all ask for it; an identical run is reused,
    not refused."""
    from apex.client import Client

    session = _session(tmp_path)
    case = linear_case(ensemble_size=20)
    file = str(case.config)
    handle = session.service.submit_run(case.config)
    assert session.service.wait(handle.id, timeout_s=TIMEOUT, poll_s=0.2).state == "complete"
    # The same run name (so the same folder), another configuration.
    other = case.directory / "other.toml"
    other.write_text(case.config.read_text().replace("ensemble_size = 20", "ensemble_size = 21"))
    assert other.read_text() != case.config.read_text()

    refused = call("validate", {"config_file": str(other)}, session)["result"]
    assert refused["valid"] is False
    assert "approval_token" not in refused  # no token for what submitting would refuse
    assert any("already holds another run" in e for e in refused["errors"])
    assert call("validate", {"config_file": file}, session)["result"]["valid"]  # reused

    root = str(tmp_path / "service")
    cli = CliRunner().invoke(app, ["validate", str(other), "--root", root])
    assert cli.exit_code == 1
    assert "already holds another run" in cli.output
    assert CliRunner().invoke(app, ["validate", file, "--root", root]).exit_code == 0

    client = Client(tmp_path / "service")
    assert not client.validate(other).valid
    assert client.validate(case.config).valid
    # Left out by default in the service (a study's cells, a diagnosis' patch).
    assert session.service.validate(other).valid
