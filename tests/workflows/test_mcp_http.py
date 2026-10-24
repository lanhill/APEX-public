"""APEX's MCP server in the service's REST app: ``/mcp`` behind the API token."""

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from apex.services.api import ApexService
from apex.services.tools import TOOLS

pytest.importorskip("mcp")
TOKEN = "test-token-0123456789abcdef"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """The REST app (with /mcp) served by uvicorn in a thread; yields its URL and root."""
    import uvicorn

    from apex.api.app import create_app

    root = tmp_path / "service"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(
        create_app(root, token=TOKEN, host="127.0.0.1"), host="127.0.0.1", port=port,
        log_level="warning",
    )  # fmt: skip
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uvicorn_server.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", root
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=30)


def _client(url: str, headers: dict[str, str]) -> Any:  # noqa: ANN401 (mcp's Client)
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client

    transport = streamable_http_client(
        f"{url}/mcp", http_client=create_mcp_http_client(headers=headers)
    )
    return Client(transport)


def test_the_mcp_endpoint_needs_the_token(server: tuple[str, Path]) -> None:
    import httpx

    url, _ = server
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}

    missing = httpx.post(f"{url}/mcp", json=body)
    wrong = httpx.post(f"{url}/mcp", json=body, headers={"Authorization": "Bearer nope"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["content-type"].startswith("application/problem+json")


def test_each_mcp_session_is_its_clients_tool_session(server: tuple[str, Path]) -> None:
    """Tools list and run behind the token; the call log names each session's client."""
    import anyio

    url, root = server

    async def scenario() -> tuple[int, str]:
        auth = {"Authorization": f"Bearer {TOKEN}"}
        async with _client(url, {**auth, "X-APEX-Client": "alpha", "X-APEX-Model": "m1"}) as a:
            tools = await a.list_tools()
            listed = await a.call_tool("list_plugins", {"registry": "localizations"})
        async with _client(url, {**auth, "X-APEX-Client": "beta"}) as b:
            await b.call_tool("list_plugins", {"registry": "methods"})
        first = json.loads(listed.content[0].text)["result"]["plugins"][0]["name"]
        return len(tools.tools), first

    assert anyio.run(scenario) == (len(TOOLS), "adaptive")
    calls = [data for _, data in ApexService(root).jobs.store.tool_calls()]
    assert [(c["client"], c["tool"]) for c in calls] == [
        ("alpha", "list_plugins"),
        ("beta", "list_plugins"),
    ]
    assert calls[0]["model"] == "m1"


def test_the_rest_api_is_unchanged_beside_mcp(server: tuple[str, Path]) -> None:
    import httpx

    url, _ = server
    health = httpx.get(f"{url}/v1/health")
    plugins = httpx.get(f"{url}/v1/plugins", headers={"Authorization": f"Bearer {TOKEN}"})

    assert health.status_code == 200
    assert plugins.status_code == 200
