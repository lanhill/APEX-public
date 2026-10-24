"""APEX's MCP server, shared by ``apex mcp`` and the service's REST app.

It serves the tool registry (:mod:`apex.services.tools`) and exposes runs and study
comparisons as resources: ``apex://runs/<id>/summary`` and
``apex://studies/<name>/comparison``. Every tool call goes through the dispatcher, so
approval tokens, session budgets, bounded results and the call log apply to MCP clients
as to any other. Over stdio (``apex mcp``) one tool session serves the host; over HTTP
(the service's ``/mcp``, behind its bearer token) each MCP session gets its own
(:class:`HttpSessions`), named by the client's ``X-APEX-Client``/``X-APEX-Model``
headers. Needs the ``mcp`` extra.

Both HTTP servers need the root's bearer token (``apex service token``): :class:`BearerGuard`
for ``apex mcp --http``, :func:`bearer_of` for the service's ``/mcp``. On a shared machine a
loopback port is reachable by every local user, so the port alone is no protection
(release tester N2, 2026-10-20).
"""

import json
import weakref
from collections.abc import Callable
from typing import Any

from apex.services.api import ApexService
from apex.services.auth import token_matches
from apex.services.tools import TOOLS, ToolSession, call

#: What a client without the token is told.
UNAUTHORISED = "send Authorization: Bearer <apex service token>"


def bearer_of(scope: Any) -> str | None:  # noqa: ANN401 (an ASGI scope)
    """Return the bearer token of an ASGI request's ``Authorization`` header, if any."""
    headers = dict(scope.get("headers") or [])
    given = headers.get(b"authorization", b"").decode("latin-1")
    scheme, _, token = given.partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


class BearerGuard:
    """ASGI middleware: HTTP requests without the bearer token get ``401``.

    Parameters
    ----------
    app
        The ASGI app behind the token (lifespan and other non-HTTP scopes pass through).
    expected
        Returns the token, read at each request so that a rotated token applies at once.
    """

    def __init__(self, app: Any, expected: Callable[[], str]) -> None:  # noqa: ANN401
        self.app = app
        self.expected = expected

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
        """Serve one request, or refuse it without the token.

        Every scope but ``lifespan`` needs the token: an HTTP request gets ``401``, a
        WebSocket is closed before it is accepted, and anything else is dropped.
        """
        kind = scope.get("type")
        if kind == "lifespan" or token_matches(bearer_of(scope), self.expected()):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if kind == "http":
            body = json.dumps({"error": "Unauthorised", "detail": UNAUTHORISED}).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            })  # fmt: skip
            await send({"type": "http.response.body", "body": body})


class HttpSessions:
    """One tool session per MCP session (its budget, its client name), made on first use.

    The client names itself with the ``X-APEX-Client`` header (default ``mcp``), and may
    declare ``X-APEX-Model`` and ``X-APEX-Host``; quotas, approvals and the call log use
    them. A session's budget defaults to its client's daily quota. Sessions are held
    weakly, so they go with the MCP session.
    """

    def __init__(self, service: ApexService) -> None:
        self.service = service
        self._sessions: weakref.WeakKeyDictionary[Any, ToolSession] = weakref.WeakKeyDictionary()

    def for_context(self, context: Any) -> ToolSession:  # noqa: ANN401 (the SDK's context)
        """Return the tool session of the request's MCP session."""
        key = context.session
        session = self._sessions.get(key)
        if session is None:
            headers = getattr(getattr(context, "request", None), "headers", {}) or {}
            session = ToolSession(
                self.service,
                client=str(headers.get("x-apex-client", "mcp")),
                model=headers.get("x-apex-model"),
                host=headers.get("x-apex-host", "http"),
            )
            self._sessions[key] = session
        return session


INSTRUCTIONS = (
    "APEX runs ensemble history matching and predictions. Typical use: validate a "
    "configuration or study file (it returns the cost and an approval token), then submit it "
    "with that token, follow it with job_status and job_events, and read results with "
    "run_summary or compare_study. Submissions above the approval limits need a person to "
    "run `apex approve TOKEN` first."
)


def build_server(sessions: ToolSession | HttpSessions) -> Any:  # noqa: ANN401 (mcp's Server)
    """Return the MCP server (the SDK's lowlevel ``Server``).

    ``sessions`` is one tool session for every call (stdio: one host, one process), or
    :class:`HttpSessions`, which gives each MCP session its own (HTTP: many hosts).
    """
    import anyio
    import mcp_types as types
    from mcp.server.lowlevel import Server

    def session_of(context: Any) -> ToolSession:  # noqa: ANN401 (the SDK's request context)
        return sessions.for_context(context) if isinstance(sessions, HttpSessions) else sessions

    async def list_tools(context: Any, params: Any) -> Any:  # noqa: ANN401
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, input_schema=t.schema())
                for t in TOOLS.values()
            ]
        )

    async def call_tool(context: Any, params: Any) -> Any:  # noqa: ANN401
        # In a worker thread: validation or submission may take seconds on large cases.
        session = session_of(context)
        result = await anyio.to_thread.run_sync(
            call, params.name, dict(params.arguments or {}), session
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
            structured_content=result,
            is_error=not result["ok"],
        )

    async def list_resources(context: Any, params: Any) -> Any:  # noqa: ANN401
        session = session_of(context)
        resources = [
            types.Resource(
                name=f"run {e.name}",
                uri=f"apex://runs/{e.run_id}/summary",
                description=f"results of run {e.run_id[:12]} ({e.path})",
                mime_type="application/json",
            )
            for e in session.service.store.entries()[-50:]
        ]
        studies = session.service.root / "studies"
        for record in sorted(studies.glob("*/study.json")) if studies.exists() else []:
            name = record.parent.name
            resources.append(
                types.Resource(
                    name=f"study {name}",
                    uri=f"apex://studies/{name}/comparison",
                    description=f"comparison of the variants of study {name}",
                    mime_type="application/json",
                )
            )
        return types.ListResourcesResult(resources=resources)

    async def read_resource(context: Any, params: Any) -> Any:  # noqa: ANN401
        session = session_of(context)
        uri = str(params.uri)
        parts = uri.removeprefix("apex://").split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "summary":
            result = call("run_summary", {"run_id": parts[1]}, session)
        elif len(parts) == 3 and parts[0] == "studies" and parts[2] == "comparison":
            result = call("compare_study", {"study": parts[1]}, session)
        else:
            result = {"ok": False, "error": f"unknown resource {uri}"}
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=uri, mime_type="application/json", text=json.dumps(result, default=str)
                )
            ]
        )

    return Server(
        "apex",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )
