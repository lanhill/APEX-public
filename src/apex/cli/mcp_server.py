"""The MCP server of APEX: ``apex mcp``.

It serves the tool registry over **stdio** (for Claude Code and other local MCP hosts) or
**Streamable HTTP** (``--http HOST:PORT``). The server itself is
:func:`apex.services.mcp.build_server`. Over HTTP every request needs the root's bearer
token (``Authorization: Bearer <apex service token>``; ``401`` without it), as the
service's REST app needs it at its own ``/mcp`` (``apex service start --http``),
which serves each MCP session a tool session of its own. Needs the ``mcp`` extra.

Example Claude Code configuration (``.mcp.json``)::

    {"mcpServers": {"apex": {"command": "uv", "args": ["run", "--project",
      "/path/to/APEX", "apex", "mcp", "--client", "planner",
      "--model", "claude-opus-5-5"]}}}
"""

from collections.abc import Callable

from apex.services.mcp import INSTRUCTIONS, BearerGuard, build_server
from apex.services.tools import ToolSession

__all__ = ["INSTRUCTIONS", "build_server", "http_app", "serve_http", "serve_stdio"]


def serve_stdio(session: ToolSession) -> None:
    """Serve over stdin and stdout until the host closes them."""
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_server(session)

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(main)


def http_app(session: ToolSession, host: str, expected: Callable[[], str]) -> BearerGuard:
    """Return the Streamable HTTP app (``/mcp``) behind the bearer token ``expected()``."""
    server = build_server(session)
    return BearerGuard(server.streamable_http_app(host=host), expected)


def serve_http(session: ToolSession, host: str, port: int) -> None:
    """Serve over Streamable HTTP at ``http://host:port/mcp``, behind the root's API token."""
    import uvicorn

    from apex.services.auth import api_token

    root = session.service.root
    api_token(root)  # created now, so that `apex service token` prints the one in force
    app = http_app(session, host, lambda: api_token(root))
    uvicorn.run(app, host=host, port=port, log_level="warning")
