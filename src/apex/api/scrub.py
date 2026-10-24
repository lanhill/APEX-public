"""No server paths in what web-UI guests receive (finding 31 of the release tester, 2026-10-19).

The web UI shows workspace-relative paths, and every route takes
them (:meth:`apex.services.workspace.Workspace.absolute`). A UI user's answers carry no
absolute server path either: :class:`PathScrubber` rewrites each workspace root as the
workspace names it (``<root name>/...``; with one root, the path under it; the root itself
``(workspace)``), and other known folders as placeholders (``<service>``, ``<code>``,
``<resources>``, ``<apex>``, ``<python>``, ``<tmp>``, ``<home>``; ``Guard.scrubber``).
:class:`GuestPaths` applies it to the JSON answers and event streams of a request whose
guard is a UI user's.
The owner and other clients get the paths as they are.

**File contents are never rewritten**: a file's text goes back to the editor as it is on
disk, since the editor may save it again: ``GET /v1/files/content`` (``FileContent``) is the
one route that returns a file's text, and it is not scrubbed. Only string *values* are
rewritten, never keys. Paths outside every known folder (a mount elsewhere) pass unchanged.
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Where a request's guard leaves its scrubber in the ASGI scope's state.
STATE_KEY = "apex_guest_paths"
#: Routes whose answers hold a file's contents: sent as they are.
CONTENT_ROUTES = ("/v1/files/content",)
#: What the only root itself is shown as, as the web UI shows it (``Guard.server_path``
#: takes it back); never ``.``, which a sentence's period would turn into ``..``.
ROOT_SHOWN = "(workspace)"


class PathScrubber:
    """Rewrite server paths as the workspace shows them.

    Parameters
    ----------
    prefixes
        ``(server path, shown)`` pairs: ``shown`` is a root's name (``""`` for the only
        root) or a placeholder such as ``<service>``. Longer paths win over the folders
        they lie in; the first pair given for a path wins.
    """

    def __init__(self, prefixes: Iterable[tuple[Path | str, str]]) -> None:
        seen: dict[str, str] = {}
        for path, shown in prefixes:
            text = str(path).rstrip("/")
            if text.startswith("/") and len(text) > 1 and text not in seen:
                seen[text] = shown
        #: The folders rewritten, and what each is shown as.
        self.prefixes = dict(seen)
        # A folder ends at "/", at a character no path name continues with, or at a "."
        # that ends a sentence ("see /srv/w."), but not at ".bak" or "-2".
        self._rules = [
            (re.compile(rf"(?<![\w./-]){re.escape(p)}(/|(?![\w-]|\.[\w-]))"), shown)
            for p, shown in sorted(seen.items(), key=lambda pair: -len(pair[0]))
        ]

    def text(self, text: str) -> str:
        """Return ``text`` with every known server path rewritten."""
        for pattern, shown in self._rules:

            def replace(match: re.Match[str], shown: str = shown) -> str:
                if match.group(1) == "/":  # a path under it
                    return f"{shown}/" if shown else ""
                return shown or ROOT_SHOWN  # the folder itself

            text = pattern.sub(replace, text)
        return text

    def value(self, value: Any) -> Any:  # noqa: ANN401 - any JSON value
        """Return a JSON value with every string value rewritten (keys are left alone)."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.value(v) for k, v in value.items()}
        return value

    def event_stream(self, chunk: str) -> str:
        r"""Return server-sent events with each ``data:`` line's JSON values rewritten.

        Parsed, not matched as text: a path after an escape (``\n``, ``\u00e5``) is found.
        A ``data:`` line that is not JSON is rewritten as text.
        """
        lines = []
        for line in chunk.split("\n"):
            if line.startswith("data:"):
                payload = line[5:].lstrip(" ")
                try:
                    data = json.loads(payload)
                except ValueError:
                    line = "data: " + self.text(payload)
                else:
                    line = "data: " + json.dumps(self.value(data), default=str)
            lines.append(line)
        return "\n".join(lines)


class GuestPaths:
    """ASGI middleware: a UI user's answers through a :class:`PathScrubber`.

    The request's guard (``api.app._authorised``) puts the scrubber in the scope's state
    for UI users only. Scrubbed: ``application/json`` and ``application/problem+json``
    answers, and ``text/event-stream``. Everything else (files, the UI's assets, ``HEAD``,
    empty bodies, bodies that are not UTF-8 JSON) passes as it is, headers and all.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one request, scrubbing a UI user's answer."""
        path = str(scope.get("path", ""))
        if (
            scope["type"] != "http"
            or scope.get("method") == "HEAD"
            or not path.startswith("/v1/")
            or path in CONTENT_ROUTES
        ):
            await self.app(scope, receive, send)
            return
        mode = "pass"
        start: Message | None = None
        chunks: list[bytes] = []
        scrubber: PathScrubber | None = None

        async def wrapped(message: Message) -> None:
            nonlocal mode, start, scrubber
            if message["type"] == "http.response.start":
                state = scope.get("state")
                found = state.get(STATE_KEY) if isinstance(state, dict) else None
                kind = _content_type(message)
                if isinstance(found, PathScrubber) and kind in (
                    "application/json",
                    "application/problem+json",
                ):
                    mode, start, scrubber = "json", message, found
                    return  # sent with the scrubbed body's length
                if isinstance(found, PathScrubber) and kind == "text/event-stream":
                    mode, scrubber = "stream", found
                await send(message)
                return
            if message["type"] == "http.response.body" and scrubber is not None:
                if mode == "json" and start is not None:
                    chunks.append(message.get("body", b""))
                    if message.get("more_body", False):
                        return
                    body = b"".join(chunks)
                    scrubbed = _scrubbed_json(scrubber, body)
                    headers = list(start.get("headers", []))
                    if scrubbed is not None:
                        headers = [(k, v) for k, v in headers if k.lower() != b"content-length"]
                        headers.append((b"content-length", str(len(scrubbed)).encode()))
                        body = scrubbed
                    await send({**start, "headers": headers})
                    await send({"type": "http.response.body", "body": body})
                    return
                if mode == "stream":
                    raw = message.get("body", b"")
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        await send(message)
                        return
                    await send({**message, "body": scrubber.event_stream(text).encode("utf-8")})
                    return
            await send(message)

        await self.app(scope, receive, wrapped)


def _content_type(message: Message) -> str:
    """Return a response's media type (lower case, without parameters)."""
    for key, value in message.get("headers", []):
        if key.lower() == b"content-type":
            return bytes(value).decode("latin-1").split(";")[0].strip().lower()
    return ""


def _scrubbed_json(scrubber: PathScrubber, body: bytes) -> bytes | None:
    """Return a JSON body scrubbed, or ``None`` to send it as it is (empty, not UTF-8 JSON)."""
    if not body:
        return None
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return json.dumps(scrubber.value(document), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
