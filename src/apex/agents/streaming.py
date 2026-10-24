"""The answer's text out of a streamed JSON reply.

The agent loop asks for a JSON object (``{"answer": "..."}``, or ``{"answer": {"answer":
"..."}}`` with prompted tools), so the raw stream is JSON. :class:`AnswerText` picks out the
first string value of an ``"answer"`` key as it arrives and decodes it (escapes included,
also when split between pieces), so a person sees the answer, not the JSON. A reply that is
a tool call, or no JSON, gives nothing.
"""

import json
import re

_KEY = re.compile(r'"answer"\s*:\s*')


class AnswerText:
    """Feed it the streamed text; it returns the answer's new characters (see above)."""

    def __init__(self) -> None:
        self._buffer = ""
        self._search = 0  # where to look for the next "answer" key
        self._at: int | None = None  # inside the string: the next character to decode
        self.done = False

    def feed(self, piece: str) -> str:
        """Add a streamed piece; return the answer's characters it completes."""
        self._buffer += piece
        if self.done:
            return ""
        if self._at is None and not self._find():
            return ""
        return self._decode()

    def _find(self) -> bool:
        while True:
            match = _KEY.search(self._buffer, self._search)
            if match is None:
                # Keep looking from near the end: the key may still be arriving.
                self._search = max(self._search, len(self._buffer) - 16)
                return False
            if match.end() >= len(self._buffer):
                return False  # the value has not started yet
            if self._buffer[match.end()] == '"':
                self._at = match.end() + 1
                return True
            self._search = match.end()  # an object (the prompted wrapper): look inside

    def _decode(self) -> str:
        assert self._at is not None
        out: list[str] = []
        i, text = self._at, self._buffer
        while i < len(text):
            char = text[i]
            if char == '"':
                self.done = True
                i += 1
                break
            if char != "\\":
                out.append(char)
                i += 1
                continue
            if i + 1 >= len(text):
                break  # an escape split between pieces
            if text[i + 1] == "u":
                if i + 6 > len(text):
                    break
                escape = text[i : i + 6]
                i += 6
            else:
                escape = text[i : i + 2]
                i += 2
            try:
                out.append(json.loads(f'"{escape}"'))
            except json.JSONDecodeError:
                out.append(escape)
        self._at = i
        return "".join(out)
