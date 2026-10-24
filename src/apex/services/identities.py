"""Kinds of web-UI user that installed packages add.

Core knows two kinds of web-UI user: the owner and guests. Another package may add a kind
(the entry points of ``apex.ui_identities``): its users sign in with tokens of their own,
work in a web-UI folder of their own, and see what the kind's access list allows. Each user
of a kind is ``<kind>:<name>``: the ``submitted_by`` of their submissions and the owner of
their chats.

- Tokens: ``<root>/<tokens>/<name>`` (:func:`identity_token`), web-UI tokens that expire as
  the others do.
- Folder: ``<first workspace root>/<ui_folder>/<folder>/<name>``
  (:meth:`apex.services.workspace.Workspace.ui_identity_folder`), apart from the owner's and
  the guests', so neither sees its work.
- Access: the kind's own list (an :class:`~apex.services.access.Access`); every guard of web-UI
  users applies.
"""

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from apex.services.access import Access
from apex.services.auth import named_ui_token, named_ui_token_files

_log = logging.getLogger(__name__)

#: The entry-point group of the kinds of web-UI user other packages add.
IDENTITIES_GROUP = "apex.ui_identities"

#: The kinds core has: their names cannot be taken.
_RESERVED = frozenset({"owner", "guest", "ui"})
#: The web-UI folders and token folders core's users have: no added kind may take them.
_CORE_FOLDERS = frozenset({"owner", "guests"})
_CORE_TOKENS = frozenset({"ui_tokens", "ui_token", "api_token", "ui_owner_token"})
#: A kind's, folder's, token folder's or user's name: one path segment, no ":".
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")

AccessFactory = Callable[[Iterable[Mapping[str, Any]], str, Sequence[str]], Access]


@dataclass(frozen=True)
class UiIdentity:
    """A kind of web-UI user another package adds.

    Parameters
    ----------
    kind
        The kind's name: its users are ``<kind>:<name>``.
    folder
        Its users' web-UI folders, ``<ui_folder>/<folder>/<name>``.
    tokens
        The folder of its users' tokens under the services root.
    access
        Returns a user's access list from the job records, the user's name and the
        workspace folders hidden from them (the web UI's other users').
    sees_owner_runs
        Whether its users' chats may also draft from the owner's runs (read-only).
    """

    kind: str
    folder: str
    tokens: str
    access: AccessFactory
    sees_owner_runs: bool = False

    def submitter(self, name: str) -> str:
        """Return a user's ``submitted_by`` (and chat owner): ``<kind>:<name>``."""
        return f"{self.kind}:{name}"

    def token(
        self, root: Path, name: str, *, create: bool = False, rotate: bool = False
    ) -> str | None:
        """Return a user's web-UI token (``None`` if never created and ``create`` is off)."""
        return named_ui_token(Path(root) / self.tokens, name, create=create, rotate=rotate)

    def token_files(self, root: Path) -> dict[str, Path]:
        """Return every user's token file, by name."""
        return named_ui_token_files(Path(root) / self.tokens)


@cache
def ui_identities() -> tuple[UiIdentity, ...]:
    """Return the kinds of web-UI user the installed packages add (a broken one is skipped)."""
    found: list[UiIdentity] = []
    for point in entry_points(group=IDENTITIES_GROUP):
        try:
            identity = point.load()()
        except Exception:  # noqa: BLE001 (a broken package must not lock everyone out)
            _log.exception("web-UI identity %s could not be loaded; skipped", point.name)
            continue
        if not isinstance(identity, UiIdentity):
            _log.error("web-UI identity %s is not a UiIdentity; skipped", point.name)
            continue
        problem = _problem(identity, found)
        if problem:
            _log.error("web-UI identity %s %s; skipped", point.name, problem)
            continue
        found.append(identity)
    return tuple(found)


def _problem(identity: UiIdentity, found: Sequence[UiIdentity]) -> str | None:
    """Return why a kind may not be added beside ``found``, or ``None``.

    Its names must be single path segments, and it may take neither core's kinds, folders
    and token files nor another kind's: users of a kind in the guests' folder, or signing in
    from another's tokens, would see and be seen as those users.
    """
    names = (identity.kind, identity.folder, identity.tokens)
    if not all(isinstance(n, str) and _NAME.fullmatch(n) for n in names):
        return "has a kind, folder or tokens that is not a plain name"
    if identity.kind in _RESERVED or any(i.kind == identity.kind for i in found):
        return f"takes the kind {identity.kind!r}"
    if identity.folder in _CORE_FOLDERS or any(i.folder == identity.folder for i in found):
        return f"takes the folder {identity.folder!r}"
    if identity.tokens in _CORE_TOKENS or any(i.tokens == identity.tokens for i in found):
        return f"takes the tokens {identity.tokens!r}"
    return None


def identity_of(submitter: str) -> tuple[UiIdentity, str] | None:
    """Return the kind and name of a ``<kind>:<name>`` of an added kind, else ``None``."""
    kind, _, name = submitter.partition(":")
    if not _NAME.fullmatch(name):
        return None
    for identity in ui_identities():
        if identity.kind == kind:
            return identity, name
    return None
