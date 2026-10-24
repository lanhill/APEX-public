"""The REST API's bearer tokens, readable by their user only.

- ``<root>/api_token``: the owner's; every request is allowed.
- guests' tokens: ``<root>/ui_token`` (the guest ``guest``) and
  ``<root>/ui_tokens/<name>``; requests are filtered through ``<root>/ui_access.toml``
  (:mod:`apex.services.access`), and each guest sees only their own work. They exist only
  once created (``apex service token --ui [--guest NAME]``).
- ``<root>/ui_owner_token``: the owner's web-UI token, with which the web UI
  sees what the owner sees there (``apex service token --ui --owner``).
- ``<root>/<folder>/<name>``: the users of a kind of web-UI user another package adds
  (:mod:`apex.services.identities`), with the kind's own access list.

Web-UI tokens (the guests' and the owner's) expire ``[ui] token_days`` after they were
issued; the issue time is ``<token file>.issued`` (ISO 8601, UTC), written when
the token is created or rotated (older tokens: the file's modification time). The owner's
API token does not expire.
"""

import datetime as dt
import hmac
import os
import secrets
from pathlib import Path

#: The owner's web-UI token.
OWNER_UI_FILE = "ui_owner_token"
#: The suffix of a token's issue-time file.
ISSUED_SUFFIX = ".issued"


def api_token(root: Path, *, rotate: bool = False) -> str:
    """Return the root's API token, creating it (or a new one, with ``rotate``) if needed."""
    return _token(Path(root) / "api_token", rotate)


#: The guest the original UI token (``<root>/ui_token``) belongs to.
DEFAULT_GUEST = "guest"


def _guest_file(root: Path, guest: str) -> Path:
    if not guest.replace("-", "").replace("_", "").isalnum() or len(guest) > 64:
        raise ValueError(f"guest names are letters, digits, '-' and '_' (got {guest!r})")
    if guest == DEFAULT_GUEST:
        return Path(root) / "ui_token"
    return Path(root) / "ui_tokens" / guest


def ui_token(
    root: Path, *, create: bool = False, rotate: bool = False, guest: str = DEFAULT_GUEST
) -> str | None:
    """Return a guest's UI token (``None`` if it was never created and ``create`` is off).

    Each guest has a token of their own: ``<root>/ui_token`` for the
    default guest, ``<root>/ui_tokens/<name>`` for the others.
    """
    path = _guest_file(root, guest)
    if not path.exists() and not (create or rotate):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return _ui_token(path, rotate)


def owner_ui_token(root: Path, *, create: bool = False, rotate: bool = False) -> str | None:
    """Return the owner's web-UI token (``None`` if never created and ``create`` is off)."""
    path = Path(root) / OWNER_UI_FILE
    if not path.exists() and not (create or rotate):
        return None
    return _ui_token(path, rotate)


def named_ui_token(
    folder: Path, name: str, *, create: bool = False, rotate: bool = False
) -> str | None:
    """Return the web-UI token ``<folder>/<name>``, ``None`` if never created (and not ``create``).

    The token of a user of a kind another package adds (:mod:`apex.services.identities`).

    Raises
    ------
    ValueError
        For a name that is not letters, digits, '-' and '_', or is the default guest's.
    """
    if name == DEFAULT_GUEST:
        raise ValueError(f"a web-UI user may not be called {name!r}, the default guest")
    path = _named_file(Path(folder), name)
    if not path.exists() and not (create or rotate):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return _ui_token(path, rotate)


def named_ui_token_files(folder: Path) -> dict[str, Path]:
    """Return every web-UI token file of ``folder``, by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {p.name: p for p in sorted(folder.iterdir()) if p.is_file() and "." not in p.name}


def _named_file(folder: Path, name: str) -> Path:
    if not name.replace("-", "").replace("_", "").isalnum() or len(name) > 64:
        raise ValueError(f"names are letters, digits, '-' and '_' (got {name!r})")
    return folder / name


def guest_token_files(root: Path) -> dict[str, Path]:
    """Return every guest's token file, by guest name."""
    files = {}
    default = Path(root) / "ui_token"
    if default.exists():
        files[DEFAULT_GUEST] = default
    folder = Path(root) / "ui_tokens"
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file() and "." not in path.name:  # not .tmp nor .issued
                files[path.name] = path
    return files


def issued_at(token_file: Path) -> dt.datetime:
    """Return when a web-UI token was issued (UTC)."""
    stamp = Path(str(token_file) + ISSUED_SUFFIX)
    if stamp.exists():
        return dt.datetime.fromisoformat(stamp.read_text().strip())
    return dt.datetime.fromtimestamp(token_file.stat().st_mtime, dt.UTC)


def expires_at(token_file: Path, days: float) -> dt.datetime:
    """Return when a web-UI token expires: ``days`` after it was issued (UTC)."""
    return issued_at(token_file) + dt.timedelta(days=days)


def _ui_token(path: Path, rotate: bool) -> str:
    """Return a web-UI token, recording its issue time when it is created or rotated."""
    new = rotate or not path.exists()
    token = _token(path, rotate)
    if new:
        stamp = Path(str(path) + ISSUED_SUFFIX)
        stamp.write_text(dt.datetime.now(dt.UTC).isoformat(timespec="seconds") + "\n")
    return token


def guest_tokens(root: Path) -> dict[str, str]:
    """Return every guest's token, by guest name."""
    return {name: path.read_text().strip() for name, path in guest_token_files(root).items()}


def revoke_guest(root: Path, guest: str) -> bool:
    """Delete a guest's token (their requests are refused at once); return whether it existed."""
    path = _guest_file(root, guest)
    if not path.exists():
        return False
    path.unlink()
    Path(str(path) + ISSUED_SUFFIX).unlink(missing_ok=True)
    return True


def _token(path: Path, rotate: bool) -> str:
    if path.exists() and not rotate:
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    # Created with user-only permissions before the token is written into it.
    descriptor = os.open(path.with_suffix(".tmp"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(token + "\n")
    os.replace(path.with_suffix(".tmp"), path)
    return token


def token_matches(given: str | None, expected: str) -> bool:
    """Compare tokens in constant time."""
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())
