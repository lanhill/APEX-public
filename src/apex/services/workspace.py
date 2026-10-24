"""The configuration files the UI's editor may list, read and write.

The service executes what configurations describe, so writing is confined:

- only ``.toml`` files, at most :data:`MAX_BYTES`, under a ``[workspace]`` root (paths
  resolved, symlinks followed, before they are matched);
- never under the policy's ``code_roots`` (templates are code);
- atomically, and only if the file is still what the editor read (its SHA-256), so that
  two editors, or an editor and a shell, cannot overwrite each other silently.

Without roots in ``service.toml`` the workspace is ``<root>/configs``, except under an
active ``[policy]``, where the editor stays off until roots are set.

**Paths are relative to the workspace**: with one root, relative to
it (``multi_5spots/configs/gies.toml``); with several, prefixed by each root's name, its
folder name (``APEX/multi_5spots/...``; a repeated name gets ``-2``, ``-3``, ...). Absolute
paths under a root are accepted too. Every listed and returned path is relative, so a
browser shows no server paths. The REST API takes such relative paths wherever a client
sends a path (:meth:`Workspace.absolute`), and gives guests no server path at all.

**The web UI's own folder**: what is submitted through the web UI writes its
runs, studies and the assistant's drafts under ``<first root>/<ui_folder>/<user>/``
(``ui_folder`` default ``UI``): ``owner`` for the owner's token, ``guests/<name>`` for each
guest (:meth:`Workspace.ui_user_folder`).
"""

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from apex.core.errors import ApexError, ConfigError
from apex.services import policy
from apex.services.settings import ServiceSettings

#: The largest file the editor writes or reads (1 MB).
MAX_BYTES = 1_000_000
SUFFIX = ".toml"
#: What the web UI shows read-only beside the configurations: the assistant's saved
#: reports (Markdown), and only in a chat's drafts folder (2026-10-05, a reported issue).
READ_ONLY_SUFFIXES = (".md",)


def _opened(fd: int, target: Path) -> Path:
    """Return the real path of an open file.

    Linux's ``/proc/self/fd``; elsewhere ``target`` if the open file is still it (same
    device and inode), else a path that lies nowhere.
    """
    try:
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    except OSError:
        try:
            opened, there = os.fstat(fd), target.stat()
        except OSError:  # gone since: not the file that was checked
            return Path("/nonexistent")
        same = (opened.st_dev, opened.st_ino) == (there.st_dev, there.st_ino)
        return target if same else Path("/nonexistent")


class WorkspaceError(ConfigError):
    """A request outside the workspace, or for a file the editor may not touch."""


class WriteConflict(ApexError):
    """The file changed since the editor read it (or exists, for a new file)."""

    def __init__(self, path: Path, current: str | None) -> None:
        state = "exists" if current is not None else "no longer exists"
        super().__init__(f"{path} changed since it was read ({state}); reload it and edit again")
        self.path, self.current = path, current


@dataclass(frozen=True, slots=True)
class Entry:
    """A folder or ``.toml`` file of the workspace."""

    path: str
    name: str
    kind: str  # "folder" or "file"
    size: int | None = None


@dataclass(frozen=True, slots=True)
class FileContent:
    """A file's text and its SHA-256 (hex), which a write names to detect conflicts."""

    path: str
    text: str
    sha256: str


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


class Workspace:
    """The editor's folders (see the module docstring).

    Parameters
    ----------
    root
        The services root (its ``configs/`` is the default workspace).
    settings
        The root's ``service.toml``.
    """

    def __init__(self, root: Path, settings: ServiceSettings) -> None:
        self.settings = settings
        configured = [Path(r) for r in settings.workspace.roots]
        if configured:
            self.roots = [r.resolve() for r in configured]
        elif policy.active(settings.policy):
            self.roots = []
        else:
            default = Path(root) / "configs"
            default.mkdir(parents=True, exist_ok=True)
            self.roots = [default.resolve()]
        self.names: list[str] = []
        for r in self.roots:
            name, n = r.name or "root", 2
            while name in self.names:
                name, n = f"{r.name}-{n}", n + 1
            self.names.append(name)

    def ui_user_folder(self, guest: str | None, *, create: bool = True) -> Path | None:
        """Return the web UI's folder of one user (created), or ``None`` without one.

        ``guest`` is the guest's name; ``None`` for the owner. Owner and guests never share
        a folder, whatever a guest is called. ``create = False`` only names it.
        """
        name = self.settings.workspace.ui_folder
        if name is None or not self.roots:
            return None
        folder = self.roots[0] / name / ("owner" if guest is None else f"guests/{guest}")
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def ui_identity_folder(self, kind: str, name: str, *, create: bool = True) -> Path | None:
        """Return the web-UI folder of a user of a kind another package adds.

        ``UI/<kind's folder>/<name>`` (:mod:`apex.services.identities`), apart from the
        owner's and the guests', so neither sees its work.
        """
        ui = self.settings.workspace.ui_folder
        if ui is None or not self.roots:
            return None
        folder = self.roots[0] / ui / kind / name
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    def in_other_ui_user_folder(self, path: str | Path) -> bool:
        """Whether ``path`` lies in a web-UI user's folder that is not the owner's.

        The guests' (``UI/guests/<name>``) and those of the kinds other packages add
        (``UI/<kind>/<name>``, e.g. an agent's): what lies there was written by someone
        other than the owner. ``UI/owner`` and everything outside ``UI`` are the owner's.
        """
        ui = self.settings.workspace.ui_folder
        if ui is None or not self.roots:
            return False
        base = (self.roots[0] / ui).resolve()
        owner = base / "owner"
        target = Path(path).expanduser().absolute().resolve()
        inside = target == base or base in target.parents
        return inside and not (target == owner or owner in target.parents)

    def _root_of(self, path: str | Path) -> tuple[Path, Path]:
        """Return (root, the path under it, unresolved) of a relative or absolute path."""
        given = Path(path)
        if ".." in given.parts:
            raise WorkspaceError(f"{path}: '..' is not allowed in workspace paths")
        if given.is_absolute():
            for r in self.roots:
                if given == r or r in given.parents:
                    return r, given
            resolved = given.resolve()
            for r in self.roots:
                if resolved == r or r in resolved.parents:
                    return r, resolved
            raise WorkspaceError(f"{path} is outside the workspace")
        if len(self.roots) == 1:
            return self.roots[0], self.roots[0] / given
        head, *rest = given.parts or ("",)
        if head not in self.names:
            raise WorkspaceError(f"{path}: begin with a workspace root ({', '.join(self.names)})")
        root = self.roots[self.names.index(head)]
        return root, root.joinpath(*rest)

    def relative(self, path: str | Path) -> str:
        """Return a path under a root as the workspace shows it (see the module docstring).

        Raises
        ------
        WorkspaceError
            If it lies under no root.
        """
        root, full = self._root_of(path)
        sub = full.relative_to(root) if full != root else Path()
        if len(self.roots) == 1:
            return sub.as_posix() if sub.parts else ""
        name = self.names[self.roots.index(root)]
        return (Path(name) / sub).as_posix()

    def absolute(self, path: str | Path) -> Path:
        """Return the server path of a workspace path (relative or absolute), unresolved.

        Symlinks are not followed: what it leads to is judged where it is used
        (:meth:`real`).

        Raises
        ------
        WorkspaceError
            If it contains ``..``, names no root, or lies under none.
        """
        if not self.roots:
            self._resolve(path)  # raises: the editor is off
        return self._root_of(path)[1]

    def describe(self, *, paths: bool = True) -> list[dict[str, str]]:
        """Return the roots: each one's name (the relative paths' prefix) and folder.

        ``paths = False`` leaves the folders out (a guest sees no server path; finding 31
        of the release tester, 2026-10-19): names alone, which every route takes.
        """
        prefix = len(self.roots) > 1
        return [
            {"name": n if prefix else "", **({"path": str(r)} if paths else {})}
            for n, r in zip(self.names, self.roots, strict=True)
        ]

    def _resolve(self, path: str | Path) -> Path:
        if not self.roots:
            raise WorkspaceError(
                "the editor is off: under a [policy], set [workspace] roots in service.toml"
            )
        _, full = self._root_of(path)
        resolved = full.resolve()  # symlinks followed: they must stay under a root
        if not any(resolved == r or r in resolved.parents for r in self.roots):
            raise WorkspaceError(f"{path} is outside the workspace")
        return resolved

    def real(self, path: str | Path) -> Path:
        """Return where a workspace path really lies: symlinks followed, under a root.

        Raises
        ------
        WorkspaceError
            If it, or what it links to, lies outside the workspace.
        """
        return self._resolve(path)

    def _file(self, path: str | Path, *, readable: bool = False) -> Path:
        resolved = self._resolve(path)
        allowed = (SUFFIX, *READ_ONLY_SUFFIXES) if readable else (SUFFIX,)
        if resolved.suffix in READ_ONLY_SUFFIXES and "drafts" not in resolved.parts:
            allowed = (SUFFIX,)  # a saved report lies among a chat's drafts, nowhere else
        if resolved.suffix not in allowed or Path(path).suffix not in allowed:
            raise WorkspaceError(
                f"{path}: the editor handles {SUFFIX} files only"
                + (f" (and shows {', '.join(READ_ONLY_SUFFIXES)} files)" if readable else "")
            )
        if any(policy.under(resolved, code) for code in self.settings.policy.code_roots):
            raise WorkspaceError(f"{path} lies under [policy] code_roots, which are not editable")
        return resolved

    def file(self, path: str | Path) -> Path:
        """Return where a ``.toml`` workspace path lies on disk (it need not exist yet).

        Raises
        ------
        WorkspaceError
            If it is outside the workspace, not TOML, or under the policy's code roots.
        """
        return self._file(path)

    def list(self, path: str | Path | None = None) -> list[Entry]:
        """Return the folders and ``.toml`` files in a workspace folder (default: the roots).

        Hidden entries are left out; folders come first, then files, each by name.

        Raises
        ------
        WorkspaceError
            If ``path`` is outside the workspace or not a folder.
        """
        if path is None or str(path) in ("", "."):
            if not self.roots:
                self._resolve("/")  # raises: the editor is off
            if len(self.roots) > 1:
                return [Entry(n, n, "folder") for n in self.names]
            path = self.roots[0]
        folder = self._resolve(path)
        base = self.relative(path)
        if not folder.is_dir():
            raise WorkspaceError(f"{path} is not a folder")
        folders: list[Entry] = []
        files: list[Entry] = []
        for child in sorted(folder.iterdir(), key=lambda p: p.name):
            if child.name.startswith("."):
                continue
            shown = f"{base}/{child.name}" if base else child.name
            if child.is_dir():
                folders.append(Entry(shown, child.name, "folder"))
            elif child.is_file() and (child.suffix == SUFFIX or (
                child.suffix in READ_ONLY_SUFFIXES and "drafts" in child.parts
            )):  # fmt: skip
                files.append(Entry(shown, child.name, "file", child.stat().st_size))
        return folders + files

    def read(self, path: str | Path, allowed: Callable[[Path], bool] | None = None) -> FileContent:
        """Return a ``.toml`` (or read-only ``.md``) file's text and SHA-256.

        With ``allowed``, the file actually opened must lie where it says (its real path,
        asked of the open file): a link swapped after the caller checked the path, and
        before it was opened, is refused rather than followed.

        Raises
        ------
        WorkspaceError
            If it is outside the workspace, not TOML, too large or not UTF-8.
        KeyError
            If it does not exist, or (with ``allowed``) is not where it may be read.
        """
        target = self._file(path, readable=True)
        if not target.is_file():
            raise KeyError(f"no file {path}")
        # Non-blocking, and a regular file only: something else swapped in after the check
        # above (a FIFO would block the open) is refused, not waited on.
        fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise KeyError(f"no file {path}")
            if allowed is not None and not allowed(_opened(fd, target)):
                raise KeyError(f"no file {path}")
            data = handle.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise WorkspaceError(f"{path} is larger than {MAX_BYTES} bytes")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise WorkspaceError(f"{path} is not UTF-8 text") from error
        return FileContent(self.relative(path), text, sha256(data))

    def write(self, path: str | Path, text: str, expected_sha256: str | None) -> FileContent:
        """Write a ``.toml`` file atomically, if it is still what the editor read.

        ``expected_sha256`` is the SHA-256 the editor read; ``None`` creates a new file,
        which must not exist. Missing folders under a root are created.

        Raises
        ------
        WorkspaceError
            If the file is outside the workspace, not TOML or too large.
        WriteConflict
            If the file changed since it was read, or exists when a new file was meant.
        """
        target = self._file(path)
        data = text.encode("utf-8")
        if len(data) > MAX_BYTES:
            raise WorkspaceError(f"{path}: larger than {MAX_BYTES} bytes")
        current = sha256(target.read_bytes()) if target.is_file() else None
        if current != expected_sha256:
            raise WriteConflict(Path(self.relative(path)), current)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, target)
        return FileContent(self.relative(path), text, sha256(data))

    def delete(self, path: str | Path, expected_sha256: str, within: Path) -> str:
        """Delete a ``.toml`` file under ``within`` (a user's web-UI folder).

        Only if it is still what the editor read; the folders it leaves empty go too, up
        to ``within``. Returns the path as the workspace shows it.

        Raises
        ------
        WorkspaceError
            If it is outside ``within`` or the workspace, or not TOML.
        KeyError
            If it does not exist.
        WriteConflict
            If it changed since it was read.
        """
        target = self._file(path, readable=True)  # a saved report too
        base = within.resolve()
        if base not in target.parents:
            raise WorkspaceError(f"{path}: only files in your own UI folder can be deleted")
        if not target.is_file() or target.is_symlink():
            raise KeyError(f"no file {path}")
        current = sha256(target.read_bytes())
        if current != expected_sha256:
            raise WriteConflict(Path(self.relative(path)), current)
        shown = self.relative(path)
        target.unlink()
        folder = target.parent
        while folder != base and base in folder.parents and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent
        return shown
