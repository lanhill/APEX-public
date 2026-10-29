"""Frozen copies of APEX's code, which jobs pinned to a commit run.

A copy is a detached ``git worktree`` of the repository at one commit, with an environment
of its own synced from the commit's ``uv.lock``: ``<code_dir>/<commit12>/`` and its
``.venv``. A job pinned to the commit runs its worker with the copy's interpreter, so
editing the checkout never reaches it, nor its automatic resumes. ``freeze_apex.sh`` (the
case studies) makes the same copies through ``apex code freeze``.

What the copy is besides the worktree lives beside it, never in it (an untracked file would
make every run's provenance say ``dirty``): ``.<commit12>.json`` (the commit, the extras
synced, when it was made), ``.<commit12>.sync.log`` and ``.<commit12>.lock``.

A copy is in use while a queued or running job pins it, or while a live process runs from
it: its command line names the copy, or its ``VIRTUAL_ENV`` is the copy's (``uv run
--project``); a process tied to it only by its working directory is not seen. Processes are
read from ``/proc``, never found with ``pgrep -f``, which matches itself. The clean-up
(:meth:`CodeCopies.collect`) removes only copies with APEX's record, unused for
``keep_days`` since their last pin.

A pinned worker starts in an environment of the copy's (:func:`worker_spawn`): its
interpreter, its ``VIRTUAL_ENV`` and ``bin/`` first on ``PATH`` (programs such as ``ert``
come from it), no ``PYTHONPATH``. The live environment travels along in ``APEX_LIVE_*``
variables, so that a job the pinned worker starts in turn (embedded mode schedules from
workers) runs live code when it is not pinned itself.
"""

import datetime as dt
import fcntl
import fnmatch
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import apex
from apex.core.errors import ConfigError
from apex.services.settings import JobsSection
from apex.storage.sqlite_store import JOB_PROTOCOL

logger = logging.getLogger(__name__)

#: The file whose ``JOB_PROTOCOL`` says which job protocol a commit speaks.
PROTOCOL_FILE = "src/apex/storage/sqlite_store.py"
_PROTOCOL = re.compile(r"^JOB_PROTOCOL = (\d+)$", re.MULTILINE)
_COMMIT = re.compile(r"^[0-9a-f]{12}$")

#: Syncs a copy's environment: (copy, extras, log file).
Sync = Callable[[Path, Sequence[str], Path], None]
#: The live environment, kept for jobs a pinned worker starts (:func:`worker_spawn`).
LIVE_ENV = {"python": "APEX_LIVE_PYTHON", "PATH": "APEX_LIVE_PATH",
            "VIRTUAL_ENV": "APEX_LIVE_VIRTUAL_ENV",
            "PYTHONPATH": "APEX_LIVE_PYTHONPATH"}  # fmt: skip


class CodeError(ConfigError):
    """A frozen copy cannot be made, found or removed."""


@dataclass(frozen=True)
class CodePin:
    """The code a job runs: a commit and the interpreter of its frozen copy."""

    commit: str
    python: Path

    def as_json(self) -> dict[str, str]:
        """Return the pin as kept in the job's record."""
        return {"commit": self.commit, "python": str(self.python)}


@dataclass(frozen=True)
class CodeCopy:
    """One frozen copy."""

    commit: str
    path: Path
    extras: tuple[str, ...]
    created: str | None
    subject: str
    #: When it was last pinned (``None``: never by APEX, e.g. a copy made by hand).
    last_used: str | None = None


def uv_sync(copy: Path, extras: Sequence[str], log: Path) -> None:
    """Sync ``copy``'s environment from its ``uv.lock`` (adding, never removing, packages).

    uv's cache is ``$UV_CACHE_DIR``, else ``.uv-cache`` beside the copies (never the home
    folder's default).
    """
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    # Every workspace package: a plain sync installs only the root project.
    command = [uv, "sync", "--frozen", "--no-dev", "--inexact", "--all-packages"]
    for extra in extras:
        command += ["--extra", extra]
    # The copy's own environment, whatever environment this process runs in.
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    }
    # uv's default cache is in the home folder, and each copy puts the whole lock in it
    # (about 1.4 GB a commit; it filled a home file system twice): without UV_CACHE_DIR it
    # goes beside the copies, on their file system, where uv can also link from it.
    env.setdefault("UV_CACHE_DIR", str(copy.parent / ".uv-cache"))
    with log.open("a") as out:
        done = subprocess.run(
            command, cwd=copy, stdout=out, stderr=subprocess.STDOUT, env=env, check=False
        )
    if done.returncode != 0:
        raise CodeError(f"uv sync failed for the frozen copy {copy}; see {log}")


def workspace_member(path: str, members: list[str], excluded: list[str]) -> bool:
    """Return whether a folder is a member of a uv workspace.

    As uv 0.12 matches them (measured): in ``members`` a ``*`` stays within one folder and a
    ``**`` segment spans any number of them (at least one when it ends the glob: uv's
    ``packages/**`` is not ``packages`` itself); in ``exclude`` a ``*`` crosses ``/`` too. Both
    are case-sensitive, with ``./`` and trailing slashes ignored.

    Parameters
    ----------
    path
        The folder relative to the workspace root, ``/``-separated.
    members, excluded
        The ``[tool.uv.workspace]`` globs.
    """
    path = posixpath.normpath(path)
    if path == ".":  # the workspace's root is no member
        return False
    if not any(_segments_match(path.split("/"), posixpath.normpath(m).split("/")) for m in members):
        return False
    return not any(fnmatch.fnmatchcase(path, posixpath.normpath(e)) for e in excluded)


def _segments_match(parts: list[str], wanted: list[str]) -> bool:
    if not wanted:
        return not parts
    if wanted[0] == "**":
        first = 1 if len(wanted) == 1 else 0
        return any(_segments_match(parts[i:], wanted[1:]) for i in range(first, len(parts) + 1))
    return (
        bool(parts)
        and fnmatch.fnmatchcase(parts[0], wanted[0])
        and _segments_match(parts[1:], wanted[1:])
    )


def _workspace_package(name: str) -> bool:
    normal = name.lower().replace("_", "-")
    return normal == "apex" or normal.startswith("apex-")


def installed_extras() -> list[str]:
    """Return APEX's extras whose requirements are all installed here (the live environment).

    A frozen copy gets the same ones by default, so a configuration that validates here
    (with live code) finds what it imports in the copy too. ``torch`` is never synced.
    """
    # APEX and the workspace's members (named apex-...): a copy syncs them all.
    requirements = [
        requirement
        for dist in metadata.distributions()
        if _workspace_package(str(dist.metadata["Name"]))
        for requirement in dist.requires or []
    ]
    if not requirements:
        return []
    needed: dict[str, list[str]] = {}
    for requirement in requirements:
        found = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", requirement)
        if found is not None:
            name = re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0]
            needed.setdefault(found.group(1), []).append(name)
    installed = []
    for extra, names in needed.items():
        try:
            for name in names:
                metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        # Never torch (a CPU build from another index), nor the documentation's browser tools:
        # jobs need neither.
        if extra not in ("torch", "docs-tools"):
            installed.append(extra)
    return sorted(installed)


def default_repository() -> Path | None:
    """Return the git checkout APEX runs from, or ``None`` (an installed APEX)."""
    top = Path(apex.__file__).resolve().parents[2]
    return top if (top / ".git").exists() else None


class CodeCopies:
    """Makes, finds and removes frozen copies.

    Parameters
    ----------
    directory
        Where the copies live.
    repository
        The git repository they are made from (``None``: none, so nothing can be frozen).
    extras
        The extras synced into each copy (``None``: :func:`installed_extras`).
    sync
        Syncs a copy's environment (default :func:`uv_sync`; tests pass a stub).
    """

    def __init__(
        self,
        directory: Path,
        repository: Path | None,
        extras: Sequence[str] | None = ("reservoir", "plots"),
        *,
        sync: Sync | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.repository = None if repository is None else Path(repository)
        self._extras = None if extras is None else tuple(extras)
        self.sync = sync or uv_sync

    @property
    def extras(self) -> tuple[str, ...]:
        """The extras synced into each copy (read from the environment only when needed)."""
        if self._extras is None:
            self._extras = tuple(installed_extras())
        return self._extras

    def _defined_extras(self, commit: str) -> set[str] | None:
        """Return the extras ``commit`` defines (``None``: unreadable).

        Its root ``pyproject.toml``'s and its workspace members': a copy syncs
        every package, so a member's extra is one the copy can have.
        """
        try:
            root = tomllib.loads(self._git("show", f"{commit}:pyproject.toml"))
        except (CodeError, tomllib.TOMLDecodeError):
            return None
        defined = set(root.get("project", {}).get("optional-dependencies", {}))
        workspace = root.get("tool", {}).get("uv", {}).get("workspace", {})
        members, excluded = workspace.get("members", []), workspace.get("exclude", [])
        try:
            files = self._git("ls-tree", "-r", "--name-only", commit).splitlines()
        except CodeError:
            return defined
        for name in files:
            folder = name.removesuffix("/pyproject.toml")
            if folder == name or not workspace_member(folder, members, excluded):
                continue
            try:
                project = tomllib.loads(self._git("show", f"{commit}:{name}"))
            except (CodeError, tomllib.TOMLDecodeError):
                continue
            defined |= set(project.get("project", {}).get("optional-dependencies", {}))
        return defined

    @classmethod
    def from_settings(cls, settings: JobsSection, root: Path) -> "CodeCopies":
        """Return the copies ``[jobs]`` of ``<root>/service.toml`` describes."""
        return cls(
            settings.code_dir or Path(root) / "code",
            settings.repository or default_repository(),
            settings.code_extras,
        )

    # ------------------------------------------------------------------ the repository

    def _git(self, *args: str, repository: Path | None = None) -> str:
        where = repository or self.repository
        if where is None:
            raise CodeError(
                "APEX does not run from a git checkout: set [jobs] repository in service.toml "
                "to freeze code"
            )
        done = subprocess.run(
            ["git", "-C", str(where), *args], capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            raise CodeError(f"git {' '.join(args)} failed in {where}: {done.stderr.strip()}")
        # Trailing space only: `git status --porcelain` lines start with a status column.
        return done.stdout.rstrip()

    def resolve(self, ref: str = "HEAD") -> str:
        """Return the 12-character commit ``ref`` names in the repository."""
        if ref.startswith("-"):
            raise CodeError(f"not a commit: {ref!r}")
        # The full name, cut: --short=12 may give more characters for an ambiguous prefix.
        return self._git("rev-parse", "--verify", f"{ref}^{{commit}}")[:12]

    def uncommitted(self) -> list[str]:
        """Return the repository's uncommitted changes to what a copy runs (a pin leaves them out).

        ``src/`` and the workspace members' (``packages/*/src``), and ``uv.lock``
        and the ``pyproject.toml`` files, which make its environment; not their tests.
        """
        status = self._git(
            "status", "--porcelain", "--untracked-files=all", "--", "src", "uv.lock",
            "pyproject.toml", ":(glob)packages/*/src/**", ":(glob)packages/*/pyproject.toml",
        )  # fmt: skip
        return [line[3:] for line in status.splitlines() if line.strip()]

    def check_pinnable(self, commit: str) -> None:
        """Refuse a commit whose workers do not speak this store's job protocol.

        Raises
        ------
        CodeError
            If the commit predates or speaks another protocol.
        """
        done = subprocess.run(
            ["git", "-C", str(self.repository), "show", f"{commit}:{PROTOCOL_FILE}"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        found = _PROTOCOL.search(done.stdout) if done.returncode == 0 else None
        if found is None:
            raise CodeError(
                f"commit {commit} predates frozen jobs: its worker cannot run a "
                "pinned job; pin a later commit"
            )
        if int(found.group(1)) != JOB_PROTOCOL:
            raise CodeError(
                f"commit {commit} speaks job protocol {found.group(1)}, this service "
                f"{JOB_PROTOCOL}: its worker would misread the job records; pin a commit "
                "with the same protocol"
            )

    # ------------------------------------------------------------------ copies

    def path(self, commit: str) -> Path:
        """Return where ``commit``'s copy is (or would be)."""
        if not _COMMIT.match(commit):
            raise CodeError(f"not a 12-character commit: {commit!r}")
        return self.directory / commit

    def _meta(self, commit: str) -> Path:
        return self.directory / f".{commit}.json"

    @contextmanager
    def _locked(self, commit: str) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / f".{commit}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def ensure(self, ref: str = "HEAD") -> CodePin:
        """Return the pin of ``ref``'s commit, making or completing its copy first.

        Two callers freezing the same commit build it once (a lock per copy). A copy synced
        with fewer extras (one ``freeze_apex.sh`` made) gets the missing ones added.

        Raises
        ------
        CodeError
            If the commit cannot be pinned, or its copy cannot be made.
        """
        commit = self.resolve(ref)
        self.check_pinnable(commit)
        path = self.path(commit)
        with self._locked(commit):
            if not path.exists():
                logger.info("freezing APEX at %s in %s", commit, path,
                            extra={"event": "code_freeze", "commit": commit})  # fmt: skip
                # A copy deleted by hand leaves its worktree registered: forget it first.
                self._git("worktree", "prune")
                self._git("worktree", "add", "--detach", str(path), commit)
            elif not (path / ".git").exists():
                raise CodeError(f"{path} exists but is not a worktree of APEX: remove it")
            meta = self._read_meta(commit)
            synced = tuple(meta.get("extras", ()))
            python = path / ".venv" / "bin" / "python"
            # Only the extras the commit itself defines: uv refuses others, and an older
            # commit lacks the extras added since (it must stay pinnable).
            defined = self._defined_extras(commit)
            wanted = [e for e in self.extras if defined is None or e in defined]
            if len(wanted) < len(self.extras):
                logger.info(
                    "commit %s does not define the extras %s: its copy goes without them",
                    commit, sorted(set(self.extras) - set(wanted)),
                    extra={"event": "code_extras_dropped", "commit": commit},
                )  # fmt: skip
            extras = sorted({*synced, *wanted})
            # A copy synced before the workspace (or by older code) lacks the
            # members: completed once, with every package.
            all_packages = bool(meta.get("all_packages"))
            if [e for e in wanted if e not in synced] or not python.exists() or not all_packages:
                # Completing a copy only adds packages (--inexact), so a job running from it
                # meanwhile keeps working.
                self.sync(path, extras, self.directory / f".{commit}.sync.log")
            # Every pin is a use: the clean-up counts its days from the last one, and the
            # record is written under the copy's lock, which the clean-up takes too.
            self._meta(commit).write_text(
                json.dumps(
                    {
                        "commit": commit,
                        "extras": extras,
                        "all_packages": True,
                        "created": meta.get("created") or _now(),
                        "last_used": _now(),
                    }
                )
            )
        return CodePin(commit, python)

    def _read_meta(self, commit: str) -> dict[str, Any]:
        try:
            data = json.loads(self._meta(commit).read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def copies(self) -> list[CodeCopy]:
        """Return the copies, oldest first."""
        found: list[CodeCopy] = []
        if not self.directory.is_dir():
            return found
        for path in sorted(self.directory.iterdir()):
            if not (path.is_dir() and _COMMIT.match(path.name) and (path / ".git").exists()):
                continue
            meta = self._read_meta(path.name)
            try:
                subject = self._git("log", "-1", "--format=%cs %s", repository=path)
            except CodeError:  # one broken copy never hides the others
                subject = "(unreadable worktree)"
            created = meta.get("created") or _mtime(path)
            last_used = meta.get("last_used") or meta.get("created")
            found.append(
                CodeCopy(path.name, path, tuple(meta.get("extras", ())), str(created), subject,
                         None if last_used is None else str(last_used))
            )  # fmt: skip
        return sorted(found, key=lambda c: c.created or "")

    def processes(self, commit: str) -> list[int]:
        """Return the live processes that run from ``commit``'s copy (not this one)."""
        return list(_processes_under(self.path(commit)))

    def in_use(self, commit: str, jobs: Iterable[dict[str, Any]]) -> list[str]:
        """Return what uses ``commit``'s copy: unfinished jobs pinning it, and processes."""
        users = [
            f"job {record['id']} ({record['state']})"
            for record in jobs
            if record.get("state") in ("queued", "running")
            and (record.get("code") or {}).get("commit") == commit
        ]
        users += [f"process {pid}" for pid in self.processes(commit)]
        return users

    def remove(
        self,
        commit: str,
        jobs: Callable[[], Iterable[dict[str, Any]]],
        *,
        unused_since: dt.datetime | None = None,
    ) -> None:
        """Remove ``commit``'s copy.

        ``jobs`` returns the job records; they are read under the copy's lock, which a pin
        (:meth:`ensure`) takes too, so a job pinned meanwhile is seen. With
        ``unused_since`` (the clean-up), a copy pinned after it is kept too: its job may not
        be inserted yet.

        Raises
        ------
        CodeError
            If it is in use, holds files that are neither tracked nor ignored (someone's
            work: removed by hand only), or there is no such copy.
        """
        path = self.path(commit)
        if not path.exists():
            raise CodeError(f"no frozen copy {path}")
        with self._locked(commit):
            used = self._read_meta(commit).get("last_used")
            try:
                recent = used is not None and _parse(str(used)) > (unused_since or _NEVER)
            except ValueError:
                recent = True
            if unused_since is not None and recent:
                raise CodeError(f"{path} was pinned at {used}, since the clean-up looked")
            users = self.in_use(commit, jobs())
            if users:
                raise CodeError(f"{path} is in use ({', '.join(users)}); not removed")
            # Without --force: git removes ignored files (the .venv), never untracked ones.
            self._git("worktree", "remove", str(path))
            for name in (f".{commit}.json", f".{commit}.sync.log"):
                (self.directory / name).unlink(missing_ok=True)
        logger.info("removed the frozen copy %s", path,
                    extra={"event": "code_removed", "commit": commit})  # fmt: skip

    def heads(self) -> frozenset[str]:
        """Return the commits the clean-up always keeps (up to three).

        ``master`` and ``origin/master`` here (what new jobs freeze, and what the hosts'
        cluster sync follows), and the repository's ``HEAD`` (a run from the checkout).
        """
        found = set()
        for ref in ("master", "origin/master", "HEAD"):
            try:
                found.add(self.resolve(ref))
            except CodeError:  # no such branch, or no repository
                continue
        return frozenset(found)

    def last_used(self, commit: str) -> str | None:
        """Return when ``commit``'s copy here was last pinned (its creation if never).

        Now, if its record is there but cannot be read: a clean-up keeps it.
        """
        meta = self._read_meta(commit)
        if not meta and self._meta(commit).exists():
            return _now()
        used = meta.get("last_used") or meta.get("created")
        return None if used is None else str(used)

    def collect(
        self,
        jobs: Callable[[], Sequence[dict[str, Any]]],
        keep_days: float,
        *,
        dry_run: bool = False,
    ) -> list[CodeCopy]:
        """Remove the copies nothing uses, pinned last more than ``keep_days`` ago.

        Only copies APEX has pinned (with its record) are collected: one made by hand is
        someone's, and goes by hand (``apex code remove``). The newest of ``master``
        (:meth:`heads`) is always kept. Returns those removed.
        """
        cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=keep_days)
        removed = []
        keep = self.heads()
        for copy in self.copies():
            if copy.commit in keep:
                continue
            try:
                stale = copy.last_used is not None and _parse(copy.last_used) <= cutoff
            except ValueError:
                logger.warning("frozen copy %s: unreadable record; kept", copy.path)
                continue
            if not stale or self.in_use(copy.commit, jobs()):
                continue
            if not dry_run:
                try:
                    self.remove(copy.commit, jobs, unused_since=cutoff)
                except CodeError as error:  # in use since the check: kept
                    logger.info("kept %s: %s", copy.path, error)
                    continue
            removed.append(copy)
        return removed


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


_NEVER = dt.datetime.min.replace(tzinfo=dt.UTC)


def _parse(time: str) -> dt.datetime:
    """Return a record's time (``ValueError`` if it is not one), in UTC."""
    parsed = dt.datetime.fromisoformat(time)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.UTC)


def _mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds")


def processes_under(path: Path) -> list[int]:
    """Return the pids of the processes running from ``path`` (``/proc``, never pgrep)."""
    return list(_processes_under(path))


def _processes_under(path: Path) -> Iterator[int]:
    """Yield the pids whose command line names ``path`` or whose virtual env is in it."""
    # As given and resolved: a pinned worker's interpreter is named as the record has it,
    # perhaps through a symlink.
    needles = {str(path).encode(), str(path.resolve()).encode()}

    def under(value: bytes) -> bool:
        return any(value == n or value.startswith(n + b"/") for n in needles)

    own = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(under(a) for a in arguments):
            yield int(entry.name)
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except OSError:  # another user's process
            continue
        for variable in environment:
            if variable.startswith(b"VIRTUAL_ENV=") and under(variable[12:]):
                yield int(entry.name)
                break


def run_commit(run_dir: Path) -> str | None:
    """Return the commit a run started on (``provenance.json``), or ``None``.

    ``None`` too when it started from an uncommitted tree: no commit holds its code.
    """
    try:
        saved = json.loads((Path(run_dir) / "provenance.json").read_text())
        code = saved["start"]["apex"]
        commit, dirty = code["commit"], code["dirty"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(commit, str) or dirty is not False:
        return None
    return commit[:12]


def worker_spawn(
    pin: dict[str, Any] | None, root: Path, job_id: str
) -> tuple[list[str], dict[str, str]]:
    """Return the command and environment that start a job's worker.

    A pinned job's worker runs in its copy's environment; any other in the live one, also
    when this process is itself a pinned worker (``APEX_LIVE_*``).

    Raises
    ------
    CodeError
        If the pinned copy's interpreter is gone.
    """
    env = dict(os.environ)
    live = {
        "python": env.get(LIVE_ENV["python"]) or sys.executable,
        "PATH": env.get(LIVE_ENV["PATH"], env.get("PATH", "")),
        "VIRTUAL_ENV": env.get(LIVE_ENV["VIRTUAL_ENV"], env.get("VIRTUAL_ENV", "")),
        "PYTHONPATH": env.get(LIVE_ENV["PYTHONPATH"], env.get("PYTHONPATH", "")),
    }
    if not pin:
        for name in ("PATH", "VIRTUAL_ENV", "PYTHONPATH"):
            if live[name]:
                env[name] = live[name]
            else:
                env.pop(name, None)
        return [live["python"], "-m", "apex.cli.worker", str(root), job_id], env
    python = Path(str(pin["python"]))
    if not python.exists():
        raise CodeError(
            f"the frozen copy of {pin.get('commit')} is gone ({python} missing): the job "
            "cannot run on its pinned code; submit it again"
        )
    venv = python.parent.parent
    for key, name in LIVE_ENV.items():
        env[name] = live[key]
    # The live environment's bin/: of its VIRTUAL_ENV, else of its interpreter.
    live_bin = str(
        Path(live["VIRTUAL_ENV"]) / "bin" if live["VIRTUAL_ENV"] else Path(live["python"]).parent
    )
    rest = [p for p in live["PATH"].split(os.pathsep) if p and p != live_bin]
    env["PATH"] = os.pathsep.join([str(venv / "bin"), *rest])
    env["VIRTUAL_ENV"] = str(venv)
    for name in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):
        env.pop(name, None)
    return [str(python), "-m", "apex.cli.worker", str(root), job_id], env
