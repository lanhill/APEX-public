"""Frozen copies of APEX's code, on a small git repository and a stub sync."""

import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from apex.services.code import (
    LIVE_ENV,
    CodeCopies,
    CodeError,
    installed_extras,
    run_commit,
    worker_spawn,
    workspace_member,
)
from apex.storage.sqlite_store import JOB_PROTOCOL, SQLiteStore


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return done.stdout.strip()


def make_repository(path: Path, protocol: int | None = JOB_PROTOCOL) -> Path:
    """A git repository whose one commit speaks ``protocol`` (``None``: predates)."""
    store = path / "src" / "apex" / "storage" / "sqlite_store.py"
    store.parent.mkdir(parents=True)
    store.write_text("" if protocol is None else f'"""Store."""\n\nJOB_PROTOCOL = {protocol}\n')
    (path / ".gitignore").write_text(".venv/\n")  # as APEX's own
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "first")
    return path


class StubSync:
    """Writes the copy's interpreter (a wrapper of this one) and counts the syncs."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, tuple[str, ...]]] = []

    def __call__(self, copy: Path, extras: Sequence[str], log: Path) -> None:
        self.calls.append((copy, tuple(extras)))
        python = copy / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)


@pytest.fixture
def copies(tmp_path: Path) -> tuple[CodeCopies, StubSync]:
    sync = StubSync()
    repo = make_repository(tmp_path / "repo")
    return CodeCopies(tmp_path / "code", repo, ["reservoir"], sync=sync), sync


def test_a_commit_is_frozen_once_into_a_worktree_with_its_own_interpreter(
    copies: tuple[CodeCopies, StubSync],
) -> None:
    code, sync = copies
    pins = []
    threads = [threading.Thread(target=lambda: pins.append(code.ensure())) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({p.commit for p in pins}) == 1
    assert len(sync.calls) == 1
    pin = pins[0]
    path = code.path(pin.commit)
    assert pin.python == path / ".venv" / "bin" / "python"
    assert (path / "src" / "apex" / "storage" / "sqlite_store.py").exists()
    # Nothing of the copy's own besides the worktree is in it: a run's provenance stays clean.
    assert _git(path, "status", "--porcelain") == ""
    meta = json.loads((code.directory / f".{pin.commit}.json").read_text())
    assert meta["extras"] == ["reservoir"]


def test_a_copy_with_fewer_extras_gets_the_missing_ones_added(
    copies: tuple[CodeCopies, StubSync],
) -> None:
    code, sync = copies
    pin = code.ensure()
    more = CodeCopies(code.directory, code.repository, ["plots", "reservoir"], sync=sync)
    more.ensure()
    more.ensure()

    assert [extras for _, extras in sync.calls] == [("reservoir",), ("plots", "reservoir")]
    assert json.loads((code.directory / f".{pin.commit}.json").read_text())["extras"] == [
        "plots",
        "reservoir",
    ]


@pytest.mark.parametrize(
    ("protocol", "refusal"),
    [(None, "predates frozen jobs"), (JOB_PROTOCOL + 1, "speaks job protocol")],
)
def test_a_commit_of_another_job_protocol_cannot_be_pinned(
    tmp_path: Path, protocol: int | None, refusal: str
) -> None:
    sync = StubSync()
    code = CodeCopies(tmp_path / "code", make_repository(tmp_path / "repo", protocol), sync=sync)

    with pytest.raises(CodeError, match=refusal):
        code.ensure()
    assert not sync.calls


def test_a_ref_that_looks_like_an_option_is_refused(copies: tuple[CodeCopies, StubSync]) -> None:
    code, _ = copies
    with pytest.raises(CodeError, match="not a commit"):
        code.ensure("--output=/tmp/x")


def _none() -> list[dict[str, object]]:
    return []


def _age(code: CodeCopies, commit: str, when: str = "2026-01-01T00:00:00+00:00") -> None:
    """Make ``commit``'s copy look last pinned at ``when``."""
    meta = code.directory / f".{commit}.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "last_used": when}))


def test_a_copy_in_use_by_a_job_or_a_process_is_kept(copies: tuple[CodeCopies, StubSync]) -> None:
    code, _ = copies
    commit = code.ensure().commit
    path = code.path(commit)
    queued = {"id": "j1", "state": "queued", "code": {"commit": commit}}
    finished = {"id": "j0", "state": "complete", "code": {"commit": commit}}
    assert code.in_use(commit, [queued]) == ["job j1 (queued)"]
    assert code.in_use(commit, [finished]) == []
    with pytest.raises(CodeError, match="in use"):
        code.remove(commit, lambda: [queued])

    # A process whose command line names the copy.
    python = path / ".venv" / "bin" / "python"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(python)])
    try:
        assert code.in_use(commit, []) == [f"process {process.pid}"]
        _age(code, commit)
        assert code.collect(_none, keep_days=0) == []
    finally:
        process.kill()
        process.wait()

    code.remove(commit, lambda: [finished])
    assert not path.exists()
    assert not (code.directory / f".{commit}.json").exists()


def test_gc_removes_only_copies_unused_since_their_last_pin(
    copies: tuple[CodeCopies, StubSync],
) -> None:
    code, _ = copies
    commit = code.ensure().commit
    assert code.repository is not None
    _git(code.repository, "commit", "-q", "--allow-empty", "-m", "next")  # master moves on
    assert code.collect(_none, keep_days=1) == []  # pinned now
    _age(code, commit)
    code.ensure(commit)  # pinned again: a use, whatever its age
    assert code.collect(_none, keep_days=1) == []
    _age(code, commit)
    assert [c.commit for c in code.collect(_none, keep_days=1, dry_run=True)] == [commit]
    assert code.path(commit).exists()
    # A job pinned between the list and the removal is seen: the jobs are read again.
    pinned = [{"id": "j", "state": "queued", "code": {"commit": commit}}]
    reads = iter([[], pinned])
    assert code.collect(lambda: next(reads), keep_days=1) == []
    assert code.path(commit).exists()

    assert [c.commit for c in code.collect(_none, keep_days=1)] == [commit]
    assert not code.path(commit).exists()


def test_gc_leaves_copies_made_by_hand_and_files_someone_left(
    copies: tuple[CodeCopies, StubSync],
) -> None:
    code, _ = copies
    commit = code.ensure().commit
    (code.directory / f".{commit}.json").unlink()  # as freeze_apex.sh made them
    assert code.collect(_none, keep_days=0) == []

    pin = code.ensure()
    _age(code, pin.commit)
    (code.path(commit) / "notes.txt").write_text("someone's work\n")  # untracked
    assert code.collect(_none, keep_days=1) == []  # git refuses: kept
    assert (code.path(commit) / "notes.txt").exists()
    with pytest.raises(CodeError, match="worktree remove"):
        code.remove(commit, _none)


def test_a_copy_deleted_by_hand_is_made_again(copies: tuple[CodeCopies, StubSync]) -> None:
    code, sync = copies
    commit = code.ensure().commit
    shutil.rmtree(code.path(commit))  # its worktree stays registered in the repository

    assert code.ensure().commit == commit
    assert (code.path(commit) / ".git").exists()
    assert len(sync.calls) == 2


def test_a_run_s_own_commit_is_read_from_its_provenance(tmp_path: Path) -> None:
    def provenance(commit: object, dirty: object) -> Path:
        run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        run.mkdir()
        record = {"start": {"apex": {"commit": commit, "dirty": dirty}}, "resumes": []}
        (run / "provenance.json").write_text(json.dumps(record))
        return run

    assert run_commit(provenance("0123456789abcdef0123", False)) == "0123456789ab"
    assert run_commit(provenance("0123456789abcdef0123", True)) is None  # no commit holds it
    assert run_commit(provenance(None, None)) is None  # not run from a checkout
    assert run_commit(tmp_path / "missing") is None


def test_a_worker_starts_in_its_pinned_copy_s_environment_or_fails_when_it_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in LIVE_ENV.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VIRTUAL_ENV", "/live/.venv")
    monkeypatch.setenv("PATH", "/live/.venv/bin:/usr/bin")
    monkeypatch.setenv("PYTHONPATH", "/checkout/src")
    live, env = worker_spawn(None, tmp_path, "j")
    assert live == [sys.executable, "-m", "apex.cli.worker", str(tmp_path), "j"]
    assert env["VIRTUAL_ENV"] == "/live/.venv"

    python = tmp_path / "copy" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    pin = {"commit": "0123456789ab", "python": str(python)}
    command, env = worker_spawn(pin, tmp_path, "j")
    assert command[0] == str(python)
    assert env["VIRTUAL_ENV"] == str(python.parent.parent)
    assert env["PATH"] == f"{python.parent}:/usr/bin"  # the live venv's bin/ left out
    assert "PYTHONPATH" not in env
    assert env["APEX_LIVE_PYTHON"] == sys.executable

    # A job a pinned worker starts in turn runs live code when it is not pinned (embedded
    # mode): the live environment came along.
    monkeypatch.setattr(os, "environ", env)
    again, back = worker_spawn(None, tmp_path, "k")
    assert again[0] == sys.executable
    assert back["VIRTUAL_ENV"] == "/live/.venv"
    assert back["PATH"] == "/live/.venv/bin:/usr/bin"
    assert back["PYTHONPATH"] == "/checkout/src"

    python.unlink()
    with pytest.raises(CodeError, match="is gone"):
        worker_spawn(pin, tmp_path, "j")


def test_copies_get_the_extras_installed_here_by_default(tmp_path: Path) -> None:
    extras = installed_extras()
    assert "torch" not in extras
    assert CodeCopies(tmp_path, None, None).extras == tuple(extras)
    # Each one's packages are installed here (CI syncs none, the workstation all but torch).
    from importlib import metadata

    requires = metadata.distribution("apex").requires or []
    for extra in extras:
        assert any(f'extra == "{extra}"' in r or f"extra == '{extra}'" in r for r in requires)


def test_the_store_keeps_the_highest_job_protocol_of_its_users(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path)
    assert store.job_protocol() == JOB_PROTOCOL
    with store.locked():
        store._execute(
            "UPDATE meta SET value = ? WHERE key = 'job_protocol'", (str(JOB_PROTOCOL + 1),)
        )
    store.close()

    # A user of an older protocol does not lower it.
    assert SQLiteStore(tmp_path).job_protocol() == JOB_PROTOCOL + 1
    assert os.path.exists(tmp_path / "apex.db")


def test_a_copy_gets_only_the_extras_its_commit_defines(tmp_path: Path) -> None:
    repo = make_repository(tmp_path / "repo")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "apex"\n[project.optional-dependencies]\nreservoir = ["mako"]\n'
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "extras")
    sync = StubSync()
    # "newer" is an extra the live checkout gained since: the older commit stays pinnable.
    code = CodeCopies(tmp_path / "code", repo, ["newer", "reservoir"], sync=sync)

    code.ensure()

    assert [extras for _, extras in sync.calls] == [("reservoir",)]


def test_gc_keeps_a_copy_pinned_after_it_looked(copies: tuple[CodeCopies, StubSync]) -> None:
    code, _ = copies
    commit = code.ensure().commit
    _age(code, commit)

    def jobs() -> list[dict[str, object]]:
        code.ensure()  # a submission pins it now; its job is not inserted yet
        return []

    assert code.collect(jobs, keep_days=1) == []
    assert code.path(commit).exists()


def test_gc_always_keeps_the_newest_of_master(copies: tuple[CodeCopies, StubSync]) -> None:
    """The user (2026-10-16): old copies go automatically, but never the newest of master
    (what new jobs freeze, and the hosts' cluster sync follows), however long unused."""
    code, _ = copies
    assert code.repository is not None
    old = code.ensure().commit
    _git(code.repository, "commit", "-q", "--allow-empty", "-m", "next")
    new = code.ensure().commit
    assert new != old
    assert new in code.heads()
    _age(code, old)
    _age(code, new)
    assert [c.commit for c in code.collect(_none, keep_days=1)] == [old]
    assert code.path(new).exists()


def test_a_members_extras_count_and_old_copies_get_every_package(tmp_path: Path) -> None:
    """The coder's review: an extra only a workspace member defines is
    one the commit defines (a copy syncs every package), an excluded folder's is not, and a
    copy synced before the workspace (no record of every package) is completed once."""
    repo = make_repository(tmp_path / "repo")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "apex"\n[project.optional-dependencies]\nreservoir = ["mako"]\n'
        '[tool.uv.workspace]\nmembers = ["packages/*"]\nexclude = ["packages/old"]\n'
    )
    for name, extra in (("apex-member", "hosts"), ("old", "gone")):
        (repo / "packages" / name).mkdir(parents=True)
        (repo / "packages" / name / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\n[project.optional-dependencies]\n{extra} = ["x"]\n'
        )
    # A pyproject.toml nested in a member (a test fixture) is not a member: uv's globs
    # never cross a "/".
    (repo / "packages" / "apex-member" / "tests" / "fixture").mkdir(parents=True)
    (repo / "packages" / "apex-member" / "tests" / "fixture" / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\n[project.optional-dependencies]\nnested = ["x"]\n'
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "workspace")
    sync = StubSync()
    code = CodeCopies(tmp_path / "code", repo, ["hosts", "gone", "nested", "reservoir"],
                      sync=sync)  # fmt: skip

    pin = code.ensure()
    assert [extras for _, extras in sync.calls] == [("hosts", "reservoir")]
    code.ensure()
    assert len(sync.calls) == 1  # complete: nothing to add

    meta = code.directory / f".{pin.commit}.json"
    record = json.loads(meta.read_text())
    del record["all_packages"]  # as a copy synced before the workspace
    meta.write_text(json.dumps(record))
    code.ensure()
    assert len(sync.calls) == 2  # completed with every package, once
    code.ensure()
    assert len(sync.calls) == 2


def test_a_members_installed_extras_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coder's review: installed_extras reads every workspace package's extras."""
    from importlib import metadata

    from apex.services import code

    class Dist:
        def __init__(self, name: str, requires: list[str]) -> None:
            self.metadata = {"Name": name}
            self.requires = requires

    dists = [Dist("apex", ['numpy; extra == "plots"']),
             Dist("apex_member", ['pytest; extra == "hosts"']),
             Dist("other", ['numpy; extra == "elsewhere"'])]  # fmt: skip
    real = metadata.distribution("numpy")

    def distribution(name: str) -> metadata.Distribution:
        if name in ("numpy", "pytest"):
            return real
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distributions", lambda: dists)
    monkeypatch.setattr(metadata, "distribution", distribution)
    assert code.installed_extras() == ["hosts", "plots"]


@pytest.mark.parametrize(
    ("path", "members", "excluded", "member"),
    [
        ("packages/a", ["packages/*"], [], True),
        ("packages/a/b", ["packages/*"], [], False),  # a member's * stays in one folder
        ("packages/a", ["packages/*/"], [], True),
        ("packages/a", ["./packages/*"], [], True),
        ("packages/a/b", ["packages/**"], [], True),
        ("packages", ["packages/**"], [], False),  # a trailing ** needs a folder
        ("packages/a", ["packages/a/**"], [], False),
        ("packages/b", ["packages/**/b"], [], True),  # elsewhere ** may span none
        ("b", ["**/b"], [], True),
        ("", ["*"], [], False),
        ("packages/a/b", ["packages/**/b"], [], True),
        ("packages/A", ["packages/a"], [], False),  # case-sensitive
        ("packages/old", ["packages/*"], ["packages/old"], False),
        ("packages/a/b", ["packages/**"], ["packages/*"], False),  # an exclude's * crosses /
        ("packages/a", ["packages/*"], ["./packages/a/"], False),
    ],
)
def test_workspace_members_follow_uvs_globs(
    path: str, members: list[str], excluded: list[str], member: bool
) -> None:
    """Uv 0.12's member and exclude globs, as measured on a workspace."""
    assert workspace_member(path, members, excluded) is member


def test_uncommitted_changes_to_a_workspace_member_count(
    copies: tuple[CodeCopies, StubSync],
) -> None:
    """The ssh executor lives in ``packages/``; an edit there is code a
    copy (and the hosts') would not run."""
    code, _ = copies
    assert code.repository is not None
    member = code.repository / "packages" / "apex-member" / "src" / "apex_member" / "x.py"
    member.parent.mkdir(parents=True)
    member.write_text("edited = True\n")
    test = code.repository / "packages" / "apex-member" / "tests_member" / "test_x.py"
    test.parent.mkdir(parents=True)
    test.write_text("edited = True\n")  # a member's tests are not code a copy runs
    assert code.uncommitted() == ["packages/apex-member/src/apex_member/x.py"]
