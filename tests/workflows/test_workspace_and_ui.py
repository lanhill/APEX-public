"""The UI's workspace (editor files), TOML validation and the served UI."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import CSP, create_app
from apex.core.errors import ConfigError
from apex.services.settings import load_settings
from apex.services.workspace import MAX_BYTES, Workspace, WorkspaceError, WriteConflict, sha256

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _root(tmp_path: Path, settings: str = "") -> Path:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(settings)
    return root


def test_the_default_workspace_is_the_roots_configs_folder(tmp_path: Path) -> None:
    root = _root(tmp_path)
    space = Workspace(root, load_settings(root))

    assert space.roots == [(root / "configs").resolve()]
    target = root / "configs" / "a" / "run.toml"
    saved = space.write(target, 'x = "1"\n', None)
    assert target.read_text() == 'x = "1"\n'
    assert space.read(target).sha256 == saved.sha256 == sha256(b'x = "1"\n')
    assert saved.path == "a/run.toml"  # relative to the one root
    names = [(e.path, e.kind) for e in space.list()]
    assert names == [("a", "folder")]
    assert [e.path for e in space.list("a")] == ["a/run.toml"]
    assert space.read("a/run.toml").text == 'x = "1"\n'  # relative paths are accepted
    assert space.describe() == [{"name": "", "path": str((root / "configs").resolve())}]


def test_several_roots_are_named_by_their_folders(tmp_path: Path) -> None:
    first, second, third = tmp_path / "cases", tmp_path / "x" / "cases", tmp_path / "studies"
    for folder in (first, second, third):
        folder.mkdir(parents=True)
    roots = ", ".join(f'"{f}"' for f in (first, second, third))
    root = _root(tmp_path, f"[workspace]\nroots = [{roots}]\n")
    space = Workspace(root, load_settings(root))

    assert [e.path for e in space.list()] == ["cases", "cases-2", "studies"]
    saved = space.write("cases-2/a/run.toml", "a = 1\n", None)
    assert (second / "a" / "run.toml").read_text() == "a = 1\n"
    assert saved.path == "cases-2/a/run.toml"
    assert space.relative(third / "s.toml") == "studies/s.toml"
    for bad in ("elsewhere/run.toml", "cases/../x.toml", "run.toml"):
        with pytest.raises(WorkspaceError):
            space.write(bad, "a = 1\n", None)


def test_writes_are_confined_to_toml_files_in_the_workspace(tmp_path: Path) -> None:
    root = _root(tmp_path)
    space = Workspace(root, load_settings(root))
    configs = root / "configs"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (configs / "out").symlink_to(outside)
    (outside / "x.py").write_text("print(1)\n")
    (configs / "link.toml").symlink_to(outside / "x.py")

    for path in (
        tmp_path / "evil.toml",  # outside the roots
        configs / ".." / "evil.toml",  # .. out of the roots
        configs / "out" / "evil.toml",  # a symlinked folder out of the roots
        configs / "deck.mako",  # not TOML
        configs / "link.toml",  # a .toml name pointing at another file
        Path("out/evil.toml"),  # relative, through the symlinked folder
        Path("../evil.toml"),
    ):
        with pytest.raises(WorkspaceError):
            space.write(path, "a = 1\n", None)
    assert not (tmp_path / "evil.toml").exists()
    assert not (outside / "evil.toml").exists()
    with pytest.raises(WorkspaceError, match="larger"):
        space.write(configs / "big.toml", "#" * (MAX_BYTES + 1), None)
    with pytest.raises(WorkspaceError):
        space.read(configs / "link.toml")


def test_a_write_is_refused_if_the_file_changed_since_it_was_read(tmp_path: Path) -> None:
    root = _root(tmp_path)
    space = Workspace(root, load_settings(root))
    target = root / "configs" / "run.toml"
    first = space.write(target, "a = 1\n", None)

    with pytest.raises(WriteConflict):  # a new file was meant, but it exists
        space.write(target, "a = 2\n", None)
    target.write_text("a = 3\n")  # changed by someone else
    with pytest.raises(WriteConflict) as conflict:
        space.write(target, "a = 2\n", first.sha256)
    assert conflict.value.current == sha256(b"a = 3\n")
    assert target.read_text() == "a = 3\n"
    space.write(target, "a = 2\n", sha256(b"a = 3\n"))
    assert target.read_text() == "a = 2\n"


def test_under_a_policy_the_workspace_must_be_set_and_readable(tmp_path: Path) -> None:
    cases, code, out = (tmp_path / n for n in ("cases", "code", "out"))
    for folder in (cases, code, out):
        folder.mkdir()
    policy = (
        f'[policy]\ninput_roots = ["{cases}"]\ncode_roots = ["{cases}/*/model"]\n'
        f'output_roots = ["{out}"]\n'
    )
    root = _root(tmp_path, policy)
    off = Workspace(root, load_settings(root))
    with pytest.raises(WorkspaceError, match="editor is off"):
        off.list()

    (root / "service.toml").write_text(policy + f'[workspace]\nroots = ["{code}"]\n')
    with pytest.raises(ConfigError, match="outside \\[policy\\] input_roots"):
        load_settings(root)
    (root / "service.toml").write_text(policy + f'[workspace]\nroots = ["{cases}/a/model"]\n')
    with pytest.raises(ConfigError, match="code_roots"):
        load_settings(root)

    (root / "service.toml").write_text(policy + f'[workspace]\nroots = ["{cases}"]\n')
    space = Workspace(root, load_settings(root))
    space.write(cases / "a" / "run.toml", "a = 1\n", None)
    with pytest.raises(WorkspaceError, match="code_roots"):
        space.write(cases / "a" / "model" / "run.toml", "a = 1\n", None)


def test_the_workspace_over_http(tmp_path: Path) -> None:
    root = _root(tmp_path)
    http = TestClient(create_app(root, token=TOKEN))
    configs = root / "configs"
    target = "run.toml"
    (configs / "sub").mkdir(parents=True)

    assert http.get("/v1/files").status_code == 401
    listed = http.get("/v1/files", headers=AUTH).json()
    assert [e["path"] for e in listed] == ["sub"]
    roots = http.get("/v1/workspace", headers=AUTH).json()["roots"]
    assert roots == [{"name": "", "path": str(configs.resolve())}]
    created = http.put(
        "/v1/files/content", headers=AUTH, json={"path": target, "text": "a = 1\n", "sha256": None}
    )
    assert created.status_code == 200
    read = http.get("/v1/files/content", headers=AUTH, params={"path": target}).json()
    assert read["path"] == "run.toml"
    assert read["text"] == "a = 1\n"
    absolute = {"path": str(configs / "run.toml")}  # absolute paths under a root also work
    assert http.get("/v1/files/content", headers=AUTH, params=absolute).json()["path"] == target
    assert read["sha256"] == created.json()["sha256"]
    stale = http.put(
        "/v1/files/content", headers=AUTH, json={"path": target, "text": "a = 2\n", "sha256": None}
    )
    assert stale.status_code == 409
    assert stale.json()["sha256"] == read["sha256"]
    outside = http.put(
        "/v1/files/content",
        headers=AUTH,
        json={"path": str(tmp_path / "x.toml"), "text": "", "sha256": None},
    )
    assert outside.status_code == 403
    assert http.get(
        "/v1/files/content", headers=AUTH, params={"path": target + "x"}
    ).status_code in (
        403,
        404,
    )
    missing = http.get("/v1/files/content", headers=AUTH, params={"path": "no.toml"})
    assert missing.status_code == 404


def test_unsaved_toml_is_validated(tmp_path: Path, linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20)
    root = _root(tmp_path)
    http = TestClient(create_app(root, token=TOKEN))
    text = case.config.read_text()

    good = http.post(
        "/v1/configs:validate",
        headers=AUTH,
        json={"config_text": text, "base_dir": str(case.directory)},
    ).json()
    assert good["valid"]
    assert good["plan"]["members"] == 20
    broken = http.post(
        "/v1/configs:validate",
        headers=AUTH,
        json={"config_text": "[run\n", "base_dir": str(case.directory)},
    ).json()
    assert not broken["valid"]
    assert broken["errors"][0].startswith("TOML")
    no_base = http.post("/v1/configs:validate", headers=AUTH, json={"config_text": text})
    assert no_base.status_code == 422


def test_the_built_ui_is_served_with_its_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>APEX</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("TOP-SECRET-CONTENT")
    monkeypatch.setenv("APEX_UI_DIST", str(dist))
    http = TestClient(create_app(_root(tmp_path), token=TOKEN), follow_redirects=False)

    assert http.get("/").headers["location"] == "/ui/"
    page = http.get("/ui/")
    assert page.status_code == 200
    assert "APEX" in page.text
    assert page.headers["content-security-policy"] == CSP
    assert http.get("/ui/assets/app.js").text == "console.log(1)"
    assert http.get("/ui/jobs").text == page.text  # every page is index.html
    assert http.get("/ui/assets/none.js").status_code == 404
    escape = http.get("/ui/..%2Fsecret.txt")
    assert "TOP-SECRET-CONTENT" not in escape.text


def test_without_a_build_the_ui_says_how_to_build_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APEX_UI_DIST", os.fspath(tmp_path / "none"))
    http = TestClient(create_app(_root(tmp_path), token=TOKEN))

    response = http.get("/ui/")
    assert response.status_code == 404
    assert "npm --prefix ui" in response.json()["detail"]


def test_the_service_says_at_start_whether_it_serves_the_ui(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A service answered 404 at /ui/ with nothing in its log: its service.toml had no
    [ui] dist_dir (2026-10-01)."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "service.toml").write_text(f'[ui]\ndist_dir = "{tmp_path / "missing"}"\n')
    with caplog.at_level("WARNING", logger="apex.service"):
        create_app(root)
    assert any("no web UI build" in r.message and "dist_dir" in r.message for r in caplog.records)
    built = tmp_path / "dist"
    built.mkdir()
    (built / "index.html").write_text("<html></html>")
    (root / "service.toml").write_text(f'[ui]\ndist_dir = "{built}"\n')
    caplog.clear()
    with caplog.at_level("INFO", logger="apex.service"):
        create_app(root)
    assert any("the web UI is served from" in r.message for r in caplog.records)
