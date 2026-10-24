"""REST routes installed packages add."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apex.api import extensions
from apex.api.app import Who, create_app
from apex.api.extensions import RouteContext
from apex.services.auth import api_token


class _EntryPoint:
    def __init__(self, name: str, load: object) -> None:
        self.name = self.value = name
        self._load = load

    def load(self) -> object:
        if isinstance(self._load, Exception):
            raise self._load
        return self._load


def _register(app: FastAPI, context: RouteContext) -> None:
    @app.get("/v1/extra/owner")
    def owner_only(who: Who, fail: bool = False) -> dict[str, Any]:
        context.owner(who)
        if fail:
            raise context.problem(409, "Not now", "the extra route refuses")
        return {"root": str(context.service.root)}


def test_a_packages_routes_are_served_owner_only_and_a_broken_one_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [_EntryPoint("broken", ImportError("no module")), _EntryPoint("extra", _register)]
    monkeypatch.setattr(extensions, "entry_points", lambda group: entries)
    root = tmp_path / "root"
    client = TestClient(create_app(root))
    owner = {"Authorization": f"Bearer {api_token(root)}"}

    assert client.get("/v1/extra/owner", headers=owner).json() == {"root": str(root)}
    refused = client.get("/v1/extra/owner", headers=owner, params={"fail": True})
    assert (refused.status_code, refused.json()["title"]) == (409, "Not now")
    assert client.get("/v1/extra/owner").status_code == 401


def test_a_packages_web_ui_page_is_served_and_listed_to_the_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.api.extensions import UiPage
    from apex.services.auth import ui_token

    built = tmp_path / "built"
    (built / "assets").mkdir(parents=True)
    (built / "index.html").write_text("<p>extra page</p>")
    (built / "assets" / "page.js").write_text("console.log(1)")
    pages = [
        _EntryPoint("extra", lambda: UiPage("extra", "Extra", built)),
        _EntryPoint("shared", lambda: UiPage("shared", "Shared", built, owner_only=False)),
        _EntryPoint("bad", lambda: UiPage("../x", "Bad", built)),
        _EntryPoint("broken", ImportError("no module")),
    ]
    wanted = extensions.UI_PAGES_GROUP
    monkeypatch.setattr(extensions, "entry_points", lambda group: pages if group == wanted else [])
    root = tmp_path / "root"
    client = TestClient(create_app(root))
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}",
             "X-APEX-Client": "ui"}  # fmt: skip

    listed = client.get("/v1/ui/pages", headers=owner).json()["pages"]
    expected = [("extra", "/ui/pages/extra/"), ("shared", "/ui/pages/shared/")]
    assert [(p["name"], p["url"]) for p in listed] == expected
    assert [p["name"] for p in client.get("/v1/ui/pages", headers=guest).json()["pages"]] == [
        "shared"
    ]
    assert client.get("/ui/pages/extra/").text == "<p>extra page</p>"
    assert client.get("/ui/pages/extra/assets/page.js").text == "console.log(1)"
    (tmp_path / "secret.txt").write_text("not for the web")
    # Encoded, so that the client does not resolve it before the page's handler sees it.
    for sneaky in (
        "/ui/pages/extra/..%2F..%2Fsecret.txt",
        "/ui/pages/extra/%2e%2e/%2e%2e/secret.txt",
    ):
        answer = client.get(sneaky)
        assert answer.status_code == 404 or "not for the web" not in answer.text
    moved = client.get("/ui/pages/extra", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (307, "/ui/pages/extra/")


def test_unbuilt_pages_and_a_demonstrations_are_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page without its build is not offered; ``[ui] contributed_pages = false`` lists none."""
    from apex.api.extensions import UiPage

    built = tmp_path / "built"
    built.mkdir()
    (built / "index.html").write_text("<p>page</p>")
    pages = [
        _EntryPoint("built", lambda: UiPage("built", "Built", built)),
        _EntryPoint("unbuilt", lambda: UiPage("unbuilt", "Unbuilt", tmp_path / "nowhere")),
        _EntryPoint("off", lambda: UiPage("off", "Off", built, listed=lambda service: False)),
    ]
    wanted = extensions.UI_PAGES_GROUP
    monkeypatch.setattr(extensions, "entry_points", lambda group: pages if group == wanted else [])
    root = tmp_path / "root"
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    listed = TestClient(create_app(root)).get("/v1/ui/pages", headers=owner).json()["pages"]
    assert [p["name"] for p in listed] == ["built"]
    (root / "service.toml").write_text("[ui]\ncontributed_pages = false\n")
    listed = TestClient(create_app(root)).get("/v1/ui/pages", headers=owner).json()["pages"]
    assert listed == []
