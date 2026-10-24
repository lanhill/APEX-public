"""Web-UI tokens expire; the owner's API token does not."""

import datetime as dt
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.services.auth import OWNER_UI_FILE, api_token, issued_at, owner_ui_token, ui_token

UI = {"X-APEX-Client": "ui"}


def _bearer(token: str | None) -> dict[str, str]:
    assert token is not None
    return {"Authorization": f"Bearer {token}"}


def _age(token_file: Path, days: float) -> None:
    """Pretend the token was issued ``days`` ago."""
    when = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    Path(str(token_file) + ".issued").write_text(when.isoformat(timespec="seconds") + "\n")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "service"
    path.mkdir()
    return path


def test_a_guests_token_expires_after_token_days(root: Path) -> None:
    http = TestClient(create_app(root))
    guest = _bearer(ui_token(root, create=True, guest="alice"))
    me = http.get("/v1/whoami", headers=guest).json()
    expires = dt.datetime.fromisoformat(me["expires"])
    issued = issued_at(root / "ui_tokens" / "alice")
    assert expires - issued == dt.timedelta(days=7)  # [ui] token_days, default 7

    _age(root / "ui_tokens" / "alice", 8)
    refused = http.get("/v1/whoami", headers=guest)
    assert refused.status_code == 401
    problem = refused.json()
    assert problem["type"].endswith("/token-expired")
    assert problem["title"] == "Token expired"
    # A guest can't run the command: they are told to ask the owner (a review).
    assert "ask the service's owner" in problem["detail"]
    assert "apex service token" not in problem["detail"]

    fresh = _bearer(ui_token(root, rotate=True, guest="alice"))  # a new one, a new date
    assert http.get("/v1/whoami", headers=fresh).status_code == 200


def test_token_days_is_configurable(root: Path) -> None:
    (root / "service.toml").write_text("[ui]\ntoken_days = 1\n")
    http = TestClient(create_app(root))
    guest = _bearer(ui_token(root, create=True))
    _age(root / "ui_token", 1.5)
    assert http.get("/v1/whoami", headers=guest).json()["title"] == "Token expired"


def test_the_owner_signs_in_to_the_web_ui_with_an_expiring_token(root: Path) -> None:
    http = TestClient(create_app(root))
    api = _bearer(api_token(root))
    # The API token does not expire, and is for the CLI, MCP, scripts and the Client.
    assert http.get("/v1/whoami", headers=api).json()["expires"] is None
    # From the web UI it is refused, pointing at the owner's web-UI token.
    refused = http.get("/v1/whoami", headers={**api, **UI})
    assert refused.status_code == 401
    assert refused.json()["title"] == "Web-UI token required"
    assert "apex service token --ui --owner" in refused.json()["detail"]

    owner = {**_bearer(owner_ui_token(root, create=True)), **UI}
    me = http.get("/v1/whoami", headers=owner).json()
    assert me["role"] == "owner"
    assert me["expires"] is not None
    _age(root / OWNER_UI_FILE, 30)
    expired = http.get("/v1/whoami", headers=owner).json()
    assert expired["title"] == "Token expired"
    assert "apex service token --ui --owner --rotate" in expired["detail"]


def test_a_token_from_before_issue_dates_uses_its_files_time(root: Path) -> None:
    ui_token(root, create=True)
    Path(str(root / "ui_token") + ".issued").unlink()  # as a token made
    issued = issued_at(root / "ui_token")
    assert abs(dt.datetime.now(dt.UTC) - issued) < dt.timedelta(minutes=5)
