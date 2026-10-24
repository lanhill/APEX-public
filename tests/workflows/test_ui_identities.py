"""Kinds of web-UI user other packages add (``apex.ui_identities``): core's plumbing.

A stand-in kind, contributed by a fake entry point: its token signs a user in as a UI user;
its work is annotated ``<kind>:<name>``, kept in its own folder, hidden from the owner's web
UI and from guests, and pruned by its limits; a broken or clashing contribution is skipped.
"""

import datetime as dt
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.services import identities
from apex.services.access import Access, ActionRules, FolderRules, NameRules, UIAccess
from apex.services.api import ApexService
from apex.services.auth import owner_ui_token, ui_token
from apex.services.guest_retention import prune_user
from apex.services.identities import UiIdentity, identity_of, ui_identities
from apex.testing import CaseFactory

NAME = "one"


class _Everything(Access):
    """Every plugin, as an added kind's own list may allow (the stand-in's)."""

    def plugin(self, registry: str, name: str) -> bool:
        """Every plugin."""
        return True


def _access(records: Iterable[Mapping[str, Any]], name: str, hidden: Sequence[str]) -> Access:
    rules = UIAccess(
        workspace=FolderRules(folders=["*"]), agents=NameRules(names=["*"]),
        actions=ActionRules(submit=True, stop=True, resume=True, delete=True),
    )  # fmt: skip
    return _Everything(rules, records, name, owner=f"tester:{name}")


TESTERS = UiIdentity(kind="tester", folder="testers", tokens="ui_tester_tokens", access=_access)


class _Point:
    def __init__(self, name: str, target: Any) -> None:  # noqa: ANN401 (an entry point's)
        self.name, self._target = name, target

    def load(self) -> Any:  # noqa: ANN401 (an entry point's object)
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


@pytest.fixture
def points(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_Point]]:
    declared: list[_Point] = []

    def found(group: str) -> list[_Point]:
        return declared if group == identities.IDENTITIES_GROUP else []

    monkeypatch.setattr(identities, "entry_points", found)
    ui_identities.cache_clear()
    yield declared
    ui_identities.cache_clear()


def _headers(token: str | None) -> dict[str, str]:
    assert token is not None
    return {"Authorization": f"Bearer {token}", "X-APEX-Client": "ui"}


def test_an_added_kinds_work_is_its_own_hidden_from_others_and_pruned(
    tmp_path: Path, linear_case: CaseFactory, points: list[_Point]
) -> None:
    points.append(_Point("tester", lambda: TESTERS))
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\n')
    case = linear_case(ensemble_size=20)
    http = TestClient(create_app(root))
    user = _headers(TESTERS.token(root, NAME, create=True))
    owner_ui = _headers(owner_ui_token(root, create=True))
    guest = _headers(ui_token(root, create=True))

    job = http.post("/v1/runs", headers=user,
                    json={"config_file": str(case.config), "client": "ui"}).json()  # fmt: skip
    service = ApexService(root)
    assert service.jobs.record(job["id"])["annotations"]["submitted_by"] == f"tester:{NAME}"
    assert http.get(f"/v1/jobs/{job['id']}", headers=user).status_code == 200
    assert http.get(f"/v1/jobs/{job['id']}", headers=owner_ui).status_code == 404
    assert http.get(f"/v1/jobs/{job['id']}", headers=guest).status_code == 404
    folder = service.workspace().ui_identity_folder(TESTERS.folder, NAME)
    assert folder is not None
    assert folder.parts[-3:] == ("UI", "testers", NAME)

    assert service.wait(job["id"], timeout_s=120, poll_s=0.1).state == "complete"
    now = dt.datetime.now(dt.UTC)
    assert prune_user(service, TESTERS.submitter(NAME), 7, 500, now=now) == []
    gone = prune_user(service, TESTERS.submitter(NAME), 7, 500, now=now + dt.timedelta(days=8))
    assert [u.guest for u in gone] == [f"tester:{NAME}"]


def test_a_broken_or_clashing_kind_is_skipped(
    points: list[_Point], caplog: pytest.LogCaptureFixture
) -> None:
    """A kind that fails to load, is not one, takes core's names or repeats another's kind
    or folder is logged and skipped; the others load."""
    points += [
        _Point("tester", lambda: TESTERS),
        _Point("broken", RuntimeError("cannot")),
        _Point("wrong", lambda: "tester"),
        _Point("guest", lambda: UiIdentity("guest", "guests2", "t2", _access)),
        _Point("again", lambda: UiIdentity("tester", "other", "t3", _access)),
        _Point("folder", lambda: UiIdentity("other", "testers", "t4", _access)),
        # Core's folders and tokens, another kind's tokens, names that are not one segment.
        _Point("guests", lambda: UiIdentity("x1", "guests", "t5", _access)),
        _Point("owner-folder", lambda: UiIdentity("x2", "owner", "t6", _access)),
        _Point("guest-tokens", lambda: UiIdentity("x3", "f3", "ui_tokens", _access)),
        _Point("shared-tokens", lambda: UiIdentity("x4", "f4", "ui_tester_tokens", _access)),
        _Point("up", lambda: UiIdentity("x5", "..", "t7", _access)),
        _Point("nested", lambda: UiIdentity("x6", "a/b", "t8", _access)),
        _Point("colon", lambda: UiIdentity("a:b", "f9", "t9", _access)),
        _Point("empty", lambda: UiIdentity("x7", "f10", "", _access)),
    ]

    assert ui_identities() == (TESTERS,)
    for name in ("broken", "wrong", "guest", "again", "folder", "guests", "owner-folder",
                 "guest-tokens", "shared-tokens", "up", "nested", "colon", "empty"):  # fmt: skip
        assert f"web-UI identity {name}" in caplog.text
    assert identity_of("tester:x") == (TESTERS, "x")
    assert identity_of("guest:x") is None
    assert identity_of("tester") is None
    for odd in ("tester:", "tester:a:b", "tester:../x", "tester:a/b"):
        assert identity_of(odd) is None, odd


def test_pruning_takes_only_a_web_ui_users_name(tmp_path: Path, points: list[_Point]) -> None:
    """``prune_user`` refuses what is no guest's nor added kind's user (the review)."""
    root = tmp_path / "service"
    root.mkdir()
    service = ApexService(root)
    for user in ("owner", "guest:", "tester:x", "ui"):
        with pytest.raises(ValueError, match="no web-UI user"):
            prune_user(service, user, 7, 500)
