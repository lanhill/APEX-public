"""The case-converter and the geomodeler over REST: drafts in the
caller's web-UI folder, private skills hidden from guests, the truth run the owner's only."""

import tomllib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.core.errors import ConfigError
from apex.services.auth import owner_ui_token, ui_token
from apex.testing import write_small_deck

TOKEN = "test-token"
#: How a refused include is answered (a configuration error).
STATUS = 422


def _world(tmp_path: Path, cap: int | None = None) -> tuple[Path, TestClient, dict[str, str]]:
    write_small_deck(tmp_path / "decks")
    root = tmp_path / "root"
    root.mkdir()
    (root / "service.toml").write_text(
        f'[workspace]\nroots = ["{tmp_path}"]\n[agents.case_converter]\nensemble_size = 20\n'
        + ("" if cap is None else f"ui_max_cells = {cap}\n")
        + "[agents.geomodeler]\nensemble_size = 20\nqc_members = 20\n"
    )

    def unavailable() -> object:
        raise ConfigError("no key for the model")

    http = TestClient(create_app(root, token=TOKEN, agent_provider=unavailable))  # type: ignore[arg-type]
    owner = {"Authorization": f"Bearer {owner_ui_token(root, create=True)}",
             "X-APEX-Client": "ui"}  # fmt: skip
    return root, http, owner


def test_the_owner_converts_a_deck_and_gives_it_a_prior(tmp_path: Path) -> None:
    _, http, owner = _world(tmp_path)

    converted = http.post("/v1/agents/convert", headers=owner,
                          json={"source": "decks/CASE.DATA", "name": "small"})  # fmt: skip
    assert converted.status_code == 200, converted.text
    body = converted.json()
    prior = http.post("/v1/agents/geomodel", headers=owner, json={"config": body["draft"]})

    assert body["draft"].startswith("UI/owner/drafts/case-converter/small-")
    assert body["valid"], body["errors"]
    assert body["decided_by"] == "rules"
    assert prior.status_code == 200, prior.text
    assert prior.json()["config"].endswith("config_prior.toml")
    assert prior.json()["valid"], (prior.json()["errors"], prior.json()["qc"])
    draft = tmp_path / body["draft"]
    assert str(tmp_path) not in draft.read_text()  # no server paths in the draft


def test_a_created_case_waits_for_its_truth_run(tmp_path: Path) -> None:
    _, http, owner = _world(tmp_path)

    created = http.post("/v1/agents/convert/create", headers=owner, json={"name": "box"})

    assert created.status_code == 200, created.text
    body = created.json()
    assert not body["valid"]
    assert "apex agent convert observe" in body["errors"][0]
    raw = tomllib.loads((tmp_path / body["draft"]).read_text())
    assert raw["case"]["kind"] == "synthetic"


def test_guests_see_public_skills_and_never_run_the_truth(tmp_path: Path) -> None:
    root, http, owner = _world(tmp_path)
    (root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["decks"]\n[agents]\nnames = ["case-converter"]\n'
        '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nlocalizations = ["*"]\n'
        'simulators = ["*"]\nexecutors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
    )
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}

    owners = {s["name"] for s in http.get("/v1/agents/convert/skills", headers=owner).json()}
    guests = {s["name"] for s in http.get("/v1/agents/convert/skills", headers=guest).json()}
    observe = http.post("/v1/agents/convert/create", headers=guest,
                        json={"name": "box", "observe": True})  # fmt: skip
    geomodel = http.post("/v1/agents/geomodel", headers=guest, json={"config": "x.toml"})

    assert {"opm_flow", "pet"} <= guests
    assert not {"intersect", "geos"} & guests
    assert guests <= owners
    assert observe.status_code == 403
    assert geomodel.status_code == 404  # the geomodeler is not in alice's list


GUEST_ACCESS = (
    '[workspace]\nfolders = ["decks"]\n[agents]\nnames = ["case-converter", "geomodeler"]\n'
    '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nlocalizations = ["*"]\n'
    'simulators = ["*"]\nexecutors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
)


def test_a_guests_deck_never_includes_files_outside_their_folders(tmp_path: Path) -> None:
    """The review's blocker (2026-10-24): a deck whose INCLUDE names a file elsewhere must
    not be read, nor copied into the guest's drafts, whether or not that file exists."""
    root, http, _ = _world(tmp_path)
    (root / "ui_access.toml").write_text(GUEST_ACCESS)
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}
    secret = tmp_path / "private" / "secret.inc"
    secret.parent.mkdir()
    secret.write_text("-- the owner's secret\n")
    deck = tmp_path / "decks" / "CASE.DATA"
    text = deck.read_text()
    for name, target in (("EVIL", secret), ("GONE", tmp_path / "private" / "none.inc")):
        evil = deck.with_name(f"{name}.DATA")
        evil.write_text(text.replace("PROPS\n", f"PROPS\nINCLUDE\n '{target}' /\n", 1))
    answers = [http.post("/v1/agents/convert", headers=guest,
                         json={"source": f"decks/{n}.DATA", "name": n.lower()})
               for n in ("EVIL", "GONE")]  # fmt: skip

    assert [a.status_code for a in answers] == [STATUS, STATUS]
    # The same answer whether the file exists or not (but for its name).
    assert answers[0].json()["detail"] == answers[1].json()["detail"].replace(
        "none.inc", "secret.inc"
    )
    copied = list((root.parent).rglob("drafts/case-converter/**/secret.inc"))
    assert copied == []


def test_guests_are_refused_what_is_not_theirs(tmp_path: Path) -> None:
    root, http, owner = _world(tmp_path)
    (root / "ui_access.toml").write_text(GUEST_ACCESS)
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}

    private = http.post("/v1/agents/convert", headers=guest,
                        json={"source": "decks/CASE.DATA", "skill": "intersect"})  # fmt: skip
    hidden = http.post("/v1/agents/convert", headers=guest, json={"source": "elsewhere/X.DATA"})
    wrong = http.post("/v1/agents/convert", headers=owner,
                      json={"source": "decks/CASE.DATA", "skill": "pet"})  # fmt: skip
    beside = http.post("/v1/agents/geomodel", headers=guest, json={"config": "decks/x.toml"})

    assert private.status_code == 404
    assert hidden.status_code == 404
    assert wrong.status_code >= 400  # pet does not read a deck
    assert beside.status_code == 403  # not in alice's own folder


def test_the_web_ui_bounds_the_work_it_starts(tmp_path: Path) -> None:
    """A guest's case is converted or created in the service's request: bounded in cells."""
    root, http, _ = _world(tmp_path, cap=100)
    (root / "ui_access.toml").write_text(GUEST_ACCESS)
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}

    created = http.post("/v1/agents/convert/create", headers=guest, json={"name": "box"})
    small = http.post("/v1/agents/convert", headers=guest, json={"source": "decks/CASE.DATA"})

    assert created.status_code == 422
    assert "at most 100" in created.json()["detail"]
    assert small.status_code == 200, small.text  # 24 cells


def test_hard_data_are_read_only_from_the_callers_folders(tmp_path: Path) -> None:
    """The review (2026-10-25): a guest's `data` outside their folders is not found, as a
    missing file is; their own is estimated from."""
    root, http, _ = _world(tmp_path)
    (root / "ui_access.toml").write_text(GUEST_ACCESS)
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}
    converted = http.post("/v1/agents/convert", headers=guest,
                          json={"source": "decks/CASE.DATA", "name": "small"})  # fmt: skip
    assert converted.status_code == 200, converted.text
    draft = converted.json()["draft"]
    rows = "i,j,k,property,value\n1,1,1,poro,0.2\n2,1,1,poro,0.22\n3,2,2,poro,0.18\n"
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "wells.csv").write_text(rows)
    (tmp_path / "decks" / "wells.csv").write_text(rows)

    hidden = http.post("/v1/agents/geomodel", headers=guest,
                       json={"config": draft, "data": "private/wells.csv"})  # fmt: skip
    missing = http.post("/v1/agents/geomodel", headers=guest,
                        json={"config": draft, "data": "private/none.csv"})  # fmt: skip
    own = http.post("/v1/agents/geomodel", headers=guest,
                    json={"config": draft, "data": "decks/wells.csv"})  # fmt: skip

    assert hidden.status_code == missing.status_code == 404
    assert own.status_code == 200, own.text
    assert any("estimated from the hard data" in n for n in own.json()["notes"])
