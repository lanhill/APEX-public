"""Budget grants for approval: the gate, the store's atomic charge, the CLI."""

import datetime as dt
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.services.api import ApexService
from apex.services.approval import ApprovalRequired, Grants, parse_duration
from apex.storage.sqlite_store import SQLiteStore

WEEK = dt.timedelta(days=7)
X = {"experiment": "X"}


@pytest.fixture
def service(tmp_path: Path) -> ApexService:
    # Over 50 simulator runs needs approval.
    return ApexService(tmp_path / "service", approval_threshold_runs=50)


def _grant(service: ApexService, budget: int = 250, client: str = "planner") -> str:
    record = service.grants.create(client, "X", budget, WEEK, by="test")
    return str(record["id"])


def test_a_grant_approves_a_matching_submission_and_is_charged(service: ApexService) -> None:
    grant = _grant(service)

    service.gate(["r1"], 100, client="planner", annotations=X)

    record = service.grants.store.grant(grant)
    assert record is not None
    assert record["used_simulator_runs"] == 100
    [use] = record["uses"]
    assert use["simulator_runs"] == 100
    approval = service.jobs.store.approval(use["token"])
    assert approval is not None
    assert approval["approved_by"] == f"grant {grant}"
    assert approval["used"]
    # A grant also covers a submission over the number of runs, when its budget holds.
    service.gate([f"r{i}" for i in range(9)], 10, client="planner", annotations=X)


def test_two_submissions_use_up_the_budget_then_approval_is_required(
    service: ApexService,
) -> None:
    _grant(service, budget=220)

    service.gate(["r1"], 100, client="planner", annotations=X)
    service.gate(["r2"], 100, client="planner", annotations=X)
    with pytest.raises(ApprovalRequired, match="has 20 simulator runs left") as refused:
        service.gate(["r3"], 100, client="planner", annotations=X)
    assert refused.value.token
    # Below the approval limits no grant is involved, and nothing is charged.
    service.gate(["r4"], 10, client="planner", annotations=X)
    [record] = service.grants.records()
    assert record["used_simulator_runs"] == 200


def test_the_grant_that_expires_first_is_charged(service: ApexService) -> None:
    late = service.grants.create("planner", "X", 500, WEEK, by="test")
    soon = service.grants.create("planner", "X", 500, dt.timedelta(hours=1), by="test")

    service.gate(["r1"], 100, client="planner", annotations=X)

    used = {g["id"]: g["used_simulator_runs"] for g in service.grants.records()}
    assert used == {soon["id"]: 100, late["id"]: 0}


def test_an_expired_grant_does_not_approve(service: ApexService) -> None:
    grant = _grant(service)
    record = service.grants.store.grant(grant)
    assert record is not None
    record["expires"] = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)).isoformat()
    service.grants.store.put_grant(grant, record)

    with pytest.raises(ApprovalRequired, match="a grant is expired"):
        service.gate(["r1"], 100, client="planner", annotations=X)
    assert service.grants.records() == []
    assert len(service.grants.records(include_inactive=True)) == 1


def test_a_revoked_grant_does_not_approve(service: ApexService) -> None:
    grant = _grant(service)
    service.grants.revoke(grant, by="test")

    with pytest.raises(ApprovalRequired, match="a grant was revoked"):
        service.gate(["r1"], 100, client="planner", annotations=X)
    with pytest.raises(ConfigError, match="already revoked"):
        service.grants.revoke(grant, by="test")
    with pytest.raises(ConfigError, match="no grant"):
        service.grants.revoke("0" * 16, by="test")


def test_several_revoked_grants_are_reported_once(service: ApexService) -> None:
    """Release tester: the refusal read "a grant was revoked; a grant was revoked"."""
    for grant in (_grant(service), _grant(service)):
        service.grants.revoke(grant, by="test")

    with pytest.raises(ApprovalRequired) as refused:
        service.gate(["r1"], 100, client="planner", annotations=X)
    assert "2 grants were revoked" in str(refused.value)
    assert "a grant was revoked" not in str(refused.value)


def _expire(service: ApexService, grant: str) -> None:
    record = service.grants.store.grant(grant)
    assert record is not None
    record["expires"] = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)).isoformat()
    service.grants.store.put_grant(grant, record)


def test_the_refusal_counts_grants_of_each_kind(service: ApexService) -> None:
    """Expired grants, grants that cover only known costs, and a mix, each said once."""
    for grant in (_grant(service), _grant(service)):
        _expire(service, grant)
    assert service.grants.why_not("planner", "X", 100) == "2 grants are expired"

    fresh = ApexService(service.root.parent / "other", approval_threshold_runs=50)
    _grant(fresh)
    _grant(fresh)
    assert fresh.grants.why_not("planner", "X", None) == (
        "2 grants cover only known costs, and this one is unknown"
    )
    _grant(fresh, budget=40)
    assert fresh.grants.why_not("planner", "X", None) == (
        "3 grants cover only known costs, and this one is unknown"
    )

    mixed = ApexService(service.root.parent / "mixed", approval_threshold_runs=50)
    mixed.grants.revoke(_grant(mixed), by="test")
    _expire(mixed, _grant(mixed))
    _grant(mixed, budget=40)
    assert mixed.grants.why_not("planner", "X", 100) == (
        "a grant was revoked; a grant is expired; a grant has 40 simulator runs left"
    )
    single = ApexService(service.root.parent / "single", approval_threshold_runs=50)
    _grant(single)
    assert single.grants.why_not("planner", "X", None) == (
        "a grant covers only known costs, and this one is unknown"
    )


def test_a_grant_is_only_for_its_client_and_experiment(service: ApexService) -> None:
    _grant(service)

    with pytest.raises(ApprovalRequired) as other_client:
        service.gate(["r1"], 100, client="someone", annotations=X)
    assert "grant" not in str(other_client.value)
    with pytest.raises(ApprovalRequired):
        service.gate(["r1"], 100, client="planner", annotations={"experiment": "Y"})
    with pytest.raises(ApprovalRequired, match="no experiment annotation"):
        service.gate(["r1"], 100, client="planner")
    [record] = service.grants.records()
    assert record["used_simulator_runs"] == 0


def test_an_unknown_cost_never_uses_a_grant(service: ApexService) -> None:
    _grant(service)

    with pytest.raises(ApprovalRequired, match="covers only known costs"):
        service.gate(["r1"], None, client="planner", annotations=X)
    [record] = service.grants.records()
    assert record["used_simulator_runs"] == 0
    assert record["uses"] == []


def test_concurrent_charges_cannot_overspend(tmp_path: Path) -> None:
    """Eight submissions of 30 against a budget of 100, each on its own connection."""
    root = tmp_path / "service"
    grant = Grants(SQLiteStore(root)).create("planner", "X", 100, WEEK, by="test")
    stores = [SQLiteStore(root) for _ in range(8)]
    start = threading.Barrier(len(stores))
    charged: list[bool] = []

    def submit(store: SQLiteStore) -> None:
        start.wait()
        result = Grants(store).charge("planner", "X", 30, record=lambda _: "token")
        charged.append(result is not None)

    threads = [threading.Thread(target=submit, args=(s,)) for s in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(charged) == [False] * 5 + [True] * 3
    record = SQLiteStore(root).grant(grant["id"])
    assert record is not None
    assert record["used_simulator_runs"] == 90
    assert len(record["uses"]) == 3


def test_grants_refuse_bad_budgets_names_and_lifetimes(service: ApexService) -> None:
    with pytest.raises(ConfigError, match="positive"):
        service.grants.create("planner", "X", 0, WEEK, by="test")
    with pytest.raises(ConfigError, match="client and an experiment"):
        service.grants.create("planner", " ", 10, WEEK, by="test")
    with pytest.raises(ConfigError, match="at most 30 days"):
        service.grants.create("planner", "X", 10, dt.timedelta(days=30, minutes=1), by="test")
    with pytest.raises(ConfigError, match="positive"):
        service.grants.create("planner", "X", 10, dt.timedelta(0), by="test")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7d", dt.timedelta(days=7)),
        ("12h", dt.timedelta(hours=12)),
        ("30m", dt.timedelta(minutes=30)),
    ],
)
def test_durations_are_parsed(text: str, expected: dt.timedelta) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["0d", "-1d", "7", "7w", "1.5d", "d", "", "7 d"])
def test_bad_durations_are_refused(text: str) -> None:
    with pytest.raises(ConfigError, match="duration"):
        parse_duration(text)


def test_the_cli_creates_lists_and_revokes_grants(tmp_path: Path) -> None:
    root = str(tmp_path / "service")
    runner = CliRunner()

    def apex(*args: str) -> tuple[int, str]:
        result = runner.invoke(app, [*args, "--root", root])
        return result.exit_code, result.output

    code, output = apex(
        "grant", "create", "planner", "X", "--max-simulator-runs", "500", "--expires", "7d"
    )
    assert code == 0, output
    grant = output.split()[1].rstrip(":")
    code, listed = apex("grant", "list")
    assert code == 0, listed
    assert grant in listed
    assert "0/500" in listed
    for bad in ("31d", "0h", "7w", "-3d"):
        code, output = apex(
            "grant", "create", "planner", "X", "--max-simulator-runs", "5", "--expires", bad
        )
        assert code == 1, (bad, output)
    code, output = apex(
        "grant", "create", "planner", "X", "--max-simulator-runs", "0", "--expires", "1d"
    )
    assert code == 1, output
    assert (
        "at most 30 days"
        in apex(
            "grant", "create", "planner", "X", "--max-simulator-runs", "5", "--expires", "721h"
        )[1]
    )

    code, output = apex("grant", "revoke", grant)
    assert code == 0, output
    assert "0 of 500" in output
    assert apex("grant", "list")[1].strip() == "no active grants"
    code, listed = apex("grant", "list", "--all")
    assert grant in listed
    assert "revoked" in listed
    assert apex("grant", "revoke", grant)[0] == 1
    assert apex("grant", "revoke", "nope")[0] == 1


def test_an_explicit_no_is_final_even_with_a_grant(service: ApexService) -> None:
    # Code review: an interactive "no" fell through to a matching grant.
    grant = service.grants.create("user", "X", 1000, WEEK, by="test")
    with pytest.raises(ApprovalRequired):
        service.gate(["r1"], 900, approve=lambda cost: False, client="user", annotations=X)
    record = service.grants.store.grant(str(grant["id"]))
    assert record is not None
    assert record["used_simulator_runs"] == 0


def test_an_absurd_duration_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="too long"):
        parse_duration("99999999999d")


def test_a_padded_name_is_refused(service: ApexService) -> None:
    for client, experiment in (("planner ", "X"), ("planner", " X")):
        with pytest.raises(ConfigError, match="spaces"):
            service.grants.create(client, experiment, 10, WEEK, by="test")
