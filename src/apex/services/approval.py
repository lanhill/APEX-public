"""Approval of spending.

A submission needs a person's approval when it may take more simulator runs than
``[approval] max_simulator_runs`` or starts more runs than ``max_runs``
(``service.toml``). An approval is a record in the store: a **token** bound to the
submission's fingerprint (its run ids) and cost, which expires and can be used once.

- Interactive front ends pass an ``approve`` hook (the CLI asks the user): its yes is
  recorded as an approval.
- Otherwise the submission is refused with a pending token; a person approves it with
  ``apex approve TOKEN``, and the same submission is made again with ``approval_token``.

A **grant** (:class:`Grants`) approves in advance a bounded budget of simulator
runs for one client's experiment; submissions it covers are charged to it and recorded
as approvals by the grant.
"""

import datetime as dt
import hashlib
import json
import re
import secrets
from collections.abc import Callable, Sequence
from typing import Any

from apex.core.errors import ConfigError
from apex.services.identity import code_version
from apex.services.settings import ApprovalSection
from apex.storage.sqlite_store import SQLiteStore

#: How long a token may wait to be used (a person looks at it, then the client submits).
TOKEN_LIFETIME = dt.timedelta(hours=24)


class ApprovalRequired(ConfigError):
    """A submission needs a person's approval first; ``token`` is pending.

    ``reason`` is why, without the API's how-to (``approval_token=``), for front ends that
    say how in their own form (the CLI's ``--approval-token``); the message by default.
    """

    def __init__(self, message: str, token: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.token = token
        self.reason = reason or message


def fingerprint(run_ids: Sequence[str]) -> str:
    """Return the fingerprint of a submission: a hash of its run ids."""
    return hashlib.sha256(json.dumps(sorted(run_ids)).encode()).hexdigest()


def _short(code: Any) -> str:  # noqa: ANN401 (a stored record's field)
    if not isinstance(code, dict):
        return "unknown"
    if not code.get("commit"):  # an installed APEX: no checkout
        return f"version {code.get('apex')}"
    commit = str(code["commit"])[:12]
    dirty = code.get("uncommitted")
    return f"{commit} with uncommitted changes {str(dirty)[:8]}" if dirty else commit


def _changed(record: dict[str, Any]) -> str:
    """Say why a submission is not the approved one: APEX's code, or its runs, changed.

    Not every fingerprint holds the code (a prediction's, a resume's), so a changed code is
    named as the likely cause, not the only one.
    """
    approved, now = record.get("code"), code_version()
    if approved is not None and approved != now:
        return (f"APEX's code changed since it was approved ({_short(approved)} -> "
                f"{_short(now)}; a run's id includes the code), or its configuration or "
                "inputs differ")  # fmt: skip
    return "its runs differ from the approved ones (another configuration, or changed inputs)"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Approvals:
    """Approval tokens of a services root."""

    def __init__(self, store: SQLiteStore, settings: ApprovalSection) -> None:
        self.store = store
        self.settings = settings

    def needed(self, simulator_runs: int | None, runs: int) -> bool:
        """Return whether a submission of this size needs approval (unknown cost: yes)."""
        too_costly = simulator_runs is None or simulator_runs > self.settings.max_simulator_runs
        return too_costly or runs > self.settings.max_runs

    def request(self, print_: str, cost: dict[str, Any], *, requested_by: str) -> str:
        """Create a pending approval; return its token."""
        # Hexadecimal: a URL-safe token may begin with "-", which `apex approve` would
        # take for an option.
        token = secrets.token_hex(8)
        self.store.put_approval(
            token,
            {
                "fingerprint": print_,
                # The code is part of every run's id: kept to say so when it changes.
                "code": code_version(),
                "cost": cost,
                "requested_by": requested_by,
                "created": _now().isoformat(timespec="seconds"),
                "expires": (_now() + TOKEN_LIFETIME).isoformat(timespec="seconds"),
                "approved_by": None,
                "approved_at": None,
                "used": False,
            },
        )
        return token

    def approve(self, token: str, *, by: str) -> dict[str, Any]:
        """Approve a pending token.

        Raises
        ------
        ConfigError
            If there is no such token, or it expired or was used.
        """
        record = self._usable(token)
        record.update(approved_by=by, approved_at=_now().isoformat(timespec="seconds"))
        self.store.put_approval(token, record)
        return record

    def consume(self, token: str, print_: str) -> None:
        """Use an approved token for the submission it was issued for.

        Raises
        ------
        ConfigError
            If the token is unknown, expired, used, not approved, or for another submission.
        """
        record = self._usable(token)
        if record["fingerprint"] != print_:
            raise ConfigError(f"approval {token} is for another submission: {_changed(record)}; "
                              "validate it again, and approve its new token")  # fmt: skip
        if not record["approved_by"]:
            raise ConfigError(f"approval {token} is still pending: run `apex approve {token}`")
        record["used"] = True
        self.store.put_approval(token, record)

    def release(self, token: str, *, unapprove: bool) -> None:
        """Make a used token usable again (its submission failed); ``unapprove``: pending.

        For a submission refused after its token was used (the tools roll back a grant's
        charge with it). Unknown tokens are left alone.
        """
        with self.store.locked():
            record = self.store.approval(token)
            if record is None:
                return
            record["used"] = False
            if unapprove:
                record.update(approved_by=None, approved_at=None)
            self.store.put_approval(token, record)

    def record(self, print_: str, cost: dict[str, Any], *, by: str) -> str:
        """Record an approval given at once (e.g. by an interactive user); return its token."""
        token = self.request(print_, cost, requested_by=by)
        self.approve(token, by=by)
        self.consume(token, print_)
        return token

    def _usable(self, token: str) -> dict[str, Any]:
        record = self.store.approval(token)
        if record is None:
            raise ConfigError(f"no approval {token!r}")
        if record["used"]:
            raise ConfigError(f"approval {token} was already used")
        if dt.datetime.fromisoformat(record["expires"]) < _now():
            raise ConfigError(f"approval {token} expired")
        return record


#: The longest a grant may last: a grant is for one experiment, not a standing permission.
GRANT_MAX_LIFETIME = dt.timedelta(days=30)

_DURATION = re.compile(r"(\d+)([dhm])")
_UNITS = {"d": "days", "h": "hours", "m": "minutes"}


def parse_duration(text: str) -> dt.timedelta:
    """Return the duration of ``7d``, ``12h`` or ``30m``.

    Raises
    ------
    ConfigError
        If it is not a positive whole number of days, hours or minutes.
    """
    match = _DURATION.fullmatch(text.strip())
    if match is None:
        raise ConfigError(f"duration {text!r}: use a whole number and d, h or m (7d, 12h, 30m)")
    try:
        duration = dt.timedelta(**{_UNITS[match[2]]: int(match[1])})
    except OverflowError as error:
        raise ConfigError(f"duration {text!r} is too long") from error
    if duration <= dt.timedelta(0):
        raise ConfigError(f"duration {text!r} must be positive")
    return duration


def grant_state(record: dict[str, Any], at: dt.datetime | None = None) -> str:
    """Return ``active``, ``expired`` or ``revoked``."""
    if record["revoked"]:
        return "revoked"
    if dt.datetime.fromisoformat(record["expires"]) <= (at or _now()):
        return "expired"
    return "active"


class Grants:
    """Budget grants of a services root.

    A grant approves in advance the submissions of one client for one experiment (the
    ``experiment`` annotation), up to a total number of simulator runs and until it
    expires. The CLI (`apex grant`) creates, lists and revokes them; no REST route or tool
    does. Tool clients use them too: ``validate`` says whether one
    covers the submission, and the submission charges it when it spends its token. As for
    tokens, write access to the root is the boundary: any process that can write it (an
    in-process ``Client`` included) could create a grant, and ``created_by`` is what that
    caller says.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def create(
        self,
        client: str,
        experiment: str,
        max_simulator_runs: int,
        lifetime: dt.timedelta,
        *,
        by: str,
    ) -> dict[str, Any]:
        """Create a grant; return its record.

        Raises
        ------
        ConfigError
            If a name is empty, the budget is not positive, or the lifetime is not
            positive or longer than :data:`GRANT_MAX_LIFETIME`.
        """
        if not client.strip() or not experiment.strip():
            raise ConfigError("a grant needs a client and an experiment")
        if client != client.strip() or experiment != experiment.strip():
            # A name with spaces around it could never match a submission's.
            raise ConfigError("a grant's client and experiment may not start or end with spaces")
        if max_simulator_runs < 1:
            raise ConfigError(f"a grant's budget must be positive, not {max_simulator_runs}")
        if lifetime <= dt.timedelta(0):
            raise ConfigError("a grant's lifetime must be positive")
        if lifetime > GRANT_MAX_LIFETIME:
            raise ConfigError(f"a grant may last at most {GRANT_MAX_LIFETIME.days} days")
        created = _now()
        record: dict[str, Any] = {
            "id": secrets.token_hex(8),  # hexadecimal, like tokens: never taken for an option
            "client": client,
            "experiment": experiment,
            "max_simulator_runs": max_simulator_runs,
            "used_simulator_runs": 0,
            "expires": (created + lifetime).isoformat(timespec="seconds"),
            "created_by": by,
            "created_at": created.isoformat(timespec="seconds"),
            "revoked": False,
            "uses": [],
        }
        self.store.put_grant(record["id"], record)
        return record

    def revoke(self, grant_id: str, *, by: str) -> dict[str, Any]:
        """Revoke a grant at once; return its record.

        Raises
        ------
        ConfigError
            If there is no such grant, or it was revoked already.
        """
        with self.store.locked():
            record = self.store.grant(grant_id)
            if record is None:
                raise ConfigError(f"no grant {grant_id!r}")
            if record["revoked"]:
                raise ConfigError(f"grant {grant_id} was already revoked")
            record.update(
                revoked=True, revoked_by=by, revoked_at=_now().isoformat(timespec="seconds")
            )
            self.store.put_grant(grant_id, record)
        return record

    def records(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        """Return the active grants (all of them with ``include_inactive``), oldest first."""
        records = self.store.grants()
        if include_inactive:
            return records
        return [r for r in records if grant_state(r) == "active"]

    def charge(
        self,
        client: str,
        experiment: str | None,
        simulator_runs: int | None,
        *,
        record: Callable[[str], str],
    ) -> dict[str, Any] | None:
        """Charge a submission to a grant that covers it; return the grant, or ``None``.

        ``record(grant_id)`` records the approval and returns its token. It runs in the
        same transaction as the charge, so two submissions cannot both spend the last of
        a budget, even from two processes.
        """
        if simulator_runs is None or experiment is None:
            return None
        with self.store.locked():
            # Read inside the transaction: another submission may have charged meanwhile.
            grant = self.covering(client, experiment, simulator_runs)
            if grant is None:
                return None
            token = record(grant["id"])
            grant["used_simulator_runs"] += simulator_runs
            grant["uses"].append(
                {
                    "token": token,
                    "simulator_runs": simulator_runs,
                    "time": _now().isoformat(timespec="seconds"),
                }
            )
            self.store.put_grant(grant["id"], grant)
        return grant

    def refund(self, grant_id: str, token: str) -> None:
        """Take back the charge of the submission ``token`` (it failed after the charge)."""
        with self.store.locked():
            grant = self.store.grant(grant_id)
            if grant is None:
                return
            kept = [u for u in grant["uses"] if u["token"] != token]
            refunded = sum(int(u["simulator_runs"]) for u in grant["uses"] if u["token"] == token)
            grant["uses"] = kept
            grant["used_simulator_runs"] -= refunded
            self.store.put_grant(grant_id, grant)

    def covering(
        self, client: str, experiment: str | None, simulator_runs: int | None
    ) -> dict[str, Any] | None:
        """Return the grant that would pay for a submission now, without charging it.

        The active grant of ``client`` for ``experiment`` with budget left for
        ``simulator_runs`` that expires first (so budgets that would lapse are used first),
        or ``None``. The tools tell ``validate``'s caller with it; :meth:`charge` charges.
        """
        if simulator_runs is None:
            return None  # an unknown cost is never pre-approved (fail closed)
        if experiment is None:
            return None  # a grant is always for a named experiment
        candidates = [
            g
            for g in self.store.grants()
            if g["client"] == client
            and g["experiment"] == experiment
            and grant_state(g) == "active"
            and g["used_simulator_runs"] + simulator_runs <= g["max_simulator_runs"]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda g: (dt.datetime.fromisoformat(g["expires"]), g["id"]))

    def why_not(self, client: str, experiment: str | None, simulator_runs: int | None) -> str:
        """Explain why no grant covered a submission ('' when no grant is relevant)."""
        mine = [g for g in self.store.grants() if g["client"] == client]
        if experiment is None:
            if any(grant_state(g) == "active" for g in mine):
                return (
                    f"client {client!r} has grants, but the submission has no experiment annotation"
                )
            return ""
        # Counted, so that several grants in one state read once ("2 grants were revoked"),
        # not "a grant was revoked; a grant was revoked".
        counts: dict[str, int] = {}
        left_runs = []
        for grant in (g for g in mine if g["experiment"] == experiment):
            state = grant_state(grant)
            if state != "active":
                counts[state] = counts.get(state, 0) + 1
            elif simulator_runs is None:
                counts["unknown"] = counts.get("unknown", 0) + 1
            else:
                left_runs.append(grant["max_simulator_runs"] - grant["used_simulator_runs"])
        reasons = []
        # A fixed order (the store returns grants in no set order).
        order = {"revoked": 0, "expired": 1, "unknown": 2}
        for state, count in sorted(counts.items(), key=lambda kv: (order.get(kv[0], 3), kv[0])):
            if state == "unknown":
                reasons.append(
                    ("a grant covers" if count == 1 else f"{count} grants cover")
                    + " only known costs, and this one is unknown"
                )
            elif count == 1:
                reasons.append(f"a grant {'is' if state == 'expired' else 'was'} {state}")
            else:
                reasons.append(f"{count} grants {'are' if state == 'expired' else 'were'} {state}")
        reasons += [f"a grant has {left} simulator runs left" for left in left_runs]
        return "; ".join(reasons)
