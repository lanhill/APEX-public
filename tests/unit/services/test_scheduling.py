"""The scheduling policy, as a pure function."""

from collections.abc import Mapping, Sequence

from apex.services.scheduling import Quota, select_starts

POOLS = {"cpu": 18, "eclipse": 3}


def _job(
    job_id: str,
    *,
    created: str = "t0",
    priority: str = "normal",
    client: str = "user",
    reservation: Mapping[str, int] | None = None,
    planned_runs: int = 0,
    after: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "id": job_id, "created": created, "priority": priority, "client": client,
        "reservation": dict(reservation or {}), "planned_runs": planned_runs, "after": list(after),
    }  # fmt: skip


def _starts(
    queued: Sequence[dict[str, object]],
    running: Sequence[dict[str, object]] = (),
    *,
    max_concurrent_jobs: int = 10,
    quotas: Mapping[str, Quota] | None = None,
    usage: int = 0,
    states: Mapping[str, str] | None = None,
) -> list[str]:
    return select_starts(
        queued,
        running,
        pools=POOLS,
        max_concurrent_jobs=max_concurrent_jobs,
        quotas=quotas or {},
        usage_today=lambda client: usage,
        finished=lambda job: (states or {}).get(job),
    )


def test_priority_then_age_within_the_job_limit() -> None:
    queued = [
        _job("old_low", created="t0", priority="low"),
        _job("new_high", created="t2", priority="high"),
        _job("old_normal", created="t1"),
        _job("new_normal", created="t3"),
    ]

    assert _starts(queued, max_concurrent_jobs=3) == ["new_high", "old_normal", "new_normal"]


def test_a_job_starts_only_when_its_reservation_fits() -> None:
    running = [_job("r", reservation={"eclipse": 3, "cpu": 3})]
    queued = [
        _job("eclipse_job", created="t0", reservation={"eclipse": 3, "cpu": 3}),
        _job("analytic", created="t1"),
    ]

    # The licences are taken: the Eclipse job waits; a job needing nothing starts.
    assert _starts(queued, running) == ["analytic"]
    assert _starts(queued) == ["eclipse_job", "analytic"]


def test_a_waiting_large_job_is_not_starved_by_smaller_ones() -> None:
    running = [_job("r", reservation={"cpu": 10})]
    queued = [
        _job("big", created="t0", reservation={"cpu": 18}),
        _job("small", created="t1", reservation={"cpu": 4}),
        _job("eclipse", created="t2", reservation={"eclipse": 1}),
    ]

    # "small" would fit, but "big" came first and needs the cpu pool: it waits behind it.
    # "eclipse" needs another pool and starts.
    assert _starts(queued, running) == ["eclipse"]


def test_reservations_larger_than_a_pool_are_capped() -> None:
    assert _starts([_job("huge", reservation={"cpu": 64})]) == ["huge"]
    assert (
        _starts([_job("huge", reservation={"cpu": 64})], [_job("r", reservation={"cpu": 1})]) == []
    )


def test_client_quotas_and_dependencies() -> None:
    quotas = {"planner": Quota(max_concurrent_jobs=1, max_simulator_runs_per_day=1000)}
    running = [_job("r", client="planner")]
    queued = [
        _job("second", client="planner"),
        _job("other_client", client="user"),
        _job("waits", after=["dep"]),
        _job("ready", after=["done"]),
    ]

    assert _starts(
        queued, running, quotas=quotas, states={"dep": "running", "done": "complete"}
    ) == [
        "other_client",
        "ready",
    ]
    # The daily budget: 800 already used, a 300-run job would exceed 1000.
    quotas = {"planner": Quota(max_simulator_runs_per_day=1000)}
    over = [
        _job("big", client="planner", planned_runs=300),
        _job("fits", client="planner", planned_runs=200),
    ]
    assert _starts(over, quotas=quotas, usage=800) == ["fits"]
