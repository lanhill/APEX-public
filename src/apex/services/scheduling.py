"""Which queued jobs to start: admission, priorities and quotas.

A pure function of the queue, the running jobs, the machine's pools and the settings, so
that the policy is tested without processes:

- **priority** ``high`` before ``normal`` before ``low``; oldest first within a priority;
- **dependencies**: a job waits for the jobs in its ``after`` to complete;
- **admission by reservation**: a job starts only when its reservation (``{pool: tokens}``,
  its runs at once times each run's tokens) fits beside those of the running jobs. A
  reservation larger than a pool is capped at the pool (the job then waits for an
  otherwise idle pool rather than never starting);
- **quotas** per client: running jobs, and simulator runs admitted per day;
- **the limit** of jobs at once.

A job waiting for resources keeps its claim on the pools it needs: later jobs that need
any of those pools wait behind it, so a large job is not starved by a stream of small
ones; jobs that need other pools (or none) still start. A job held back by its client's
quota claims nothing.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from apex.core.jobs import JobRecord

PRIORITIES = {"high": 0, "normal": 1, "low": 2}


@dataclass(frozen=True)
class Quota:
    """A client's limits (``None``: unlimited)."""

    max_concurrent_jobs: int | None = None
    max_simulator_runs_per_day: int | None = None


def reservation(record: JobRecord, pools: Mapping[str, int]) -> dict[str, int]:
    """Return a job's reservation, capped at each pool's size (unknown pools ignored)."""
    wanted: Mapping[str, int] = record.get("reservation") or {}
    return {pool: min(int(n), pools[pool]) for pool, n in wanted.items() if pool in pools and n > 0}


def select_starts(
    queued: Sequence[JobRecord],
    running: Sequence[JobRecord],
    *,
    pools: Mapping[str, int],
    max_concurrent_jobs: int,
    quotas: Mapping[str, Quota],
    usage_today: Callable[[str], int],
    finished: Callable[[str], str | None],
) -> list[str]:
    """Return the ids of the queued jobs to start now, in the order to start them.

    ``finished(job_id)`` gives a dependency's state (``None`` if unknown), and
    ``usage_today(client)`` the simulator runs already admitted for that client today.
    """
    reserved = dict.fromkeys(pools, 0)
    per_client: dict[str, int] = {}
    for record in running:
        for pool, n in reservation(record, pools).items():
            reserved[pool] += n
        client = str(record.get("client", "user"))
        per_client[client] = per_client.get(client, 0) + 1
    admitted_runs: dict[str, int] = {}
    slots = max_concurrent_jobs - len(running)
    starts: list[str] = []
    claimed: set[str] = set()  # pools a waiting, earlier job needs
    order = sorted(
        queued, key=lambda r: (PRIORITIES.get(str(r.get("priority")), 1), r["created"], r["id"])
    )
    for record in order:
        if slots <= 0:
            break
        if any(finished(d) != "complete" for d in record.get("after", [])):
            continue
        client = str(record.get("client", "user"))
        quota = quotas.get(client, Quota())
        if (
            quota.max_concurrent_jobs is not None
            and per_client.get(client, 0) >= quota.max_concurrent_jobs
        ):
            continue
        planned = int(record.get("planned_runs") or 0)
        if quota.max_simulator_runs_per_day is not None:
            today = usage_today(client) + admitted_runs.get(client, 0)
            if today + planned > quota.max_simulator_runs_per_day:
                continue
        wanted = reservation(record, pools)
        if claimed & set(wanted) or any(
            reserved[pool] + n > pools[pool] for pool, n in wanted.items()
        ):
            claimed |= set(wanted)
            continue
        for pool, n in wanted.items():
            reserved[pool] += n
        per_client[client] = per_client.get(client, 0) + 1
        admitted_runs[client] = admitted_runs.get(client, 0) + planned
        slots -= 1
        starts.append(record["id"])
    return starts
