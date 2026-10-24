"""Executors that run ensemble members on the local machine.

Every run holds the machine-wide resource tokens its simulator declares (``resources``,
e.g. Eclipse ``{"eclipse": 1, "cpu": 1}``) for as long as it runs, so that
concurrent jobs and command-line runs together never exceed a pool (the Eclipse
licences, the cores). Simulators without ``resources`` hold none.
"""

import logging
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError, TransientSimulationError
from apex.core.plugins import executors
from apex.core.resources import ResourcePools
from apex.core.simulation import MemberCallback, MemberFailure, MemberInput, Simulator
from apex.domain.observations import DataKey
from apex.execution.resources import FilePools

logger = logging.getLogger(__name__)

#: How worker processes start. ``forkserver`` forks them from a small server process that
#: imports nothing heavy, so they never inherit threads that a library started in the run's
#: own process (OpenBLAS, pyarrow through pandas, ...): forking a multi-threaded process may
#: deadlock the child. It is Python 3.14's default; results do not depend on it. Workers
#: get the parent's import path, but not its loaded modules: a simulator must be importable
#: by its module name (any installed plugin is; so is a script's module on sys.path).
_START_METHOD = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


def _needs(simulator: Simulator) -> Mapping[str, int]:
    needs: Mapping[str, int] = getattr(simulator, "resources", None) or {}
    return needs


@dataclass(frozen=True)
class RetryPolicy:
    """How often a member's transient failures are retried (licences, file systems).

    Attributes
    ----------
    retries
        Runs after the first, for failures the simulator marks transient
        (:class:`~apex.core.errors.TransientSimulationError`). Other failures are never
        retried: a simulation that failed once fails the same way again.
    delay_s
        Seconds before the first retry, doubled for each further one. The member's
        resource tokens (e.g. an Eclipse licence) are released while it waits.
    """

    retries: int = 2
    delay_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate the policy."""
        if self.retries < 0 or self.delay_s < 0:
            raise ValueError(f"retries and delay_s must be non-negative, got {self}")


def run_one(
    simulator: Simulator,
    member: MemberInput,
    keys: Sequence[DataKey],
    pools: ResourcePools | None = None,
    retry: RetryPolicy | None = None,
) -> NDArray[np.float64] | MemberFailure:
    """Run one member here, retrying transient failures; return its forecast or failure.

    Public for other executors: only simulation failures become
    :class:`MemberFailure`; anything else is a bug and propagates.
    """
    if member.work_dir is not None:
        member.work_dir.mkdir(parents=True, exist_ok=True)
    needs = _needs(simulator)
    retry = retry or RetryPolicy(retries=0)
    attempt = 0
    while True:
        attempt += 1
        try:
            if needs and pools is not None:
                # Waits here, in the worker, until the tokens are free on this machine.
                with pools.hold(needs):
                    prediction = simulator.run(member, keys)
            else:
                prediction = simulator.run(member, keys)
            break
        except TransientSimulationError as error:
            if attempt > retry.retries:
                return MemberFailure(member.member_id, str(error), transient=True, attempts=attempt)
            wait = retry.delay_s * 2 ** (attempt - 1)
            logger.warning(
                "member %d: transient failure, retrying in %.0f s (%d of %d): %s",
                member.member_id, wait, attempt, retry.retries, str(error).splitlines()[0],
                extra={"event": "member_retry", "member_id": member.member_id,
                       "attempt": attempt, "wait_s": wait},
            )  # fmt: skip
            time.sleep(wait)  # outside the token hold: another member may use the licence
        except SimulationError as error:
            return MemberFailure(member.member_id, str(error), attempts=attempt)
    prediction = np.asarray(prediction, dtype=np.float64)
    # A simulator may return more than one value per key (a sensitivity run).
    output_size = getattr(simulator, "output_size", None)
    expected = len(keys) if output_size is None else int(output_size(len(keys)))
    if prediction.shape != (expected,):
        return MemberFailure(
            member.member_id, f"simulator returned shape {prediction.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(prediction)):
        return MemberFailure(member.member_id, "simulator returned non-finite predictions")
    return prediction


@executors.register("serial")
class SerialExecutor:
    """Runs members one after another in the calling process.

    Parameters
    ----------
    pools
        Machine-wide resource pools (default: this machine's, :class:`FilePools`).
    retries, retry_delay_s
        Retries of a member's transient failures, and the first wait (see
        :class:`RetryPolicy`).
    """

    def __init__(
        self,
        pools: ResourcePools | None = None,
        retries: int = 2,
        retry_delay_s: float = 30.0,
    ) -> None:
        self.pools: ResourcePools = FilePools() if pools is None else pools
        self.retry = RetryPolicy(retries, retry_delay_s)

    def run_all(
        self,
        simulator: Simulator,
        members: Sequence[MemberInput],
        keys: Sequence[DataKey],
        *,
        on_result: MemberCallback | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run every member in order."""
        outputs = []
        for member in members:
            output = run_one(simulator, member, keys, self.pools, self.retry)
            notify(on_result, member, output)
            outputs.append(output)
        return outputs


@executors.register("process")
class LocalProcessExecutor:
    """Runs members in a pool of worker processes.

    Parameters
    ----------
    workers
        Number of worker processes; defaults to the CPU count. Simulators that launch
        multi-threaded or MPI programs should budget their own cores within this. A
        simulator that declares ``max_parallel_runs`` (e.g. Eclipse's ``licenses``) caps
        it: never more runs at once than the simulator allows.

    pools
        Machine-wide resource pools (default: this machine's, :class:`FilePools`). Each
        worker holds its run's tokens, so other jobs' runs count against the same pools.
    retries, retry_delay_s
        Retries of a member's transient failures, and the first wait (see
        :class:`RetryPolicy`).

    Notes
    -----
    Per-member time limits belong to the simulator (e.g. a subprocess timeout), which
    can stop its own work cleanly; a pool cannot kill a running task.
    """

    def __init__(
        self,
        workers: int | None = None,
        pools: ResourcePools | None = None,
        retries: int = 2,
        retry_delay_s: float = 30.0,
    ) -> None:
        if workers is not None and workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self.workers = workers or os.cpu_count() or 1
        self.pools: ResourcePools = FilePools() if pools is None else pools
        self.retry = RetryPolicy(retries, retry_delay_s)

    def run_all(
        self,
        simulator: Simulator,
        members: Sequence[MemberInput],
        keys: Sequence[DataKey],
        *,
        on_result: MemberCallback | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run every member in the pool; results keep the input order."""
        if not members:
            return []
        workers = min(self.workers, len(members))
        cap = getattr(simulator, "max_parallel_runs", None)
        if cap is not None and cap < workers:
            logger.info(
                "running %d members at a time (the simulator allows %d), not %d",
                cap,
                cap,
                workers,
                extra={"event": "parallel_runs_capped", "workers": workers, "cap": cap},
            )
            workers = cap
        context = multiprocessing.get_context(_START_METHOD)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [
                pool.submit(run_one, simulator, member, keys, self.pools, self.retry)
                for member in members
            ]
            if on_result is not None:
                member_of = dict(zip(futures, members, strict=True))
                for future in as_completed(futures):
                    # A member that raised (a bug) is not reported; the list below
                    # re-raises the first such member in input order, as without callback.
                    if future.exception() is None:
                        notify(on_result, member_of[future], future.result())
            return [future.result() for future in futures]


def notify(
    on_result: MemberCallback | None,
    member: MemberInput,
    output: NDArray[np.float64] | MemberFailure,
) -> None:
    """Report one finished member to ``on_result``, if there is one."""
    if on_result is not None:
        on_result(member.member_id, not isinstance(output, MemberFailure))
