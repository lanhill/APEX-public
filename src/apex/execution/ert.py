"""The ``ert`` executor: each forecast is one ``ert ensemble_experiment``.

The members of a forecast become the realizations of a generated ERT case: the case's own
lines (:meth:`apex.simulation.ert.ErtCase.forward_config`) plus ``NUM_REALIZATIONS``,
``RUNPATH``, ``ENSPATH`` and ``RUNPATH_FILE`` in the forecast's folder, ``QUEUE_SYSTEM
LOCAL`` with ``MAX_RUNNING`` = the workers, and ``MAX_SUBMIT`` = 1 + the retries. The
generated case is written next to the case's own file (``.apex_batch_<pid>_<n>.ert``), so
that its relative paths and ``<CONFIG_PATH>`` mean what they mean there, and removed
afterwards.

ERT runs in ``<forecast folder>/.ert/runs/realization-<i>``, where the members' fields are
written first (the simulator's ``write_parameters``); each realization is then moved to
its member's work directory and its summary read there. A realization without a summary
is a member failure. ERT's storage and logs stay in ``.ert``.

It runs the ``ert`` program (ERT 26.x, the ``ert`` extra) and imports nothing from ERT.
While a batch runs it holds the machine-wide tokens of as many runs as run at once.
"""

import itertools
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import SimulationError
from apex.core.options import Program
from apex.core.plugins import executors
from apex.core.resources import ResourcePools
from apex.core.simulation import MemberCallback, MemberFailure, MemberInput, Simulator
from apex.domain.observations import DataKey
from apex.execution.resources import FilePools

logger = logging.getLogger(__name__)


class ErtBatchCase(Protocol):
    """What the executor needs of the ``ert`` simulator (:mod:`apex.simulation.ert`)."""

    case_file: Path
    resources: dict[str, int]

    def write_parameters(self, member: MemberInput) -> None:
        """Write the member's parameters into its run path."""

    def forward_config(self) -> list[str]:
        """Return the case's lines that ERT runs."""

    def read(
        self, run_path: Path, realization: int, keys: Sequence[DataKey], iteration: int = 0
    ) -> NDArray[np.float64]:
        """Read a realization's results at ``keys``."""

    def read_batch(
        self, storage: Path, keys: Sequence[DataKey], n_realizations: int,
        environment: Mapping[str, str],
    ) -> NDArray[np.float64]:  # fmt: skip
        """Read a batch's results at ``keys`` from its ERT storage."""


_batches = itertools.count()


@executors.register("ert")
class ErtExecutor:
    """Runs a forecast's members as one ERT ensemble experiment, locally.

    Parameters
    ----------
    workers
        Realizations ERT runs at once (``MAX_RUNNING``).
    retries
        Resubmissions of a failed realization (``MAX_SUBMIT`` = 1 + retries).
    timeout_s
        Longest a batch may take.
    ert_executable
        The ``ert`` program; default the one in APEX's environment, else on ``PATH``.
    pools
        Machine-wide resource pools (set by the composition).
    """

    #: ERT itself (the ``ert`` extra; checked by ``apex plugins``, not imported).
    requires: ClassVar[Mapping[str, str]] = MappingProxyType({"ert": "ert"})

    def __init__(
        self,
        workers: int = 1,
        retries: int = 0,
        timeout_s: float = 24 * 3600.0,
        ert_executable: Program | None = None,
        pools: ResourcePools | None = None,
    ) -> None:
        if workers < 1 or retries < 0 or timeout_s <= 0:
            raise ValueError("workers >= 1, retries >= 0 and timeout_s > 0 are required")
        self.workers = int(workers)
        self.retries = int(retries)
        self.timeout_s = float(timeout_s)
        self.ert_executable = ert_executable
        self.pools: ResourcePools = FilePools() if pools is None else pools

    def run_all(
        self,
        simulator: Simulator,
        members: Sequence[MemberInput],
        keys: Sequence[DataKey],
        *,
        on_result: MemberCallback | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run every member in one ERT experiment; results keep the input order.

        Raises
        ------
        SimulationError
            If the simulator is not ``ert``, or ERT cannot be started.
        """
        # The simulator may come wrapped (its capabilities forwarded), so it is recognised by
        # its capability rather than its class.
        if not getattr(simulator, "ert_batches", False):
            raise SimulationError("the 'ert' executor runs the 'ert' simulator only")
        case = cast(ErtBatchCase, simulator)
        if not members:
            return []
        work_dirs = [m.work_dir for m in members]
        if any(d is None for d in work_dirs):
            raise SimulationError("the 'ert' executor needs a work directory per member")
        # ERT runs in folders of its own, beside the members' (never inside one), and each
        # realization is moved into its member's folder afterwards. Links would not do: ERT's
        # forward-model runner reports its resolved working directory, which then matches no
        # run path, and ERT waits two minutes per realization for its checksums.
        parents = [str(d.parent) for d in work_dirs if d is not None]
        batch = Path(os.path.commonpath(parents)) / ".ert"
        if batch.exists():
            shutil.rmtree(batch)
        runs = batch / "runs"
        runs.mkdir(parents=True)
        for i, member in enumerate(members):
            (runs / _name(i)).mkdir()
            case.write_parameters(MemberInput(member.member_id, member.state, runs / _name(i)))
        running = min(self.workers, len(members))
        config = case.case_file.parent / f".apex_batch_{os.getpid()}_{next(_batches)}.ert"
        config.write_text("\n".join([
            *case.forward_config(),
            "",
            "-- APEX's batch: realization i is the batch's i-th member.",
            f"NUM_REALIZATIONS {len(members)}",
            f"RUNPATH {runs}/realization-<IENS>",
            f"ENSPATH {batch / 'storage'}",
            f"RUNPATH_FILE {batch / 'runpath_list'}",
            "QUEUE_SYSTEM LOCAL",
            f"QUEUE_OPTION LOCAL MAX_RUNNING {running}",
            f"MAX_SUBMIT {1 + self.retries}",
            "MIN_REALIZATIONS 1",
        ]) + "\n")  # fmt: skip
        try:
            needs = {k: v * running for k, v in case.resources.items()}
            with self.pools.hold(needs):
                self._run_ert(config, batch)
        finally:
            config.unlink(missing_ok=True)
        batch_values = None
        if getattr(case, "responses", "summary") == "storage":
            # Every observation type, matched by ERT in the batch's storage.
            batch_values = case.read_batch(batch / "storage", keys, len(members), ert_environment())
        outputs: list[NDArray[np.float64] | MemberFailure] = []
        for i, member in enumerate(members):
            assert member.work_dir is not None
            if member.work_dir.exists():
                shutil.rmtree(member.work_dir)
            member.work_dir.parent.mkdir(parents=True, exist_ok=True)
            (runs / _name(i)).rename(member.work_dir)
            result: NDArray[np.float64] | MemberFailure
            if batch_values is not None:
                column = np.ascontiguousarray(batch_values[:, i])
                result = (
                    column
                    if np.isfinite(column).all()
                    else MemberFailure(
                        member.member_id,
                        f"ERT realization {i}: no response at "
                        f"{int((~np.isfinite(column)).sum())} observations",
                    )
                )
            else:
                try:
                    result = case.read(member.work_dir, i, keys)
                except SimulationError as error:
                    result = MemberFailure(member.member_id, f"ERT realization {i}: {error}")
            outputs.append(result)
            if on_result is not None:
                on_result(member.member_id, not isinstance(result, MemberFailure))
        return outputs

    def _run_ert(self, config: Path, batch: Path) -> None:
        environment = ert_environment()
        executable = self.ert_executable or shutil.which("ert", path=environment["PATH"])
        if executable is None:
            raise SimulationError("`ert` is not installed: install apex[ert] (ERT 26.x)")
        log = batch / "ert.log"
        with log.open("w") as handle:
            try:
                done = subprocess.run(
                    [executable, "--logdir", str(batch / "logs"), "ensemble_experiment",
                     config.name],
                    cwd=config.parent, env=environment, stdout=handle,
                    stderr=subprocess.STDOUT, timeout=self.timeout_s, check=False,
                )  # fmt: skip
            except subprocess.TimeoutExpired as error:
                raise SimulationError(f"ERT took longer than {self.timeout_s:g} s") from error
        if done.returncode != 0:
            # Failed realizations are reported per member; a failure of ERT itself (a bad
            # configuration) leaves every member without a summary, and says why here.
            logger.warning(
                "ert exited with %d (see %s)", done.returncode, log,
                extra={"event": "ert_batch_failed", "log": str(log)},
            )  # fmt: skip


def ert_environment() -> dict[str, str]:
    """Return the environment ERT runs in.

    ERT's local queue runs its fm_dispatch.py, and APEX's forward-model steps and readers
    run this Python: both live in this environment's bin folder, which goes first on PATH.
    A shell PYTHONPATH (e.g. a PET checkout) is dropped.
    """
    bin_dir = str(Path(sys.executable).parent)
    environment = {**os.environ, "PATH": os.pathsep.join([bin_dir, os.environ.get("PATH", "")])}
    environment.pop("PYTHONPATH", None)
    return environment


def _name(realization: int) -> str:
    return f"realization-{realization}"
