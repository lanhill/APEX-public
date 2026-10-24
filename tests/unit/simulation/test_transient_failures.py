"""Which failed simulator runs are transient (worth retrying) and which are not."""

import stat
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import SimulationError, TransientSimulationError
from apex.core.simulation import MemberInput
from apex.domain import DataKey
from apex.simulation.deck import failure_error
from apex.simulation.eclipse import EclipseSimulator
from apex.simulation.opm_flow import OPMFlowSimulator

#: What Open MPI printed on a cluster host before ``MPI_Init`` failed (2026-10-14); successful
#: runs can print it too.
PMIX = "PMIX ERROR: PMIX_ERR_FILE_OPEN_FAILURE in file pmix_shmem.c at line 156"
#: Flow's way out after an error of its own, once MPI is running.
MPI_ABORT = "MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD"


@pytest.mark.parametrize(
    ("log", "returncode", "transient"),
    [
        ("FLEXnet Licensing error:-15,570. System Error: 115", 1, True),
        ("@--  ERROR  LICENSE CHECKOUT FAILED FOR ECLIPSE100", 1, True),
        ("Cannot connect to license server system.", 1, True),
        ("cp: cannot stat 'MODEL.UNSMRY': Stale file handle", 1, True),
        ("write error: Input/output error", 1, True),
        # Flow's MPI start-up on a busy host (Open MPI 4, 2026-10-14).
        (f"{PMIX}\n*** An error occurred in MPI_Init\n*** on a NULL communicator", 1, True),
        ("*** An error occurred in MPI_Init_thread", 1, True),
        ("", -9, True),  # killed by a signal, e.g. the out-of-memory killer
        ("Problem: Solver convergence failure - Iteration limit reached", 1, False),
        ("Error: keyword PERMX has 99 values, expected 100", 1, False),
        # PMIx's warning alone is no start-up failure: a later deck error stays permanent.
        (f"{PMIX}\nError: keyword PERMX has 99 values\n{MPI_ABORT}", 1, False),
        ("", 1, False),
    ],
)
def test_failures_are_classified(
    tmp_path: Path, log: str, returncode: int, transient: bool
) -> None:
    path = tmp_path / "run.log"
    path.write_text(log)

    error = failure_error("the run failed", returncode, path)

    assert isinstance(error, SimulationError)
    assert isinstance(error, TransientSimulationError) is transient


def test_site_patterns_add_transient_failures(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text("MPI_ABORT: node n042 unreachable")

    assert not isinstance(failure_error("x", 1, path), TransientSimulationError)
    assert isinstance(
        failure_error("x", 1, path, [r"node \w+ unreachable"]), TransientSimulationError
    )


def _member(tmp_path: Path) -> MemberInput:
    work_dir = tmp_path / "member_0"
    work_dir.mkdir()
    return MemberInput(0, {"permx": np.array([1.0])}, work_dir)


def _script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _template(tmp_path: Path) -> Path:
    path = tmp_path / "deck.mako"
    path.write_text("${permx[0]}\n")
    return path


def test_a_licence_failure_of_eclrun_is_transient(tmp_path: Path) -> None:
    simulator = EclipseSimulator(
        _template(tmp_path),
        executable=_script(
            tmp_path, "eclrun", "echo 'FLEXlm error: license not available'; exit 3\n"
        ),
    )

    with pytest.raises(TransientSimulationError, match="transient: output matches"):
        simulator.run(_member(tmp_path), [DataKey("2024-01-01", "WOPR:P1")])


ECLEND = """ Error summary
  Comments               0
  Warnings               0
  Problems               0
  Errors                 1
  Bugs                   0
  Final cpu       0.1 elapsed       0.1
"""
PRT = " @--  ERROR  AT TIME        0.0   DAYS\n @           LICENSE SERVER NOT AVAILABLE\n"


def test_a_licence_failure_in_the_prt_is_transient(tmp_path: Path) -> None:
    # eclrun exits 0; the error summary counts an error, which the PRT names.
    (tmp_path / "report.ECLEND").write_text(ECLEND)
    (tmp_path / "report.PRT").write_text(PRT)
    body = f"cp {tmp_path}/report.ECLEND MODEL.ECLEND\ncp {tmp_path}/report.PRT MODEL.PRT\n"
    simulator = EclipseSimulator(_template(tmp_path), executable=_script(tmp_path, "eclrun", body))

    with pytest.raises(TransientSimulationError):
        simulator.run(_member(tmp_path), [DataKey("2024-01-01", "WOPR:P1")])


def test_a_flow_convergence_failure_is_not_transient_and_a_killed_flow_is(tmp_path: Path) -> None:
    keys = [DataKey("2024-01-01", "WOPR:P1")]
    failing = OPMFlowSimulator(
        _template(tmp_path),
        executable=_script(tmp_path, "flow", "echo 'Solver convergence failure'; exit 1\n"),
    )
    with pytest.raises(SimulationError) as raised:
        failing.run(_member(tmp_path), keys)
    assert not isinstance(raised.value, TransientSimulationError)

    killed = OPMFlowSimulator(
        _template(tmp_path), executable=_script(tmp_path, "flow2", "kill -9 $$\n")
    )
    work = tmp_path / "second"
    work.mkdir()
    with pytest.raises(TransientSimulationError, match="killed by signal 9"):
        killed.run(MemberInput(1, {"permx": np.array([1.0])}, work), keys)
