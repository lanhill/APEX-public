"""OPMFlowSimulator behaviour that does not need OPM Flow (a fake executable stands in)."""

import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import SimulationError
from apex.core.simulation import MemberInput
from apex.domain import DataKey
from apex.simulation.opm_flow import OPMFlowSimulator

KEYS = [DataKey("2024-01-01", "WOPR:P1")]


def _template(
    tmp_path: Path,
    body: str = "PERMX\n% for v in permx:\n${v}\n% endfor\n/\n",
    name: str = "deck.mako",
) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _fake_flow(tmp_path: Path, script: str) -> str:
    path = tmp_path / "fake_flow"
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _member(tmp_path: Path) -> MemberInput:
    work_dir = tmp_path / "member_3"
    work_dir.mkdir()
    return MemberInput(3, {"permx": np.array([1.5, 2.5])}, work_dir)


def test_deck_is_rendered_with_state_member_and_case_dir(tmp_path: Path) -> None:
    template = _template(tmp_path, "${member_id} ${case_dir}\n% for v in permx:\n${v}\n% endfor\n")
    simulator = OPMFlowSimulator(template, executable=_fake_flow(tmp_path, "exit 1\n"))
    member = _member(tmp_path)

    with pytest.raises(SimulationError):
        simulator.run(member, KEYS)

    deck = (member.work_dir / "MODEL.DATA").read_text().split()  # type: ignore[operator]
    assert deck == ["3", str(tmp_path), "1.5", "2.5"]


def test_flow_failure_reports_the_log_tail(tmp_path: Path) -> None:
    simulator = OPMFlowSimulator(
        _template(tmp_path), executable=_fake_flow(tmp_path, "echo 'Error: bad deck'\nexit 2\n")
    )

    with pytest.raises(SimulationError, match=r"exited with 2(.|\n)*bad deck"):
        simulator.run(_member(tmp_path), KEYS)


def test_timeout_missing_executable_and_bad_template_fail_the_member(tmp_path: Path) -> None:
    member = _member(tmp_path)
    slow = OPMFlowSimulator(
        _template(tmp_path), executable=_fake_flow(tmp_path, "sleep 5\n"), timeout_s=0.2
    )
    missing = OPMFlowSimulator(_template(tmp_path), executable="no-such-flow-binary")
    broken = OPMFlowSimulator(_template(tmp_path, "${undefined_name}\n", name="broken.mako"))

    with pytest.raises(SimulationError, match="exceeded"):
        slow.run(member, KEYS)
    with pytest.raises(SimulationError, match="not found on PATH"):
        missing.run(member, KEYS)
    with pytest.raises(SimulationError, match="deck template failed"):
        broken.run(member, KEYS)
    with pytest.raises(SimulationError, match="needs a work directory"):
        missing.run(MemberInput(0, {"permx": np.zeros(2)}), KEYS)


def test_constructor_validates_its_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        OPMFlowSimulator(tmp_path / "missing.mako")
    with pytest.raises(ValueError, match="timeout_s"):
        OPMFlowSimulator(_template(tmp_path), timeout_s=0)


def test_summary_reader_refuses_field_keys(tmp_path: Path) -> None:
    from apex.simulation.opm_flow import read_summary

    with pytest.raises(SimulationError, match="summary vectors only"):
        read_summary(tmp_path / "CASE", [DataKey("2024-01-01", "PRESSURE", 0)])


def test_case_dir_names_the_folder_of_the_decks_includes(tmp_path: Path) -> None:
    """A copied example keeps its template in the examples root and reads its own data
    files (2026-10-06: ``${case_dir}`` was always the template's folder)."""
    template = _template(tmp_path, "${case_dir}\n")
    copy = tmp_path / "copy"
    copy.mkdir()
    simulator = OPMFlowSimulator(template, executable=_fake_flow(tmp_path, "exit 1\n"),
                                 case_dir=copy)  # fmt: skip
    member = _member(tmp_path)

    with pytest.raises(SimulationError):
        simulator.run(member, KEYS)

    assert (member.work_dir / "MODEL.DATA").read_text().split() == [str(copy)]  # type: ignore[operator]
    with pytest.raises(ValueError, match="case_dir is not a folder"):
        OPMFlowSimulator(template, case_dir=tmp_path / "nowhere")


class _RecordingRunner:
    """A runner that records what Flow would run elsewhere, and exits ``code``."""

    def __init__(self, code: int | None, log: str = "") -> None:
        self.code = code
        self.log = log
        self.calls: list[dict[str, object]] = []

    def static_folder(self, local: Path, files: Sequence[str] | None = None) -> Path:
        self.files = files
        return Path("/remote/static") / local.name

    def run(
        self,
        command: Sequence[str],
        work_dir: Path,
        *,
        env: Mapping[str, str],
        timeout_s: float,
        log: Path,
    ) -> int | None:
        self.calls.append({"command": list(command), "work_dir": work_dir, "env": dict(env),
                           "timeout_s": timeout_s, "log": log})  # fmt: skip
        log.write_text(self.log)
        return self.code


def test_a_runner_runs_flow_elsewhere_on_the_decks_includes_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """${case_dir} names the runner's copy, and the command's paths are relative."""
    from dataclasses import replace

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)  # its default, 1, goes to the runner

    case = tmp_path / "case"
    case.mkdir()
    simulator = OPMFlowSimulator(_template(case, "${case_dir}\n"), flow_args=["--x=1"],
                                 timeout_s=99.0)  # fmt: skip
    runner = _RecordingRunner(0)
    member = replace(_member(tmp_path), runner=runner)

    case_path = simulator.simulate(member)

    assert (member.work_dir / "MODEL.DATA").read_text().strip() == "/remote/static/case"  # type: ignore[operator]
    [call] = runner.calls
    assert call["command"] == ["flow", "MODEL.DATA", "--output-dir=.", "--x=1"]
    assert call["work_dir"] == member.work_dir
    assert call["env"] == {"OMP_NUM_THREADS": "1"}
    assert call["timeout_s"] == 99.0
    assert case_path == member.work_dir / "MODEL"  # type: ignore[operator]
    assert runner.files == []  # the template names nothing under ${case_dir} but itself


@pytest.mark.parametrize(("code", "log", "error"), [
    (None, "", "exceeded 99.0 s"),
    (1, "Error: bad deck", "exited with 1(.|\n)*bad deck"),
    (-9, "", r"\[transient: killed by signal 9\]"),
])  # fmt: skip
def test_flow_failing_elsewhere_fails_the_member_as_here(
    tmp_path: Path, code: int | None, log: str, error: str
) -> None:
    from dataclasses import replace

    simulator = OPMFlowSimulator(_template(tmp_path), timeout_s=99.0)
    member = replace(_member(tmp_path), runner=_RecordingRunner(code, log))

    with pytest.raises(SimulationError, match=error):
        simulator.simulate(member)
