"""EclipseSimulator behaviour that does not need Eclipse (a fake ``eclrun`` stands in).

The ``.ECLEND`` and ``.PRT`` snippets follow real Eclipse 100 output (opm-tests'
``spe9/eclipse-simulation/SPE9_CP.ECLEND``). No test here uses an Eclipse licence.
"""

import datetime as dt
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import SimulationError
from apex.core.simulation import MemberInput
from apex.domain import DataKey
from apex.simulation.eclipse import EclipseSimulator, error_summary, prt_errors

KEYS = [DataKey("2024-02-01", "WOPR:P1"), DataKey("2024-03-01", "WOPR:P1")]

ECLEND = """
 Error summary
 Comments               1
 Warnings              10
 Problems               0
 Errors                 {errors}
 Bugs                   0

 This was an ECLIPSE 100 run. 4 diagnostics follow:-
"""
PRT_ERROR = """
 @--ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2024):
 @           KEYWORD PERMX HAS TOO FEW VALUES
 @           200 EXPECTED, 199 READ
    1 READING RUNSPEC
"""


@pytest.fixture
def summary_dir(tmp_path: Path) -> Path:
    """A summary (``MODEL.SMSPEC``/``.UNSMRY``) with WOPR:P1 = 100, 200 at two reports."""
    from resdata.summary import Summary

    directory = tmp_path / "summary"
    directory.mkdir()
    writer = Summary.writer(str(directory / "MODEL"), dt.datetime(2024, 1, 1), 1, 1, 1)
    writer.add_variable("WOPR", wgname="P1")
    for step, days in ((1, 31), (2, 60)):
        writer.add_t_step(step, days)["WOPR:P1"] = 100.0 * step
    writer.fwrite()
    return directory


def _template(tmp_path: Path, body: str = "% for v in permx:\n${v}\n% endfor\n") -> Path:
    path = tmp_path / "deck.mako"
    path.write_text(body)
    return path


def _fake_eclrun(tmp_path: Path, script: str) -> str:
    """A shell script standing in for eclrun; it records its arguments in ``args``."""
    path = tmp_path / "fake_eclrun"
    path.write_text('#!/bin/sh\necho "$@" > args\n' + script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _finishing(summary_dir: Path, errors: int = 0, prt: str = "") -> str:
    """A fake run that writes the reports and copies the summary into the work directory."""
    eclend = ECLEND.format(errors=errors).replace("\n", "\\n")
    return (
        f"printf '{eclend}' > MODEL.ECLEND\n"
        f"cat > MODEL.PRT <<'PRT'\n{prt}\nPRT\n"
        f"cp {summary_dir}/MODEL.SMSPEC {summary_dir}/MODEL.UNSMRY .\n"
    )


def _member(tmp_path: Path) -> MemberInput:
    work_dir = tmp_path / "member_3"
    work_dir.mkdir()
    return MemberInput(3, {"permx": np.array([1.5, 2.5])}, work_dir)


def test_a_clean_run_returns_the_summary_values(tmp_path: Path, summary_dir: Path) -> None:
    simulator = EclipseSimulator(
        _template(tmp_path),
        executable=_fake_eclrun(tmp_path, _finishing(summary_dir)),
        eclrun_args=["--version=2023.1"],
    )
    member = _member(tmp_path)

    values = simulator.run(member, KEYS)

    np.testing.assert_array_equal(values, [100.0, 200.0])
    assert member.work_dir is not None
    assert (member.work_dir / "args").read_text().split() == [
        "--version=2023.1",
        "eclipse",
        "MODEL.DATA",
    ]
    assert (member.work_dir / "MODEL.DATA").read_text().split() == ["1.5", "2.5"]


def test_e300_runs_the_compositional_program(tmp_path: Path, summary_dir: Path) -> None:
    simulator = EclipseSimulator(
        _template(tmp_path),
        version="e300",
        executable=_fake_eclrun(tmp_path, _finishing(summary_dir)),
    )
    member = _member(tmp_path)
    simulator.run(member, KEYS)
    assert member.work_dir is not None
    assert (member.work_dir / "args").read_text().split() == ["e300", "MODEL.DATA"]


def test_errors_in_the_error_summary_fail_the_member_with_the_prt_messages(
    tmp_path: Path, summary_dir: Path
) -> None:
    # eclrun can exit 0 after a failed run; the .ECLEND error summary decides (as PET).
    simulator = EclipseSimulator(
        _template(tmp_path),
        executable=_fake_eclrun(tmp_path, _finishing(summary_dir, errors=1, prt=PRT_ERROR)),
    )
    with pytest.raises(SimulationError, match=r"1 error\(s\)(.|\n)*PERMX HAS TOO FEW VALUES"):
        simulator.run(_member(tmp_path), KEYS)


def test_a_run_without_an_error_summary_did_not_finish(tmp_path: Path) -> None:
    simulator = EclipseSimulator(
        _template(tmp_path), executable=_fake_eclrun(tmp_path, "echo started\n")
    )
    with pytest.raises(SimulationError, match="did not finish"):
        simulator.run(_member(tmp_path), KEYS)


def test_a_non_zero_exit_reports_the_log_tail(tmp_path: Path) -> None:
    simulator = EclipseSimulator(
        _template(tmp_path),
        executable=_fake_eclrun(tmp_path, "echo 'no licence available'\nexit 3\n"),
    )
    with pytest.raises(SimulationError, match=r"exited with 3(.|\n)*no licence available"):
        simulator.run(_member(tmp_path), KEYS)


def test_a_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    # The fake eclrun starts a child (as eclrun starts eclipse) that would outlive it.
    marker = tmp_path / "child_survived"
    script = f"(sleep 3; touch {marker}) &\nsleep 30\n"
    simulator = EclipseSimulator(
        _template(tmp_path), executable=_fake_eclrun(tmp_path, script), timeout_s=0.5
    )
    with pytest.raises(SimulationError, match="exceeded"):
        simulator.run(_member(tmp_path), KEYS)
    time.sleep(3.5)
    assert not marker.exists()  # the child was killed with its group: no held licence


def test_pruning_keeps_the_reports_and_the_summary(tmp_path: Path, summary_dir: Path) -> None:
    script = _finishing(summary_dir) + "touch MODEL.UNRST MODEL.EGRID MODEL.INIT\n"
    simulator = EclipseSimulator(
        _template(tmp_path), executable=_fake_eclrun(tmp_path, script), keep_files=False
    )
    member = _member(tmp_path)
    simulator.run(member, KEYS)
    assert member.work_dir is not None
    assert {p.name for p in member.work_dir.iterdir()} == {
        "MODEL.DATA",
        "MODEL.ECLEND",
        "MODEL.PRT",
        "MODEL.SMSPEC",
        "MODEL.UNSMRY",
        "eclrun.log",
    }


def test_licenses_cap_the_parallel_runs_and_options_are_validated(tmp_path: Path) -> None:
    template = _template(tmp_path)
    assert EclipseSimulator(template, licenses=3).max_parallel_runs == 3
    assert EclipseSimulator(template).max_parallel_runs is None
    for bad, message in (
        ({"licenses": 0}, "licenses"),
        ({"licenses": True}, "licenses"),
        ({"version": "e500"}, "version"),
        ({"timeout_s": 0}, "timeout_s"),
    ):
        with pytest.raises(ValueError, match=message):
            EclipseSimulator(template, **bad)
    with pytest.raises(ValueError, match="not found"):
        EclipseSimulator(tmp_path / "missing.mako")
    missing = EclipseSimulator(template, executable="no-such-eclrun")
    with pytest.raises(SimulationError, match="not found on PATH"):
        missing.run(_member(tmp_path), KEYS)


def test_report_parsers_on_real_formats(tmp_path: Path) -> None:
    eclend = tmp_path / "RUN.ECLEND"
    eclend.write_text(ECLEND.format(errors=0))
    assert error_summary(eclend) == {
        "Comments": 1,
        "Warnings": 10,
        "Problems": 0,
        "Errors": 0,
        "Bugs": 0,
    }
    assert error_summary(tmp_path / "missing.ECLEND") is None
    prt = tmp_path / "RUN.PRT"
    prt.write_text(PRT_ERROR * 5)
    messages = prt_errors(prt)
    assert len(messages) == 3
    assert messages[0].startswith("@--ERROR  AT TIME")
    assert "200 EXPECTED, 199 READ" in messages[0]


def test_the_variant_is_not_the_software_version(tmp_path: Path) -> None:
    """Regression: ``version = "e100"`` (the variant) was stored as ``simulator.version``,
    which services call as the optional software-version capability, so validating any
    Eclipse configuration raised ``TypeError: 'str' object is not callable``."""
    import json

    from apex.services.identity import _simulator_version

    simulator = EclipseSimulator(_template(tmp_path), version="e300")
    assert simulator.variant == "e300"
    assert getattr(simulator, "version", None) is None
    options = {"deck_template_file": str(_template(tmp_path)), "version": "e300"}
    assert _simulator_version("eclipse", json.dumps(options)) is None
