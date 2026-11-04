"""A member stopped at a report date runs the whole deck's schedule up to it, bit for bit.

The early-start surrogates continue a member's run from the data of a run stopped at a
report date (``MemberInput.stop_at``); their gate takes the held-out runs' own values up to
that date for what a stopped run gives. Here OPM Flow runs the 3dBox example's truth once
whole and once stopped at 2022-07-01, its report dates in the deck's main text, and once
stopped there with the report dates moved into a schedule include (inlined by the cut):
each stopped run's summary at the dates up to the stop is the whole run's, exactly.

Opt-in (it runs OPM Flow three times, a few minutes):
``APEX_RUN_STOP_AT=1 uv run pytest -m integration tests/integration/test_stop_at_opm.py``.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey
from apex.simulation.opm_flow import OPMFlowSimulator

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production"
STOP = "2022-07-01"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
    pytest.mark.skipif(
        os.environ.get("APEX_RUN_STOP_AT") != "1", reason="set APEX_RUN_STOP_AT=1 to run"
    ),
]


def _keys() -> list[DataKey]:
    """The example's observed vectors at every observed date, and the injectors' rates."""
    rows = (EXAMPLE / "observations.csv").read_text().splitlines()[1:]
    observed = {(r.split(",")[0], r.split(",")[1]) for r in rows}
    times = sorted({t for t, _ in observed})
    datatypes = sorted({d for _, d in observed} | {f"WWIR:INJ{i}" for i in (1, 2, 3)})
    return [DataKey(t, d) for d in datatypes for t in times]


def _member(work_dir: Path, stop_at: str | None) -> MemberInput:
    truth = np.load(EXAMPLE / "truth.npz")
    state = {name: np.asarray(truth[name], dtype=np.float64) for name in ("permx", "poro")}
    work_dir.mkdir(parents=True)
    return MemberInput(0, state, work_dir=work_dir, stop_at=stop_at)


def _dates_in_an_include(folder: Path) -> Path:
    """The example's template with its report dates moved into ``dates.sch``."""
    shutil.copytree(EXAMPLE, folder)
    template = folder / "deck.mako"
    lines = template.read_text().splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines) if line.startswith("-- Monthly report times"))
    end = max(i for i, line in enumerate(lines) if line.strip() == "END")
    (folder / "dates.sch").write_text("".join(lines[first:end]))
    lines[first:end] = ["INCLUDE\n", " '${case_dir}/dates.sch' /\n", "\n"]
    template.write_text("".join(lines))
    return template


def test_a_stopped_run_is_the_whole_runs_prefix(tmp_path: Path) -> None:
    keys = _keys()
    early = [k for k in keys if k.time <= STOP]
    assert len(early) < len(keys)
    options = {"template_values": {"permx_cap": 8.0}, "timeout_s": 1800.0}
    whole = OPMFlowSimulator(deck_template_file=EXAMPLE / "deck.mako", **options)  # type: ignore[arg-type]
    full = whole.run(_member(tmp_path / "whole", None), keys)
    expected = full[[keys.index(k) for k in early]]
    stopped = whole.run(_member(tmp_path / "stopped", STOP), early)
    np.testing.assert_array_equal(stopped, expected)
    template = _dates_in_an_include(tmp_path / "included_case")
    included = OPMFlowSimulator(deck_template_file=template, **options)  # type: ignore[arg-type]
    cut = included.run(_member(tmp_path / "included", STOP), early)
    np.testing.assert_array_equal(cut, expected)
    deck = next((tmp_path / "included").glob("*.DATA")).read_text()
    assert "the INCLUDE of dates.sch inlined" in deck
    assert deck.rstrip().endswith("END")
