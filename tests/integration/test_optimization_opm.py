"""PET's popt tutorial (3WELL) with OPM Flow: NPV through simulations, journal and resume.

The example ``examples/optimization/3well/enopt.toml`` runs one EnOpt
iteration with 4 members; a resume then adds one more. Every simulation is journaled once,
and the resume's restart of the iteration is answered from the journal.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from apex.services import optimization
from apex.storage import optimization_directory as layout

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "optimization" / "3well"


def test_enopt_raises_the_3well_npv_and_resumes_from_the_journal(tmp_path: Path) -> None:
    case = tmp_path / "3well"
    shutil.copytree(EXAMPLE, case)
    config = case / "enopt.toml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("ne = 10, maxiter = 5", "ne = 4, maxiter = 5")
    config.write_text(text, encoding="utf-8")

    first = optimization.optimize(config, max_iterations=1)
    assert not first.finished
    history = layout.read_history(first.run_dir)
    start, step = history[0], history[-1]
    # obj_scaling = -1e6: the objective is -NPV in millions, so a lower value is a higher NPV.
    assert np.isfinite(start["objective"])
    assert start["objective"] < 0
    assert step["objective"] < start["objective"]
    journal = layout.read_evaluations(first.run_dir)
    assert all(row["value"] is not None for row in journal)
    assert len({row["key"] for row in journal}) == len(journal)

    resumed = optimization.resume(first.run_dir, max_iterations=1)
    rows = layout.read_evaluations(resumed.run_dir)
    assert rows[: len(journal)] == journal  # the first invocation's evaluations, kept
    assert len(layout.read_history(resumed.run_dir)) == len(history) + 1
    # keep_simulations = "failed": the successful simulations' folders are gone.
    assert not any((resumed.run_dir / layout.SIMULATIONS).glob("eval_*/*.DATA"))
