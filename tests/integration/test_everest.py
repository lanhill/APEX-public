"""A whole Everest run on APEX's optimizer (opt-in: APEX_RUN_EVEREST=1).

Everest (ERT 26) runs ``optimization: algorithm: apex/enopt`` on three generic controls
whose forward model returns minus the squared distance to (0.5, 0.5, 0.5): Everest's own
loop, run paths, queue and storage, with APEX's EnOpt choosing the points. From 0.1 in
every control (objective -0.48), four iterations must come within 0.02 of the target's 0.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("APEX_RUN_EVEREST") != "1", reason="set APEX_RUN_EVEREST=1"),
]

FORWARD_MODEL = '''#!{python}
"""Minus the squared distance of point.json to (0.5, 0.5, 0.5)."""
import json
point = json.load(open("point.json"))
value = -sum((point[k] - 0.5) ** 2 for k in ("x", "y", "z"))
open("distance", "w").write(f"{{value:.15g}}\\n")
'''

CONFIG = """
controls:
  - name: point
    min: -1.0
    max: 1.0
    initial_guess: 0.1
    perturbation_magnitude: 0.05
    variables:
      - name: x
      - name: y
      - name: z
objective_functions:
  - name: distance
optimization:
  algorithm: apex/enopt
  max_iterations: 4
  options:
    ne: 8
    alpha: 0.2
model:
  realizations: [0]
install_jobs:
  - name: distance
    executable: jobs/distance.py
forward_model:
  - distance
environment:
  simulation_folder: sims
  output_folder: output
  random_seed: 123
"""


def test_everest_runs_apex_enopt_to_the_optimum(tmp_path: Path) -> None:
    everest = shutil.which("everest", path=str(Path(sys.executable).parent))
    if everest is None:
        pytest.skip("Everest is not installed (the ert extra)")
    (tmp_path / "jobs").mkdir()
    job = tmp_path / "jobs" / "distance.py"
    job.write_text(FORWARD_MODEL.format(python=sys.executable))
    job.chmod(0o755)
    (tmp_path / "config.yml").write_text(CONFIG)
    done = subprocess.run(
        [everest, "run", "config.yml"], cwd=tmp_path, capture_output=True, text=True,
        timeout=1800, check=False,
    )  # fmt: skip
    output = done.stdout + done.stderr
    assert done.returncode == 0, output[-3000:]
    assert "Optimization completed" in output, output[-3000:]
    # The last progress block's objective (Everest maximizes it).
    values = [float(v) for v in re.findall(r"^\s*distance: (-?[0-9.eE+-]+)\s*$", output, re.M)]
    assert values, output[-3000:]
    assert max(values) > -0.02
