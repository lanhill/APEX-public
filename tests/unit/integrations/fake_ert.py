"""A stand-in for ``ert ensemble_experiment`` in tests: no ERT needed.

Called as ``fake_ert.py --logdir DIR ensemble_experiment CONFIG`` in the configuration's
folder. For each realization of the generated batch, it reads the field ``X`` that APEX
wrote into the run path, and writes ``pred.npy`` = ``forward @ x`` (``forward.npz`` next
to the configuration), as a deterministic forward model. Realizations listed in
``FAIL_REALIZATIONS`` (a file next to the configuration, removed by the first batch that
reads it) are left without results.
"""

import sys
from pathlib import Path

import numpy as np

from apex.simulation.grdecl import read_grdecl


def main(argv: list[str]) -> int:
    """Run the batch named by ``argv`` (see the module docstring)."""
    if argv[0] != "--logdir" or argv[2] != "ensemble_experiment":
        raise SystemExit(f"unexpected arguments {argv}")
    config = Path(argv[3])
    settings = {}
    for line in config.read_text().splitlines():
        words = line.split()
        if words and not words[0].startswith("--"):
            settings[words[0]] = " ".join(words[1:])
    forward = np.load(config.parent / "forward.npz")["matrix"]
    failing = config.parent / "FAIL_REALIZATIONS"
    fail = set()
    if failing.exists():  # in this batch only
        fail = {int(i) for i in failing.read_text().split()}
        failing.unlink()
    Path(argv[1]).mkdir(parents=True, exist_ok=True)
    for i in range(int(settings["NUM_REALIZATIONS"])):
        run_path = Path(settings["RUNPATH"].replace("<IENS>", str(i)))
        if i in fail:
            continue
        x = read_grdecl(run_path / "x.grdecl", "X")
        np.save(run_path / "pred.npy", forward @ x)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
