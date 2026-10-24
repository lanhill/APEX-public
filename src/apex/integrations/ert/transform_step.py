r"""The forward-model step ``APEX_TRANSFORM``: APEX's data transforms in ERT.

An exported case with ``[[data_transforms]]`` has ERT assimilate the transformed
data. After the simulator, this step reads the summary at the case's raw keys and applies
the configuration's transforms, fitted to its observations exactly as APEX fits them, and
writes the result as the ``GEN_DATA`` ``APEX_TRANSFORMED`` (one value per line).

    python -m apex.integrations.ert.transform_step --export apex_export.json \
        --output apex_transformed_0.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from apex.simulation.deck import read_summary


def transform(export: Path, output: Path, run_path: Path = Path()) -> None:
    """Write the transformed data of the run path's summary into ``output``.

    Raises
    ------
    SimulationError
        If the summary is missing or lacks a raw key.
    """
    from apex.services import composition

    manifest = json.loads(export.read_text())
    config = composition.load_config(Path(manifest["config"]))
    workflow = composition.build_run(config).workflow
    raw = read_summary(run_path / manifest["eclbase"], workflow.raw_observations.keys)
    values = workflow.transform(raw[:, None])[:, 0]
    (run_path / output).write_text("".join(f"{float(v)!r}\n" for v in np.asarray(values)))


def main(argv: list[str] | None = None) -> int:
    """Run the step (see the module docstring)."""
    parser = argparse.ArgumentParser(description="APEX's data transforms in an ERT run path.")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    transform(args.export, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
