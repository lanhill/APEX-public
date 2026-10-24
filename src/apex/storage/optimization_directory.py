"""The layout of an optimization run directory.

::

    runs/<name>/
        config.json        the resolved configuration (resume reads it)
        history.jsonl      one row per accepted iterate (and per EPF update)
        checkpoint.npz     the optimizer's arrays at the last checkpoint
        checkpoint.json    its other values, and the random stream's state
        result.json        the final result, once the run has stopped
        pareto.json        a multi-objective run's front at its last checkpoint:
                           controls, objectives, violation, the representative, the
                           hypervolume and its reference point
        evaluations.jsonl  the journal of simulated evaluations (phase 2), one row each
        simulations/       the simulations' folders (``[output] keep_simulations``)
        run.lock           held while a process runs the optimization

A checkpoint is written to temporary files and renamed, so a crash leaves the previous one.
The journal is appended (and flushed to disk) after each batch of simulations: a resumed run
replays the evaluations it holds without simulating them again.
"""

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG = "config.json"
HISTORY = "history.jsonl"
CHECKPOINT_ARRAYS = "checkpoint.npz"
CHECKPOINT_VALUES = "checkpoint.json"
RESULT = "result.json"
EVALUATIONS = "evaluations.jsonl"
PARETO = "pareto.json"
#: The job service's folder of optimizations it placed (``<root>/optimizations/``).
SERVICE_FOLDER = "optimizations"
SIMULATIONS = "simulations"


def write_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Write the resolved configuration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CONFIG).write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def read_config(run_dir: Path) -> dict[str, Any]:
    """Read the resolved configuration."""
    data: dict[str, Any] = json.loads((run_dir / CONFIG).read_text(encoding="utf-8"))
    return data


def save_checkpoint(run_dir: Path, state: Mapping[str, object]) -> None:
    """Save a state of arrays and JSON values atomically.

    Arrays go to ``checkpoint.npz``; ``None``, numbers, booleans and strings to
    ``checkpoint.json`` (which also lists the arrays' keys).
    """
    # Typed as Any: numpy's stubs confuse **arrays with the allow_pickle keyword.
    arrays: dict[str, Any] = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
    values = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    for key, value in values.items():
        if not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError(f"checkpoint value {key!r} is a {type(value).__name__}")
    tmp_arrays = run_dir / (CHECKPOINT_ARRAYS + ".tmp")
    with tmp_arrays.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_values = run_dir / (CHECKPOINT_VALUES + ".tmp")
    tmp_values.write_text(json.dumps({"values": values, "arrays": sorted(arrays)}), "utf-8")
    os.replace(tmp_arrays, run_dir / CHECKPOINT_ARRAYS)
    os.replace(tmp_values, run_dir / CHECKPOINT_VALUES)


def load_checkpoint(run_dir: Path) -> dict[str, object] | None:
    """Return the last checkpoint's state, or ``None`` if there is none."""
    values_path = run_dir / CHECKPOINT_VALUES
    if not values_path.exists():
        return None
    meta = json.loads(values_path.read_text(encoding="utf-8"))
    state: dict[str, object] = dict(meta["values"])
    with np.load(run_dir / CHECKPOINT_ARRAYS) as arrays:
        missing = set(meta["arrays"]) - set(arrays.files)
        if missing:
            raise ValueError(f"checkpoint arrays {sorted(missing)} are missing in {run_dir}")
        state.update({key: arrays[key] for key in meta["arrays"]})
    return state


def append_history(run_dir: Path, row: Mapping[str, Any]) -> None:
    """Append one row to ``history.jsonl``."""
    with (run_dir / HISTORY).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=float) + "\n")


def read_history(run_dir: Path) -> list[dict[str, Any]]:
    """Return the history's rows (without a last row still being written).

    Raises
    ------
    ValueError
        If a row before the last is not JSON (a damaged history).
    """
    path = run_dir / HISTORY
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for n, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if n < len(lines) - 1:
                raise
            # A running optimization appends its rows: the last may be half written.
    return rows


def truncate_history(run_dir: Path, rows: int) -> None:
    """Keep the first ``rows`` rows (those up to the checkpoint a resume continues from)."""
    kept = read_history(run_dir)[:rows]
    (run_dir / HISTORY).write_text(
        "".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8"
    )


def read_result(run_dir: Path) -> dict[str, Any]:
    """Read the final result."""
    data: dict[str, Any] = json.loads((run_dir / RESULT).read_text(encoding="utf-8"))
    return data


def write_result(run_dir: Path, result: Mapping[str, Any]) -> None:
    """Write the final result."""
    (run_dir / RESULT).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")


def write_pareto(run_dir: Path, front: Mapping[str, Any]) -> None:
    """Write a multi-objective run's front, atomically (a reader never sees half of it)."""
    target = run_dir / PARETO
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(front, indent=1, default=float), encoding="utf-8")
    os.replace(temporary, target)


def read_pareto(run_dir: Path) -> dict[str, Any] | None:
    """Read a multi-objective run's front (``None`` for a single-objective run)."""
    path = run_dir / PARETO
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def append_evaluations(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append evaluations to the journal, durably (flushed and synced)."""
    if not rows:
        return
    with (run_dir / EVALUATIONS).open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, default=float) + "\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())


def read_evaluations(run_dir: Path, *, repair: bool = False) -> list[dict[str, Any]]:
    """Return the journal's rows, in order.

    A torn last line (a crash mid-write) is dropped; with ``repair`` the file is rewritten
    without it, so that rows appended next start on a line of their own.
    """
    path = run_dir / EVALUATIONS
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    if repair and (len(rows) != len(text.splitlines()) or (text and not text.endswith("\n"))):
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        os.replace(tmp, path)
    return rows


def simulation_dir(run_dir: Path, member_id: int) -> Path:
    """Return the folder of one simulated evaluation."""
    return run_dir / SIMULATIONS / f"eval_{member_id:06d}"
