"""APEX's cross-validation criterion for ERT runs (``APEX_CV``).

An exported case keeps a ``[validation]`` split's held-out data out of ERT's
``OBS_CONFIG``, in ``observations/held_out.csv``; ERT simulates them (their keys are in
``SUMMARY``) but never assimilates them. After each ERT iterate is simulated, the workflow
job ``APEX_CV`` scores them as APEX does (``cv_normalized_data_mismatch`` and the ``test``
counterpart: per realization, the mean of ``((d_obs - y) / sigma)^2``) into
``apex_cv.json``; at the end, ``APEX_CV_SELECT`` applies APEX's stopping rule to the
iterates' mean CV score (``select``: the minimum-CV iterate is the posterior; ``patience``:
also where the rule would have stopped; ERT's schedule runs on).

This module needs no ERT: the jobs in :mod:`.plugin` call it with ERT's run paths.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import apex.algorithms.validation  # noqa: F401  (registers the stopping rules)
from apex.core.errors import SimulationError
from apex.core.plugins import stopping_rules
from apex.domain.observations import DataKey
from apex.simulation.deck import read_summary

#: The file the jobs write, next to ``case.ert``.
RESULTS = "apex_cv.json"


@dataclass(frozen=True)
class HeldOut:
    """One role's held-out data: keys, observed values and standard deviations."""

    keys: list[DataKey]
    values: NDArray[np.float64]
    errors: NDArray[np.float64]


def held_out(case_dir: Path) -> dict[str, HeldOut]:
    """Return the exported case's held-out data by role (``cv``, ``test``); ``{}`` if none."""
    path = case_dir / "observations" / "held_out.csv"
    if not path.exists():
        return {}
    rows: dict[str, list[dict[str, str]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["role"], []).append(row)
    return {
        role: HeldOut(
            [
                DataKey(r["date"], f"{r['keyword']}:{r['well']}" if r["well"] else r["keyword"])
                for r in items
            ],
            np.array([float(r["value"]) for r in items]),
            np.array([float(r["error"]) for r in items]),
        )
        for role, items in rows.items()
    }


def score(case: Path, data: HeldOut) -> float:
    """Return one run's normalized mismatch of held-out data, ``mean(((d - y) / sigma)^2)``.

    Raises
    ------
    SimulationError
        If the run's summary cannot be read.
    """
    predicted = read_summary(case, data.keys)
    return float(np.mean(((data.values - predicted) / data.errors) ** 2))


def score_iteration(case_dir: Path, iteration: int, run_paths: Sequence[Path]) -> dict[str, Any]:
    """Score every run path of one ERT iteration and record it in ``apex_cv.json``.

    Run paths whose summary cannot be read (failed realizations) are left out.
    """
    manifest = json.loads((case_dir / "apex_export.json").read_text())
    data = held_out(case_dir)
    entry: dict[str, Any] = {}
    for role, values in data.items():
        members = {}
        for path in run_paths:
            try:
                members[int(_realization(path))] = score(path / manifest["eclbase"], values)
            except SimulationError:
                continue
        entry[role] = {
            "mean": float(np.mean(list(members.values()))) if members else None,
            "members": members,
        }
    results = _read(case_dir)
    results["iterations"][str(iteration)] = entry
    _write(case_dir, results)
    return entry


def select(case_dir: Path, rule: str = "select", patience: int = 2) -> dict[str, Any]:
    """Apply APEX's stopping rule to the scored iterates' mean CV; record the selection.

    Returns ``{"rule", "selected", "would_stop_at", "minimum_overall"}``: the minimum-CV
    iterate among those the rule saw (``patience`` stops seeing them after
    ``would_stop_at``, ``None`` if it would not have stopped), and the minimum over all.

    Raises
    ------
    ValueError
        If no iterate has CV scores.
    """
    results = _read(case_dir)
    scored = sorted(
        (int(k), v["cv"]["mean"])
        for k, v in results["iterations"].items()
        if v.get("cv") and v["cv"]["mean"] is not None
    )
    if not scored:
        raise ValueError(f"{case_dir} has no CV scores: is APEX_CV hooked at POST_SIMULATION?")
    options = {"patience": patience} if rule == "patience" else {}
    stopping = stopping_rules.create(rule, **options)
    would_stop = None
    for iteration, value in scored:
        # As in an APEX run: the rule sees the iterates up to where it stops, and its
        # selection is the minimum among those (ERT's schedule ran on regardless).
        if stopping.observe(iteration, value):
            would_stop = iteration
            break
    overall = min(scored, key=lambda item: item[1])[0]
    selection = {
        "rule": rule, "selected": stopping.selected, "would_stop_at": would_stop,
        "minimum_overall": overall,
    }  # fmt: skip
    results["selection"] = selection
    _write(case_dir, results)
    return selection


def _realization(path: Path) -> str:
    """Return the realization number in a run path ``.../realization-<i>/iter-<k>``."""
    for part in reversed(path.parts):
        if part.startswith("realization-"):
            return part.split("-", 1)[1]
    raise ValueError(f"{path} is not an ERT run path (no realization-<i>)")


def _read(case_dir: Path) -> dict[str, Any]:
    path = case_dir / RESULTS
    if path.exists():
        results: dict[str, Any] = json.loads(path.read_text())
        return results
    return {"written_by": "APEX_CV", "iterations": {}}


def _write(case_dir: Path, results: dict[str, Any]) -> None:
    (case_dir / RESULTS).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
