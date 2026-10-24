"""Golden outputs of the history-matching workflow (the refactor safety net).

Eight representative runs cover:
- ES-MDA and GIES, each with and without localization;
- wavelet compression;
- failed members and rejected trials;
- a resumed run.

Each run's posterior, member ids, history and per-forecast data mismatches, and the set of
result files it saved, are compared with references recorded from the code before
the generic driver.

- **Tolerance:** bit-exact results differ between CPU types (AVX2 and AVX-512 round
  some sums differently), so values are compared to ``RTOL``. The set of result files
  must match exactly.
- **Exact check:** with ``APEX_GOLDEN_EXACT=1`` the SHA-256 digests must match too.
  Use it on the machine that recorded the references.
- **Recording:** ``APEX_RECORD_GOLDEN=1 uv run pytest tests/workflows/test_golden_outputs.py``
  rewrites the references. Only record when a change of results is intended.
"""

import hashlib
import importlib.util
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import pytest

from apex.domain.results import HistoryMatchResult
from apex.services import composition
from apex.storage.run_directory import RunDirectory

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden"


def _load_compression_cases() -> ModuleType:
    # The compressed-data case builders of test_data_compression_e2e.py (tests/ is not a
    # package, so load the module from its file).
    path = Path(__file__).with_name("test_data_compression_e2e.py")
    spec = importlib.util.spec_from_file_location("_compression_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compression = _load_compression_cases()
#: Relative tolerance across CPU types (measured differences ~1e-16 per step).
RTOL = 1e-9
#: Result files compared by name (bookkeeping files such as state.json are not).
RESULT_PREFIXES = (
    "iter_",
    "trial_",
    "data_mismatch_",
    "perturbed_observations",
    "posterior",
    "truth",
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

TANH = 'nonlinearity = "tanh"'
ESMDA = (
    'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)
#: The references were recorded with PET-fork's halving after an accepted step, the
#: default until 2026-10-13: ``reduction = 0.5`` is that rule bit for bit.
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9, reduction = 0.5 }\nmax_iterations = 12"
)


def _localized(method: str, localization: str) -> str:
    return f'{method}\nlocalization = {{ name = "{localization}", options = {{ taper = "soft" }} }}'


LINEAR_CASES: dict[str, dict[str, str]] = {
    "esmda": {"method": ESMDA},
    "esmda_adaptive": {"method": _localized(ESMDA, "adaptive_per_update")},
    "esmda_adaptive_prior": {"method": _localized(ESMDA, "adaptive_prior")},
    "esmda_failures": {"method": ESMDA, "simulator_extra": "fail_members = [5]"},
    "gies_failures_trials": {
        "method": GIES,
        "simulator_extra": f"fail_members = [3, 11]\n{TANH}",
        "failures": "[output]\nsave_rejected_trials = true",
    },
    "gies_adaptive_prior": {"method": _localized(GIES, "adaptive_prior"), "simulator_extra": TANH},
    "gies_resumed": {"method": GIES, "simulator_extra": TANH},
    # Members fail as iterates move (at iterations 4, 5, 6 and 11): the accepted iterate
    # loses them too and the mean model is re-forecast.
    "gies_midrun_failures": {"method": GIES, "simulator_extra": f"{TANH}\nfail_above = 2.0"},
    "gies_midrun_failures_resumed": {
        "method": GIES,
        "simulator_extra": f"{TANH}\nfail_above = 2.0",
    },
}
COMPRESSED_CASES = {
    "esmda_wavelet": compression.ESMDA,
    "gies_wavelet_raw": compression.GIES.replace("schedule = { ", "schedule = { reduction = 0.5, "),
}


def _summary(run_dir: Path, result: HistoryMatchResult) -> dict[str, np.ndarray]:
    run = RunDirectory(run_dir)
    history = run.history_dicts()
    data_mismatches = [
        run.load_arrays(f"data_mismatch_{r['iteration']}")["data_mismatch"] for r in history
    ]
    files = sorted(
        p.stem for p in run.checkpoints.glob("*.npz") if p.stem.startswith(RESULT_PREFIXES)
    )
    return {
        "posterior": result.posterior,
        "member_ids": result.member_ids,
        "data_mismatch_means": np.array([r["data_mismatch_mean"] for r in history]),
        "accepted": np.array([r["accepted"] for r in history]),
        "data_mismatches": np.concatenate(data_mismatches),
        "files": np.array(json.dumps(files)),
    }


def _digest(summary: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(summary):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(summary[key]).tobytes())
    return digest.hexdigest()


#: Names the references were recorded under, before the rename of 2026-09-30:
#: their values are compared unchanged, so the rename is shown to change no result.
OLD_KEYS = {"data_mismatch_means": "misfit_means", "data_mismatches": "misfits"}


def _old_names(summary: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``summary`` under the names of references recorded before the rename."""
    files = [
        f"misfit_{f.removeprefix('data_mismatch_')}" if f.startswith("data_mismatch_") else f
        for f in json.loads(str(summary["files"]))
    ]
    renamed = {OLD_KEYS.get(k, k): v for k, v in summary.items()}
    return {**renamed, "files": np.array(json.dumps(sorted(files)))}


def _check(name: str, summary: dict[str, np.ndarray]) -> None:
    path = GOLDEN / f"workflow_{name}.npz"
    if os.environ.get("APEX_RECORD_GOLDEN") == "1":
        GOLDEN.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {**summary, "digest": np.array(_digest(summary))}
        np.savez(path, **payload)
        return
    assert path.exists(), f"no golden reference {path}; record it (see module docstring)"
    with np.load(path) as reference:
        if "misfit_means" in reference.files:
            summary = _old_names(summary)
            keys = ("posterior", "misfit_means", "misfits")
        else:
            keys = ("posterior", "data_mismatch_means", "data_mismatches")
        assert json.loads(str(summary["files"])) == json.loads(str(reference["files"]))
        for key in ("member_ids", "accepted"):
            np.testing.assert_array_equal(summary[key], reference[key])
        for key in keys:
            np.testing.assert_allclose(summary[key], reference[key], rtol=RTOL, atol=1e-12)
        if os.environ.get("APEX_GOLDEN_EXACT") == "1":
            assert _digest(summary) == str(reference["digest"]), f"{name}: not bit-identical"


@pytest.mark.parametrize("name", sorted(LINEAR_CASES))
def test_linear_workflow_matches_its_golden_outputs(linear_case: CaseFactory, name: str) -> None:
    kwargs = LINEAR_CASES[name]
    case = linear_case(name=name, ensemble_size=24, **kwargs)
    run_dir = case.directory / "runs" / name
    if name.endswith("_resumed"):
        composition.start(case.config, max_steps=2)
        result = composition.resume(run_dir)
    else:
        result = composition.start(case.config)
    _check(name, _summary(run_dir, result))


@pytest.mark.parametrize("name", sorted(COMPRESSED_CASES))
def test_compressed_workflow_matches_its_golden_outputs(tmp_path: Path, name: str) -> None:
    section = compression._wavelet_section()
    if name.endswith("_raw"):
        section += "\n\n[output]\nsave_raw_forecasts = true"
    config = compression._write_case(
        tmp_path / name, transform=section, method=COMPRESSED_CASES[name]
    )
    result = composition.start(config)
    _check(name, _summary(tmp_path / name / "runs" / "case", result))
