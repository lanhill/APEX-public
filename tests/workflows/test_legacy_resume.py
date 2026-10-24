"""Run directories saved by the old drivers still resume to the same result.

The fixtures in ``tests/data/legacy_runs/<case>/`` were written by the two old drivers
(``state.json`` with ``driver = "schedule" | "iterative"``) and stopped mid-run. Each holds:
- the case inputs;
- the run directory, with ``@CASE@`` in place of the absolute case path;
- ``expected.npz``: the posterior and history of an uninterrupted run of the old code.

Three format generations are covered:
- ES-MDA: current, and older (perturbations in every step file);
- GIES: current (``current_<k>``), and older (one overwritten ``accepted.npz``);
- GIES with a prior-only localization taper.

Record them (with the old drivers only) by ``APEX_RECORD_LEGACY=1 uv run pytest
tests/workflows/test_legacy_resume.py``. Values are compared to ``rtol = 1e-9``; set
``APEX_GOLDEN_EXACT=1`` to require bit-identity (on the recording machine).
"""

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest

from apex.services import composition

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "legacy_runs"
PLACEHOLDER = "@CASE@"
TANH = 'nonlinearity = "tanh"'
ESMDA = (
    'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
    'strategy = { name = "approx", options = { energy = 1.0 } }'
)
GIES = (
    'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-9 }\nmax_iterations = 12"
)
LOCALIZED_GIES = (
    f'{GIES}\nlocalization = {{ name = "adaptive_prior", options = {{ taper = "soft" }} }}'
)

#: case -> (method, simulator options, steps before the stop, rewrite into the older format)
CASES: dict[str, tuple[str, str, int, str | None]] = {
    "esmda": (ESMDA, "", 2, None),
    "esmda_older": (ESMDA, "", 2, "schedule"),
    # Members fail at iterations 4, 5 and 6: the stop at 5 falls between two drops.
    "gies": (GIES, f"{TANH}\nfail_above = 2.0", 5, None),
    "gies_older": (GIES, f"{TANH}\nfail_above = 2.0", 5, "iterative"),
    "gies_adaptive_prior": (LOCALIZED_GIES, TANH, 2, None),
}


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _older_format(run_dir: Path, driver: str) -> None:
    """Rewrite a run directory of the last old version as the version before it saved it."""
    checkpoints = run_dir / "checkpoints"
    saved = dict(np.load(checkpoints / "perturbed_observations.npz"))
    (checkpoints / "perturbed_observations.npz").unlink()
    state = json.loads((run_dir / "state.json").read_text())
    by_member = dict(
        zip(saved["member_ids"].tolist(), saved["perturbed_observations"].T, strict=True)
    )
    if driver == "schedule":
        # Every step file carried the scoring perturbations of its members.
        for path in checkpoints.glob("iter_*.npz"):
            arrays = dict(np.load(path))
            arrays["scoring_observations"] = np.column_stack(
                [by_member[m] for m in arrays["member_ids"].tolist()]
            )
            np.savez(path, **arrays)
        return
    # One overwritten `accepted` checkpoint with the perturbations; no `current`.
    current = dict(np.load(checkpoints / f"{state['current']}.npz"))
    for path in checkpoints.glob("current_*.npz"):
        path.unlink()
    del state["current"]
    accepted = dict(np.load(checkpoints / f"{state['checkpoint']}.npz"))
    columns = [accepted["member_ids"].tolist().index(m) for m in current["member_ids"]]
    legacy = {
        "state": accepted["state"][:, columns],
        "predicted": accepted["predicted"][:, columns],
        "member_ids": current["member_ids"],
        "perturbed_observations": np.column_stack(
            [by_member[m] for m in current["member_ids"].tolist()]
        ),
    }
    if "reference_prediction" in current:
        legacy["reference_prediction"] = current["reference_prediction"]
    np.savez(checkpoints / "accepted.npz", **legacy)
    state["checkpoint"] = "accepted"
    (run_dir / "state.json").write_text(json.dumps(state))


def _record(linear_case: CaseFactory, name: str) -> None:
    method, simulator, steps, older = CASES[name]
    whole = linear_case(
        name=f"{name}_whole", ensemble_size=24, method=method, simulator_extra=simulator
    )
    reference = composition.start(whole.config)
    case = linear_case(name=name, ensemble_size=24, method=method, simulator_extra=simulator)
    composition.start(case.config, max_steps=steps)
    run_dir = case.directory / "runs" / name
    if older is not None:
        _older_format(run_dir, older)
    target = FIXTURES / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(case.directory, target, ignore=shutil.ignore_patterns("simulations", "*.jsonl"))
    snapshot = target / "runs" / name / "config.json"
    snapshot.write_text(snapshot.read_text().replace(str(case.directory), PLACEHOLDER))
    np.savez(
        target / "expected.npz",
        posterior=reference.posterior,
        member_ids=reference.member_ids,
        data_mismatch_means=np.array([r.data_mismatch_mean for r in reference.history]),
    )


def _means(expected: Any) -> Any:  # noqa: ANN401 (an NpzFile)
    """The recorded mean data mismatches (fixtures before 2026-09-30 call them misfit_means)."""
    key = "data_mismatch_means" if "data_mismatch_means" in expected.files else "misfit_means"
    return expected[key]


@pytest.mark.parametrize("name", sorted(CASES))
def test_old_run_directories_resume_to_the_old_result(
    linear_case: CaseFactory, tmp_path: Path, name: str
) -> None:
    if os.environ.get("APEX_RECORD_LEGACY") == "1":
        _record(linear_case, name)
        return
    source = FIXTURES / name
    assert source.exists(), f"no legacy fixture {source}; record it (see module docstring)"
    case_dir = tmp_path / name
    shutil.copytree(source, case_dir)
    snapshot = case_dir / "runs" / name / "config.json"
    snapshot.write_text(snapshot.read_text().replace(PLACEHOLDER, str(case_dir)))

    resumed = composition.resume(case_dir / "runs" / name)

    with np.load(source / "expected.npz") as expected:
        np.testing.assert_array_equal(resumed.member_ids, expected["member_ids"])
        data_mismatches = np.array([r.data_mismatch_mean for r in resumed.history])
        if os.environ.get("APEX_GOLDEN_EXACT") == "1":
            np.testing.assert_array_equal(resumed.posterior, expected["posterior"])
            np.testing.assert_array_equal(data_mismatches, _means(expected))
        np.testing.assert_allclose(resumed.posterior, expected["posterior"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(data_mismatches, _means(expected), rtol=1e-9)


def test_diagnostics_of_a_legacy_resume_are_saved_under_their_iteration(tmp_path: Path) -> None:
    # The old drivers checkpointed before the next update, so the restore runs it; its
    # diagnostics belong to the iteration it produces (3 here: stopped after step 2).
    case_dir = tmp_path / "esmda"
    shutil.copytree(FIXTURES / "esmda", case_dir)
    snapshot = case_dir / "runs" / "esmda" / "config.json"
    config = json.loads(snapshot.read_text().replace(PLACEHOLDER, str(case_dir)))
    config["output"]["diagnostics"] = ["step"]
    snapshot.write_text(json.dumps(config))

    composition.resume(case_dir / "runs" / "esmda")

    results = composition.load_results(case_dir / "runs" / "esmda")
    assert results.diagnostics_iterations() == [3, 4]
    for k in (3, 4):
        step = results.iterate(k).state["x"] - results.iterate(k - 1).state["x"]
        np.testing.assert_allclose(results.diagnostics(k)[0]["method.step"], step, atol=1e-12)
