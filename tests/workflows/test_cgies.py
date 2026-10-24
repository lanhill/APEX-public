"""C-GIES end to end on the linear cases of conftest.py.

Independent references: a feasible box keeps a run feasible and finishes as unconstrained
GIES does in data mismatch; the per-source localization resumes bit for bit; and an
infeasible member stops the run with the cause and the fixes.
"""

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
import tomli_w

from apex.algorithms.data_assimilation.gmac.constraints import ConstraintViolation
from apex.services import composition
from apex.services.policy import named
from apex.storage.run_directory import RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

BOX = {"function": {"name": "box", "options": {"bounds": {"x": [-6.0, 6.0]}}}}
HISTOGRAM = {
    "function": {
        "name": "histogram",
        "options": {"variable": "x", "range": [-3.0, 3.0], "bins": 3,
                    "reference_file": "truth.npz"},
    },
    "weight": {"rule": "trace", "w": 0.5},
}  # fmt: skip


def _configure(case: Case, constraints: list[dict[str, Any]], **method: dict[str, str]) -> None:
    raw = tomllib.loads(case.config.read_text())
    raw["method"] = {
        "name": "gies",
        "strategy": {"name": "gmac", "options": {"constraints": constraints}},
        "schedule": {"data_mismatch_tolerance": 1e-6},
        "max_iterations": 6,
        **method,
    }
    raw["metrics"] = [
        {
            "name": "constraint_cost",
            "label": f"constraint_{k}",
            "options": {"function": c["function"]},
        }
        for k, c in enumerate(constraints)
    ]
    case.config.write_text(tomli_w.dumps(raw))


def test_a_constrained_gies_run_fits_the_data_and_records_the_constraints(
    linear_case: CaseFactory,
) -> None:
    case = linear_case(ensemble_size=20)
    _configure(case, [BOX, HISTOGRAM])

    result = composition.start(case.config)

    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert accepted[-1] < 0.5 * accepted[0]
    assert np.all(np.abs(result.posterior) < 6.0)  # the box held
    scores = RunDirectory(case.directory / "runs" / "case").load_arrays("data_mismatch_0")
    assert np.all(np.isfinite(scores["metric:constraint_0:cost"]))
    assert np.all(scores["metric:constraint_0:violations"] == 0)
    # the histogram's reference_file, relative, resolved in the metric's function too
    assert np.all(np.isfinite(scores["metric:constraint_1:cost"]))


def test_a_localized_constrained_run_resumes_identically(linear_case: CaseFactory) -> None:
    whole = linear_case(name="whole", ensemble_size=20)
    parts = linear_case(name="parts", ensemble_size=20)
    for case in (whole, parts):
        _configure(case, [BOX, HISTOGRAM], localization={"name": "adaptive_per_update"})
    uninterrupted = composition.start(whole.config)
    composition.start(parts.config, max_steps=2)

    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.data_mismatch_mean for r in resumed.history] == [
        r.data_mismatch_mean for r in uninterrupted.history
    ]


def test_an_infeasible_member_stops_the_run_with_the_cause_and_the_fixes(
    linear_case: CaseFactory,
) -> None:
    case = linear_case(ensemble_size=20)
    _configure(case, [{"function": {"name": "box", "options": {"bounds": {"x": [-0.5, 0.5]}}}}])

    with pytest.raises(ConstraintViolation, match="set lower/upper within the box"):
        composition.start(case.config)


def test_a_constrained_configuration_names_only_its_files(linear_case: CaseFactory) -> None:
    """The policy (and every web-UI user) sees the constraints' files as inputs
    and nothing undeclared, in gmac's options and in the metric's."""
    case = linear_case(ensemble_size=20)
    _configure(case, [BOX, HISTOGRAM])

    found = named(composition.load_config(case.config))

    kinds = {(kind, Path(value).name) for kind, _, value in found if "truth.npz" in value}
    assert kinds == {("input", "truth.npz")}
    assert not [item for item in found if item[0] == "undeclared"]
    # the case's truth_file, the strategy's histogram and the metric's
    plugins = [key for _, key, value in found if "truth.npz" in value and "reference_file" in key]
    assert len(plugins) == 2
