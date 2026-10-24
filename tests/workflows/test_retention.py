"""Retention of simulator work directories (``[output] keep_simulations``)."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

#: Every run writes a work directory; member 3 fails in the prior (dropped afterwards).
EXTRA = "write_files = true\nfail_members = [3]"
FAILURES = "[failures]\nmin_success_fraction = 0.5"


def _kept(case: Case, name: str) -> set[str]:
    root = case.directory / "runs" / name / "simulations"
    if not root.exists():
        return set()
    return {f"{d.parent.name}/{d.name}" for d in root.glob("iter_*/*")}


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("failed", {"iter_0/member_3"}),
        ("last", {f"iter_4/member_{m}" for m in range(8) if m != 3}),
        (
            "failed_and_last",
            {"iter_0/member_3"} | {f"iter_4/member_{m}" for m in range(8) if m != 3},
        ),
        ("none", set()),
    ],
)
def test_each_policy_keeps_what_it_says(
    linear_case: CaseFactory, policy: str, expected: set[str]
) -> None:
    everything = linear_case(name="all", ensemble_size=8, simulator_extra=EXTRA, failures=FAILURES)
    pruned = linear_case(
        name=policy,
        ensemble_size=8,
        simulator_extra=EXTRA,
        failures=f'{FAILURES}\n\n[output]\nkeep_simulations = "{policy}"',
    )

    a, b = composition.start(everything.config), composition.start(pruned.config)

    assert len(_kept(everything, "all")) == 8 + 4 * 7  # the prior, then 4 steps of 7
    assert _kept(pruned, policy) == expected
    np.testing.assert_array_equal(a.posterior, b.posterior)  # results never depend on them
    if "failed" in policy:
        marker = pruned.directory / "runs" / policy / "simulations/iter_0/member_3/APEX_FAILED"
        assert "configured to fail" in marker.read_text()
