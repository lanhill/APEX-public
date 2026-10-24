"""LM's ``confirm_convergence`` in the loop (tester finding 34 and its numerics review).

A small improvement made with the damping above its minimum is accepted but is not yet
convergence; two in a row are, and a rejection in between starts the count again. Old
run snapshots resume with PET's rule (snapshot version 5).
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.algorithms.data_assimilation.protocols import DampingDecision
from apex.algorithms.data_assimilation.registries import damping_schemes
from apex.domain.results import HistoryMatchResult
from apex.services import composition
from apex.storage.run_directory import SNAPSHOT_VERSION, RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

#: tentative, rejected, tentative, tentative: the loop must not stop at the third trial.
SCRIPT = ("tentative", "rejected", "tentative", "tentative", "tentative")


@dataclass
class Scripted:
    """The decisions of ``SCRIPT``, in order, whatever the data mismatches."""

    initial: float = 1.0
    calls: list[str] = field(default_factory=list)

    def decide(
        self,
        damping: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float | None = None,
        spread: float | None = None,
    ) -> DampingDecision:
        kind = SCRIPT[len(self.calls)]
        self.calls.append(kind)
        if kind == "rejected":
            return DampingDecision(False, False, damping * 2)
        return DampingDecision(True, False, damping / 2, tentative=True)


@pytest.fixture
def scripted_scheme() -> Iterator[None]:
    damping_schemes.register("test_scripted")(Scripted)
    try:
        yield
    finally:
        damping_schemes._factories.pop("test_scripted")


def test_a_rejection_between_small_improvements_restarts_the_count(
    linear_case: CaseFactory, scripted_scheme: None
) -> None:
    case = linear_case(
        method='name = "lm_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
        'damping = { name = "test_scripted" }\nmax_iterations = 10',
        ensemble_size=20,
    )
    config = composition.load_config(case.config)
    result = composition.start_config(config, None)

    assert result.complete
    assert result.converged
    assert config.run.output_dir is not None
    history = json.loads((config.run.output_dir / "history.json").read_text())
    trials = [h["accepted"] for h in history if h["iteration"] > 0]
    assert trials == [True, False, True, True]  # not stopped at the third trial


SUBSPACE = (
    'name = "lm_enrml"\nstrategy = { name = "subspace", options = { energy = 1.0 } }\n'
    "max_iterations = 40"
)


@pytest.mark.parametrize("steps", [2, 3])
def test_a_run_resumed_after_a_small_improvement_is_identical(
    linear_case: CaseFactory, steps: int
) -> None:
    """The first trial (lambda 100) is a small improvement: the count is in the checkpoint."""
    whole = linear_case(name="whole", method=SUBSPACE, ensemble_size=20)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=SUBSPACE, ensemble_size=20)
    composition.start(case.config, max_steps=steps)

    resumed = composition.resume(case.directory / "runs" / "parts")

    dampings = [r.damping for r in uninterrupted.history if r.accepted][:3]
    assert dampings[1:] == [100.0, 20.0]  # the first trial was tentative: relaxed
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.data_mismatch_mean for r in resumed.history] == [
        r.data_mismatch_mean for r in uninterrupted.history
    ]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ({"name": "lm_enrml"}, {"name": "lm_enrml", "schedule": {"confirm_convergence": False}}),
        (
            {"name": "lm_enrml", "schedule": {"confirm_convergence": True}},
            {"name": "lm_enrml", "schedule": {"confirm_convergence": True}},
        ),
        (
            {"name": "gies", "damping": {"name": "lm"}},
            {"name": "gies", "damping": {"name": "lm", "options": {"confirm_convergence": False}}},
        ),
        (
            {"name": "chop", "options": {"inner": "lm_enrml"}},
            {
                "name": "chop",
                "options": {"inner": "lm_enrml"},
                "schedule": {"confirm_convergence": False},
            },
        ),
        ({"name": "gies"}, {"name": "gies"}),  # rlmmac
        ({"name": "gn_enrml"}, {"name": "gn_enrml"}),
        (
            {"name": "lm_enrml", "damping": {"name": "rlmmac"}},
            {"name": "lm_enrml", "damping": {"name": "rlmmac"}},
        ),
        ({"name": "esmda"}, {"name": "esmda"}),
    ],
)
def test_old_snapshots_keep_pets_stopping_rule(
    method: dict[str, object], expected: dict[str, object]
) -> None:
    raw = {"method": json.loads(json.dumps(method))}

    composition._migrate_lm_confirmation(raw)

    assert raw["method"] == expected


def test_a_version_4_snapshot_resumes_with_pets_rule(linear_case: CaseFactory) -> None:
    pets = SUBSPACE.replace(
        "max_iterations", "schedule = { confirm_convergence = false }\nmax_iterations"
    )
    reference = composition.start(linear_case(name="pet", method=pets, ensemble_size=20).config)
    ours = composition.start(linear_case(name="ours", method=SUBSPACE, ensemble_size=20).config)
    case = linear_case(name="old", method=SUBSPACE, ensemble_size=20)
    run_dir = case.directory / "runs" / "old"
    composition.start(case.config, max_steps=0)  # the prior forecast only
    assert SNAPSHOT_VERSION == 5
    (run_dir / "snapshot_version").write_text("4\n")

    resumed = composition.resume(run_dir)

    def means(result: HistoryMatchResult) -> list[float]:
        return [r.data_mismatch_mean for r in result.history]

    assert means(resumed) == means(reference)
    assert means(ours) != means(reference)  # the rules differ on this case
    assert RunDirectory(run_dir).snapshot_version() == 4


def test_rlmmac_confirms_only_when_asked() -> None:
    """GIES's scheme has the option off by default (its docstring says why)."""
    from apex.algorithms.data_assimilation import DampingSchedule

    assert DampingSchedule().confirm_convergence is False
    off = DampingSchedule().decide(4.0, 252.0, 251.8)
    assert (off.accepted, off.converged, off.tentative) == (True, True, False)
    on = DampingSchedule(confirm_convergence=True).decide(4.0, 252.0, 251.8)
    assert (on.accepted, on.converged, on.tentative) == (True, False, True)
    assert on.damping == pytest.approx(3.6)  # reduced as after any accepted trial
    halved = DampingSchedule(confirm_convergence=True, reduction=None).decide(4.0, 252.0, 251.8)
    assert halved.damping == 2.0  # PET-fork's division by the factor
    floor = DampingSchedule(confirm_convergence=True, minimum=4.0).decide(4.0, 252.0, 251.8)
    assert (floor.converged, floor.tentative) == (True, False)


def test_gies_with_heavy_damping_goes_on_when_confirming(linear_case: CaseFactory) -> None:
    """A GIES run started at a large damping stops at its first small step unless asked."""

    def run(name: str, confirm: bool) -> HistoryMatchResult:
        method = (
            'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
            'damping = { name = "rlmmac", options = { initial = 100.0, '
            f"confirm_convergence = {str(confirm).lower()} }} }}\nmax_iterations = 40"
        )
        return composition.start(linear_case(name=name, method=method, ensemble_size=20).config)

    off, on = run("off", False), run("on", True)

    assert off.converged
    assert on.converged
    accepted_off = [r for r in off.history if r.accepted]
    accepted_on = [r for r in on.history if r.accepted]
    assert len(accepted_on) > len(accepted_off)
    assert accepted_on[-1].data_mismatch_mean < accepted_off[-1].data_mismatch_mean


def test_a_fixed_threshold_with_too_few_members_is_refused_before_simulating(
    linear_case: CaseFactory,
) -> None:
    """The review of the fixed threshold (2026-10-20): ``3/sqrt(N)`` needs 10 members, and
    validation says so, as for ``adaptive_rows`` (#216)."""
    from apex.core.errors import ConfigError

    method = (
        'name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
        'localization = { name = "adaptive_prior", options = { threshold = "fixed" } }\n'
        "max_iterations = 2"
    )
    case = linear_case(method=method, ensemble_size=8)

    with pytest.raises(ConfigError, match="keeps nothing with 8 members"):
        composition.build_run(composition.load_config(case.config))
