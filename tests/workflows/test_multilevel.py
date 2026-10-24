"""Multilevel ES-MDA end to end on linear cases (PET parity: tests/parity/)."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError, WorkflowError
from apex.services import composition
from apex.storage.run_directory import RunDirectory


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray
    observations: np.ndarray
    variances: np.ndarray

    def exact_posterior(self) -> tuple[np.ndarray, np.ndarray]: ...


CaseFactory = Callable[..., Case]


def _method(levels: str, energy: float = 0.98) -> str:
    return (
        'name = "multilevel_esmda"\n'
        f"options = {{ inflation = [4, 4, 4, 4], levels = [{levels}] }}\n"
        f'strategy = {{ name = "hybrid", options = {{ energy = {energy} }} }}'
    )


def test_one_level_is_es_mda_with_the_approx_update(linear_case: CaseFactory) -> None:
    """Same seed, same perturbations: the hybrid gain of one level is the approx gain."""
    single = linear_case(
        name="ml", method=_method("{ members = 60, weight = 1.0 }"), ensemble_size=60
    )
    esmda = linear_case(
        name="esmda",
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "approx", options = { energy = 0.98 } }',
        ensemble_size=60,
    )

    a, b = composition.start(single.config), composition.start(esmda.config)

    np.testing.assert_allclose(a.posterior, b.posterior, rtol=1e-10, atol=1e-12)


def test_levels_run_their_own_model_and_resume_identically(linear_case: CaseFactory) -> None:
    """Level 0 runs a biased coarse operator (its own forward_file, resolved against the
    configuration); level 1 the fine one."""
    levels = (
        '{ members = 8, weight = 0.3, forward_file = "coarse.npz" }, { members = 16, weight = 0.7 }'
    )
    whole = linear_case(name="whole", method=_method(levels), ensemble_size=24)
    parts = linear_case(name="parts", method=_method(levels), ensemble_size=24)
    for case in (whole, parts):
        np.savez(case.directory / "coarse.npz", matrix=1.1 * case.forward + 0.05)
    uninterrupted = composition.start(whole.config)
    composition.start(parts.config, max_steps=2)

    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    run = RunDirectory(whole.directory / "runs" / "whole")
    prior = run.load_arrays("iter_0")
    ids = prior["member_ids"]
    coarse = ids < 8
    expected = np.where(
        coarse[None, :],
        (1.1 * whole.forward + 0.05) @ prior["state"],
        whole.forward @ prior["state"],
    )
    np.testing.assert_allclose(prior["predicted"], expected, rtol=1e-12)
    kept = [r.data_mismatch_mean for r in uninterrupted.history]
    assert kept[-1] < kept[0] / 5


def test_the_hybrid_estimate_approaches_the_kalman_posterior(linear_case: CaseFactory) -> None:
    """With the same model at both levels and many members (1000 + 1000), the posterior
    tends to the exact one. Monte Carlo errors with 2000 members: the mean's is
    sqrt(var / N) <= 0.006, so 0.03 is ~5 sigma; a covariance entry's is about
    sqrt(2 / N) var <= 0.002 (largest posterior variance ~0.07), so 0.01 is ~5 sigma."""
    case = linear_case(
        method=_method("{ members = 1000, weight = 0.5 }, { members = 1000, weight = 0.5 }", 1.0),
        ensemble_size=2000,
    )

    result = composition.start(case.config)

    mean, covariance = case.exact_posterior()
    np.testing.assert_allclose(result.posterior.mean(axis=-1), mean, atol=0.03)
    np.testing.assert_allclose(np.cov(result.posterior), covariance, atol=0.01)


def test_members_are_dropped_within_their_level(linear_case: CaseFactory) -> None:
    levels = "{ members = 3, weight = 0.5 }, { members = 5, weight = 0.5 }"
    ok = linear_case(
        name="ok",
        method=_method(levels),
        ensemble_size=8,
        simulator_extra="fail_members = [7]",
        failures="[failures]\nmin_success_fraction = 0.5",
    )
    result = composition.start(ok.config)
    assert result.posterior.shape[-1] == 7

    too_few = linear_case(
        name="few",
        method=_method(levels),
        ensemble_size=8,
        simulator_extra="fail_members = [0, 1]",
        failures="[failures]\nmin_success_fraction = 0.5",
    )
    with pytest.raises(WorkflowError, match="every level needs at least 2"):
        composition.start(too_few.config)


def test_the_levels_must_add_up_to_the_ensemble(linear_case: CaseFactory) -> None:
    case = linear_case(method=_method("{ members = 5, weight = 1.0 }"), ensemble_size=8)

    with pytest.raises(ConfigError, match="levels have 5 members"):
        composition.start(case.config)


def test_a_level_without_members_is_named(linear_case: CaseFactory) -> None:
    """``levels = [{}]`` said "levels have 0 members" (the release tester, 2026-10-19)."""
    case = linear_case(method=_method("{ weight = 1.0 }"), ensemble_size=8)

    with pytest.raises(ConfigError, match="level 0 needs members"):
        composition.start(case.config)
