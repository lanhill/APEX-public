"""GN-EnRML and the subspace analysis end to end on linear cases (PET parity: tests/parity/).

Independent references: for a linear model the Gauss-Newton subspace iteration reaches, in
one full step, Evensen et al. (2019)'s subspace solution ``X = X_p + A_p S^T (S S^T + P E
E^T P)^+ (D - G X_p)`` (``S = L^-1 G A_p``, ``E = L^-1 D Pi``, ``P`` the projection on the
range of ``S``), computed here with a pseudo-inverse instead of the eigen-decomposition;
and PET's step-length rules recomputed from the run's own history.
"""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.algorithms.data_assimilation.kernels import anomalies
from apex.core.errors import WorkflowError
from apex.services import composition
from apex.storage.run_directory import RunDirectory

#: With the tanh model members whose state exceeds 2 fail (dropped mid-run).
FAILING = 'nonlinearity = "tanh"\nfail_above = 2.0'

GN = (
    'name = "gn_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
    "schedule = { data_mismatch_tolerance = 1e-10 }\nmax_iterations = 60"
)
#: One full Gauss-Newton step (gamma fixed at 1) in the weight space, no truncation.
SUBSPACE = (
    'name = "gn_enrml"\nstrategy = { name = "subspace", options = { energy = 1.0 } }\n'
    "schedule = { initial = 1.0, maximum = 1.0, factor = 1.0, data_mismatch_tolerance = 1e-10 }\n"
    "max_iterations = 5"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray
    variances: np.ndarray


CaseFactory = Callable[..., Case]


def test_one_gauss_newton_subspace_step_reaches_the_subspace_solution(
    linear_case: CaseFactory,
) -> None:
    """Needs a linear model, energy 1 and gamma 1; ``S`` has rank 3 < 8 data, so the
    projection ``P`` matters (and the five null singular values are dropped)."""
    case = linear_case(method=SUBSPACE, ensemble_size=30)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    prior = run.load_arrays("iter_0")["state"]
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    sigma = np.sqrt(case.variances)[:, None]
    a_p = anomalies(prior)
    s = case.forward @ a_p / sigma
    e = anomalies(perturbed) / sigma
    u = np.linalg.svd(s, full_matrices=False)[0][:, :3]
    projection = u @ u.T
    gain = a_p @ s.T @ np.linalg.pinv(s @ s.T + projection @ e @ e.T @ projection, rcond=1e-12)
    expected = prior + gain @ ((perturbed - case.forward @ prior) / sigma)
    np.testing.assert_allclose(result.posterior, expected, rtol=1e-8, atol=1e-10)
    assert result.converged


def test_the_step_length_follows_pets_rules(linear_case: CaseFactory) -> None:
    """Every gamma of the run, recomputed from its history by PET's GN rules."""
    method = GN.replace("1e-10", "1e-6")
    case = linear_case(method=method, ensemble_size=30, simulator_extra='nonlinearity = "tanh"')

    result = composition.start(case.config)

    history = result.history
    gamma, std, accepted = 0.2, history[0].data_mismatch_std, 0
    for record in history[1:]:
        assert record.damping == gamma  # the step length the trial ran with
        if record.accepted:
            if record.data_mismatch_std < std:
                gamma = gamma + (0.5 - gamma) * 2.0 ** (-(accepted + 1) / 1.5)
            std, accepted = record.data_mismatch_std, accepted + 1
        else:
            gamma /= 2.5
    assert any(not r.accepted for r in history), "the case should reject some trials"
    means = [r.data_mismatch_mean for r in history if r.accepted]
    assert all(b < a for a, b in itertools.pairwise(means))
    assert result.converged


@pytest.mark.parametrize("max_steps", [3, 7])
@pytest.mark.parametrize("strategy", ["approx", "subspace"])
def test_a_resumed_run_is_identical(
    linear_case: CaseFactory, strategy: str, max_steps: int
) -> None:
    """Full-length steps on the tanh model are rejected from the 3rd trial (subspace) or
    the 6th (approx) on, so stopping after 3 or 7 steps resumes before, between or after
    rejected retries (the subspace strategy's pending weights)."""
    method = (
        GN.replace('name = "approx"', f'name = "{strategy}"')
        .replace(
            "data_mismatch_tolerance = 1e-10",
            "initial = 1.0, maximum = 1.0, data_mismatch_tolerance = 1e-10",
        )
        .replace("60", "8")
    )
    extra = 'nonlinearity = "tanh"'
    whole = linear_case(name="whole", method=method, ensemble_size=20, simulator_extra=extra)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=20, simulator_extra=extra)
    composition.start(case.config, max_steps=max_steps)

    resumed = composition.resume(case.directory / "runs" / "parts")

    if max_steps == 7:  # the cut comes after a rejected trial
        assert any(not r.accepted for r in uninterrupted.history[: max_steps + 1])
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [(r.data_mismatch_mean, r.damping) for r in resumed.history] == [
        (r.data_mismatch_mean, r.damping) for r in uninterrupted.history
    ]


def test_lm_enrml_runs_the_subspace_analysis(linear_case: CaseFactory) -> None:
    method = (
        'name = "lm_enrml"\nstrategy = { name = "subspace", options = { energy = 1.0 } }\n'
        "schedule = { initial = 1.0, data_mismatch_tolerance = 1e-8 }\nmax_iterations = 40"
    )
    case = linear_case(method=method, ensemble_size=30)

    result = composition.start(case.config)

    means = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert len(means) > 2
    assert all(b < a for a, b in itertools.pairwise(means))
    assert result.converged


def test_the_subspace_analysis_refuses_to_drop_members_mid_run(linear_case: CaseFactory) -> None:
    method = GN.replace('name = "approx"', 'name = "subspace"').replace("60", "8")
    case = linear_case(method=method, ensemble_size=24, simulator_extra=FAILING)

    with pytest.raises(WorkflowError, match="cannot drop members after the prior forecast"):
        composition.start(case.config)


SUBSPACE2 = SUBSPACE.replace('name = "subspace", options = { energy = 1.0 }', 'name = "subspace2"')


def test_one_gauss_newton_transform_step_reaches_each_members_rml_solution(
    linear_case: CaseFactory,
) -> None:
    """Independent reference (Raanes et al. 2019): for a linear model one full step of the
    ensemble-transform IES from W = I gives each member's RML solution with the prior's
    sample covariance, m_j = x_j + C_p G^T (G C_p G^T + C_d)^-1 (d_j - G x_j), for any
    ensemble size (push-through identity; no rank condition), here 10 members."""
    case = linear_case(method=SUBSPACE2, ensemble_size=10)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    prior = run.load_arrays("iter_0")["state"]
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    g, c_d, c_p = case.forward, np.diag(case.variances), np.cov(prior)
    gain = c_p @ g.T @ np.linalg.inv(g @ c_p @ g.T + c_d)
    np.testing.assert_allclose(
        result.posterior, prior + gain @ (perturbed - g @ prior), rtol=1e-8, atol=1e-10
    )
    assert result.converged


@pytest.mark.parametrize("strategy", ["subspace2", "margis"])
def test_the_transform_analyses_run_and_resume_identically(
    linear_case: CaseFactory, strategy: str
) -> None:
    method = (
        GN.replace('name = "approx", options = { energy = 1.0 }', f'name = "{strategy}"')
        .replace(
            "data_mismatch_tolerance = 1e-10",
            "initial = 1.0, maximum = 1.0, data_mismatch_tolerance = 1e-10",
        )
        .replace("60", "8")
    )
    extra = 'nonlinearity = "tanh"'
    whole = linear_case(name="whole", method=method, ensemble_size=20, simulator_extra=extra)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=20, simulator_extra=extra)
    composition.start(case.config, max_steps=3)

    resumed = composition.resume(case.directory / "runs" / "parts")

    # The tanh model saturates: the fit halves the data mismatch, with every accepted step better.
    means = [r.data_mismatch_mean for r in uninterrupted.history if r.accepted]
    assert means[-1] < means[0] / 2
    assert all(b < a for a, b in itertools.pairwise(means))
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [(r.data_mismatch_mean, r.damping) for r in resumed.history] == [
        (r.data_mismatch_mean, r.damping) for r in uninterrupted.history
    ]
