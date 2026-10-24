"""LM-EnRML end to end on linear cases (driver; PET parity in tests/parity/).

Independent references: the per-member weighted least-squares fit of the perturbed data
(the minimum of each member's data mismatch, which the approx form approaches for a linear,
overdetermined model), and PET's damping rules recomputed from the run's own history.
"""

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_directory import RunDirectory

#: With the tanh model members whose state exceeds 2 fail (dropped mid-run).
FAILING = 'nonlinearity = "tanh"\nfail_above = 2.0'

LM = (
    'name = "lm_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
    "schedule = { initial = 1.0, data_mismatch_tolerance = 1e-10 }\nmax_iterations = 40"
)


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray
    variances: np.ndarray

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray: ...


CaseFactory = Callable[..., Case]


def test_lm_enrml_converges_to_the_least_squares_fit(linear_case: CaseFactory) -> None:
    case = linear_case(method=LM, ensemble_size=30)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    # lambda stays at 0.008 (1 -> 0.2 -> 0.04 -> 0.008) because PET reduces it only when the
    # data mismatch spread falls too, and near the fit it does not (the same with minimum = 0,
    # checked); with lambda > 0 each step leaves (1 + lambda) / (1 + lambda + s^2) of the
    # error along a singular direction, so the run stops on the data mismatch tolerance ~7e-6
    # from the fit.
    np.testing.assert_allclose(
        result.posterior, case.least_squares(perturbed), rtol=1e-4, atol=1e-5
    )
    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert all(b < a for a, b in itertools.pairwise(accepted))
    assert result.converged


def test_the_damping_follows_pets_rules(linear_case: CaseFactory) -> None:
    """Every lambda of the run, recomputed from its history by PET's LM rules."""
    method = LM.replace("initial = 1.0", "initial = 100.0").replace("1e-10", "1e-6")
    case = linear_case(method=method, ensemble_size=30, simulator_extra='nonlinearity = "tanh"')

    result = composition.start(case.config)

    history = result.history
    lam, std = 100.0, history[0].data_mismatch_std
    for record in history[1:]:
        assert record.damping == lam  # the lambda the trial ran with
        if record.accepted:
            if record.data_mismatch_std < std and lam > 0.01:  # PET's lambda_min
                lam /= 5
            std = record.data_mismatch_std
        else:
            lam *= 5
    assert any(not r.accepted for r in history), "the case should reject some trials"


@pytest.mark.parametrize("initial", ["1.0", '"auto"'])
def test_a_resumed_run_is_identical(linear_case: CaseFactory, initial: str) -> None:
    method = LM.replace("initial = 1.0", f"initial = {initial}")
    whole = linear_case(name="whole", method=method, ensemble_size=20)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=20)
    run_dir = case.directory / "runs" / "parts"
    composition.start(case.config, max_steps=3)

    resumed = composition.resume(run_dir)

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.data_mismatch_mean for r in resumed.history] == [
        r.data_mismatch_mean for r in uninterrupted.history
    ]


def test_rejections_in_a_row_stop_the_run(linear_case: CaseFactory) -> None:
    # A huge first step on the tanh model overshoots; one rejection is allowed in a row.
    method = LM.replace("initial = 1.0", "initial = 1e-6").replace(
        "max_iterations = 40", "max_iterations = 40\noptions = { max_inner_iterations = 1 }"
    )
    case = linear_case(method=method, ensemble_size=20, simulator_extra='nonlinearity = "tanh"')

    result = composition.start(case.config)

    assert result.converged
    assert not result.history[-1].accepted
    state = json.loads((case.directory / "runs" / "case" / "state.json").read_text())
    assert state["algorithm"]["trials"] == 1


FULL = (
    LM.replace('name = "approx"', 'name = "full"')
    .replace("1e-10", "1e-12")
    .replace("max_iterations = 40", "max_iterations = 80")
)


def test_the_full_analysis_converges_to_each_members_rml_solution(
    linear_case: CaseFactory,
) -> None:
    """Independent reference: with the prior term, LM-EnRML minimises each member's RML
    objective (m - m_pr,j)^T C_p^-1 (m - m_pr,j) + (d_j - G m)^T C_d^-1 (d_j - G m). With 30
    members for 3 unknowns and energy 1, Am Am^T is exactly the inverse of the prior sample
    covariance C_p, and the minimiser is m*_j = m_pr,j + C_p G^T (G C_p G^T + C_d)^-1
    (d_j - G m_pr,j), whatever lambda did on the way. This needs N - 1 >= n_state (so the
    sample covariance is invertible), energy 1 (no truncation of Am or of the data
    anomalies) and G of full column rank. The case has one state variable, so per-variable
    scaling and truncation are not exercised here (the parity test covers both).
    """
    case = linear_case(method=FULL, ensemble_size=30)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    prior = run.load_arrays("iter_0")["state"]
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    g, c_d = case.forward, np.diag(case.variances)
    c_p = np.cov(prior)
    gain = c_p @ g.T @ np.linalg.inv(g @ c_p @ g.T + c_d)
    expected = prior + gain @ (perturbed - g @ prior)
    np.testing.assert_allclose(result.posterior, expected, rtol=1e-6, atol=1e-8)
    assert result.converged


def test_the_full_analysis_resumes_and_survives_failed_members(linear_case: CaseFactory) -> None:
    method = FULL.replace("max_iterations = 80", "max_iterations = 8")
    whole = linear_case(name="whole", method=method, ensemble_size=24, simulator_extra=FAILING)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=24, simulator_extra=FAILING)
    composition.start(case.config, max_steps=3)

    resumed = composition.resume(case.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert uninterrupted.history[-1].n_members < 24  # members failed on the way


@pytest.mark.parametrize(
    ("name", "cls"),
    [
        ("full", "FullUpdate"),
        ("subspace", "SubspaceUpdate"),
        ("subspace2", "TransformUpdate"),
        ("margis", "MarginalISUpdate"),
    ],
)
def test_es_mda_refuses_the_prior_based_analyses(
    linear_case: CaseFactory, name: str, cls: str
) -> None:
    """PET's ES-MDA re-imposes the original prior with these (tests/parity/
    test_esmda_prior_term.py): not a valid ES-MDA, so APEX refuses them."""
    method = (
        'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        f'strategy = {{ name = "{name}" }}'
    )
    case = linear_case(method=method, ensemble_size=10)

    with pytest.raises(ConfigError, match=f"{cls} analysis is not supported"):
        composition.start(case.config)


@pytest.mark.parametrize(
    ("method_name", "strategy", "cls"),
    [
        ("esmda", "rlmmac", "RLMMACUpdate"),
        ("esmda", "local_rlmmac", "LocalRLMMACUpdate"),
        ("es", "rlmmac", "RLMMACUpdate"),
    ],
)
def test_es_mda_refuses_the_mean_model_analyses(
    linear_case: CaseFactory, method_name: str, strategy: str, cls: str
) -> None:
    """RLM-MAC needs the forecast at the mean state, which ES and ES-MDA never make. A study
    that switched GIES to ES-MDA by a patch inherited ``strategy = rlmmac`` (2026-09-30);
    it must be refused before any simulation, not at the first analysis."""
    options = "{ inflation = [4, 4, 4, 4] }" if method_name == "esmda" else "{}"
    method = f'name = "{method_name}"\noptions = {options}\nstrategy = {{ name = "{strategy}" }}'
    case = linear_case(method=method, ensemble_size=10)

    with pytest.raises(ConfigError, match=f"{cls} analysis is not supported"):
        composition.start(case.config)


@pytest.mark.parametrize("confirm", [True, False])
def test_heavy_initial_damping_does_not_stop_at_the_prior(
    linear_case: CaseFactory, confirm: bool
) -> None:
    """Tester finding 34: PET's defaults (lambda 100, tolerance 1 %) with the subspace
    analysis take a first step 1/101 of Gauss-Newton's, which improves by less than 1 %;
    PET's rule (``confirm_convergence = false``) stops there, at the prior. APEX's default
    accepts it, relaxes lambda and stops only after two small improvements in a row.
    """
    method = (
        'name = "lm_enrml"\nstrategy = { name = "subspace", options = { energy = 1.0 } }\n'
        f"schedule = {{ confirm_convergence = {str(confirm).lower()} }}\nmax_iterations = 40"
    )
    case = linear_case(method=method, ensemble_size=30)

    result = composition.start(case.config)

    accepted = [r for r in result.history if r.accepted]
    prior = result.history[0].data_mismatch_mean
    assert result.converged
    if confirm:
        assert len(accepted) > 3
        assert accepted[-1].data_mismatch_mean < 0.5 * prior
    else:
        assert len(accepted) == 2  # the prior and the one small step
        assert accepted[-1].data_mismatch_mean > 0.98 * prior
