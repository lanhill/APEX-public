"""LM-EnRML against PET's ``LMEnRML`` with the approx analysis (upstream ``e65fb10``).

The references (``tools/pet_reference/generate.py``): one damped step from PET's
``approx_update`` with injected prior, forecast and perturbations; and PET's own
``score_and_commit`` damping rules on scripted data mismatch sequences.
"""

import math
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import ApproxUpdate, DataErrorCovariance
from apex.algorithms.data_assimilation.lm_enrml import LMDampingSchedule, LMEnRML

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: Relative to the largest step component: SVD and summation order differ (as ES-MDA's).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
def test_the_damped_step_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("lmenrml_approx_step")
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    method = LMEnRML()

    # The same z and factor sqrt(variances) as PET's gen_real: identical arithmetic.
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    assert isinstance(method.strategy, ApproxUpdate)
    assert method.strategy.energy == float(_floats(ref, "in_energy")) == 0.95
    step = method.strategy.step(
        state, predicted, perturbed, error, damping=float(_floats(ref, "in_lambda"))
    )

    expected = _floats(ref, "out_posterior") - state
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())


@pytest.mark.pet_parity
@pytest.mark.parametrize("scenario", ["mixed", "worse_converged", "lambda_max", "lambda_min"])
def test_the_damping_rules_match_pet(pet_reference: ReferenceLoader, scenario: str) -> None:
    """Replay PET's scripted trials; the decisions and lambdas must be PET's exactly."""
    ref = pet_reference("lmenrml_schedule")
    lam, factor, minimum, maximum, tolerance = _floats(ref, f"in_{scenario}_settings")
    schedule = LMDampingSchedule(
        lam,
        factor,
        minimum,
        maximum,
        tolerance,
        converge_on_rejection=True,  # PET's rule
        confirm_convergence=False,  # PET's rule: the first small trial stops
    )
    accepted = _floats(ref, f"in_{scenario}_prior")
    rows = []
    for trial in _floats(ref, f"in_{scenario}_trials"):
        decision = schedule.decide(
            lam,
            float(accepted.mean()),
            float(trial.mean()),
            previous_spread=float(accepted.std()),
            spread=float(trial.std()),
        )
        rows.append([lam, float(decision.accepted), float(decision.converged), decision.damping])
        lam = decision.damping
        if decision.accepted:
            accepted = trial
        if decision.converged:
            break

    np.testing.assert_array_equal(np.array(rows), _floats(ref, f"out_{scenario}"))


@pytest.mark.pet_parity
def test_auto_lambda_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("lmenrml_schedule")
    perturbed = _floats(ref, "in_auto_perturbed")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_auto_variances"))
    data_mismatch = float(
        np.mean(error.data_mismatch(perturbed - _floats(ref, "in_auto_predicted")))
    )
    method = LMEnRML(schedule=LMDampingSchedule(initial="auto"))

    lam = method._initial_damping(SimpleNamespace(data_mismatch=data_mismatch, perturbed=perturbed))  # type: ignore[arg-type]

    assert math.isclose(lam, float(_floats(ref, "out_auto_lambda")), rel_tol=1e-14)


@pytest.mark.pet_parity
def test_the_full_step_matches_pet(pet_reference: ReferenceLoader) -> None:
    """The full analysis: the approx data term plus the pull toward each member's prior."""
    from apex.algorithms.data_assimilation import FullUpdate

    ref = pet_reference("lmenrml_full_step")
    prior, state = _floats(ref, "in_prior"), _floats(ref, "in_state")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    strategy = FullUpdate(
        float(_floats(ref, "in_energy")), state_scaling=_floats(ref, "in_scaling")
    )

    strategy.prepare(prior)
    step = strategy.step(
        state,
        _floats(ref, "in_predicted"),
        perturbed,
        error,
        damping=float(_floats(ref, "in_lambda")),
        prior_state=prior,
    )

    # Am's columns are singular vectors, defined up to sign: compare Am Am^T.
    am, am_pet = strategy.export_state()["am"], _floats(ref, "out_am")
    assert am.shape == am_pet.shape  # the same truncation of the prior anomalies
    np.testing.assert_allclose(
        am @ am.T, am_pet @ am_pet.T, rtol=0, atol=1e-10 * np.abs(am_pet @ am_pet.T).max()
    )
    expected = _floats(ref, "out_step")
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())
