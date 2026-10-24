"""GN-EnRML and the subspace analysis against PET (upstream ``e65fb10``).

The references (``tools/pet_reference/generate.py``): two iterations of PET's
``subspace_update`` with ``propose_state``'s weight branch (``subspace_steps``), and PET's
own ``GNEnRML.score_and_commit`` step-length rules on scripted data mismatches
(``gnenrml_schedule``). The approx and full steps GN-EnRML scales are pinned at
``lambda = 0`` by the ES and LM-EnRML cases; ``gamma`` multiplies them as PET's
``propose_state`` does (``enX + step_scale * step``).
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import DataErrorCovariance, SubspaceUpdate
from apex.algorithms.data_assimilation.gn_enrml import GNStepSchedule

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: Relative to the largest component: SVD, eigh and summation order differ, and APEX
#: divides by the diagonal of I + (1 + lambda) Lambda where PET solves with it (as the
#: other steps, measured ~1e-15).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
def test_two_subspace_iterations_match_pet(pet_reference: ReferenceLoader) -> None:
    """Iteration 0 from W = 0, then iteration 1 from its committed weights."""
    ref = pet_reference("subspace_steps")
    prior, forward = _floats(ref, "in_prior"), _floats(ref, "in_forward")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    lam, gamma = float(_floats(ref, "in_lambda")), float(_floats(ref, "in_gamma"))
    strategy = SubspaceUpdate(float(_floats(ref, "in_energy")))
    strategy.prepare(prior)

    state = prior
    for k in (0, 1):
        predicted = np.tanh(forward @ state)
        np.testing.assert_allclose(predicted, _floats(ref, f"out_predicted_{k}"), rtol=1e-12)
        step = strategy.step(
            state, predicted, perturbed, error, damping=lam, prior_state=prior, step_length=gamma
        )
        weights = strategy.export_state()["pending_weights"]
        expected_w = _floats(ref, f"out_weights_{k}")
        np.testing.assert_allclose(
            weights, expected_w, rtol=0, atol=STEP_RTOL * np.abs(expected_w).max()
        )
        expected = _floats(ref, f"out_state_{k}")
        increment = expected - state
        np.testing.assert_allclose(
            step, increment, rtol=0, atol=STEP_RTOL * np.abs(increment).max()
        )
        strategy.accept_trial()  # PET's score_and_commit on acceptance
        state = state + step


@pytest.mark.pet_parity
@pytest.mark.parametrize("scenario", ["mixed", "worse_converged", "fixed"])
def test_the_step_length_rules_match_pet(pet_reference: ReferenceLoader, scenario: str) -> None:
    """Replay PET's scripted trials; the decisions and gammas must be PET's exactly."""
    ref = pet_reference("gnenrml_schedule")
    gamma, maximum, factor, tolerance = _floats(ref, f"in_{scenario}_settings")
    schedule = GNStepSchedule(gamma, maximum, factor, tolerance, converge_on_rejection=True)
    accepted, accepted_iterations = _floats(ref, f"in_{scenario}_prior"), 0
    rows = []
    for trial in _floats(ref, f"in_{scenario}_trials"):
        decision = schedule.decide(
            gamma,
            float(accepted.mean()),
            float(trial.mean()),
            previous_spread=float(accepted.std()),
            spread=float(trial.std()),
            accepted_iterations=accepted_iterations,
        )
        rows.append([gamma, float(decision.accepted), float(decision.converged), decision.damping])
        gamma = decision.damping
        if decision.converged:
            break
        if decision.accepted:
            accepted, accepted_iterations = trial, accepted_iterations + 1

    np.testing.assert_array_equal(np.array(rows), _floats(ref, f"out_{scenario}"))


@pytest.mark.pet_parity
def test_auto_gamma_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("gnenrml_schedule")
    assert GNStepSchedule(initial="auto").initial_gamma == float(_floats(ref, "out_auto_gamma"))


class _UpperWhitening:
    """PET's dense-C_d whitening: ``U^-1`` with ``C_d = U^T U`` (the ES-MDA bug)."""

    def __init__(self, upper: NDArray[np.float64]) -> None:
        self.upper, self.n_data = upper, upper.shape[0]

    def whiten(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        solved: NDArray[np.float64] = np.linalg.solve(self.upper, values)
        return solved


@pytest.mark.pet_parity
def test_the_dense_subspace_step_matches_pet_only_when_s_spans_the_data(
    pet_reference: ReferenceLoader,
) -> None:
    """With a correlated C_d, PET whitens with ``U^-1`` (C_d = U^T U) where APEX uses
    ``L^-1`` (L = U^T). When ``S`` spans the data space (8 data, 15 members, no truncation)
    that cancels: the inverse is exact and every term is whitened by the same matrix, so
    APEX matches PET. Otherwise the step projects on the kept range of ``S``, which depends
    on the whitening: with truncation, and with more data than members (30 data, 12
    members, only the null singular value dropped). PET's output is then pinned as the step
    with ``U^-1`` whitening, and APEX's differs.
    """
    ref = pet_reference("subspace_dense")
    prior, predicted = _floats(ref, "in_prior"), _floats(ref, "in_predicted")
    covariance = _floats(ref, "in_covariance")
    upper = _floats(ref, "out_scale_data")
    np.testing.assert_allclose(upper.T @ upper, covariance, rtol=1e-12)  # PET's factor
    perturbed = _floats(ref, "out_perturbed_observations")
    lam = float(_floats(ref, "in_lambda"))

    def weights(error: object, energy: float) -> NDArray[np.float64]:
        strategy = SubspaceUpdate(energy)
        strategy.prepare(prior)
        strategy.step(prior, predicted, perturbed, error, damping=lam, prior_state=prior)  # type: ignore[arg-type]
        return strategy.export_state()["pending_weights"]

    correct, pet_whitening = DataErrorCovariance.dense(covariance), _UpperWhitening(upper)
    full, truncated = _floats(ref, "out_weights"), _floats(ref, "out_weights_truncated")
    atol = STEP_RTOL * np.abs(full).max()
    np.testing.assert_allclose(weights(correct, 1.0), full, rtol=0, atol=atol)
    np.testing.assert_allclose(weights(pet_whitening, 1.0), full, rtol=0, atol=atol)
    atol = STEP_RTOL * np.abs(truncated).max()
    np.testing.assert_allclose(weights(pet_whitening, 0.8), truncated, rtol=0, atol=atol)
    difference = np.abs(weights(correct, 0.8) - truncated).max()
    assert difference > 0.01 * np.abs(truncated).max()

    # More data than members: S cannot span the data space, even untruncated.
    prior, predicted = _floats(ref, "in_wide_prior"), _floats(ref, "in_wide_predicted")
    perturbed = _floats(ref, "out_wide_perturbed_observations")
    wide = _floats(ref, "out_wide_weights")
    upper = _floats(ref, "out_wide_scale_data")
    correct = DataErrorCovariance.dense(_floats(ref, "in_wide_covariance"))
    atol = STEP_RTOL * np.abs(wide).max()
    energy = 1.0 - 1e-9  # only S's null singular value is dropped
    np.testing.assert_allclose(weights(_UpperWhitening(upper), energy), wide, rtol=0, atol=atol)
    assert np.abs(weights(correct, energy) - wide).max() > 0.01 * np.abs(wide).max()
