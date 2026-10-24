"""Unit tests of GN-EnRML's step-length schedule and the subspace strategy's guards."""

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    DataErrorCovariance,
    GNEnRML,
    GNStepSchedule,
    SubspaceUpdate,
)
from apex.core.errors import WorkflowError

N_STATE, N_DATA, N = 4, 6, 8


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, DataErrorCovariance]:
    rng = np.random.default_rng(7)
    prior = rng.standard_normal((N_STATE, N))
    predicted = rng.standard_normal((N_DATA, N))
    perturbed = rng.standard_normal((N_DATA, N))
    return prior, predicted, perturbed, DataErrorCovariance.diagonal(np.full(N_DATA, 0.5))


@pytest.mark.parametrize(
    "options",
    [{"initial": 0.0}, {"initial": "big"}, {"maximum": 0.0}, {"factor": 0.0},
     {"data_mismatch_tolerance": 0.0}],
)  # fmt: skip
def test_invalid_schedules_are_refused(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        GNStepSchedule(**options)  # type: ignore[arg-type]


def test_a_factor_of_one_fixes_the_step_length() -> None:
    schedule = GNStepSchedule(initial=0.3, factor=1.0)
    kwargs = {"previous_spread": 5.0, "accepted_iterations": 0}
    assert schedule.decide(0.3, 100.0, 50.0, spread=1.0, **kwargs).damping == 0.3  # type: ignore[arg-type]
    assert schedule.decide(0.3, 100.0, 150.0, spread=1.0, **kwargs).damping == 0.3  # type: ignore[arg-type]


def test_gn_enrml_starts_at_the_initial_step_length() -> None:
    assert GNEnRML().initial_damping == 0.2
    assert GNEnRML(schedule=GNStepSchedule(initial="auto")).initial_damping == 0.1


def test_the_subspace_step_is_deterministic_and_pending_until_accepted() -> None:
    prior, predicted, perturbed, error = _inputs()
    first, second = SubspaceUpdate(), SubspaceUpdate()
    first.prepare(prior)
    second.prepare(prior)

    a = first.step(prior, predicted, perturbed, error, prior_state=prior, step_length=0.5)
    b = second.step(prior, predicted, perturbed, error, prior_state=prior, step_length=0.5)

    np.testing.assert_array_equal(a, b)
    assert a.shape == (N_STATE, N)
    state = first.export_state()
    np.testing.assert_array_equal(state["weights"], np.zeros((N, N)))
    first.accept_trial()
    np.testing.assert_array_equal(first.export_state()["weights"], state["pending_weights"])
    assert "pending_weights" not in first.export_state()
    with pytest.raises(RuntimeError, match="no pending trial"):
        first.accept_trial()


def test_the_subspace_weights_round_trip() -> None:
    prior, predicted, perturbed, error = _inputs()
    strategy = SubspaceUpdate()
    strategy.prepare(prior)
    strategy.step(prior, predicted, perturbed, error, prior_state=prior)
    restored = SubspaceUpdate()

    restored.restore_state(strategy.export_state())

    for key, value in strategy.export_state().items():
        np.testing.assert_array_equal(restored.export_state()[key], value)


def test_the_subspace_step_is_guarded() -> None:
    prior, predicted, perturbed, error = _inputs()
    strategy = SubspaceUpdate()
    with pytest.raises(ValueError, match="not prepared"):
        strategy.step(prior, predicted, perturbed, error, prior_state=prior)
    strategy.prepare(prior)
    with pytest.raises(ValueError, match="prior state"):
        strategy.step(prior, predicted, perturbed, error)
    with pytest.raises(ValueError, match="damping"):
        strategy.step(prior, predicted, perturbed, error, prior_state=prior, damping=-1.0)
    with pytest.raises(ValueError, match="step_length"):
        strategy.step(prior, predicted, perturbed, error, prior_state=prior, step_length=0.0)
    with pytest.raises(ValueError, match="ensemble mean"):
        strategy.step(
            prior,
            predicted,
            perturbed,
            error,
            prior_state=prior,
            reference_prediction=predicted[:, 0],
        )
    flat = np.ones_like(predicted)
    with pytest.raises(ValueError, match="no spread"):
        strategy.step(prior, flat, perturbed, error, prior_state=prior)
    with pytest.raises(ValueError, match="energy"):
        SubspaceUpdate(energy=0.0)


def test_the_subspace_strategy_refuses_to_drop_members() -> None:
    strategy = SubspaceUpdate()
    strategy.drop_members(np.ones(N, dtype=bool))  # nothing dropped: fine
    with pytest.raises(WorkflowError, match="1 member"):
        strategy.drop_members(np.array([True] * (N - 1) + [False]))


def test_gn_enrml_has_no_lm_schedule() -> None:
    method = GNEnRML()
    assert not hasattr(method, "lm_schedule")
    with pytest.raises(TypeError, match="step_schedule"):
        method.decide(0.2, 10.0, 5.0)


def test_margis_with_a_huge_nu_is_the_transform_update() -> None:
    """Stordal et al. (2023): the data weights (M + nu)/(chi + nu s^2) tend to 1 as nu grows."""
    from apex.algorithms.data_assimilation import MarginalISUpdate, TransformUpdate
    from apex.domain.observations import DataKey

    prior, predicted, perturbed, error = _inputs()
    keys = [DataKey(f"t{i}", "A" if i < 3 else "B") for i in range(N_DATA)]
    margis, plain = MarginalISUpdate(nu=1e15), TransformUpdate()
    margis.use_data_keys(keys)
    margis.prepare(prior)
    plain.prepare(prior)

    a = margis.step(prior, predicted, perturbed, error, prior_state=prior)
    b = plain.step(prior, predicted, perturbed, error, prior_state=prior)

    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_margis_weights_each_type_by_its_data_mismatch() -> None:
    """A type whose residuals are large relative to C_d gets a small weight (its error scale
    is inferred to be larger), and M_k + nu over chi_k + nu otherwise."""
    from apex.algorithms.data_assimilation import MarginalISUpdate
    from apex.domain.observations import DataKey

    _, predicted, perturbed, error = _inputs()
    perturbed = perturbed.copy()
    perturbed[:3] += 30.0  # type A far off
    keys = [DataKey(f"t{i}", "A" if i < 3 else "B") for i in range(N_DATA)]
    strategy = MarginalISUpdate(nu=4.0)
    strategy.use_data_keys(keys)
    residuals = error.whiten(perturbed) - error.whiten(predicted)

    weights = strategy._data_weights(residuals, residuals)

    chi_a = np.mean(np.sum(residuals[:3] ** 2, axis=0))
    assert weights[0] == pytest.approx((3 + 4.0) / (chi_a + 4.0))
    assert weights[0] < 0.1 < weights[-1]
    with pytest.raises(ValueError, match="use_data_keys"):
        MarginalISUpdate()._data_weights(residuals, residuals)


def test_one_transform_step_is_the_rml_solution_with_a_singular_sample_covariance() -> None:
    """Raanes et al. (2019): for a linear model one full step of the transform IES from
    W = I gives x_j + C_p G^T (G C_p G^T + C_d)^-1 (d_j - G x_j) with C_p the prior's sample
    covariance, by the push-through identity, whatever its rank: 10 members, 30 unknowns."""
    from apex.algorithms.data_assimilation import TransformUpdate

    rng = np.random.default_rng(11)
    n_state, n_data, n = 30, 8, 10
    forward = rng.standard_normal((n_data, n_state))
    prior = rng.standard_normal((n_state, n))
    variances = rng.uniform(0.05, 0.2, n_data)
    error = DataErrorCovariance.diagonal(variances)
    perturbed = rng.standard_normal((n_data, n))
    strategy = TransformUpdate()
    strategy.prepare(prior)

    step = strategy.step(prior, forward @ prior, perturbed, error, prior_state=prior)

    c_p = np.cov(prior)
    assert np.linalg.matrix_rank(c_p) == n - 1
    gain = c_p @ forward.T @ np.linalg.inv(forward @ c_p @ forward.T + np.diag(variances))
    np.testing.assert_allclose(
        prior + step, prior + gain @ (perturbed - forward @ prior), rtol=1e-9, atol=1e-11
    )


def test_a_worse_trial_within_the_tolerance_is_rejected_not_converged() -> None:
    """PET stops on it; APEX rejects it."""
    from apex.algorithms.data_assimilation.gn_enrml import GNStepSchedule

    decision = GNStepSchedule().decide(
        0.2, 100.0, 100.5, previous_spread=1.0, spread=1.0, accepted_iterations=0
    )
    assert (decision.accepted, decision.converged, decision.stalled) == (False, False, True)
    assert decision.damping < 0.2  # the step length shrinks, as on any rejection
    pet = GNStepSchedule(converge_on_rejection=True).decide(
        0.2, 100.0, 100.5, previous_spread=1.0, spread=1.0, accepted_iterations=0
    )
    assert (pet.accepted, pet.converged) == (False, True)


def test_a_fixed_step_length_stops_at_once_on_a_stalled_trial() -> None:
    """With factor <= 1 the retry would repeat the trial: stop rather than spend forecasts."""
    from apex.algorithms.data_assimilation.gn_enrml import GNStepSchedule

    decision = GNStepSchedule(factor=1.0).decide(
        0.2, 100.0, 100.5, previous_spread=1.0, spread=1.0, accepted_iterations=0
    )
    assert (decision.accepted, decision.converged, decision.stalled) == (False, True, True)
