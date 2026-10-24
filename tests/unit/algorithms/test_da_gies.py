import itertools
import math

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    GIES,
    ApproxUpdate,
    DampingSchedule,
    DataErrorCovariance,
    RLMMACUpdate,
    iterative_methods,
    update_strategies,
)

# Non-square, with n_state != n_ensemble, so a transposed axis fails loudly.
N_STATE, N_DATA, N_ENSEMBLE = 6, 4, 25


def _problem(seed: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((N_STATE, N_ENSEMBLE))
    forward = rng.standard_normal((N_DATA, N_STATE))
    predicted = np.tanh(forward @ state)
    reference = np.tanh(forward @ state.mean(axis=-1))
    perturbed = rng.standard_normal((N_DATA, N_ENSEMBLE))
    return state, predicted, reference, perturbed


def _correlated_covariance() -> np.ndarray:
    a = np.random.default_rng(21).standard_normal((N_DATA, N_DATA))
    return a @ a.T + N_DATA * np.eye(N_DATA)


# ---------------------------------------------------------------- RLMMACUpdate


@pytest.mark.parametrize("dense", [False, True])
def test_untruncated_rlmmac_step_equals_its_closed_form(dense: bool) -> None:
    # Independent reference: dX dY^T (dY dY^T + gamma C_d)^-1 (D - Y), with dY centred on
    # the mean-model forecast and gamma = lambda * mean(s^2).
    state, predicted, reference, perturbed = _problem()
    matrix = _correlated_covariance() if dense else np.diag([0.5, 1.0, 2.0, 0.25])
    error = (
        DataErrorCovariance.dense(matrix)
        if dense
        else DataErrorCovariance.diagonal(np.diag(matrix))
    )
    damping = 0.3

    step = RLMMACUpdate(energy=1.0).step(
        state, predicted, perturbed, error, damping=damping, reference_prediction=reference
    )

    d_x = (state - state.mean(axis=-1, keepdims=True)) / math.sqrt(N_ENSEMBLE - 1)
    d_y = (predicted - reference[:, None]) / math.sqrt(N_ENSEMBLE - 1)
    s = np.linalg.svd(error.whiten(d_y), compute_uv=False)
    gamma = damping * np.mean(s**2)
    expected = d_x @ d_y.T @ np.linalg.solve(d_y @ d_y.T + gamma * matrix, perturbed - predicted)
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


def test_truncated_rlmmac_step_uses_the_kept_spectrum() -> None:
    # gamma is lambda * mean(s_r^2) over the kept rank r only; a regression computing it from
    # the full spectrum would change the step.
    state, predicted, reference, perturbed = _problem()
    error = DataErrorCovariance.diagonal(np.array([0.5, 1.0, 2.0, 0.25]))
    d_x = (state - state.mean(axis=-1, keepdims=True)) / math.sqrt(N_ENSEMBLE - 1)
    whitened = error.whiten((predicted - reference[:, None]) / math.sqrt(N_ENSEMBLE - 1))
    u, s, vt = np.linalg.svd(whitened, full_matrices=False)
    fractions = np.cumsum(s) / s.sum()
    energy = float((fractions[1] + fractions[2]) / 2)  # mid-gap: keeps exactly 3 of 4
    damping = 0.3

    step = RLMMACUpdate(energy=energy).step(
        state, predicted, perturbed, error, damping=damping, reference_prediction=reference
    )

    u, s, vt = u[:, :3], s[:3], vt[:3]
    gamma = damping * np.mean(s**2)
    expected = d_x @ vt.T @ np.diag(s / (s**2 + gamma)) @ u.T @ error.whiten(perturbed - predicted)
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("energy", [1.0, 0.8])
def test_unscaled_data_anomalies_shrink_every_step_by_sqrt_n_minus_1(energy: float) -> None:
    # PET-fork's unscaled dY (scale_data_anomalies = False): s and gamma = lambda mean(s^2)
    # both grow by sqrt(N-1), so each weight s / (s^2 + gamma), hence the step, shrinks by
    # it at the same lambda. The energy truncation is scale-free, so it holds when truncated.
    state, predicted, reference, perturbed = _problem()
    error = DataErrorCovariance.diagonal(np.array([0.5, 1.0, 2.0, 0.25]))
    steps = [
        RLMMACUpdate(energy=energy, scale_data_anomalies=scaled).step(
            state, predicted, perturbed, error, damping=0.3, reference_prediction=reference
        )
        for scaled in (True, False)
    ]
    np.testing.assert_allclose(steps[0], math.sqrt(N_ENSEMBLE - 1) * steps[1], rtol=1e-12)


def test_rlmmac_ignores_numerically_null_directions_at_tiny_damping() -> None:
    # A linear model makes g(x_mean) = mean(Y), so L^-1 dY has rank n_state < n_data and
    # numerically null singular values; they must not turn into a 1/s pseudo-inverse.
    rng = np.random.default_rng(3)
    forward = rng.standard_normal((8, 3))
    state = rng.standard_normal((3, 30))
    perturbed = forward @ np.array([1.0, -1.0, 2.0])[:, None] + rng.standard_normal((8, 30))
    error = DataErrorCovariance.diagonal(np.ones(8))

    steps = [
        RLMMACUpdate(energy=1.0).step(
            state,
            forward @ state,
            perturbed,
            error,
            damping=damping,
            reference_prediction=forward @ state.mean(axis=-1),
        )
        for damping in (1e-20, 1e-40, 0.0)
    ]

    np.testing.assert_allclose(steps[1], steps[0], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(steps[2], steps[0], rtol=1e-8, atol=1e-10)


def test_rlmmac_rejects_data_insensitive_to_the_state() -> None:
    state, _, _, perturbed = _problem()
    constant = np.ones((N_DATA, N_ENSEMBLE))

    with pytest.raises(ValueError, match="all zero"):
        RLMMACUpdate().step(
            state,
            constant,
            perturbed,
            DataErrorCovariance.diagonal(np.ones(N_DATA)),
            damping=1.0,
            reference_prediction=np.ones(N_DATA),
        )


def test_strategies_declare_whether_they_need_the_mean_forecast() -> None:
    assert RLMMACUpdate.requires_reference_prediction
    assert not ApproxUpdate.requires_reference_prediction
    assert GIES().requires_reference_prediction
    assert not GIES(ApproxUpdate()).requires_reference_prediction


def test_rlmmac_requires_a_valid_reference_prediction() -> None:
    state, predicted, reference, perturbed = _problem()
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    strategy = RLMMACUpdate()

    with pytest.raises(ValueError, match="needs reference_prediction"):
        strategy.step(state, predicted, perturbed, error, damping=1.0)
    with pytest.raises(ValueError, match="shape"):
        strategy.step(
            state, predicted, perturbed, error, damping=1.0, reference_prediction=reference[:-1]
        )
    reference[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        strategy.step(
            state, predicted, perturbed, error, damping=1.0, reference_prediction=reference
        )


def test_approx_update_rejects_a_reference_prediction() -> None:
    state, predicted, reference, perturbed = _problem()

    with pytest.raises(ValueError, match="RLMMACUpdate"):
        ApproxUpdate().step(
            state,
            predicted,
            perturbed,
            DataErrorCovariance.diagonal(np.ones(N_DATA)),
            reference_prediction=reference,
        )


def test_larger_damping_gives_a_smaller_step() -> None:
    state, predicted, reference, perturbed = _problem()
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    strategy = RLMMACUpdate(energy=1.0)

    def norm(damping: float) -> float:
        step = strategy.step(
            state, predicted, perturbed, error, damping=damping, reference_prediction=reference
        )
        return float(np.linalg.norm(step))

    assert norm(10.0) < norm(1.0) < norm(0.1)


# ---------------------------------------------------------------- DampingSchedule


def test_a_rejected_trial_never_converges_by_default() -> None:
    """Regression (5Spots_snesim, 2026-10-04): GIES stopped "converged" after one rejected
    trial whose data mismatch rose by 6e-5; a worse trial is rejected and the damping raised."""
    decision = DampingSchedule().decide(1.0, 100.0, 100.006)
    assert (decision.accepted, decision.converged, decision.damping) == (False, False, 2.0)
    assert decision.stalled  # counted towards max_stalled_rejections by the GIES loop
    assert not DampingSchedule().decide(1.0, 100.0, 150.0).stalled  # a real deterioration
    improved = DampingSchedule().decide(1.0, 100.0, 99.994)
    assert (improved.accepted, improved.converged) == (True, True)


@pytest.mark.parametrize(
    ("previous", "data_mismatch", "accepted", "converged", "damping"),
    [
        (100.0, 50.0, True, False, 0.5),  # improvement: accept, lambda / 2
        (100.0, 150.0, False, False, 2.0),  # deterioration: reject, lambda * 2
        (100.0, 99.5, True, True, 1.0),  # |1 - 0.995| < 0.01 and improved: stop, accept
        (100.0, 100.5, False, True, 1.0),  # |1 - 1.005| < 0.01 but worse: stop, reject
    ],
)
def test_schedule_follows_pet_check_convergence(
    previous: float, data_mismatch: float, accepted: bool, converged: bool, damping: float
) -> None:
    # PET-fork's rule: no reduction factor, so an acceptance divides by the factor.
    schedule = DampingSchedule(converge_on_rejection=True, reduction=None)
    decision = schedule.decide(1.0, previous, data_mismatch)

    assert (decision.accepted, decision.converged, decision.damping) == (
        accepted,
        converged,
        damping,
    )


def test_schedule_stops_reducing_damping_at_the_minimum() -> None:
    schedule = DampingSchedule(factor=4.0, minimum=0.1, reduction=None)

    assert schedule.decide(0.4, 100.0, 50.0).damping == 0.1
    assert schedule.decide(0.1, 100.0, 50.0).damping == 0.1
    assert schedule.decide(0.1, 100.0, 150.0).damping == 0.4


def test_a_reduction_factor_lowers_the_damping_as_the_matlab_rlm_mac() -> None:
    # MATLAB main/Kalman/RLM_average_cost.m: lambda * lambda_reduction_factor (0.9) after
    # an accepted step, lambda * lambda_increment_factor (2) after a rejected one.
    schedule = DampingSchedule(factor=2.0, reduction=0.9)

    accepted = schedule.decide(1.0, 100.0, 50.0)
    rejected = schedule.decide(1.0, 100.0, 150.0)

    assert (accepted.accepted, accepted.damping) == (True, 0.9)
    assert (rejected.accepted, rejected.damping) == (False, 2.0)
    # The floor still applies; 0.9 is the default (the user, 2026-10-13), and without a
    # reduction the schedule is PET-fork's.
    assert DampingSchedule(reduction=0.5, minimum=0.3).decide(0.3, 100.0, 50.0).damping == 0.3
    assert DampingSchedule().decide(1.0, 100.0, 50.0).damping == 0.9
    assert DampingSchedule(reduction=None).decide(1.0, 100.0, 50.0).damping == 0.5
    assert DampingSchedule(reduction=1.0).decide(1.0, 100.0, 50.0).damping == 1.0


def test_a_reduction_of_one_half_is_pet_forks_division_bit_for_bit() -> None:
    """The Brugge MEC replication gives PET-fork's rule as ``reduction = 0.5``."""
    rng = np.random.default_rng(7)
    halving, dividing = DampingSchedule(reduction=0.5), DampingSchedule(reduction=None)
    extremes = [5e-324, 3 * 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308]
    for damping in [*rng.uniform(1e-6, 1e6, 1000), *extremes]:
        assert (
            halving.decide(damping, 2.0, 1.0).damping == dividing.decide(damping, 2.0, 1.0).damping
        )


def test_successive_acceptances_scale_the_damping_geometrically() -> None:
    # Three improving trials in a row: lambda 1 -> 0.9 -> 0.81 -> 0.729 (MATLAB RLM-MAC).
    schedule = DampingSchedule(reduction=0.9)
    lambdas = [1.0]
    for data_mismatch in (80.0, 60.0, 45.0):
        lambdas.append(schedule.decide(lambdas[-1], data_mismatch * 1.5, data_mismatch).damping)
    np.testing.assert_allclose(lambdas, [1.0, 0.9, 0.81, 0.729], rtol=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial": -1.0},
        {"initial": 0.0},
        {"factor": 1.0},
        {"minimum": -0.1},
        {"data_mismatch_tolerance": 0.0},
        {"reduction": 0.0},
        {"reduction": 1.5},
        {"reduction": float("nan")},
    ],
)
def test_schedule_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"damping|factor|tolerance"):
        DampingSchedule(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(("previous", "data_mismatch"), [(0.0, 1.0), (1.0, np.nan), (np.inf, 1.0)])
def test_schedule_rejects_invalid_data_mismatches(previous: float, data_mismatch: float) -> None:
    with pytest.raises(ValueError, match=r"data[ _]mismatch"):
        DampingSchedule().decide(1.0, previous, data_mismatch)


# ---------------------------------------------------------------- GIES


def test_perturbations_are_reproducible_and_bounds_apply() -> None:
    state, predicted, reference, _ = _problem()
    error = DataErrorCovariance.diagonal(np.full(N_DATA, 1e-4))
    method = GIES(lower_bounds=np.full(N_STATE, -0.1), upper_bounds=np.full(N_STATE, 0.1))

    first = method.perturb_observations(np.zeros(N_DATA), error, np.random.default_rng(1), 25)
    again = method.perturb_observations(np.zeros(N_DATA), error, np.random.default_rng(1), 25)
    trial = method.update(
        state, predicted, first + 10.0, error, damping=1.0, reference_prediction=reference
    )

    np.testing.assert_array_equal(first, again)
    assert trial.min() >= -0.1
    assert trial.max() <= 0.1


def test_mean_data_mismatch_is_the_average_weighted_residual() -> None:
    _, predicted, _, perturbed = _problem()
    variances = np.array([0.5, 1.0, 2.0, 0.25])

    data_mismatch = GIES().mean_data_mismatch(
        predicted, perturbed, DataErrorCovariance.diagonal(variances)
    )

    expected = np.mean(np.sum((perturbed - predicted) ** 2 / variances[:, None], axis=0))
    assert data_mismatch == pytest.approx(expected, rel=1e-12)


def test_gies_iterations_converge_to_the_weighted_least_squares_fit() -> None:
    # Independent reference: RLM-MAC minimises each member's data mismatch (there is no prior
    # term). For a linear, overdetermined model whose ensemble spans the state space, every
    # member therefore converges to the weighted least-squares fit of its perturbed data.
    # This drives the whole accept/reject loop a workflow would run.
    rng = np.random.default_rng(7)
    n_state, n_data, n_ensemble = 3, 8, 30
    forward = rng.standard_normal((n_data, n_state))
    variances = rng.uniform(0.1, 0.5, n_data)
    error = DataErrorCovariance.diagonal(variances)
    observations = forward @ np.array([1.0, -2.0, 0.5]) + np.sqrt(variances) * rng.standard_normal(
        n_data
    )
    method = GIES(RLMMACUpdate(energy=1.0), DampingSchedule(data_mismatch_tolerance=1e-10))
    perturbed = method.perturb_observations(observations, error, rng, n_ensemble)

    state = rng.standard_normal((n_state, n_ensemble))
    damping = method.initial_damping
    data_mismatch = method.mean_data_mismatch(forward @ state, perturbed, error)
    accepted_data_mismatches = [data_mismatch]
    for _ in range(60):
        trial = method.update(
            state,
            forward @ state,
            perturbed,
            error,
            damping=damping,
            reference_prediction=forward @ state.mean(axis=-1),
        )
        trial_data_mismatch = method.mean_data_mismatch(forward @ trial, perturbed, error)
        decision = method.decide(damping, data_mismatch, trial_data_mismatch)
        if decision.accepted:
            state, data_mismatch = trial, trial_data_mismatch
            accepted_data_mismatches.append(data_mismatch)
        damping = decision.damping
        if decision.converged:
            break

    weights = 1.0 / variances
    normal = forward.T @ (weights[:, None] * forward)
    least_squares = np.linalg.solve(normal, forward.T @ (weights[:, None] * perturbed))
    assert len(accepted_data_mismatches) > 3  # the loop actually iterated
    assert all(b < a for a, b in itertools.pairwise(accepted_data_mismatches))
    np.testing.assert_allclose(state, least_squares, rtol=1e-4, atol=1e-6)


def test_gies_plugins_are_registered_by_name() -> None:
    assert "rlmmac" in update_strategies.names()
    assert "gies" in iterative_methods.names()

    method = iterative_methods.create("gies", strategy=update_strategies.create("rlmmac"))

    assert isinstance(method, GIES)
    assert method.initial_damping == 1.0


def test_exhausted_retries_raise_the_damping_no_lower_than_the_initial() -> None:
    # MATLAB RLM_average_cost.m: after maxInnerIter - 1 rejected trials, lambda is
    # multiplied by lambda_increment_factor once more and raised to the initial lambda.
    from apex.algorithms.data_assimilation.gies import _GiesRun

    method = GIES(schedule=DampingSchedule(initial=1.0, factor=2.0), max_trials_per_iteration=2)
    run = _GiesRun(None, np.zeros((1, 2)), np.zeros((1, 2)), np.arange(2))  # type: ignore[arg-type]
    run.damping = 0.1  # after one rejection (0.05 * 2)
    method._count_trial(run, accepted=False)
    assert (run.outer, run.trials, run.damping) == (0, 1, 0.1)
    run.damping = 0.2  # after the second rejection
    method._count_trial(run, accepted=False)
    assert (run.outer, run.trials, run.damping) == (1, 0, 1.0)  # 0.4 raised to the initial 1
    run.damping = 3.0
    method._count_trial(run, accepted=False)
    method._count_trial(run, accepted=False)
    assert (run.outer, run.trials, run.damping) == (2, 0, 6.0)
    method._count_trial(run, accepted=True)
    assert (run.outer, run.trials) == (3, 0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_max_trials_per_iteration_is_validated(bad: object) -> None:
    with pytest.raises(ValueError, match="max_trials_per_iteration"):
        GIES(max_trials_per_iteration=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["auto", 0.0, -1.0, math.inf, True])
def test_the_rlmmac_start_is_a_positive_number(bad: object) -> None:
    with pytest.raises(ValueError, match="initial damping"):
        DampingSchedule(initial=bad)  # type: ignore[arg-type]
