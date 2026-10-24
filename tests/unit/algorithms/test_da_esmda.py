import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    ESMDA,
    ApproxUpdate,
    DataErrorCovariance,
    methods,
    update_strategies,
)

# Deliberately non-square and with n_state != n_ensemble so that a transposed axis
# fails loudly instead of passing by coincidence.
N_STATE, N_DATA, N_ENSEMBLE = 6, 4, 25


def _ensembles(seed: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((N_STATE, N_ENSEMBLE))
    predicted = rng.standard_normal((N_DATA, N_STATE)) @ state + 0.1 * rng.standard_normal(
        (N_DATA, N_ENSEMBLE)
    )
    perturbed = rng.standard_normal((N_DATA, N_ENSEMBLE))
    return state, predicted, perturbed


def _correlated_covariance() -> np.ndarray:
    a = np.random.default_rng(11).standard_normal((N_DATA, N_DATA))
    return a @ a.T + N_DATA * np.eye(N_DATA)


# ---------------------------------------------------------------- ApproxUpdate


@pytest.mark.parametrize("dense", [False, True])
def test_untruncated_approx_step_equals_the_ensemble_kalman_gain(dense: bool) -> None:
    # Independent reference: with no truncation and damping 0 the TSVD form must equal
    # C_xy (C_yy + C_d)^-1 (D - Y) with sample covariances. The dense case is where PET's
    # upper-Cholesky whitening diverges.
    state, predicted, perturbed = _ensembles()
    matrix = _correlated_covariance() if dense else np.diag([0.5, 1.0, 2.0, 0.25])
    error = (
        DataErrorCovariance.dense(matrix)
        if dense
        else DataErrorCovariance.diagonal(np.diag(matrix))
    )

    step = ApproxUpdate(energy=1.0).step(state, predicted, perturbed, error)

    c_xy = np.cov(state, predicted)[:N_STATE, N_STATE:]
    c_yy = np.cov(predicted)
    expected = c_xy @ np.linalg.solve(c_yy + matrix, perturbed - predicted)
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", ["state", "predicted", "perturbed_observations"])
def test_approx_step_rejects_non_finite_ensembles(name: str) -> None:
    state, predicted, perturbed = _ensembles()
    arrays = {"state": state, "predicted": predicted, "perturbed_observations": perturbed}
    arrays[name][0, 0] = np.nan

    with pytest.raises(ValueError, match=f"{name} contains non-finite"):
        ApproxUpdate().step(
            arrays["state"],
            arrays["predicted"],
            arrays["perturbed_observations"],
            DataErrorCovariance.diagonal(np.ones(N_DATA)),
        )


def test_damping_shrinks_the_step() -> None:
    state, predicted, perturbed = _ensembles()
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    strategy = ApproxUpdate(energy=1.0)

    undamped = strategy.step(state, predicted, perturbed, error)
    damped = strategy.step(state, predicted, perturbed, error, damping=10.0)

    assert np.linalg.norm(damped) < np.linalg.norm(undamped)


@pytest.mark.parametrize(
    ("state_shape", "predicted_shape", "perturbed_shape", "message"),
    [
        ((N_STATE, N_ENSEMBLE), (N_DATA, N_ENSEMBLE + 1), (N_DATA, N_ENSEMBLE + 1), "ensemble"),
        ((N_STATE, N_ENSEMBLE), (N_DATA, N_ENSEMBLE), (N_DATA, N_ENSEMBLE - 1), "perturbed"),
        ((N_STATE, N_ENSEMBLE), (N_DATA + 1, N_ENSEMBLE), (N_DATA + 1, N_ENSEMBLE), "data rows"),
        ((N_STATE,), (N_DATA, N_ENSEMBLE), (N_DATA, N_ENSEMBLE), "2-D"),
    ],
)
def test_approx_step_rejects_inconsistent_shapes(
    state_shape: tuple[int, ...],
    predicted_shape: tuple[int, ...],
    perturbed_shape: tuple[int, ...],
    message: str,
) -> None:
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))

    with pytest.raises(ValueError, match=message):
        ApproxUpdate().step(
            np.ones(state_shape), np.ones(predicted_shape), np.ones(perturbed_shape), error
        )


# ---------------------------------------------------------------- ESMDA schedule


def test_schedule_accepts_rounded_factors_that_pet_rejects() -> None:
    # Emerick & Reynolds' 4-step schedule; sum(1/alpha) = 1 + 3.6e-11 after rounding.
    method = ESMDA([9.333333333, 7.0, 4.0, 2.0])

    assert method.n_steps == 4


@pytest.mark.parametrize("inflation", [[], [2.0, 3.0], [1.0, -1.0], [np.inf, 1.0]])
def test_schedule_rejects_invalid_factors(inflation: list[float]) -> None:
    with pytest.raises(ValueError, match=r"alpha|inflation"):
        ESMDA(inflation)


def test_uniform_schedule_matches_pet_default() -> None:
    assert ESMDA.uniform(3).inflation == (3.0, 3.0, 3.0)


# ---------------------------------------------------------------- ESMDA analysis


def test_analysis_is_reproducible_for_a_seed_and_varies_between_seeds() -> None:
    state, predicted, _ = _ensembles()
    observations = np.zeros(N_DATA)
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    method = ESMDA.uniform(2)

    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return method.analysis(0, state, predicted, observations, error, rng).posterior

    np.testing.assert_array_equal(run(1), run(1))
    assert not np.allclose(run(1), run(2))


def test_analysis_perturbs_with_the_inflated_covariance() -> None:
    state, predicted, _ = _ensembles()
    observations = np.arange(N_DATA, dtype=np.float64)
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))

    result = ESMDA([2.0, 2.0]).analysis(
        1, state, predicted, observations, error, np.random.default_rng(3)
    )

    z = np.random.default_rng(3).standard_normal((N_DATA, N_ENSEMBLE))
    np.testing.assert_allclose(
        result.perturbed_observations, observations[:, None] + np.sqrt(2) * z
    )
    np.testing.assert_allclose(
        result.posterior,
        ESMDA([2.0, 2.0]).update(1, state, predicted, result.perturbed_observations, error),
    )


def test_analysis_clips_the_posterior_to_bounds() -> None:
    state, predicted, _ = _ensembles()
    error = DataErrorCovariance.diagonal(np.full(N_DATA, 1e-4))
    lower, upper = np.full(N_STATE, -0.1), np.full(N_STATE, 0.1)

    result = ESMDA([1.0], lower_bounds=lower, upper_bounds=upper).analysis(
        0, state, predicted, np.full(N_DATA, 10.0), error, np.random.default_rng(0)
    )

    assert result.posterior.min() >= -0.1
    assert result.posterior.max() <= 0.1


def test_uniform_forwards_bounds() -> None:
    method = ESMDA.uniform(2, lower_bounds=np.zeros(N_STATE), upper_bounds=np.ones(N_STATE))

    assert method.lower_bounds is not None
    assert method.upper_bounds is not None


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (np.ones(N_STATE), np.zeros(N_STATE), "must not exceed"),
        (np.zeros(N_STATE), np.ones(N_STATE + 1), "differ in shape"),
        (np.zeros((N_STATE, 1)), None, "1-D"),
    ],
)
def test_invalid_bounds_are_rejected_at_construction(
    lower: np.ndarray, upper: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ESMDA([1.0], lower_bounds=lower, upper_bounds=upper)


def test_analysis_rejects_out_of_range_step_and_bad_observations() -> None:
    state, predicted, _ = _ensembles()
    error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    rng = np.random.default_rng(0)

    with pytest.raises(IndexError, match="step_index"):
        ESMDA.uniform(2).analysis(2, state, predicted, np.zeros(N_DATA), error, rng)
    with pytest.raises(ValueError, match="observations"):
        ESMDA.uniform(2).analysis(0, state, predicted, np.zeros((N_DATA, 1)), error, rng)


@pytest.mark.parametrize("dense", [False, True])
def test_esmda_recovers_the_linear_gaussian_posterior(dense: bool) -> None:
    # Independent reference: for a linear model and Gaussian prior, ES-MDA with
    # sum(1/alpha) = 1 converges to the exact posterior as the ensemble grows. The dense
    # (correlated C_d) case is where APEX deliberately differs from PET, so it needs this
    # oracle rather than PET.
    rng = np.random.default_rng(2026)
    n_state, n_data, n_ensemble = 3, 5, 20_000
    g = rng.standard_normal((n_data, n_state))
    prior_mean = np.array([1.0, -1.0, 0.5])
    prior_cov = np.diag([1.0, 2.0, 0.5])
    if dense:
        a = rng.standard_normal((n_data, n_data))
        data_cov = 0.1 * (a @ a.T) + 0.1 * np.eye(n_data)
        error = DataErrorCovariance.dense(data_cov)
    else:
        data_cov = np.diag(np.full(n_data, 0.3))
        error = DataErrorCovariance.diagonal(np.diag(data_cov))
    observations = g @ np.array([0.2, 0.4, -0.3]) + rng.multivariate_normal(
        np.zeros(n_data), data_cov
    )

    gain = prior_cov @ g.T @ np.linalg.inv(g @ prior_cov @ g.T + data_cov)
    posterior_mean = prior_mean + gain @ (observations - g @ prior_mean)
    posterior_cov = prior_cov - gain @ g @ prior_cov

    state = rng.multivariate_normal(prior_mean, prior_cov, size=n_ensemble).T
    method = ESMDA([9.333333333, 7.0, 4.0, 2.0], ApproxUpdate(energy=1.0))
    for step in range(method.n_steps):
        state = method.analysis(step, state, g @ state, observations, error, rng).posterior

    # Monte Carlo error of the mean is ~sqrt(var / N) ~ 0.01; 0.03 leaves margin for the
    # O(1/N) sampling error of the gains.
    np.testing.assert_allclose(state.mean(axis=-1), posterior_mean, atol=0.03)
    np.testing.assert_allclose(np.cov(state), posterior_cov, atol=0.03)


# ---------------------------------------------------------------- registries


def test_plugins_are_registered_by_name() -> None:
    assert "approx" in update_strategies.names()
    assert "esmda" in methods.names()

    method = methods.create(
        "esmda", inflation=[2.0, 2.0], strategy=update_strategies.create("approx")
    )

    assert isinstance(method, ESMDA)
    assert method.n_steps == 2


def test_esmda_with_prior_taper_localizes_the_gain() -> None:
    """ES-MDA + adaptive_prior: step = (T o (dX G)) r, T the prior taper.

    An APEX extension (PET-fork's approx update has no prior-only branch): the taper
    multiplies ES-MDA's gain as in PET-fork's RLM-MAC ini_only branch. With T = 1 it is
    the unlocalized step.
    """
    from apex.algorithms.data_assimilation import (
        ApproxUpdate,
        DataErrorCovariance,
        PriorAdaptiveLocalization,
    )
    from apex.algorithms.data_assimilation.kernels import anomalies

    rng = np.random.default_rng(5)
    state = rng.standard_normal((12, 20))
    predicted = np.tanh(rng.standard_normal((9, 12)) @ state)
    perturbed = predicted.mean(axis=1, keepdims=True) + rng.standard_normal((9, 20))
    error = DataErrorCovariance.diagonal(rng.uniform(0.1, 0.5, 9))
    local = PriorAdaptiveLocalization([12], taper="soft")
    local.prepare(state, predicted, [], rng)
    strategy = ApproxUpdate(energy=1.0, localization=local)

    step = strategy.step(state, predicted, perturbed, error, rng=rng)

    u, s, vt = np.linalg.svd(error.whiten(anomalies(predicted)), full_matrices=False)
    gain = anomalies(state) @ (vt.T @ np.diag(s / (1 + s**2)) @ u.T)
    assert local.taper_matrix is not None
    np.testing.assert_allclose(
        step, (local.taper_matrix * gain) @ error.whiten(perturbed - predicted), atol=1e-12
    )
    local.taper_matrix = np.ones_like(local.taper_matrix)
    unlocalized = ApproxUpdate(energy=1.0).step(state, predicted, perturbed, error)
    np.testing.assert_allclose(
        strategy.step(state, predicted, perturbed, error, rng=rng), unlocalized, atol=1e-12
    )
    with pytest.raises(ValueError, match="needs rng"):
        strategy.step(state, predicted, perturbed, error)


@pytest.mark.parametrize("method", ["esmda", "es"])
@pytest.mark.parametrize("strategy", ["rlmmac", "local_rlmmac"])
def test_es_and_es_mda_refuse_the_mean_model_strategies(method: str, strategy: str) -> None:
    """RLM-MAC centres on the forecast at the mean state, which ES and ES-MDA never make:
    refused when the method is built, before any forecast."""
    from apex.algorithms.data_assimilation.es import ES
    from apex.algorithms.data_assimilation.gies import RLMMACUpdate
    from apex.algorithms.data_assimilation.local_analysis import LocalRLMMACUpdate

    update = RLMMACUpdate() if strategy == "rlmmac" else LocalRLMMACUpdate(block_sizes=[3])
    name = "ESMDA" if method == "esmda" else "ES"
    with pytest.raises(ValueError, match=f"^{name} with the .* analysis is not supported"):
        ESMDA((4.0, 4.0, 4.0, 4.0), update) if method == "esmda" else ES(update)


@pytest.mark.parametrize("strategy", ["full", "subspace", "subspace2", "margis"])
def test_es_mda_refuses_the_prior_increment_strategies_factually(strategy: str) -> None:
    """The refusal states why, without claims about other codes (release tester, 2026-10-19).

    These strategies add an iterative method's prior-increment term, which ES-MDA's steps
    do not have.
    """
    from apex.algorithms.data_assimilation.registries import update_strategies

    with pytest.raises(ValueError, match="prior-increment term") as refused:
        ESMDA((2.0, 2.0), update_strategies.create(strategy))
    assert "PET" not in str(refused.value)
    assert "gies, lm_enrml or gn_enrml" in str(refused.value)
