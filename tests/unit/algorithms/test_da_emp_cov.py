"""The approx update's empirical data error (PET ``emp_cov``): an independent check."""

import numpy as np

from apex.algorithms.data_assimilation import ApproxUpdate, DataErrorCovariance


def test_it_is_the_update_with_the_perturbations_covariance() -> None:
    """With energy 1 and fewer data than members, the step is
    dX dY^T (dY dY^T + (1 + lambda) E E^T)^-1 (D - Y), E the perturbations' anomalies."""
    rng = np.random.default_rng(3)
    n = 30
    state = rng.standard_normal((5, n))
    predicted = np.tanh(rng.standard_normal((6, 5)) @ state)
    error = DataErrorCovariance.diagonal(rng.uniform(0.1, 0.4, 6))
    perturbed = rng.standard_normal(6)[:, None] + error.perturbations(rng, n)
    damping = 0.8

    step = ApproxUpdate(1.0, empirical_data_error=True).step(
        state, predicted, perturbed, error, damping=damping
    )

    def anomalies(a: np.ndarray) -> np.ndarray:
        centred: np.ndarray = (a - a.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
        return centred

    dx, dy, e = anomalies(state), anomalies(predicted), anomalies(perturbed)
    expected = (
        dx @ dy.T @ np.linalg.solve(dy @ dy.T + (1 + damping) * e @ e.T, perturbed - predicted)
    )
    np.testing.assert_allclose(step, expected, rtol=1e-9, atol=1e-11)


def test_null_singular_values_are_dropped() -> None:
    """Energy 1 with more data than members: dY has N - 1 non-zero singular values and the
    rest are rounding noise; dividing by them would blow the step up. The step equals
    dX dY^T (dY dY^T + (1 + lambda) P E E^T P)^+ (D - Y), P the projection on range(dY)."""
    rng = np.random.default_rng(4)
    n, n_data = 8, 20
    state = rng.standard_normal((5, n))
    predicted = np.tanh(rng.standard_normal((n_data, 5)) @ state)
    error = DataErrorCovariance.diagonal(np.full(n_data, 0.2))
    perturbed = rng.standard_normal(n_data)[:, None] + error.perturbations(rng, n)

    step = ApproxUpdate(1.0, empirical_data_error=True).step(state, predicted, perturbed, error)

    def anomalies(a: np.ndarray) -> np.ndarray:
        centred: np.ndarray = (a - a.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
        return centred

    sigma = np.sqrt(0.2)
    dx, dy, e = anomalies(state), anomalies(predicted) / sigma, anomalies(perturbed) / sigma
    u = np.linalg.svd(dy, full_matrices=False)[0][:, : n - 1]
    p = u @ u.T
    inverse = np.linalg.pinv(dy @ dy.T + p @ e @ e.T @ p, rcond=1e-10)
    expected = dx @ dy.T @ inverse @ ((perturbed - predicted) / sigma)
    assert np.all(np.isfinite(step))
    np.testing.assert_allclose(step, expected, rtol=1e-7, atol=1e-9)
