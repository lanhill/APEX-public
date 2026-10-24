import numpy as np
import pytest

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance


def _correlated_covariance(n: int) -> np.ndarray:
    a = np.random.default_rng(3).standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


@pytest.fixture(params=["diagonal", "dense"])
def covariance(request: pytest.FixtureRequest) -> tuple[DataErrorCovariance, np.ndarray]:
    if request.param == "diagonal":
        variances = np.array([0.5, 2.0, 1.0, 4.0])
        return DataErrorCovariance.diagonal(variances), np.diag(variances)
    matrix = _correlated_covariance(4)
    return DataErrorCovariance.dense(matrix), matrix


def test_correlate_produces_the_covariance(
    covariance: tuple[DataErrorCovariance, np.ndarray],
) -> None:
    error, matrix = covariance
    factor = error.correlate(np.eye(4))  # L

    np.testing.assert_allclose(factor @ factor.T, matrix, rtol=1e-12)


def test_whiten_inverts_correlate(covariance: tuple[DataErrorCovariance, np.ndarray]) -> None:
    error, _ = covariance
    z = np.random.default_rng(4).standard_normal((4, 6))

    np.testing.assert_allclose(error.whiten(error.correlate(z)), z, rtol=1e-12, atol=1e-14)


def test_whitened_residual_norm_is_the_mahalanobis_distance(
    covariance: tuple[DataErrorCovariance, np.ndarray],
) -> None:
    # |L^-1 r|^2 = r^T C^-1 r only for the factor with C = L L^T; PET's upper factor fails
    # this for a correlated C.
    error, matrix = covariance
    residual = np.random.default_rng(5).standard_normal((4, 1))

    whitened = error.whiten(residual)

    expected = residual.T @ np.linalg.solve(matrix, residual)
    np.testing.assert_allclose(whitened.T @ whitened, expected, rtol=1e-12)


def test_inflated_scales_the_covariance(covariance: tuple[DataErrorCovariance, np.ndarray]) -> None:
    error, matrix = covariance
    factor = error.inflated(3.0).correlate(np.eye(4))

    np.testing.assert_allclose(factor @ factor.T, 3.0 * matrix, rtol=1e-12)


def test_perturbations_depend_only_on_the_seed(
    covariance: tuple[DataErrorCovariance, np.ndarray],
) -> None:
    error, _ = covariance

    first = error.perturbations(np.random.default_rng(6), 10)
    again = error.perturbations(np.random.default_rng(6), 10)
    other = error.perturbations(np.random.default_rng(7), 10)

    assert first.shape == (4, 10)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


@pytest.mark.parametrize("variances", [[1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [[1.0]], []])
def test_diagonal_rejects_invalid_variances(variances: list[float]) -> None:
    with pytest.raises(ValueError, match="variances"):
        DataErrorCovariance.diagonal(variances)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.ones((2, 3)), "square"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
    ],
)
def test_dense_rejects_invalid_matrices(matrix: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DataErrorCovariance.dense(matrix)


def test_dense_accepts_round_off_asymmetry_next_to_zero_entries() -> None:
    matrix = np.array([[1.0, 0.0], [1e-300, 2.0]])

    assert DataErrorCovariance.dense(matrix).n_data == 2


@pytest.mark.parametrize("alpha", [3.0, 4.0, 7.3])
def test_diagonal_inflation_uses_pet_arithmetic(alpha: float) -> None:
    # sqrt(alpha * var), not sqrt(alpha) * sqrt(var): the two differ in the last ulp for
    # most alpha, and PET uses the former.
    variances = np.random.default_rng(8).uniform(0.05, 0.5, 50)
    factor = DataErrorCovariance.diagonal(variances).inflated(alpha).correlate(np.eye(50))

    np.testing.assert_array_equal(np.diagonal(factor), np.sqrt(alpha * variances))


def test_whiten_rejects_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="expected shape"):
        DataErrorCovariance.diagonal([1.0, 1.0]).whiten(np.ones((3, 2)))


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf])
def test_inflated_rejects_invalid_factors(alpha: float) -> None:
    with pytest.raises(ValueError, match="inflation"):
        DataErrorCovariance.diagonal([1.0]).inflated(alpha)
