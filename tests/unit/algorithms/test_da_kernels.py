import numpy as np
import pytest

from apex.algorithms.data_assimilation.kernels import anomalies, clip_to_bounds, truncated_svd


def test_anomalies_are_centred_and_give_the_unbiased_sample_covariance() -> None:
    ensemble = np.random.default_rng(1).standard_normal((5, 30))

    result = anomalies(ensemble)

    np.testing.assert_allclose(result.mean(axis=-1), 0.0, atol=1e-15)
    np.testing.assert_allclose(result @ result.T, np.cov(ensemble), rtol=1e-12)


@pytest.mark.parametrize("shape", [(5,), (5, 1)])
def test_anomalies_reject_non_ensembles(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="n_ensemble"):
        anomalies(np.ones(shape))


def _matrix_with_singular_values(values: list[float]) -> np.ndarray:
    rng = np.random.default_rng(2)
    u, _ = np.linalg.qr(rng.standard_normal((7, len(values))))
    v, _ = np.linalg.qr(rng.standard_normal((9, len(values))))
    return u @ np.diag(values) @ v.T


@pytest.mark.parametrize(
    ("energy", "rank"),
    # Singular values 4, 3, 2, 1 have cumulative fractions 0.4, 0.7, 0.9, 1.0. The kept
    # rank is the first whose fraction reaches `energy` (PET's nuclear-norm rule). Energies
    # sit off the exact boundaries: the computed singular values carry rounding error, so
    # e.g. a cumulative 0.7 may come out as 0.69999999999999996.
    [(0.3, 1), (0.39, 1), (0.41, 2), (0.69, 2), (0.71, 3), (0.89, 3), (0.91, 4), (1.0, 4)],
)
def test_truncated_svd_keeps_first_rank_reaching_energy(energy: float, rank: int) -> None:
    u, s, vt = truncated_svd(_matrix_with_singular_values([4.0, 3.0, 2.0, 1.0]), energy)

    assert s.shape == (rank,)
    assert u.shape == (7, rank)
    assert vt.shape == (rank, 9)
    np.testing.assert_allclose(s, [4.0, 3.0, 2.0, 1.0][:rank], rtol=1e-12)


def test_truncated_svd_keeps_everything_for_a_zero_matrix() -> None:
    _, s, _ = truncated_svd(np.zeros((3, 4)), 0.5)

    assert s.shape == (3,)


@pytest.mark.parametrize("energy", [0.0, -0.1, 1.5, 98.0])
def test_truncated_svd_rejects_energy_outside_unit_interval(energy: float) -> None:
    with pytest.raises(ValueError, match="energy"):
        truncated_svd(np.eye(3), energy)


def test_clip_to_bounds_clips_rows_and_allows_missing_bounds() -> None:
    state = np.array([[-2.0, 0.5, 3.0], [-2.0, 0.5, 3.0]])

    clipped = clip_to_bounds(state, lower=np.array([-1.0, -np.inf]), upper=None)

    np.testing.assert_array_equal(clipped, [[-1.0, 0.5, 3.0], [-2.0, 0.5, 3.0]])
    np.testing.assert_array_equal(clip_to_bounds(state, None, None), state)
