import logging
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.geostatistics import (
    covariance_factor,
    layer_covariance,
    variogram_covariance,
)
from apex.algorithms.priors import GaussianFieldPrior, GeostatsPySgsimPrior, IidGaussianPrior
from apex.core.plugins import priors
from apex.domain import StateVariable
from apex.storage.priors import FilePrior, read_array


@pytest.mark.parametrize("model", ["spherical", "cubic", "exponential", "gaussian", "stable"])
def test_variogram_models_start_at_the_sill_and_decay(model: str) -> None:
    u = np.linspace(0, 3, 31)

    values = variogram_covariance(u, model, sill=2.0, exponent=1.5)

    assert values[0] == pytest.approx(2.0)
    assert np.all(np.diff(values) <= 1e-12)
    if model in ("spherical", "cubic"):
        assert values[u >= 1].max() == 0.0  # compact support
    else:
        # Practical range: 5% of the sill at u = 1 (factor 3 in the exponent).
        assert values[10] == pytest.approx(2.0 * np.exp(-3), rel=1e-12)


def test_unknown_models_and_bad_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown variogram"):
        variogram_covariance(np.zeros(1), "wave")
    with pytest.raises(ValueError, match="exponent"):
        variogram_covariance(np.zeros(1), "stable", exponent=2.5)
    with pytest.raises(ValueError, match="sill"):
        variogram_covariance(np.zeros(1), "spherical", sill=0.0)


def test_layer_covariance_uses_eclipse_order_and_rotates_the_major_axis() -> None:
    nx, ny = 6, 4
    east = 1  # cell (i=1, j=0)
    north = nx  # cell (i=0, j=1)

    along_i = layer_covariance(nx, ny, model="spherical", sill=1.0, major_range=5, minor_range=1.5)
    along_j = layer_covariance(
        nx, ny, model="spherical", sill=1.0, major_range=5, minor_range=1.5, azimuth=90.0
    )

    assert along_i.shape == (24, 24)
    np.testing.assert_allclose(along_i, along_i.T)
    assert along_i[0, east] > along_i[0, north]  # major axis along i (x)
    assert along_j[0, north] > along_j[0, east]  # rotated by 90 degrees: along j (y)
    assert along_i[0, east] == pytest.approx(along_j[0, north])


def test_zero_range_is_uncorrelated() -> None:
    np.testing.assert_array_equal(
        layer_covariance(3, 2, model="spherical", sill=2.0, major_range=0.0), 2.0 * np.eye(6)
    )


def test_singular_covariances_fall_back_to_clipped_eigenvalues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    covariance = layer_covariance(10, 10, model="gaussian", sill=1.0, major_range=30.0)

    with caplog.at_level(logging.WARNING):
        factor = covariance_factor(covariance)

    np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-8)
    assert "not numerically positive definite" in caplog.text


def test_gaussian_field_reproduces_its_covariance_statistically() -> None:
    prior = GaussianFieldPrior([5, 3, 2], mean=1.0, variance=[1.0, 4.0], major_range=[3.0, 6.0])
    field = prior.generate(StateVariable("k", 30), 20_000, np.random.default_rng(0))

    np.testing.assert_allclose(field.mean(axis=-1), 1.0, atol=0.05)
    expected_top = layer_covariance(5, 3, model="spherical", sill=1.0, major_range=3.0)
    # Sample covariance error ~ sqrt(2/N) ~ 0.01 for unit variance.
    np.testing.assert_allclose(np.cov(field[:15]), expected_top, atol=0.04)
    np.testing.assert_allclose(np.cov(field[:15], field[15:])[:15, 15:], 0.0, atol=0.06)
    assert np.cov(field[15:]).diagonal().mean() == pytest.approx(4.0, rel=0.05)


def test_gaussian_field_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="one per layer"):
        GaussianFieldPrior([2, 2, 3], variance=[1.0, 2.0])
    with pytest.raises(ValueError, match="grid"):
        GaussianFieldPrior([2, 2])
    prior = GaussianFieldPrior([2, 2, 1])
    with pytest.raises(ValueError, match="has 5 entries"):
        prior.generate(StateVariable("k", 5), 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="ensemble_size"):
        prior.generate(StateVariable("k", 4), None, np.random.default_rng(0))


def test_iid_gaussian_prior() -> None:
    draws = IidGaussianPrior(mean=[0.0, 10.0], std=2.0).generate(
        StateVariable("x", 2), 5000, np.random.default_rng(1)
    )

    np.testing.assert_allclose(draws.mean(axis=-1), [0.0, 10.0], atol=0.1)
    np.testing.assert_allclose(draws.std(axis=-1), 2.0, rtol=0.05)


def test_file_prior_reads_trims_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "prior.npz"
    np.savez(path, k=np.arange(12.0).reshape(3, 4), other=np.zeros((2, 4)))
    rng = np.random.default_rng(0)

    np.testing.assert_array_equal(
        FilePrior(path).generate(StateVariable("k", 3), None, rng).shape, (3, 4)
    )
    np.testing.assert_array_equal(
        FilePrior(path).generate(StateVariable("k", 3), 2, rng), [[0, 1], [4, 5], [8, 9]]
    )
    with pytest.raises(ValueError, match="has 4 members"):
        FilePrior(path).generate(StateVariable("k", 3), 5, rng)
    with pytest.raises(ValueError, match="no array"):
        FilePrior(path).generate(StateVariable("missing", 3), None, rng)
    with pytest.raises(ValueError, match="must be"):
        FilePrior(path, key="other").generate(StateVariable("k", 3), None, rng)
    np.testing.assert_array_equal(read_array(path, "other"), np.zeros((2, 4)))


def test_geostatspy_sgsim_prior_follows_the_major_axis_and_variance() -> None:
    prior = GeostatsPySgsimPrior(
        [16, 10, 1], mean=2.0, variance=4.0, major_range=8.0, minor_range=2.0, azimuth=0.0
    )
    rng = np.random.default_rng(3)
    state = np.random.get_state()  # noqa: NPY002 - check the global generator is untouched

    field = prior.generate(StateVariable("k", 160), 40, rng).reshape(10, 16, 40)

    after = np.random.get_state()  # noqa: NPY002
    assert after[1].tolist() == state[1].tolist()
    lag_i = np.corrcoef(field[:, :-1].ravel(), field[:, 1:].ravel())[0, 1]
    lag_j = np.corrcoef(field[:-1].ravel(), field[1:].ravel())[0, 1]
    assert lag_i > lag_j + 0.2  # major axis along i
    assert field.mean() == pytest.approx(2.0, abs=0.5)
    assert field.var() == pytest.approx(4.0, rel=0.3)
    again = GeostatsPySgsimPrior(
        [16, 10, 1], mean=2.0, variance=4.0, major_range=8.0, minor_range=2.0
    ).generate(StateVariable("k", 160), 40, np.random.default_rng(3))
    np.testing.assert_array_equal(again.reshape(10, 16, 40), field)  # reproducible from the rng


def _diagonal_correlations(field: np.ndarray) -> tuple[float, float]:
    """Lag-(1, 1) and lag-(1, -1) correlations of ``(ny, nx, n_ensemble)`` fields."""
    up = np.corrcoef(field[:-1, :-1].ravel(), field[1:, 1:].ravel())[0, 1]
    down = np.corrcoef(field[1:, :-1].ravel(), field[:-1, 1:].ravel())[0, 1]
    return float(up), float(down)


def test_geostatspy_sgsim_and_gaussian_field_agree_on_a_rotated_major_axis() -> None:
    options = {"mean": 0.0, "variance": 1.0, "major_range": 8.0, "minor_range": 2.0}
    variable = StateVariable("k", 256)
    rng = np.random.default_rng(5)

    sgsim = GeostatsPySgsimPrior([16, 16, 1], azimuth=45.0, **options)  # type: ignore[arg-type]
    field = GaussianFieldPrior([16, 16, 1], azimuth=45.0, **options)  # type: ignore[arg-type]
    sgsim_up, sgsim_down = _diagonal_correlations(
        sgsim.generate(variable, 40, rng).reshape(16, 16, 40)
    )
    field_up, field_down = _diagonal_correlations(
        field.generate(variable, 400, rng).reshape(16, 16, 400)
    )

    # Azimuth 45 (counter-clockwise from i): the major axis runs along (i, j) = (1, 1).
    assert field_up > field_down + 0.3
    assert sgsim_up > sgsim_down + 0.2
    assert sgsim_up == pytest.approx(field_up, abs=0.15)


def test_geostatspy_sgsim_validates_its_ranges() -> None:
    with pytest.raises(ValueError, match="major_range >= minor_range > 0"):
        GeostatsPySgsimPrior([4, 4, 1], major_range=2.0, minor_range=3.0)
    with pytest.raises(ValueError, match="major_range >= minor_range > 0"):
        GeostatsPySgsimPrior([4, 4, 1], major_range=0.0)


def test_layers_with_the_same_variogram_share_one_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def counting_factor(covariance: np.ndarray) -> np.ndarray:
        calls.append(covariance.shape)
        return covariance_factor(covariance)

    monkeypatch.setattr("apex.algorithms.priors.covariance_factor", counting_factor)
    prior = GaussianFieldPrior([3, 3, 4], variance=[1.0, 1.0, 2.0, 1.0])

    prior.generate(StateVariable("k", 36), 5, np.random.default_rng(0))

    assert len(calls) == 2


def test_prior_generators_are_registered() -> None:
    assert {"gaussian", "file", "gaussian_field", "geostatspy_sgsim"} <= set(priors.names())
