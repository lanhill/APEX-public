"""The ``wls`` variogram estimator against GSTools and synthetic recovery.

References (Validation 1):

- **GSTools** 1.7 (``gstools.vario_estimate``, Matheron's estimator, an independent
  implementation): the experimental semivariogram and its pair counts, omnidirectional and
  directional (angle tolerance and bandwidth), agree to round-off on the same data.
- **Synthetic recovery:** fields drawn from known variograms (APEX's grid covariances,
  :func:`~apex.algorithms.geostatistics.covariance.layer_covariance`); over 20 seeds the
  median fitted range is within 20 % and the median total sill within 15 % of the truth,
  and the right model is picked in most seeds.
"""

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geostatistics.covariance import covariance_factor, layer_covariance
from apex.algorithms.geostatistics.variogram import (
    MIN_PAIRS,
    WeightedLeastSquares,
    experimental_variogram,
    fit_variogram,
    model_variogram,
)
from apex.core.geomodeling import VariogramEstimator
from apex.core.plugins import variogram_estimators

GRID = 60
POINTS = 1200
SEEDS = 20


def _points(seed: int, n: int = 300) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 40, (n, 2)), rng.standard_normal(n)


def test_the_experimental_variogram_is_gstools_to_round_off() -> None:
    gs = pytest.importorskip("gstools")
    xy, z = _points(0)
    edges = np.linspace(0, 20, 11)

    mine = experimental_variogram(xy, z, edges)
    _, gamma, counts = gs.vario_estimate(xy.T, z, edges, return_counts=True)

    np.testing.assert_allclose(mine.gamma, gamma, rtol=1e-12)
    np.testing.assert_array_equal(mine.counts, counts)


@pytest.mark.parametrize(("azimuth", "tolerance", "bandwidth"),
                         [(30.0, 20.0, 4.0), (0.0, 22.5, None), (112.5, 11.25, 2.5)])  # fmt: skip
def test_a_directional_variogram_is_gstools_to_round_off(
    azimuth: float, tolerance: float, bandwidth: float | None
) -> None:
    """GSTools counts a pair in a direction or its opposite, as APEX does."""
    gs = pytest.importorskip("gstools")
    xy, z = _points(1)
    edges = np.linspace(0, 20, 11)
    theta = math.radians(azimuth)

    mine = experimental_variogram(xy, z, edges, azimuth=azimuth, angle_tolerance=tolerance,
                                  bandwidth=bandwidth)  # fmt: skip
    _, gamma, counts = gs.vario_estimate(
        xy.T, z, edges, direction=[[math.cos(theta), math.sin(theta)]],
        angles_tol=math.radians(tolerance), bandwidth=bandwidth, return_counts=True,
    )  # fmt: skip

    np.testing.assert_array_equal(mine.counts, np.asarray(counts).ravel())
    np.testing.assert_allclose(mine.gamma, np.asarray(gamma).ravel(), rtol=1e-12)


def test_only_pairs_in_one_layer_count() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    z = np.array([0.0, 2.0, 10.0, 30.0])

    found = experimental_variogram(xy, z, [0.5, 1.5], layers=np.array([1, 1, 2, 2]))

    # (0 - 2)^2 and (10 - 30)^2: never a pair across the layers.
    assert found.counts.tolist() == [2]
    assert found.gamma[0] == pytest.approx((4 + 400) / 4)


def _sample(
    model: str, major: float, minor: float, azimuth: float
) -> tuple[NDArray[np.float64], list[tuple[NDArray[np.int64], NDArray[np.float64]]]]:
    """Fields drawn from a known variogram (unit sill), POINTS cells of each, per seed."""
    factor = covariance_factor(layer_covariance(GRID, GRID, model=model, sill=1.0,
                                                major_range=major, minor_range=minor,
                                                azimuth=azimuth))  # fmt: skip
    data = []
    for seed in range(SEEDS):
        field = factor @ np.random.default_rng(seed).standard_normal(GRID * GRID)
        index = np.random.default_rng(1000 + seed).choice(GRID * GRID, POINTS, replace=False)
        j, i = np.divmod(index, GRID)  # Eclipse order: index = i + nx * j
        data.append((np.c_[i + 1, j + 1, np.ones_like(i)], field[index]))
    return factor, data


@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian"])
def test_wls_recovers_an_isotropic_variogram(model: str) -> None:
    _, data = _sample(model, 10.0, 10.0, 0.0)
    estimator = WeightedLeastSquares()

    found = [estimator.estimate(cells, values, anisotropic=False) for cells, values in data]

    ranges = [e.major_range for e in found]
    sills = [e.sill + e.nugget for e in found]
    assert np.median(ranges) == pytest.approx(10.0, rel=0.20)
    assert np.median(sills) == pytest.approx(1.0, rel=0.15)
    assert sum(e.model == model for e in found) > SEEDS / 2


def test_wls_recovers_an_anisotropic_variogram() -> None:
    """Within 20 %, though the fit has a known bias: it takes each 22.5-degree sector's
    pairs to lie along its centre, which lengthens the major range (about 16 % on these
    fields, 16 cells read as 18.6 before mean pair distances were used)."""
    _, data = _sample("spherical", 16.0, 6.0, 30.0)
    estimator = WeightedLeastSquares()

    found = [estimator.estimate(cells, values, model="spherical") for cells, values in data]

    assert np.median([e.major_range for e in found]) == pytest.approx(16.0, rel=0.20)
    assert np.median([e.minor_range for e in found]) == pytest.approx(6.0, rel=0.20)
    assert np.median([e.azimuth for e in found]) == pytest.approx(30.0, abs=10.0)
    assert np.median([e.sill + e.nugget for e in found]) == pytest.approx(1.0, rel=0.15)


def test_the_fit_reproduces_a_noise_free_model() -> None:
    """Values of the model itself, with many pairs: the fit returns its parameters."""
    lags = np.linspace(1.0, 20.0, 12)
    gamma = model_variogram(lags, np.zeros(12), "exponential", 0.1, 0.9, 8.0, 8.0, 0.0)
    from apex.algorithms.geostatistics.variogram import ExperimentalVariogram

    observed = ExperimentalVariogram(lags, gamma, np.full(12, 1000, dtype=np.int64))
    fit = fit_variogram([observed], "exponential")

    assert (fit.nugget, fit.sill, fit.major_range) == pytest.approx((0.1, 0.9, 8.0), rel=1e-4)
    assert fit.residual == pytest.approx(0.0, abs=1e-10)


def test_too_few_data_are_refused_and_the_plugin_is_registered() -> None:
    estimator = variogram_estimators.create("wls")
    assert isinstance(estimator, VariogramEstimator)
    cells = np.array([[1, 1, 1], [2, 1, 1], [3, 1, 1]])

    with pytest.raises(ValueError, match="all equal"):
        estimator.estimate(cells, np.ones(3))
    with pytest.raises(ValueError, match=f"{MIN_PAIRS} pairs"):
        estimator.estimate(cells, np.array([0.0, 1.0, 3.0]), anisotropic=False)
    with pytest.raises(ValueError, match="model must be"):
        estimator.estimate(cells, np.array([0.0, 1.0, 3.0]), model="cubic")


def test_the_fit_agrees_with_gstools_within_its_tolerance() -> None:
    """GSTools' own fit of an exponential model with a nugget to the same experimental
    variogram. The fits' weights differ (APEX: Cressie's ``N / gamma_model^2``; GSTools:
    ``N`` here), so they agree within 25 % on the range and 10 % on the sill, not to
    round-off. GSTools' exponential is ``exp(-h / l)``: its practical range is ``3 l``.
    """
    gs = pytest.importorskip("gstools")
    _, data = _sample("exponential", 12.0, 12.0, 0.0)
    cells, values = data[0]
    xy = cells[:, :2].astype(np.float64)
    edges = np.linspace(0.0, 30.0, 21)
    experimental = experimental_variogram(xy, values, edges)

    mine = fit_variogram([experimental], "exponential")
    theirs = gs.Exponential(dim=2)
    theirs.fit_variogram(experimental.lags, experimental.gamma, nugget=True,
                         weights=experimental.counts.astype(np.float64),
                         max_eval=100_000)  # fmt: skip

    assert mine.major_range == pytest.approx(3.0 * theirs.len_scale, rel=0.25)
    assert mine.sill + mine.nugget == pytest.approx(theirs.var + theirs.nugget, rel=0.10)


def test_the_orthant_probability_is_scipys_bivariate_normal() -> None:
    """Against scipy's multivariate normal CDF, and the ends p^2 (rho 0) and p (rho 1)."""
    from scipy.special import ndtr
    from scipy.stats import multivariate_normal

    from apex.algorithms.geostatistics.variogram import orthant_probability

    for t in (-1.2, 0.0, 0.7):
        for rho in (0.0, 0.3, 0.8, 0.95):
            expected = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([t, t])
            assert orthant_probability(t, np.array([rho]))[0] == pytest.approx(expected, abs=1e-6)
        ends = orthant_probability(t, np.array([0.0, 1.0]))
        assert ends == pytest.approx([ndtr(t) ** 2, ndtr(t)], abs=1e-8)


@pytest.mark.parametrize("proportion", [0.5, 0.25])
def test_a_thresholded_fields_latent_range_is_recovered(proportion: float) -> None:
    """Facies from a latent gaussian field (range 12) cut at a proportion: the latent
    variogram behind the indicator gives the range back (median of 20 seeds within 20 %).
    The indicator's own fit does not: it is linear at the origin (the reviewer's 16 %
    short at p = 0.5, runaway fits at p = 0.16)."""
    from scipy.special import ndtri

    from apex.algorithms.geostatistics.variogram import latent_variogram

    factor, _ = _sample("gaussian", 12.0, 12.0, 0.0)
    threshold = float(ndtri(proportion))
    edges = np.linspace(0.0, 30.0, 21)
    ranges = []
    for seed in range(SEEDS):
        field = factor @ np.random.default_rng(seed).standard_normal(GRID * GRID)
        index = np.random.default_rng(1000 + seed).choice(GRID * GRID, POINTS, replace=False)
        j, i = np.divmod(index, GRID)
        below = (field[index] < threshold).astype(np.float64)
        indicator = experimental_variogram(np.c_[i, j].astype(np.float64), below, edges)
        latent = latent_variogram(indicator, float(below.mean()))
        ranges.append(fit_variogram([latent], "gaussian", max_range=60.0).major_range)

    assert np.median(ranges) == pytest.approx(12.0, rel=0.20)


def test_values_that_vary_only_between_layers_are_refused_plainly() -> None:
    cells = np.array([[i, j, k] for k in (1, 2) for i in range(1, 11) for j in range(1, 11)])
    values = (cells[:, 2] == 2).astype(np.float64)

    with pytest.raises(ValueError, match="vary only between layers"):
        WeightedLeastSquares().estimate(cells, values, anisotropic=False)


@pytest.mark.benchmark(group="geostatistics")
def test_an_anisotropic_estimate_of_1000_points(benchmark: pytest.FixtureRequest) -> None:
    """Hard data of 40 wells x 25 layers: eight directions of 20,000 same-layer pairs.

    2.1 s on the development workstation (2026-10-25), mostly the fits: three models from
    24 starts each (8 azimuths x 3 ranges). 5,000 points (50 wells x 100 layers) took 7.8 s.
    """
    rng = np.random.default_rng(4)
    wells = rng.choice(10_000, 40, replace=False)
    cells = np.array([[w % 100 + 1, w // 100 + 1, k] for w in wells for k in range(1, 26)])
    values = rng.standard_normal(cells.shape[0])

    found = benchmark(WeightedLeastSquares().estimate, cells, values)  # type: ignore[operator]

    assert found.n_data == 1000
