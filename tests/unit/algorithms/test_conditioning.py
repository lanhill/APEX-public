"""Conditioning Gaussian fields to hard data by kriging residuals.

Independent reference: the conditional Gaussian in closed form. For ``x ~ N(m, C)`` and
data ``d = x_D + e``, ``e ~ N(0, R)``, the conditional mean is
``m + C_xD (C_DD + R)^-1 (d - m_D)`` and the covariance ``C - C_xD (C_DD + R)^-1 C_Dx``
(e.g. Rasmussen & Williams 2006, eq. 2.19 and A.6). The prior's draws are compared with it.
"""

import numpy as np
import pytest

from apex.algorithms.geostatistics import (
    condition_by_kriging,
    kriging_factor,
    layer_covariance,
    layer_cross_covariance,
)
from apex.algorithms.geostatistics.conditioning import JITTER, LARGE_CORRECTION, MAX_CONDITION
from apex.algorithms.priors import GaussianFieldPrior, GeostatsPySgsimPrior
from apex.domain import StateVariable

GRID = [6, 5, 2]
CELLS = [[2, 2, 1], [5, 4, 1], [3, 1, 2]]  # 1-based
VALUES = [1.5, -0.5, 2.0]
VARIOGRAM = {"variance": 2.0, "major_range": 4.0, "minor_range": 2.5, "azimuth": 30.0,
             "variogram": "spherical"}  # fmt: skip


def _layer_rows(layer: int) -> tuple[list[int], list[float]]:
    """Return a layer's data as (0-based columns within the layer, values)."""
    nx = GRID[0]
    found = [(c[0] - 1 + nx * (c[1] - 1), v) for c, v in zip(CELLS, VALUES, strict=True)
             if c[2] == layer + 1]  # fmt: skip
    return [r for r, _ in found], [v for _, v in found]


def _exact(layer: int, mean: float, error_std: float) -> tuple[np.ndarray, np.ndarray]:
    """The layer's conditional mean and covariance in closed form."""
    nx, ny, _ = GRID
    cov = layer_covariance(nx, ny, model="spherical", sill=2.0, major_range=4.0,
                           minor_range=2.5, azimuth=30.0)  # fmt: skip
    rows, values = _layer_rows(layer)
    c_dd = cov[np.ix_(rows, rows)] + error_std**2 * np.eye(len(rows))
    gain = np.linalg.solve(c_dd, cov[rows]).T
    return mean + gain @ (np.array(values) - mean), cov - gain @ cov[rows]


@pytest.mark.parametrize("error_std", [0.0, 0.4])
def test_conditioned_draws_follow_the_exact_conditional_gaussian(error_std: float) -> None:
    mean, members = 0.7, 40_000
    prior = GaussianFieldPrior(grid=GRID, mean=mean, **VARIOGRAM,  # type: ignore[arg-type]
                               conditioning={"cells": CELLS, "values": VALUES,
                                             "error_std": error_std})  # fmt: skip

    drawn = prior.generate(StateVariable(name="x", size=60), members, np.random.default_rng(1))

    layer = 30
    for k in range(2):
        block = drawn[k * layer : (k + 1) * layer]
        exact_mean, exact_cov = _exact(k, mean, error_std)
        # Monte Carlo errors: std/sqrt(N) for means, about sill*sqrt(2/N) for covariances.
        np.testing.assert_allclose(block.mean(axis=1), exact_mean,
                                   atol=5 * np.sqrt(2.0 / members))  # fmt: skip
        np.testing.assert_allclose(np.cov(block), exact_cov, atol=6 * 2.0 * np.sqrt(2 / members))
    if error_std == 0.0:
        rows = [c[0] - 1 + GRID[0] * (c[1] - 1) + 30 * (c[2] - 1) for c in CELLS]
        np.testing.assert_allclose(drawn[rows], np.array(VALUES)[:, None] * np.ones(members),
                                   rtol=0, atol=1e-12)  # fmt: skip


def test_exact_data_draw_nothing_more_and_leave_other_layers_alone() -> None:
    """Exact data add no random draws: the stream ends where the unconditioned one does, and
    a layer without data is the unconditioned layer bit for bit."""
    plain = GaussianFieldPrior(grid=GRID, **VARIOGRAM)  # type: ignore[arg-type]
    only_first = {"cells": CELLS[:2], "values": VALUES[:2]}
    conditioned = GaussianFieldPrior(grid=GRID, **VARIOGRAM, conditioning=only_first)  # type: ignore[arg-type]
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)

    a = plain.generate(StateVariable(name="x", size=60), 5, rng_a)
    b = conditioned.generate(StateVariable(name="x", size=60), 5, rng_b)

    assert rng_a.random() == rng_b.random()
    np.testing.assert_array_equal(a[30:], b[30:])
    assert not np.allclose(a[:30], b[:30])
    assert b.flags.c_contiguous


def test_close_data_of_a_smooth_model_are_still_honoured(caplog: pytest.LogCaptureFixture) -> None:
    """Adjacent cells of a long-range Gaussian model: a singular data covariance, factored
    with a tiny nugget; the data are still met to far below a std."""
    prior = GaussianFieldPrior(grid=[20, 1, 1], variogram="gaussian", major_range=30.0,
                               conditioning={"cells": [[i, 1, 1] for i in range(5, 12)],
                                             "values": [0.1 * i for i in range(7)]})  # fmt: skip

    drawn = prior.generate(StateVariable(name="x", size=20), 50, np.random.default_rng(2))

    np.testing.assert_allclose(drawn[4:11], np.arange(7)[:, None] * 0.1 * np.ones(50), atol=1e-3)


@pytest.mark.parametrize(
    ("conditioning", "message"),
    [
        ({"cells": [[7, 1, 1]], "values": [1.0]}, "outside the grid"),
        ({"cells": [[0, 1, 1]], "values": [1.0]}, "outside the grid"),
        ({"cells": [[1, 1, 1], [1, 1, 1]], "values": [1.0, 2.0]}, "given twice"),
        ({"cells": [[1, 1, 1]], "values": [1.0, 2.0]}, "one value each"),
        ({"cells": [[1, 1, 1]], "values": [float("nan")]}, "finite"),
        ({"cells": [[1, 1, 1]], "values": [1.0], "error_std": -1.0}, "non-negative"),
        ({"cells": [[1, 1, 1]], "values": [1.0], "noise": 1.0}, "takes cells"),
        ({"cells": [], "values": [1.0]}, "need their cells"),
    ],
)  # fmt: skip
def test_bad_conditioning_is_refused(conditioning: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GaussianFieldPrior(grid=GRID, conditioning=conditioning)


def test_an_uncorrelated_field_is_conditioned_at_its_cells_only() -> None:
    prior = GaussianFieldPrior(grid=[3, 3, 1], major_range=0.0,
                               conditioning={"cells": [[2, 2, 1]], "values": [9.0]})  # fmt: skip
    plain = GaussianFieldPrior(grid=[3, 3, 1], major_range=0.0)

    a = prior.generate(StateVariable(name="x", size=9), 4, np.random.default_rng(3))
    b = plain.generate(StateVariable(name="x", size=9), 4, np.random.default_rng(3))

    np.testing.assert_array_equal(a[4], 9.0)
    np.testing.assert_array_equal(np.delete(a, 4, axis=0), np.delete(b, 4, axis=0))


def test_cross_covariance_is_the_layer_covariance_columns() -> None:
    nx, ny = 7, 4
    columns = np.array([0, 9, 27])
    options = {"model": "exponential", "sill": 1.5, "major_range": 3.0, "minor_range": 1.5,
               "azimuth": -40.0}  # fmt: skip

    cross = layer_cross_covariance(nx, ny, columns, **options)  # type: ignore[arg-type]

    np.testing.assert_allclose(cross, layer_covariance(nx, ny, **options)[:, columns],  # type: ignore[arg-type]
                               rtol=1e-14, atol=1e-15)  # fmt: skip


def test_kriging_needs_noise_for_data_with_errors() -> None:
    with pytest.raises(ValueError, match="noise"):
        condition_by_kriging(np.zeros((3, 2)), np.eye(3)[:, :1], np.array([0]), np.ones(1),
                             np.ones(1))  # fmt: skip


def test_sgsim_realisations_take_the_data_exactly() -> None:
    pytest.importorskip("geostatspy")
    prior = GeostatsPySgsimPrior(grid=[12, 10, 1], mean=0.2, variance=0.01, major_range=5.0,
                                 conditioning={"cells": [[3, 3, 1], [9, 7, 1]],
                                               "values": [0.3, 0.1]})  # fmt: skip

    drawn = prior.generate(StateVariable(name="poro", size=120), 6, np.random.default_rng(4))

    np.testing.assert_allclose(drawn[[2 + 12 * 2, 8 + 12 * 6]], [[0.3] * 6, [0.1] * 6],
                               atol=1e-12)  # fmt: skip


def test_data_a_smooth_variogram_cannot_fit_are_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """Rough data a few cells apart under a long Gaussian variogram: the exact conditional
    field swings far outside the prior. It is logged, and the kriged mean's excursion (what
    the geomodeler refuses on) is far beyond LARGE_CORRECTION, while the same data under a
    short spherical variogram are ordinary."""
    cells = [[i, 1, 1] for i in (10, 13, 16, 19)]
    rough = {"cells": cells, "values": [0.0, 1.0, 0.0, 1.0]}
    smooth = GaussianFieldPrior(grid=[60, 1, 1], variogram="gaussian", major_range=40.0,
                                conditioning=rough)  # fmt: skip
    short = GaussianFieldPrior(grid=[60, 1, 1], major_range=10.0, conditioning=rough)

    with caplog.at_level("WARNING"):
        smooth.generate(StateVariable(name="x", size=60), 5, np.random.default_rng(0))

    assert "improbable under this variogram" in caplog.text
    assert smooth.hard is not None
    assert short.hard is not None
    assert smooth.hard.excursion(smooth) > LARGE_CORRECTION
    assert short.hard.excursion(short) < 3.0


def test_a_near_singular_data_covariance_is_jittered_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Adjacent cells of a very long Gaussian model: condition number beyond MAX_CONDITION,
    so a small nugget is added and the largest miss logged; the field stays bounded."""
    prior = GaussianFieldPrior(grid=[40, 1, 1], variogram="gaussian", major_range=200.0,
                               conditioning={"cells": [[i, 1, 1] for i in range(10, 16)],
                                             "values": [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]})  # fmt: skip

    with caplog.at_level("WARNING"):
        drawn = prior.generate(StateVariable(name="x", size=40), 5, np.random.default_rng(1))

    assert "near singular" in caplog.text
    assert np.all(np.isfinite(drawn))
    assert np.abs(drawn).max() < 1e3


def test_well_data_from_a_slightly_wrong_variogram_are_not_flagged() -> None:
    """25 wells x 20 layers drawn from a spherical field, judged under a sill 25 % too low
    (estimated sills routinely are): the kriged mean stays within a few prior stds, so the
    data are kept (a chi-square test refused 18 of 20 such draws)."""
    nx, ny, nz = 30, 30, 20
    rng = np.random.default_rng(8)
    truth = GaussianFieldPrior(grid=[nx, ny, nz], variance=1.0, major_range=8.0)
    field = truth.generate(StateVariable(name="x", size=nx * ny * nz), 2, rng)[:, 0]
    wells = rng.choice(nx * ny, 25, replace=False)
    cells = [[int(w % nx) + 1, int(w // nx) + 1, k + 1] for k in range(nz) for w in wells]
    values = [float(field[(c[2] - 1) * nx * ny + (c[0] - 1) + nx * (c[1] - 1)]) for c in cells]
    judged = GaussianFieldPrior(grid=[nx, ny, nz], variance=0.75, major_range=8.0,
                                conditioning={"cells": cells, "values": values})  # fmt: skip

    assert judged.hard is not None
    assert judged.hard.excursion(judged) < 0.5 * LARGE_CORRECTION


def test_the_jitter_is_relative_to_each_datums_variance() -> None:
    """Variables of very different scales (ln k std 1.5, porosity 0.03): the condition
    number is the correlation matrix's, and the nugget each datum's own share, so neither
    fires for scale alone and both are met alike when it does."""
    unit = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    stds = np.array([1.5, 0.03])
    covariance = unit * np.outer(stds, stds)

    _, condition, jittered = kriging_factor(covariance)

    assert condition == pytest.approx(np.linalg.cond(unit))
    assert not jittered
    near = np.array([[1.0, 1 - 1e-10], [1 - 1e-10, 1.0]]) * np.outer(stds, stds)
    factor, condition, jittered = kriging_factor(near)
    assert jittered
    assert condition > MAX_CONDITION
    restored = np.tril(factor[0]) @ np.tril(factor[0]).T
    np.testing.assert_allclose(np.diag(restored), (1 + JITTER) * stds**2, rtol=1e-12)
