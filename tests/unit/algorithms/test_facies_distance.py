"""Signed distances and ``facies_distance``: exact references."""

import numpy as np
import pytest

from apex.algorithms.facies import FaciesDistance, facies_from_distances, signed_distances
from apex.core.plugins import parameterizations


def test_a_half_plane_has_the_analytic_signed_distance() -> None:
    """Facies 1 in columns i < 20 of a 40 x 10 layer: d_1 = 20 - i inside, -(i - 19) outside.

    (Cell-centre distances: the nearest other-facies cell of column i < 20 is column 20.)
    """
    nx, ny = 40, 10
    i = np.tile(np.arange(nx), ny)  # Eclipse order: i fastest
    codes = (i < 20).astype(np.int64)
    d = signed_distances(codes, (nx, ny, 1), 2, cell_size=(2.0, 5.0, 1.0))
    expected = np.where(i < 20, 20 - i, -(i - 19)) * 2.0
    np.testing.assert_allclose(d[1], expected)
    np.testing.assert_allclose(d[0], -expected)


def test_a_disc_matches_brute_force_distances() -> None:
    """Every cell's signed distance against the brute-force minimum over the other side."""
    nx = ny = 21
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    inside = ((i - 10) ** 2 + (j - 10) ** 2 <= 6**2).ravel(order="F")
    x, y = i.ravel(order="F") * 1.5, j.ravel(order="F") * 1.0
    d = signed_distances(inside.astype(np.int64), (nx, ny, 1), 2, cell_size=(1.5, 1.0, 1.0))
    expected = np.empty(nx * ny)
    for c in range(nx * ny):
        other = inside != inside[c]
        nearest = np.min(np.hypot(x[other] - x[c], y[other] - y[c]))
        expected[c] = nearest if inside[c] else -nearest
    np.testing.assert_allclose(d[1], expected, rtol=1e-12)


def test_argmax_recovers_any_realisation_exactly() -> None:
    rng = np.random.default_rng(3)
    grid = (12, 9, 4)
    codes = rng.integers(0, 3, size=int(np.prod(grid)))
    codes[: 12 * 9] = 1  # a layer filled by one facies; the others absent there
    d = signed_distances(codes, grid, 3, cell_size=(10.0, 20.0, 2.0))
    np.testing.assert_array_equal(facies_from_distances(d), codes)
    layer = slice(0, 12 * 9)
    assert np.all(d[1][layer] > 0)
    assert np.all(d[0][layer] < 0)


def test_signed_distances_check_their_input() -> None:
    with pytest.raises(ValueError, match="cells"):
        signed_distances(np.zeros(5, dtype=int), (2, 2, 1), 2)
    with pytest.raises(ValueError, match=r"0\.\.1"):
        signed_distances(np.full(4, 2), (2, 2, 1), 2)


def test_facies_distance_maps_to_properties() -> None:
    plugin = parameterizations.create(
        "facies_distance",
        fields=["d_shale", "d_sand"],
        facies=["shale", "sand"],
        values={"shale": {"poro": 0.05}, "sand": {"poro": 0.25}},
        property_fields={"permx": {"shale": "k_shale", "sand": "k_sand"}},
    )
    assert isinstance(plugin, FaciesDistance)
    assert plugin.outputs == ("facies", "poro", "permx")
    state = {
        "d_shale": np.array([1.0, -1.0, -2.0]),
        "d_sand": np.array([-1.0, 1.0, 3.0]),
        "k_shale": np.array([1.0, 2.0, 3.0]),
        "k_sand": np.array([100.0, 200.0, 300.0]),
    }
    out = plugin.properties(state)
    np.testing.assert_array_equal(out["facies"], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(out["poro"], [0.05, 0.25, 0.25])
    np.testing.assert_array_equal(out["permx"], [1.0, 200.0, 300.0])
    with pytest.raises(ValueError, match="needs"):
        plugin.properties({"d_shale": state["d_shale"]})
    with pytest.raises(ValueError, match="one field per facies"):
        FaciesDistance(fields=["a"], facies=["x", "y"])


def test_a_band_bounds_an_absent_facies_and_keeps_recovery_exact() -> None:
    """A facies missing from a layer sits at -band, not at minus the layer's diagonal."""
    grid = (30, 20, 2)
    codes = np.zeros(30 * 20 * 2, dtype=np.int64)
    codes[600 + 5 : 600 + 25] = 1  # facies 1 only in the second layer
    unbounded = signed_distances(codes, grid, 2)
    assert unbounded[1][:600].min() == pytest.approx(-np.hypot(30, 20))
    bounded = signed_distances(codes, grid, 2, band=4.0)
    assert bounded[1][:600].min() == -4.0
    assert np.abs(bounded).max() <= 4.0
    np.testing.assert_array_equal(facies_from_distances(bounded), codes)
    with pytest.raises(ValueError, match="positive"):
        signed_distances(codes, grid, 2, cell_size=(0.0, 1.0, 1.0))
