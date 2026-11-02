"""The parameterization ``block_average``: property fields averaged onto a coarser grid.

References: hand-computed block means of a 4 x 2 x 2 grid, ragged blocks included; the
geometric mean of ``k`` equal to the arithmetic mean of ``ln k``; ``PropertyMap`` placing the
outputs on the coarse grid.
"""

import numpy as np
import pytest

from apex.algorithms.upscaling import BlockAverage, block_average, block_index
from apex.domain.state import StateLayout, StateVariable
from apex.services.plugins import check_options, is_private
from apex.workflows.parameterization import PropertyMap


def test_block_means_by_hand() -> None:
    # A 4 x 2 x 2 grid, values 1..16 in Eclipse order (i fastest), blocks of 2 x 2 x 1.
    index, n = block_index((4, 2, 2), (2, 2, 1))
    assert n == 4
    values = np.arange(1.0, 17.0)
    # Block (0, 0, 0): cells 1, 2, 5, 6; block (1, 0, 0): 3, 4, 7, 8; layer 2 adds 8.
    np.testing.assert_allclose(
        block_average(values, index, n, "arithmetic"), [3.5, 5.5, 11.5, 13.5]
    )
    np.testing.assert_allclose(
        block_average(values, index, n, "harmonic")[0], 4 / (1 + 1 / 2 + 1 / 5 + 1 / 6)
    )
    k = np.exp(np.random.default_rng(0).standard_normal(16))
    np.testing.assert_allclose(
        block_average(k, index, n, "geometric"),
        np.exp(block_average(np.log(k), index, n, "arithmetic")),
    )
    with pytest.raises(ValueError, match="positive"):
        block_average(values - 5.0, index, n, "geometric")


def test_ragged_blocks_take_what_is_left() -> None:
    index, n = block_index((5, 1, 1), (2, 1, 1))  # blocks of 2, 2, 1
    assert n == 3
    np.testing.assert_allclose(block_average(np.arange(1.0, 6.0), index, n, "arithmetic"),
                               [1.5, 3.5, 5.0])  # fmt: skip


def test_three_dimensional_ragged_blocks_match_a_direct_loop() -> None:
    dims, factors = (5, 3, 3), (2, 2, 2)
    values = np.random.default_rng(1).uniform(1.0, 2.0, 45)
    index, n = block_index(dims, factors)
    cx, cy, cz = 3, 2, 2
    expected = np.zeros(n)
    for c in range(n):
        bi, bj, bk = c % cx, (c // cx) % cy, c // (cx * cy)
        cells = [
            i + 5 * (j + 3 * k)
            for k in range(2 * bk, min(2 * bk + 2, 3))
            for j in range(2 * bj, min(2 * bj + 2, 3))
            for i in range(2 * bi, min(2 * bi + 2, 5))
        ]
        expected[c] = values[cells].mean()
    assert n == cx * cy * cz
    np.testing.assert_allclose(block_average(values, index, n, "arithmetic"), expected)


def test_the_parameterization_puts_its_outputs_on_the_coarse_grid() -> None:
    plugin = BlockAverage(dims=[4, 2, 2], factors=[2, 2, 1],
                          means={"poro": "arithmetic", "permx": "arithmetic"})  # fmt: skip
    assert plugin.output_grid == (2, 1, 2)
    layout = StateLayout([StateVariable("poro", 16, grid=(4, 2, 2)),
                          StateVariable("permx", 16, grid=(4, 2, 2))])  # fmt: skip
    mapped = PropertyMap([plugin], layout)
    assert mapped.grids == {"poro": (2, 1, 2), "permx": (2, 1, 2)}
    out = mapped.member({"poro": np.arange(1.0, 17.0), "permx": np.zeros(16)})
    np.testing.assert_allclose(out["poro"], [3.5, 5.5, 11.5, 13.5])
    with pytest.raises(ValueError, match="has 15 values"):
        plugin.properties({"poro": np.zeros(15), "permx": np.zeros(16)})


def test_options_are_checked_and_the_plugin_is_public() -> None:
    assert not is_private("parameterizations", "block_average")
    options = {"dims": [4, 2, 2], "factors": [2, 2, 1], "means": {"poro": "arithmetic"}}
    assert check_options("parameterizations", "block_average", options) == []
    with pytest.raises(ValueError, match="unknown means"):
        BlockAverage(dims=[2, 2, 1], factors=[1, 1, 1], means={"x": "median"})  # type: ignore[dict-item]
