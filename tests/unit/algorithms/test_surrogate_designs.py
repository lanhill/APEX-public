"""Designs (validation 1): reproducible from the seed, held-out runs drawn first.

The Latin hypercube is checked against its definition (McKay, Beckman & Conover 1979): every
one of the n equal strata of every coordinate holds exactly one point.
"""

from collections.abc import Mapping

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.surrogates.designs import (
    ensemble_design,
    latin_hypercube,
    lhs_design,
    sampled_design,
)


def test_a_latin_hypercube_has_one_point_per_stratum_of_every_coordinate() -> None:
    lower, upper = np.array([0.0, -5.0, 10.0]), np.array([1.0, 5.0, 10.5])

    points = latin_hypercube(lower, upper, 40, np.random.default_rng(1))

    assert points.shape == (3, 40)
    assert points.flags.c_contiguous
    unit = (points - lower[:, None]) / (upper - lower)[:, None]
    assert np.all((unit >= 0) & (unit < 1))
    for row in unit:
        np.testing.assert_array_equal(np.sort(np.floor(row * 40)), np.arange(40))
    np.testing.assert_array_equal(
        points, latin_hypercube(lower, upper, 40, np.random.default_rng(1))
    )
    assert not np.array_equal(points, latin_hypercube(lower, upper, 40, np.random.default_rng(2)))


def test_an_lhs_design_keeps_its_held_out_runs_whatever_the_training_size() -> None:
    box = {"poro": (np.full(2, 0.1), np.full(2, 0.3)), "mult": (np.array([0.5]), np.array([2.0]))}

    small = lhs_design(box, 10, 5, np.random.default_rng(7))
    large = lhs_design(box, 30, 5, np.random.default_rng(7))

    assert small.plan == "lhs"
    for name in box:
        np.testing.assert_array_equal(small.held_out[name], large.held_out[name])
        assert small.training[name].shape == (box[name][0].size, 10)
        assert large.training[name].flags.c_contiguous
    # Each set is a hypercube of its own: the held-out runs' strata are complete too.
    unit = (small.held_out["mult"] - 0.5) / 1.5
    np.testing.assert_array_equal(np.sort(np.floor(unit[0] * 5)), np.arange(5))


def test_a_sampled_design_draws_the_held_out_runs_first() -> None:
    calls: list[int] = []

    def draw(n: int, rng: np.random.Generator) -> Mapping[str, NDArray[np.float64]]:
        calls.append(n)
        return {"permx": rng.normal(size=(4, n))}

    design = sampled_design(draw, 12, 3, np.random.default_rng(5))

    assert calls == [3, 12]
    assert design.plan == "prior"
    rng = np.random.default_rng(5)
    np.testing.assert_array_equal(design.held_out["permx"], rng.normal(size=(4, 3)))
    np.testing.assert_array_equal(design.training["permx"], rng.normal(size=(4, 12)))


def test_an_ensemble_design_takes_disjoint_columns_reproducibly() -> None:
    ensemble = {"x": np.arange(20.0).reshape(2, 10)}

    design = ensemble_design("run", ensemble, 3, np.random.default_rng(9), n_training=5)
    again = ensemble_design("run", ensemble, 3, np.random.default_rng(9))

    assert design.held_out_columns is not None
    assert design.training_columns is not None
    assert not set(design.held_out_columns) & set(design.training_columns)
    np.testing.assert_array_equal(design.held_out["x"], ensemble["x"][:, design.held_out_columns])
    np.testing.assert_array_equal(design.training["x"], ensemble["x"][:, design.training_columns])
    # The same held-out columns with all the rest for training.
    assert again.held_out_columns == design.held_out_columns
    assert again.training_columns is not None
    assert len(again.training_columns) == 7
    with pytest.raises(ValueError, match="fewer than"):
        ensemble_design("file", ensemble, 3, np.random.default_rng(9), n_training=8)
    with pytest.raises(ValueError, match="at least one"):
        lhs_design({"x": (np.zeros(1), np.ones(1))}, 5, 0, np.random.default_rng(0))


def test_held_out_runs_never_repeat_training_runs() -> None:
    """Review finding: a generator that ignores its stream (a stored prior gives its first
    columns) made the held-out runs the first training runs."""
    stored = np.arange(40.0).reshape(2, 20)

    def first_columns(n: int, rng: np.random.Generator) -> Mapping[str, NDArray[np.float64]]:
        return {"x": stored[:, :n]}

    with pytest.raises(ValueError, match=r"3 held-out run\(s\) repeat training runs"):
        sampled_design(first_columns, 10, 3, np.random.default_rng(0))
    repeated = {"x": np.hstack([stored[:, :5], stored[:, :5]])}  # every column twice
    with pytest.raises(ValueError, match="repeat training runs"):
        ensemble_design("file", repeated, 5, np.random.default_rng(1))


def test_coarse_only_runs_are_drawn_last_and_leave_the_others_unchanged() -> None:
    """Decision 13: coarse-only runs after the held-out and training runs, so a
    design without them is drawn exactly as before, for every plan."""
    box = {"x": (np.zeros(2), np.ones(2))}
    plain = lhs_design(box, 6, 3, np.random.default_rng(4))
    more = lhs_design(box, 6, 3, np.random.default_rng(4), n_coarse_only=8)
    assert plain.coarse_only is None
    assert more.coarse_only is not None
    assert more.coarse_only["x"].shape == (2, 8)
    np.testing.assert_array_equal(plain.held_out["x"], more.held_out["x"])
    np.testing.assert_array_equal(plain.training["x"], more.training["x"])
    unit = more.coarse_only["x"]  # a hypercube of its own
    np.testing.assert_array_equal(np.sort(np.floor(unit[0] * 8)), np.arange(8))

    calls: list[int] = []

    def draw(n: int, rng: np.random.Generator) -> Mapping[str, NDArray[np.float64]]:
        calls.append(n)
        return {"x": rng.normal(size=(2, n))}

    drawn = sampled_design(draw, 5, 2, np.random.default_rng(3), n_coarse_only=4)
    assert calls == [2, 5, 4]
    rng = np.random.default_rng(3)
    np.testing.assert_array_equal(drawn.held_out["x"], rng.normal(size=(2, 2)))
    np.testing.assert_array_equal(drawn.training["x"], rng.normal(size=(2, 5)))
    assert drawn.coarse_only is not None
    np.testing.assert_array_equal(drawn.coarse_only["x"], rng.normal(size=(2, 4)))

    ensemble = {"x": np.arange(24.0).reshape(2, 12)}
    without = ensemble_design("file", ensemble, 3, np.random.default_rng(9), n_training=5)
    within = ensemble_design("file", ensemble, 3, np.random.default_rng(9), 5, n_coarse_only=4)
    assert within.held_out_columns == without.held_out_columns
    assert within.training_columns == without.training_columns
    order = np.random.default_rng(9).permutation(12)
    assert within.coarse_only_columns == tuple(int(i) for i in order[8:12])
    assert within.coarse_only is not None
    np.testing.assert_array_equal(
        within.coarse_only["x"], ensemble["x"][:, within.coarse_only_columns]
    )
    with pytest.raises(ValueError, match="5 training and 5 coarse-only runs"):
        ensemble_design("file", ensemble, 3, np.random.default_rng(9), 5, n_coarse_only=5)
