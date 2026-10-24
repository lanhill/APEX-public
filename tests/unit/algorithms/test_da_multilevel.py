"""Multilevel ES-MDA: the gain kernel, guards and level assignment."""

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    ESMDA,
    GIES,
    DataErrorCovariance,
    HybridUpdate,
    LMEnRML,
    LocalAnalysisUpdate,
    MultilevelESMDA,
)
from apex.algorithms.data_assimilation.multilevel import hybrid_gain


def test_the_step_is_the_hybrid_gain_on_the_whitened_innovations() -> None:
    """The step never forms K (memory); it must equal K r."""
    rng = np.random.default_rng(1)
    state, predicted = rng.standard_normal((7, 12)), rng.standard_normal((5, 12))
    perturbed = rng.standard_normal((5, 12))
    error = DataErrorCovariance.diagonal(rng.uniform(0.1, 0.4, 5))
    levels = np.array([0] * 4 + [1] * 8)

    step = HybridUpdate(0.9).step(
        state, predicted, perturbed, error, levels=levels, weights=[0.25, 0.75], damping=0.5
    )

    gain = hybrid_gain(
        [state[:, :4], state[:, 4:]],
        [predicted[:, :4], predicted[:, 4:]],
        error,
        [0.25, 0.75],
        energy=0.9,
        damping=0.5,
    )
    np.testing.assert_allclose(step, gain @ error.whiten(perturbed - predicted), rtol=1e-12)


def test_levels_are_consecutive_member_ids_and_weights_are_normalised() -> None:
    method = MultilevelESMDA([2.0, 2.0], [{"members": 3, "weight": 1}, {"members": 5, "weight": 3}])

    assert method.weights == (0.25, 0.75)
    np.testing.assert_array_equal(method.level_of(np.array([0, 2, 3, 7])), [0, 0, 1, 1])
    with pytest.raises(ValueError, match="beyond"):
        method.level_of(np.array([8]))


def test_the_levels_and_strategies_are_guarded() -> None:
    with pytest.raises(ValueError, match="at least 2 members"):
        MultilevelESMDA([1.0], [{"members": 1, "weight": 1}])
    with pytest.raises(ValueError, match="weights"):
        MultilevelESMDA([1.0], [{"members": 3, "weight": -1}])
    with pytest.raises(ValueError, match="multilevel_esmda"):
        ESMDA([1.0], HybridUpdate())
    with pytest.raises(ValueError, match="multilevel_esmda"):
        LMEnRML(HybridUpdate())
    local = LocalAnalysisUpdate(block_sizes=[3])
    with pytest.raises(ValueError, match="GIES runs local analysis with RLM-MAC"):
        GIES(local)
    assert LMEnRML(local).strategy is local  # LM-/GN-EnRML accept local analysis


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ([{}], "level 0 needs members and weight"),
        ([{"members": 250}], "level 0 needs weight"),
        ([{"members": 3, "weight": 1.0}, {"weight": 1.0}], "level 1 needs members"),
    ],
)
def test_a_level_without_its_keys_says_which(levels: list[dict[str, float]], message: str) -> None:
    """Missing keys are named, not read as 0 members or NaN weights (release tester)."""
    with pytest.raises(ValueError, match=message):
        MultilevelESMDA([1.0], levels)  # type: ignore[arg-type]


def test_the_levels_have_a_typed_schema() -> None:
    """``apex plugins --json`` describes a level's keys (release tester, 2026-10-19)."""
    from apex.services.plugins import check_options, plugin_info

    schema = plugin_info("methods", "multilevel_esmda").options
    level = schema["$defs"]["Level"]
    assert set(level["required"]) == {"members", "weight"}
    assert {"upscale", "deck_template_file"} <= set(level["properties"])
    assert check_options("methods", "multilevel_esmda", {"inflation": [1.0], "levels": [{}]})
    assert not check_options(
        "methods",
        "multilevel_esmda",
        {"inflation": [2.0, 2.0], "levels": [{"members": 20, "weight": 0.5}] * 2},
    )
