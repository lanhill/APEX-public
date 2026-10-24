"""Performance-metric kernels and plugins."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from skimage.metrics import structural_similarity as _structural_similarity

from apex.algorithms.metrics import (
    ModelRMSE,
    ModelSSIM,
    NormalizedDataMismatch,
    normalized_l2,
    ssim,
)
from apex.core.metrics import MetricContext
from apex.core.plugins import metrics
from apex.domain.observations import DataKey, ObservationSet

# scikit-image is untyped; the independent SSIM reference.
structural_similarity = cast(Callable[..., float], _structural_similarity)


@pytest.mark.parametrize(
    ("shape", "win_size"), [((40,), 3), ((30, 25), 7), ((12, 11, 5), 5), ((20, 15, 9), 7)]
)
def test_ssim_equals_scikit_image(shape: tuple[int, ...], win_size: int) -> None:
    """Independent reference: scikit-image's structural_similarity, member by member."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape)
    members = x[..., None] + 0.5 * rng.standard_normal((*shape, 4))
    data_range = float(max(x.max(), members.max()) - min(x.min(), members.min()))
    expected = [
        structural_similarity(x, members[..., j], data_range=data_range, win_size=win_size)
        for j in range(4)
    ]
    np.testing.assert_allclose(ssim(x, members, data_range, win_size), expected, atol=1e-14)
    np.testing.assert_allclose(ssim(x, x[..., None], data_range, win_size), 1.0, atol=1e-14)


def test_ssim_refuses_windows_that_do_not_fit() -> None:
    with pytest.raises(ValueError, match="exceeds the grid"):
        ssim(np.zeros((10, 10, 2)), np.zeros((10, 10, 2, 3)), 1.0, 7)
    for win_size in (4, 1):  # scikit-image refuses 1 too
        with pytest.raises(ValueError, match="odd"):
            ssim(np.zeros((10, 10)), np.zeros((10, 10, 3)), 1.0, win_size)


def test_normalized_l2_is_an_rmse() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    ensemble = reference[:, None] + np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(normalized_l2(reference, ensemble), [1.0, 1.0])


def _context(state: dict[str, np.ndarray], truth: dict[str, np.ndarray] | None) -> MetricContext:
    keys = [DataKey("t0", "a"), DataKey("t1", "a"), DataKey("t0", "b")]
    observations = ObservationSet(keys, [1.0, 2.0, 3.0], [1.0, 4.0, 0.25])
    predicted = np.array([[1.0, 2.0], [2.0, 0.0], [3.0, 3.5]])
    return MetricContext(state, predicted, observations, truth)


def test_normalized_data_mismatch_is_pet_forks_mean_square() -> None:
    """``mean(((d_obs - y) / sigma)^2)``, as PET-fork's calculate_data_mismatch."""
    observations = ObservationSet(
        [DataKey("t0", "a"), DataKey("t1", "a"), DataKey("t0", "b")],
        [1.0, 2.0, 3.0],
        [1.0, 4.0, 0.25],
    )
    # member 0 fits exactly; member 1: residuals (-1, 4, -1.5) / (1, 2, 0.5) = (-1, 2, -3)
    predicted = np.array([[1.0, 2.0], [2.0, -2.0], [3.0, 4.5]])
    context = MetricContext({"x": np.zeros((2, 2))}, predicted, observations, None)
    result = NormalizedDataMismatch().evaluate(context)
    np.testing.assert_allclose(result["all"], [0.0, (1 + 4 + 9) / 3])
    np.testing.assert_allclose(result["a"], [0.0, (1 + 4) / 2])
    np.testing.assert_allclose(result["b"], [0.0, 9.0])


def test_rmse_per_variable_and_overall() -> None:
    truth = {"x": np.zeros(2), "y": np.zeros(3)}
    state = {"x": np.full((2, 2), 1.0), "y": np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])}
    result = ModelRMSE().evaluate(_context(state, truth))
    np.testing.assert_allclose(result["x"], [1.0, 1.0])
    np.testing.assert_allclose(result["y"], [0.0, 2.0])
    np.testing.assert_allclose(result["all"], [np.sqrt(2 / 5), np.sqrt(14 / 5)])
    with pytest.raises(ValueError, match="needs the true model"):
        ModelRMSE().evaluate(_context(state, None))
    with pytest.raises(ValueError, match="unknown state variables"):
        ModelRMSE(variables=["z"]).evaluate(_context(state, truth))
    # actnum applies to the variables of its size only (x: 2 values; y is unmasked).
    masked = ModelRMSE(actnum=[1, 0]).evaluate(_context(state, truth))
    np.testing.assert_allclose(masked["x"], [1.0, 1.0])
    np.testing.assert_allclose(masked["y"], [0.0, 2.0])
    with pytest.raises(ValueError, match="actnum has 2 cells"):
        ModelRMSE(variables=["y"], actnum=[1, 0]).evaluate(_context(state, truth))


def test_ssim_per_layer_for_thin_grids() -> None:
    """A 10x10x2 grid: 3-D windows cannot fit, per-layer 2-D SSIM does."""
    rng = np.random.default_rng(1)
    truth = rng.standard_normal(200)
    state = {"permx": truth[:, None] + 0.3 * rng.standard_normal((200, 3))}
    with pytest.raises(ValueError, match="per_layer"):
        ModelSSIM(grid=[10, 10, 2])
    metric = ModelSSIM(grid=[10, 10, 2], per_layer=True, win_size=3)
    result = metric.evaluate(_context(state, {"permx": truth}))["permx"]

    field = truth.reshape(10, 10, 2, order="F")
    members = state["permx"].reshape(10, 10, 2, 3, order="F")
    data_range = float(
        max(truth.max(), state["permx"].max()) - min(truth.min(), state["permx"].min())
    )
    expected = [
        np.mean(
            [
                structural_similarity(
                    field[..., k], members[..., k, j], data_range=data_range, win_size=3
                )
                for k in range(2)
            ]
        )
        for j in range(3)
    ]
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_ssim_picks_the_field_variables() -> None:
    truth = {"permx": np.ones(64) + np.arange(64), "skin": np.zeros(2)}
    state = {"permx": truth["permx"][:, None] * np.ones((1, 2)), "skin": np.zeros((2, 2))}
    result = ModelSSIM(grid=[8, 8], win_size=3).evaluate(_context(state, truth))
    assert set(result) == {"permx"}  # only variables with prod(grid) values
    np.testing.assert_allclose(result["permx"], 1.0)


def test_metrics_are_registered() -> None:
    assert {"normalized_data_mismatch", "rmse", "ssim"} <= set(metrics.names())
    assert ModelRMSE.requires_truth
    assert ModelSSIM.requires_truth
    assert not NormalizedDataMismatch.requires_truth


def test_ssim_takes_the_grids_of_the_state_variables() -> None:
    rng = np.random.default_rng(2)
    truth = {"permx": rng.standard_normal(64), "poro": rng.standard_normal(64), "skin": np.zeros(1)}
    state = {name: v[:, None] + 0.2 * rng.standard_normal((v.size, 3)) for name, v in truth.items()}
    context = MetricContext(
        state,
        np.zeros((3, 3)),
        ObservationSet(
            [DataKey("t0", "a"), DataKey("t1", "a"), DataKey("t0", "b")], [1, 2, 3], [1, 1, 1]
        ),
        truth,
        grids={"permx": (8, 8), "poro": (4, 16)},
    )
    result = ModelSSIM(win_size=3).evaluate(context)
    assert set(result) == {"permx", "poro"}  # the gridded variables only
    explicit = ModelSSIM(grid=[4, 16], win_size=3).evaluate(context)
    np.testing.assert_allclose(result["poro"], explicit["poro"])
    with pytest.raises(ValueError, match="no grid for skin"):
        ModelSSIM(variables=["skin"], win_size=3).evaluate(context)
    with pytest.raises(ValueError, match="does not fit"):
        ModelSSIM(win_size=5).evaluate(context)  # poro's 4 rows


def test_quantity_metrics_leave_categorical_outputs_out_unless_named() -> None:
    """Facies codes are not quantities; facies_mismatch scores them."""
    rng = np.random.default_rng(0)
    state = {"facies": rng.integers(0, 3, (6, 4)).astype(float), "poro": rng.random((6, 4))}
    truth = {"facies": np.zeros(6), "poro": np.full(6, 0.2)}
    observations = ObservationSet([DataKey("t", "y")], np.zeros(1), np.ones(1))
    context = MetricContext(
        state=state, predicted=np.zeros((1, 4)), observations=observations, truth=truth,
        categorical=frozenset({"facies"}),
    )  # fmt: skip

    assert set(metrics.create("rmse").evaluate(context)) == {"poro", "all"}
    assert set(metrics.create("rmse", variables=["facies"]).evaluate(context)) == {"facies", "all"}
    mismatch = metrics.create("facies_mismatch").evaluate(context)["facies"]
    np.testing.assert_allclose(mismatch, (state["facies"] != 0).mean(axis=0))
    only_codes = MetricContext(
        state={"facies": state["facies"]}, predicted=np.zeros((1, 4)), observations=observations,
        truth={"facies": truth["facies"]}, categorical=frozenset({"facies"}),
    )  # fmt: skip
    assert metrics.create("rmse").evaluate(only_codes) == {}


def _facies_context(ensemble: list[list[int]], truth: list[int]) -> MetricContext:
    observations = ObservationSet([DataKey("t", "y")], np.zeros(1), np.ones(1))
    return MetricContext(
        state={"facies": np.array(ensemble, dtype=float)},
        predicted=np.zeros((1, len(ensemble[0]))),
        observations=observations, truth={"facies": np.array(truth, dtype=float)},
        categorical=frozenset({"facies"}),
    )  # fmt: skip


def test_the_facies_brier_score_and_its_baselines_by_hand() -> None:
    # Two cells, four members, two facies. Cell 0 (truth 0): p = (3/4, 1/4); cell 1 (truth 1):
    # p = (1/2, 1/2). Brier: cell 0 (1/16 + 1/16), cell 1 (1/4 + 1/4): mean = 0.3125.
    context = _facies_context([[0, 0, 0, 1], [0, 1, 0, 1]], [0, 1])

    scores = {k: float(v[0]) for k, v in metrics.create("facies_brier").evaluate(context).items()}

    assert scores["facies"] == pytest.approx(0.3125)
    assert scores["facies:all_0"] == pytest.approx(1.0)  # 2 x half the cells are not 0
    assert scores["facies:all_1"] == pytest.approx(1.0)
    # Overall proportions (5/8, 3/8) in both cells: (9/64 + 9/64 + 25/64 + 25/64) / 2.
    assert scores["facies:proportions"] == pytest.approx(68 / 128)
    assert scores["facies:skill"] == pytest.approx(1 - 0.3125 / (68 / 128))
    assert all(v.shape == (4,) for v in metrics.create("facies_brier").evaluate(context).values())


def test_a_perfect_ensemble_scores_best_and_an_absent_facies_stays_finite() -> None:
    perfect = _facies_context([[0, 0, 0], [2, 2, 2]], [0, 2])  # facies 1 in no member

    brier = metrics.create("facies_brier").evaluate(perfect)
    log = metrics.create("facies_log_score").evaluate(perfect)

    assert brier["facies"][0] == 0.0
    # Smoothed p_truth = (3 + 0.5) / (3 + 0.5 x 3) = 7/9 in both cells.
    assert log["facies"][0] == pytest.approx(-np.log(7 / 9))
    assert all(np.isfinite(v).all() for v in log.values())
    wrong = _facies_context([[1, 1, 1], [0, 0, 0]], [0, 2])
    assert metrics.create("facies_log_score").evaluate(wrong)["facies"][0] > log["facies"][0]
