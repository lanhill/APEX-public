"""``adaptive_rows``: ERT's per-row adaptive localization, native.

Independent references: a brute-force NumPy computation of each row's own restricted gain
(no grouping), and, where installed, ERT's library (``AdaptiveESMDA`` through
``ert_adaptive``) without truncation.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import AdaptiveRowsUpdate, update_strategies
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance

type Problem = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
                     DataErrorCovariance]  # fmt: skip


def _problem(seed: int, n_state: int = 30, n_data: int = 12, n: int = 40) -> Problem:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_state, n))
    forward = rng.standard_normal((n_data, n_state)) * (rng.random((n_data, n_state)) < 0.2)
    predicted = forward @ state + 0.3 * rng.standard_normal((n_data, n))
    variances = rng.uniform(0.5, 2.0, n_data)
    perturbed = rng.standard_normal(n_data)[:, None] + np.sqrt(variances)[:, None] * (
        rng.standard_normal((n_data, n))
    )
    return state, predicted, perturbed, DataErrorCovariance.diagonal(variances)


def _brute_force(
    state: NDArray[np.float64],
    predicted: NDArray[np.float64],
    perturbed: NDArray[np.float64],
    error: DataErrorCovariance,
    threshold: float,
    damping: float = 0.0,
) -> NDArray[np.float64]:
    """Each row's own gain: C_xd[i,S] (C_dd[S,S] + (1 + damping) C_d[S,S])^-1."""
    n = state.shape[1]
    dx = state - state.mean(1, keepdims=True)
    dy = predicted - predicted.mean(1, keepdims=True)
    corr = np.asarray(np.corrcoef(np.vstack([state, predicted])), dtype=np.float64)[
        : state.shape[0], state.shape[0] :
    ]
    step = np.zeros_like(state)
    for i in range(state.shape[0]):
        keep = np.flatnonzero(np.abs(corr[i]) > threshold)
        if keep.size == 0:
            continue
        c_xd = dx[i] @ dy[keep].T / (n - 1)
        c_dd = dy[keep] @ dy[keep].T / (n - 1)
        full = error.values if not error.is_diagonal else np.diag(error.values)
        c_d = (1 + damping) * full[np.ix_(keep, keep)]
        step[i] = c_xd @ np.linalg.solve(c_dd + c_d, (perturbed - predicted)[keep])
    return step


def _dense(error: DataErrorCovariance, seed: int) -> DataErrorCovariance:
    """A correlated C_d with the same variances (exponential correlation along the data)."""
    n_data = error.n_data
    lag = np.abs(np.subtract.outer(np.arange(n_data), np.arange(n_data)))
    sd = np.sqrt(error.values)
    return DataErrorCovariance.dense(np.outer(sd, sd) * np.exp(-lag / (2.0 + seed)))


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("damping", [0.0, 2.5])
def test_each_row_gets_its_own_restricted_gain(seed: int, dense: bool, damping: float) -> None:
    """LM-EnRML's damping enters as (1 + damping) C_d (review 2026-10-04)."""
    state, predicted, perturbed, error = _problem(seed)
    if dense:
        error = _dense(error, seed)
    threshold = 3 / np.sqrt(state.shape[1])
    step = AdaptiveRowsUpdate(energy=1.0).step(state, predicted, perturbed, error, damping=damping)
    expected = _brute_force(state, predicted, perturbed, error, threshold, damping)
    np.testing.assert_allclose(step, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("truncation", [0.9, 0.95, 0.98, 1.0])
@pytest.mark.parametrize(("n_data", "n"), [(10, 50), (60, 30)])
def test_it_equals_erts_library(truncation: float, n_data: int, n: int) -> None:
    """The truncation rule is ERT's (a fraction of the singular-value sum), n_data < N or > N."""
    pytest.importorskip("iterative_ensemble_smoother")
    from apex.integrations.ert.smoothers import ErtAdaptive

    for seed in range(3):
        state, predicted, perturbed, error = _problem(seed, n_state=25, n_data=n_data, n=n)
        ours = AdaptiveRowsUpdate(energy=truncation).step(state, predicted, perturbed, error)
        erts = ErtAdaptive(truncation=truncation).step(state, predicted, perturbed, error)
        np.testing.assert_allclose(ours, erts, rtol=1e-8, atol=1e-10)


def test_rows_without_spread_or_selection_are_not_updated_and_batches_agree() -> None:
    state, predicted, perturbed, error = _problem(4)
    state[3] = 1.5  # no spread
    strategy = AdaptiveRowsUpdate(energy=1.0, threshold=0.6)
    step = strategy.step(state, predicted, perturbed, error)
    assert np.all(step[3] == 0)
    mask = strategy.selection(state, predicted, np.arange(state.shape[0]))
    assert np.all(step[~mask.any(axis=1)] == 0)
    batched = AdaptiveRowsUpdate(energy=1.0, threshold=0.6, max_batch_mb=1e-4).step(
        state, predicted, perturbed, error
    )
    np.testing.assert_allclose(batched, step, rtol=1e-12, atol=1e-14)


def test_the_default_threshold_needs_ten_members() -> None:
    state, predicted, perturbed, error = _problem(5, n=8)
    with pytest.raises(ValueError, match="3/sqrt"):
        AdaptiveRowsUpdate().step(state, predicted, perturbed, error)
    assert isinstance(update_strategies.create("adaptive_rows"), AdaptiveRowsUpdate)
    with pytest.raises(ValueError, match="threshold"):
        AdaptiveRowsUpdate(threshold=1.0)


class _Recorder:
    """A diagnostics sink that keeps every item."""

    def __init__(self) -> None:
        self.items: dict[str, object] = {}

    def wants(self, role: str, name: str) -> bool:
        return True

    def add(self, role: str, name: str, item: object, *, nbytes: int | None = None) -> None:
        self.items[name] = item


def test_groups_count_distinct_subsets_across_batches_and_fixed_groups_are_refused() -> None:
    state, predicted, perturbed, error = _problem(6)
    one = AdaptiveRowsUpdate(energy=1.0, threshold=0.6)
    batched = AdaptiveRowsUpdate(energy=1.0, threshold=0.6, max_batch_mb=1e-4)
    sinks = [_Recorder(), _Recorder()]
    for strategy, sink in zip((one, batched), sinks, strict=True):
        strategy.attach_diagnostics(sink)
        strategy.step(state, predicted, perturbed, error)
    mask = one.selection(state, predicted, np.arange(state.shape[0]))
    subsets = {np.flatnonzero(row).tobytes() for row in mask if row.any()}
    assert len(subsets) > 1
    for sink in sinks:
        assert int(np.asarray(sink.items["groups"])) == len(subsets)
    with pytest.raises(TypeError, match="every update"):
        one.groups(state.shape[0], predicted.shape[0])


def test_selection_groups_keeps_first_appearance_order() -> None:
    from apex.algorithms.data_assimilation.local_analysis import selection_groups

    mask = np.array([[1, 0], [0, 1], [1, 0], [0, 0], [0, 1], [1, 1]], dtype=bool)
    groups = selection_groups(mask)
    assert [g[0].tolist() for g in groups] == [[0, 2], [1, 4], [5]]
    assert [g[1].tolist() for g in groups] == [[0], [1], [0, 1]]
    assert selection_groups(np.zeros((0, 3), dtype=bool)) == []


def test_a_spread_that_underflows_selects_nothing() -> None:
    """|x| ~ 1e-170 squares to 0, so the correlation is ±inf: not a selection (review)."""
    state, predicted, _, _ = _problem(7)
    state[2] = 1e-170 * np.random.default_rng(0).standard_normal(state.shape[1])
    mask = AdaptiveRowsUpdate(threshold=0.3).selection(state, predicted, np.arange(5))
    assert not mask[2].any()


@pytest.mark.parametrize("n_data", [1, 7, 8, 9, 650])
def test_selection_groups_equal_a_per_row_reference(n_data: int) -> None:
    """Grouping by packed bits (2026-10-04) equals grouping each row by its own data
    subset, in order of first appearance; widths not a multiple of 8 included.
    """
    from apex.algorithms.data_assimilation.local_analysis import selection_groups

    rng = np.random.default_rng(n_data)
    mask = rng.random((300, n_data)) < min(0.5, 3 / n_data)
    mask[::7] = mask[3]  # repeated patterns
    expected: dict[bytes, list[int]] = {}
    for row, pattern in enumerate(mask):
        if pattern.any():
            expected.setdefault(np.flatnonzero(pattern).tobytes(), []).append(row)
    groups = selection_groups(mask)
    assert [g[0].tolist() for g in groups] == list(expected.values())
    assert [g[1].tobytes() for g in groups] == [
        np.flatnonzero(mask[rows[0]]).astype(np.intp).tobytes() for rows in expected.values()
    ]
