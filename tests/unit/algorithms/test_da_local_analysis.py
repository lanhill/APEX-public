"""Local analysis: independent checks (PET's localanalysis does not run)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    ApproxUpdate,
    DataErrorCovariance,
    LocalAnalysisUpdate,
)
from apex.algorithms.data_assimilation.local_analysis import selection_groups
from apex.algorithms.data_assimilation.localization import AdaptiveLocalization


class _FixedSelection:
    """A localization stand-in whose prepared taper is given."""

    def __init__(self, taper: NDArray[np.float64]) -> None:
        self.taper_matrix = taper

    def prepare(self, *args: object) -> None:
        pass

    def localized_step(self, *args: object) -> NDArray[np.float64]:
        raise AssertionError("local analysis does not taper a global gain")

    def export_state(self) -> dict[str, NDArray[np.float64]]:
        return {"taper": self.taper_matrix}

    def restore_state(self, arrays: object) -> None:
        pass


def _problem(
    rng: np.random.Generator, n_state: int = 6, n_data: int = 9, n: int = 25
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], DataErrorCovariance]:
    state = rng.standard_normal((n_state, n))
    predicted = np.tanh(rng.standard_normal((n_data, n_state)) @ state)
    perturbed = rng.standard_normal((n_data, n))
    return state, predicted, perturbed, DataErrorCovariance.diagonal(rng.uniform(0.1, 0.5, n_data))


def test_selecting_every_datum_is_the_approx_update() -> None:
    state, predicted, perturbed, error = _problem(np.random.default_rng(1))
    local = LocalAnalysisUpdate(energy=0.9, localization=_FixedSelection(np.ones((6, 9))))

    step = local.step(state, predicted, perturbed, error, damping=2.0)

    expected = ApproxUpdate(energy=0.9).step(state, predicted, perturbed, error, damping=2.0)
    np.testing.assert_allclose(step, expected, rtol=1e-12, atol=1e-14)


def test_grouping_equals_one_local_update_per_parameter() -> None:
    rng = np.random.default_rng(2)
    state, predicted, perturbed, error = _problem(rng)
    mask = rng.uniform(size=(6, 9)) > 0.5
    mask[3] = mask[1]  # two rows with the same subset form one group
    mask[5] = False  # a row with no data is not updated
    local = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(mask.astype(float)))

    step = local.step(state, predicted, perturbed, error)

    for i in range(6):
        data = np.flatnonzero(mask[i])
        if not data.size:
            np.testing.assert_array_equal(step[i], 0.0)
            continue
        one = ApproxUpdate(energy=1.0).step(
            state[[i]],
            predicted[data],
            perturbed[data],
            DataErrorCovariance.diagonal(error.values[data]),
        )
        np.testing.assert_allclose(step[i], one[0], rtol=1e-11, atol=1e-13)
    assert len(local.groups(6, 9)) == len({tuple(r) for r in mask if r.any()})


def test_decoupled_subsystems_are_updated_as_two_separate_problems() -> None:
    """Parameters A affect only data A and B only data B; with the block mask, local
    analysis equals the approx update of each subsystem alone (a global update with 30
    members would also use the spurious A-B sample correlations)."""
    rng = np.random.default_rng(3)
    n = 30
    state = rng.standard_normal((4, n))
    g_a, g_b = rng.standard_normal((5, 2)), rng.standard_normal((4, 2))
    predicted = np.vstack([g_a @ state[:2], g_b @ state[2:]])
    perturbed = predicted.mean(axis=1, keepdims=True) + rng.standard_normal((9, n))
    variances = rng.uniform(0.1, 0.3, 9)
    mask = np.zeros((4, 9))
    mask[:2, :5] = mask[2:, 5:] = 1.0
    local = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(mask))

    step = local.step(state, predicted, perturbed, DataErrorCovariance.diagonal(variances))

    approx = ApproxUpdate(energy=1.0)
    a = approx.step(
        state[:2], predicted[:5], perturbed[:5], DataErrorCovariance.diagonal(variances[:5])
    )
    b = approx.step(
        state[2:], predicted[5:], perturbed[5:], DataErrorCovariance.diagonal(variances[5:])
    )
    np.testing.assert_allclose(step, np.vstack([a, b]), rtol=1e-12, atol=1e-14)
    whole = approx.step(state, predicted, perturbed, DataErrorCovariance.diagonal(variances))
    assert np.abs(whole - step).max() > 1e-3  # the global update differs


def test_a_soft_taper_can_weight_the_local_increment() -> None:
    rng = np.random.default_rng(4)
    state, predicted, perturbed, error = _problem(rng)
    taper = np.full((6, 9), 0.5)
    weighted = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(taper), weight=True)
    plain = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(taper))

    np.testing.assert_allclose(
        weighted.step(state, predicted, perturbed, error),
        0.5 * plain.step(state, predicted, perturbed, error),
        rtol=1e-12,
    )


def test_selection_groups() -> None:
    mask = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)

    groups = selection_groups(mask)

    assert [(list(r), list(d)) for r, d in groups] == [([0, 2], [0, 2]), ([3], [1])]


def test_guards() -> None:
    with pytest.raises(ValueError, match="adaptive_per_update"):
        LocalAnalysisUpdate(localization=AdaptiveLocalization([3]))
    with pytest.raises(ValueError, match="block_sizes"):
        LocalAnalysisUpdate()
    local = LocalAnalysisUpdate(block_sizes=[6])
    state, predicted, perturbed, error = _problem(np.random.default_rng(5))
    with pytest.raises(ValueError, match="not prepared"):
        local.step(state, predicted, perturbed, error)


def test_distance_selects_the_data_within_the_taper_support() -> None:
    """Gaspari-Cohn with length c = 1 vanishes at distance 2c: W1 at cell 1 selects cells 1
    and 2, W2 at cell 5 cells 4 and 5, and the middle cell selects nothing."""
    from apex.algorithms.data_assimilation.localization import DistanceLocalization
    from apex.domain.observations import DataKey

    selection = DistanceLocalization(
        grid=[5, 1, 1],
        positions={"W1": [[1, 1, 1]], "W2": [[5, 1, 1]]},
        radius=1.0,
        block_sizes=[5],
    )
    keys = [DataKey("t0", "WOPR:W1"), DataKey("t0", "WOPR:W2")]
    rng = np.random.default_rng(0)
    selection.prepare(rng.standard_normal((5, 10)), rng.standard_normal((2, 10)), keys, rng)
    local = LocalAnalysisUpdate(localization=selection)

    groups = [(list(r), list(d)) for r, d in local.groups(5, 2)]

    assert groups == [([0, 1], [0]), ([3, 4], [1])]


def test_correlation_selection_drops_independent_parameters() -> None:
    """A 50-cell field where only cells 0-9 drive the data, 500 members: the driving cells
    select every datum; the independent ones almost none (the universal threshold,
    ``sqrt(2 ln 50) sigma``, admits noise rarely). The threshold is per state variable, so
    the selection is meant for fields, not single scalars."""
    from apex.algorithms.data_assimilation.localization import PriorAdaptiveLocalization
    from apex.domain.observations import DataKey

    rng = np.random.default_rng(6)
    n = 500
    prior = rng.standard_normal((50, n))
    forward = np.zeros((6, 50))
    forward[:, :10] = rng.uniform(0.5, 1.5, (6, 10))
    predicted = forward @ prior + 0.3 * rng.standard_normal((6, n))
    keys = [DataKey(f"t{i}", "y") for i in range(6)]
    selection = PriorAdaptiveLocalization([50], taper="hard")
    selection.prepare(prior, predicted, keys, rng)
    local = LocalAnalysisUpdate(localization=selection)

    mask = np.zeros((50, 6), dtype=bool)
    for rows, data in local.groups(50, 6):
        mask[np.ix_(rows, data)] = True

    assert mask[:10].all()
    assert mask[10:].mean() < 0.05


def test_a_dense_inflated_c_d_uses_the_selected_block() -> None:
    """step[G] = dX[G] dY_J^T (dY_J dY_J^T + alpha C_d[J, J] (1 + lambda))^-1 (D - Y)_J with no
    truncation: the closed form of the local approx update (energy 1, lambda folds as
    (1 + lambda) into the data-error term)."""
    rng = np.random.default_rng(7)
    state, predicted, perturbed, _ = _problem(rng, n=40)
    a = rng.standard_normal((9, 9))
    covariance = 0.1 * (a @ a.T + 9 * np.eye(9))
    error = DataErrorCovariance.dense(covariance).inflated(3.0)
    mask = np.zeros((6, 9))
    mask[:3, [0, 2, 4, 6]] = 1.0
    mask[3:, 5:] = 1.0
    damping = 0.7
    local = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(mask))

    step = local.step(state, predicted, perturbed, error, damping=damping)

    dx = (state - state.mean(axis=1, keepdims=True)) / np.sqrt(39)
    dy = (predicted - predicted.mean(axis=1, keepdims=True)) / np.sqrt(39)
    for rows, data in ((slice(0, 3), [0, 2, 4, 6]), (slice(3, 6), [5, 6, 7, 8])):
        c = 3.0 * covariance[np.ix_(data, data)] * (1 + damping)
        expected = (
            dx[rows]
            @ dy[data].T
            @ np.linalg.solve(dy[data] @ dy[data].T + c, (perturbed - predicted)[data])
        )
        np.testing.assert_allclose(step[rows], expected, rtol=1e-9, atol=1e-11)


def test_decoupled_subsystems_approach_each_kalman_posterior() -> None:
    """Two independent linear-Gaussian subsystems, 4000 members: the local update's
    posterior mean and covariance of each tend to that subsystem's Kalman posterior."""
    rng = np.random.default_rng(8)
    n = 4000
    g_a, g_b = rng.standard_normal((5, 2)), rng.standard_normal((4, 2))
    variances = rng.uniform(0.1, 0.3, 9)
    truth = rng.standard_normal(4)
    observed = np.concatenate([g_a @ truth[:2], g_b @ truth[2:]]) + np.sqrt(
        variances
    ) * rng.standard_normal(9)
    prior = rng.standard_normal((4, n))
    predicted = np.vstack([g_a @ prior[:2], g_b @ prior[2:]])
    error = DataErrorCovariance.diagonal(variances)
    perturbed = observed[:, None] + error.perturbations(rng, n)
    mask = np.zeros((4, 9))
    mask[:2, :5] = mask[2:, 5:] = 1.0
    local = LocalAnalysisUpdate(energy=1.0, localization=_FixedSelection(mask))

    posterior = prior + local.step(prior, predicted, perturbed, error)

    for rows, g, data in ((slice(0, 2), g_a, slice(0, 5)), (slice(2, 4), g_b, slice(5, 9))):
        gain = g.T @ np.linalg.inv(g @ g.T + np.diag(variances[data]))  # prior N(0, I)
        mean, cov = gain @ observed[data], np.eye(2) - gain @ g
        # Monte Carlo error with 4000 members: ~0.02 on the mean (5 sigma), ~0.01 on cov.
        np.testing.assert_allclose(posterior[rows].mean(axis=1), mean, atol=0.05)
        np.testing.assert_allclose(np.cov(posterior[rows]), cov, atol=0.02)


def test_the_local_rlmmac_with_every_datum_is_rlmmac() -> None:
    from apex.algorithms.data_assimilation import LocalRLMMACUpdate, RLMMACUpdate

    state, predicted, perturbed, error = _problem(np.random.default_rng(9))
    reference = predicted.mean(axis=1) + 0.05
    local = LocalRLMMACUpdate(energy=0.9, localization=_FixedSelection(np.ones((6, 9))))

    step = local.step(
        state, predicted, perturbed, error, damping=0.3, reference_prediction=reference
    )

    expected = RLMMACUpdate(energy=0.9).step(
        state, predicted, perturbed, error, damping=0.3, reference_prediction=reference
    )
    np.testing.assert_allclose(step, expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="reference_prediction"):
        local.step(state, predicted, perturbed, error)


def test_the_local_rlmmac_damps_each_group_by_its_own_spectrum() -> None:
    """gamma_g = lambda * mean(s_g^2) per group: each group equals RLM-MAC on its rows."""
    from apex.algorithms.data_assimilation import LocalRLMMACUpdate, RLMMACUpdate

    state, predicted, perturbed, error = _problem(np.random.default_rng(10))
    reference = predicted.mean(axis=1) - 0.02
    mask = np.zeros((6, 9))
    mask[:3, :4] = mask[3:, 4:] = 1.0
    local = LocalRLMMACUpdate(energy=1.0, localization=_FixedSelection(mask))

    step = local.step(
        state, predicted, perturbed, error, damping=0.5, reference_prediction=reference
    )

    for rows, data in ((slice(0, 3), slice(0, 4)), (slice(3, 6), slice(4, 9))):
        expected = RLMMACUpdate(energy=1.0).step(
            state[rows],
            predicted[data],
            perturbed[data],
            DataErrorCovariance.diagonal(error.values[data]),
            damping=0.5,
            reference_prediction=reference[data],
        )
        np.testing.assert_allclose(step[rows], expected, rtol=1e-12, atol=1e-14)
