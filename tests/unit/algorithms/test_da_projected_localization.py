"""Upstream PET's adaptive localization in the projected-data space (``adaptive_projected``).

PET parity is in ``tests/parity/test_approx_autoadaloc.py``; these check the kernels, the
limits and the composition rules.
"""

import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    GIES,
    ApproxUpdate,
    DataErrorCovariance,
    GNEnRML,
    LMEnRML,
    LocalAnalysisUpdate,
    ProjectedAdaptiveLocalization,
    ProjectedLocalization,
    RLMMACUpdate,
    localizations,
)
from apex.algorithms.data_assimilation.localization import (
    Taper,
    Threshold,
    pearson_correlation,
    threshold_taper,
)


def _case(
    seed: int = 0, n_state: int = 40, n_data: int = 15, n_ensemble: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DataErrorCovariance]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_state, n_ensemble))
    predicted = np.tanh(rng.standard_normal((n_data, n_state)) @ state / 4)
    error = DataErrorCovariance.diagonal(rng.uniform(0.05, 0.2, n_data))
    perturbed = rng.standard_normal(n_data)[:, None] + error.correlate(
        rng.standard_normal((n_data, n_ensemble))
    )
    return state, predicted, perturbed, error


def test_pearson_correlation_is_numpys_with_constant_rows_zeroed() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((5, 30)), rng.standard_normal((4, 30))
    a[2] = 3.0  # constant: PET zeroes its row
    b[1] = 1e-8 * rng.standard_normal(30)  # std below 1e-6: PET zeroes its column

    corr = pearson_correlation(a, b)

    with np.errstate(divide="ignore", invalid="ignore"):  # numpy's NaN, zeroed below
        expected = np.corrcoef(a, b)[:5, 5:]
    expected[2, :] = 0
    expected[:, 1] = 0
    np.testing.assert_allclose(corr, expected, atol=1e-14)


def test_the_thresholds_follow_pets_rules() -> None:
    rng = np.random.default_rng(2)
    corr = rng.uniform(-1, 1, (50, 3))
    shuffled = rng.uniform(-0.3, 0.3, (50, 3))
    noise = np.median(np.abs(shuffled), axis=0) / 0.6745
    levels: dict[Threshold, np.ndarray] = {
        "adaptive": 0.3 * noise,
        "fixed": np.full(3, 0.3),
        "universal": np.sqrt(2 * np.log(50)) * noise,
    }
    for threshold, level in levels.items():
        hard = threshold_taper(corr, shuffled, [50], threshold, 0.3, "hard")
        np.testing.assert_array_equal(hard, np.abs(corr) > level)
        # PET's sigmoid switches at the threshold, unlike PET-fork's.
        sigmoid = threshold_taper(corr, shuffled, [50], threshold, 0.3, "sigmoid")
        np.testing.assert_allclose(sigmoid, 1 / (1 + np.exp(-50 * (np.abs(corr) - level))))


def test_each_block_has_its_own_noise_level() -> None:
    corr = np.full((4, 1), 0.5)
    shuffled = np.array([[0.1], [0.1], [0.9], [0.9]])

    taper = threshold_taper(corr, shuffled, [2, 2], "universal", 0.3, "hard")

    # Block 1: sqrt(2 ln 2) * 0.1 / 0.6745 = 0.17 < 0.5; block 2: 1.56 >= 1, zeroed.
    np.testing.assert_array_equal(taper[:, 0], [1, 1, 0, 0])


@pytest.mark.parametrize("empirical", [False, True])
def test_a_taper_of_ones_gives_the_unlocalized_step(empirical: bool) -> None:
    state, predicted, perturbed, error = _case()
    # A fixed threshold below every |rho| keeps everything (hard taper of ones).
    localization = ProjectedAdaptiveLocalization([40], threshold="fixed", cutoff=1e-12)
    localized = ApproxUpdate(0.95, localization, empirical_data_error=empirical)
    plain = ApproxUpdate(0.95, empirical_data_error=empirical)

    for damping in (0.0, 2.0):
        step = localized.step(
            state, predicted, perturbed, error, damping=damping, rng=np.random.default_rng(3)
        )
        expected = plain.step(state, predicted, perturbed, error, damping=damping)
        np.testing.assert_allclose(step, expected, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize("taper", ["hard", "soft", "sigmoid"])
def test_the_step_is_reproducible_from_the_generator(taper: Taper) -> None:
    state, predicted, perturbed, error = _case(seed=4)
    strategy = ApproxUpdate(
        0.95, ProjectedAdaptiveLocalization([20, 20], threshold="universal", taper=taper)
    )

    def step(seed: int) -> np.ndarray:
        return strategy.step(state, predicted, perturbed, error, rng=np.random.default_rng(seed))

    np.testing.assert_array_equal(step(5), step(5))
    assert not np.array_equal(step(5), step(6))  # the shuffle sets the noise level


def test_the_options_and_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="block_sizes"):
        ProjectedAdaptiveLocalization([])
    with pytest.raises(ValueError, match="threshold"):
        ProjectedAdaptiveLocalization([3], threshold="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cutoff"):
        ProjectedAdaptiveLocalization([3], cutoff=0)
    with pytest.raises(ValueError, match="fixed threshold"):
        ProjectedAdaptiveLocalization([3], threshold="fixed", cutoff=1.0)
    with pytest.raises(ValueError, match="taper"):
        ProjectedAdaptiveLocalization([3], taper="sigm")  # type: ignore[arg-type]
    localization = ProjectedAdaptiveLocalization([3])
    dx = np.ones((3, 4))
    with pytest.raises(ValueError, match="random generator"):
        localization.projected_step(dx, np.ones((2, 4)), np.ones((2, 4)), None)
    with pytest.raises(ValueError, match=r"\(r, 4\)"):
        localization.projected_step(dx, np.ones((2, 4)), np.ones((3, 4)), np.random.default_rng(0))
    with pytest.raises(ValueError, match="only the approx update"):
        localization.localized_step(dx, np.ones((4, 2)), np.ones((2, 4)), None)


def test_only_the_approx_update_and_its_methods_take_it() -> None:
    localization = ProjectedAdaptiveLocalization([3])
    assert isinstance(localization, ProjectedLocalization)
    assert localizations.factory("adaptive_projected") is ProjectedAdaptiveLocalization
    with pytest.raises(ValueError, match="projected-data space"):
        RLMMACUpdate(localization=localization)
    with pytest.raises(ValueError, match="projected-data space"):
        LocalAnalysisUpdate(localization=localization)
    # Upstream PET localizes LM-EnRML and GN-EnRML (approx) in this space.
    LMEnRML(strategy=ApproxUpdate(localization=localization))
    GNEnRML(strategy=ApproxUpdate(localization=localization))
    GIES(strategy=ApproxUpdate(localization=localization))


def test_rows_below_pets_absolute_std_floor_are_reported(caplog: pytest.LogCaptureFixture) -> None:
    state, predicted, perturbed, error = _case()
    state[0] = 1e-9 * state[0]  # e.g. a compressibility: below PET's 1e-6 floor
    strategy = ApproxUpdate(0.95, ProjectedAdaptiveLocalization([40]))

    step = strategy.step(state, predicted, perturbed, error, rng=np.random.default_rng(0))

    assert np.all(step[0] == 0)  # PET's behaviour, kept for parity
    assert "not updated" in caplog.text
