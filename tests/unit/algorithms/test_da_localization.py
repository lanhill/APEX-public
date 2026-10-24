import numpy as np
import pytest

from apex.algorithms.data_assimilation import (
    AdaptiveLocalization,
    DataErrorCovariance,
    DistanceLocalization,
    PriorAdaptiveLocalization,
    RLMMACUpdate,
    localizations,
)
from apex.algorithms.data_assimilation.localization import (
    Taper,
    _fix_fixed_points,
    adaptive_taper,
    correlation,
    derangement,
    gaspari_cohn,
)
from apex.domain import DataKey


@pytest.mark.parametrize(
    "permutation", [[0, 1, 2, 3], [2, 1, 0], [1, 0, 2], [3, 1, 2, 0], [1, 2, 0], [0]]
)
def test_fixed_points_are_removed_and_the_result_stays_a_permutation(
    permutation: list[int],
) -> None:
    result = _fix_fixed_points(np.array(permutation))

    assert sorted(result.tolist()) == list(range(len(permutation)))
    if len(permutation) > 1:
        assert not np.any(result == np.arange(len(permutation)))


def test_derangement_is_reproducible_from_the_generator() -> None:
    a = derangement(np.random.default_rng(1), 50)
    b = derangement(np.random.default_rng(1), 50)

    np.testing.assert_array_equal(a, b)
    assert not np.any(a == np.arange(50))


def test_correlation_matches_numpy_and_zeroes_rows_without_spread() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((4, 30)), rng.standard_normal((3, 30))
    a[0] = 5.0

    corr = correlation(a, b)

    np.testing.assert_allclose(corr[1:], np.corrcoef(a[1:], b)[:3, 3:], atol=1e-12)
    np.testing.assert_array_equal(corr[0], 0.0)


def test_gaspari_cohn_is_one_at_zero_and_vanishes_beyond_two() -> None:
    np.testing.assert_allclose(gaspari_cohn(np.array([0.0, 2.0, 3.0])), [1.0, 0.0, 0.0], atol=1e-15)
    values = gaspari_cohn(np.linspace(0, 2, 50))
    assert np.all(np.diff(values) <= 1e-15)  # monotone decreasing


def test_hard_taper_keeps_correlations_above_the_universal_threshold() -> None:
    corr = np.array([[0.9], [0.01], [-0.8], [0.02]])
    shuffled = np.full((4, 1), 0.1)

    taper = adaptive_taper(corr, shuffled, [4], threshold_scale=1.0, taper="hard")

    # threshold = sqrt(2 ln 4) * 0.1 / 0.6745 = 0.247
    np.testing.assert_array_equal(taper[:, 0], [1.0, 0.0, 1.0, 0.0])


def test_blocks_are_thresholded_separately() -> None:
    corr = np.array([[0.3], [0.3], [0.3], [0.3]])
    shuffled = np.array([[0.01], [0.01], [0.5], [0.5]])

    taper = adaptive_taper(corr, shuffled, [2, 2], threshold_scale=1.0, taper="hard")

    np.testing.assert_array_equal(taper[:, 0], [1.0, 1.0, 0.0, 0.0])


def test_localized_rlmmac_without_effective_tapering_equals_the_unlocalized_step() -> None:
    # A taper of ones must reproduce the unlocalized update exactly (both code paths).
    rng = np.random.default_rng(3)
    state = rng.standard_normal((6, 20))
    predicted = rng.standard_normal((4, 20))
    perturbed = rng.standard_normal((4, 20))
    reference = predicted.mean(axis=-1) + 0.1
    error = DataErrorCovariance.diagonal(np.ones(4))
    # Radius 1e9 cells: the Gaspari-Cohn taper is 1 - O(1e-18), i.e. 1 in double precision.
    distance = DistanceLocalization([6, 1, 1], {"P": [[3, 1, 1]]}, 1e9, [6])
    keys = [DataKey(f"t{i}", "WOPR:P") for i in range(4)]
    distance.prepare(state, predicted, keys, rng)

    plain = RLMMACUpdate(energy=1.0).step(
        state, predicted, perturbed, error, damping=0.5, reference_prediction=reference
    )
    localized = RLMMACUpdate(energy=1.0, localization=distance).step(
        state, predicted, perturbed, error, damping=0.5, reference_prediction=reference
    )

    np.testing.assert_allclose(localized, plain, rtol=1e-12, atol=1e-14)


def test_distance_taper_follows_wells_layers_and_unknown_wells() -> None:
    local = DistanceLocalization(
        [5, 4, 2],
        {"P1": [[1, 1, 1], [1, 1, 2]], "P2": [[5, 4, 2]]},
        1.5,
        [40, 40],
        unlisted="zero",
    )
    keys = [DataKey("t", "WOPR:P1"), DataKey("t", "WWPR:P2"), DataKey("t", "FOPR")]

    local.prepare(np.zeros((80, 10)), np.zeros((3, 10)), keys, np.random.default_rng(0))

    taper = local.taper_matrix
    assert taper is not None
    assert taper.shape == (80, 3)
    p1 = taper[:40, 0].reshape(2, 4, 5)  # (k, j, i)
    assert p1[0, 0, 0] == 1.0
    assert p1[1, 0, 0] == 1.0
    assert p1[0, 3, 4] == 0.0
    p2 = taper[:40, 1].reshape(2, 4, 5)
    assert p2[1, 3, 4] == 1.0
    assert p2[0].max() == 0.0  # same layer only (APEX's 3-D extension; PET is 2-D)
    np.testing.assert_array_equal(taper[:, 2], 0.0)  # field data: no well, no update
    np.testing.assert_array_equal(taper[40:], taper[:40])  # every variable tapered alike


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid": [5, 4]}, "grid"),
        ({"block_sizes": [39]}, "nx\\*ny\\*nz"),
        ({"positions": {"P": [[6, 1, 1]]}}, "outside the grid"),
        ({"radius": 0.0}, "radius"),
        ({"taper": "box"}, "taper"),
    ],
)
def test_distance_localization_validates_its_options(
    kwargs: dict[str, object], message: str
) -> None:
    options: dict[str, object] = {
        "grid": [5, 4, 2],
        "positions": {"P": [[1, 1, 1]]},
        "radius": 2.0,
        "block_sizes": [40],
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        DistanceLocalization(**options)  # type: ignore[arg-type]


def test_gain_tapers_must_be_prepared_and_adaptive_needs_a_generator() -> None:
    zeros = np.zeros((3, 4))
    with pytest.raises(ValueError, match="prepare"):
        PriorAdaptiveLocalization([3]).localized_step(
            zeros, np.zeros((4, 2)), np.zeros((2, 4)), None
        )
    with pytest.raises(ValueError, match="random generator"):
        AdaptiveLocalization([3]).localized_step(zeros, np.zeros((4, 2)), np.zeros((2, 4)), None)


def test_prior_taper_round_trips_through_export_and_restore() -> None:
    rng = np.random.default_rng(4)
    original = PriorAdaptiveLocalization([5], taper="soft")
    original.prepare(rng.standard_normal((5, 20)), rng.standard_normal((3, 20)), [], rng)

    restored = PriorAdaptiveLocalization([5], taper="soft")
    restored.restore_state(original.export_state())

    assert restored.taper_matrix is not None
    np.testing.assert_array_equal(restored.taper_matrix, original.export_state()["taper"])


def test_localizations_are_registered() -> None:
    assert {"adaptive", "adaptive_per_update", "adaptive_prior", "distance"} <= set(
        localizations.names()
    )
    # The default "adaptive" is prior-only; "adaptive_prior" is the same plugin.
    assert localizations.factory("adaptive") is localizations.factory("adaptive_prior")
    assert localizations.factory("adaptive_per_update") is AdaptiveLocalization


def test_distance_localization_refuses_data_without_well_positions() -> None:
    local = DistanceLocalization([5, 4, 1], {"PRO1": [[1, 1, 1]]}, 1.5, [20])
    keys = [DataKey("t", "WOPR:PRO1"), DataKey("t", "WOPR:PROD2")]

    with pytest.raises(ValueError, match="no positions for PROD2"):
        local.prepare(np.zeros((20, 5)), np.zeros((2, 5)), keys, np.random.default_rng(0))


def test_distance_taper_is_rebuilt_from_its_exported_state() -> None:
    keys = [DataKey("t", "WOPR:P1"), DataKey("t", "WOPR:P2")]
    positions = {"P1": [[1, 1, 1]], "P2": [[5, 4, 1]]}
    original = DistanceLocalization([5, 4, 1], positions, 2.0, [20], taper="furrer_bengtsson")
    original.prepare(np.zeros((20, 7)), np.zeros((2, 7)), keys, np.random.default_rng(0))

    restored = DistanceLocalization([5, 4, 1], positions, 2.0, [20], taper="furrer_bengtsson")
    restored.restore_state(original.export_state())

    assert "taper" not in original.export_state()  # geometry, not a dense matrix, is saved
    assert restored.taper_matrix is not None
    np.testing.assert_array_equal(restored.taper_matrix, original.taper_matrix)


def test_columns_whose_cutoff_reaches_one_get_no_update() -> None:
    # PET's soft taper would reverse here (more noise, less localization).
    corr = np.array([[0.0, 0.0], [0.3, 0.3], [0.6, 0.6], [0.95, 0.95]])
    shuffled = np.array([[0.01, 1.0], [0.01, 1.0], [0.01, 1.0], [0.01, 1.0]])

    tapers: tuple[Taper, ...] = ("hard", "soft", "sigmoid")
    for taper in tapers:
        result = adaptive_taper(corr, shuffled, [4], 1.0, taper)
        np.testing.assert_array_equal(result[:, 1], 0.0)
        assert result[:, 0].max() > 0


def test_distance_localization_refuses_field_data() -> None:
    """Field components (and their wavelet coefficients) have no well to measure from."""
    local = DistanceLocalization([5, 4, 1], {"PRO1": [[1, 1, 1]]}, 1.5, [20])
    keys = [DataKey("t", "WOPR:PRO1"), DataKey("t", "wavelet(avo)", 0)]

    with pytest.raises(ValueError, match=r"cannot localize field data: wavelet\(avo\)@t"):
        local.prepare(np.zeros((20, 5)), np.zeros((2, 5)), keys, np.random.default_rng(0))


def test_a_fixed_threshold_keeps_what_exceeds_it_whatever_the_noise() -> None:
    """``threshold = "fixed"`` (the user, 2026-10-20): ``|rho| > cutoff`` everywhere."""
    corr = np.array([[0.31, -0.29], [0.9, -0.5]])

    taper = adaptive_taper(corr, None, [2], 1.0, "hard", fixed=0.3)

    np.testing.assert_array_equal(taper, [[1.0, 0.0], [1.0, 1.0]])


def test_the_fixed_threshold_defaults_to_three_over_root_n() -> None:
    rng = np.random.default_rng(3)
    state, forecast = rng.standard_normal((40, 100)), rng.standard_normal((6, 100))
    forecast[0] = state[0] + 0.1 * rng.standard_normal(100)  # one strong correlation
    loc = PriorAdaptiveLocalization([40], threshold="fixed")

    loc.prepare(state, forecast, [], np.random.default_rng(0))

    from apex.algorithms.data_assimilation.localization import correlation

    assert loc._fixed_cutoff(100) == pytest.approx(0.3)
    expected = (np.abs(correlation(state, forecast)) > 3 / np.sqrt(100)).astype(float)
    np.testing.assert_array_equal(loc.taper_matrix, expected)
    assert expected[0].any()


def test_the_per_update_plugin_takes_the_fixed_threshold_too() -> None:
    """``adaptive_per_update`` correlates ``dX`` with ``X r``: ``N`` columns, ``N`` members."""
    from apex.algorithms.data_assimilation.localization import AdaptiveLocalization, correlation

    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((50, 30)), rng.standard_normal((30, 30))
    loc = AdaptiveLocalization([20, 30], threshold="fixed")

    taper = loc._taper(a, b, np.random.default_rng(0))

    np.testing.assert_array_equal(
        taper, (np.abs(correlation(a, b)) > 3 / np.sqrt(30)).astype(float)
    )


def test_the_universal_default_is_unchanged_and_the_sigmoid_centres_on_the_cutoff() -> None:
    corr = np.array([[0.3, 0.1], [0.5, 0.9]])
    shuffled = np.array([[0.05, -0.1], [0.08, 0.02]])
    universal = adaptive_taper(corr, shuffled, [2], 1.0, "hard")
    assert np.array_equal(universal, adaptive_taper(corr, shuffled, [2], 1.0, "hard", fixed=None))
    sigmoid = adaptive_taper(corr, None, [2], 1.0, "sigmoid", fixed=0.3)
    assert sigmoid[0, 0] == pytest.approx(0.5)


def test_the_fixed_threshold_draws_the_same_stream_as_the_universal_one() -> None:
    """Choosing the threshold leaves the run's localization stream where it was."""
    rng = np.random.default_rng(5)
    state, forecast = rng.standard_normal((10, 20)), rng.standard_normal((4, 20))
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    PriorAdaptiveLocalization([10]).prepare(state, forecast, [], a)
    PriorAdaptiveLocalization([10], threshold="fixed").prepare(state, forecast, [], b)
    assert a.random() == b.random()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"cutoff": 0.3}, "threshold = 'fixed'"),
        ({"threshold": "fixed", "threshold_scale": 0.5}, "give the threshold itself"),
        ({"threshold": "fixed", "cutoff": 1.2}, r"\(0, 1\)"),
        ({"threshold": "adaptive"}, "'universal' or 'fixed'"),
    ],
)
def test_inconsistent_threshold_options_are_refused(
    options: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PriorAdaptiveLocalization([4], **options)  # type: ignore[arg-type]


def test_three_over_root_n_needs_ten_members() -> None:
    loc = PriorAdaptiveLocalization([4], threshold="fixed")
    with pytest.raises(ValueError, match="keeps nothing with 9 members"):
        loc._fixed_cutoff(9)
