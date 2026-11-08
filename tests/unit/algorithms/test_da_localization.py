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


@pytest.mark.parametrize("cls", [PriorAdaptiveLocalization, AdaptiveLocalization])
@pytest.mark.parametrize("threshold", ["universal", "fixed"])
def test_unlocalized_variables_get_rows_of_ones_and_the_rest_is_unchanged(
    cls: type[PriorAdaptiveLocalization] | type[AdaptiveLocalization], threshold: str
) -> None:
    """The 5Spots MEC run (2026-11-07): the taper kept 0.0025% of the network's weights.
    ``unlocalized`` leaves a named variable's rows at 1; the other rows, and the run's
    localization stream, are those of the plain localization."""
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((12, 30)), rng.standard_normal((5, 30))
    sizes, names = [4, 6, 2], ["poro", "nn_params", "permx"]
    plain = cls(sizes, threshold=threshold)  # type: ignore[arg-type]
    exempt = cls(sizes, threshold=threshold, unlocalized=["nn_params"],  # type: ignore[arg-type]
                 variable_names=names)  # fmt: skip
    first, second = np.random.default_rng(1), np.random.default_rng(1)

    reference = plain._taper(a, b, first)
    taper = exempt._taper(a, b, second)

    assert first.random() == second.random()
    np.testing.assert_array_equal(taper[4:10], 1.0)
    np.testing.assert_array_equal(np.delete(taper, slice(4, 10), axis=0),
                                  np.delete(reference, slice(4, 10), axis=0))  # fmt: skip
    assert not np.all(reference[4:10] == 1.0)  # the exemption changed something


def test_an_unlocalized_variable_takes_the_unlocalized_rlmmac_step() -> None:
    """With every variable unlocalized the prior-taper step is the plain RLM-MAC step."""
    rng = np.random.default_rng(3)
    state = rng.standard_normal((6, 20))
    predicted = rng.standard_normal((4, 20))
    perturbed = rng.standard_normal((4, 20))
    reference = predicted.mean(axis=-1) + 0.1
    error = DataErrorCovariance.diagonal(np.ones(4))
    localization = PriorAdaptiveLocalization([2, 4], unlocalized=["a", "b"],
                                             variable_names=["a", "b"])  # fmt: skip
    localization.prepare(state, predicted, [], rng)

    plain = RLMMACUpdate(energy=1.0).step(
        state, predicted, perturbed, error, damping=0.5, reference_prediction=reference
    )
    localized = RLMMACUpdate(energy=1.0, localization=localization).step(
        state, predicted, perturbed, error, damping=0.5, reference_prediction=reference
    )

    np.testing.assert_allclose(localized, plain, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unlocalized": ["c"], "variable_names": ["a", "b"]}, r"\['c'\], which are not state"),
        ({"unlocalized": ["a"]}, "needs the variables' names"),
        ({"unlocalized": "a", "variable_names": ["a", "b"]}, "must be a list"),
    ],
)
def test_unlocalized_names_are_checked(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PriorAdaptiveLocalization([2, 3], **options)  # type: ignore[arg-type]


def _steps(loc: PriorAdaptiveLocalization | AdaptiveLocalization, path: str) -> tuple:  # type: ignore[type-arg]
    """``(localized, unlocalized)`` steps of one entry point on a non-square case."""
    rng = np.random.default_rng(8)
    n_state, n, n_data = 13, 17, 7
    dx = rng.standard_normal((n_state, n))
    gain = rng.standard_normal((n, n_data))
    r = rng.standard_normal((n_data, n))
    gains = [rng.standard_normal((n, n_data)) for _ in range(n)]
    sources = [rng.standard_normal((n, n)) for _ in range(2)]
    if isinstance(loc, PriorAdaptiveLocalization):
        loc.prepare(dx, rng.standard_normal((n_data, n)), [], rng)
    stream = np.random.default_rng(0)
    if path == "step":
        return loc.localized_step(dx, gain, r, stream), dx @ gain @ r
    if path == "members":
        plain = np.stack([dx @ g @ r[:, j] for j, g in enumerate(gains)], axis=-1)
        return loc.localized_member_steps(dx, iter(gains), r, stream), plain
    assert isinstance(loc, AdaptiveLocalization)
    return loc.localized_source_steps(dx, sources, stream), sum(dx @ p for p in sources)


@pytest.mark.parametrize(
    ("cls", "path"),
    [(PriorAdaptiveLocalization, "step"), (PriorAdaptiveLocalization, "members"),
     (AdaptiveLocalization, "step"), (AdaptiveLocalization, "members"),
     (AdaptiveLocalization, "sources")],
)  # fmt: skip
def test_unlocalized_rows_take_the_unlocalized_step_in_every_entry_point(
    cls: type[PriorAdaptiveLocalization] | type[AdaptiveLocalization], path: str
) -> None:
    """The numerics review: a mixed case (the middle block exempt) through each entry point
    (the gain, GIES's per-member gains, C-GIES's sources); the other rows stay tapered."""
    loc = cls([4, 5, 4], unlocalized=["b"], variable_names=["a", "b", "c"])
    localized, plain = _steps(loc, path)
    np.testing.assert_allclose(localized[4:9], plain[4:9], rtol=1e-12, atol=1e-12)
    assert not np.allclose(np.delete(localized, slice(4, 9), axis=0),
                           np.delete(plain, slice(4, 9), axis=0))  # fmt: skip


def test_an_unlocalized_block_is_not_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """The numerics review: no "updates are zeroed" warning about a block left unlocalized."""
    corr = np.full((3, 2), 0.1)
    shuffled = np.full((3, 2), 0.9)  # cutoff >= 1 in every block
    taper = adaptive_taper(corr, shuffled, [2, 1], 1.0, "hard", exempt=(False, True))
    np.testing.assert_array_equal(taper, [[0, 0], [0, 0], [1, 1]])
    assert caplog.text.count("block of 2") == 1
    taper = adaptive_taper(corr, shuffled, [1, 2], 1.0, "hard", exempt=(False, True))
    np.testing.assert_array_equal(taper[1:], 1.0)
    assert caplog.text.count("block of 2") == 1  # not warned again: exempt


def test_every_variable_unlocalized_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    PriorAdaptiveLocalization([2, 3], unlocalized=["a", "b"], variable_names=["a", "b"])
    assert "every state variable is unlocalized" in caplog.text


@pytest.mark.parametrize("threshold", ["universal", "fixed"])
def test_exempt_rows_are_not_correlated_and_the_rest_is_tapered_as_before(threshold: str) -> None:
    """The coder's review: the exempt rows' correlations are skipped (a network's would be two
    dense temporaries); the other rows' taper is the one over every row, and the stream is
    unchanged."""
    from apex.algorithms.data_assimilation.localization import correlation as corr

    rng = np.random.default_rng(9)
    a, b = rng.standard_normal((40, 25)), rng.standard_normal((6, 25))
    sizes, names = [10, 22, 8], ["poro", "nn_params", "permx"]
    loc = PriorAdaptiveLocalization(sizes, threshold=threshold,  # type: ignore[arg-type]
                                    unlocalized=["NN_PARAMS"], variable_names=names)  # fmt: skip
    first, second = np.random.default_rng(2), np.random.default_rng(2)
    taper = loc._taper(a, b, first)
    order = derangement(second, 25)
    fixed = 3 / np.sqrt(25) if threshold == "fixed" else None
    reference = adaptive_taper(
        corr(a, b), None if fixed else corr(a[:, order], b), sizes, 1.0, "hard",
        fixed=fixed, exempt=(False, True, False),
    )  # fmt: skip
    assert first.random() == second.random()
    np.testing.assert_array_equal(taper[10:32], 1.0)  # matched in any case
    np.testing.assert_allclose(taper, reference, rtol=0, atol=0)
