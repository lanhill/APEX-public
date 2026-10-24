"""Wavelet sparse representation: kernels, fitted fields and the ``wavelet`` transform."""

from typing import Literal

import numpy as np
import pytest

from apex.algorithms.data_compression import (
    WaveletCompression,
    WaveletOptions,
    decompose,
    fit_field,
    recompose,
)
from apex.core.plugins import data_transforms
from apex.domain.observations import DataKey, ObservationSet


def _smooth(grid: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axes = np.meshgrid(*[np.linspace(0, 1, n) for n in grid], indexing="ij")
    field = np.zeros(grid)
    for _ in range(4):
        phase = np.pi * sum(rng.uniform(0.5, 3) * a for a in axes) + rng.uniform(0, 6)
        field += rng.normal() * np.cos(phase)
    return field


@pytest.mark.parametrize(
    ("grid", "wavelet", "level"),
    [((37,), "db2", 3), ((16, 11), "haar", 2), ((9, 7, 5), "sym3", 1), ((8, 8, 4), "db2", 2)],
)
def test_decompose_and_recompose_are_inverse(
    grid: tuple[int, ...], wavelet: str, level: int
) -> None:
    options = WaveletOptions(grid=grid, wavelet=wavelet, level=level)
    signal = np.random.default_rng(1).standard_normal((*grid, 3))
    coefficients, layout = decompose(signal, options)
    assert coefficients.shape == (layout.n_coefficients, 3)
    assert layout.keys[0] == ""
    assert len(layout.shapes) == 1 + level * (2 ** len(grid) - 1)
    np.testing.assert_allclose(recompose(coefficients, layout, options), signal, atol=1e-12)


def test_batched_decomposition_equals_member_by_member() -> None:
    options = WaveletOptions(grid=(12, 10), wavelet="db2", level=2)
    signal = np.random.default_rng(2).standard_normal((12, 10, 4))
    batched, _ = decompose(signal, options)
    for j in range(4):
        single, _ = decompose(signal[..., j : j + 1], options)
        np.testing.assert_array_equal(batched[:, j], single[:, 0])


@pytest.mark.parametrize("order", ["C", "F"])
def test_zero_threshold_keeps_every_coefficient_and_reconstructs_exactly(
    order: Literal["C", "F"],
) -> None:
    grid = (16, 12)
    field = np.random.default_rng(3).standard_normal(grid).reshape(-1, order=order)
    options = WaveletOptions(
        grid=grid, wavelet="haar", level=2, order=order, threshold_multiplier=0
    )
    fitted = fit_field(field, options)
    assert fitted.n_retained == fitted.layout.n_coefficients  # no exact zeros in random data
    np.testing.assert_allclose(
        fitted.reconstruct(fitted.observed[:, None])[:, 0], field, atol=1e-12
    )


def test_mask_follows_the_data_order() -> None:
    """Data and mask are both flat in ``order`` (PET flattens the mask in C order, D2)."""
    grid = (10, 8, 3)
    rng = np.random.default_rng(4)
    values = rng.standard_normal(grid)
    mask = rng.uniform(size=grid) > 0.3
    fill = -2.0
    options = WaveletOptions(grid=grid, wavelet="haar", level=1, order="F", threshold_multiplier=0)
    data = values.reshape(-1, order="F")[mask.reshape(-1, order="F")]
    fitted = fit_field(data, options, active=mask, inactive_value=fill)

    expected, _ = decompose(np.where(mask, values, fill)[..., None], options)
    np.testing.assert_allclose(fitted.coefficients(data[:, None]), expected, atol=1e-13)
    np.testing.assert_allclose(
        fitted.reconstruct(fitted.coefficients(data[:, None])[fitted.retained]),
        data[:, None],
        atol=1e-12,
    )


def test_white_noise_level_is_estimated_from_the_finest_diagonal_band() -> None:
    """MAD estimate (Donoho & Johnstone 1994) of the noise on a smooth field.

    Haar, whose detail coefficients of white noise are exactly white. Longer wavelets add
    boundary coefficients (symmetric extension) with less noise, which biases the MAD low
    (about 5% for db4 on 128x128); see the notes in docs/migration/pet_parity.md.
    """
    grid, sigma = (128, 128), 0.1
    rng = np.random.default_rng(5)
    field = _smooth(grid).ravel() + sigma * rng.standard_normal(128 * 128)
    fitted = fit_field(field, WaveletOptions(grid=grid, wavelet="haar", level=3))
    assert np.unique(fitted.noise_std).size == 1
    assert fitted.noise_std[0] == pytest.approx(sigma, rel=0.05)
    # Smooth fields are sparse: thresholding keeps a small fraction of the coefficients.
    assert fitted.n_retained < 0.1 * fitted.layout.n_coefficients
    # And the retained ones reproduce the field to about the noise level.
    rebuilt = fitted.reconstruct(fitted.observed[:, None])[:, 0]
    assert np.sqrt(np.mean((rebuilt - _smooth(grid).ravel()) ** 2)) < sigma


@pytest.mark.parametrize("thresholding", ["hard", "soft"])
def test_the_observation_goes_through_the_same_operator_as_predictions(thresholding: str) -> None:
    """Compressing the observed field gives the fitted observation (D1: soft shrinks both)."""
    grid = (32, 16)
    field = _smooth(grid).ravel() + 0.05 * np.random.default_rng(6).standard_normal(32 * 16)
    options = WaveletOptions(grid=grid, wavelet="db2", level=2, thresholding=thresholding)  # type: ignore[arg-type]
    fitted = fit_field(field, options)
    np.testing.assert_allclose(fitted.compress(field[:, None])[:, 0], fitted.observed, atol=1e-14)
    if thresholding == "soft":
        assert np.all(np.abs(fitted.observed) > 0)
        raw = fitted.coefficients(field[:, None])[fitted.retained, 0]
        np.testing.assert_allclose(np.abs(raw) - np.abs(fitted.observed), fitted.thresholds)


def test_keep_approximation_retains_the_whole_approximation_band() -> None:
    grid = (16, 16)
    field = _smooth(grid).ravel()
    fitted = fit_field(
        field,
        WaveletOptions(
            grid=grid, wavelet="haar", level=2, noise="coloured", keep_approximation=True
        ),
    )
    n_approx = int(np.prod(fitted.layout.shapes[0]))
    np.testing.assert_array_equal(fitted.retained[:n_approx], np.arange(n_approx))
    np.testing.assert_array_equal(fitted.thresholds[:n_approx], 0.0)


def test_coloured_noise_has_one_level_per_band() -> None:
    grid = (32, 32)
    field = _smooth(grid).ravel() + 0.05 * np.random.default_rng(7).standard_normal(32 * 32)
    fitted = fit_field(
        field,
        WaveletOptions(
            grid=grid, wavelet="haar", level=2, noise="coloured", threshold_multiplier=0.5
        ),
    )
    bands = fitted.layout.band[fitted.retained]
    for band in np.unique(bands):
        assert np.unique(fitted.noise_std[bands == band]).size == 1


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"threshold_rule": "bayesian"}, "coloured"),
        ({"wavelet": "nope"}, "wavelet"),
        ({"level": 0}, "level"),
        ({"threshold_multiplier": -1.0}, "threshold_multiplier"),
        ({"grid": (2, 2, 2, 2)}, "grid"),
        ({"min_noise": 0.0}, "min_noise"),
        ({"thresholding": "medium"}, "thresholding"),
    ],
)
def test_invalid_options_are_refused(options: dict[str, object], message: str) -> None:
    base: dict[str, object] = {"grid": (8, 8), "wavelet": "haar", "level": 1}
    with pytest.raises(ValueError, match=message):
        WaveletOptions(**{**base, **options})  # type: ignore[arg-type]


def test_fit_validates_the_field() -> None:
    options = WaveletOptions(grid=(8, 8), wavelet="haar", level=1)
    with pytest.raises(ValueError, match="one value per active cell"):
        fit_field(np.zeros(63), options)
    with pytest.raises(ValueError, match="mask must have shape"):
        fit_field(np.zeros(64), options, active=np.ones(10, dtype=bool))
    with pytest.raises(ValueError, match="finite"):
        fit_field(np.full(64, np.nan), options)
    with pytest.raises(ValueError, match="no wavelet coefficient"):
        fit_field(np.zeros(64), options)


# ---------------------------------------------------------------------- the transform

GRID = (16, 8)
N_CELLS = 16 * 8


def _observations() -> ObservationSet:
    """A scalar, a field at t1, a scalar, a field at t2 (keys interleaved as a CSV + fields)."""
    rng = np.random.default_rng(8)
    keys, values = [DataKey("t0", "q")], [1.0]
    for time in ("t1", "t2"):
        keys += [DataKey(time, "avo", i) for i in range(N_CELLS)]
        values += list(
            _smooth(GRID, seed=int(time[1])).ravel() + 0.02 * rng.standard_normal(N_CELLS)
        )
        keys.append(DataKey(time, "q"))
        values.append(2.0)
    return ObservationSet(keys, values, np.full(len(keys), 0.5))


def _transform(**options: object) -> WaveletCompression:
    return data_transforms.create(
        "wavelet", datatype="avo", grid=GRID, wavelet="haar", level=2, **options
    )  # type: ignore[return-value]


def test_transform_compresses_each_field_in_place() -> None:
    observations = _observations()
    transform = _transform()
    out = transform.fit(observations)

    t1, t2 = transform.fields["t1"], transform.fields["t2"]
    assert out.n_data == 3 + t1.n_retained + t2.n_retained
    assert out.keys[0] == DataKey("t0", "q")
    assert out.keys[1] == DataKey("t1", "wavelet(avo)", 0)
    assert out.keys[1 + t1.n_retained] == DataKey("t1", "q")
    np.testing.assert_array_equal(out.values[1 : 1 + t1.n_retained], t1.observed)
    np.testing.assert_array_equal(out.variances[1 : 1 + t1.n_retained], t1.noise_std**2)
    assert out.variances[0] == 0.5

    predicted = np.tile(observations.values[:, None], (1, 4)) + np.arange(4)
    compressed = transform.apply(predicted)
    assert compressed.shape == (out.n_data, 4)
    np.testing.assert_array_equal(compressed[0], predicted[0])
    np.testing.assert_array_equal(compressed[-1], predicted[-1])
    np.testing.assert_allclose(
        compressed[1 : 1 + t1.n_retained], t1.compress(predicted[1 : 1 + N_CELLS])
    )


def test_transform_times_select_fields() -> None:
    transform = _transform(times=["t2"])
    out = transform.fit(_observations())
    assert set(transform.fields) == {"t2"}
    assert out.keys[1] == DataKey("t1", "avo", 0)
    with pytest.raises(ValueError, match="no field avo observed at t9"):
        _transform(times=["t9"]).fit(_observations())


def test_transform_errors() -> None:
    with pytest.raises(ValueError, match="no field observations"):
        data_transforms.create("wavelet", datatype="xyz", grid=GRID, wavelet="haar", level=1).fit(
            _observations()
        )
    shuffled = ObservationSet(
        [DataKey("t", "avo", 1), DataKey("t", "avo", 0)], [1.0, 2.0], [1.0, 1.0]
    )
    with pytest.raises(ValueError, match=r"0\.\.n-1"):
        _transform().fit(shuffled)
    transform = _transform()
    with pytest.raises(RuntimeError, match="fit"):
        transform.apply(np.zeros((3, 2)))
    transform.fit(_observations())
    with pytest.raises(ValueError, match="predicted must be"):
        transform.apply(np.zeros((3, 2)))


def test_a_field_and_a_scalar_cannot_share_a_time_and_type() -> None:
    with pytest.raises(ValueError, match="both as a scalar and as a field"):
        ObservationSet([DataKey("t", "a"), DataKey("t", "a", 0)], [1.0, 2.0], [1.0, 1.0])
    assert str(DataKey("t", "a", 3)) == "a@t[3]"


def test_exact_zeros_are_not_retained_and_noise_matches_the_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """D6: with a zero threshold PET counts zero coefficients in the noise, not the data."""
    grid = (16, 16)
    mask = np.zeros(grid, dtype=bool)
    mask[:, :8] = True  # the inactive half is constant, so its haar details are exactly 0
    data = np.random.default_rng(9).standard_normal(int(mask.sum()))
    options = WaveletOptions(
        grid=grid, wavelet="haar", level=1, noise="coloured", threshold_multiplier=0
    )
    fitted = fit_field(data, options, active=mask, inactive_value=0.0)

    coefficients = fitted.coefficients(data[:, None])[:, 0]
    assert np.count_nonzero(coefficients == 0) > 0
    np.testing.assert_array_equal(fitted.retained, np.flatnonzero(coefficients))
    assert fitted.noise_std.shape == fitted.observed.shape == fitted.thresholds.shape
    assert "min_noise" not in caplog.text  # every band still has active cells


def test_a_noise_estimate_at_the_floor_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    grid = (16, 16)
    i, j = np.meshgrid(np.arange(16.0), np.arange(16.0), indexing="ij")
    field = (2 * i + j).ravel()  # linear: haar details of noise-free data are ~constant
    options = WaveletOptions(grid=grid, wavelet="haar", level=1, noise="coloured")
    fitted = fit_field(field, options)
    assert fitted.noise_std.min() == options.min_noise
    assert "hit min_noise" in caplog.text


def test_white_noise_with_a_non_orthogonal_wavelet_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    field = _smooth((16, 16)).ravel() + 0.05 * np.random.default_rng(10).standard_normal(256)
    fit_field(field, WaveletOptions(grid=(16, 16), wavelet="bior2.2", level=1))
    assert "not orthogonal" in caplog.text
    caplog.clear()
    fit_field(field, WaveletOptions(grid=(16, 16), wavelet="bior2.2", level=1, noise="coloured"))
    assert "not orthogonal" not in caplog.text


def test_fields_default_to_eclipse_cell_order() -> None:
    """Flat fields are in F order (i fastest), as elsewhere in APEX, unless told otherwise."""
    assert WaveletOptions(grid=(4, 3), wavelet="haar", level=1).order == "F"
    assert WaveletCompression("avo", (4, 3), "haar", 1).options.order == "F"
