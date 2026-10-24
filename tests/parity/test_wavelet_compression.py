"""Wavelet sparse representation against PET's ``SparseRepresentation`` (upstream e65fb10).

Each reference compresses one observed field with thresholding (fixing the leading
indices and the noise) and then several predicted members with those indices, exactly as
PET's reader and forecast do.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_compression import FieldCompression, WaveletOptions, fit_field
from apex.algorithms.data_compression.wavelet import soft_threshold

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

CASES: dict[str, dict[str, Any]] = {
    "wavelet_white_hard_3d": {"wavelet": "db2", "order": "F", "noise": "white"},
    "wavelet_coloured_masked_2d": {"wavelet": "sym4", "order": "C", "noise": "coloured"},
    "wavelet_bayesian_keepca_3d": {
        "wavelet": "haar",
        "order": "C",
        "noise": "coloured",
        "threshold_rule": "bayesian",
        "keep_approximation": True,
    },
    "wavelet_odd_soft_coloured_3d": {
        "wavelet": "db2",
        "order": "F",
        "noise": "coloured",
        "thresholding": "soft",
        "keep_approximation": True,
    },
    "wavelet_white_soft_2d": {
        "wavelet": "db3",
        "order": "F",
        "noise": "white",
        "thresholding": "soft",
    },
}


def _f(array: NDArray[np.generic]) -> NDArray[np.float64]:
    return np.asarray(array, dtype=np.float64)


def _fit(ref: dict[str, NDArray[np.generic]], case: str) -> FieldCompression:
    options = WaveletOptions(
        grid=tuple(int(n) for n in np.asarray(ref["in_grid"])),
        level=int(np.asarray(ref["in_level"])),
        threshold_multiplier=float(np.asarray(ref["in_th_mult"])),
        min_noise=float(np.asarray(ref["in_min_noise"])),
        **CASES[case],
    )
    inactive = float(np.asarray(ref["in_inactive_value"]))
    return fit_field(
        np.asarray(ref["in_observed"], dtype=np.float64),
        options,
        active=np.asarray(ref["in_mask"]),
        inactive_value=None if np.isnan(inactive) else inactive,
    )


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", sorted(CASES))
def test_observed_compression_matches_pet(pet_reference: ReferenceLoader, case: str) -> None:
    """Leading indices, compressed observation, noise and fill value, as PET's reader."""
    ref = pet_reference(case)
    fitted = _fit(ref, case)

    assert fitted.layout.n_coefficients == int(np.asarray(ref["out_num_total_coeff"]))
    np.testing.assert_array_equal(fitted.retained, ref["out_retained"])
    # rtol: the same pywt transform and threshold arithmetic; measured bit-identical (PET's
    # env has pywt 1.8.0, APEX 1.10), the margin only absorbs a future pywt's rounding.
    np.testing.assert_allclose(fitted.observed, _f(ref["out_compressed"]), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(fitted.noise_std, _f(ref["out_est_noise"]), rtol=1e-12)
    np.testing.assert_allclose(
        np.float64(fitted.inactive_value), _f(ref["out_inactive_value"]), rtol=1e-14
    )


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", sorted(CASES))
def test_predicted_compression_matches_pet(pet_reference: ReferenceLoader, case: str) -> None:
    """Members reduced to the observed leading indices, as PET's forecast.

    With soft thresholding APEX also shrinks the predicted coefficients by the observed
    thresholds (PET does not; pet_parity.md D1), so it must equal PET's output shrunk.
    """
    ref = pet_reference(case)
    fitted = _fit(ref, case)
    members = np.asarray(ref["in_members"], dtype=np.float64)
    pet = np.asarray(ref["out_predicted"], dtype=np.float64)

    selected = fitted.coefficients(members)[fitted.retained]
    # rtol: as above, the same transform on the same inputs.
    np.testing.assert_allclose(selected, pet, rtol=1e-12, atol=1e-13)
    soft = CASES[case].get("thresholding") == "soft"
    expected = soft_threshold(pet, fitted.thresholds[:, None]) if soft else pet
    np.testing.assert_allclose(fitted.compress(members), expected, rtol=1e-12, atol=1e-13)


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", sorted(CASES))
def test_reconstruction_matches_pet(pet_reference: ReferenceLoader, case: str) -> None:
    """The observation rebuilt from its retained coefficients, dropped ones set to zero.

    PET rebuilds from the unshrunk coefficients even with soft thresholding, so the raw
    retained coefficients are used (for hard thresholding they are the observation).
    Covers the crop of odd-sized axes (``wavelet_odd_soft_coloured_3d``).
    """
    ref = pet_reference(case)
    fitted = _fit(ref, case)
    observed = _f(ref["in_observed"])[:, None]
    raw = fitted.coefficients(observed)[fitted.retained]
    if CASES[case].get("thresholding") != "soft":
        np.testing.assert_array_equal(raw[:, 0], fitted.observed)
    rebuilt = fitted.reconstruct(raw)[:, 0]
    # atol: an inverse transform of the same coefficients; rounding ~1e-15 of field ~1.
    np.testing.assert_allclose(rebuilt, _f(ref["out_reconstruction"]), rtol=1e-12, atol=1e-12)
