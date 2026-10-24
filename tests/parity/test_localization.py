"""Localization methods against PET-fork 1348861 (see docs/migration/pet_parity.md)."""

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    GIES,
    AdaptiveLocalization,
    DataErrorCovariance,
    DistanceLocalization,
    PriorAdaptiveLocalization,
    RLMMACUpdate,
)

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

# Measured: <= 7e-15 relative to the largest step entry; rounding only.
STEP_RTOL = 1e-12


class _InjectedPermutation:
    """Generator stand-in returning PET's shuffle (a derangement, which APEX keeps as is)."""

    def __init__(self, permutation: NDArray[np.generic]) -> None:
        self._permutation = np.asarray(permutation)

    def permutation(self, n: int) -> NDArray[np.generic]:
        assert n == self._permutation.size
        return self._permutation.copy()


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
@pytest.mark.parametrize(
    ("case", "localization", "taper"),
    [
        ("gies_localized_adaptive_hard", AdaptiveLocalization, "hard"),
        ("gies_localized_adaptive_soft", AdaptiveLocalization, "soft"),
        ("gies_localized_adaptive_sigmoid", AdaptiveLocalization, "sigmoid"),
        ("gies_localized_prior_soft", PriorAdaptiveLocalization, "soft"),
    ],
)
@pytest.mark.parametrize("scale_data_anomalies", [True, False])
def test_adaptive_localization_matches_pet_fork(
    pet_reference: ReferenceLoader,
    case: str,
    localization: type[AdaptiveLocalization] | type[PriorAdaptiveLocalization],
    taper: str,
    scale_data_anomalies: bool,
) -> None:
    """With the same shuffle, APEX = sqrt(N-1) x PET-fork at the same lambda.

    Localized RLM-MAC uses gamma = lambda * mean(s^2) in PET-fork too, so only the
    1/sqrt(N-1) data-anomaly scaling differs; correlations, hence tapers, are scale-free.
    With ``scale_data_anomalies = False`` (PET-fork's unscaled dY) the steps are equal.
    """
    ref = pet_reference(case)
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    rng = _InjectedPermutation(ref["in_permutation"])
    local = localization([state.shape[0]], float(_floats(ref, "in_threshold_scale")), taper)  # type: ignore[arg-type]
    method = GIES(
        RLMMACUpdate(energy=1.0, localization=local, scale_data_anomalies=scale_data_anomalies)
    )

    method.prepare(state, predicted, [], rng)  # type: ignore[arg-type]
    step = (
        method.update(
            state,
            predicted,
            _floats(ref, "out_perturbed_observations"),
            error,
            damping=float(_floats(ref, "in_damping")),
            reference_prediction=_floats(ref, "in_reference_prediction"),
            rng=rng,  # type: ignore[arg-type]
        )
        - state
    )

    expected = _floats(ref, "out_posterior") - state
    if scale_data_anomalies:
        expected *= math.sqrt(state.shape[-1] - 1)
    if isinstance(local, AdaptiveLocalization):
        # Hard tapers are bit-identical; continuous ones differ by rounding (<= 1e-14).
        np.testing.assert_allclose(local.last_taper, _floats(ref, "out_taper"), rtol=0, atol=1e-12)
    if localization is PriorAdaptiveLocalization:
        # PET-fork's ini_only step is divided by the prior std (a bug; APEX rescales).
        expected *= _floats(ref, "in_state_std")[:, None]
        assert isinstance(local, PriorAdaptiveLocalization)
        # Identical up to rounding: bit-identical on the development machine, but BLAS
        # kernels differ between CPUs (CI runners show 1.8e-15), so compare to 1e-12.
        assert local.taper_matrix is not None
        np.testing.assert_allclose(
            local.taper_matrix, _floats(ref, "out_taper"), rtol=0, atol=1e-12
        )
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())


@pytest.mark.pet_parity
def test_distance_tapers_match_pet_fork_masks(pet_reference: ReferenceLoader) -> None:
    """Direct evaluation equals PET's cropped-and-padded stencils wherever PET succeeds.

    PET positions are 0-based ``x y z``; APEX uses 1-based Eclipse ``(i, j, k)``.
    """
    ref = pet_reference("distance_masks")
    nx, ny, nz = (int(n) for n in ref["in_grid"])
    names = {"gc": "gaspari_cohn", "fb": "furrer_bengtsson"}
    checked = 0
    for s_index, (taper, radius, anisotropy) in enumerate(
        zip(ref["in_taper"], _floats(ref, "in_radius"), _floats(ref, "in_anisotropy"), strict=True)
    ):
        for p_index, (x, y, z) in enumerate(ref["in_positions"]):
            if ref["out_failed"][s_index, p_index]:
                continue  # PET's stencil placement crashes here; APEX has no such case
            local = DistanceLocalization(
                [nx, ny, nz],
                {"W": [[int(x) + 1, int(y) + 1, int(z) + 1]]},
                float(radius),
                [nx * ny * nz],
                names[str(taper)],  # type: ignore[arg-type]
                anisotropy,
            )
            np.testing.assert_allclose(
                local.well_taper("W", int(ref["in_n_ensemble"])),
                _floats(ref, "out_masks")[s_index, p_index],
                rtol=0,
                atol=1e-13,
            )
            checked += 1
    assert checked == 22  # PET crashes at 2 of 24 (rotated stencil at two corners)


@pytest.mark.pet_parity
@pytest.mark.parametrize(
    ("case", "taper"),
    [("esmda_localized_adaptive_soft", "soft"), ("esmda_localized_adaptive_hard", "hard")],
)
def test_localized_esmda_matches_pet_fork(
    pet_reference: ReferenceLoader, case: str, taper: str
) -> None:
    """ES-MDA's approx update with adaptive localization = PET-fork's approx_update.

    Same perturbed observations and shuffle; the step uses the inflated C_d. Both scale
    the anomalies by 1/sqrt(N-1), so the steps agree directly (no rescaling).
    """
    from apex.algorithms.data_assimilation import ESMDA, ApproxUpdate

    ref = pet_reference(case)
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    alpha = float(_floats(ref, "in_alpha"))
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    local = AdaptiveLocalization([state.shape[0]], float(_floats(ref, "in_threshold_scale")), taper)  # type: ignore[arg-type]
    # One step of weight alpha: the schedule [alpha, alpha/(alpha-1)] sums 1/alpha to 1.
    method = ESMDA([alpha, alpha / (alpha - 1)], ApproxUpdate(energy=1.0, localization=local))
    rng = _InjectedPermutation(ref["in_permutation"])

    posterior = method.update(
        0,
        state,
        predicted,
        _floats(ref, "out_perturbed_observations"),
        error,
        rng=rng,  # type: ignore[arg-type]
    )

    expected = _floats(ref, "out_posterior")
    scale = np.abs(expected - state).max()
    np.testing.assert_allclose(posterior - state, expected - state, rtol=0, atol=STEP_RTOL * scale)
    # atol: correlations of the same anomalies; measured <= 4e-15.
    assert local.last_taper is not None
    np.testing.assert_allclose(local.last_taper, _floats(ref, "out_taper"), rtol=0, atol=1e-12)
