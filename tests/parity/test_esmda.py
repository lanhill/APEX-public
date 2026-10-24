"""ES-MDA with the approx analysis against PET e65fb10 (see docs/migration/pet_parity.md)."""

from collections.abc import Callable

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import ESMDA, ApproxUpdate, DataErrorCovariance
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

# Measured APEX-vs-PET step difference on the diagonal case: 8.5e-16 relative to the largest
# step entry. Sources are rounding only: mean subtraction vs PET's (ne, ne) projection
# matmul, and PET's state scaling, which cancels exactly in exact arithmetic.
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


def _assert_steps_close(
    step: NDArray[np.float64], expected: NDArray[np.float64], rtol: float
) -> None:
    # Compare increments, not posteriors: the posterior is dominated by the prior, so a
    # relative check on it would hide errors in the update itself.
    np.testing.assert_allclose(step, expected, rtol=0, atol=rtol * np.abs(expected).max())


def _method(ref: dict[str, NDArray[np.generic]]) -> tuple[ESMDA, float]:
    alpha = float(_floats(ref, "in_alpha"))
    # Step 0 of a uniform schedule has inflation alpha; the schedule length is irrelevant.
    return ESMDA.uniform(round(alpha), ApproxUpdate(float(_floats(ref, "in_energy")))), alpha


@pytest.mark.pet_parity
def test_esmda_approx_diagonal_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("esmda_approx_diagonal")
    method, alpha = _method(ref)
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    state = _floats(ref, "in_state")

    perturbed = _floats(ref, "in_observations")[:, None] + error.inflated(alpha).correlate(
        _floats(ref, "in_z")
    )
    posterior = method.update(
        0, state, _floats(ref, "in_predicted"), _floats(ref, "out_perturbed_observations"), error
    )

    # Same z, same factor sqrt(alpha * variances) as PET: identical arithmetic, bit-for-bit.
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    _assert_steps_close(posterior - state, _floats(ref, "out_posterior") - state, STEP_RTOL)


@pytest.mark.pet_parity
def test_esmda_approx_truncated_matches_pet(pet_reference: ReferenceLoader) -> None:
    """n_data > n_ensemble with active truncation: the rank PET kept, and the same step."""
    ref = pet_reference("esmda_approx_truncated")
    method, alpha = _method(ref)
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")

    _, kept, _ = truncated_svd(
        error.inflated(alpha).whiten(anomalies(predicted)), float(_floats(ref, "in_energy"))
    )
    posterior = method.update(
        0, state, predicted, _floats(ref, "out_perturbed_observations"), error
    )

    # The generator put the energy mid-gap between cumulative fractions, so rounding
    # cannot flip the rank; PET dropped singular values here (20 of 25 kept).
    assert kept.size == int(ref["out_rank"])
    assert kept.size < predicted.shape[-1] - 1
    _assert_steps_close(posterior - state, _floats(ref, "out_posterior") - state, STEP_RTOL)


@pytest.mark.pet_parity
def test_esmda_approx_dense_diverges_from_pet_whitening(pet_reference: ReferenceLoader) -> None:
    """Intentional divergence: PET whitens a correlated C_d with the wrong Cholesky factor.

    PET factors ``alpha C_d = U^T U`` and whitens with ``U^-1``, so its gain uses ``U U^T``
    instead of ``alpha C_d``. This test pins that diagnosis against PET's actual output and
    checks that APEX uses the correct covariance instead.
    """
    ref = pet_reference("esmda_approx_dense")
    method, alpha = _method(ref)
    covariance = _floats(ref, "in_covariance")
    error = DataErrorCovariance.dense(covariance)
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    perturbed_pet = _floats(ref, "out_perturbed_observations")
    step_pet = _floats(ref, "out_posterior") - state

    perturbed = _floats(ref, "in_observations")[:, None] + error.inflated(alpha).correlate(
        _floats(ref, "in_z")
    )
    step = method.update(0, state, predicted, perturbed_pet, error) - state

    # Sampling agrees: PET's U^T z equals APEX's L z (L = U^T); only matmul order differs.
    np.testing.assert_allclose(perturbed, perturbed_pet, rtol=0, atol=1e-13)

    c_xy = np.cov(state, predicted)[: state.shape[0], state.shape[0] :]
    c_yy = np.cov(predicted)
    upper = scipy.linalg.cholesky(alpha * covariance)  # PET's factor
    innovations = perturbed_pet - predicted
    # energy = 1 (no truncation), so both reduce to the closed-form gain up to rounding.
    _assert_steps_close(
        step_pet, c_xy @ np.linalg.solve(c_yy + upper @ upper.T, innovations), STEP_RTOL
    )
    _assert_steps_close(
        step, c_xy @ np.linalg.solve(c_yy + alpha * covariance, innovations), STEP_RTOL
    )
    # And the divergence is material on this case (measured: 17% of the largest entry).
    assert np.abs(step - step_pet).max() > 0.05 * np.abs(step_pet).max()
