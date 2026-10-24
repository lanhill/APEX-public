"""GIES/RLM-MAC against PET-fork 1348861 (see docs/migration/pet_parity.md)."""

import math
from collections.abc import Callable

import numpy as np
import pytest
import scipy.linalg
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import GIES, DataErrorCovariance, RLMMACUpdate
from apex.algorithms.data_assimilation.kernels import truncated_svd

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

# Measured: 1.1e-15 (diagonal case) and 3.1e-15 (wide case) relative to the largest step
# entry, from rounding only (mean subtraction vs PET's projection matmul, state scaling).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", ["gies_rlmmac_diagonal", "gies_rlmmac_wide"])
def test_gies_rlmmac_matches_pet_fork_up_to_the_agreed_rescaling(
    pet_reference: ReferenceLoader, case: str
) -> None:
    """APEX = sqrt(N-1) x PET-fork at lambda_apex = lambda_pet / ((N-1) mean(s^2)).

    The two agreed differences (docs/migration/pet_parity.md) are data anomalies scaled by
    1/sqrt(N-1) and gamma = lambda * mean(s^2). With no truncation they combine into this
    exact mapping, so matching it shows that nothing else differs: the mean-model centring,
    whitening, innovations and state handling.
    """
    ref = pet_reference(case)
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    reference_prediction = _floats(ref, "in_reference_prediction")
    perturbed_pet = _floats(ref, "out_perturbed_observations")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    n_ensemble = state.shape[-1]
    method = GIES(RLMMACUpdate(energy=1.0))

    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    _, s, _ = truncated_svd(
        error.whiten((predicted - reference_prediction[:, None]) / math.sqrt(n_ensemble - 1)), 1.0
    )
    damping = float(_floats(ref, "in_damping")) / ((n_ensemble - 1) * float(np.mean(s**2)))
    step = (
        method.update(
            state,
            predicted,
            perturbed_pet,
            error,
            damping=damping,
            reference_prediction=reference_prediction,
        )
        - state
    )
    expected = math.sqrt(n_ensemble - 1) * (_floats(ref, "out_posterior") - state)

    # Same z and factor sqrt(variances) as PET's gen_real: identical arithmetic.
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())
    # PET's calc_objectivefun, per member; rounding differs only in summation order.
    data_mismatch_pet = _floats(ref, "out_misfit")
    np.testing.assert_allclose(
        error.data_mismatch(perturbed_pet - predicted), data_mismatch_pet, rtol=1e-12, atol=0
    )
    assert method.mean_data_mismatch(predicted, perturbed_pet, error) == pytest.approx(
        float(np.mean(data_mismatch_pet)), rel=1e-12
    )


@pytest.mark.pet_parity
def test_gies_rlmmac_dense_diverges_from_pet_fork_whitening(
    pet_reference: ReferenceLoader,
) -> None:
    """PET-fork whitens a correlated C_d with U^-1 (C_d = U^T U), as upstream ES-MDA does.

    Its untruncated step is therefore dX dYu^T (dYu dYu^T + lambda U U^T)^-1 (D - Y), with
    unscaled dYu. APEX uses C_d itself: dX dY^T (dY dY^T + gamma C_d)^-1 (D - Y).
    """
    ref = pet_reference("gies_rlmmac_dense")
    state, predicted = _floats(ref, "in_state"), _floats(ref, "in_predicted")
    reference_prediction = _floats(ref, "in_reference_prediction")
    covariance = _floats(ref, "in_covariance")
    perturbed_pet = _floats(ref, "out_perturbed_observations")
    error = DataErrorCovariance.dense(covariance)
    n_ensemble = state.shape[-1]
    damping_pet = float(_floats(ref, "in_damping"))

    d_x = (state - state.mean(axis=-1, keepdims=True)) / math.sqrt(n_ensemble - 1)
    d_y_unscaled = predicted - reference_prediction[:, None]
    innovations = perturbed_pet - predicted
    upper = scipy.linalg.cholesky(covariance)  # PET-fork's factor
    step_pet = _floats(ref, "out_posterior") - state
    pet_formula = (
        d_x
        @ d_y_unscaled.T
        @ np.linalg.solve(
            d_y_unscaled @ d_y_unscaled.T + damping_pet * upper @ upper.T, innovations
        )
    )

    method = GIES(RLMMACUpdate(energy=1.0))
    step = (
        method.update(
            state,
            predicted,
            perturbed_pet,
            error,
            damping=damping_pet,
            reference_prediction=reference_prediction,
        )
        - state
    )
    d_y = d_y_unscaled / math.sqrt(n_ensemble - 1)
    s = np.linalg.svd(error.whiten(d_y), compute_uv=False)
    apex_formula = (
        d_x
        @ d_y.T
        @ np.linalg.solve(d_y @ d_y.T + damping_pet * np.mean(s**2) * covariance, innovations)
    )

    # Sampling agrees: PET's U^T z equals APEX's L z (L = U^T); only matmul order differs.
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_allclose(perturbed, perturbed_pet, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        step_pet, pet_formula, rtol=0, atol=STEP_RTOL * np.abs(step_pet).max()
    )
    np.testing.assert_allclose(step, apex_formula, rtol=0, atol=STEP_RTOL * np.abs(step).max())
