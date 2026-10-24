"""The ensemble-transform analyses (``subspace2``, ``margis``) against PET (upstream ``e65fb10``).

The references (``tools/pet_reference/generate.py``, ``_transform_steps``) run two
iterations of PET's ``subspace2_update`` and ``margIS_update`` with ``propose_state``'s
matrix branch, the second from the committed non-identity transform, with ``lambda = 1.5``
and ``gamma = 0.4`` active; and one ``subspace2`` iteration with a correlated ``C_d``.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    DataErrorCovariance,
    MarginalISUpdate,
    TransformUpdate,
)
from apex.domain.observations import DataKey

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: Relative to the largest component: solves and summation order differ (measured ~1e-15).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


def _keys(ref: dict[str, NDArray[np.generic]]) -> list[DataKey]:
    return [DataKey(f"t{i}", str(t)) for i, t in enumerate(np.asarray(ref["in_types"]))]


class _UpperWhitening:
    """PET's dense-C_d whitening: ``U^-1`` with ``C_d = U^T U`` (the ES-MDA bug)."""

    def __init__(self, upper: NDArray[np.float64]) -> None:
        self.upper, self.n_data = upper, upper.shape[0]

    def whiten(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        solved: NDArray[np.float64] = np.linalg.solve(self.upper, values)
        return solved


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", ["subspace2_steps", "margis_steps"])
def test_two_transform_iterations_match_pet(pet_reference: ReferenceLoader, case: str) -> None:
    ref = pet_reference(case)
    prior, forward = _floats(ref, "in_prior"), _floats(ref, "in_forward")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_covariance"))
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    lam, gamma = float(_floats(ref, "in_lambda")), float(_floats(ref, "in_gamma"))
    strategy: TransformUpdate | MarginalISUpdate
    if case == "margis_steps":
        strategy = MarginalISUpdate()
        strategy.use_data_keys(_keys(ref))
    else:
        strategy = TransformUpdate()
    strategy.prepare(prior)

    state = prior
    for k in (0, 1):
        predicted = np.tanh(forward @ state)
        np.testing.assert_allclose(predicted, _floats(ref, f"out_predicted_{k}"), rtol=1e-12)
        step = strategy.step(
            state, predicted, perturbed, error, damping=lam, prior_state=prior, step_length=gamma
        )
        weights, expected_w = (
            strategy.export_state()["pending_weights"],
            _floats(ref, f"out_weights_{k}"),
        )
        np.testing.assert_allclose(
            weights, expected_w, rtol=0, atol=STEP_RTOL * np.abs(expected_w).max()
        )
        increment = _floats(ref, f"out_state_{k}") - state
        np.testing.assert_allclose(
            step, increment, rtol=0, atol=STEP_RTOL * np.abs(increment).max()
        )
        strategy.accept_trial()
        state = state + step


@pytest.mark.pet_parity
def test_the_dense_transform_step_diverges_from_pet_whitening(
    pet_reference: ReferenceLoader,
) -> None:
    """PET whitens a correlated C_d with ``U^-1``, so its step uses ``(U U^T)^-1`` for
    ``C_d^-1``. PET's output is pinned as the step with that whitening; APEX's differs."""
    ref = pet_reference("subspace2_dense")
    prior, forward = _floats(ref, "in_prior"), _floats(ref, "in_forward")
    covariance, upper = _floats(ref, "in_covariance"), _floats(ref, "out_scale_data")
    np.testing.assert_allclose(upper.T @ upper, covariance, rtol=1e-12)
    perturbed = _floats(ref, "out_perturbed_observations")
    predicted = np.tanh(forward @ prior)
    expected = _floats(ref, "out_weights_0")

    def weights(error: object) -> NDArray[np.float64]:
        strategy = TransformUpdate()
        strategy.prepare(prior)
        strategy.step(
            prior,
            predicted,
            perturbed,
            error,  # type: ignore[arg-type]
            damping=float(_floats(ref, "in_lambda")),
            prior_state=prior,
            step_length=float(_floats(ref, "in_gamma")),
        )
        return strategy.export_state()["pending_weights"]

    atol = STEP_RTOL * np.abs(expected).max()
    np.testing.assert_allclose(weights(_UpperWhitening(upper)), expected, rtol=0, atol=atol)
    apex = weights(DataErrorCovariance.dense(covariance))
    assert np.abs(apex - expected).max() > 0.01 * np.abs(expected - np.eye(len(expected))).max()
