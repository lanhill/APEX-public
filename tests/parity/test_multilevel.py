"""Multilevel ES-MDA's hybrid gain against PET's ``hybrid_update`` (upstream ``e65fb10``).

The reference (``hybrid_step``) runs one step with two fidelity levels (10 and 25
members, weights 0.3/0.7), per-level injected perturbations, ``alpha = 3`` and energy 0.95.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import DataErrorCovariance, HybridUpdate

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: Relative to the largest component: SVD and summation order, and PET's state scaling
#: (which cancels; APEX omits it). Measured ~1e-15.
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
def test_the_hybrid_step_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("hybrid_step")
    sizes = [int(n) for n in np.asarray(ref["in_sizes"])]
    alpha = float(_floats(ref, "in_alpha"))
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances")).inflated(alpha)
    observations = _floats(ref, "in_observations")
    states, predicted, perturbed = [], [], []
    for level in range(len(sizes)):
        d = observations[:, None] + error.correlate(_floats(ref, f"in_z_{level}"))
        np.testing.assert_array_equal(d, ref[f"out_perturbed_{level}"])
        states.append(_floats(ref, f"in_state_{level}"))
        predicted.append(_floats(ref, f"in_predicted_{level}"))
        perturbed.append(d)
    levels = np.repeat(np.arange(len(sizes)), sizes)

    step = HybridUpdate(float(_floats(ref, "in_energy"))).step(
        np.hstack(states),
        np.hstack(predicted),
        np.hstack(perturbed),
        error,
        levels=levels,
        weights=[float(w) for w in _floats(ref, "in_weights")],
    )

    expected = np.hstack([_floats(ref, f"out_step_{level}") for level in range(len(sizes))])
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())
