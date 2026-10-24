"""The approx analysis with adjoint sensitivities against PET (upstream ``e65fb10``).

The reference (``approx_adjoint``) runs ``approx_update`` with injected per-member
sensitivities ``enAdj`` ``(n_d, n_x, N)``; PET uses their member mean, as APEX does.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import ApproxUpdate, DataErrorCovariance

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: As the other steps: SVD and summation order (measured ~1e-15).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
def test_the_adjoint_step_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("approx_adjoint")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])

    step = ApproxUpdate(float(_floats(ref, "in_energy")), sensitivities=True).step(
        _floats(ref, "in_state"),
        _floats(ref, "in_predicted"),
        perturbed,
        error,
        damping=float(_floats(ref, "in_lambda")),
        sensitivity=_floats(ref, "in_adjoints").mean(axis=-1),
    )

    expected = _floats(ref, "out_step")
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())
