"""ES with the approx analysis against PET e65fb10 (EnKF.calc_analysis, which ES inherits)."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import ES, ApproxUpdate, DataErrorCovariance

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: Relative to the largest step entry: rounding only, as for ES-MDA (whose step this is).
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
def test_es_update_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("es_approx_step")
    state = _floats(ref, "in_state")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    method = ES(ApproxUpdate(float(_floats(ref, "in_energy"))))

    # PET perturbs with the un-inflated C_d: the same z and factor give the same draw.
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    posterior = method.update(0, state, _floats(ref, "in_predicted"), perturbed, error)

    expected = _floats(ref, "out_posterior") - state
    np.testing.assert_allclose(
        posterior - state, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max()
    )
    assert method.inflation == (1.0,)
