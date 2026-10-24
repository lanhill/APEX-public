"""The PG-LSTM well residual model against PET-fork 1348861's torch model."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.model_error.lstm import WellNetwork, well_residuals

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

# Both compute in float32 and sum the gate products in different orders. At the prior scale
# (theta ~ N(0, 0.001^2)) they agree to ~1e-7 of each output's scale u_k. With weights of
# 0.3 on features of ~1e3, pre-activations of ~1e3 round at ~1e-4 in float32: both results
# are then within ~5e-5 u_k of a float64 evaluation (measured: torch 6e-6, APEX 4e-5).
PRIOR_ATOL = 1e-6
LARGE_ATOL = 1e-4


@pytest.mark.pet_parity
@pytest.mark.parametrize("well", ["producer", "injector"])
def test_well_residuals_match_pet_fork(pet_reference: ReferenceLoader, well: str) -> None:
    ref = pet_reference("mec_lstm_forward")
    features = np.asarray(ref[f"in_{well}_features"], dtype=np.float64)
    theta = np.asarray(ref[f"in_{well}_theta"], dtype=np.float64)
    scale = np.asarray(ref[f"in_{well}_scale"], dtype=np.float64)
    expected = np.asarray(ref[f"out_{well}_residuals"], dtype=np.float64)
    n_steps, n_out, n = expected.shape
    network = WellNetwork(features.shape[1], int(ref["in_hidden_size"]), n_out, n_steps)
    assert network.n_params == theta.shape[0]  # the same layout size as torch's

    # PET-fork never zeroes a shut well's residuals (see the reference case's docstring),
    # so its outputs are those of an all-open mask.
    residuals = well_residuals(
        network,
        np.repeat(features[:, :, None], n, axis=-1),
        theta,
        scale,
        np.ones((n_steps, n_out), dtype=bool),
    )

    # Relative to each output's own scale u_k, so that small-scale outputs are checked too.
    error = np.abs(residuals - expected) / scale[None, :, None]
    assert error[..., :3].max() <= PRIOR_ATOL  # measured 1.8e-7
    assert error[..., 3:].max() <= LARGE_ATOL  # measured 4.2e-5
    # The reference covers both regimes: tiny residuals at the prior scale, and saturated
    # gates (residuals near +-u) at the larger one.
    assert np.abs(expected[..., :3]).max() < 0.05 * scale.max()
    assert np.abs(expected[..., 3:]).max() > 0.5 * scale.min()
