"""The PG-LSTM well residual kernel: shapes, layout, masks, bounds."""

import numpy as np
import pytest

from apex.algorithms.model_error.lstm import WellNetwork, well_residuals


def _network() -> WellNetwork:
    return WellNetwork(input_size=6, hidden_size=4, output_size=3, n_steps=5)


def test_parameter_count_follows_torchs_layout() -> None:
    network = _network()
    # weight_ih 16x6, weight_hh 16x4, two biases of 16, then per step a 3x4 FC and 3 biases.
    assert network.n_params == 16 * 6 + 16 * 4 + 2 * 16 + 5 * (3 * 4 + 3)
    arrays = network.unpack(np.arange(network.n_params, dtype=float))
    assert arrays["weight_ih"].shape == (16, 6)
    assert arrays["fc_4.bias"].shape == (3,)
    assert arrays["fc_4.bias"][-1] == network.n_params - 1  # the last parameter
    assert all(a.dtype == np.float32 for a in arrays.values())


def test_residuals_are_bounded_by_the_scale_and_zero_where_shut() -> None:
    network = _network()
    rng = np.random.default_rng(0)
    n = 4
    features = rng.normal(size=(5, 6, n))
    theta = rng.normal(0, 3.0, size=(network.n_params, n))  # saturating gates
    scale = np.array([10.0, 2.0, 0.5])
    mask = np.ones((5, 3), dtype=bool)
    mask[:2, 0] = False

    r = well_residuals(network, features, theta, scale, mask)

    assert r.shape == (5, 3, n)
    assert np.all(np.abs(r) < scale[None, :, None])
    assert np.all(r[:2, 0] == 0.0)
    assert np.all(r[2:, 0] != 0.0)


def test_zero_parameters_give_zero_residuals() -> None:
    network = _network()
    r = well_residuals(
        network,
        np.ones((5, 6, 2)),
        np.zeros((network.n_params, 2)),
        np.ones(3),
        np.ones((5, 3), dtype=bool),
    )
    # sigma(0) = 1/2, so u (2 * 1/2 - 1) = 0: a network at zero leaves the simulation as is.
    assert np.all(r == 0.0)


def test_each_member_uses_its_own_parameters_and_is_deterministic() -> None:
    network = _network()
    rng = np.random.default_rng(1)
    features = np.repeat(rng.normal(size=(5, 6, 1)), 2, axis=-1)
    theta = rng.normal(0, 0.5, size=(network.n_params, 2))
    args = (np.ones(3), np.ones((5, 3), dtype=bool))

    r = well_residuals(network, features, theta, *args)
    single = well_residuals(network, features[..., 1:], theta[:, 1:], *args)

    np.testing.assert_array_equal(r[..., 1:], single)
    assert not np.allclose(r[..., 0], r[..., 1])
    np.testing.assert_array_equal(r, well_residuals(network, features, theta, *args))


def test_shapes_are_checked() -> None:
    network = _network()
    good = (np.ones((5, 6, 2)), np.zeros((network.n_params, 2)), np.ones(3), np.ones((5, 3), bool))
    with pytest.raises(ValueError, match="features must be"):
        well_residuals(network, np.ones((4, 6, 2)), *good[1:])
    with pytest.raises(ValueError, match="theta must be"):
        well_residuals(network, good[0], np.zeros((network.n_params - 1, 2)), *good[2:])
    with pytest.raises(ValueError, match="scale must be"):
        well_residuals(network, *good[:2], np.ones(2), good[3])
    with pytest.raises(ValueError, match="open_mask must be"):
        well_residuals(network, *good[:3], np.ones((5, 2), bool))
    with pytest.raises(ValueError, match="hidden_size must be positive"):
        WellNetwork(1, 0, 1, 1)
