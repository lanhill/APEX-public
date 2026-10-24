"""CHOP's kernels against explicit formulas (Luo et al., P-AutoAdaLoc)."""

import math

import numpy as np
import pytest

from apex.algorithms.data_assimilation.chop.components import (
    ComponentContext,
    DataWeights,
    GainInflation,
    LengthScale,
    stack_sizes,
)
from apex.algorithms.data_assimilation.chop.method import Chop
from apex.algorithms.data_assimilation.chop.taper import ChopTaper, HyperTaper
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.gies import GIES
from apex.algorithms.data_assimilation.localization import correlation, gaspari_cohn

N_STATE, N_DATA, N = 7, 5, 12


def _context(blocks: tuple[int, ...] = (3, 4)) -> ComponentContext:
    return ComponentContext(
        n_state=N_STATE, n_data=N_DATA, block_sizes=blocks,
        datatypes=("a", "a", "b", "b", "b"), n_ensemble=N,
    )  # fmt: skip


def _arrays(seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((N_STATE, N))
    forward = rng.standard_normal((N_DATA, N_STATE))
    return {
        "state": state,
        "predicted": forward @ state + 0.1 * rng.standard_normal((N_DATA, N)),
        "gain_factor": rng.standard_normal((N, N_DATA)),
        "innovations": rng.standard_normal((N_DATA, N)),
        "theta": rng.uniform(0.23, 0.43, (N_DATA, N)),
    }


@pytest.mark.parametrize("max_batch_mb", [256.0, 1e-5])
def test_the_model_taper_is_each_members_gaspari_cohn_taper_on_the_shared_gain(
    max_batch_mb: float,
) -> None:
    """dm_j = (T(l_j) o K) r_j with T[k, s] = f_GC((1 - |rho|) / (factor l_j[s])) (the
    paper's Eq. eq:GC_tapering_rule_parameterized_chop); any row blocking."""
    a = _arrays()
    component = LengthScale(mode="data", correlation="current", factor=2.0)
    taper = ChopTaper([component], max_batch_mb=max_batch_mb)
    context = _context()
    taper.configure(context, stack_sizes([component], context))
    taper.theta = a["theta"]
    taper.set_current(a["state"], a["predicted"])
    anomalies = (a["state"] - a["state"].mean(axis=1, keepdims=True)) / math.sqrt(N - 1)

    step = taper.localized_step(anomalies, a["gain_factor"], a["innovations"], None)

    gain = anomalies @ a["gain_factor"]
    rho = correlation(a["state"], a["predicted"])
    expected = np.column_stack([
        (gaspari_cohn((1 - np.abs(rho)) / (2.0 * a["theta"][:, j])[None, :]) * gain)
        @ a["innovations"][:, j]
        for j in range(N)
    ])  # fmt: skip
    np.testing.assert_allclose(step, expected, rtol=1e-12, atol=1e-14)


def test_very_long_length_scales_leave_the_gain_untapered() -> None:
    a = _arrays()
    component = LengthScale(mode="scalar", correlation="current")
    taper = ChopTaper([component])
    context = _context()
    taper.configure(context, stack_sizes([component], context))
    taper.theta = np.full((1, N), 1e9)
    taper.set_current(a["state"], a["predicted"])
    anomalies = a["state"] - a["state"].mean(axis=1, keepdims=True)

    step = taper.localized_step(anomalies, a["gain_factor"], a["innovations"], None)

    np.testing.assert_allclose(step, anomalies @ a["gain_factor"] @ a["innovations"], rtol=1e-8)


def test_prior_correlations_are_the_initial_ensembles() -> None:
    a = _arrays()
    later = _arrays(seed=4)
    component = LengthScale(mode="data", correlation="prior")
    taper = ChopTaper([component])
    context = _context()
    taper.configure(context, stack_sizes([component], context))
    taper.prepare(a["state"], a["predicted"], [], np.random.default_rng(0))
    taper.set_current(later["state"], later["predicted"])  # ignored: correlation = prior
    taper.theta = a["theta"]
    anomalies = later["state"] - later["state"].mean(axis=1, keepdims=True)

    step = taper.localized_step(anomalies, a["gain_factor"], a["innovations"], None)

    rho0 = correlation(a["state"], a["predicted"])
    gain = anomalies @ a["gain_factor"]
    j = 5
    expected = (gaspari_cohn((1 - np.abs(rho0)) / (2 * a["theta"][:, j])[None, :]) * gain) @ (
        a["innovations"][:, j]
    )
    np.testing.assert_allclose(step[:, j], expected, rtol=1e-12)


def test_inflation_and_weights_are_schur_factors_of_the_gain() -> None:
    a = _arrays()
    inflation = GainInflation(mode="variable_datatype")
    weights = DataWeights(mode="datatype")
    context = _context()
    taper = ChopTaper([inflation, weights])
    slices = stack_sizes([inflation, weights], context)
    taper.configure(context, slices)
    rng = np.random.default_rng(1)
    theta = np.vstack([inflation.prior(rng, context), weights.prior(rng, context)])
    taper.theta = theta
    anomalies = a["state"] - a["state"].mean(axis=1, keepdims=True)

    step = taper.localized_step(anomalies, a["gain_factor"], a["innovations"], None)

    gain = anomalies @ a["gain_factor"]
    variables = np.repeat([0, 1], [3, 4])
    types = np.array([0, 0, 1, 1, 1])
    j = 2
    table = theta[slices[0], j].reshape(2, 2)  # variables x data types
    factor = table[variables][:, types] * theta[slices[1], j][types][None, :]
    np.testing.assert_allclose(step[:, j], (factor * gain) @ a["innovations"][:, j], rtol=1e-12)


def _chop(hyper_localization: str) -> Chop:
    return Chop(GIES(), [{"name": "length_scale", "mode": "data"}],
                hyper_localization=hyper_localization)  # fmt: skip


def _started(chop: Chop, a: dict[str, np.ndarray]) -> None:
    from types import SimpleNamespace

    keys = [SimpleNamespace(datatype="y") for _ in range(N_DATA)]
    setup = SimpleNamespace(
        prior=np.zeros((N_STATE, N)), observations=SimpleNamespace(keys=keys),
        streams={"chop": np.random.default_rng(0)},
    )  # fmt: skip
    chop._configure(setup, N)  # type: ignore[arg-type]
    chop.theta_u = a["theta"].copy()


def test_the_unlocalized_hyper_update_is_an_rlm_mac_step_on_the_hyper_parameters() -> None:
    """l_j <- l_j + S_l S_g^T (S_g S_g^T + gamma I)^-1 L^-1 (d_j - g(m_j^{i+1})), S_g about
    g(mean m^{i+1}), gamma = lambda mean(s^2) (the paper's Eq. eq:rlm_mac_chop_2, the
    code's update_hyper_no_loc); energy 1, no truncation."""
    a = _arrays()
    chop = _chop("none")
    chop.hyper_strategy.energy = 1.0
    _started(chop, a)
    variances = np.linspace(0.5, 2.0, N_DATA)
    data_error = DataErrorCovariance.diagonal(variances)
    rng = np.random.default_rng(9)
    perturbed = rng.standard_normal((N_DATA, N))
    reference = a["predicted"].mean(axis=1) + 0.05
    damping = 0.7

    chop.after_accept(a["state"], a["predicted"], reference, perturbed, data_error, damping)

    whiten = 1 / np.sqrt(variances)[:, None]
    s_theta = (a["theta"] - a["theta"].mean(axis=1, keepdims=True)) / math.sqrt(N - 1)
    s_g = whiten * (a["predicted"] - reference[:, None]) / math.sqrt(N - 1)
    singular = np.linalg.svd(s_g, compute_uv=False)
    gamma = damping * np.mean(singular[singular > 1e-12] ** 2)
    gain = s_theta @ s_g.T @ np.linalg.inv(s_g @ s_g.T + gamma * np.eye(N_DATA))
    expected = np.maximum(a["theta"] + gain @ (whiten * (perturbed - a["predicted"])), 1e-6)
    assert chop.theta_u is not None
    np.testing.assert_allclose(chop.theta_u, expected, rtol=1e-9, atol=1e-12)


def test_the_localized_hyper_update_tapers_by_each_members_length_scales() -> None:
    """The code's update_hyper: T_l[s', s] = f_GC((1 - |corr(l_s', g_s)|) / (f l_j[s]))."""
    a = _arrays()
    chop = _chop("same")
    _started(chop, a)
    hyper = chop._hyper_taper
    assert isinstance(hyper, HyperTaper)
    hyper.theta = a["theta"]
    hyper.correlation = correlation(a["theta"], a["predicted"])
    anomalies = a["theta"] - a["theta"].mean(axis=1, keepdims=True)

    step = hyper.localized_step(anomalies, a["gain_factor"], a["innovations"], None)

    gain = anomalies @ a["gain_factor"]
    rho = correlation(a["theta"], a["predicted"])
    j = 7
    taper = gaspari_cohn((1 - np.abs(rho)) / (2 * a["theta"][:, j])[None, :])
    np.testing.assert_allclose(step[:, j], (taper * gain) @ a["innovations"][:, j], rtol=1e-12)


@pytest.mark.parametrize("transform", ["none", "log", "logit"])
def test_transforms_go_there_and_back(transform: str) -> None:
    component = LengthScale(transform=transform)  # type: ignore[arg-type]
    values = np.array([[0.25, 0.3, 0.42]])

    np.testing.assert_allclose(component.from_u(component.to_u(values)), values, rtol=1e-12)


def test_the_floor_and_the_logit_bounds_hold() -> None:
    assert LengthScale().from_u(np.array([-1.0]))[0] == 1e-6
    bounded = GainInflation(transform="logit", lower=0.0, upper=2.0)
    assert 0.0 < bounded.from_u(np.array([-50.0]))[0] < bounded.from_u(np.array([50.0]))[0] <= 2.0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"mode": "rows"}, "mode"),
        ({"correlation": "later"}, "correlation"),
        ({"factor": 0.0}, "factor"),
        ({"low": 0.0}, "positive"),
        ({"transform": "logit", "lower": 0.3}, "lower < low"),
    ],
)
def test_bad_length_scale_options_are_refused(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LengthScale(**options)  # type: ignore[arg-type]


def test_component_sizes_follow_their_modes() -> None:
    context = _context()

    assert LengthScale(mode="data").size(context) == N_DATA
    assert LengthScale(mode="scalar").size(context) == 1
    assert GainInflation(mode="variable").size(context) == 2
    assert GainInflation(mode="variable_datatype").size(context) == 4
    assert DataWeights(mode="datatype").size(context) == 2
    assert DataWeights(mode="datum").size(context) == N_DATA


@pytest.mark.parametrize("transform", ["log", "logit"])
def test_a_large_step_stays_finite_in_the_bounded_transforms(transform: str) -> None:
    """Review finding: a round trip through the natural values overflowed to infinity."""
    component = GainInflation(transform=transform)  # type: ignore[arg-type]
    u = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])

    bounded = component.bounded(u)

    np.testing.assert_array_equal(bounded[1:4], u[1:4])  # moderate values unchanged
    natural = component.from_u(bounded)
    assert np.all(np.isfinite(natural))
    assert np.all(natural >= 0)


def test_a_negative_lower_bound_is_refused_and_the_default_is_not_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        DataWeights(transform="logit", lower=-0.5)
    assert GainInflation(transform="logit").lower == 0.0  # 0.5 - 1.0, floored at 0
    assert GainInflation(transform="logit").from_u(np.array([-50.0]))[0] >= 0


def test_after_accept_tapers_by_the_hyper_parameters_before_the_update() -> None:
    """hyper_localization = "same": T_l from corr(l^i, g(m^{i+1})) and member j's l^i."""
    a = _arrays()
    chop = _chop("same")
    chop.hyper_strategy.energy = 1.0
    _started(chop, a)
    data_error = DataErrorCovariance.diagonal(np.ones(N_DATA))
    rng = np.random.default_rng(9)
    perturbed = rng.standard_normal((N_DATA, N))
    reference = a["predicted"].mean(axis=1)

    chop.after_accept(a["state"], a["predicted"], reference, perturbed, data_error, 0.7)

    s_theta = (a["theta"] - a["theta"].mean(axis=1, keepdims=True)) / math.sqrt(N - 1)
    s_g = (a["predicted"] - reference[:, None]) / math.sqrt(N - 1)
    singular = np.linalg.svd(s_g, compute_uv=False)
    gamma = 0.7 * np.mean(singular[singular > 1e-12] ** 2)
    gain = s_theta @ s_g.T @ np.linalg.inv(s_g @ s_g.T + gamma * np.eye(N_DATA))
    rho = correlation(a["theta"], a["predicted"])
    expected = np.column_stack([
        a["theta"][:, j]
        + (gaspari_cohn((1 - np.abs(rho)) / (2 * a["theta"][:, j])[None, :]) * gain)
        @ (perturbed[:, j] - a["predicted"][:, j])
        for j in range(N)
    ])  # fmt: skip
    assert chop.theta_u is not None
    np.testing.assert_allclose(chop.theta_u, np.maximum(expected, 1e-6), rtol=1e-9, atol=1e-12)
