"""ERT's smoothers as APEX update strategies: independent references.

- each plugin's step equals ``iterative_ensemble_smoother`` called directly on the same
  arrays, with ERT's own observations, ``alpha`` and perturbations (the plumbing);
- independently of the library, each equals its formula written out in NumPy: ES-MDA's
  ``dX dY^T (dY dY^T + (N-1) C_d)^-1 (D - Y)``, the localized gain times the taper, and the
  adaptive per-row inversion over the data a row correlates with;
- ``ert_esmda`` without truncation equals APEX's ``approx`` update at energy 1;
- on a linear-Gaussian problem a 2000-member ES with ``ert_esmda`` approaches the Kalman
  posterior (the independent reference of a method with no PET equivalent).
"""

from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.data_assimilation import ApproxUpdate
from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.registries import update_strategies

ies = pytest.importorskip("iterative_ensemble_smoother")

ALPHA = 4.0


def _problem(
    n_state: int = 7, n_data: int = 5, n: int = 40, seed: int = 0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_state, n))
    forward = rng.standard_normal((n_data, n_state))
    predicted = forward @ state + 0.1 * rng.standard_normal((n_data, n))
    observations = rng.standard_normal(n_data)
    variances = rng.uniform(0.5, 2.0, n_data)
    perturbations = np.sqrt(variances)[:, None] * rng.standard_normal((n_data, n))
    return {
        "state": state,
        "predicted": predicted,
        "observations": observations,
        "variances": variances,
        "perturbations": perturbations,
        # What APEX's ES-MDA hands a strategy: D and the inflated C_d.
        "perturbed": observations[:, None] + np.sqrt(ALPHA) * perturbations,
    }


def _apex_step(name: str, p: dict[str, np.ndarray], **options: object) -> np.ndarray:
    strategy = update_strategies.create(name, **options)
    error = DataErrorCovariance.diagonal(p["variances"]).inflated(ALPHA)
    return np.asarray(p["state"] + strategy.step(p["state"], p["predicted"], p["perturbed"], error))


def _library(
    kind: str, p: dict[str, np.ndarray], truncation: float = 0.99, **assimilate: object
) -> np.ndarray:
    # ERT's own call: the real observations, its ES-MDA schedule (the library normalises it
    # to sum 1/alpha = 1, so four steps of 4; this is the first), and its perturbations.
    smoother = getattr(ies, kind)(
        covariance=p["variances"],
        observations=p["observations"],
        alpha=np.full(4, ALPHA),
        seed=1,
    )
    smoother.prepare_assimilation(
        Y=p["predicted"].copy(), truncation=truncation, observation_perturbations=p["perturbations"]
    )
    return np.asarray(smoother.assimilate_batch(X=p["state"].copy(), **assimilate))


def test_ert_esmda_is_the_librarys_esmda() -> None:
    p = _problem()
    np.testing.assert_allclose(
        _apex_step("ert_esmda", p), _library("ESMDA", p), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _apex_step("ert_esmda", p, truncation=0.8),
        _library("ESMDA", p, truncation=0.8),
        rtol=1e-12,
        atol=1e-12,
    )


def test_ert_adaptive_is_the_librarys_adaptive_esmda() -> None:
    p = _problem(n=30)
    np.testing.assert_allclose(
        _apex_step("ert_adaptive", p), _library("AdaptiveESMDA", p), rtol=1e-12, atol=1e-12
    )
    # A threshold of 0 drops nothing: ES-MDA itself.
    np.testing.assert_allclose(
        _apex_step("ert_adaptive", p, threshold=0.0), _library("ESMDA", p), rtol=1e-10, atol=1e-10
    )


def test_ert_localized_multiplies_the_gain_by_the_taper(tmp_path: Path) -> None:
    p = _problem()
    rng = np.random.default_rng(3)
    taper = rng.uniform(size=(7, 5))
    np.savez(tmp_path / "taper.npz", taper=taper)
    np.save(tmp_path / "ones.npy", np.ones((7, 5)))

    def localize(gain: np.ndarray) -> np.ndarray:
        return gain * taper

    np.testing.assert_allclose(
        _apex_step("ert_localized", p, taper_file=str(tmp_path / "taper.npz")),
        _library("LocalizedESMDA", p, localization_callback=localize),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _apex_step("ert_localized", p, taper_file=str(tmp_path / "ones.npy")),
        _library("ESMDA", p),
        rtol=1e-10,
        atol=1e-10,
    )
    np.save(tmp_path / "wrong.npy", np.ones((3, 5)))
    with pytest.raises(ValueError, match="the taper is"):
        _apex_step("ert_localized", p, taper_file=str(tmp_path / "wrong.npy"))


def test_without_truncation_ert_esmda_is_apexs_approx_update() -> None:
    p = _problem(n=12)  # fewer members than state: the ensemble subspace matters
    ours = _apex_step("ert_esmda", p, truncation=1.0)
    error = DataErrorCovariance.diagonal(p["variances"]).inflated(ALPHA)
    theirs = p["state"] + ApproxUpdate(energy=1.0).step(
        p["state"], p["predicted"], p["perturbed"], error
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-10)


def test_a_large_es_approaches_the_kalman_posterior() -> None:
    """Independent reference: the linear-Gaussian posterior mean and covariance."""
    rng = np.random.default_rng(7)
    n_state, n_data, n = 3, 4, 2000
    forward = rng.standard_normal((n_data, n_state))
    prior_cov = np.diag([1.0, 0.5, 2.0])
    variances = np.full(n_data, 0.3)
    observations = forward @ np.array([1.0, -1.0, 0.5]) + rng.normal(0, 0.3**0.5, n_data)
    state = rng.multivariate_normal(np.zeros(n_state), prior_cov, n).T
    predicted = forward @ state
    perturbed = observations[:, None] + np.sqrt(variances)[:, None] * rng.standard_normal(
        (n_data, n)
    )
    error = DataErrorCovariance.diagonal(variances)
    posterior = state + update_strategies.create("ert_esmda", truncation=1.0).step(
        state, predicted, perturbed, error
    )
    gain = (
        prior_cov @ forward.T @ np.linalg.inv(forward @ prior_cov @ forward.T + np.diag(variances))
    )
    mean = gain @ observations
    cov = prior_cov - gain @ forward @ prior_cov
    np.testing.assert_allclose(posterior.mean(axis=1), mean, atol=0.06)
    np.testing.assert_allclose(np.cov(posterior), cov, atol=0.06)


def test_iterative_use_is_refused() -> None:
    p = _problem()
    strategy = update_strategies.create("ert_esmda")
    error = DataErrorCovariance.diagonal(p["variances"])
    with pytest.raises(ValueError, match="no damping"):
        strategy.step(p["state"], p["predicted"], p["perturbed"], error, damping=1.0)
    with pytest.raises(ValueError, match="reference"):
        strategy.step(
            p["state"], p["predicted"], p["perturbed"], error,
            reference_prediction=p["predicted"].mean(axis=1),
        )  # fmt: skip
    with pytest.raises(ValueError, match="truncation"):
        update_strategies.create("ert_esmda", truncation=0.0)


@pytest.mark.parametrize("method", ["gies", "lm_enrml"])
def test_damping_methods_refuse_erts_smoothers(method: str) -> None:
    from apex.algorithms.data_assimilation.registries import methods

    with pytest.raises(ValueError, match="damps its updates"):
        methods.create(method, strategy=update_strategies.create("ert_esmda"))
    # GN-EnRML runs its analysis at damping 0, so it takes them.
    methods.create("gn_enrml", strategy=update_strategies.create("ert_esmda"))


def _anomalies(p: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = p["state"] - p["state"].mean(axis=1, keepdims=True)
    dy = p["predicted"] - p["predicted"].mean(axis=1, keepdims=True)
    return dx, dy, p["perturbed"] - p["predicted"]


def test_the_localized_step_is_the_tapered_kalman_gain(tmp_path: Path) -> None:
    """Independent reference (Emerick 2016): ((dX dY^T (dY dY^T + (N-1) C_d)^-1) o rho)(D - Y)."""
    p = _problem(n=15)
    n = p["state"].shape[1]
    taper = np.random.default_rng(5).uniform(size=(7, 5))
    np.savez(tmp_path / "taper.npz", taper=taper)
    dx, dy, innovation = _anomalies(p)
    c_d = ALPHA * np.diag(p["variances"])
    gain = dx @ dy.T @ np.linalg.inv(dy @ dy.T + (n - 1) * c_d)
    expected = p["state"] + (gain * taper) @ innovation
    ours = _apex_step("ert_localized", p, taper_file=str(tmp_path / "taper.npz"), truncation=1.0)
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-12)


def test_the_adaptive_step_inverts_over_each_rows_correlated_data() -> None:
    """Independent reference: each row's ES update over the data it correlates with only."""
    p = _problem(n_state=9, n_data=6, n=15, seed=4)
    n = p["state"].shape[1]
    threshold = 0.4
    dx, dy, innovation = _anomalies(p)
    c_d = ALPHA * np.diag(p["variances"])
    corr = np.corrcoef(p["state"], p["predicted"])[:9, 9:]
    expected = p["state"].copy()
    kept_any = 0
    for i in range(9):
        s = np.flatnonzero(np.abs(corr[i]) > threshold)
        kept_any += s.size
        if s.size:
            inverse = np.linalg.inv(dy[s] @ dy[s].T + (n - 1) * c_d[np.ix_(s, s)])
            expected[i] += dx[i] @ dy[s].T @ inverse @ innovation[s]
    assert 0 < kept_any < 9 * 6  # the threshold keeps some pairs, not all
    ours = _apex_step("ert_adaptive", p, threshold=threshold, truncation=1.0)
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-12)


def test_ert_esmda_and_approx_truncate_alike() -> None:
    p = _problem(n=12)
    error = DataErrorCovariance.diagonal(p["variances"]).inflated(ALPHA)
    for fraction in (0.9, 0.98):
        ours = _apex_step("ert_esmda", p, truncation=fraction)
        theirs = p["state"] + ApproxUpdate(energy=fraction).step(
            p["state"], p["predicted"], p["perturbed"], error
        )
        np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("name", ["ert_esmda", "ert_adaptive"])
def test_batches_do_not_change_the_step(name: str) -> None:
    p = _problem(n_state=40, n=30)
    whole = _apex_step(name, p)
    # A budget of a few kilobytes: batches of a handful of rows.
    np.testing.assert_allclose(
        _apex_step(name, p, max_batch_mb=1e-3), whole, rtol=1e-12, atol=1e-13
    )


def test_rows_without_spread_are_not_updated() -> None:
    p = _problem()
    p["state"][2] = 1.5
    step = _apex_step("ert_adaptive", p) - p["state"]
    np.testing.assert_array_equal(step[2], 0.0)


def test_the_default_threshold_needs_ten_members(caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="keeps no data"):
        _apex_step("ert_adaptive", _problem(n=9))
    with caplog.at_level("WARNING"):
        _apex_step("ert_adaptive", _problem(n=20), threshold=0.999)
    assert "kept no state-datum pairs" in caplog.text
