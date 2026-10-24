"""Every iterative method with every update strategy.

PET ports one strategy per method in its parity tests. The other pairs are APEX
combinations: each needs an independent reference, or is refused. On the linear-Gaussian
case (30 members, 3 unknowns, 8 data, energy 1) each pair's fixed point is known:

- the approximate form (``approx``, ``rlmmac``) drops the prior term and converges to each
  member's weighted least-squares fit of its perturbed data (the case is overdetermined);
- ``full`` and ``subspace2`` keep it and converge to each member's RML solution
  ``m_j + C_p G^T (G C_p G^T + C_d)^-1 (d_j - G m_j)`` (``C_p`` the prior sample covariance);
- ``subspace`` (Evensen et al. 2019) represents ``C_d`` by the sample ``E E^T`` of the
  perturbations, projected on the range of ``S`` (``P = U U^T``), and converges to the
  weight-space ES ``W = -S^T (S S^T + (1 + lambda) P E E^T P)^+ L^-1 (G X_p - D)``,
  recomputed here. ``lambda`` is the damping at convergence: PET's step (ported exactly)
  is not the LM step of the weight-space objective, and its fixed point moves with
  ``lambda``. LM-EnRML's stays at 0.008 here; GIES's falls to ~1e-11; GN-EnRML's is 0;
- ``margis`` (Stordal et al. 2023), with one data type, converges to the RML solution with
  ``C_d / w``, ``w = (M + nu) / (chi + nu s^2)``, ``chi`` the members' mean sum of squared
  whitened residuals at the posterior (``nu = N - 1``, ``s = 1``);
- the localized strategies have no closed form; they must reach the least-squares data
  mismatch within 0.2 % (the RML solution's is 1.5 % above it).

On the tanh case every pair must lower the mean data mismatch below 40 % of the prior's
(the pairs reach 30-34 %). GN-EnRML with RLM-MAC stalled at 52-74 % before it was refused:
it runs at damping 0, where RLM-MAC's relative damping leaves no regularisation.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_directory import RunDirectory

METHODS = ("gies", "lm_enrml", "gn_enrml")
STRATEGIES = {
    "approx": '{ name = "approx", options = { energy = 1.0 } }',
    "rlmmac": '{ name = "rlmmac", options = { energy = 1.0 } }',
    "full": '{ name = "full", options = { energy = 1.0 } }',
    "subspace": '{ name = "subspace", options = { energy = 1.0 } }',
    "subspace2": '{ name = "subspace2" }',
    "margis": '{ name = "margis" }',
    "local_rlmmac": '{ name = "local_rlmmac", options = { energy = 1.0 } }',
    "local_analysis": '{ name = "local_analysis", options = { energy = 1.0 } }',
}
SCHEDULES = {
    "gies": "schedule = { data_mismatch_tolerance = 1e-12 }",
    "lm_enrml": "schedule = { initial = 1.0, data_mismatch_tolerance = 1e-12 }",
    "gn_enrml": "schedule = { data_mismatch_tolerance = 1e-12 }",
}
#: Pairs refused when the method is built, with the reason's words.
REFUSED = {
    ("gn_enrml", "rlmmac"): "leaves RLM-MAC unregularised",
    ("gn_enrml", "local_rlmmac"): "leaves RLM-MAC unregularised",
    ("gies", "local_analysis"): "use strategy local_rlmmac",
}
PAIRS = [(m, s) for m in METHODS for s in STRATEGIES]
SUPPORTED = [p for p in PAIRS if p not in REFUSED]


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray
    variances: np.ndarray

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray: ...


CaseFactory = Callable[..., Case]


def _case(linear_case: CaseFactory, method: str, strategy: str, extra: str = "") -> Case:
    localization = '\nlocalization = { name = "adaptive" }' if strategy.startswith("local") else ""
    text = (
        f'name = "{method}"\nstrategy = {STRATEGIES[strategy]}\n{SCHEDULES[method]}\n'
        f"max_iterations = 80{localization}"
    )
    return linear_case(method=text, ensemble_size=30, simulator_extra=extra)


def _evensen(case: Case, prior: np.ndarray, perturbed: np.ndarray, damping: float) -> np.ndarray:
    """Return the subspace update's fixed point on a linear model (module docstring)."""
    n = prior.shape[1]
    pi = (np.eye(n) - 1.0 / n) / np.sqrt(n - 1)
    whiten = 1.0 / np.sqrt(case.variances)[:, None]
    s_mat = whiten * (case.forward @ prior) @ pi
    e_mat = np.sqrt(1.0 + damping) * whiten * perturbed @ pi
    u, s, _ = np.linalg.svd(s_mat, full_matrices=False)
    keep = s > s[0] * 1e-10
    u, s = u[:, keep], s[keep]
    x = (u.T @ e_mat) / s[:, None]
    lam, q = np.linalg.eigh(x @ x.T)
    x2 = (u / s) @ q
    inverse = x2 @ np.diag(1.0 / (1.0 + lam)) @ x2.T
    weights = -s_mat.T @ inverse @ (whiten * (case.forward @ prior - perturbed))
    return np.asarray(prior @ (np.eye(n) + weights / np.sqrt(n - 1)))


@pytest.mark.parametrize(("method", "strategy"), SUPPORTED)
def test_each_pair_converges_to_its_fixed_point(
    linear_case: CaseFactory, method: str, strategy: str
) -> None:
    case = _case(linear_case, method, strategy)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    prior = run.load_arrays("iter_0")["state"]
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    g, c_d = case.forward, np.diag(case.variances)
    least_squares = case.least_squares(perturbed)
    c_p = np.cov(prior)
    rml = prior + c_p @ g.T @ np.linalg.inv(g @ c_p @ g.T + c_d) @ (perturbed - g @ prior)
    if strategy in ("approx", "rlmmac"):
        np.testing.assert_allclose(result.posterior, least_squares, atol=1e-5)
    elif strategy in ("full", "subspace2"):
        np.testing.assert_allclose(result.posterior, rml, atol=1e-6)
    elif strategy == "subspace":
        # The damping the converged iterate was formed with (GN-EnRML's records hold its
        # step length; its strategy always runs at 0).
        last = [r for r in result.history if r.accepted][-1]
        damping = 0.0 if method == "gn_enrml" else float(last.damping or 0.0)
        expected = _evensen(case, prior, perturbed, damping)
        np.testing.assert_allclose(result.posterior, expected, atol=1e-9)
    elif strategy == "margis":
        whitened = (g @ result.posterior - perturbed) / np.sqrt(case.variances)[:, None]
        chi = float(np.mean(np.sum(whitened**2, axis=0)))
        n_data, members = perturbed.shape
        weight = (n_data + members - 1) / (chi + members - 1)
        scaled = np.linalg.inv(g @ c_p @ g.T + c_d / weight)
        np.testing.assert_allclose(
            result.posterior, prior + c_p @ g.T @ scaled @ (perturbed - g @ prior), atol=1e-9
        )
    else:
        residual = (perturbed - g @ least_squares) / np.sqrt(case.variances)[:, None]
        minimum = float(np.mean(np.sum(residual**2, axis=0)))
        final = [r.data_mismatch_mean for r in result.history if r.accepted][-1]
        assert minimum <= final <= 1.002 * minimum


@pytest.mark.parametrize(("method", "strategy"), SUPPORTED)
def test_each_pair_lowers_the_data_mismatch_on_a_nonlinear_case(
    linear_case: CaseFactory, method: str, strategy: str
) -> None:
    case = _case(linear_case, method, strategy, 'nonlinearity = "tanh"')

    result = composition.start(case.config)

    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert np.isfinite(result.posterior).all()
    assert accepted[-1] < 0.4 * accepted[0]


@pytest.mark.parametrize("strategy", ["rlmmac", "local_rlmmac"])
def test_lm_enrml_with_rlm_mac_needs_a_damping_floor(
    linear_case: CaseFactory, strategy: str
) -> None:
    """Without a floor LM-EnRML's lambda may reach 0, where RLM-MAC is unregularised."""
    case = _case(linear_case, "lm_enrml", strategy)
    text = case.config.read_text().replace("initial = 1.0,", "initial = 1.0, minimum = 0.0,")
    case.config.write_text(text)

    with pytest.raises(ConfigError, match="needs a positive damping minimum"):
        composition.start(case.config)
    approx = case.config.read_text().replace(f'name = "{strategy}"', 'name = "approx"')
    case.config.write_text(approx.replace('\nlocalization = { name = "adaptive" }', ""))
    composition.start(case.config)  # with approx, 0 is regularised by 1 + lambda


@pytest.mark.parametrize(("method", "strategy"), list(REFUSED))
def test_pairs_without_a_valid_step_are_refused(
    linear_case: CaseFactory, method: str, strategy: str
) -> None:
    case = _case(linear_case, method, strategy)

    with pytest.raises(ConfigError, match=REFUSED[(method, strategy)]):
        composition.start(case.config)
    assert not (case.directory / "runs" / "case" / "checkpoints").exists()  # nothing ran
