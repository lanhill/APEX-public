"""Kernels of ensemble optimization against independent references."""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import rosen_der, rosen_hess

from apex.algorithms.optimization import (
    CMA,
    Adam,
    GeneratorDraws,
    GradientDescent,
    Steihaug,
    beta_concentration,
    ensemble_gradient,
    ensemble_hessian,
    external_penalty,
    gaussian_perturbations,
    smc_weights,
)
from apex.algorithms.optimization.bounds import BoundTransform
from apex.algorithms.optimization.ensemble_gradient import member_realizations
from apex.algorithms.optimization.generalized import BetaMC, TruncGaussian
from apex.algorithms.optimization.line_search import bfgs_update, newton_cg, wolfe_line_search
from apex.algorithms.optimization.matrices import cov_to_corr, floor_eigenvalues
from apex.algorithms.optimization.objectives import Quadratic, Rosenbrock


def _quadratic_ensemble(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A quadratic f(x) = 0.5 x^T A x + b^T x, its perturbations around x and their values."""
    rng = np.random.default_rng(seed)
    a = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])
    b = np.array([1.0, -2.0, 0.5])
    x = np.array([0.3, -0.1, 0.7])
    cov = np.diag([0.04, 0.01, 0.02])
    perturbed = gaussian_perturbations(x, cov, n, GeneratorDraws(rng))
    values = 0.5 * np.einsum("in,ij,jn->n", perturbed, a, perturbed) + b @ perturbed
    return x, cov, perturbed, values


def test_ensemble_gradient_estimates_the_natural_gradient() -> None:
    """Independent reference: for a quadratic, E[dX dF] / N -> C grad f (Stordal et al. 2016)."""
    x, cov, perturbed, values = _quadratic_ensemble(200_000, 1)
    a = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])
    grad = a @ x + np.array([1.0, -2.0, 0.5])
    f_x = 0.5 * x @ a @ x + np.array([1.0, -2.0, 0.5]) @ x
    baseline = np.full(values.size, f_x)
    natural = ensemble_gradient(x, perturbed, values, baseline, cov)
    plain = ensemble_gradient(x, perturbed, values, baseline, cov, natural=False)
    # Monte Carlo error with 2e5 members is ~1% of the largest component.
    np.testing.assert_allclose(natural, cov @ grad, atol=0.01 * np.max(np.abs(cov @ grad)))
    np.testing.assert_allclose(plain, grad, atol=0.03 * np.max(np.abs(grad)))


def test_ensemble_hessian_estimates_c_hess_c() -> None:
    """Independent reference: for a quadratic the natural Hessian tends to C A C (Zhang 2023)."""
    x, cov, perturbed, values = _quadratic_ensemble(400_000, 2)
    a = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])
    f_x = 0.5 * x @ a @ x + np.array([1.0, -2.0, 0.5]) @ x
    baseline = np.full(values.size, f_x)
    natural = ensemble_hessian(x, perturbed, values, baseline, cov)
    plain = ensemble_hessian(x, perturbed, values, baseline, cov, natural=False)
    np.testing.assert_allclose(natural, cov @ a @ cov, atol=0.05 * np.max(np.abs(cov @ a @ cov)))
    np.testing.assert_allclose(plain, a, atol=0.1 * np.max(np.abs(a)))


def test_perturbations_are_centred_clipped_and_reproducible() -> None:
    x = np.array([0.0, 1.9])
    cov = 0.25 * np.eye(2)
    first = gaussian_perturbations(x, cov, 50, GeneratorDraws(np.random.default_rng(3)))
    again = gaussian_perturbations(x, cov, 50, GeneratorDraws(np.random.default_rng(3)))
    other = gaussian_perturbations(x, cov, 50, GeneratorDraws(np.random.default_rng(4)))
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert first.shape == (2, 50)
    np.testing.assert_allclose(first.mean(axis=1), x, atol=1e-12)
    clipped = gaussian_perturbations(
        x, cov, 50, GeneratorDraws(np.random.default_rng(3)), np.full(2, -2.0), np.full(2, 2.0)
    )
    assert clipped.max() <= 2.0
    with pytest.raises(ValueError, match="covariance"):
        gaussian_perturbations(x, np.eye(3), 5, GeneratorDraws(np.random.default_rng(0)))


def test_member_realizations_cycle_in_blocks() -> None:
    np.testing.assert_array_equal(member_realizations(6, 6, 3), [0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError, match="multiple"):
        member_realizations(5, 5, 2)


def test_smc_weights_are_tempered_and_floored() -> None:
    weights = smc_weights(np.array([0.0, 1.0, 100.0]), 2.0)
    expected = np.array([1.0, np.exp(-2.0), 0.0]) + 1e-6
    np.testing.assert_allclose(weights, expected / expected.sum())


def test_gradient_descent_with_momentum_and_backtracking() -> None:
    rule = GradientDescent(0.1, 0.5)
    g = np.array([1.0, -2.0])
    x1, v1 = rule.apply_update(np.zeros(2), g, iteration=1)
    np.testing.assert_allclose(v1, -0.1 * g)
    rule.restore_parameters()
    rule.apply_backtracking(0.5)
    x2, v2 = rule.apply_update(x1, g, iteration=2)
    # v = (0.5 * 0.5) v1 - (0.1 * 0.5) g
    np.testing.assert_allclose(v2, 0.25 * v1 - 0.05 * g)
    np.testing.assert_allclose(x2, x1 + v2)


def test_adam_first_step_has_unit_length_per_component() -> None:
    """Kingma & Ba (2014): the first bias-corrected step is alpha * g / (|g| + eps)."""
    rule = Adam(0.1, 0.9, 0.999)
    g = np.array([3.0, -0.5])
    _, step = rule.apply_update(np.zeros(2), g, iteration=1)
    np.testing.assert_allclose(step, 0.1 * g / (np.abs(g) + 1e-7))


def test_steihaug_stops_on_the_boundary_at_negative_curvature() -> None:
    """At negative curvature the step goes to the trust-region boundary (Nocedal & Wright 7.2).

    PET measures that step from the residual instead of the iterate and misses the boundary.
    """
    rule = Steihaug(step_size=0.5)
    g = np.array([1.0, 0.2])
    b = np.diag([-1.0, 2.0])
    _, step = rule.apply_update(np.zeros(2), g, iteration=1, hessian=b)
    assert np.linalg.norm(step) == pytest.approx(0.5)
    # A positive definite B with a small Newton step stays inside: the Newton step.
    _, inside = rule.apply_update(np.zeros(2), g, iteration=1, hessian=np.diag([10.0, 10.0]))
    np.testing.assert_allclose(inside, -g / 10.0)
    with pytest.raises(ValueError, match="Hessian"):
        rule.apply_update(np.zeros(2), g, iteration=1, hessian=None)


def test_beta_concentration_reproduces_the_variance() -> None:
    for mode, var in [(0.3, 0.01), (0.8, 0.05), (0.5, 0.001)]:
        c = beta_concentration(mode, var)
        a, b = 1 + c * mode, 1 + c * (1 - mode)
        assert stats.beta(a, b).var() == pytest.approx(var, rel=1e-10)
    # On [lb, ub] the variance scales with the width squared.
    assert beta_concentration(2.0, 0.04, 0.0, 10.0) == pytest.approx(
        beta_concentration(0.2, 0.0004)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        beta_concentration(0.5, 0.2)
    assert caught


def _finite_difference(
    f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    return np.asarray((f(theta + h) - f(theta - h)) / (2 * h))


def test_theta_derivatives_match_finite_differences() -> None:
    """Independent reference: central differences of scipy's log densities in theta."""
    lb, ub = np.array([-2.0, 0.0]), np.array([2.0, 3.0])
    mean = np.array([0.4, 1.0])
    x = np.array([[0.1, -1.0, 1.5], [0.5, 2.0, 1.2]])
    tg = TruncGaussian(lb, ub)
    sigma = np.array([0.7, 0.9])

    def log_tg(s: np.ndarray) -> np.ndarray:
        a, b = (lb - mean) / s, (ub - mean) / s
        return np.asarray(
            stats.truncnorm.logpdf(x, a[:, None], b[:, None], mean[:, None], s[:, None])
        )

    np.testing.assert_allclose(
        tg.grad_theta_log_pdf(x, sigma, mean), _finite_difference(log_tg, sigma), rtol=1e-6
    )
    np.testing.assert_allclose(
        tg.hess_theta_log_pdf(x, sigma, mean),
        _finite_difference(lambda s: tg.grad_theta_log_pdf(x, s, mean), sigma),
        rtol=1e-5,
    )
    bm = BetaMC(lb, ub, 0.01)
    c = np.array([8.0, 5.0])

    def log_bm(cc: np.ndarray) -> np.ndarray:
        return np.log(bm.pdf(x, cc, mean))

    np.testing.assert_allclose(
        bm.grad_theta_log_pdf(x, c, mean), _finite_difference(log_bm, c), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.broadcast_to(bm.hess_theta_log_pdf(x, c, mean)[:, None], x.shape),
        _finite_difference(lambda cc: bm.grad_theta_log_pdf(x, cc, mean), c),
        rtol=1e-5,
    )


def test_cma_learning_rates_follow_hansen() -> None:
    """Hansen (2006): mu_eff = 1 / sum(w^2); equal weights give mu_eff = n_mu."""
    cma = CMA(20, 5)
    assert cma.n_mu == 10
    assert cma.mu_eff == pytest.approx(10.0)
    c_cov = 1 / 10 * 2 / (5 + 2**0.5) ** 2 + (1 - 1 / 10) * min(1, 19 / (49 + 10))
    assert cma.alpha_1 == pytest.approx(c_cov / 10)
    assert cma.alpha_mu == pytest.approx(c_cov * 0.9)
    assert cma.alpha_c == pytest.approx(4 / 9)
    rng = np.random.default_rng(0)
    corr = CMA(20, 5, corr_update=True)(
        np.eye(5), rng.standard_normal(5), rng.standard_normal((5, 20)), rng.standard_normal(20)
    )
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_bounds_transform_round_trip_and_projected_gradient() -> None:
    bounds = BoundTransform(np.array([0.0, -1.0]), np.array([2.0, 1.0]), transform=True)
    x = np.array([0.5, 0.0])
    u = bounds.to_internal(x)
    np.testing.assert_allclose(u, [0.25, 0.5])
    np.testing.assert_allclose(bounds.to_physical(u), x)
    np.testing.assert_allclose(bounds.gradient_to_internal(np.array([1.0, 1.0])), [2.0, 2.0])
    at_bounds = np.array([0.0, 1.0])
    np.testing.assert_allclose(
        bounds.project_gradient(at_bounds, np.array([1.0, -1.0])), [0.0, 0.0]
    )
    np.testing.assert_allclose(
        bounds.project_gradient(at_bounds, np.array([-1.0, 1.0])), [-1.0, 1.0]
    )
    with pytest.raises(ValueError, match="finite"):
        BoundTransform(np.array([0.0]), None, transform=True)


def test_matrix_helpers() -> None:
    a = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    floored = floor_eigenvalues(a)
    assert np.linalg.eigvalsh(floored).min() == pytest.approx(1e-5)
    np.testing.assert_array_equal(floor_eigenvalues(np.eye(2)), np.eye(2))
    np.testing.assert_allclose(np.diag(cov_to_corr(np.array([[4.0, 1.0], [1.0, 9.0]]))), 1.0)


def test_penalty_counts_only_violations() -> None:
    c_eq = np.array([[0.0, 1.0]])
    c_iq = np.array([[1.0, -2.0]])
    np.testing.assert_allclose(external_penalty(2.0, c_eq, c_iq), [0.0, 1.0 + 4.0])


def test_line_search_kernels_on_rosenbrock() -> None:
    x = np.array([-1.2, 1.0])
    g = rosen_der(x)

    def f(v: np.ndarray) -> float:
        return float(Rosenbrock()(v[:, None], np.zeros(1, np.int64))[0])

    alpha, f_new, g_new = wolfe_line_search(1e-3, x, -g, f, rosen_der)
    assert alpha is not None
    assert f_new is not None
    assert f_new < f(x)
    # Strong Wolfe curvature condition at the accepted step.
    assert g_new is not None
    assert abs(np.dot(g_new, -g)) <= 0.9 * abs(np.dot(g, -g))
    direction = newton_cg(g, rosen_hess(x))
    assert np.dot(direction, g) < 0
    s, y = np.array([0.1, 0.0]), np.array([0.2, 0.05])
    h = bfgs_update(np.eye(2), s, y)
    np.testing.assert_allclose(h @ y, s)  # the secant equation
    np.testing.assert_array_equal(bfgs_update(np.eye(2), s, -y), np.eye(2))


def test_quadratic_constraints() -> None:
    controls = np.array([[1.0, -1.0], [2.0, 3.0]])
    c_eq, c_iq = Quadratic().constraints(controls)
    np.testing.assert_allclose(c_eq, [[0.0, -1.0]])
    np.testing.assert_allclose(c_iq, [[-1.0, 1.0]])


def test_bfgs_skips_a_vanishing_curvature() -> None:
    """PET divides by zero when y^T s = 0."""
    np.testing.assert_array_equal(
        bfgs_update(np.eye(2), np.zeros(2), np.array([1.0, 0.0])), np.eye(2)
    )


def test_correlation_of_a_degenerate_matrix_is_finite() -> None:
    """A zero evolution-path component (a projected step) gives NaN in PET's cov2corr."""
    corr = cov_to_corr(np.outer([1.0, 0.0, 2.0], [1.0, 0.0, 2.0]))
    np.testing.assert_allclose(corr, [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])


def test_one_controls_ensemble_gradient_is_a_vector_and_enopt_runs() -> None:
    """Regression (2026-10-03): np.squeeze made a one-control gradient 0-d, and EnOpt's
    normalisation (an inf-norm) then failed; the gradient is (n_x,) for every n_x."""
    from apex.algorithms.optimization import EnOpt, GeneratorDraws, Problem, Quadratic

    x = np.array([0.3])
    perturbed = x[:, None] + np.array([[0.1, -0.1, 0.05]])
    values = 0.5 * (perturbed[0] - 1.0) ** 2
    gradient = ensemble_gradient(x, perturbed, values, np.full(3, 0.5 * 0.49), np.eye(1))
    assert gradient.shape == (1,)
    problem = Problem(Quadratic(), x, lower=np.array([-2.0]), upper=np.array([2.0]))
    result = EnOpt(ne=3, maxiter=5).minimize(
        problem, GeneratorDraws(np.random.default_rng(1)), cov=np.eye(1) * 0.01
    )
    assert result.fun[0] < 0.5 * 0.7**2
