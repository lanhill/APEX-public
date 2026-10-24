"""Non-Gaussian perturbations: Gaussian-copula marginals and score-function gradients.

Ported from PET
---------------
PET module : src/popt/ensembles/ensemble_generalized.py (GeneralizedEnsemble, BetaMC, Beta,
             Logistic, TruncGaussian, Gaussian, epsilon_trafo, var_to_concentration)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Stordal, A. S., Szklarz, S. P. & Leeuwenburgh, O. (2016). A theoretical look at
             ensemble-based optimization in reservoir management. Mathematical Geosciences,
             48, 399-417. Nelsen, R. B. (2006). An Introduction to Copulas (2nd ed.).
             Springer.
Differences: ensemble last (PET keeps the members in rows); the BetaMC concentration is the
             largest real root of a cubic from NumPy instead of sympy (equal to 1e-12); the
             gradient's baseline is ``f(x_k)`` (ensemble_gradient.py); the truncated
             Gaussian's ``theta`` (sigma) derivatives are exact, where PET's first one is the
             derivative in the mean; BetaMC's mode clip ``eps`` is relative to the width of
             the bounds, where PET's is in physical units (docs/migration/pet_parity.md).

The perturbations: correlated normals ``Z ~ N(0, R)`` ``(d, N)`` map to the controls through
each control's marginal, ``X = F^-1(Phi(Z))``. The gradient of ``E[f(X)]`` with respect to the
centre ``x`` is the score-function estimate

    g = -(1/N) sum_n (F_n - f) grad_X log p(X_n)

where ``log p`` is the copula density's log: the marginal scores ``G`` plus the copula's
``D = -rho * (H Z)``, ``H = R^-1 - I``, ``rho = p_marg(X) / phi(Z)``. The mutation gradient is the
same estimate with respect to the marginal's own parameter ``theta`` (PET's
``mutation_gradient``), which GenOpt follows to adapt the distribution.
"""

import warnings
from collections.abc import Mapping
from typing import Protocol

import numpy as np
import scipy.stats as stats
from scipy.special import polygamma

from apex.algorithms.optimization.ensemble_gradient import EnsembleEvaluator, member_realizations
from apex.algorithms.optimization.protocols import Draws, FloatArray

#: BetaMC's variance cap: a Beta variance is below 1/12; PET then uses 0.08.
BETA_VARIANCE_CAP = 0.08


class Marginal(Protocol):
    """A marginal distribution of the controls, centred on the current controls.

    Every array is ``(d, N)`` (or broadcasts to it); ``theta`` is ``(d,)`` (``(d, 2)`` for
    Beta) and ``mean`` ``(d,)``.
    """

    name: str

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        ...

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        ...

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d log p / dx``."""
        ...

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d^2 log p / dx^2`` (``(d,)`` when it does not depend on ``x``)."""
        ...


class AdaptableMarginal(Marginal, Protocol):
    """A marginal whose parameter ``theta`` GenOpt can follow."""

    def grad_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d log p / d theta``."""
        ...

    def hess_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d^2 log p / d theta^2``."""
        ...


def _col(a: FloatArray) -> FloatArray:
    # Per-control parameters (d,) as a column against the members (d, N).
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


class BetaMC:
    """Beta marginal on ``[lb, ub]`` with mode the current control and concentration ``theta``.

    ``a = 1 + c m``, ``b = 1 + c (1 - m)``, ``m`` the mode in ``[0, 1]`` clipped to
    ``[eps, 1 - eps]``.
    """

    name = "BetaMC"

    def __init__(self, lb: FloatArray, ub: FloatArray, eps: float = 0.01) -> None:
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.eps = eps

    def _mode(self, mean: FloatArray) -> FloatArray:
        return np.asarray(np.clip((mean - self.lb) / (self.ub - self.lb), self.eps, 1 - self.eps))

    def _ab(self, mean: FloatArray, theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        m = self._mode(mean)
        return _col(1 + theta * m), _col(1 + theta * (1 - m))

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        a, b = self._ab(mean, theta)
        loc, scale = _col(self.lb), _col(self.ub - self.lb)
        return np.asarray(stats.beta.pdf(x, a, b, loc=loc, scale=scale), dtype=float)

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        a, b = self._ab(mean, theta)
        loc, scale = _col(self.lb), _col(self.ub - self.lb)
        return np.asarray(stats.beta.ppf(u, a, b, loc=loc, scale=scale), dtype=float)

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``c m / (s u) - c (1 - m) / (s (1 - u))``, ``u`` the scaled ``x``."""
        scale, u, m, c = self._parts(x, theta, mean)
        return np.asarray((c * m) / (scale * u) - (c * (1 - m)) / (scale * (1 - u)))

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the second derivative of the log density in ``x``."""
        scale, u, m, c = self._parts(x, theta, mean)
        return np.asarray(-c * m / (scale**2 * u**2) - c * (1 - m) / (scale**2 * (1 - u) ** 2))

    def grad_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d log p / dc`` (digamma terms of the Beta normaliser)."""
        _, u, m, c = self._parts(x, theta, mean)
        a, b = 1 + c * m, 1 + c * (1 - m)
        return np.asarray(
            m * np.log(u)
            + (1 - m) * np.log(1 - u)
            + polygamma(0, c + 2)
            - m * polygamma(0, a)
            - (1 - m) * polygamma(0, b)
        )

    def hess_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d^2 log p / dc^2`` (``(d,)``: it does not depend on ``x``)."""
        m = self._mode(mean)
        a, b = 1 + theta * m, 1 + theta * (1 - m)
        return np.asarray(
            polygamma(1, theta + 2) - m**2 * polygamma(1, a) - (1 - m) ** 2 * polygamma(1, b)
        )

    def _parts(
        self, x: FloatArray, theta: FloatArray, mean: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        scale = _col(self.ub - self.lb)
        return scale, (x - _col(self.lb)) / scale, _col(self._mode(mean)), _col(theta)


class Beta:
    """Beta marginal on ``[0, 1]`` with parameters ``theta = (a, b)`` per control ``(d, 2)``."""

    name = "Beta"

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        return np.asarray(stats.beta(_col(theta[:, 0]), _col(theta[:, 1])).pdf(x), dtype=float)

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        return np.asarray(stats.beta(_col(theta[:, 0]), _col(theta[:, 1])).ppf(u), dtype=float)

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``(a - 1) / x - (b - 1) / (1 - x)``."""
        a, b = _col(theta[:, 0]), _col(theta[:, 1])
        return np.asarray((a - 1) / x - (b - 1) / (1 - x))

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-(a - 1) / x^2 - (b - 1) / (1 - x)^2``."""
        a, b = _col(theta[:, 0]), _col(theta[:, 1])
        return np.asarray(-(a - 1) / x**2 - (b - 1) / (1 - x) ** 2)


class Logistic:
    """Logistic marginal centred on the current control with scale ``theta``."""

    name = "Logistic"

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        return np.asarray(stats.logistic(loc=_col(mean), scale=_col(theta)).pdf(x), dtype=float)

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        return np.asarray(stats.logistic(loc=_col(mean), scale=_col(theta)).ppf(u), dtype=float)

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-tanh((x - mu) / (2 s)) / s``."""
        s = _col(theta)
        return np.asarray(-np.tanh((x - _col(mean)) / (2 * s)) / s)

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-1 / (2 s^2 cosh((x - mu) / (2 s))^2)``."""
        s = _col(theta)
        return np.asarray(-1 / (2 * s**2 * np.cosh((x - _col(mean)) / (2 * s)) ** 2))

    @staticmethod
    def var_to_scale(var: FloatArray) -> FloatArray:
        """Return the logistic scale of variance ``var``: ``sqrt(3 var) / pi``."""
        return np.asarray(np.sqrt(3 * var) / np.pi)


class TruncGaussian:
    """Gaussian marginal truncated to ``[lb, ub]``, centred on the control, std ``theta``."""

    name = "TruncGaussian"

    def __init__(self, lb: FloatArray, ub: FloatArray) -> None:
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)

    def _params(self, mean: FloatArray, theta: FloatArray) -> tuple[FloatArray, ...]:
        a, b = (self.lb - mean) / theta, (self.ub - mean) / theta
        return _col(a), _col(b), _col(mean), _col(theta)

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        a, b, loc, scale = self._params(mean, theta)
        return np.asarray(stats.truncnorm.pdf(x, a, b, loc=loc, scale=scale), dtype=float)

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        a, b, loc, scale = self._params(mean, theta)
        return np.asarray(stats.truncnorm.ppf(u, a, b, loc=loc, scale=scale), dtype=float)

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-(x - mu) / sigma^2``."""
        return np.asarray(-(x - _col(mean)) / _col(theta) ** 2)

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-1 / sigma^2`` ``(d,)``."""
        return np.asarray(-1 / np.asarray(theta, dtype=float) ** 2)

    def _standardized(
        self, theta: FloatArray, mean: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        # alpha, beta: the standardized bounds; phi(z) z vanishes at an infinite bound.
        mu, sig = np.asarray(mean, dtype=float), np.asarray(theta, dtype=float)
        alpha, beta = (self.lb - mu) / sig, (self.ub - mu) / sig
        with np.errstate(invalid="ignore"):
            pa = np.where(np.isfinite(alpha), stats.norm.pdf(alpha) * alpha, 0.0)
            pb = np.where(np.isfinite(beta), stats.norm.pdf(beta) * beta, 0.0)
        cdf_d = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
        return alpha, beta, pa, pb, np.asarray(cdf_d, dtype=float)

    def grad_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d log p / d sigma``.

        ``log p = -log sigma - (x - mu)^2 / (2 sigma^2) - log(Phi(beta) - Phi(alpha))``, so
        ``d/dsigma = -1/sigma + (x - mu)^2 / sigma^3 + (phi(beta) beta - phi(alpha) alpha) /
        (sigma (Phi(beta) - Phi(alpha)))``. PET returns the derivative in ``mu`` instead
        (docs/migration/pet_parity.md).
        """
        sig = np.asarray(theta, dtype=float)
        _, _, pa, pb, cdf_d = self._standardized(theta, mean)
        tail = (pb - pa) / (sig * cdf_d)
        return np.asarray(-1 / _col(sig) + (x - _col(mean)) ** 2 / _col(sig) ** 3 + _col(tail))

    def hess_theta_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``d^2 log p / d sigma^2`` ``(d, N)``.

        With ``N = phi(beta) beta - phi(alpha) alpha`` and ``Delta = Phi(beta) - Phi(alpha)``:
        ``1/sigma^2 - 3 (x - mu)^2 / sigma^4 + N' / (sigma Delta) - N (Delta - N) / (sigma
        Delta)^2``, ``N' = (phi(beta) beta (beta^2 - 1) - phi(alpha) alpha (alpha^2 - 1)) /
        sigma``.
        """
        sig = np.asarray(theta, dtype=float)
        alpha, beta, pa, pb, cdf_d = self._standardized(theta, mean)
        n_term = pb - pa
        with np.errstate(invalid="ignore"):
            n_prime = (
                np.where(np.isfinite(beta), pb * (beta**2 - 1), 0.0)
                - np.where(np.isfinite(alpha), pa * (alpha**2 - 1), 0.0)
            ) / sig
        tail = n_prime / (sig * cdf_d) - n_term * (cdf_d - n_term) / (sig * cdf_d) ** 2
        return np.asarray(
            1 / _col(sig) ** 2 - 3 * (x - _col(mean)) ** 2 / _col(sig) ** 4 + _col(tail)
        )


class Gaussian:
    """Gaussian marginal centred on the control with standard deviation ``theta``."""

    name = "Gaussian"

    def pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the density at ``x``."""
        return np.asarray(stats.norm(loc=_col(mean), scale=_col(theta)).pdf(x), dtype=float)

    def ppf(self, u: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return the quantile function at ``u``."""
        return np.asarray(stats.norm(loc=_col(mean), scale=_col(theta)).ppf(u), dtype=float)

    def grad_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-(x - mu) / sigma^2``."""
        return np.asarray(-(x - _col(mean)) / _col(theta) ** 2)

    def hess_log_pdf(self, x: FloatArray, theta: FloatArray, mean: FloatArray) -> FloatArray:
        """Return ``-1 / sigma^2`` ``(d,)``."""
        return np.asarray(-1 / np.asarray(theta, dtype=float) ** 2)


def beta_concentration(mode: float, var: float, lb: float = 0.0, ub: float = 1.0) -> float:
    """Return the BetaMC concentration ``c`` with variance ``var`` at ``mode`` on ``[lb, ub]``.

    With ``a = 1 + c m`` and ``b = 1 + c (1 - m)``, ``a + b = c + 2`` and the Beta variance
    ``a b / ((a + b)^2 (a + b + 1)) = v`` is the cubic
    ``v (c + 2)^2 (c + 3) - (1 + c + m (1 - m) c^2) = 0``; ``c`` is its largest real root.
    A variance of 1/12 or more (beyond any Beta) is set to 0.08 with a warning, as in PET.
    """
    m = (mode - lb) / (ub - lb)
    v = var / (ub - lb) ** 2
    if v >= 1 / 12:
        warnings.warn(
            "the largest Beta variance is 1/12; the variance is set to 0.08",
            RuntimeWarning,
            stacklevel=2,
        )
        v = BETA_VARIANCE_CAP
    cubic = np.polysub(v * np.polymul([1.0, 4.0, 4.0], [1.0, 3.0]), [m * (1 - m), 1.0, 1.0])
    roots = np.roots(cubic)
    real = roots[np.abs(roots.imag) < 1e-10].real
    if real.size == 0:
        raise ValueError(f"no real concentration for mode {mode} and variance {var}")
    return float(np.max(real))


def epsilon_trafo(
    x: FloatArray, samples: FloatArray, eps: FloatArray, lower: FloatArray, upper: FloatArray
) -> FloatArray:
    """Map unit-interval Beta samples ``(d, N)`` to a window of half-width ``eps`` around ``x``.

    The window is shifted to stay inside ``[lower, upper]``.
    """
    x, eps, lower, upper = _col(x), _col(eps), _col(lower), _col(upper)
    psi = x + (samples - 0.5) * np.minimum(2 * eps, upper - lower)
    return np.asarray(psi + np.maximum(0, lower - (x - eps)) - np.maximum(0, x + eps - upper))


#: The marginals by PET's name.
MARGINALS = ("BetaMC", "Beta", "Logistic", "TruncGaussian", "Gaussian")


class GeneralizedEnsemble:
    """GenOpt's perturbations and their score-function gradient estimates.

    Parameters
    ----------
    evaluate
        Evaluates a set of physical controls (see ``EnsembleEvaluator``).
    x0
        Initial controls ``(d,)`` (BetaMC's default concentration is computed there).
    cov
        Initial covariance ``(d, d)``: its correlation is the copula's, its variances set the
        default ``theta``.
    n_samples
        Ensemble size ``N``.
    draws
        Source of the random draws.
    lower, upper
        Bounds ``(d,)`` (needed by BetaMC, Beta's window and TruncGaussian).
    marginal
        One of :data:`MARGINALS` (PET's default ``BetaMC``).
    theta
        The marginal's parameter (default from the variances, as PET).
    n_realizations
        Model realizations of robust optimization.
    """

    def __init__(
        self,
        evaluate: EnsembleEvaluator,
        x0: FloatArray,
        cov: FloatArray,
        n_samples: int,
        draws: Draws,
        lower: FloatArray | None,
        upper: FloatArray | None,
        marginal: str = "BetaMC",
        theta: FloatArray | None = None,
        n_realizations: int = 1,
    ) -> None:
        member_realizations(n_samples, n_samples, n_realizations)
        self.evaluate = evaluate
        self.n_samples = n_samples
        self.draws = draws
        self.n_realizations = n_realizations
        d = x0.size
        self.lower = np.full(d, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.full(d, np.inf) if upper is None else np.asarray(upper, dtype=float)
        variances = np.diag(cov)
        std = np.sqrt(variances)
        self.corr = np.asarray(cov / np.outer(std, std), dtype=float)
        self.grad_scale = 1.0
        self.hess_scale = 1.0
        self.eps: FloatArray | None = None
        self.margs: Marginal
        default_theta: FloatArray
        if marginal == "Beta":
            self.margs = Beta()
            default_theta = np.array([[20.0, 20.0]] * d)
        elif marginal == "BetaMC":
            self._need_finite_bounds(marginal)
            # The mode is clipped to [eps, 1 - eps] in the unit interval. PET computes eps
            # from the physical standard deviation (0.1 sqrt(var)), which pins the mode to a
            # bound once sqrt(var) >= 5; APEX scales it by the width of the bounds.
            width = float(self.upper[0] - self.lower[0])
            self.margs = BetaMC(self.lower, self.upper, 0.1 * float(np.sqrt(variances[0])) / width)
            default_theta = np.array(
                [
                    beta_concentration(x0[i], variances[i], self.lower[i], self.upper[i])
                    for i in range(d)
                ]
            )
        elif marginal == "Logistic":
            self.margs = Logistic()
            default_theta = Logistic.var_to_scale(variances)
        elif marginal == "TruncGaussian":
            self.margs = TruncGaussian(self.lower, self.upper)
            default_theta = std.copy()
        elif marginal == "Gaussian":
            self.margs = Gaussian()
            default_theta = std.copy()
        else:
            raise ValueError(f"unknown marginal {marginal!r}; expected one of {MARGINALS}")
        self.theta = default_theta if theta is None else np.asarray(theta, dtype=float)
        if marginal == "Beta":
            self._need_finite_bounds(marginal)
            # The window half-width whose Beta(a, b) samples have the controls' variances.
            a, b = self.theta[:, 0], self.theta[:, 1]
            self.eps = np.sqrt(0.25 * variances / (a * b / ((a + b) ** 2 * (a + b + 1))))
            self.grad_scale = 1 / (2 * self.eps)  # type: ignore[assignment]
            self.hess_scale = 1 / (4 * self.eps**2)  # type: ignore[assignment]
        self.last: dict[str, FloatArray] = {}

    def _need_finite_bounds(self, marginal: str) -> None:
        if not (np.all(np.isfinite(self.lower)) and np.all(np.isfinite(self.upper))):
            raise ValueError(f"the {marginal} marginal needs finite bounds on every control")

    def sample(
        self, x: FloatArray, theta: FloatArray, corr: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Draw the members around ``x``: ``(X, Z)``, both ``(d, N)``; ``X`` is clipped."""
        z = self.draws.multivariate_normal(np.zeros(x.size), corr, self.n_samples).T
        members = self.margs.ppf(stats.norm.cdf(z), theta, x)
        return np.asarray(np.clip(members, self.lower[:, None], self.upper[:, None])), z

    def draw_and_evaluate(
        self, x: FloatArray, theta: FloatArray, corr: FloatArray
    ) -> dict[str, FloatArray]:
        """Draw an ensemble around ``x`` and evaluate it; returns ``X``, ``Z`` and ``F``."""
        members, z = self.sample(x, theta, corr)
        evaluated = members
        if isinstance(self.margs, Beta) and self.eps is not None:
            evaluated = epsilon_trafo(x, members, self.eps, self.lower, self.upper)
        values = np.asarray(
            self.evaluate(
                evaluated, member_realizations(self.n_samples, self.n_samples, self.n_realizations)
            ),
            dtype=float,
        )
        return {"X": members, "Z": z, "F": values}

    def gradient(
        self,
        x: FloatArray,
        theta: FloatArray,
        corr: FloatArray,
        ensemble: Mapping[str, FloatArray],
        f_x: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the score-function gradient ``(d,)`` and Hessian ``(d, d)`` of ``E[f]``.

        Parameters
        ----------
        x
            Centre ``(d,)``.
        theta, corr
            The marginal's parameter and the copula's correlation ``(d, d)``.
        ensemble
            ``X``, ``Z``, ``F`` from :meth:`draw_and_evaluate`.
        f_x
            The objective at the iterate per realization ``(n_realizations,)``.
        """
        members, z = ensemble["X"], ensemble["Z"]
        d, n = members.shape
        d_f = ensemble["F"] - self._baseline(f_x, n)
        h = np.linalg.inv(corr) - np.eye(d)
        g = self.margs.grad_log_pdf(members, theta, x)
        k = np.asarray(self.margs.hess_log_pdf(members, theta, x), dtype=float)
        rho = self.margs.pdf(members, theta, x) / stats.norm.pdf(z)
        # The copula's score: D = -rho (H Z); grad log p = G + D.
        dz = h @ z
        d_copula = -rho * dz
        grad_log_p = g + d_copula
        # Hessian of log p per member: diag(K + M_ii) + off-diagonal -(rho_i rho_j) H_ij.
        m_ii = (g + rho * z) * d_copula - np.diag(h)[:, None] * rho**2
        weighted = grad_log_p * d_f
        hess = weighted @ grad_log_p.T
        k_sum = np.sum(d_f) * k if k.ndim == 1 else k @ d_f
        hess += np.diag(k_sum + m_ii @ d_f)
        off = np.ones((d, d)) - np.eye(d)
        hess += -(h * off) * ((rho * d_f) @ rho.T)
        gradient = -(grad_log_p @ d_f) * self.grad_scale / n
        return np.asarray(gradient, dtype=float), np.asarray(
            hess * self.hess_scale / n, dtype=float
        )

    def mutation_gradient(
        self, x: FloatArray, theta: FloatArray, ensemble: Mapping[str, FloatArray], f_x: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return the gradient ``(d,)`` and diagonal Hessian ``(d, d)`` of ``E[f]`` in ``theta``.

        Raises
        ------
        ValueError
            If the marginal cannot be adapted (PET's Beta, Logistic and Gaussian).
        """
        margs = self.margs
        if not hasattr(margs, "grad_theta_log_pdf"):
            raise ValueError(f"the {margs.name} marginal has no theta gradient")
        adaptable: AdaptableMarginal = margs  # type: ignore[assignment]
        members = ensemble["X"]
        n = members.shape[1]
        d_f = ensemble["F"] - self._baseline(f_x, n)
        dm = np.broadcast_to(adaptable.grad_theta_log_pdf(members, theta, x), members.shape)
        hm = np.asarray(adaptable.hess_theta_log_pdf(members, theta, x), dtype=float)
        hm = np.broadcast_to(hm[:, None] if hm.ndim == 1 else hm, members.shape)
        gradient = dm @ d_f / n
        hessian = np.diag((hm + dm**2) @ d_f / n)
        return np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)

    def _baseline(self, f_x: FloatArray, n: int) -> FloatArray:
        realizations = member_realizations(n, self.n_samples, self.n_realizations)
        return np.asarray(f_x, dtype=float).reshape(-1)[realizations]
