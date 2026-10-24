"""Line searches, the BFGS update, Newton-CG and the trust-region subproblems.

Ported from PET
---------------
PET module : src/popt/optimization_methods/subroutines/subroutines.py (line_search, zoom,
             line_search_backtracking, bfgs_update, newton_cg,
             solve_trust_region_subproblem)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Nocedal, J. & Wright, S. J. (2006). Numerical Optimization (2nd ed.),
             algorithms 3.5 and 3.6 (Wolfe line search and zoom), 6.1 (BFGS), 7.1
             (Newton-CG), 4.3 and 7.2 (trust-region subproblems). Springer.
Differences: none. The interpolation helpers are SciPy's ``_cubicmin`` and ``_quadmin``
             (BSD-3-Clause, scipy/optimize/_linesearch.py, SciPy 1.18), which PET imports
             privately; the trust-region subproblems are SciPy's own classes, as in PET.

All vectors are ``(n_x,)``; ``fun`` and ``jac`` take the controls in the optimizer's
coordinates. The line searches evaluate each step length at most once (PET caches them), and
return the objective and gradient of the last step length evaluated.
"""

from collections.abc import Callable

import numpy as np
from scipy.optimize._trustregion_exact import IterativeSubproblem
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem

from apex.algorithms.optimization.protocols import FloatArray

type ScalarFunction = Callable[[FloatArray], float]
type GradientFunction = Callable[[FloatArray], FloatArray]


def _cubicmin(
    a: float, fa: float, fpa: float, b: float, fb: float, c: float, fc: float
) -> float | None:
    # The minimiser of the cubic through (a, fa), (b, fb), (c, fc) with slope fpa at a
    # (SciPy's _cubicmin); None when it does not exist.
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            db, dc = b - a, c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
            coef_a, coef_b = d1 @ np.array([fb - fa - fpa * db, fc - fa - fpa * dc])
            coef_a /= denom
            coef_b /= denom
            radical = coef_b * coef_b - 3 * coef_a * fpa
            xmin = a + (-coef_b + np.sqrt(radical)) / (3 * coef_a)
        except ArithmeticError:
            return None
    return float(xmin) if np.isfinite(xmin) else None


def _quadmin(a: float, fa: float, fpa: float, b: float, fb: float) -> float | None:
    # The minimiser of the quadratic through (a, fa), (b, fb) with slope fpa at a
    # (SciPy's _quadmin); None when it does not exist.
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            db = b - a * 1.0
            coef_b = (fb - fa - fpa * db) / (db * db)
            xmin = a - fpa / (2.0 * coef_b)
        except ArithmeticError:
            return None
    return float(xmin) if np.isfinite(xmin) else None


class _LineFunction:
    """``phi(alpha) = f(x + alpha p)`` and ``phi'(alpha)``, each evaluated once per alpha."""

    def __init__(
        self,
        x: FloatArray,
        p: FloatArray,
        fun: ScalarFunction,
        jac: GradientFunction,
        f0: float | None,
        g0: FloatArray | None,
    ) -> None:
        self.x, self.p, self.fun, self.jac = x, p, fun, jac
        self.f0, self.g0 = f0, g0
        self._phi: dict[float, float] = {}
        self._dphi: dict[float, float] = {}
        self.last_f: float | None = None
        self.last_g: FloatArray | None = None

    def phi(self, alpha: float) -> float:
        if alpha not in self._phi:
            if alpha == 0 and self.f0 is not None:
                value = self.f0
            else:
                value = float(self.fun(self.x + alpha * self.p))
            self._phi[alpha] = value
            self.last_f = value
        return self._phi[alpha]

    def dphi(self, alpha: float) -> float:
        if alpha not in self._dphi:
            if alpha == 0 and self.g0 is not None:
                gradient = self.g0
            else:
                gradient = np.asarray(self.jac(self.x + alpha * self.p), dtype=float)
            self._dphi[alpha] = float(np.dot(gradient, self.p))
            self.last_g = gradient
        return self._dphi[alpha]


def wolfe_line_search(
    step_size: float,
    x: FloatArray,
    p: FloatArray,
    fun: ScalarFunction,
    jac: GradientFunction,
    f0: float | None = None,
    g0: FloatArray | None = None,
    *,
    amax: float = 1000.0,
    maxiter: int = 10,
    c1: float = 1e-4,
    c2: float = 0.9,
) -> tuple[float | None, float | None, FloatArray | None]:
    """Find a step length satisfying the strong Wolfe conditions (Nocedal & Wright, alg. 3.5).

    Parameters
    ----------
    step_size
        First trial step length (at most ``amax``).
    x, p
        Current point and search direction.
    fun, jac
        Objective and gradient.
    f0, g0
        Objective and gradient at ``x``, if known.
    amax
        Largest step length; the trial doubles up to it.
    maxiter
        Trial step lengths (shared with the zoom).
    c1, c2
        Sufficient-decrease and curvature constants.

    Returns
    -------
    tuple
        The step length (``None`` on failure), and the objective and gradient last evaluated.
    """
    if step_size > amax:
        raise ValueError(f"the initial step length {step_size} exceeds amax = {amax}")
    line = _LineFunction(x, p, fun, jac, f0, g0)
    phi_0, dphi_0 = line.phi(0.0), line.dphi(0.0)
    a = [0.0, step_size]
    for i in range(1, maxiter + 1):
        phi_i = line.phi(a[i])
        if phi_i > phi_0 + c1 * a[i] * dphi_0 or phi_i >= line.phi(a[i - 1]):
            alpha = _zoom(a[i - 1], a[i], line, phi_0, dphi_0, maxiter + 1 - i, c1, c2)
            return alpha, line.last_f, line.last_g
        dphi_i = line.dphi(a[i])
        if abs(dphi_i) <= -c2 * dphi_0:
            return a[i], line.last_f, line.last_g
        if dphi_i >= 0:
            alpha = _zoom(a[i], a[i - 1], line, phi_0, dphi_0, maxiter + 1 - i, c1, c2)
            return alpha, line.last_f, line.last_g
        a.append(min(2 * a[i], amax))
    return None, None, None


def _zoom(
    alo: float,
    ahi: float,
    line: _LineFunction,
    phi_0: float,
    dphi_0: float,
    maxiter: int,
    c1: float,
    c2: float,
) -> float | None:
    # Nocedal & Wright, algorithm 3.6, with SciPy's safeguarded interpolation.
    phi_lo, phi_hi, dphi_lo = line.phi(alo), line.phi(ahi), line.dphi(alo)
    aold: float | None = None
    phi_old: float | None = None
    aj: float | None = None
    for j in range(maxiter):
        tol_cubic = 0.2 * (ahi - alo)
        tol_quad = 0.1 * (ahi - alo)
        if j > 0:
            aj = _cubicmin(alo, phi_lo, dphi_lo, ahi, phi_hi, aold, phi_old)  # type: ignore[arg-type]
        if j == 0 or aj is None or aj < alo + tol_cubic or aj > ahi - tol_cubic:
            aj = _quadmin(alo, phi_lo, dphi_lo, ahi, phi_hi)
        if aj is None or aj < alo + tol_quad or aj > ahi - tol_quad:
            aj = alo + 0.5 * (ahi - alo)
        phi_j = line.phi(aj)
        if phi_j > phi_0 + c1 * aj * dphi_0 or phi_j >= phi_lo:
            aold, phi_old = ahi, phi_hi
            ahi, phi_hi = aj, phi_j
        else:
            dphi_j = line.dphi(aj)
            if abs(dphi_j) <= -c2 * dphi_0:
                return aj
            if dphi_j * (ahi - alo) >= 0:
                aold, phi_old = ahi, phi_hi
                ahi, phi_hi = alo, phi_lo
            else:
                aold, phi_old = alo, phi_lo
            alo, phi_lo, dphi_lo = aj, phi_j, dphi_j
    return None


def backtracking_line_search(
    step_size: float,
    x: FloatArray,
    p: FloatArray,
    fun: ScalarFunction,
    jac: GradientFunction,
    f0: float,
    g0: FloatArray,
    *,
    rho: float = 0.5,
    maxiter: int = 10,
    c1: float = 1e-4,
) -> tuple[float | None, float | None, FloatArray | None]:
    """Shrink the step length by ``rho`` until it gives sufficient decrease (Armijo).

    Returns the step length (``None`` on failure), its objective and the gradient there.
    """
    line = _LineFunction(x, p, fun, jac, f0, g0)
    slope = float(np.dot(g0, p))
    for _ in range(maxiter):
        phi_i = line.phi(step_size)
        if phi_i <= line.phi(0.0) + c1 * step_size * slope:
            return step_size, phi_i, np.asarray(jac(x + step_size * p), dtype=float)
        step_size *= rho
    return None, None, None


def bfgs_update(h: FloatArray, s: FloatArray, y: FloatArray) -> FloatArray:
    """Return the BFGS update of the inverse Hessian ``h`` ``(n, n)`` (Nocedal & Wright, 6.17).

    ``H+ = V H V^T + rho s s^T``, ``V = I - rho s y^T``, ``rho = 1 / (y^T s)``; skipped (``h``
    returned) when the curvature ``y^T s`` is not positive.
    """
    s, y = s.reshape(-1, 1), y.reshape(-1, 1)
    curvature = (y.T @ s).item()
    # PET skips only y^T s < 0 (and divides by zero at y^T s = 0); a vanishing curvature
    # would give a huge rho, so it is skipped too.
    if curvature <= 1e-12 * float(np.linalg.norm(y) * np.linalg.norm(s)):
        return h
    rho = 1.0 / curvature
    v = np.eye(h.shape[0]) - rho * s @ y.T
    return np.asarray(v @ h @ v.T + rho * s @ s.T)


def newton_cg(g: FloatArray, h: FloatArray, maxiter: int | None = None) -> FloatArray:
    """Return the truncated Newton-CG direction for gradient ``g`` and Hessian ``h``.

    CG on ``h z = -g`` to the relative tolerance ``min(0.5, sqrt(||g||))``; at negative
    curvature it returns the direction so far (``-g`` on the first iteration).
    """
    if maxiter is None:
        maxiter = 20 * g.size
    tol = min(0.5, np.sqrt(np.linalg.norm(g))) * np.linalg.norm(g)
    z = np.zeros_like(g, dtype=float)
    r = g
    d = -r
    for j in range(maxiter):
        hd = h @ d
        curvature = np.dot(d, hd)
        if curvature <= 0:
            return -g if j == 0 else z
        r_old = r
        a = np.dot(r, r) / curvature
        z = z + a * d
        r = r + a * hd
        if np.linalg.norm(r) < tol:
            return np.asarray(z, dtype=float)
        d = -r + np.dot(r, r) / np.dot(r_old, r_old) * d
    return z if maxiter > 0 else -g


def trust_region_step(
    x: FloatArray, f: float, g: FloatArray, h: FloatArray, radius: float, method: str = "iterative"
) -> tuple[FloatArray, bool]:
    """Solve ``min g^T p + p^T h p / 2`` over ``||p|| <= radius``.

    ``method`` is ``iterative`` (Moré-Sorensen, SciPy's ``IterativeSubproblem``) or
    ``CG-Steihaug`` (SciPy's ``CGSteihaugSubproblem``). Returns the step and whether it lies on
    the boundary.
    """

    def model(p: FloatArray) -> float:
        return float(f + np.dot(g, p) + 0.5 * np.dot(p, h @ p))

    def gradient(_: FloatArray) -> FloatArray:
        return g

    def hessian(_: FloatArray) -> FloatArray:
        return h

    if method == "iterative":
        step, hits = IterativeSubproblem(x, model, gradient, hessian).solve(radius)
    elif method == "CG-Steihaug":
        step, hits = CGSteihaugSubproblem(x, model, gradient, hessian).solve(radius)
    else:
        raise ValueError(f"unknown trust-region method {method!r}; use iterative or CG-Steihaug")
    return np.asarray(step, dtype=float), bool(hits)
