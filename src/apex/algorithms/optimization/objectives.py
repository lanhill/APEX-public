"""Analytic objectives: Rosenbrock, the (constrained) quadratic, Rastrigin and Ackley.

Rastrigin and Ackley are not PET's: standard multimodal tests of global search.

Ported from PET
---------------
PET module : src/popt/cost_functions/rosenbrock.py, src/popt/cost_functions/quadratic.py
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least
             value of a function. The Computer Journal, 3(3), 175-184.
Differences: the quadratic's constraints come through ``ConstrainedObjective`` (the loop adds
             the penalty); PET's quadratic reads the penalty factor from a keyword the
             optimizers never pass, so its constraints are never applied there
             (docs/migration/pet_parity.md). Both objectives have their exact gradient and
             Hessian, for line search and trust region.
"""

import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess

from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.registries import objectives


@objectives.register("rosenbrock")
class Rosenbrock:
    r"""SciPy's Rosenbrock function ``sum 100 (x_i+1 - x_i^2)^2 + (1 - x_i)^2``; minimum 0 at 1."""

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the function of each column ``(n,)`` of ``controls`` ``(n_x, n)``."""
        return np.asarray(rosen(controls), dtype=float).reshape(controls.shape[1])

    def gradient(self, x: FloatArray) -> FloatArray:
        """Return the gradient ``(n_x,)``."""
        return np.asarray(rosen_der(x), dtype=float)

    def hessian(self, x: FloatArray) -> FloatArray:
        """Return the Hessian ``(n_x, n_x)``."""
        return np.asarray(rosen_hess(x), dtype=float)


@objectives.register("quadratic")
class Quadratic:
    r"""PET's quadratic ``f(x) = (x - 1)^T A (x - 1)``, ``A = I / 2``; minimum 0 at ``x = 1``.

    Its constraints (applied with EPF): ``sum(x) = n_x + 1`` and ``x_0 <= 0``.
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``0.5 ||x - 1||^2`` of each column ``(n,)``."""
        u = controls - 1.0
        return np.asarray(0.5 * np.sum(u * u, axis=0), dtype=float)

    def gradient(self, x: FloatArray) -> FloatArray:
        """Return ``x - 1``."""
        return np.asarray(x - 1.0, dtype=float)

    def hessian(self, x: FloatArray) -> FloatArray:
        """Return the identity."""
        return np.eye(x.size)

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``c_eq = sum(x) - (n_x + 1)`` and ``c_iq = -x_0`` (each ``(1, n)``)."""
        n_x = controls.shape[0]
        c_eq = np.sum(controls, axis=0, keepdims=True) - (n_x + 1)
        return np.asarray(c_eq, dtype=float), np.asarray(-controls[:1], dtype=float)


@objectives.register("rastrigin")
class Rastrigin:
    r"""Rastrigin's function ``10 n + sum x_i^2 - 10 cos(2 pi x_i)``; minimum 0 at 0.

    Multimodal: a local minimum near every integer point. A test of global search.
    Reference: Rastrigin, L. A. (1974). Systems of extremal control. Nauka, Moscow;
    Muhlenbein, H. et al. (1991). Parallel Computing, 17, 619-632 (the n-dimensional form).
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the function of each column ``(n,)`` of ``controls`` ``(n_x, n)``."""
        n_x = controls.shape[0]
        terms = controls * controls - 10.0 * np.cos(2.0 * np.pi * controls)
        return np.asarray(10.0 * n_x + np.sum(terms, axis=0), dtype=float)


@objectives.register("ackley")
class Ackley:
    r"""Ackley's function; minimum 0 at 0, a nearly flat outer region with many local minima.

    ``-20 exp(-0.2 sqrt(mean x_i^2)) - exp(mean cos(2 pi x_i)) + 20 + e``. A test of global
    search. Reference: Ackley, D. H. (1987). A connectionist machine for genetic
    hillclimbing. Kluwer.
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return the function of each column ``(n,)`` of ``controls`` ``(n_x, n)``."""
        square = np.sqrt(np.mean(controls * controls, axis=0))
        cosine = np.mean(np.cos(2.0 * np.pi * controls), axis=0)
        value = -20.0 * np.exp(-0.2 * square) - np.exp(cosine) + 20.0 + np.e
        return np.asarray(value, dtype=float)
