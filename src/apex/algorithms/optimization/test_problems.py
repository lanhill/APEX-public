"""Analytic multi-objective test problems with known Pareto fronts.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/problems/multi/zdt.py (ZDT1-3), bnh.py, tnk.py, srn.py, kursawe.py, and
             pymoo/problems/many/dtlz.py (DTLZ2)
Reference  : Schaffer, J. D. (1985). Proc. 1st ICGA, 93-100 (SCH). Fonseca, C. M. & Fleming,
             P. J. (1995). Evolutionary Computation, 3, 1-16 (FON). Kursawe, F. (1991). PPSN I,
             193-197 (KUR). Zitzler, E., Deb, K. & Thiele, L. (2000). Evolutionary
             Computation, 8, 173-195 (ZDT). Deb, K., Thiele, L., Laumanns, M. & Zitzler, E.
             (2002). Proc. IEEE CEC, 825-830 (DTLZ). Binh, T. T. & Korn, U. (1997). Mendel 97,
             176-182 (BNH). Tanaka, M. et al. (1995). Proc. IEEE SMC, 1556-1561 (TNK).
             Srinivas, N. & Deb, K. (1994). Evolutionary Computation, 2, 221-248 (SRN).
Differences: columns are members (pymoo: rows); constraints follow APEX's sign (an inequality
             is met when non-negative: pymoo's ``G <= 0`` becomes ``-G >= 0``). SCH and FON,
             which pymoo 0.6.2 no longer ships, follow the papers (Deb's 2001 book's bounds).
             TNK's and KUR's fronts have no closed form: they are not here but in the tests'
             references (pymoo's stored fronts).

Each problem is an ``objectives`` plugin returning ``(n_obj, n)``; its ``bounds`` are the
problem's box (an example's configuration uses them), and those with a closed-form front give
``pareto_front(n_points)``. They cost microseconds per evaluation: for testing the
multi-objective machinery, not for reservoirs.
"""

from typing import ClassVar

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray, IntArray
from apex.algorithms.optimization.reference_directions import das_dennis, number_of_directions
from apex.algorithms.optimization.registries import objectives


class _Problem:
    """What every test problem declares."""

    objective_names: ClassVar[tuple[str, ...]] = ("f1", "f2")
    #: The variables' box ``(lower, upper)`` as the paper (and pymoo) gives it.
    bounds: tuple[float, float] = (0.0, 1.0)


@objectives.register("sch")
class Schaffer(_Problem):
    r"""SCH: ``f1 = x^2``, ``f2 = (x - 2)^2`` on ``[-1000, 1000]``; front for ``x`` in [0, 2]."""

    bounds = (-1000.0, 1000.0)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(1, n)``."""
        x = controls[0]
        return np.vstack([x * x, (x - 2.0) ** 2])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return the convex front ``(2, n_points)``."""
        x = np.linspace(0.0, 2.0, n_points)
        return np.vstack([x * x, (x - 2.0) ** 2])


@objectives.register("fon")
class Fonseca(_Problem):
    r"""FON (n = 3): ``f1,2 = 1 - exp(-sum (x_i -+ 1/sqrt(3))^2)`` on ``[-4, 4]``; non-convex."""

    bounds = (-4.0, 4.0)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(n_x, n)``."""
        shift = 1.0 / np.sqrt(controls.shape[0])
        f1 = 1.0 - np.exp(-np.sum((controls - shift) ** 2, axis=0))
        f2 = 1.0 - np.exp(-np.sum((controls + shift) ** 2, axis=0))
        return np.vstack([f1, f2])

    def pareto_front(self, n_points: int, n_x: int = 3) -> FloatArray:
        """Return the front ``(2, n_points)``: ``x_i = t``, ``t`` in ``[-1/sqrt(n), 1/sqrt(n)]``."""
        shift = 1.0 / np.sqrt(n_x)
        t = np.linspace(-shift, shift, n_points)
        X = np.repeat(t[None, :], n_x, axis=0)
        return self(X, np.zeros(n_points, dtype=np.int64))


@objectives.register("kur")
class Kursawe(_Problem):
    r"""KUR (n = 3) on ``[-5, 5]``: a disconnected front (pymoo's form)."""

    bounds = (-5.0, 5.0)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(n_x, n)``."""
        x = controls
        f1 = np.sum(-10.0 * np.exp(-0.2 * np.sqrt(x[:-1] ** 2 + x[1:] ** 2)), axis=0)
        f2 = np.sum(np.abs(x) ** 0.8 + 5.0 * np.sin(x**3), axis=0)
        return np.vstack([f1, f2])


class _Zdt(_Problem):
    """ZDT's shared ``g``: ``1 + 9 / (n - 1) sum_{i>1} x_i``; ``x`` in ``[0, 1]``."""

    @staticmethod
    def g(x: FloatArray) -> FloatArray:
        return np.asarray(1.0 + 9.0 / (x.shape[0] - 1) * np.sum(x[1:], axis=0), dtype=float)


@objectives.register("zdt1")
class Zdt1(_Zdt):
    """ZDT1: ``f2 = g (1 - sqrt(f1 / g))``; convex front ``f2 = 1 - sqrt(f1)``."""

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(n_x, n)``."""
        f1, g = controls[0], self.g(controls)
        return np.vstack([f1, g * (1.0 - np.power(f1 / g, 0.5))])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return the front ``(2, n_points)``."""
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.vstack([f1, 1.0 - np.sqrt(f1)])


@objectives.register("zdt2")
class Zdt2(_Zdt):
    """ZDT2: ``f2 = g (1 - (f1 / g)^2)``; non-convex front ``f2 = 1 - f1^2``."""

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(n_x, n)``."""
        f1, g = controls[0], self.g(controls)
        return np.vstack([f1, g * (1.0 - np.power(f1 / g, 2))])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return the front ``(2, n_points)``."""
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.vstack([f1, 1.0 - f1**2])


@objectives.register("zdt3")
class Zdt3(_Zdt):
    """ZDT3: ``f2 = g (1 - sqrt(f1/g) - f1/g sin(10 pi f1))``; five disconnected pieces."""

    #: The front's pieces in ``f1`` (pymoo's).
    REGIONS: ClassVar[tuple[tuple[float, float], ...]] = (
        (0.0, 0.0830015349), (0.182228780, 0.2577623634), (0.4093136748, 0.4538821041),
        (0.6183967944, 0.6525117038), (0.8233317983, 0.8518328654),
    )  # fmt: skip

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(n_x, n)``."""
        f1, g = controls[0], self.g(controls)
        ratio = f1 / g
        return np.vstack([f1, g * (1.0 - np.power(ratio, 0.5) - ratio * np.sin(10 * np.pi * f1))])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return the front ``(2, ~n_points)``, spread over its pieces (pymoo's).

        pymoo's piece boundaries are rounded, so a few points at their ends are dominated;
        they are dropped.
        """
        from apex.algorithms.optimization.multiobjective import non_dominated

        pieces = []
        for lo, hi in self.REGIONS:
            f1 = np.linspace(lo, hi, max(n_points // len(self.REGIONS), 2))
            pieces.append(np.vstack([f1, 1.0 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)]))
        front = np.hstack(pieces)
        return np.ascontiguousarray(front[:, non_dominated(front)])


class _Dtlz:
    """DTLZ's scalable family (pymoo's ``problems/many/dtlz.py``): ``n_obj`` objectives.

    The first ``n_obj - 1`` variables place a point on the front, the rest (``k`` of them)
    give its distance ``g`` (Deb, Thiele, Laumanns & Zitzler 2005).
    """

    bounds: tuple[float, float] = (0.0, 1.0)

    def __init__(self, n_obj: int = 3) -> None:
        if n_obj < 2:
            raise ValueError(f"n_obj must be at least 2, got {n_obj}")
        self.n_obj = n_obj
        self.objective_names: tuple[str, ...] = tuple(f"f{i + 1}" for i in range(n_obj))

    def _split(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        if controls.shape[0] < self.n_obj:
            raise ValueError(f"DTLZ with {self.n_obj} objectives needs at least {self.n_obj} "
                             f"variables, got {controls.shape[0]}")  # fmt: skip
        return controls[: self.n_obj - 1], controls[self.n_obj - 1 :]

    def _directions(self, n_points: int) -> FloatArray:
        partitions = 1
        while number_of_directions(partitions + 1, self.n_obj) <= n_points:
            partitions += 1
        return das_dennis(partitions, self.n_obj).T  # (n_obj, m)


@objectives.register("dtlz1")
class Dtlz1(_Dtlz):
    r"""DTLZ1: a linear front, the simplex ``sum f = 0.5``, with ``11^k - 1`` local fronts.

    ``g = 100 (k + sum_{M-th on} ((x_i - 0.5)^2 - cos(20 pi (x_i - 0.5))))``;
    ``f_i = 0.5 (1+g) prod_{j < M-i} x_j (1 - x_{M-i})`` (no last factor for ``i = 1``).
    pymoo's default: 7 variables for 3 objectives (``k = 5``).

    Parameters
    ----------
    n_obj
        Number of objectives, at least 2 (default 3). The controls must number at least
        ``n_obj``: the first ``n_obj - 1`` place a point on the front and the rest set its
        distance from it.
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(n_obj, n)`` for ``controls`` ``(n_x, n)``, ``n_x >= n_obj``."""
        X, XM = self._split(controls)
        k = XM.shape[0]
        g = 100.0 * (k + np.sum((XM - 0.5) ** 2 - np.cos(20.0 * np.pi * (XM - 0.5)), axis=0))
        rows = []
        for i in range(self.n_obj):
            f = 0.5 * (1.0 + g) * np.prod(X[: X.shape[0] - i], axis=0)
            if i > 0:
                f = f * (1.0 - X[X.shape[0] - i])
            rows.append(f)
        return np.vstack(rows)

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return about ``n_points`` points of the front ``(n_obj, m)``: 0.5 x Das-Dennis."""
        return 0.5 * self._directions(n_points)


@objectives.register("dtlz2")
class Dtlz2(_Dtlz):
    r"""DTLZ2: the unit sphere's positive part (``n_x`` = 12 by default for 3 objectives).

    ``g = sum_{M-th on} (x_i - 0.5)^2``; ``f_i = (1+g) prod_{j < M-i} cos(x_j pi/2)
    sin(x_{M-i} pi/2)`` (no sine for ``i = 1``). For three objectives:
    ``f1 = (1+g) cos(x1 pi/2) cos(x2 pi/2)``, ``f2 = (1+g) cos(x1 pi/2) sin(x2 pi/2)``,
    ``f3 = (1+g) sin(x1 pi/2)``.

    Parameters
    ----------
    n_obj
        Number of objectives, at least 2 (default 3). The controls must number at least
        ``n_obj``: the first ``n_obj - 1`` place a point on the front and the rest set its
        distance from it.
    """

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(n_obj, n)`` for ``controls`` ``(n_x, n)``, ``n_x >= n_obj``."""
        X, XM = self._split(controls)
        g = np.sum((XM - 0.5) ** 2, axis=0)
        rows = []
        for i in range(self.n_obj):
            f = (1.0 + g) * np.prod(np.cos(X[: X.shape[0] - i] * np.pi / 2.0), axis=0)
            if i > 0:
                f = f * np.sin(X[X.shape[0] - i] * np.pi / 2.0)
            rows.append(f)
        return np.vstack(rows)

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return about ``n_points`` points of the front ``(n_obj, m)``.

        Three objectives: a grid of the angles (as the tests' pymoo reference); otherwise
        Das-Dennis directions projected on the sphere (pymoo's ``generic_sphere``).
        """
        if self.n_obj == 3:
            side = max(int(np.sqrt(n_points)), 2)
            grid_a, grid_b = np.meshgrid(
                np.linspace(0, np.pi / 2, side), np.linspace(0, np.pi / 2, side)
            )
            a, b = grid_a.ravel(), grid_b.ravel()
            return np.vstack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])
        directions = self._directions(n_points)
        return directions / np.linalg.norm(directions, axis=0, keepdims=True)


@objectives.register("bnh")
class BinhKorn(_Problem):
    r"""BNH: ``f1 = 4 x1^2 + 4 x2^2``, ``f2 = (x1-5)^2 + (x2-5)^2`` with two constraints.

    ``x1`` in [0, 5], ``x2`` in [0, 3]; constraints ``(x1-5)^2 + x2^2 <= 25`` and
    ``(x1-8)^2 + (x2+3)^2 >= 7.7`` (pymoo's scaling).
    """

    #: Per-variable bounds (the box is not square).
    LOWER: ClassVar[tuple[float, float]] = (0.0, 0.0)
    UPPER: ClassVar[tuple[float, float]] = (5.0, 3.0)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(2, n)``."""
        x1, x2 = controls[0], controls[1]
        return np.vstack([4 * x1**2 + 4 * x2**2, (x1 - 5) ** 2 + (x2 - 5) ** 2])

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return no equalities and the two inequalities ``(2, n)``, met when >= 0."""
        x1, x2 = controls[0], controls[1]
        g1 = (1 / 25) * ((x1 - 5) ** 2 + x2**2 - 25)
        g2 = -1 / 7.7 * ((x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7)
        return np.zeros((0, controls.shape[1])), np.vstack([-g1, -g2])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return pymoo's front ``(2, n_points)``: ``x2 = x1`` up to 3, then ``x2 = 3``."""
        x1 = np.linspace(0.0, 5.0, n_points)
        x2 = np.minimum(x1, 3.0)
        return self(np.vstack([x1, x2]), np.zeros(n_points, dtype=np.int64))


@objectives.register("tnk")
class Tanaka(_Problem):
    r"""TNK: ``f = (x1, x2)`` on ``[0, pi]^2`` with two constraints; a disconnected front.

    ``x1^2 + x2^2 - 1 - 0.1 cos(16 atan(x1 / x2)) >= 0`` and
    ``(x1 - 0.5)^2 + (x2 - 0.5)^2 <= 0.5``. ``x2`` starts at 1e-30 (pymoo's), so that the
    arctangent is defined.
    """

    LOWER: ClassVar[tuple[float, float]] = (0.0, 1e-30)
    UPPER: ClassVar[tuple[float, float]] = (np.pi, np.pi)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(2, n)``."""
        return np.vstack([controls[0], controls[1]])

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return no equalities and the two inequalities ``(2, n)``, met when >= 0."""
        x1, x2 = controls[0], controls[1]
        g1 = -(x1**2 + x2**2 - 1.0 - 0.1 * np.cos(16.0 * np.arctan(x1 / x2)))
        g2 = 2 * ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) - 1
        return np.zeros((0, controls.shape[1])), np.vstack([-g1, -g2])


@objectives.register("srn")
class Srinivas(_Problem):
    r"""SRN: two quadratic objectives on ``[-20, 20]^2`` with two constraints.

    ``f1 = 2 + (x1-2)^2 + (x2-1)^2``, ``f2 = 9 x1 - (x2-1)^2``; ``x1^2 + x2^2 <= 225`` and
    ``x1 - 3 x2 + 10 <= 0``. The front: ``x1 = -2.5``, ``x2`` in [2.5, 14.79].
    """

    bounds = (-20.0, 20.0)

    def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
        """Return ``(2, n)`` for ``controls`` ``(2, n)``."""
        x1, x2 = controls[0], controls[1]
        return np.vstack([2 + (x1 - 2) ** 2 + (x2 - 1) ** 2, 9 * x1 - (x2 - 1) ** 2])

    def constraints(self, controls: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return no equalities and the two inequalities ``(2, n)``, met when >= 0."""
        x1, x2 = controls[0], controls[1]
        g1 = x1**2 + x2**2 - 225
        g2 = x1 - 3 * x2 + 10
        return np.zeros((0, controls.shape[1])), np.vstack([-g1, -g2])

    def pareto_front(self, n_points: int) -> FloatArray:
        """Return pymoo's front ``(2, n_points)``."""
        X = np.vstack([np.full(n_points, -2.5), np.linspace(2.5, 14.7902, n_points)])
        return self(X, np.zeros(n_points, dtype=np.int64))
