"""Particle swarm optimization (``pso``): inertia weight, constriction defaults, bounded moves.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/algorithms/soo/nonconvex/pso.py (``pso_equation``, the initial velocity,
             ``max_velocity_rate``)
Reference  : Kennedy, J. & Eberhart, R. (1995). Particle swarm optimization. Proc. IEEE ICNN,
             1942-1948. Shi, Y. & Eberhart, R. (1998). A modified particle swarm optimizer.
             Proc. IEEE CEC, 69-73. Clerc, M. & Kennedy, J. (2002). The particle swarm:
             explosion, stability, and convergence in a multidimensional complex space. IEEE
             TEC, 6, 58-73.
Differences: particles are columns ``(n_x, P)`` (pymoo: rows); the velocity update draws
             pymoo's ``r1`` and ``r2`` in its order and shapes. pymoo's parameter adaptation
             and its perturbation of the best particle (``pertube_best``) are left out: the
             weights are fixed (``w``, ``c1``, ``c2``; Clerc and Kennedy's constriction
             equivalents by default). Bounds: pymoo redraws an out-of-bounds particle's
             velocity up to 20 times, then re-initialises it at random; here it is reflected
             (its velocity component reversed) or clipped (``bounds_handling``), so a particle
             keeps its course. ``topology = "ring"`` (each particle's two neighbours) is an
             option pymoo lacks. Personal and global bests follow the ranking (feasible
             first, or the penalized value with EPF); between two infeasible members of
             equal violation the smaller objective wins (pymoo's ``ImprovementReplacement``
             keeps the old one). The initial velocities are drawn in ``[0, v_max]``, all
             non-negative, as pymoo's (``initial_velocity = "zero"`` avoids the drift).

One generation: ``v <- w v + c1 r1 (p_best - x) + c2 r2 (s_best - x)``, limited to
``v_max`` times each control's range; ``x <- x + v`` within the bounds; evaluate; replace a
personal best that the new position beats. The population the base class ranks is the
personal bests (it never gets worse); the spread that ends the run is the positions'.
"""

from collections.abc import Mapping
from typing import ClassVar, Literal, cast

import numpy as np

from apex.algorithms.optimization.base import EpfOptions
from apex.algorithms.optimization.population import (
    ConstraintHandling,
    Initialization,
    PopulationOptimizer,
    rank_order,
)
from apex.algorithms.optimization.protocols import (
    FloatArray,
    OptimizerState,
    StateValue,
    Status,
)
from apex.algorithms.optimization.registries import optimizers

type Topology = Literal["global", "ring"]
type BoundsHandling = Literal["reflect", "clip"]

#: Clerc and Kennedy's constriction: chi = 0.7298 with phi = 4.1, i.e. c = chi * 2.05.
INERTIA = 0.7298
ACCELERATION = 1.49618


def velocity_update(
    rng: np.random.Generator,
    X: FloatArray,
    best: FloatArray,
    social: FloatArray,
    V: FloatArray,
    v_max: FloatArray,
    w: float,
    c1: float,
    c2: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the new positions and velocities ``(n_x, P)`` (pymoo's ``pso_equation``).

    Parameters
    ----------
    rng
        The generator; ``r1`` then ``r2`` are drawn ``(P, n_x)`` each, as pymoo does.
    X, best, social, V
        Positions, personal bests, the social bests and velocities ``(n_x, P)``.
    v_max
        Velocity limit per control ``(n_x,)``.
    w, c1, c2
        Inertia, cognitive and social weights.
    """
    n_x, n = X.shape
    r1 = rng.random((n, n_x)).T
    r2 = rng.random((n, n_x)).T
    Vp = w * V + c1 * r1 * (best - X) + c2 * r2 * (social - X)
    Vp = np.clip(Vp, -v_max[:, None], v_max[:, None])  # pymoo's set_to_bounds_if_outside
    return np.ascontiguousarray(X + Vp), np.ascontiguousarray(Vp)


def keep_in_bounds(
    X: FloatArray, V: FloatArray, lower: FloatArray, upper: FloatArray, how: BoundsHandling
) -> tuple[FloatArray, FloatArray]:
    """Return positions within the bounds and their velocities ``(n_x, P)``.

    ``"reflect"`` mirrors a coordinate that left at the bound it crossed and reverses that
    velocity component (clipped afterwards in case it overshoots the other bound);
    ``"clip"`` puts it on the bound and stops that component.
    """
    lo, hi = lower[:, None], upper[:, None]
    X, V = X.copy(), V.copy()
    below, above = lo > X, hi < X
    if how == "reflect":
        X = np.where(below, 2.0 * lo - X, X)
        X = np.where(above, 2.0 * hi - X, X)
        V = np.where(below | above, -V, V)
    else:
        V = np.where(below | above, 0.0, V)
    return np.ascontiguousarray(np.clip(X, lo, hi)), np.ascontiguousarray(V)


@optimizers.register("pso")
class ParticleSwarm(PopulationOptimizer):
    """Particle swarm optimization: particles drawn to their own and the swarm's best.

    Parameters
    ----------
    w, c1, c2
        Inertia, cognitive and social weights (Clerc and Kennedy's constriction equivalents).
    topology
        ``"global"`` (the swarm's best) or ``"ring"`` (each particle's and its two
        neighbours' best).
    v_max
        Velocity limit as a fraction of each control's range (pymoo's 0.2).
    initial_velocity
        ``"random"`` (uniform in ``[0, v_max]``, pymoo's) or ``"zero"``.
    bounds_handling
        ``"reflect"`` or ``"clip"`` (see :func:`keep_in_bounds`).
    pop_size
        Members of the population, ``x0`` included (at least 2, default 20); each
        generation evaluates ``pop_size`` new members on every realization.
    stall
        Generations in a row without the best member improving by more than ``ftol``
        (relative) after which the run counts as converged (default 20).
    init
        How the start population besides ``x0`` is drawn within the bounds:
        ``"uniform"`` (pymoo's random sampling) or ``"lhs"`` (Latin hypercube).
    constraint_handling
        ``"feasibility"`` (Deb's rules, as pymoo: a smaller constraint violation ranks
        first, then a smaller objective) or ``"epf"`` (the external penalty, which needs
        ``epf``). Matters only with a constrained objective.
    eq_tolerance
        Equality constraints within this absolute distance of zero count as met under the
        feasibility rules (default 1e-4).
    maxiter
        Generations at most (per outer EPF iteration with ``"epf"``; default 100).
    ftol
        Relative improvement of the best member's objective that resets the ``stall``
        count (default 1e-6).
    xtol
        The run counts as converged when the population's spread, the largest standard
        deviation of a control (in optimizer coordinates), falls below it (default 1e-8).
    transform
        Optimize in the unit cube of the bounds instead of the controls' own units
        (default off).
    epf
        External penalty options with ``constraint_handling = "epf"``: ``r`` (required),
        ``r_factor``, ``tol_factor``, ``conv_crit``, ``max_epf_iter``.
    """

    status: ClassVar[Status] = "experimental"
    label: ClassVar[str] = "PSO"

    def __init__(
        self,
        *,
        pop_size: int = 20,
        w: float = INERTIA,
        c1: float = ACCELERATION,
        c2: float = ACCELERATION,
        topology: Topology = "global",
        v_max: float = 0.2,
        initial_velocity: Literal["random", "zero"] = "random",
        bounds_handling: BoundsHandling = "reflect",
        stall: int = 20,
        init: Initialization = "uniform",
        constraint_handling: ConstraintHandling = "feasibility",
        eq_tolerance: float = 1e-4,
        maxiter: int = 100,
        ftol: float = 1e-6,
        xtol: float = 1e-8,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        super().__init__(
            pop_size=pop_size, stall=stall, init=init,
            constraint_handling=constraint_handling, eq_tolerance=eq_tolerance,
            maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf,
        )  # fmt: skip
        if topology not in ("global", "ring"):
            raise ValueError(f"topology must be 'global' or 'ring', got {topology!r}")
        if bounds_handling not in ("reflect", "clip"):
            raise ValueError(
                f"bounds_handling must be 'reflect' or 'clip', got {bounds_handling!r}"
            )
        if initial_velocity not in ("random", "zero"):
            raise ValueError(
                f"initial_velocity must be 'random' or 'zero', got {initial_velocity!r}"
            )
        if not 0.0 < v_max <= 1.0:
            raise ValueError(f"v_max must lie in (0, 1], got {v_max}")
        if w < 0 or c1 < 0 or c2 < 0:
            raise ValueError("w, c1 and c2 must be non-negative")
        self.w, self.c1, self.c2 = w, c1, c2
        self.topology: Topology = topology
        self.v_max = v_max
        self.initial_velocity = initial_velocity
        self.bounds_handling: BoundsHandling = bounds_handling
        self.positions_ = np.empty((0, 0))
        self.limit = np.empty(0)
        self.velocities = np.empty((0, 0))

    def initialize(self) -> None:
        """Start the positions at the first bests, and the velocities as pymoo draws them."""
        self.positions_ = self.X.copy()
        n_x, n = self.X.shape
        self.limit = self.v_max * (self.xu - self.xl)
        if self.initial_velocity == "random":
            self.velocities = np.ascontiguousarray(
                (self.rng.random((n, n_x)) * self.limit[None, :]).T
            )
        else:
            self.velocities = np.zeros((n_x, n))

    def social(self) -> FloatArray:
        """Return each particle's social best ``(n_x, P)``: the swarm's, or its ring's."""
        value, cv, _ = self.scores(self.F, self.c_eq, self.c_iq)
        n = value.size
        if self.topology == "global":
            best = int(rank_order(value, cv)[0])
            return np.repeat(self.X[:, best : best + 1], n, axis=1)
        out = np.empty_like(self.X)
        for i in range(n):
            ring = np.array([(i - 1) % n, i, (i + 1) % n])
            out[:, i] = self.X[:, ring[rank_order(value[ring], cv[ring])[0]]]
        return out

    def generation(self) -> None:
        """Move every particle, evaluate, and keep each personal best that is beaten."""
        self.limit = self.v_max * (self.xu - self.xl)
        X, V = velocity_update(
            self.rng, self.positions_, self.X, self.social(), self.velocities, self.limit,
            self.w, self.c1, self.c2,
        )  # fmt: skip
        X, V = keep_in_bounds(X, V, self.xl, self.xu, self.bounds_handling)
        self.positions_, self.velocities = X, V
        F, c_eq, c_iq = self.evaluate_population(X)
        new_value, new_cv, _ = self.scores(F, c_eq, c_iq)
        old_value, old_cv, _ = self.scores(self.F, self.c_eq, self.c_iq)
        better = (new_cv < old_cv) | ((new_cv == old_cv) & (new_value < old_value))
        best = self.X.copy()
        best[:, better] = X[:, better]
        Fb, eq, iq = self.F.copy(), self.c_eq.copy(), self.c_iq.copy()
        Fb[:, better] = F[:, better]
        eq[:, better] = c_eq[:, better]
        iq[:, better] = c_iq[:, better]
        self._set_population(best, Fb, eq, iq)

    def positions(self) -> FloatArray:
        """Return the particles' positions: their spread ends the run."""
        return self.positions_

    def _export_population_extra(self) -> OptimizerState:
        return {"positions": self.positions_, "velocities": self.velocities}

    def _restore_population_extra(self, state: Mapping[str, StateValue]) -> None:
        self.positions_ = np.ascontiguousarray(
            np.array(cast(FloatArray, state["positions"]), float)
        )
        self.velocities = np.ascontiguousarray(
            np.array(cast(FloatArray, state["velocities"]), float)
        )
        self.limit = self.v_max * (self.xu - self.xl)
