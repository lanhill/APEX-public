"""MOEA/D (``moead``): decomposition into neighbouring scalar subproblems.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Modules    : pymoo/algorithms/moo/moead.py (``ParallelMOEAD``, ``NeighborhoodSelection``,
             ``MOEAD._setup``/``_replace``, ``default_decomp``), pymoo/decomposition/
             tchebicheff.py and pbi.py, pymoo/core/decomposition.py,
             pymoo/functions/standard/decomposition.py (``calc_distance_to_weights``)
Reference  : Zhang, Q. & Li, H. (2007). MOEA/D: a multiobjective evolutionary algorithm based
             on decomposition. IEEE TEC, 11, 712-731.
Differences: - ``ParallelMOEAD``, not ``MOEAD``: one child per subproblem, evaluated as one
               batch (``MOEAD`` evaluates a child at a time: one simulation per batch).
             - Each subproblem's child is one of the two children of its own mating, drawn at
               random. pymoo's ``ParallelMOEAD`` reshapes the flattened children (all first
               children, then all second) into pairs, so subproblem ``k`` gets a child of the
               matings ``2k`` and ``2k+1`` (or of the second children): a child of other
               subproblems' parents, against its comment ("select a random offspring from
               each mating") and the method's neighbourhoods. Recorded in
               docs/migration/pymoo.md.
             - Constraints by EPF only: pymoo's MOEA/D refuses constraints (its
               constraint code is commented out as "not tested enough").
             - Defaults are smaller than the paper's: the weights by
               :mod:`.reference_directions` and ``n_neighbors = min(20, max(2, ceil(N/5)))``
               for ``N`` weights (pymoo: 20; 20 of 28 would make mating global).
             - As ``nsga2``: ``x0`` is a member, robust objectives are reduced
               over the realizations by their statistic, and the stall is the front's
               movement.

A generation: the subproblems in a random order; for each, two parents from its neighbourhood
(the ``n_neighbors`` nearest weights, itself included) with probability
``prob_neighbor_mating``, else from the whole population; SBX and polynomial mutation; one of
the two children. The batch is evaluated, the ideal point updated from it, and then, child by
child, every neighbour whose decomposed value the child improves is replaced by it. The
population stays in the weights' order (member ``k`` serves weight ``k``).
"""

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from scipy.spatial.distance import cdist

from apex.algorithms.optimization.base import EpfOptions
from apex.algorithms.optimization.ga import polynomial_mutation, sbx
from apex.algorithms.optimization.nsga2 import NSGA2
from apex.algorithms.optimization.population import Initialization
from apex.algorithms.optimization.protocols import (
    FloatArray,
    IntArray,
    OptimizerState,
    StateValue,
    Status,
)
from apex.algorithms.optimization.reference_directions import directions_for
from apex.algorithms.optimization.registries import optimizers

#: The decompositions: Tchebicheff (pymoo's default for 2 objectives) and PBI (3 or more).
type Decomposition = Literal["auto", "tchebicheff", "pbi"]

# ---------------------------------------------------------------------------- kernels


def tchebicheff(F: FloatArray, weights: FloatArray, ideal: FloatArray) -> FloatArray:
    """Return ``max_i w_i |f_i - z_i|`` for paired rows ``F``, ``weights`` ``(m, n_obj)``."""
    return np.asarray(np.max(np.abs(F - ideal) * weights, axis=1), dtype=float)


def pbi(F: FloatArray, weights: FloatArray, ideal: FloatArray, theta: float) -> FloatArray:
    """Return ``d1 + theta d2`` (penalty boundary intersection) for paired rows ``(m, n_obj)``.

    ``d1`` is the projection of ``F - z`` on the weight's line, ``d2`` the distance to it.
    """
    norm = np.linalg.norm(weights, axis=1)
    shifted = F - ideal
    d1 = np.sum(shifted * weights, axis=1) / norm
    d2 = np.linalg.norm(shifted - d1[:, None] * weights / norm[:, None], axis=1)
    return np.asarray(d1 + theta * d2, dtype=float)


def neighbourhoods(weights: FloatArray, n_neighbors: int) -> IntArray:
    """Return each weight's ``n_neighbors`` nearest weights ``(N, T)``, itself first.

    pymoo's ``argsort(cdist(w, w), kind="quicksort")[:, :T]``.
    """
    # SciPy's cdist, as pymoo: the directions' grid has exact ties, which its last bits order.
    distance = cdist(weights, weights)
    return np.ascontiguousarray(np.argsort(distance, axis=1, kind="quicksort")[:, :n_neighbors])


def neighbourhood_parents(
    rng: np.random.Generator, neighbors: IntArray, n: int, prob: float
) -> IntArray:
    """Return two parents per subproblem ``(len(neighbors), 2)`` (pymoo's selection).

    With probability ``prob`` two distinct neighbours, else two members of the whole
    population of ``n``.
    """
    parents = np.full((len(neighbors), 2), -1, dtype=np.int64)
    for k, near in enumerate(neighbors):
        if rng.random() < prob:
            parents[k] = rng.choice(near, 2, replace=False)
        else:
            parents[k] = rng.permutation(n)[:2]
    return parents


# ---------------------------------------------------------------------------- the optimizer


@optimizers.register("moead")
class MOEAD(NSGA2):
    """MOEA/D: multi-objective optimization by neighbouring scalar subproblems.

    Parameters
    ----------
    n_partitions, inner_partitions
        Das-Dennis divisions, as ``nsga3``'s (default by objectives: 2 -> 19, 3 -> 6, 4 -> 4;
        from 5, two layers). The population is the number of weights: 20 for 2 objectives,
        28 for 3, 20 for 5.
    n_neighbors
        Each subproblem's neighbourhood (default ``min(20, max(2, ceil(N / 5)))`` for ``N``
        weights; pymoo and the paper: 20).
    prob_neighbor_mating
        Probability of mating within the neighbourhood (pymoo: 0.9).
    decomposition
        ``auto`` (Tchebicheff for 2 objectives, PBI for more, as pymoo), ``tchebicheff`` or
        ``pbi``.
    theta
        PBI's penalty (pymoo: 5).
    eta_c, crossover_prob, crossover_prob_var
        SBX (pymoo's MOEA/D: 20, 1, 0.5).
    eta_m, mutation_prob, mutation_prob_var
        Polynomial mutation (pymoo: 20, 0.9, ``1/n_x``).
    constraint_handling
        ``epf`` for constrained objectives (EPF only); ``feasibility`` (the default)
        refuses them.
    ref_point, stall, ftol, init, eq_tolerance, maxiter, xtol, transform, epf
        As ``nsga2``.
    """

    status: ClassVar[Status] = "experimental"
    label: ClassVar[str] = "MOEA/D"
    #: Constraints only through EPF; validation refuses the rest before a run.
    constraints_by_epf_only: ClassVar[bool] = True

    def __init__(
        self,
        *,
        n_partitions: int | None = None,
        inner_partitions: int | None = None,
        n_neighbors: int | None = None,
        prob_neighbor_mating: float = 0.9,
        decomposition: Decomposition = "auto",
        theta: float = 5.0,
        crossover_prob: float = 1.0,
        crossover_prob_var: float = 0.5,
        eta_c: float = 20.0,
        mutation_prob: float = 0.9,
        mutation_prob_var: float | None = None,
        eta_m: float = 20.0,
        ref_point: list[float] | None = None,
        stall: int = 30,
        init: Initialization = "uniform",
        constraint_handling: Literal["feasibility", "epf"] = "feasibility",
        eq_tolerance: float = 1e-4,
        maxiter: int = 100,
        ftol: float = 0.0025,
        xtol: float = 1e-8,
        transform: bool = False,
        epf: EpfOptions = None,
    ) -> None:
        if n_partitions is not None and n_partitions < 1:
            raise ValueError(f"n_partitions must be positive, got {n_partitions}")
        if inner_partitions is not None and inner_partitions < 0:
            raise ValueError(f"inner_partitions must not be negative, got {inner_partitions}")
        if n_neighbors is not None and n_neighbors < 2:
            raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}")
        if not 0.0 <= prob_neighbor_mating <= 1.0:
            raise ValueError(f"prob_neighbor_mating must lie in [0, 1], got {prob_neighbor_mating}")
        if decomposition not in ("auto", "tchebicheff", "pbi"):
            raise ValueError(f"decomposition must be auto, tchebicheff or pbi: {decomposition!r}")
        if theta < 0:
            raise ValueError(f"theta must not be negative, got {theta}")
        super().__init__(
            pop_size=4, crossover_prob=crossover_prob, crossover_prob_var=crossover_prob_var,
            eta_c=eta_c, mutation_prob=mutation_prob, mutation_prob_var=mutation_prob_var,
            eta_m=eta_m, eliminate_duplicates=False, ref_point=ref_point, stall=stall,
            init=init, constraint_handling=constraint_handling, eq_tolerance=eq_tolerance,
            maxiter=maxiter, ftol=ftol, xtol=xtol, transform=transform, epf=epf,
        )  # fmt: skip
        self.n_partitions = n_partitions
        self.inner_partitions = inner_partitions
        self.given_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition: Decomposition = decomposition
        #: The decomposition in use: ``auto`` resolved by the number of objectives.
        self.resolved: Decomposition = decomposition
        self.theta = theta
        self.weights: FloatArray | None = None
        self.neighbors: IntArray | None = None
        self.ideal: FloatArray | None = None

    def set_objectives(self, n_obj: int) -> None:
        """Make the weights for ``n_obj`` objectives, the population and the neighbourhoods."""
        self.weights = directions_for(n_obj, self.n_partitions, self.inner_partitions)
        n = self.weights.shape[0]
        if n < 2:
            raise ValueError(f"MOEA/D needs at least two weights, got {n}")
        self.pop_size = self.n_new = n
        t = self.given_neighbors or min(20, max(2, math.ceil(n / 5)))
        self.neighbors = neighbourhoods(self.weights, min(t, n))
        self.resolved = (
            ("tchebicheff" if n_obj <= 2 else "pbi") if self.decomposition == "auto"
            else self.decomposition
        )  # fmt: skip

    def _setup(self, problem: Any, draws: Any, cov: FloatArray | None) -> None:  # noqa: ANN401
        names = getattr(problem.objective, "objective_names", None)
        if names is not None and (self.weights is None or self.weights.shape[1] != len(names)):
            self.set_objectives(len(names))
        super()._setup(problem, draws, cov)
        if self.constrained and self.constraint_handling != "epf":
            raise ValueError(
                "moead handles constraints by the external penalty only: set "
                'constraint_handling = "epf" with an epf factor r, or use nsga2/nsga3'
            )

    def decompose(self, F: FloatArray, weights: FloatArray) -> FloatArray:
        """Return the decomposed values of paired rows ``F`` and ``weights`` ``(m, n_obj)``."""
        assert self.ideal is not None
        if self.resolved == "pbi":
            return pbi(F, weights, self.ideal, self.theta)
        return tchebicheff(F, weights, self.ideal)

    # ------------------------------------------------------------------ the loop

    def initialize(self) -> None:
        """Set the ideal point from the first population (no survival: member k serves weight k)."""
        mean, _ = self.means(self.F, self.c_eq, self.c_iq)
        self.ideal = np.min(mean, axis=1)
        n = self.X.shape[1]
        self.ranks, self.crowding = np.zeros(n, dtype=np.int64), np.zeros(n)

    def generation(self) -> None:
        """One child per subproblem, evaluated as a batch, then the neighbours replaced."""
        assert self.neighbors is not None
        assert self.weights is not None
        assert self.ideal is not None
        n = self.X.shape[1]
        order = self.rng.permutation(n)[: self.n_new]
        parents = neighbourhood_parents(
            self.rng, self.neighbors[order], n, self.prob_neighbor_mating
        )
        c1, c2 = sbx(
            self.rng, self.X[:, parents[:, 0]], self.X[:, parents[:, 1]], self.xl, self.xu,
            eta=self.eta_c, prob=self.crossover_prob, prob_var=self.crossover_prob_var,
        )  # fmt: skip
        children = polynomial_mutation(
            self.rng, np.column_stack([c1, c2]), self.xl, self.xu, eta=self.eta_m,
            prob=self.mutation_prob, prob_var=self.mutation_prob_var,
        )  # fmt: skip
        m = order.size
        # One of its own mating's two children per subproblem (see the module docstring).
        picks = np.array([int(self.rng.choice(np.array([k, m + k]))) for k in range(m)])
        X = np.ascontiguousarray(children[:, picks])
        F, c_eq, c_iq = self.evaluate_population(X)
        mean, _ = self.means(F, c_eq, c_iq)
        self.ideal = np.min(np.column_stack([self.ideal[:, None], mean]), axis=1)
        for j, k in enumerate(order):
            self._replace(int(k), X[:, j], F[:, :, j], c_eq[:, j], c_iq[:, j])

    def _replace(
        self, k: int, x: FloatArray, f: FloatArray, c_eq: FloatArray, c_iq: FloatArray
    ) -> None:
        """Replace every neighbour of subproblem ``k`` the child improves (pymoo's)."""
        assert self.neighbors is not None
        assert self.weights is not None
        near = self.neighbors[k]
        current, _ = self.means(self.F[:, :, near], self.c_eq[:, near], self.c_iq[:, near])
        child, _ = self.means(f[:, :, None], c_eq[:, None], c_iq[:, None])
        weights = self.weights[near]
        better = near[self.decompose(np.repeat(child.T, len(near), axis=0), weights)
                      < self.decompose(np.ascontiguousarray(current.T), weights)]  # fmt: skip
        if better.size:
            self.X[:, better] = x[:, None]
            self.F[:, :, better] = f[:, :, None]
            self.c_eq[:, better] = c_eq[:, None]
            self.c_iq[:, better] = c_iq[:, None]

    def survivors(
        self, F: FloatArray, cv: FloatArray, n_survive: int
    ) -> tuple[IntArray, IntArray, FloatArray]:
        """Unused: MOEA/D replaces by subproblem (see :meth:`generation`)."""
        raise NotImplementedError("moead replaces neighbours instead of a survival")

    # ------------------------------------------------------------------ state

    def _export_population_extra(self) -> OptimizerState:
        state = dict(super()._export_population_extra())
        state["moead_ideal"] = self.ideal
        return state

    def _restore_population_extra(self, state: Mapping[str, StateValue]) -> None:
        super()._restore_population_extra(state)
        ideal = state.get("moead_ideal")
        self.ideal = None if ideal is None else np.array(ideal, dtype=float)
