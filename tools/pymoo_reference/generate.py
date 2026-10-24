"""Generate pymoo reference outputs for the ``ga`` and ``pso`` operator tests.

Run with a pymoo interpreter, never the APEX environment (pymoo is not an APEX dependency,
as PET is not), with ``PYTHONPATH`` cleared::

    uv venv .venv-pymoo --python 3.12 && uv pip install --python .venv-pymoo/bin/python \
        "pymoo==0.6.2"
    env -u PYTHONPATH .venv-pymoo/bin/python tools/pymoo_reference/generate.py

It writes ``tests/data/reference/pymoo/operators.npz``: for each case the inputs
(``<case>_in_*``), pymoo's outputs (``<case>_out_*``) and the seed of the
``numpy.random.Generator`` pymoo drew from (``<case>_seed``). APEX's operators, given a
generator with the same seed, must return the same arrays. pymoo's own classes are used
(``SBX().do``, ``PM().do``, ``TournamentSelection(...).do``, ``pso_equation``), so the
references test APEX's port against pymoo, not against a copy of it. pymoo's layout is rows
(one member per row); the APEX tests transpose at their boundary.

This script is outside the ``apex`` package: it is the only place that imports pymoo.
"""

import json
import os
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "pymoo"
PYMOO_VERSION = "0.6.2"
#: The release's commit (tag 0.6.2), recorded with the references.
PYMOO_COMMIT = "217af1554f864ab9358d88eb883091adb78cb44a"


def main() -> None:
    """Write the reference file."""
    if os.environ.get("PYTHONPATH"):
        sys.exit("clear PYTHONPATH (env -u PYTHONPATH ...) so no other checkout shadows pymoo")
    version = metadata.version("pymoo")
    if version != PYMOO_VERSION:
        sys.exit(f"pymoo {PYMOO_VERSION} is the reference version; this is {version}")
    from pymoo.algorithms.soo.nonconvex.ga import comp_by_cv_and_fitness
    from pymoo.algorithms.soo.nonconvex.pso import pso_equation
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.selection.tournament import TournamentSelection

    arrays: dict[str, np.ndarray] = {}
    setup = np.random.default_rng(20261006)  # the inputs; each case's draws use its own seed

    # SBX on m pairs, with bounds that some children reach (clamping) and one equal pair.
    n_x, m = 4, 12
    xl, xu = np.array([-1.0, 0.0, 2.0, -5.0]), np.array([1.0, 3.0, 2.5, 5.0])
    parents = xl + setup.random((2 * m, n_x)) * (xu - xl)
    parents[1] = parents[0]  # pair 0: equal parents, no crossover
    problem = Problem(n_var=n_x, xl=xl, xu=xu)
    for case, kwargs in (("sbx", {}), ("sbx_eta5", {"eta": 5.0, "prob": 1.0})):
        seed = 11 if case == "sbx" else 12
        pop = Population.new(X=parents)
        mating = np.arange(2 * m).reshape(m, 2)
        off = SBX(**kwargs).do(problem, pop, mating, random_state=np.random.default_rng(seed))
        arrays[f"{case}_in_parents"] = parents
        arrays[f"{case}_in_xl"], arrays[f"{case}_in_xu"] = xl, xu
        arrays[f"{case}_out_children"] = off.get("X")
        arrays[f"{case}_seed"] = np.array(seed)
        arrays[f"{case}_eta"] = np.array(kwargs.get("eta", 15.0))
        arrays[f"{case}_prob"] = np.array(kwargs.get("prob", 0.9))

    # Polynomial mutation of n members, some on a bound.
    members = xl + setup.random((30, n_x)) * (xu - xl)
    members[0, 0], members[1, 3] = xl[0], xu[3]
    for case, kwargs in (("pm", {}), ("pm_eta5", {"eta": 5.0, "prob": 1.0})):
        seed = 21 if case == "pm" else 22
        pop = Population.new(X=members.copy())
        out = PM(**kwargs).do(problem, pop, random_state=np.random.default_rng(seed))
        arrays[f"{case}_in_members"] = members
        arrays[f"{case}_out_members"] = out.get("X")
        arrays[f"{case}_seed"] = np.array(seed)
        arrays[f"{case}_eta"] = np.array(kwargs.get("eta", 20.0))
        arrays[f"{case}_prob"] = np.array(kwargs.get("prob", 0.9))

    # Binary tournaments: some infeasible members, and ties drawn at random.
    n = 10
    F = np.round(setup.random(n), 1)  # rounded: ties
    CV = np.where(setup.random(n) < 0.3, np.round(setup.random(n), 1), 0.0)
    pop = Population.new(X=setup.random((n, 2)), F=F[:, None], CV=CV[:, None])
    selection = TournamentSelection(func_comp=comp_by_cv_and_fitness)
    picks = selection.do(problem, pop, 16, 2, False, random_state=np.random.default_rng(31))
    arrays["tournament_in_value"], arrays["tournament_in_cv"] = F, CV
    arrays["tournament_out_winners"] = np.asarray(picks, dtype=np.int64).reshape(-1)
    arrays["tournament_seed"] = np.array(31)

    # Elitist survival (FitnessSurvival): the best 7 of 12, infeasible ones and ties included.
    from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival

    n_s = 12
    F_s = np.round(setup.random(n_s), 1)
    CV_s = np.where(setup.random(n_s) < 0.4, np.round(setup.random(n_s), 1), 0.0)
    pop = Population.new(X=np.arange(n_s, dtype=float)[:, None], F=F_s[:, None], CV=CV_s[:, None])
    kept = FitnessSurvival().do(problem, pop, n_survive=7)
    arrays["survival_in_value"], arrays["survival_in_cv"] = F_s, CV_s
    arrays["survival_out_kept"] = np.asarray(kept.get("X")[:, 0], dtype=np.int64)

    # One PSO update, every particle staying within the bounds (no bound handling).
    n_p = 8
    X = 0.4 + 0.2 * setup.random((n_p, n_x))
    P_X = 0.4 + 0.2 * setup.random((n_p, n_x))
    S_X = np.repeat(P_X[[3]], n_p, axis=0)
    V = 0.01 * (setup.random((n_p, n_x)) - 0.5)
    V_max = 0.2 * np.ones(n_x)
    Xp, Vp = pso_equation(X, P_X, S_X, V, V_max, 0.7298, 1.49618, 1.49618,
                          random_state=np.random.default_rng(41))  # fmt: skip
    for key, value in (("X", X), ("P_X", P_X), ("S_X", S_X), ("V", V), ("V_max", V_max)):
        arrays[f"pso_in_{key}"] = value
    arrays["pso_out_X"], arrays["pso_out_V"] = Xp, Vp
    arrays["pso_seed"] = np.array(41)

    # Statistics: the best of 50 generations on Rastrigin (n = 5, [-5.12, 5.12]) over 20 seeds,
    # population 20, pymoo's default operators; PSO with APEX's fixed weights (no adaptation,
    # no perturbation of the best). Termination: exactly 50 generations.
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.optimize import minimize
    from pymoo.problems.single import Rastrigin

    rastrigin = Rastrigin(n_var=5)
    for case, make in (
        ("stats_ga", lambda: GA(pop_size=20, eliminate_duplicates=True)),
        ("stats_pso", lambda: PSO(pop_size=20, w=0.7298, c1=1.49618, c2=1.49618,
                                  adaptive=False, pertube_best=False)),
    ):  # fmt: skip
        best = [float(minimize(rastrigin, make(), ("n_gen", 50), seed=seed, verbose=False).F[0])
                for seed in range(20)]  # fmt: skip
        arrays[f"{case}_best"] = np.array(best)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "operators.npz", **arrays)
    np.savez(OUT / "multiobjective.npz", **_multiobjective(setup))
    np.savez(OUT / "manyobjective.npz", **_manyobjective())
    meta = {
        "pymoo": version,
        "pymoo_commit": PYMOO_COMMIT,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    (OUT / "operators.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT / 'operators.npz'} ({len(arrays)} arrays), multiobjective.npz and "
        "manyobjective.npz "
        f"with pymoo {version}"
    )


def _multiobjective(setup: np.random.Generator) -> dict[str, np.ndarray]:
    """Return the references: NSGA-II's kernels, hypervolumes, stored fronts, statistics."""
    from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.operators.selection.tournament import TournamentSelection
    from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
    from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    out: dict[str, np.ndarray] = {}
    # Fronts of 30 members (rounded: ties and duplicates), sorting and crowding.
    F = np.round(setup.random((30, 2)) * 4) / 4
    fronts, rank = NonDominatedSorting().do(F, return_rank=True)
    out["nds_in_F"], out["nds_out_rank"] = F, np.asarray(rank, dtype=np.int64)
    front = F[fronts[0]]
    out["crowding_in_F"] = front
    out["crowding_out"] = calc_crowding_distance(front)
    F3 = setup.random((25, 3))
    out["nds3_in_F"] = F3
    out["nds3_out_rank"] = np.asarray(NonDominatedSorting().do(F3, return_rank=True)[1],
                                      dtype=np.int64)  # fmt: skip
    # RankAndCrowding survival of 12 of 30 (the last front cut by crowding, ties drawn).
    pop = Population.new(X=np.arange(30, dtype=float)[:, None], F=F)
    problem = Problem(n_var=1, n_obj=2, xl=0.0, xu=30.0)
    kept = RankAndCrowding().do(problem, pop, n_survive=12, random_state=np.random.default_rng(51))
    out["survival_in_F"], out["survival_seed"] = F, np.array(51)
    out["survival_out_kept"] = np.asarray(kept.get("X")[:, 0], dtype=np.int64)
    # NSGA-II's binary tournament with constraints, rank and crowding (pymoo's attributes).
    CV = np.where(setup.random(30) < 0.3, np.round(setup.random(30), 1), 0.0)
    pop = Population.new(X=np.arange(30, dtype=float)[:, None], F=F, CV=CV[:, None])
    crowding = np.full(30, np.inf)
    _, rank = NonDominatedSorting().do(F, return_rank=True)
    for r in range(int(rank.max()) + 1):
        members = np.flatnonzero(rank == r)
        crowding[members] = calc_crowding_distance(F[members])
    pop.set("rank", rank, "crowding", crowding)

    class Algorithm:
        tournament_type = "comp_by_dom_and_crowding"
        random_state = np.random.default_rng(61)

    selection = TournamentSelection(func_comp=binary_tournament)
    picks = selection.do(problem, pop, 20, 2, False, random_state=Algorithm.random_state,
                         algorithm=Algorithm())  # fmt: skip
    out["tournament_in_F"], out["tournament_in_cv"] = F, CV
    out["tournament_in_crowding"] = crowding
    out["tournament_out_winners"] = np.asarray(picks, dtype=np.int64).reshape(-1)
    out["tournament_seed"] = np.array(61)
    # Hypervolumes of random fronts (2, 3 and 4 objectives) against pymoo's HV.
    for n_obj in (2, 3, 4):
        H = setup.random((20, n_obj))
        ref = np.full(n_obj, 1.1)
        out[f"hv{n_obj}_in_F"], out[f"hv{n_obj}_in_ref"] = H, ref
        out[f"hv{n_obj}_out"] = np.array(HV(ref_point=ref)(H))
    # The front's movement between generations (pymoo's MultiObjectiveSpaceTermination), the
    # nsga2 stall's measure: 2 and 3 objectives, and one with a degenerate objective.
    from pymoo.termination.ftol import MultiObjectiveSpaceTermination

    def movement(prev: np.ndarray, cur: np.ndarray) -> float:
        def data(F: np.ndarray) -> dict:
            return {"ideal": F.min(axis=0), "nadir": F.max(axis=0), "F": F, "feas": True}

        return float(MultiObjectiveSpaceTermination()._delta(data(prev), data(cur)))

    for tag, (m_prev, m_cur, n_obj) in {"2": (15, 12, 2), "3": (20, 18, 3)}.items():
        prev, cur = setup.random((m_prev, n_obj)), setup.random((m_cur, n_obj)) * 0.9
        out[f"move{tag}_in_prev"], out[f"move{tag}_in_cur"] = prev, cur
        out[f"move{tag}_out"] = np.array(movement(prev, cur))
    prev, cur = setup.random((6, 2)), setup.random((5, 2))
    cur[:, 1] = 0.4  # no range in the second objective
    out["moveflat_in_prev"], out["moveflat_in_cur"] = prev, cur
    out["moveflat_out"] = np.array(movement(prev, cur))
    # The fronts pymoo stores (TNK, Kursawe), for the analytic problems' tests.
    out["front_tnk"] = get_problem("tnk").pareto_front()
    out["front_kur"] = get_problem("kursawe").pareto_front()
    # NSGA-II's IGD after a fixed budget over 60 seeds (statistics validation 3).
    for name, problem_name, kwargs, n_gen in (
        ("zdt1", "zdt1", {"n_var": 10}, 60), ("dtlz2", "dtlz2", {"n_var": 12, "n_obj": 3}, 60),
    ):  # fmt: skip
        prob = get_problem(problem_name, **kwargs)
        indicator = IGD(prob.pareto_front() if name == "zdt1" else _sphere_front())
        igds = [float(indicator(minimize(prob, NSGA2(pop_size=40), ("n_gen", n_gen), seed=s,
                                         verbose=False).F)) for s in range(60)]  # fmt: skip
        out[f"stats_{name}_igd"] = np.array(igds)
    return out


def _manyobjective() -> dict[str, np.ndarray]:
    """Return the references: NSGA-III's and MOEA/D's kernels, and their statistics."""
    from pymoo.algorithms.moo.moead import MOEAD, NeighborhoodSelection, ParallelMOEAD
    from pymoo.algorithms.moo.nsga3 import (
        NSGA3,
        HyperplaneNormalization,
        ReferenceDirectionSurvival,
        associate_to_niches,
        comp_by_cv_then_random,
        niching,
    )
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.decomposition.pbi import PBI
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.indicators.igd import IGD
    from pymoo.operators.selection.tournament import TournamentSelection
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.util.reference_direction import das_dennis
    from scipy.spatial.distance import cdist

    setup = np.random.default_rng(20261007)  # this file's inputs, apart from the 2a ones
    out: dict[str, np.ndarray] = {}
    # Das-Dennis directions: APEX's defaults (2, 3, 5 objectives), the papers' 3, and none.
    for p, m in ((19, 2), (6, 3), (3, 5), (12, 3), (0, 3)):
        out[f"das_{p}_{m}"] = das_dennis(p, m)
    # APEX's defaults from 5 objectives: two layers (boundary 2, inner 1 scaled by 0.5).
    from pymoo.util.ref_dirs import get_reference_directions

    def two_layers(m: int) -> np.ndarray:
        return get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", m, n_partitions=2, scaling=1.0),
            get_reference_directions("das-dennis", m, n_partitions=1, scaling=0.5),
        )

    for m in (5, 6, 8):
        out[f"layers_{m}"] = two_layers(m)
    ref = das_dennis(6, 3)
    # NSGA-III: association with the reference lines.
    F = setup.random((15, 3))
    ideal, nadir = F.min(axis=0) - 0.05, F.max(axis=0) + 0.1
    niche, dist, matrix = associate_to_niches(F, ref, ideal, nadir)
    out.update(
        {
            "assoc_in_F": F,
            "assoc_in_ideal": ideal,
            "assoc_in_nadir": nadir,
            "assoc_out_niche": niche,
            "assoc_out_dist": dist,
            "assoc_out_matrix": matrix,
        }
    )
    # Niching: 9 of a last front of 14 members, given niche counts.
    counts = setup.integers(0, 3, len(ref))
    of = setup.integers(0, len(ref), 14)
    d = setup.random(14)
    picks = niching(
        Population.new(X=np.zeros((14, 1))),
        9,
        counts.copy(),
        of,
        d,
        random_state=np.random.default_rng(71),
    )
    out.update(
        {
            "niching_in_counts": counts,
            "niching_in_niche": of,
            "niching_in_dist": d,
            "niching_seed": np.array(71),
            "niching_out": np.asarray(picks, dtype=np.int64),
        }
    )
    # The hyperplane normalisation over two updates (its state persists).
    norm = HyperplaneNormalization(3)
    for case, scale in (("1", 1.0), ("2", 0.8)):
        Fn = setup.random((20, 3)) * scale
        nds = NonDominatedSorting().do(Fn, only_non_dominated_front=True)
        norm.update(Fn, nds=nds)
        out.update(
            {
                f"norm{case}_in_F": Fn,
                f"norm{case}_in_nds": np.asarray(nds),
                f"norm{case}_out_ideal": norm.ideal_point.copy(),
                f"norm{case}_out_worst": norm.worst_point.copy(),
                f"norm{case}_out_extremes": norm.extreme_points.copy(),
                f"norm{case}_out_nadir": norm.nadir_point.copy(),
            }
        )
    # Survival of 28 of 40, twice with one survival (its normalisation persists), and once
    # with constraints (distinct violations: pymoo draws ties from an unseeded generator).
    survival = ReferenceDirectionSurvival(ref)
    problem = Problem(n_var=1, n_obj=3, xl=0.0, xu=100.0)
    for case, seed in (("a", 81), ("b", 82)):
        Fs = setup.random((40, 3))
        pop = Population.new(X=np.arange(40, dtype=float)[:, None], F=Fs)
        kept = survival.do(problem, pop, n_survive=28, random_state=np.random.default_rng(seed))
        out.update(
            {
                f"survival_{case}_in_F": Fs,
                f"survival_{case}_seed": np.array(seed),
                f"survival_{case}_out_kept": np.asarray(kept.get("X")[:, 0], dtype=np.int64),
            }
        )
    Fc = setup.random((40, 3))
    CV = np.where(setup.random(40) < 0.4, 0.01 + setup.permutation(40) / 40.0, 0.0)
    problem_c = Problem(n_var=1, n_obj=3, n_ieq_constr=1, xl=0.0, xu=100.0)
    pop = Population.new(X=np.arange(40, dtype=float)[:, None], F=Fc, CV=CV[:, None])
    kept = ReferenceDirectionSurvival(ref).do(
        problem_c, pop, n_survive=28, random_state=np.random.default_rng(83)
    )
    out.update(
        {
            "survival_c_in_F": Fc,
            "survival_c_in_cv": CV,
            "survival_c_seed": np.array(83),
            "survival_c_out_kept": np.asarray(kept.get("X")[:, 0], dtype=np.int64),
        }
    )
    # NSGA-III's tournament (comp_by_cv_then_random): distinct violations, so no tie draws.
    tour = TournamentSelection(func_comp=comp_by_cv_then_random)
    pop = Population.new(X=np.arange(40, dtype=float)[:, None], F=Fc, CV=CV[:, None])
    winners = tour.do(problem_c, pop, 20, 2, False, random_state=np.random.default_rng(91))
    out.update(
        {
            "tournament3_in_cv": CV,
            "tournament3_seed": np.array(91),
            "tournament3_out": np.asarray(winners, dtype=np.int64).reshape(-1),
        }
    )
    # MOEA/D: neighbourhoods, the neighbourhood selection, and both decompositions.
    neighbors = np.argsort(cdist(ref, ref), axis=1, kind="quicksort")[:, :6]
    order = setup.permutation(len(ref))
    pop = Population.new(X=np.zeros((len(ref), 1)))
    parents = NeighborhoodSelection(prob=0.9).do(
        problem,
        pop,
        len(ref),
        2,
        False,
        neighbors=neighbors[order],
        random_state=np.random.default_rng(101),
    )
    out.update(
        {
            "neighbors_out": neighbors,
            "selection_in_order": order,
            "selection_seed": np.array(101),
            "selection_out": np.asarray(parents, dtype=np.int64),
        }
    )
    Fd, z = setup.random((6, 3)), np.array([-0.1, 0.0, -0.05])
    weights = ref[neighbors[3]]
    child = setup.random(3)
    out.update(
        {
            "decomp_in_F": Fd,
            "decomp_in_weights": weights,
            "decomp_in_ideal": z,
            "decomp_in_child": child,
            "tcheb_out": Tchebicheff().do(Fd, weights=weights, ideal_point=z),
            "tcheb_child_out": Tchebicheff().do(child, weights=weights, ideal_point=z),
            "pbi_out": PBI(theta=5.0).do(Fd, weights=weights, ideal_point=z),
            "pbi_child_out": PBI(theta=5.0).do(child, weights=weights, ideal_point=z),
        }
    )
    # Statistics over 60 seeds: NSGA-III on DTLZ2 (3 and 5 objectives, APEX's default
    # directions), MOEA/D (pymoo's loop-wise, and its ParallelMOEAD) on ZDT1 and DTLZ2.
    for name, n_obj, n_var, refs, pop_size in (("3", 3, 12, das_dennis(6, 3), 28),
                                               ("5", 5, 14, two_layers(5), 20)):  # fmt: skip
        prob = get_problem("dtlz2", n_var=n_var, n_obj=n_obj)
        front = _sphere_front() if n_obj == 3 else _sphere_points(n_obj, 400)
        indicator = IGD(front)
        igds = [float(indicator(minimize(prob, NSGA3(ref_dirs=refs, pop_size=pop_size),
                                         ("n_gen", 60), seed=s, verbose=False).F))
                for s in range(60)]  # fmt: skip
        out[f"stats_nsga3_dtlz2_{name}_igd"] = np.array(igds)
    for name, problem_name, kwargs, p, n_obj, t in (
        ("zdt1", "zdt1", {"n_var": 10}, 19, 2, 4), ("dtlz2", "dtlz2", {"n_var": 12, "n_obj": 3},
                                                    6, 3, 6),
    ):  # fmt: skip
        prob = get_problem(problem_name, **kwargs)
        indicator = IGD(prob.pareto_front() if name == "zdt1" else _sphere_front())
        for variant, cls in (("moead", MOEAD), ("parallel", ParallelMOEAD)):
            igds = [
                float(indicator(minimize(prob, cls(ref_dirs=das_dennis(p, n_obj), n_neighbors=t),
                                         ("n_gen", 100), seed=s, verbose=False).F))
                for s in range(60)
            ]  # fmt: skip
            out[f"stats_{variant}_{name}_igd"] = np.array(igds)
    return out


def _sphere_points(n_obj: int, n_points: int) -> np.ndarray:
    """DTLZ2's front for n_obj objectives as APEX's tests make it (Das-Dennis on the sphere)."""
    from math import comb

    from pymoo.util.reference_direction import das_dennis

    p = 1
    while comb(n_obj + p, p + 1) <= n_points:
        p += 1
    d = das_dennis(p, n_obj)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _sphere_front() -> np.ndarray:
    """DTLZ2's front as APEX's tests grid it (angles), rows."""
    side = 20
    a, b = np.meshgrid(np.linspace(0, np.pi / 2, side), np.linspace(0, np.pi / 2, side))
    a, b = a.ravel(), b.ravel()
    return np.column_stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])


if __name__ == "__main__":
    main()
