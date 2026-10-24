# Third-party notices

APEX is GPL-3.0-or-later (`LICENSE`). Parts of it are adapted from other open-source projects,
as each module docstring records.

## pymoo

- Project: pymoo, multi-objective optimization in Python, github.com/anyoptimization/pymoo
  (Blank, J. & Deb, K. (2020). pymoo: Multi-objective optimization in Python. IEEE Access, 8,
  89497-89509).
- Version: 0.6.2, commit `217af1554f864ab9358d88eb883091adb78cb44a`.
- Licence: Apache License 2.0, reproduced in `LICENSES/Apache-2.0-pymoo.txt`.
- Adapted in APEX: `src/apex/algorithms/optimization/population.py`, `ga.py` and
  `pso.py`, which port pymoo's sampling, constraint violation, fitness survival, tournament
  selection, SBX crossover, polynomial mutation, mating loop and PSO update. The changes are
  listed under "Differences" in each module docstring.
- Adapted in APEX: `multiobjective.py` and `nsga2.py` (non-dominated sorting,
  crowding distance, rank-and-crowding survival, the binary tournament, the hypervolume
  indicator and NSGA-II), and `test_problems.py` with the examples in
  `examples/toy/multi_objective/` (pymoo's ZDT, DTLZ2, Kursawe, BNH, TNK and SRN problems).
- Adapted in APEX: `reference_directions.py` (Das-Dennis), `nsga3.py` (NSGA-III's
  survival, normalisation, niching and tournament), `moead.py` (MOEA/D's ParallelMOEAD,
  neighbourhoods, Tchebicheff and PBI), and DTLZ1 and the scalable DTLZ2 in
  `test_problems.py`, with their examples.
- pymoo itself is not a dependency of APEX: it is installed only in a separate environment
  that writes the reference outputs of the tests (`tools/pymoo_reference/generate.py`).

The methods ported from PET (GPL-3.0) and the code adapted from ERT and
iterative_ensemble_smoother (GPL-3.0) are under the same licence family as APEX; their
origins are recorded in the module docstrings.
