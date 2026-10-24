# pymoo references

`operators.npz` holds pymoo 0.6.2's outputs (commit `217af15`, the tag 0.6.2) for the `ga` and `pso` tests
(`tests/unit/algorithms/test_ga_pso.py`):

- SBX, polynomial mutation, binary tournaments and one PSO update, given seeded generators: APEX's
  operators must return the same arrays;
- elitist survival (`FitnessSurvival`) of 7 of 12 members, infeasible ones and ties included;
- the best of 50 generations on Rastrigin (n = 5) over 20 seeds, for GA and PSO, compared
  statistically.

`multiobjective.npz` holds, for `tests/unit/algorithms/test_nsga2.py`:

- non-dominated sorting (2 and 3 objectives), crowding distances, `RankAndCrowding` survival (seed 51) and the
  binary tournament with constraints (seed 61);
- pymoo's hypervolume of random fronts with 2, 3 and 4 objectives, and its front movement
  (`MultiObjectiveSpaceTermination._delta`) between random fronts;
- pymoo's fronts of TNK and Kursawe (no closed form);
- NSGA-II's IGD on ZDT1 (n = 10) and DTLZ2 (n = 12) after 60 generations of 40 members, over 60 seeds.

`manyobjective.npz` holds, for `tests/unit/algorithms/test_nsga3_moead.py` (inputs from a generator
seeded 20261007, each case's draws from its own seed):

- Das-Dennis directions (2, 3, 5 objectives at APEX's defaults; 3 at the papers' 12 partitions; none);
- NSGA-III: the association with reference lines, niching (seed 71), the hyperplane normalisation over two updates,
  the survival of 28 of 40 (twice with one survival, seeds 81 and 82; and with constraints, seed 83), and the
  tournament `comp_by_cv_then_random` (seed 91; distinct violations, as pymoo draws ties unseeded);
- MOEA/D: neighbourhoods, the neighbourhood selection (seed 101), Tchebicheff and PBI;
- statistics over 60 seeds: NSGA-III's IGD on DTLZ2 (3 and 5 objectives, 60 generations), MOEA/D's and
  ParallelMOEAD's on ZDT1 and DTLZ2 (100 generations).

`operators.json` records the pymoo, NumPy and Python versions.

Regenerate in a separate environment; pymoo is never in APEX's environment, as PET is not:

```bash
uv venv .venv-pymoo --python 3.12
uv pip install --python .venv-pymoo/bin/python "pymoo==0.6.2"
env -u PYTHONPATH .venv-pymoo/bin/python tools/pymoo_reference/generate.py
```

The inputs come from the generator seeded with 20261006; each case's draws come from its own seed (in the file).
