---
audience: public
---

# pymoo ports

APEX's population optimizers port the mathematics of pymoo 0.6.2 (commit `217af15`,
Apache-2.0; `THIRD_PARTY_NOTICES.md`). pymoo is not a dependency: it writes the reference
outputs in a separate environment (`tools/pymoo_reference/generate.py`,
`tests/data/reference/pymoo/`).

| pymoo | APEX | Reference test | Tolerance |
|---|---|---|---|
| `SBX` / `cross_sbx` with `Crossover.do` | `ga.sbx` | `test_sbx_is_pymoos` (two cases) | `rtol` 1e-13, `atol` 1e-15 (bit-exact on the platform that generated the reference; `pow` rounds by the platform's libm) |
| `PM` / `mut_pm` with `Mutation.do` | `ga.polynomial_mutation` | `test_polynomial_mutation_is_pymoos` (two cases) | as SBX |
| `TournamentSelection(comp_by_cv_and_fitness)` | `ga.tournament` (size 2) | `test_tournament_is_pymoos` | exact |
| `FitnessSurvival` | `population.rank_order` | `test_survival_is_pymoos` | exact |
| `pso_equation` | `pso.velocity_update` | `test_pso_update_is_pymoos` | exact |
| `GA` on Rastrigin (n = 5), 50 generations, 20 seeds | `ga` | `test_the_best_after_50_generations_agrees_with_pymoo` | Mann-Whitney p > 0.05; medians within 1.5x |
| `PSO` (fixed weights) on the same | `pso` | the same | the same |
| `NonDominatedSorting` (`fast_non_dominated_sort`), 2 and 3 objectives | `multiobjective.pareto_ranks` | `test_non_dominated_sorting_is_pymoos` | exact |
| `calc_crowding_distance` | `multiobjective.crowding_distance` | `test_crowding_distance_is_pymoos` | `rtol` 1e-13 |
| `RankAndCrowding` | `multiobjective.survive` | `test_rank_and_crowding_survival_is_pymoos` | exact |
| `binary_tournament` (`comp_by_dom_and_crowding`, with constraints) | `multiobjective.tournament` | `test_the_binary_tournament_is_pymoos` | exact |
| `HV` indicator, 2 to 4 objectives | `multiobjective.hypervolume` | `test_the_hypervolume_is_pymoos` | `rtol` 1e-12 |
| `MultiObjectiveSpaceTermination._delta` | `multiobjective.front_movement` | `test_the_front_movement_is_pymoos` (2, 3 objectives, a degenerate one) | `rtol` 1e-12 |
| `NSGA2` on ZDT1 (n = 10) and DTLZ2 (n = 12), population 40, 60 generations, 60 seeds | `nsga2` | `test_the_igd_after_60_generations_agrees_with_pymoos` | Mann-Whitney p > 0.05; median IGD within 1.25x |
| `ZDT1`–`ZDT3`, `DTLZ2`, `Kursawe`, `BNH`, `TNK`, `SRN` and their fronts | `test_problems` | `test_every_test_problem_reaches_its_known_front` (KUR and TNK against pymoo's fronts) | IGD below 2.5 % of the front's range (DTLZ2: 15 %), every member feasible |

## Known differences

- **Layout.** Members are columns `(n_x, P)`; pymoo uses rows. The operators draw pymoo's random numbers in pymoo's order and shapes, then transpose.
- **Population.** `pop_size` defaults to 20 (pymoo: 100), because evaluations here are simulator runs. Crossover and mutation probabilities are options.
- **Start.** `x0` is a member of the initial population.
- **GA.** Tournaments larger than 2 take the best of the group (pymoo's comparison is binary). Duplicates are judged by the largest coordinate difference (pymoo: Euclidean distance; at 1e-16 the two agree).
- **PSO.** pymoo's parameter adaptation and its perturbation of the best particle are left out. Out of bounds, a particle is reflected or clipped (pymoo redraws its velocity, then re-initialises it). A ring topology is offered. Between two infeasible members of equal violation, the smaller objective wins (pymoo keeps the old one).
- **Constraints.** pymoo's violation (sum, equalities within `eq_tolerance`), or APEX's external penalty (`constraint_handling = "epf"`), which pymoo lacks.
- **Convergence.** `stall` generations without a relative improvement above `ftol`, or the population's spread below `xtol` (pymoo: its default termination).
- **NSGA-II**. `pop_size` defaults to 40 (pymoo: 100). It stops after `stall` (30) generations in a row in which the front moved less than `ftol` (0.0025), by pymoo's `MultiObjectiveSpaceTermination` measure taken every generation (pymoo's default: every sixth, `ftol` 0.005, over 50 generations). The front's representative (closest to the ideal point in normalised objectives) and the hypervolume history are APEX's. ZDT3's `pareto_front` drops the points of pymoo's rounded regions that other points dominate. SCH and FON are not in pymoo 0.6.2 and are written from their papers.
- **NSGA-III and MOEA/D**.
  - Defaults are smaller than the papers' (see docs/optimization.md).
  - NSGA-III's tournament draws a tie between two infeasible members of equal violation from the run's generator. pymoo's `comp_by_cv_then_random` calls `compare` without its generator, so pymoo draws such a tie from a fresh, unseeded one, which cannot be reproduced.
  - MOEA/D is pymoo's `ParallelMOEAD` (one batch per generation), with two differences:
    - **A pymoo bug, not reproduced.** Each subproblem takes one of its own mating's two children. pymoo flattens the children (all first children, then all second) and pairs consecutive ones, so subproblem `k` gets a child of the matings `2k` and `2k + 1`, or a second child: a child of other subproblems' parents, against its comment ("select a random offspring from each mating").
    - Constraints only through EPF; pymoo's MOEA/D refuses constraints.

