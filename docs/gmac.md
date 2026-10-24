---
audience: public
---

# GMAC and C-GIES: GIES variants and soft constraints

## Goal

The update strategy `gmac` generalises GIES's RLM-MAC update. Its `constraints` option adds soft equality and inequality constraints, which gives C-GIES. By the end of this page you will be able to:

- choose the step penalty of the update (the metric terms);
- add soft constraints, such as a box on PERMX or a target histogram;
- read the constraint cost and find out why an update was refused;
- add your own transform, norm, constraint function or distance metric.

Both methods are published:
- GMAC: X. Luo (2021). Novel iterative ensemble smoothers derived from a class of generalized cost functions. *Computational Geosciences* 25(3), 1159–1189. [doi:10.1007/s10596-021-10046-1](https://doi.org/10.1007/s10596-021-10046-1)
- C-GIES: X. Luo and W. C. Cruz (2022). Data assimilation with soft constraints (DASC) through a generalized iterative ensemble smoother. *Computational Geosciences* 26(3), 571–594. [doi:10.1007/s10596-022-10137-7](https://doi.org/10.1007/s10596-022-10137-7)

This page is the user's guide. The differences from the authors' MATLAB implementation, and how APEX's was validated against it, are in [migration/gmac.md](migration/gmac.md).

## Prerequisites

`gmac` is a strategy, so it needs a method whose loop supplies the forecast of the mean model (the model at the ensemble mean). Two methods do:

- `gies`, with its own damping or with `lm`;
- `lm_enrml`, with a positive damping minimum. The default of 0.01 in the `lm` scheme is fine.

`es`, `esmda` and `gn_enrml` refuse `gmac`.

**GMAC suits small ensembles.** With member-dependent terms (any `l1sq` or `lpq` norm, or a `state` transform with a non-quadratic norm) or with `constraints`, every trial forms and decomposes one N × N metric per member. The cost grows as N⁴: doubling the ensemble makes each trial about 16 times slower. APEX warns above 200 members ("cost ~N^4"). At N = 500 a run takes tens of minutes even on a model with three parameters, where the simulations themselves are instant. With only `l2sq` terms the metric is the same for every member, and a trial costs about as much as RLM-MAC's.

A minimal valid pairing with `gies` needs only the method and the strategy. Without `terms` or `code`, the metric is one identity `l2sq` term (the same as `code = "100"`), which is RLM-MAC exactly:

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"
```

With `lm_enrml` it is the same, and you leave the `lm` damping at its default (0.01), which is positive:

```toml
[method]
name = "lm_enrml"

[method.strategy]
name = "gmac"
```

If you set a damping minimum yourself, keep it above zero. Otherwise `lm_enrml` and `gmac` cannot be combined.

## The update

Symbols used below:

- N is the ensemble size, and m_j is member j. m̄ is the ensemble mean.
- ΔX holds the state anomalies (members minus the mean), an (n × N) matrix.
- S holds the simulated-data anomalies, whitened by the observation-error factor L. So M_D = SᵀS is the whitened data metric.
- d_j is the perturbed observation of member j, and y_j its simulated data. The whitened innovation is L⁻¹(d_j − y_j).
- γ is the damping scale the loop supplies (the damping of `gies`, or the `lm` damping).

For each member j, GMAC minimises the data mismatch plus a step penalty R[S(m⁺) − S(m_j)]. The penalty is a sum of **metric terms**, each a transform S and a norm. Together they give the member's step metric M_j (N × N). The update is

m_j⁺ = m_j + ΔX (M_D + γ M_j)⁻¹ Sᵀ L⁻¹ (d_j − y_j).

C-GIES adds **constraint terms** on the data side. A constraint c(m) = 0 (equality) or c(m) ≤ 0 (inequality) is a "perfect measurement" of 0 with a distance metric D. Let x_j = 0 − c(m_j), let w_j be the constraint's weight, and let S_c be the ensemble anomalies of c about its value at the mean model (no gradient of c is needed). Then M_D and the innovation term each get an extra term:

- w_j S_cᵀ ∇²D(x_j) S_c is added to M_D;
- w_j S_cᵀ ∇D(x_j) is added to Sᵀ L⁻¹ (d_j − y_j).

## Steps

1. Pick a method (`gies` or `lm_enrml`) and set `[method.strategy] name = "gmac"`.
2. Choose the step penalty: `terms` or the shorthand `code`. With one identity `l2sq` term you get RLM-MAC.
3. Optionally choose `form`, `subspace` or `direct`.
4. Optionally add `constraints`, each a `{ function, metric, weight }` table.
5. Add a `constraint_cost` metric to see how well the constraints are met.

### Metric terms (`terms` or `code`)

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[method.strategy.options]
energy = 0.99
form = "subspace"

[[method.strategy.options.terms]]                # I l2^2: RLM-MAC
transform = { name = "identity" }
norm = { name = "l2sq" }
weight = { rule = "trace", alpha = 0.5 }

[[method.strategy.options.terms]]                # TV l1^2 of PERMX
transform = { name = "tv", options = { variables = ["permx"] } }
norm = { name = "l1sq" }
weight = { rule = "trace", alpha = 0.5 }
```

- **Transforms** (registry `gmac_transforms`):
  - `identity`: RLM-MAC's I. It takes only a norm with a constant Hessian (`l2sq`).
  - `state` (`variables`): the raw variables.
  - `tv` (`variables`, `scale_per_variable`): first differences along each grid axis, in Eclipse order. It needs `[[state.variables]] grid`.
  - `histogram_entropy` (`variables`, `bins` = 100, `ranges`): count-weighted entropy per bin. Within an update all members and the mean share the bins. `ranges` fixes them per variable. Otherwise they span the ensemble's range at each update.
- **Norms** (registry `gmac_norms`):
  - `l2sq`: ‖x‖₂²;
  - `l1sq`: ‖x‖₁², with a rank-one Hessian;
  - `lpq` (`p`, `q`, `floor`): ‖x‖_p^q with its exact Hessian. p = q = 1 is refused.
- **Weights:**
  - `trace` (`alpha`): α N / tr(M_k), the paper's rule, so each term counts by α whatever its scale;
  - `fixed` (`value`): multiplies half the Hessian, as the paper's R = ½ Σ w‖·‖.
- **`code`** is the paper's shorthand instead of `terms`, with equal α over the bits set (`variables` names the variables for V and H):
  - 5 bits over (I ℓ₂², V ℓ₂², V ℓ₁², H ℓ₂², H ℓ₁²), e.g. `code = "10100"`;
  - or 3 bits over (I ℓ₂², V ℓ₂², V ℓ₁²).
- **`form`:**
  - `subspace` (default): the paper's algorithm. The data are reduced by a truncated SVD (TSVD) that keeps the fraction `energy` of the singular-value energy (0.99 keeps 99 %). The step metric is cut with `metric_floor` and `metric_energy`, which discard its weakest directions. The examples on this page leave both at their defaults; change them only if you see an unstable or over-smoothed step, and then change one at a time.
  - `direct`: the closed form (M_D + ρ M_j)⁺, with ρ = λ tr(M_D)/tr(M_j). Here λ is the damping the loop supplies, the same one as γ in the update above, and ρ is not the term weight `alpha`. With constraints, M_D is replaced by M_eff (M_D plus the constraint terms) and the scale becomes γ = λ tr(M_eff)/tr(M_R), where M_R is the sum of the metric-term matrices. With an `absolute_trace` channel (see below), the sum of |eigenvalues| of M_eff replaces its trace.
- With **`[[parameterizations]]`**, non-identity terms see the simulator's properties (by their property names), in secants about P(m̄). So do constraints and the `constraint_cost` metric.

One identity `l2sq` term is `rlmmac` exactly, bit for bit, in the subspace form.

### Constraints (`constraints`)

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[method.strategy.options]
form = "direct"                                  # regularisation defaults to the identity (code "100")

[[method.strategy.options.constraints]]          # inequality: a box on PERMX
function = { name = "box", options = { bounds = { permx = [100.0, 15000.0] } } }
metric = { name = "log_barrier", options = { a = 0.1 } }
weight = { rule = "trace", w = 0.5 }

[[method.strategy.options.constraints]]          # equality: PERMX's histogram as the truth's
function = { name = "histogram", options = { variable = "permx", range = [100.0, 15000.0], bins = 50, reference_file = "../data/truth.npz" } }
metric = { name = "log_channel" }                # or { name = "quadratic" }, ...
weight = { rule = "trace", w = 0.5 }
```

Every entry is a table `{ function, metric, weight }`. `function` and `metric` are themselves tables `{ name, options }`. There is no string shorthand, because a `[policy]` and the web UI could not check one. Files named `*_file` are resolved against the configuration's folder.

#### Constraint functions (registry `gmac_constraints`)

| Function | Kind | Options | c(m) |
|---|---|---|---|
| `box` | inequality | `bounds = { variable = [lower, upper] }` | [lower − x; x − upper] per variable, in the order of `bounds` |
| `histogram` | equality | `variable`, `range`, `bins` (50), `reference_file` (`.npy`, or `.npz` with the variable's array) or `counts` | counts(x) − target counts, on fixed bins; the target must sum to the variable's size |
| `tv` | equality | `variables` | TV(x): first differences, target 0 |
| `linear` | per `kind` | `matrix_file` (`.npz` with `matrix`, optionally `rhs`), `kind` (`equality`, `inequality`), `variables` | A x − b |

#### Distance metrics (registry `gmac_constraint_metrics`)

Any metric can go with any function. The defaults are `log_barrier` for inequalities and `log_channel` for equalities. `metric` replaces them, e.g. the channel by the quadratic: `metric = { name = "quadratic" }`.

| Metric | D(x) | Gradient | Hessian | Options |
|---|---|---|---|---|
| `quadratic` | ½‖x‖² | x | I | none |
| `l1sq` | ‖x‖₁² | 2‖x‖₁ sgn(x) | 2 sgn(x) sgn(x)ᵀ (rank one) | none |
| `log_barrier` | −Σ log(x + a) | −1/(x + a) | diag(1/(x + a)²) | `a` (0.1) |
| `log_channel` | Σ log(\|x\| + b) | sgn(x)/(\|x\| + b) | see `curvature` | `b` (0.1), `epsilon` (0.001), `curvature` |

- **`log_channel`'s `curvature`:** the channel is concave on each side of 0, so its Hessian is negative.
  - `absolute_hessian` (the default): diag(1/(\|x\| + b)²) with a positive weight, so the step pulls toward the constraint.
  - `paper`: the paper's negative Hessian and negative trace weight, with their 1/(x + b sgn x + ε). This form pushes |c| away from 0.
  - `absolute_trace`: the negative Hessian with |tr| in the weight. The metric may be indefinite.

  On a 2-D facies test case, `absolute_hessian` with box + histogram constraints at w_eq = 1 was the only setting better than no constraints in data mismatch, RMSE and SSIM.
- **Pairings:** `quadratic` on an inequality also penalises the feasible side. `log_barrier` on an equality is undefined where c ≥ a.
- **A perfect measurement:** with a linear function, `quadratic` and a fixed weight w, a constraint is exactly an extra observation 0 of A x − b with error variance 1/w (a unit test checks it).

#### Weights

- `trace` (`w`): w_j = w tr(M_D)/tr(M_c,j), the paper's rule (its α_j and β_j). Its sign follows the metric: |tr| except for `log_channel` with `curvature = "paper"`.
- `fixed` (`value`): w_j = value.

Note the keys: terms use `alpha`, constraints `w`.

#### Localization

With constraints, only `adaptive_per_update` localizes. Each source (the data, then each constraint) gets its own correlation taper. `localize_constraints = false` tapers only the data and applies the constraints globally. This helps a box, whose pull toward its centre is the same for every member.

## Worked example: PERMX with a box and a histogram

The constraints example above is the worked case. It assimilates with `gies`, keeps the default identity regularisation, and adds two constraints on PERMX:

1. A `box` between 100.0 and 15000.0 with a `log_barrier` (`a` = 0.1). It is an inequality, so members are pushed back toward the box as they approach an edge.
2. A `histogram` of PERMX (50 bins over the same range), with the truth's histogram read from `reference_file`, and a `log_channel`. It is an equality that pulls the ensemble's value distribution toward the reference.

Both use `rule = "trace"` with `w = 0.5`, so each constraint counts by half the data metric whatever its scale.

!!! warning "The prior must lie inside the box"
    A `log_barrier` is undefined for a member that violates its inequality by `a` or more. If the prior has members outside the box (PERMX below 100.0 or above 15000.0 by `a` or more), the first update is refused before it is made, with the constraint, the members and the fixes. Give the PERMX entry of `[[state.variables]]` the bounds `lower = 100.0` and `upper = 15000.0`, so that a sampled prior is clipped into the box. A `file` prior is not clipped: store one whose members lie inside the box. See the *Infeasible members* pitfall below.

To see what the constraints do, add a `constraint_cost` metric. It reports each member's D(0 − c(m)) as `cost` and, for inequalities, the violated rows as `violations`, for any run. Give each instance a `label`:

```toml
[[metrics]]
name = "constraint_cost"
label = "histogram_channel"
options = { function = { name = "histogram", options = { variable = "permx", range = [100.0, 15000.0], bins = 50, reference_file = "../data/truth.npz" } }, metric = { name = "log_channel" } }
```

The `label` is how you tell instances apart in the results, so give the box its own metric entry with its own label. A falling `cost` over the iterations means the ensemble is moving toward the constraint. For the box, a non-zero `violations` means some members are still outside it.

The strategy's diagnostics (`constraint_cost`, `constraint_weight`, `constraint_violations`) record the same quantities per update. Request them in `[output] diagnostics` or in the strategy's `diagnostics` table.

## Pitfalls

- **Wrong method:** `es`, `esmda` and `gn_enrml` refuse `gmac`, and `lm_enrml` needs a positive damping minimum.
- **Infeasible members:** the barrier is undefined once a member violates an inequality by `a` or more. The update is then refused, with the constraint, the members, the worst violation, the causes and the fixes. The usual fix is to give the PERMX entry of `[[state.variables]]` the bounds `lower = 100.0` and `upper = 15000.0`, the same as the box (or slightly inside it). The loop then clips every member into the box. Clipped members violate by at most 0, so they stay well below `a`. Bounds wider than the box would not help.
- **A box barrier pulls every member toward the box's centre:** the `log_barrier` of a `box` sums a barrier from each bound, and their gradients cancel only at the centre. So the constraint moves members (and their mean) toward the centre, and narrows the spread, even when no member is near a bound. With a wide box this bias is small but not zero. If the box only guards physical limits, give the variable the bounds in `[[state.variables]]` (members are clipped, nothing is pulled) or a small weight `w`; to guard one side only, use a `linear` inequality on that side.
- **Wrong `log_channel` curvature:** only `absolute_hessian` pulls toward the constraint. `paper` pushes |c| away from 0, and `absolute_trace` may be indefinite.
- **Mismatched pairings:** `log_barrier` on an equality is undefined where c ≥ a. `quadratic` on an inequality also penalises the feasible side.
- **Histogram target:** the target counts must sum to the variable's size.
- **Mixed-up weight keys:** terms use `alpha`, constraints use `w`.
- **String options in plugins:** see the next section.

## Adding your own transform, norm, constraint function or metric

Each part is a plugin in a registry. A new one is a class registered in APEX (`@gmac_constraint_metrics.register("my_metric")`, etc.) or in another package, through the entry-point group `apex.<registry>`, e.g. `apex.gmac_constraint_metrics`. `gmac` needs no change.

| Part | Registry | What it implements |
|---|---|---|
| transform | `gmac_transforms` | `linear: bool`; `anomalies(state, layout, centre=None) -> (A, T)`, feature anomalies and Hessian points (n_f, N); optionally `check(layout)` |
| norm | `gmac_norms` | `constant: bool`; `metric(a, t)`, AᵀH(t_j)A: (N, N) if constant, else (N, N, N) |
| constraint function | `gmac_constraints` | `kind` (`"equality"` or `"inequality"`); `values(x, layout)`, c (n_c, k) of the columns of x (state or properties); `check(layout)` |
| constraint metric | `gmac_constraint_metrics` | `constant: bool`; `cost(x)` (k,), `gradient(x)` (n_c, k), `metric(a, x)` (N, N, N), with x = 0 − c |

`layout` maps each variable to its rows and grid (`gmac.parts.Block`). For a large constraint system, keep the Hessian diagonal (as the barrier and the channel do). `metric(a, x)` is then N products of n_c × N × N, and no n_c × n_c matrix is ever formed.

A minimal constraint metric, here the quadratic ½‖x‖² (Hessian I), looks like this. Imports of `gmac_constraint_metrics` are omitted; take them from the built-in metrics:

```python
import numpy as np


@gmac_constraint_metrics.register("my_quadratic")
class MyQuadratic:
    constant = True  # the Hessian does not depend on x

    def cost(self, x):  # x: (n_c, k) -> (k,)
        return 0.5 * np.sum(x**2, axis=0)

    def gradient(self, x):  # (n_c, k)
        return x

    def metric(self, a, x):  # a: (n_c, N) -> (N, N, N)
        k = x.shape[1]
        aha = a.T @ a  # aᵀ H a with H = I
        return np.repeat(aha[:, :, None], k, axis=2)
```

Compare the array layout of `metric` with the built-in `quadratic` before relying on it.

String options must be declared: add them to `FUNCTION_ROLES` or `METRIC_ROLES` in `gmac/constraints.py` (input files as `INPUT`, named `*_file`). Otherwise a `[policy]` and the web UI refuse them as undeclared. Numeric options need nothing.

A new plugin also needs its validation:

- a reference to compare against: an analytic case, a transcription (an independent re-implementation of the formula from the paper, such as the authors' MATLAB code), or finite differences of its gradient and Hessian;
- a test run on a reservoir case such as 5Spots, using a configuration like the ones above with your plugin in place of a built-in part.

How the built-in `gmac` was validated against the authors' implementation is described in [migration/gmac.md](migration/gmac.md).

## What to read next

- [migration/gmac.md](migration/gmac.md): differences from the MATLAB implementation and the validation.
- The pages for `gies` and `lm_enrml`, for their damping settings.
- The two papers above, for the theory behind the weights and metrics.
