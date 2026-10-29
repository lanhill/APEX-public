---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - config:history_match:method
  - plugin:gmac_constraint_metrics:l1sq
  - plugin:gmac_constraints:tv
  - plugin:gmac_norms:l1sq
  - plugin:gmac_norms:l2sq
  - plugin:gmac_transforms:histogram_entropy
  - plugin:gmac_transforms:identity
  - plugin:gmac_transforms:state
  - plugin:gmac_transforms:tv
  - plugin:methods:gies
  - plugin:update_strategies:gmac
---

# Gmac transforms

The `gmac_transforms` registry holds four plugins: `state`, `identity`, `tv` and `histogram_entropy`. Each one turns the ensemble state into a different quantity. The `gmac` update strategy uses that quantity to build its step metric.

## Goal

By the end of this page you will know what each of the four transforms does, which options each accepts, and how to put one into a `gmac` term in a configuration.

## Prerequisites

- A working APEX installation and a TOML run configuration to edit.
- The `gmac` strategy used with the `gies` or `lm_enrml` method. Other methods do not support it; see [GMAC and C-GIES](../../gmac.md).
- The names of the state variables you want to transform. The example below uses `PORO` and `PERMX` as placeholders. Replace them with the variables in your own state.
- For `tv`: variables that live on a grid.

## The four transforms

| Plugin | What it produces | Options |
|---|---|---|
| `state` | The raw state, or some of its variables. This is the l₁² term on the state `m` in the MATLAB routine `RLM_MAC_L1`. | `variables` |
| `identity` | The "I term" of the GMAC metric: the ensemble-whitened state, scaled so that `S_Sᵀ S_S = I_N`. Here `S_S` is the transformed ensemble matrix and `I_N` is the identity matrix with one row and column per ensemble member. | none |
| `tv` | First differences along each grid axis, per variable, stacked (called `S_V`; it follows the MATLAB routine `getTV2D`). | `variables`, `scale_per_variable` |
| `histogram_entropy` | A histogram's count-weighted entropy per bin, per variable (called `S_H`; it follows the MATLAB routine `getHistEntropy`). | `variables`, `bins`, `ranges` |

Every plugin rejects options it does not define. `identity` takes no options at all, so giving it one is an error.

## Steps

1. Decide which quantity you need: raw (`state`), whitened (`identity`), spatial differences (`tv`) or histogram entropy (`histogram_entropy`).
2. In the `gmac` strategy's options, add an entry to `terms` (a `[[method.strategy.options.terms]]` table). Each term has three parts:
   - `transform`: `{ name = "<plugin>", options = { ... } }`, which is the transform you chose.
   - `norm`: `{ name = "<norm>" }`, a plugin of the `gmac_norms` registry. `l2sq` is the squared 2-norm, whose Hessian is `2I`. `l1sq` is the squared 1-norm, whose Hessian is `2 sgn(x) sgn(x)ᵀ`.
   - `weight`: `{ rule = "trace", alpha = ... }` or `{ rule = "fixed", value = ... }`.
3. Set only the transform options you need. Anything you leave out keeps its default.
4. To restrict a transform to some variables, list them in `variables`. When `variables` is omitted (default `null`), `state` and `histogram_entropy` use all state variables, and `tv` uses every variable that has a grid.
5. To check the options your installation accepts, list the plugins:

```bash
uv run apex plugins
```

Add `--json` for the full descriptions as JSON.

### Option reference

`state`

- `variables`: list of strings or `null`. Default `null` (all state variables).

`identity`

- No options.

`tv`

- `variables`: list of strings or `null`. Default `null` (every variable with a grid).
- `scale_per_variable`: boolean. Default `false`. When `true`, each variable's differences are divided by their ensemble RMS, so no variable dominates because of its units. When `false`, the variables are stacked unscaled.

`histogram_entropy`

- `variables`: list of strings or `null`. Default `null` (all state variables).
- `bins`: integer. Default `100`. The number of bins per variable.
- `ranges`: a table that maps a variable name to `[low, high]`, or `null`. Default `null`. With `null`, the range at each update is the one spanned by the ensemble and its mean. With a fixed range, values outside it fall in the end bins. Either way the bins are the same for every member and for the mean, so the differences compare like with like.

## Worked example

This configuration uses all four transforms in one `gmac` strategy. `identity` takes no options. `state` and `tv` are restricted to two variables, and `tv` scales each variable. `histogram_entropy` uses 50 bins and a fixed range per variable. Every term has a `norm` and a `weight`.

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.terms]]
transform = { name = "identity" }
norm = { name = "l2sq" }
weight = { rule = "trace", alpha = 1.0 }

[[method.strategy.options.terms]]
transform = { name = "state", options = { variables = ["PORO", "PERMX"] } }
norm = { name = "l1sq" }
weight = { rule = "fixed", value = 0.5 }

[[method.strategy.options.terms]]
transform = { name = "tv", options = { variables = ["PORO", "PERMX"], scale_per_variable = true } }
norm = { name = "l2sq" }
weight = { rule = "fixed", value = 0.5 }

[[method.strategy.options.terms]]
transform = { name = "histogram_entropy", options = { variables = ["PORO", "PERMX"], bins = 50, ranges = { PORO = [0.1, 0.4], PERMX = [6.0, 8.0] } } }
norm = { name = "l2sq" }
weight = { rule = "fixed", value = 0.5 }
```

This is the method and strategy part of a configuration. A full run also needs the rest of your setup, which this page does not cover. Start the run as usual:

```bash
uv run apex run config.toml
```

How to read the example:

- `identity` returns the whitened ensemble state.
- `state` passes the two named variables through unchanged.
- `tv` takes first differences along each grid axis for each variable and stacks them. With `scale_per_variable = true`, each variable's differences are divided by their ensemble RMS.
- `histogram_entropy` builds a 50-bin histogram per variable over the given range and returns the count-weighted entropy per bin.

## Pitfalls

- **Passing options to `identity`.** It defines none, so any option you add is rejected.
- **Misspelled option names.** Unknown options are rejected rather than ignored, so a typo such as `bin` instead of `bins` raises an error.
- **Wrong types.** `variables` must be a list of strings, `bins` an integer and `scale_per_variable` a boolean. Write `bins = 50`, not `bins = "50"`.
- **`ranges` that do not match `variables`.** Use the names of the variables the transform works on as keys. If you set `ranges`, give one for each variable you list.
- **Using `tv` on a variable without a grid.** `tv` works on gridded variables only.
- **Reading `tv` as one value per cell.** It returns first differences along every grid axis, stacked, so its output is larger than a single field.
- **Forgetting the method.** `gmac` works with `gies` or `lm_enrml`; methods such as `es`, `esmda` and `gn_enrml` do not support it.

## What to read next

- [GMAC and C-GIES](../../gmac.md), for the terms' norms and weights, the `code` shorthand and the constraints.
- The other plugin pages in the guide, to see how transforms fit into a complete run.
