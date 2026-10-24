---
audience: public
items:
  - plugin:gmac_transforms:histogram_entropy
  - plugin:gmac_transforms:identity
  - plugin:gmac_transforms:state
  - plugin:gmac_transforms:tv
---

# Gmac transforms

The `gmac_transforms` registry holds four plugins. Each one turns an ensemble state into a different representation, so you can choose which quantity a GMAC-style analysis works on.

## Goal

By the end of this page you will know what each of the four transforms does: `state`, `identity`, `tv` and `histogram_entropy`. You will also know which options each one accepts and how to choose it in a configuration.

## Prerequisites

- A working APEX setup and a TOML configuration file to edit.
- The names of the state variables you want to transform. The examples below use `pressure` and `saturation` as placeholder names. Replace them with the variables in your own state.

## The four transforms

| Plugin | What it produces | Options |
|---|---|---|
| `state` | The raw state, or some of its variables. This is the l₁² term on `m` from MATLAB `RLM_MAC_L1`. | `variables` |
| `identity` | The paper's I term: the ensemble-whitened state, so that `S_Sᵀ S_S = I_N`. | none |
| `tv` | First differences along each grid axis, per variable, stacked (`S_V`, `getTV2D`). | `variables`, `scale_per_variable` |
| `histogram_entropy` | A histogram's count-weighted entropy per bin, per variable (`S_H`, `getHistEntropy`). | `variables`, `bins`, `ranges` |

Every plugin rejects options it does not define. For example, `identity` takes no options at all, and giving it one is an error.

## Steps

1. Decide which representation you need: raw (`state`), whitened (`identity`), spatial differences (`tv`) or histogram entropy (`histogram_entropy`).
2. Add a term to the `gmac` strategy's `terms`, an `[[method.strategy.options.terms]]` entry, with `transform = "<plugin>"`, or `transform = { name = "<plugin>", options = { ... } }` to set options.
3. Set only the options you need. Anything you leave out keeps its default.
4. If a transform should act on only some variables, list them in `variables`. The default is `null`, meaning no explicit selection.

### Option reference

`state`

- `variables`: list of strings or `null`. Default `null`.

`identity`

- No options.

`tv`

- `variables`: list of strings or `null`. Default `null`.
- `scale_per_variable`: boolean. Default `false`.

`histogram_entropy`

- `variables`: list of strings or `null`. Default `null`.
- `bins`: integer. Default `100`.
- `ranges`: a table that maps a variable name to a list of numbers, or `null`. Default `null`.

## Worked example

A transform is chosen per metric term, as the `transform` of an entry of the `gmac` strategy's `terms` (see [GMAC and C-GIES](../../gmac.md)), with a `norm` and a `weight`. This configuration uses all four: `identity` (no options, so its name alone), `state` and `tv` restricted to two variables with per-variable scaling for `tv`, and `histogram_entropy` with 50 bins and an explicit range per variable.

```toml
[method]
name = "gies"

[method.strategy]
name = "gmac"

[[method.strategy.options.terms]]
transform = "identity"
norm = "l2sq"

[[method.strategy.options.terms]]
transform = { name = "state", options = { variables = ["pressure", "saturation"] } }
norm = "l1sq"

[[method.strategy.options.terms]]
transform = { name = "tv", options = { variables = ["pressure", "saturation"], scale_per_variable = true } }
norm = "l2sq"

[[method.strategy.options.terms]]
transform = { name = "histogram_entropy", options = { variables = ["pressure", "saturation"], bins = 50, ranges = { pressure = [1.0, 5.0], saturation = [0.0, 1.0] } } }
norm = "l2sq"
```

How to read it:

- `state` passes the two named variables through unchanged.
- `identity` returns the whitened ensemble state.
- `tv` computes first differences along each grid axis for each variable and stacks them. With `scale_per_variable = true`, each variable is scaled individually.
- `histogram_entropy` builds a 50-bin histogram per variable over the given range and returns the count-weighted entropy per bin.

## Pitfalls

- **Passing options to `identity`.** It defines none, so any option you add is rejected.
- **Misspelled option names.** Unknown options are rejected rather than ignored, so a typo such as `bin` instead of `bins` raises an error.
- **Wrong types.** `variables` must be a list of strings, `bins` an integer and `scale_per_variable` a boolean. Write `bins = 50`, not `bins = "50"`.
- **`ranges` keys that do not match `variables`.** Each key in `ranges` should be the name of a variable that the transform actually uses.
- **Reading `tv` as one value per cell.** It returns first differences along every grid axis, stacked, so its output is larger than a single field.

## What to read next

- The other plugin pages in the guide, to see how transforms fit into a complete run.
- [GMAC and C-GIES](../../gmac.md), for the terms' norms and weights.
