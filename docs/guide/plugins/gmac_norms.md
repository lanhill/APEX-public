---
audience: public
items:
  - cli:apex plugins
  - cli:apex run
  - plugin:gmac_constraint_metrics:l1sq
  - plugin:gmac_norms:l1sq
  - plugin:gmac_norms:l2sq
  - plugin:gmac_norms:lpq
---

# Gmac norms

The `gmac_norms` registry holds the norm plugins you can choose from: `l1sq`, `l2sq` and `lpq`. Each one is a squared or powered norm together with its Hessian (the matrix of second derivatives). This page explains what each computes, how they differ, and how to set the options of `lpq`.

## Goal

By the end of this page you will be able to:

- say what `l1sq`, `l2sq` and `lpq` compute;
- explain how the Hessian of each gives the per-member metric `M_j` or the shared metric `2AᵀA`;
- choose a plugin and set the `p`, `q` and `floor` options of `lpq`.

## Prerequisites

- Basic familiarity with norms and Hessians.
- A run configuration you can pass to `uv run apex run <config>`. This page covers what the `gmac_norms` plugins compute and what their options mean. It does not cover the configuration section that selects a plugin, so use the output of `uv run apex plugins` (see step 4) to check names and options.

## Terms used on this page

- **Member**: one of the members (indexed by `j`) of the ensemble in a history-matching run. Each member has its own vector `x`.
- **`v_j`**: the sign vector `sgn(x)` of member `j`. Each entry is +1, −1 or 0.
- **`M_j`**: the metric of member `j`. For `l1sq` it is the Hessian, `M_j = 2 v_j v_jᵀ`.
- **`A`**: the matrix inside the norm. For `l2sq` the metric is `2AᵀA`, which does not depend on the member.

## The plugins at a glance

| Plugin | Norm | Hessian | Options |
|---|---|---|---|
| `l1sq` | ‖x‖₁² | 2 sgn(x) sgn(x)ᵀ (rank one) | none |
| `l2sq` | ‖x‖₂² | 2I | none |
| `lpq` | ‖x‖_p^q | exact Hessian (manuscript, App. C) | `p`, `q`, `floor` |

A plugin named `l1sq` also exists in a different registry, `gmac_constraint_metrics`. It computes `D(x) = ||x||_1^2` and also takes no options. It is a separate plugin; this page is about the one in `gmac_norms`.

## Steps

### 1. Decide which norm you need

- **`l2sq`** is the squared Euclidean norm ‖x‖₂². Its Hessian is the constant `2I`, so the metric `2AᵀA` is the same for every member.
- **`l1sq`** is the squared one-norm ‖x‖₁². Its Hessian is `2 sgn(x) sgn(x)ᵀ`, which has rank one. The metric is `M_j = 2 v_j v_jᵀ`, and it differs from member to member whenever the sign patterns differ.
- **`lpq`** is the general form ‖x‖_p^q. It uses its exact Hessian, as derived in App. C of the manuscript.

### 2. Leave `l1sq` and `l2sq` as they are

Neither `l1sq` nor `l2sq` has any options. There is nothing to set.

### 3. Set the options of `lpq`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `p` | number | `2.0` | The norm used inside. At least 1. |
| `q` | number | `2.0` | The power the norm is raised to. At least 1. |
| `floor` | number | `1e-06` | Guards the Hessian when `1 < p < 2` (see below). |

`p = q = 1` (‖x‖₁) is refused: its Hessian is 0 almost everywhere, so it adds nothing.

**What `floor` does.** For `1 < p < 2`, the Hessian contains `|x|^{p-2}`, which is infinite where an entry of `x` is 0. To avoid this, `lpq` evaluates it at `max(|x|, floor · max|x|)`, where `max|x|` is the largest absolute entry of `x`. So an entry smaller than `floor` times the largest entry is treated as if it were that size. `floor` has no role for other values of `p`. Raise it if tiny entries make the Hessian huge. Lower it to stay closer to the exact Hessian. Most users can leave it at `1e-06`.

**A short rule for choosing `p` and `q`:**

- `p = 2`, `q = 2` (the defaults) is the squared Euclidean norm, the same as `l2sq`. Use `l2sq` directly if that is what you want.
- `p = 1`, `q = 2` gives the squared one-norm value, the same as `l1sq`. Use `l1sq` directly if that is what you want.
- Other combinations (for example `p = 3`, `q = 2`) are what `lpq` is for. Keep both at 1 or above, and not both equal to 1.

### 4. Check what is installed

List every plugin with its options:

```
uv run apex plugins
```

For full descriptions as JSON:

```
uv run apex plugins --json
```

When you are ready, start a run with your configuration file:

```
uv run apex run config.toml
```

`apex run` also accepts `--max-steps` to stop after a number of analysis steps, and `--progress` or `--no-progress` to turn progress display on or off.

## Worked example

Take two members with `x = (3, −2)` and `x′ = (1, 4)`.

**With `l2sq`:** for `x` the norm is 3² + (−2)² = 13. The Hessian is `2I`, and it is the same for `x′`:

```
[ 2  0 ]
[ 0  2 ]
```

For a small matrix `A = [[1, 0], [1, 1]]` we get `AᵀA = [[2, 1], [1, 1]]`, so the metric is `2AᵀA = [[4, 2], [2, 2]]`. It is identical for both members, because `x` does not enter it.

**With `l1sq`:** for `x` the norm is (|3| + |−2|)² = 25. The sign vector is `v₁ = (1, −1)`, so `M₁ = 2 v₁ v₁ᵀ` is:

```
[  2  -2 ]
[ -2   2 ]
```

For `x′`, the sign vector is `v₂ = (1, 1)`, so `M₂ = 2 v₂ v₂ᵀ` is:

```
[ 2  2 ]
[ 2  2 ]
```

Both matrices have rank one (the second row is ±1 times the first), but they point in different directions. That is how the metric differs from member to member.

**With `lpq` at non-default values**, still for `x = (3, −2)`:

- `p = 1`, `q = 2`: ‖x‖₁² = (3 + 2)² = 25, the same value as `l1sq`.
- `p = 3`, `q = 2`: ‖x‖₃ = (27 + 8)^(1/3) = 35^(1/3) ≈ 3.271, so ‖x‖₃² ≈ 10.70.
- `p = 2`, `q = 1`: ‖x‖₂ = √13 ≈ 3.606.

**Where `floor` matters.** Take `p = 1.5` and `x = (3, 0)`. The Hessian needs `|x|^{-0.5}`, which is infinite for the entry 0. With the default `floor = 1e-06`, that entry is evaluated at `max(0, 1e-06 · 3) = 3e-06` instead, giving a large but finite value.

## Pitfalls

- **Rank one means a single direction.** The `l1sq` metric `M_j = 2 v_j v_jᵀ` constrains only one direction per member. Do not expect it to behave like the full-rank `2I` of `l2sq`.
- **`l2sq` is shared.** Its metric `2AᵀA` is identical for every member. If you need member-specific metrics, `l2sq` will not give them.
- **No options on `l1sq` and `l2sq`.** Only `lpq` accepts `p`, `q` and `floor`. Supplying options to the other two is an error, because they accept none.
- **`p = q = 1` is refused.** The Hessian of ‖x‖₁ is 0 almost everywhere, so `lpq` rejects it. Use `p = 1`, `q = 2` or another valid pair.
- **`p` and `q` below 1 are not allowed.** Both must be at least 1.
- **Defaults make `lpq` equal to `l2sq`.** If you pick `lpq` and leave `p` and `q` alone, you get the squared Euclidean norm.
- **Do not confuse the two `l1sq` plugins.** One is in `gmac_norms`, the other in `gmac_constraint_metrics`. Check the registry name in the `uv run apex plugins` output.

## What to read next

- The output of `uv run apex plugins`, which lists every plugin, including those of other registries that work with these norms, together with their options.
- App. C of the manuscript, for the exact Hessian of ‖x‖_p^q used by `lpq`.
