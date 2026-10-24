---
audience: public
items:
  - plugin:gmac_norms:l1sq
  - plugin:gmac_norms:l2sq
  - plugin:gmac_norms:lpq
---

# Gmac norms

The `gmac_norms` registry holds the norm plugins you can choose from. Each one is a squared or powered norm together with its Hessian. This page explains what the three plugins, `l1sq`, `l2sq` and `lpq`, compute, how they differ, and how to choose between them.

## Goal

By the end of this page you will be able to:

- say what each of `l1sq`, `l2sq` and `lpq` computes;
- explain how its Hessian shapes the metric `M_j` or `2AᵀA`;
- pick the right plugin and set the `p`, `q` and `floor` options of `lpq`.

## Prerequisites

- Basic familiarity with norms and Hessians (the matrix of second derivatives).
- Some knowledge of how APEX plugins are selected in a run. This page covers only what the `gmac_norms` plugins are and what their options mean.

## The plugins at a glance

| Plugin | Norm | Hessian | Options |
|---|---|---|---|
| `l1sq` | ‖x‖₁² | 2 sgn(x) sgn(x)ᵀ (rank one) | none |
| `l2sq` | ‖x‖₂² | 2I | none |
| `lpq` | ‖x‖_p^q | exact Hessian (manuscript, App. C) | `p`, `q`, `floor` |

## Steps

### 1. Decide which norm you need

- **`l2sq`** is the squared Euclidean norm ‖x‖₂². Its Hessian is the constant `2I`, so the metric `2AᵀA` is the same for every member.
- **`l1sq`** is the squared one-norm ‖x‖₁². Its Hessian is `2 sgn(x) sgn(x)ᵀ`, which has rank one. The metric is therefore `M_j = 2 v_j v_jᵀ`, and it differs from member to member.
- **`lpq`** is the general form ‖x‖_p^q. It uses its exact Hessian, as derived in App. C of the manuscript.

### 2. Leave `l1sq` and `l2sq` as they are

Neither `l1sq` nor `l2sq` has any options. There is nothing to set.

### 3. Set the options of `lpq` if you use it

| Option | Type | Default |
|---|---|---|
| `p` | number | `2.0` |
| `q` | number | `2.0` |
| `floor` | number | `1e-06` |

With the defaults, `p = 2` and `q = 2`, so `lpq` gives ‖x‖₂², the same norm as `l2sq`. Change `p` to change the norm used inside, and `q` to change the power it is raised to. `floor` is a small positive number with default `1e-06`.

## Worked example

Take a vector x = (3, −2).

**With `l2sq`:** the norm is 3² + (−2)² = 13. The Hessian is 2I:

```
[ 2  0 ]
[ 0  2 ]
```

It does not depend on x, which is why the metric is shared by all members.

**With `l1sq`:** the norm is (|3| + |−2|)² = 25. The sign vector is sgn(x) = (1, −1), so the Hessian 2 sgn(x) sgn(x)ᵀ is:

```
[  2  -2 ]
[ -2   2 ]
```

The second row is −1 times the first, so the matrix has rank one. Because the sign vector changes with x, each member j gets its own `M_j = 2 v_j v_jᵀ`.

**With `lpq` at its defaults:** `p = 2.0`, `q = 2.0` and `floor = 1e-06` give ‖x‖₂², the same norm as `l2sq`.

## Pitfalls

- **Rank one means a single direction.** The `l1sq` metric `M_j = 2 v_j v_jᵀ` constrains only one direction per member. Do not expect it to behave like the full-rank `2I` of `l2sq`.
- **`l2sq` is shared.** Its metric `2AᵀA` is identical for every member. If you need member-specific metrics, `l2sq` will not give them.
- **No options on `l1sq` and `l2sq`.** Only `lpq` accepts `p`, `q` and `floor`. Supplying options to the other two is an error, because they accept no options at all.
- **Defaults make `lpq` equal to `l2sq`.** If you pick `lpq` and leave `p` and `q` alone, you get the squared Euclidean norm.

## What to read next

- The other plugin pages in the guide, for the registries that use these norms.
- App. C of the manuscript, for the exact Hessian of ‖x‖_p^q used by `lpq`.
