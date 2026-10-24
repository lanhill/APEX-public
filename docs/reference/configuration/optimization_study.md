---
audience: public
generated: reference
---

# Configuring optimization study: reference

Every one of these 1 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `optimization_study:study`

`[study]` with `kind = "optimization"`.

| option | type | default | description |
|---|---|---|---|
| `annotations` | object | | |
| `budget` | | | |
| `cases` | object | required | Name -> optimization configuration. |
| `kind` | string | required | |
| `methods` | object | | Name -> a complete [optimizer] section; none: each case's own. |
| `metrics` | array | | What to compare: objective:final, objective:best, evaluations:all, iterations:all; for several objectives also hypervolume:final and front_size:final. |
| `name` | string | required | |
| `reference_point` | array or null | | The hypervolume's reference point for every cell (several objectives); default: the nadir of all cells' fronts plus 10 % of each objective's range. |
| `replicates` | integer | 1 | Seeds per case, method and variant. |
| `seed` | integer | 0 | Seeds of replicates 1, 2, ... derive from it. |
| `variants` | array | required | |

**`BudgetSection`**: `[study] budget`: refused before anything runs when the plan exceeds it.

| option | type | default | description |
|---|---|---|---|
| `max_simulator_runs` | integer or null | | |

**`OptimizationVariant`**: `[[study.variants]]` of an optimization study: one change to every case.

| option | type | default | description |
|---|---|---|---|
| `name` | string | required | |
| `patch` | object | | JSON merge patch (RFC 7386) over each case's optimization configuration. |

**`PluginSection`**: A registry name plus keyword options for its factory.

`diagnostics`, for plugins that produce diagnostics (the method, its
strategy and localization), switches this plugin's items on or off and **overrides**
`[output] diagnostics` for it: `{ taper = true }` saves the taper even if
`[output]` does not list it, `{ taper = false }` suppresses it even if it does.
`max_mb` and `every` override `[output] diagnostics_max_mb` and
`diagnostics_every` for this plugin. The run logs the resulting plan and warns when
a plugin switch contradicts `[output]`.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | Per-plugin diagnostics switches (name = true/false, max_mb, every); they override [output] diagnostics for this plugin. |
| `name` | string | required | |
| `options` | object | | |
