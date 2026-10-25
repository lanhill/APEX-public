---
audience: public
generated: reference
---

# Configuring study: reference

Every one of these 1 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `study:study`

`[study]`.

| option | type | default | description |
|---|---|---|---|
| `annotations` | object | | |
| `budget` | | | |
| `cases` | object | required | Name -> history-matching configuration. |
| `methods` | object | | Name -> algorithm, each complete; none: each case's own. |
| `metrics` | array | | 'metric:label' to compare at each run's posterior. An unknown metric, or a label none of the cases has (a state variable or data type; not checked where a case's labels are known only when it runs), is refused at validation. A metric a cell does not record (e.g. rmse in a real case) is left out of its comparison. |
| `name` | string | required | |
| `predictions` | array | | Prediction configurations without run_dir, run after each run. |
| `replicates` | integer | 1 | Seeds per case and variant. |
| `seed` | integer | 0 | Seeds of replicates 1, 2, ... derive from it. |
| `simulator` | string or null | | Run every case's deck on this simulator (opm_flow or eclipse): it replaces the case's [simulator] name and keeps the options both share. Unset: each case's own. |
| `variants` | array | required | |

**`BudgetSection`**: `[study] budget`: refused before anything runs when the plan exceeds it.

| option | type | default | description |
|---|---|---|---|
| `max_simulator_runs` | integer or null | | |

**`MethodEntry`**: `[study.methods.<name>]`: one algorithm, complete.

Its fields are the algorithm's fields of `[method]` (`name`, `options`,
`diagnostics`, `strategy`, `schedule`, `max_iterations`); those left out take
`[method]`'s defaults, not the case's values.

| option | type | default | description |
|---|---|---|---|
| `diagnostics` | object or null | | |
| `engine` | string | apex | Where it runs: 'apex', or another engine such as 'ert'. |
| `max_iterations` | integer or null | | |
| `name` | string | required | The method plugin (esmda, gies, ...), or the engine's algorithm (ERT's es_mda or ensemble_smoother). |
| `options` | object | | |
| `schedule` | object or null | | |
| `strategy` | PluginSection or null | | |

**`VariantSection`**: `[[study.variants]]`: one change to every case.

| option | type | default | description |
|---|---|---|---|
| `name` | string | required | |
| `patch` | object | | JSON merge patch (RFC 7386) over each case's configuration. |
