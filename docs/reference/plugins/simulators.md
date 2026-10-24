---
audience: public
generated: reference
---

# Simulators: reference

Every one of these 5 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `eclipse`

Runs Eclipse (E100 or E300) through `eclrun` for one member in its work directory.

Select it with:

```toml
[simulator]
name = "eclipse"
```

| option | type | default | description |
|---|---|---|---|
| `case_dir` | string or string or null | | |
| `deck_template_file` | string or string | required | |
| `eclrun_args` | array | | |
| `executable` | string | eclrun | |
| `keep_files` | boolean | true | |
| `levels` | array or null | | |
| `licenses` | integer or null | | |
| `template_values` | object or null | | |
| `timeout_s` | number | 3600.0 | |
| `transient_patterns` | array | | |
| `version` | string | e100 | |

## `ert`

An ERT case's forward model, run by ERT through the `ert` executor.

Select it with:

```toml
[simulator]
name = "ert"
```

| option | type | default | description |
|---|---|---|---|
| `case_file` | string or string | required | |
| `eclbase` | string | required | |
| `fields` | object | required | |
| `parameter_step` | string or null | | |
| `responses` | string | summary | |
| `results_program` | array | | |
| `skip_forward_steps` | integer | 0 | |
| `step_parameters` | object or null | | |

## `linear`

`y = f(G x)` for a fixed matrix `G`, with `f` the identity or `tanh`.

Select it with:

```toml
[simulator]
name = "linear"
```

| option | type | default | description |
|---|---|---|---|
| `delay_s` | number | 0.0 | |
| `fail_above` | number or null | | |
| `fail_members` | array | | |
| `fields` | array | | |
| `flaky_members` | object or null | | |
| `forward_file` | string or string or null | | |
| `levels` | array or null | | |
| `matrix` | ArrayLike or null | | |
| `nonlinearity` | string | none | |
| `outputs` | array | required | |
| `resources` | object or null | | |
| `variables` | array | required | |
| `write_files` | boolean | false | |

## `opm_flow`

Runs OPM Flow for one member in its work directory.

Select it with:

```toml
[simulator]
name = "opm_flow"
```

| option | type | default | description |
|---|---|---|---|
| `case_dir` | string or string or null | | |
| `deck_template_file` | string or string | required | |
| `executable` | string | flow | |
| `flow_args` | array | | |
| `keep_files` | boolean | true | |
| `levels` | array or null | | |
| `template_values` | object or null | | |
| `timeout_s` | number | 3600.0 | |
| `transient_patterns` | array | | |

## `toy_reservoir`

Field totals of a toy waterflood, for optimization tests and demos (/0043).

Select it with:

```toml
[simulator]
name = "toy_reservoir"
```

| option | type | default | description |
|---|---|---|---|
| `control` | string | inj | |
| `oil_rate` | number | 20.0 | |
| `start` | string | 2000-01-01 | |
| `water_rate` | number | 1.0 | |
