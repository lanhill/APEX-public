---
audience: public
generated: reference
---

# Configuring service: reference

Every one of these 18 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `service:agents.diagnosis`

`[agents.diagnosis]`: the diagnosis agent's thresholds.

| option | type | default | description |
|---|---|---|---|
| `coverage_outside_fraction` | number | 0.2 | |
| `coverage_significance` | number | 0.01 | |
| `mahalanobis_limit` | number | 10.83 | |
| `overcorrection_percent` | number | 10.0 | |
| `recurring_failure_members` | integer | 3 | |

## `service:agents.llm`

`[agents.llm]`: the model runtime agents use.

Keys come from the environment variable named by `api_key_env`; they are never
written to a file or a log.

| option | type | default | description |
|---|---|---|---|
| `api_key_env` | string or null | | |
| `base_url` | string or null | | |
| `json_mode` | boolean or null | | |
| `max_context` | integer or null | | |
| `max_tokens` | integer | 2048 | |
| `model` | string or null | | |
| `options` | object | | Further provider options (e.g. Ollama keep_alive). |
| `provider` | string | openai_compatible | |
| `temperature` | number | 0.0 | |
| `timeout_s` | number | 120.0 | |
| `tool_calls` | boolean or null | | Override the provider's native tool calling (false: JSON). |

## `service:agents.monitor`

`[agents.monitor]`: the run monitor's thresholds.

| option | type | default | description |
|---|---|---|---|
| `auto` | boolean | false | |
| `collapse_spread_ratio` | number | 0.05 | |
| `cv_patience` | integer | 2 | |
| `max_consecutive_rejections` | integer | 5 | |
| `max_failed_fraction` | number | 0.2 | |
| `min_mismatch_reduction` | number | 0.1 | |
| `notify_on` | array | ["warning", "critical"] | |
| `optimization_hypervolume_drop` | number | 0.05 | |
| `optimization_infeasible_generations` | integer | 10 | |
| `optimization_min_front_generations` | integer | 10 | |
| `optimization_min_front_share` | number | 0.1 | |
| `optimization_stall_tolerance` | number | 0.001 | |
| `optimization_stall_window` | integer | 3 | |
| `overfit_mismatch` | number | 0.5 | |
| `poor_fit_mismatch` | number | 10000.0 | |
| `stagnation_tolerance` | number | 0.01 | |
| `stagnation_window` | integer | 3 | |

## `service:agents.planner`

`[agents.planner]`: the planning agent's bounds.

| option | type | default | description |
|---|---|---|---|
| `max_replicates` | integer | 5 | |
| `max_revisions` | integer | 2 | |
| `max_variants` | integer | 6 | |

## `service:agents.plotter`

`[agents.plotter]`: the plotter agent's bounds.

| option | type | default | description |
|---|---|---|---|
| `dpi_scale` | number | 1.0 | |
| `max_drawings` | integer | 2 | |
| `max_figures` | integer | 40 | |

## `service:agents.visibility`


## `service:approval`

`[approval]`: when a person must approve a submission.

| option | type | default | description |
|---|---|---|---|
| `max_runs` | integer | 20 | |
| `max_simulator_runs` | integer | 20000 | |

## `service:chat`

`[chat]`: the chat assistant of the web UI.

| option | type | default | description |
|---|---|---|---|
| `context_turns` | integer | 20 | |
| `isolate_tools` | boolean | true | |
| `llm` | LLMSection or null | | |
| `max_history_mb` | number | 200.0 | |
| `max_model_calls` | integer | 10 | |
| `max_turns_per_chat` | integer | 500 | |
| `refusal_turns` | integer | 4 | |
| `retention_days` | number | 92.0 | |
| `stream` | boolean | true | |
| `tool_timeout_s` | number | 120.0 | |

**`LLMSection`**: `[agents.llm]`: the model runtime agents use.

Keys come from the environment variable named by `api_key_env`; they are never
written to a file or a log.

| option | type | default | description |
|---|---|---|---|
| `api_key_env` | string or null | | |
| `base_url` | string or null | | |
| `json_mode` | boolean or null | | |
| `max_context` | integer or null | | |
| `max_tokens` | integer | 2048 | |
| `model` | string or null | | |
| `options` | object | | Further provider options (e.g. Ollama keep_alive). |
| `provider` | string | openai_compatible | |
| `temperature` | number | 0.0 | |
| `timeout_s` | number | 120.0 | |
| `tool_calls` | boolean or null | | Override the provider's native tool calling (false: JSON). |

## `service:examples`

`[examples]`: the catalogue of examples and toy problems (`services/examples.py`).

| option | type | default | description |
|---|---|---|---|
| `enabled` | boolean | true | Offer the catalogue (the Examples page). |
| `root` | string or null | | Folder holding the examples (each with an example.toml); default: the repository's examples/ when APEX runs from a checkout. Its files are trusted as the server's: a copied example's deck templates are read from here. |

## `service:guests`

`[guests]`: how long, and how much, the web UI's guests' work is kept.

Applies only to work a guest submitted (`submitted_by = guest:<name>`); the
owner's, and other local programs' (the CLI, the Python client), is never pruned.

| option | type | default | description |
|---|---|---|---|
| `max_storage_mb` | number | 500.0 | |
| `retention_days` | number | 30.0 | |

## `service:jobs`

`[jobs]`: which code jobs run.

| option | type | default | description |
|---|---|---|---|
| `cli_code` | string | frozen | Code of jobs submitted from the command line (apex submit, apex study run): 'frozen' pins each to the repository's HEAD at submission, 'live' runs the checkout as it is when the worker starts ($APEX_CLI_CODE overrides it). Other clients run live code unless they ask. |
| `code_dir` | string or null | | Where frozen copies live; default <root>/code. |
| `code_extras` | array or null | | Extras synced into each frozen copy's environment; default: those installed in the environment APEX runs in (torch never). |
| `keep_code_days` | number | 2.0 | apex gc --code (and the service, daily) removes frozen copies that no job or process uses and that were last pinned longer ago than this; never the newest copy of master. |
| `repository` | string or null | | The git repository frozen copies are made from; default: the checkout APEX runs from. |

## `service:notifications`

`[notifications]`: optional calls when a job ends.

| option | type | default | description |
|---|---|---|---|
| `command` | string or null | | |
| `webhook` | string or null | | |

## `service:policy`

`[policy]`: roots a submission may read, execute and write.

Off by default (all lists empty); once one list is set, all three must be. Roots are
absolute paths, optionally with glob components (`/cases/*/runs/planner`).

| option | type | default | description |
|---|---|---|---|
| `code_roots` | array | | |
| `executables` | array | | Program names (found on the service's PATH) allowed under code_roots. |
| `hosts` | array | | Other machines that members may run on; none by default. |
| `input_roots` | array | | |
| `output_roots` | array | | |

## `service:quotas`


## `service:service`

`[service]`.

| option | type | default | description |
|---|---|---|---|
| `max_concurrent_jobs` | integer | 2 | |
| `max_retries` | integer | 2 | |
| `poll_s` | number | 1.0 | |
| `routines` | boolean | true | |

## `service:tools`

`[tools]`: what tool clients (MCP, `apex tools call`, agent loops) may read.

Without a `[policy]`, a tool client names files only under the workspace's roots
(`[workspace] roots`, by default `<root>/configs`), these `roots` and the
service's `runs/`, `studies/` and `optimizations/`. With a `[policy]`, its
readable roots decide instead.

| option | type | default | description |
|---|---|---|---|
| `roots` | array | | More folders whose files tool clients may name (absolute; no [policy]). |

## `service:ui`

`[ui]`: the built web UI the service serves at `/ui/`.

| option | type | default | description |
|---|---|---|---|
| `contributed_pages` | boolean | true | |
| `dist_dir` | string or null | | |
| `token_days` | number | 7.0 | |

## `service:workspace`

`[workspace]`: folders whose `.toml` files the UI's editor may write.

Empty means `<root>/configs`, except under a `[policy]`, where the editor is off
until roots are set (each under the policy's `input_roots`).

| option | type | default | description |
|---|---|---|---|
| `roots` | array | | |
| `ui_folder` | string or null | UI | Folder of the first root where the web UI's submissions write their runs, studies and drafts: UI/owner/ and UI/guests/<name>/; "" turns it off (their configurations say where). |
