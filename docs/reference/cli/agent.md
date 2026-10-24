---
audience: public
generated: reference
---

# The apex agent commands: reference

Every one of these 8 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex agent ask`

Ask APEX's agent a question; it may use APEX's tools to answer.

```bash
apex agent ask QUESTION
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |
| `--tools` (on) / `--no-tools` (off) | true | Offer APEX's tools. |
| `--max-turns` | 8 | Model calls at most. |
| `--client` | agent | Client name for quotas and the call log. |

## `apex agent diagnose`

Explain what went wrong in a run and what to change (advisory).

```bash
apex agent diagnose TARGET
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--no-llm` (on) / `--no-no-llm` (off) | false | Rules only: no model is asked. |
| `--json` | false | Print the result as JSON. |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |

## `apex agent draft-optimization`

Draft an optimization of a history-matched case's forecast (no model).

```bash
apex agent draft-optimization CASE
```

| option | default | description |
|---|---|---|
| `--forecast-end` | | The forecast's last date (ISO), after the history. |
| `--wop` | | Oil price per unit of FOPT. |
| `--wwp` | | Cost per unit of produced water. |
| `--wwi` | | Cost per unit of injected water. |
| `--disc` | | Yearly discount rate. |
| `--wgp` | | Gas price (default: not priced). |
| `--obj-scaling` | | Divides NPV (default -1e6: millions, maximised). |
| `--producer-bhp` | | Producers' pressure bounds LOW,HIGH (with bhp control). |
| `--producer-control` | bhp | Producers on 'bhp' or 'oil_rate'. |
| `--control-periods` | 4 | |
| `--run` | | The run whose posterior is the model (id, name or path). |
| `--members` | 10 | Posterior members used. |
| `--optimizer` | enopt | The optimizer plugin. |
| `--name` | | The draft's name (default <run>_npv). |
| `--out` | | The folder to write it in (default: beside the case). |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--json` | false | Print the result as JSON. |

## `apex agent monitor`

Check a run's health: rules on its facts, and a model's judgement.

```bash
apex agent monitor TARGET
```

| option | default | description |
|---|---|---|
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--watch` (on) / `--no-watch` (off) | false | Check whenever a new forecast is scored, until the job ends. |
| `--interval` | 60 | Seconds between looks, with --watch. |
| `--act` (on) / `--no-act` (off) | false | Offer to stop the job when a stop is recommended (running iterative runs only); you are asked to confirm first. |
| `--no-llm` (on) / `--no-no-llm` (off) | false | Rules only: no model is asked. |
| `--json` | false | Print the result as JSON. |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |

## `apex agent plan`

Propose a study for a case and a goal; submit it only if you confirm.

```bash
apex agent plan CONFIG
```

| option | default | description |
|---|---|---|
| `--goal` | | What the experiment should find out. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--budget` | | Simulator runs for the whole study (default: the approval limit of service.toml). |
| `--out` | | Where to write the study (default: beside the case). |
| `--submit` (on) / `--no-submit` (off) | false | Submit the study after you confirm (through the tools, with an approval token). |
| `--approval-token` | | Submit with this token, approved by `apex approve` after an earlier --submit was pending (implies --submit). |
| `--no-llm` (on) / `--no-no-llm` (off) | false | Rules only: no model is asked. |
| `--json` | false | Print the result as JSON. |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |

## `apex agent plot`

Draw a run's result figures: a preset, or what --request asks for.

```bash
apex agent plot TARGET
```

| option | default | description |
|---|---|---|
| `--request` | | What to draw, in words (default: the classic set). |
| `--preset` | | Draw this preset (e.g. brugge_classic); no model. |
| `--with` | | Switch on a preset's option, e.g. --with model_change --with normalized (brugge_classic). |
| `--output` | | Output folder (default <run>/report/plots). |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--no-llm` (on) / `--no-no-llm` (off) | false | Rules only: no model is asked. |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |

## `apex agent providers`

List the LLM providers.

```bash
apex agent providers
```


## `apex agent report`

Write a narrative report of a run or a study.

```bash
apex agent report TARGET
```

| option | default | description |
|---|---|---|
| `--study` | | Report on this study instead. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |
| `--out` | | Where to write the Markdown. |
| `--no-llm` (on) / `--no-no-llm` (off) | false | Rules only: no model is asked. |
| `--provider` | | Provider (default: [agents.llm] provider). |
| `--model` | | Model (default: [agents.llm] model). |
| `--base-url` | | Server address. |
| `--api-key-env` | | Variable holding the key (never the key itself). |
| `--no-tool-calls` (on) / `--no-no-tool-calls` (off) | false | The model has no native tool calling: prompt for JSON. |
