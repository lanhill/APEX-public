---
audience: public
---

# Getting started: the agents from the command line

## Goal

By the end of this page you can run APEX's agents in a terminal with a language model: ask questions about APEX and your work, check a run's health, diagnose it, plan a study, and write a report. You will also connect Claude Code to APEX's tools, so that Claude Code can drive APEX in a conversation.

## Prerequisites

- APEX installed as the README describes. Run every command below from its folder.
- A language model. This page uses two examples:
  - **Claude through Claude Code**, with your Claude subscription;
  - **Gemma through Ollama**, a model you run yourself, or Ollama's cloud Gemma.
- A finished run to look at. The README's first example makes one:

  ```bash
  uv run apex run examples/linear_gaussian/config_esmda.toml
  ```

  Its run folder is `examples/linear_gaussian/runs/linear_esmda`. Runs started with `apex run` in a terminal are not in the service's catalogue, so the agents' `list_runs` does not show them; give the agents the run folder, as Step 3 does. Runs submitted with `apex submit` (or from the web UI) are catalogued.

Every agent except `apex agent ask` also works without a model, with `--no-llm`: it then decides by its rules only. Start with that if you have no model yet.

## Step 1: choose a model

The agents use `[agents.llm]` in `service.toml`, in the *services root*: `$APEX_HOME`, by default `~/.apex/service`. Create the file if it is missing.

### Example A: Claude, through Claude Code and your subscription

```bash
curl -fsSL https://claude.ai/install.sh | bash   # or: npm install -g @anthropic-ai/claude-code
claude                                           # log in once with /login, with your subscription
```

```toml
[agents.llm]
provider = "claude_code"
model = "sonnet"          # or "opus", "haiku"
timeout_s = 300
```

APEX runs `claude -p` as a model only, with none of Claude Code's own tools, MCP servers, settings or hooks, and without any `ANTHROPIC_API_KEY`. The calls count against your subscription's usage.

### Example B: Gemma, through Ollama

```bash
ollama pull gemma4:31b      # runs on your machine; needs a large GPU or much memory
```

Pick a Gemma size that fits your machine from [Ollama's library](https://ollama.com/library/gemma4). Ollama also offers `gemma4:31b-cloud`, run by Ollama's service after `ollama signin`: no download, but your data leaves your machine, as with any hosted model.

```toml
[agents.llm]
provider = "ollama"
model = "gemma4:31b"
timeout_s = 300
```

Ollama listens on `http://localhost:11434` by default; set `base_url` for another address. Gemma 4 calls tools natively. For a model that cannot, add `tool_calls = false`, and APEX asks it for JSON instead.

### Let the agents read the examples

The agents use APEX's tools, and tool clients read only the service's workspace folders unless `service.toml` adds more. To use the examples in the APEX folder, add it:

```toml
[tools]
roots = ["/path/to/apex/examples"]    # the absolute path of your APEX folder's examples
```

### Check the model

```bash
uv run apex agent providers        # the providers your install offers
uv run apex agent ask "hello"      # a first answer through the model
```

Each agent command also takes `--provider` and `--model` for one call, e.g. `--provider ollama --model gemma4:31b`, without changing `service.toml`.

## Step 2: ask the agent

```bash
uv run apex agent ask "Which localization methods does APEX have, and which is the default?"
uv run apex agent ask "Validate examples/linear_gaussian/config_gies.toml and tell me its cost"
```

The agent answers with APEX's tools: plugins, schemas, validation, runs and jobs. It may submit work only within the approval limits of `service.toml`. Above them it gets a pending token, which only a person approves (`uv run apex approve TOKEN`).

## Step 3: check, diagnose and report a run

```bash
RUN=examples/linear_gaussian/runs/linear_esmda
uv run apex agent monitor "$RUN"       # is it healthy, should it stop?
uv run apex agent diagnose "$RUN"      # causes, and what to change (with validated patches)
uv run apex agent report "$RUN"        # a narrative of the results, in Markdown
uv run apex agent plot "$RUN" --request "box plots of the data mismatch"   # figures
```

`RUN` may also be a run id or a job id. In each agent, code computes the facts and named rules decide first. The model then judges with read-only tools, and its answer is held to the facts: a finding no rule backs is dropped, and each correction is printed. Every call is recorded in the run's `agents/` folder, prompt included.

For a running job, `apex agent monitor JOB --watch` checks again after every scored forecast. The monitor only advises. `--act` stops a run only after you confirm at the terminal.

## Step 4: plan a study

```bash
uv run apex agent plan examples/linear_gaussian/config_esmda.toml \
    --goal "does a longer inflation schedule improve the RMSE?" --budget 20000
```

The planner writes a study within the budget, revising it when validation refuses it. It submits nothing unless you add `--submit` and confirm. Run a written study with `uv run apex study run STUDY.toml`, and compare its cells with `uv run apex study compare NAME`.

## Step 5: let Claude Code drive APEX (MCP)

Instead of APEX calling a model, Claude Code can call APEX's tools in a conversation: list plugins, validate and submit runs, read results. In the APEX folder:

```bash
claude mcp add apex -- uv run --project "$PWD" apex mcp --client claude-code --budget-runs 20000
claude                       # then ask, e.g.: "list APEX's update strategies"
```

`--budget-runs` limits the simulator runs this session may submit. Like the agents, Claude Code reads only the folders `[tools] roots` allows (Step 1). Submissions above the service's approval limits still need `uv run apex approve TOKEN` from you. [The tools page](../tools.md) covers other MCP hosts and the tools' path rules.

## What the model receives

The model receives the facts the agents compute from your runs and cases: configuration settings, metric values, well and data-type names, and paths.

- **Claude:** these go to Anthropic.
- **Gemma on your machine:** nothing leaves it. A cloud Gemma model (`gemma4:31b-cloud`) sends them to Ollama's service.
- **No model at all:** use `--no-llm`.

See [what a provider receives](../agents.md#what-a-provider-receives-and-how-to-keep-it-local).

## Pitfalls

- **`claude` not found, or not logged in:** run `claude` once and `/login`. The agents fall back to their rules and say so when the model is unavailable.
- **Ollama answers nothing:** check `ollama list` and that the server runs (`ollama serve`). A large model on a CPU is slow; raise `timeout_s` or use a smaller one.
- **The wrong settings are used:** the agents read the `service.toml` of `$APEX_HOME`. Check `echo $APEX_HOME`, or pass `--root`.
- **"refused" for a file:** tool clients read only the service's workspace roots unless `service.toml` adds folders (`[tools] roots`); see [the tools page](../tools.md#path-rules).

## What to read next

- [The web UI and its AI assistant](web_ui.md): the same models in the browser.
- [Runtime agents and LLM providers](../agents.md): every agent, provider and setting.
- [The tools](../tools.md): the tools the agents and MCP hosts use.
