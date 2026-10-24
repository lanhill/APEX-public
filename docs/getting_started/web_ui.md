---
audience: public
---

# Getting started: the web UI and its AI assistant

## Goal

By the end of this page you have APEX's web UI running in your browser, you have run an example from it, and you can ask its AI assistant about your runs and have it plan a study for you.

The web UI is served by APEX's job service. It shows jobs, runs, studies, predictions, optimizations, plugins and examples, and has an editor for configuration files. The **AI Assistant** is a chat panel on every page. It answers from APEX's own tools and can plan studies as drafts that you launch.

## Prerequisites

- APEX installed as the README describes (`git clone`, then `uv sync --locked` in its folder). Run every command below from that folder.
- [Node.js](https://nodejs.org) 22.12 or later, to build the web UI once (18.18 still builds it, with npm's warnings).
- For the assistant: a language model. This page uses two examples:
  - **Claude through Claude Code**, with your Claude subscription (no API key);
  - **Gemma through Ollama**, a model you run yourself.

  The web UI works without a model. Only the assistant needs one.

## Step 1: build the web UI

```bash
npm --prefix ui ci && npm --prefix ui run build
```

This writes `ui/dist/`. Build it again after `git pull` when the UI has changed.

## Step 2: start the service and sign in

```bash
uv run apex service start --http 127.0.0.1:8765
uv run apex service token --ui --owner
```

The first command starts the service in the background. It keeps its data in the *services root*: `$APEX_HOME`, by default `~/.apex/service`. The second prints your web-UI token.

1. Open `http://127.0.0.1:8765/ui/`.
2. Sign in with the user name `owner` and the token from `apex service token --ui --owner`. The plain `apex service token` prints the API token, which the web UI refuses.
3. The token expires after 7 days. `uv run apex service token --ui --owner --rotate` gives you a new one.

If the service runs on another machine, keep it bound to `127.0.0.1` and open an SSH tunnel from your own computer (`ssh -N -L 8765:127.0.0.1:8765 you@server`). Then open `http://localhost:8765/ui/` there. [The web UI page](../ui.md#using-it-from-another-machine-eg-a-windows-laptop) has the details for Windows, PuTTY and VS Code.

Stop the service with `uv run apex service stop`.

## Step 3: run an example from the web UI

1. Open **Examples**, choose the **History matching** tab, and on the linear-Gaussian example click **Copy (ES-MDA) to my folder**. The copy opens in the **Editor**.
2. Click **Validate**. The editor shows the run's plan: members, forecasts, simulator runs.
3. Click **Submit**. The job appears on the **Jobs** page with live events.
4. When it is complete, open its run. The run page has a chart per metric, the steps and the run's figures.

Only work submitted from the web UI is listed there. Runs started with `apex run` in a terminal do not appear in the web UI.

## Step 4: give the assistant a model

The assistant uses the model of `[chat.llm]` in `service.toml`, or else `[agents.llm]`, which the other agents also use. Choose one of the examples below and put it in `service.toml` in the services root (`$APEX_HOME`, by default `~/.apex/service`; create the file if it is missing).

### Example A: Claude, through Claude Code and your subscription

1. Install Claude Code and log in once:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash   # or: npm install -g @anthropic-ai/claude-code
   claude                                           # then /login, with your Claude subscription
   ```

   Check it with `claude --version`.
2. In `service.toml`:

   ```toml
   [agents.llm]
   provider = "claude_code"
   model = "sonnet"          # or "opus", "haiku"
   timeout_s = 300
   ```

APEX runs `claude -p` as a model only. It turns off Claude Code's own tools, MCP servers, settings, hooks and `CLAUDE.md` files, and it never passes an `ANTHROPIC_API_KEY` on. The calls therefore count against your subscription's usage, not API billing. Each reply takes a few model calls of a few seconds each.

### Example B: Gemma, through Ollama

1. Install [Ollama](https://ollama.com/download) and get a Gemma model:

   ```bash
   ollama pull gemma4:31b         # runs on your machine; needs a large GPU or much memory
   ollama list                    # check that it is there
   ```

   Pick a Gemma size that fits your machine from [Ollama's library](https://ollama.com/library/gemma4). Ollama also offers `gemma4:31b-cloud`, run by Ollama's service after `ollama signin`. With a cloud model, your data leaves your machines, as with any hosted model.
2. In `service.toml`:

   ```toml
   [agents.llm]
   provider = "ollama"
   model = "gemma4:31b"
   timeout_s = 300
   ```

   `provider = "ollama"` talks to Ollama's own API at `http://localhost:11434` (set `base_url` for another address). It asks Ollama for a context large enough for each conversation, up to `max_context` (65536 tokens by default, and never more than the model's own context).

### Restart the service

The service reads `service.toml` when it starts:

```bash
uv run apex service stop && uv run apex service start --http 127.0.0.1:8765
```

To switch between the two models, change `[agents.llm]` (or give the assistant its own `[chat.llm]`) and restart the service.

To keep both and choose in the web UI instead, name them in `[chat.models]` (the keys of `[agents.llm]`, plus a `label`). A selector then appears at the top of the assistant's panel, and `[chat] default_model` is the one it starts with. Your guests use the models that `ui_access.toml` lists under `[chat] models`, tried in that order; see [The assistant's models](../ui.md#the-assistants-models).

```toml
[chat]
default_model = "claude"

[chat.models.claude]
label = "Claude"
provider = "claude_code"
model = "sonnet"

[chat.models.gemma]
label = "Gemma"
provider = "ollama"
model = "gemma4:31b"
```

## Step 5: talk to the assistant

Click **AI Assistant** in the top bar. The panel stays open as you move between pages, and it knows the page you are on. Try:

- *"Which update strategies does ES-MDA accept?"*
- On the run you submitted in Step 3: *"Is this run healthy? What would you change?"* The assistant uses the monitor and diagnosis agents and quotes their findings.
- *"Write a short report of this run."*
- *"Plan a comparison of ES-MDA with and without adaptive localization on my copy of the linear-Gaussian example, within 20,000 simulator runs."*

When it plans a study, it writes a draft into your folder and shows a **card** with the draft's cost and **Launch**, **Review in the editor** and **Discard**. Nothing runs until you click **Launch**. The assistant itself never submits, stops or resumes anything. **Stop** ends an answer at once.

## What the model receives

The model receives your messages and the answers of the assistant's tools: configuration settings, metric values, well and data-type names, and paths from your runs and cases.

- **Claude:** these go to Anthropic, under your subscription's terms.
- **Gemma on your machine:** nothing leaves it. A cloud Gemma model sends them to Ollama's service.

Every call is recorded in `<services root>/chats/<chat id>/agents/`, so you can see what was sent. See [what a provider receives](../agents.md#what-a-provider-receives-and-how-to-keep-it-local).

## Pitfalls

- **"The model is not available"** in the chat: check the model from a terminal first, with `uv run apex agent ask "hello"` (see [the agents from the command line](agents_cli.md)). For Claude Code, run `claude` and check that you are logged in; for Ollama, check that `ollama list` shows the model.
- **Changes to `service.toml` do nothing:** restart the service, and check that you edited the `service.toml` of the services root the service uses (`$APEX_HOME`).
- **The page says the UI is not built:** run Step 1, then reload the page.
- **Answers are slow:** a large local model on a CPU can take minutes per answer; use a smaller Gemma, a GPU, or Claude. `[chat] tool_timeout_s` (120 s) limits each tool.

## What to read next

- [The agents from the command line](agents_cli.md): the same models for the monitor, diagnosis, planner, report and plotter in a terminal.
- [The web UI](../ui.md): every page, guests and their tokens, and the editor.
- [Runtime agents and LLM providers](../agents.md): all providers and settings.
