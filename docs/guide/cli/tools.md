---
audience: public
items:
  - cli:apex tools call
  - cli:apex tools export
  - cli:apex tools list
---

# The apex tools commands

The `apex tools` command group lets you see which tools APEX offers, hand their definitions to a local model, and run a single tool call yourself. It has three commands: `apex tools list`, `apex tools export` and `apex tools call`.

## Goal

By the end of this lesson you will be able to:

- list the available tools and spot the ones that spend compute;
- export the tool definitions in a format your model server understands (for example Ollama or vLLM);
- run one tool call and read its result, as an agent loop over a local model would.

## Prerequisites

- APEX is installed and the `apex` command works in your shell.
- A services root exists. By default `apex tools call` uses `$APEX_HOME`, or `~/.apex/service` if that is not set. You can point it elsewhere with `--root`.
- Optional: a local model served through something like Ollama or vLLM, if you want to drive the tools from an agent loop.

## Steps

### 1. List the tools

```bash
apex tools list
```

`apex tools list` takes no options. It prints the tools and marks those that spend compute. A tool marked this way needs an approval token before it will run, so check the marks before you build a loop around a tool.

Note the exact tool names in the output. You will pass one of them to `apex tools call`.

### 2. Export the definitions for your model

```bash
apex tools export
```

`apex tools export` prints the tools as function-calling definitions. The `--format` option picks the shape:

| Value | Meaning |
| --- | --- |
| `openai` | The default. |
| `anthropic` | The Anthropic tool format. |
| `json-schema` | Plain JSON Schema. |

For example:

```bash
apex tools export --format anthropic
```

Give the printed definitions to your model server so the model knows which tools it can request.

### 3. Run one tool call

```bash
apex tools call <name>
```

`apex tools call` runs a single tool call and prints its result. This is the command an agent loop uses when the model asks for a tool. Files the call names, and the files their configurations name, must lie in the workspace's roots (by default `<root>/configs`) or in a folder listed under `[tools] roots` in `service.toml`; add your cases' folder there to use them.

`<name>` is required. It is the name of the tool, as shown by `apex tools list`. The options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--json` | `{}` | The tool's arguments, as a JSON string. |
| `--client` | `llm` | Client name, used for quotas and the call log. |
| `--model` | none | The model driving the calls, recorded in the log. |
| `--root` | `$APEX_HOME`, else `~/.apex/service` | The services root. |

## Worked example

Suppose `apex tools list` showed a tool called `my_tool` that takes one argument, `query`. These names are placeholders; use a real name and its real arguments from your own output.

First, export the definitions for an OpenAI-compatible server (the default format):

```bash
apex tools export --format openai
```

Your model replies that it wants to call `my_tool` with `query` set to `hello`. Your loop then runs:

```bash
apex tools call my_tool --json '{"query": "hello"}' --client my-agent --model my-local-model
```

The result is printed, and you feed it back to the model. Because you passed `--client` and `--model`, the call is logged under those names, and quotas apply to the `my-agent` client.

If the tool needs no arguments, leave `--json` out. It defaults to `{}`:

```bash
apex tools call my_tool
```

## Pitfalls

- **Tools that spend compute need an approval token.** `apex tools list` marks them. Do not expect them to run unattended.
- **Quote the JSON.** In a shell, wrap the value of `--json` in single quotes so the double quotes inside survive.
- **Use the exact tool name.** The `name` argument must match what `apex tools list` prints.
- **Everyone shares the default client name.** If you leave `--client` alone, calls are logged and counted under `llm`. Set a distinct name per agent to keep quotas and logs separate.
- **Set `--model` if you want it in the log.** It has no default, so the log has no model name unless you pass one.
- **Check the services root.** If results look wrong, confirm that `$APEX_HOME` or `--root` points where you expect.

## What to read next

- The other pages in the CLI section of this guide, for the rest of the `apex` commands.
- The documentation of your model server (Ollama or vLLM) for how to supply function-calling definitions to a model.
