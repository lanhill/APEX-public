---
audience: public
generated: reference
---

# The apex tools commands: reference

Every one of these 3 items, with its options, as APEX's code defines them. Written from the code; the user guide teaches their use.

## `apex tools call`

Run one tool call (e.g. from an agent loop over a local model); print its result.

```bash
uv run apex tools call NAME
```

| option | default | description |
|---|---|---|
| `--json` | {} | Its arguments, as JSON. |
| `--client` | llm | Client name, for quotas and the call log. |
| `--model` | | The model driving the calls, for the log. |
| `--root` | | Services root (default: $APEX_HOME, else ~/.apex/service). |

## `apex tools export`

Print the tools as function-calling definitions (e.g. for Ollama or vLLM).

```bash
uv run apex tools export
```

| option | default | description |
|---|---|---|
| `--format` | openai | openai, anthropic or json-schema. |

## `apex tools list`

List the tools, marking those that spend compute (they need an approval token).

```bash
uv run apex tools list
```
