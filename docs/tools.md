---
audience: public
---

# Tools for LLM clients

APEX exposes its services as one set of **tools**. Claude Code and other MCP hosts use them, and so do agent loops over OpenAI-compatible APIs such as Ollama, vLLM, llama.cpp, LM Studio, and hosted Kimi or GLM. The tools are the same whatever the model.

## Goal

By the end of this page you can connect a model to APEX's tools, check that the connection works, and have the model validate a configuration, get approval for it, and follow the job.

## Prerequisites

- APEX installed, so that the `apex` command works.
- A configuration file that the model will work with (a `config_file`). It must sit in a folder tool clients may read: see [Path rules](#path-rules).
- The stdio setup for Claude Code needs nothing else: `apex mcp` starts the tool server itself.

## Terms

- **Client**: the name a tool session carries (for example `planner`). It is used for quotas and the call log. Set it with `apex mcp --client`; the default is `mcp`.
- **Session**: one connection's tool state: its client name, the model it declares, and its budget.
- **Approval token**: what `validate` returns for a valid plan. A tool that spends compute needs one.
- **Budget**: the simulator runs a session may submit. Set it with `apex mcp --budget-runs`. The default is no budget.
- **Roots**: the folders whose files a tool client may name. Without a `[policy]` they are the workspace's roots and `[tools] roots`; with one, the policy's roots.

## Check the tools locally

```bash
apex tools list                          # all tools; those that spend compute need an approval token
apex tools export --format openai        # or anthropic, json-schema: function-calling definitions
apex tools call validate --json '{"config_file": "/path/run.toml"}' --model glm-4.6
```

- `apex tools list` prints the tools. If it does, the `apex` command and its tool set work. It does not test a host connection.
- `apex tools call validate ...` runs the `validate` tool once, as a model would, with the model name `glm-4.6` recorded for the session. A good result contains either errors that say what to fix, or the plan and an approval token. A refused path means the file is outside the folders tool clients may read (see [Path rules](#path-rules)).

## The workflow a model follows

1. Call `validate` with the `config_file`. It returns errors, or the plan and an approval token.
   - Whether the token is approved at once or **pending** is stated in the `validate` result.
   - A pending token needs a person to run `apex approve TOKEN`, using the token string from that result. Until then the model should report the situation and wait, not submit.
   - With `annotations` such as `{"experiment": "exp-042"}`, `validate` also checks the grants (made with `apex grant create`). If an active grant of the session's client for that experiment covers the cost, the result says `"approval": "granted"` and names the grant. The grant is charged only when the submission is made with the same `experiment` annotation.
2. Call the submit tool for the kind of job, passing the approval token (and, for a granted token, the same `annotations`). Run `apex tools list` to see the submit tools, and `apex tools export --format json-schema` to see each tool's arguments.
3. Follow the job with `job_status` and `job_events`. Their arguments, including how `job_events` picks up where the last call stopped, are in the exported definitions.
4. Read the results as numbers with `run_summary`, `compare_study` and `prediction_summary`.

Also available:

- `list_plugins` and `get_schema` (the whole schema of a kind, or one `section`, for example `prior`), for discovery;
- `list_runs`, to see runs, newest first (`offset` and the result's `next_offset` page through older ones).

Errors come back as results that say what to do (`did you mean ...`, the allowed arguments, "call validate again"), so a model can correct itself. A validation error names the offending keys and says what is wrong with them; it never repeats the values in the file.

## Path rules

A tool client acts with the rights of the service's owner, so what it may make the service read, render or run is confined. Without a `[policy]` (the default), the readable folders are the workspace's roots (`[workspace] roots` in `service.toml`, by default `<root>/configs`), the folders in `[tools] roots`, and the service's own `runs/`, `studies/` and `optimizations/`. Then:

- The files a call names (`config_file`, `study_file`, `prediction_file`, `optimization_file`) and a study's cases must lie in those folders.
- Every file the configuration reads must lie in them too: data and truth files, prior files, the folder of a deck's includes (`case_dir`), and any plugin option declared as an input file. This holds for each cell of a study and its predictions, for predictions and optimizations, and after `overrides` are applied. A prediction's run is checked before its snapshot is read.
- Deck templates and programs written in the configuration file itself (a GEOS or MPSlib build in a home folder, say) are the owner's choice and are accepted wherever they lie. Not so in a web-UI user's folder other than the owner's (`UI/guests/<name>/`, an agent's): a template or program written by a configuration there, or lying there, is refused, since running it would run someone else's code as the owner. A `[policy]` decides those instead.
- `overrides` may change settings and file paths (within the folders), but not deck templates, programs, program arguments or hosts.
- A study or optimization file that does not load is refused, so nothing it names is read unchecked.
- An output folder or a prediction's run may also be a run of the catalogue; so may the folders given to `prediction_summary` and `optimization_summary`.
- Paths are checked before anything is opened, so a refusal does not tell whether a file exists, and a refusal or a validation error never shows a file's contents.

With a `[policy]`, the policy decides instead: its `input_roots`, `code_roots` and `output_roots`, as for every other client (see the job service page).

To let tool clients use cases elsewhere, add their folder to `service.toml`:

```toml
[tools]
roots = ["/data/cases"]
```

If a path is refused, move the file into one of those folders, or ask the owner of the service to add its folder.

## Connect Claude Code (MCP over stdio)

Put this in `.mcp.json` in a project:

```json
{"mcpServers": {"apex": {
  "command": "uv",
  "args": ["run", "--project", "/path/to/APEX", "apex", "mcp",
           "--client", "planner", "--model", "claude-opus-5-5"]
}}}
```

Or add the same server from the command line:

```bash
claude mcp add apex -- uv run --project /path/to/APEX apex mcp --client planner --model claude-opus-5-5
```

**What the model receives.** Claude Code's model (and any MCP host's) receives the tools' answers: configuration settings, metric values, well and data-type names, and paths from your runs and cases, on the provider's servers under its terms. To keep them on your machines, use a local model; see [what a provider receives, and how to keep it local](agents.md#what-a-provider-receives-and-how-to-keep-it-local).

To confirm that Claude Code has picked up the server, ask the model to call `list_plugins`. A list of plugins means the connection works. If the tool is missing, the server did not start: run the `apex mcp` command by hand and read its error.

## Connect other MCP hosts (Streamable HTTP)

`apex mcp --http HOST:PORT` serves Streamable HTTP. Every request needs the service's API token, the one `apex service token` prints (the same as the REST API's); a request without it gets `401`:

```bash
apex mcp --http 127.0.0.1:8770 --client planner --model kimi-k2
# the host connects to http://127.0.0.1:8770/mcp with the header
#   Authorization: Bearer <the output of `apex service token`>
```

Use a port of its own: 8765 is the service's (`apex service start --http`), and the service already serves MCP at `/mcp` behind its token.

Give the token to the host as a header in its server configuration, for example in Claude Code's `.mcp.json`:

```json
{"mcpServers": {"apex": {
  "type": "http",
  "url": "http://127.0.0.1:8770/mcp",
  "headers": {"Authorization": "Bearer <token>"}
}}}
```

On a shared machine, other users can reach a loopback port too, so the token, not the address, keeps them out. Keep the address on loopback all the same; for remote hosts, the service's own `/mcp` endpoint (`apex service start --http`, see the REST API page) serves the same tools behind the same token, with a tool session per MCP session.

## Local open-source models (OpenAI-compatible APIs)

To drive a local model such as one served by Ollama, write a loop that offers the model the definitions from `apex tools export --format openai`, then runs each tool call the model makes. You can test each call by hand with `apex tools call NAME --json '{...}' --model NAME`.

## Worked example

1. Start the tool server: `apex mcp --http 127.0.0.1:8770 --client planner --model kimi-k2`.
2. Connect the host to `http://127.0.0.1:8770/mcp`, with the header `Authorization: Bearer TOKEN`, where `TOKEN` is the output of `apex service token`.
3. Ask the model to call `list_plugins`. A list of plugins confirms the connection.
4. The model calls `validate` with `config_file` set to `/path/run.toml`, and gets a plan and an approval token, or errors to fix.
5. If the token is pending, a person runs `apex approve TOKEN`.
6. The model calls the submit tool shown by `apex tools list`, passing the token, and follows the job with `job_status` and `job_events`.
7. When the job finishes, the model calls `run_summary`.

## Pitfalls

- Submitting a pending token. A person must run `apex approve TOKEN` first.
- Changing the plan after validation. Call `validate` again.
- Files outside the folders tool clients may read are refused (see [Path rules](#path-rules)).
- Connecting to `apex mcp --http` without the token: every request gets `401`.
- Exposing `apex mcp --http` beyond loopback. Keep it on `127.0.0.1`.
- Submitting a granted token without the `experiment` annotation it was validated with. The grant is not used, and the token stays pending.
- Taking a successful `apex tools list` as proof that a host is connected. Ask the model to call `list_plugins` instead.

## What to read next

The pages on the job service and its `[policy]` settings, and on approvals, grants and quotas, explain how to change the roots and limits mentioned here.
