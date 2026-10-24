r"""A minimal agent loop over any OpenAI-compatible chat API, using APEX's tools.

For local open-source models (Ollama, vLLM, llama.cpp's server, LM Studio) and hosted
ones with the same API (Kimi, GLM, ...). It uses only the standard library: the model
gets APEX's tools in the OpenAI format (``apex.services.tools.export``), and each tool
call it makes is run by APEX's dispatcher (``apex.services.tools.call``), with the same
approval tokens, budgets and call log as the MCP server.

    # e.g. with Ollama running locally and a tool-calling model pulled:
    uv run python examples/llm/openai_compatible_agent.py \\
        --base-url http://localhost:11434/v1 --model qwen3:32b \\
        "Which localization methods does APEX have? Validate /path/to/config.toml."

Spending tools need an approval token; above the approval limits a person must run
``apex approve TOKEN`` before the model can submit, whatever the model does.

This example shows how little an external loop needs. APEX's own loop (``apex agent
ask``, ``apex.agents.loop``) adds prompted JSON for models without tool calling,
validated answers, a rule-based fallback, and a trace of every call.
"""

import argparse
import json
import os
import urllib.request
from typing import Any

from apex.services.api import ApexService
from apex.services.tools import ToolSession, call, export

SYSTEM = (
    "You operate APEX, an ensemble history-matching platform, through its tools. Validate "
    "before submitting; submit only with the approval token validate returned; if a token "
    "is pending, tell the user to run `apex approve TOKEN`. Answer briefly."
)


def chat(
    base_url: str, model: str, messages: list[dict[str, Any]], api_key: str | None
) -> dict[str, Any]:
    """One chat completion with APEX's tools offered."""
    body = json.dumps(
        {"model": model, "messages": messages, "tools": export("openai"), "temperature": 0}
    )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", body.encode(), headers
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        reply: dict[str, Any] = json.loads(response.read())
    return reply


def run(question: str, base_url: str, model: str, api_key: str | None, max_turns: int = 8) -> str:
    """Let the model answer ``question`` with APEX's tools; return its final answer."""
    session = ToolSession(ApexService(), client="agent-loop", model=model, host=base_url)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    for _ in range(max_turns):
        message = chat(base_url, model, messages, api_key)["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            return str(message.get("content") or "")
        for tool_call in calls:
            function = tool_call["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as error:
                result: dict[str, Any] = {"ok": False, "error": f"arguments are not JSON: {error}"}
            else:
                result = call(function["name"], arguments, session)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", ""),
                    "content": json.dumps(result, default=str),
                }
            )
    return "(stopped after the maximum number of turns)"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "--base-url", default=os.environ.get("APEX_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    parser.add_argument("--model", default=os.environ.get("APEX_LLM_MODEL", "qwen3:32b"))
    parser.add_argument(
        "--api-key-env", default="APEX_LLM_API_KEY", help="variable holding the key, if any"
    )
    args = parser.parse_args()
    print(run(args.question, args.base_url, args.model, os.environ.get(args.api_key_env)))


if __name__ == "__main__":
    main()
