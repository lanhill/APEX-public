"""The web UI's model chooser, on scripted models (no live model).

Covered: the fallback over a guest's models, who may use which model (the owner any,
a guest the list of ``ui_access.toml``, another kind of web-UI user the owner's default),
the settings' checks, and the REST routes: the models offered, a model per message, and
a guest's agents answering with the guest's models.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.agents.providers.fallback import FallbackProvider
from apex.agents.providers.stub import StubProvider
from apex.api.app import create_app
from apex.core.errors import ConfigError
from apex.core.llm import LLMInvalidOutput, LLMTimeout, LLMUnavailable, Message
from apex.services.api import ApexService
from apex.services.auth import api_token, ui_token
from apex.services.chat_models import choice
from apex.services.settings import ChatSection

ASK = [Message(role="user", content="hi")]


def _answer(text: str) -> str:
    return json.dumps({"answer": text})


def _complete(provider: FallbackProvider) -> str:
    return provider.complete(ASK, max_tokens=64, temperature=0.0).text


def test_a_guests_models_are_tried_in_order_and_a_failed_one_is_left() -> None:
    first = StubProvider([LLMUnavailable("over its limit")], model="beam")
    second = StubProvider(["from gemma", "again"], model="gemma")
    made: list[str] = []

    def factory(name: str, stub: StubProvider) -> Callable[[], StubProvider]:
        def make() -> StubProvider:
            made.append(name)
            return stub

        return make

    provider = FallbackProvider([("beam", factory("beam", first)),
                                 ("gemma", factory("gemma", second))])  # fmt: skip
    assert made == []  # nothing is made before it is needed
    assert _complete(provider) == "from gemma"
    assert provider.chat_model == "gemma"
    assert provider.model_id == "gemma"
    assert provider.describe()["unavailable"] == ["beam: over its limit"]
    assert _complete(provider) == "again"  # the failed model is not tried again
    assert len(first.requests) == 1


def test_a_timeout_moves_on_but_an_invalid_answer_stays_with_its_model() -> None:
    timed_out = FallbackProvider([("a", lambda: StubProvider([LLMTimeout("slow")])),
                                  ("b", lambda: StubProvider(["b"]))])  # fmt: skip
    assert _complete(timed_out) == "b"
    invalid = FallbackProvider([("a", lambda: StubProvider([LLMInvalidOutput("tool call")])),
                                ("b", lambda: StubProvider(["b"]))])  # fmt: skip
    with pytest.raises(LLMInvalidOutput):
        _complete(invalid)  # the loop asks again; it is the model's answer, not its absence
    assert invalid.chat_model == "a"


def test_a_model_that_cannot_be_made_is_skipped_and_all_failing_says_why() -> None:
    def no_key() -> StubProvider:
        raise ConfigError("no key")

    skipped = FallbackProvider([("a", no_key), ("b", lambda: StubProvider(["b"]))])
    assert _complete(skipped) == "b"
    none = FallbackProvider([("a", no_key), ("b", lambda: StubProvider([]))])
    with pytest.raises(LLMUnavailable, match="no model available: a: no key; b: "):
        _complete(none)
    # Afterwards the turn still names the model it tried last, for its record.
    assert none.model_id == "stub-model"
    assert none.chat_model == "b"
    nothing = FallbackProvider([("a", no_key)])
    with pytest.raises(LLMUnavailable):
        nothing.model_id  # noqa: B018 (no model was ever made)


def test_the_stop_flag_and_streaming_reach_the_answering_model() -> None:
    class Stoppable(StubProvider):
        should_stop: Callable[[], bool] | None = None

    stoppable = Stoppable(["whole"])
    provider = FallbackProvider([("a", lambda: stoppable)])

    def flag() -> bool:
        return False

    provider.should_stop = flag
    pieces: list[str] = []
    done = provider.stream_complete(ASK, on_text=pieces.append, max_tokens=64, temperature=0.0)
    assert stoppable.should_stop is flag
    assert done.text == "whole"
    assert pieces == ["whole"]  # a model that cannot stream hands over its whole text
    streaming = FallbackProvider([("s", lambda: StubProvider(["abcdefghij"], streaming=True,
                                                              chunk=4))])  # fmt: skip
    pieces.clear()
    streaming.stream_complete(ASK, on_text=pieces.append, max_tokens=64, temperature=0.0)
    assert pieces == ["abcd", "efgh", "ij"]


def _chat(**models: str) -> ChatSection:
    return ChatSection.model_validate(
        {"models": {n: {"provider": "stub", "label": label} for n, label in models.items()}}
    )


def test_the_owner_chooses_a_guest_gets_the_list_another_user_the_default() -> None:
    chat = _chat(claude="Claude", beam="Beam", gemma="Gemma")
    owner = choice(chat, "owner")
    assert (owner.names, owner.default, owner.choosable) == (("claude", "beam", "gemma"),
                                                             "claude", True)  # fmt: skip
    assert owner.ordered(None) == ("claude",)
    assert owner.ordered("gemma") == ("gemma",)  # the owner's pick, without fallbacks
    guest = choice(chat, "guest", ["beam", "absent", "gemma"])
    assert (guest.names, guest.choosable) == (("beam", "gemma"), False)
    assert guest.note == "These are the models available for your account, used in this order."
    assert guest.ordered(None) == ("beam", "gemma")
    assert guest.ordered("gemma") == ("gemma", "beam")
    with pytest.raises(PermissionError):
        guest.ordered("claude")
    assert choice(chat, "guest", ["gemma"]).note == "This is the model available for your account."
    nothing = choice(chat, "guest", [])
    assert nothing.names == ()
    assert "No assistant model" in (nothing.note or "")
    assert choice(chat, "other").names == ("claude",)
    assert not choice(ChatSection(), "owner").configured


def test_the_settings_refuse_an_unknown_default_a_bad_name_and_two_ways_of_saying_it() -> None:
    assert _chat(beam="Beam").default == "beam"  # the first, without default_model
    with pytest.raises(ValidationError, match="default_model 'x'"):
        ChatSection.model_validate({"models": {"a": {"provider": "stub"}}, "default_model": "x"})
    with pytest.raises(ValidationError, match="lower-case names"):
        ChatSection.model_validate({"models": {"Big Model": {"provider": "stub"}}})
    with pytest.raises(ValidationError, match="exclude each other"):
        ChatSection.model_validate(
            {"llm": {"provider": "stub"}, "models": {"a": {"provider": "stub"}}}
        )


SERVICE = """
[chat]
default_model = "claude"
stream = false

[chat.models.claude]
label = "Claude"
provider = "stub"
model = "claude-stub"
options = {{ replies = [{claude}] }}

[chat.models.beam]
label = "Beam"
provider = "stub"
model = "beam-stub"
options = {{ replies = [] }}

[chat.models.gemma]
label = "Gemma"
provider = "stub"
model = "gemma-stub"
options = {{ replies = [{gemma}] }}
"""


def _answered(http: TestClient, headers: dict[str, str], body: dict[str, str]) -> dict[str, Any]:
    chat = http.post("/v1/chats", headers=headers, json={}).json()
    sent = http.post(f"/v1/chats/{chat['id']}/messages", headers=headers, json=body)
    assert sent.status_code == 202, sent.text
    for _ in range(200):
        full = http.get(f"/v1/chats/{chat['id']}", headers=headers).json()
        if not full["busy"] and len(full["messages"]) == 2:
            reply: dict[str, Any] = full["messages"][1]
            return reply
        time.sleep(0.05)
    raise AssertionError("no answer")


def test_the_chooser_over_http(tmp_path: Path) -> None:
    root = tmp_path / "service"
    root.mkdir()
    replies = ", ".join(json.dumps(_answer(f"answer {i}")) for i in range(3))
    (root / "service.toml").write_text(SERVICE.format(claude=replies, gemma=replies))
    (root / "ui_access.toml").write_text('[chat]\nmodels = ["beam", "gemma"]\n')
    http = TestClient(create_app(root))
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}

    offered = http.get("/v1/chats/models", headers=owner).json()
    assert offered == {
        "models": [{"name": "claude", "label": "Claude", "local": True},
                   {"name": "beam", "label": "Beam", "local": True},
                   {"name": "gemma", "label": "Gemma", "local": True}],
        "default": "claude", "choosable": True, "note": None,
    }  # fmt: skip
    theirs = http.get("/v1/chats/models", headers=guest).json()
    assert theirs["models"] == [{"name": "beam", "label": "Beam", "local": True},
                                {"name": "gemma", "label": "Gemma", "local": True}]  # fmt: skip
    assert theirs["choosable"] is False
    assert theirs["note"].startswith("These are the models available for your account")

    by_default = _answered(http, owner, {"text": "hi"})
    assert by_default["data"]["chat_model"] == "claude"
    picked = _answered(http, owner, {"text": "hi", "model": "gemma"})
    assert picked["data"]["chat_model"] == "gemma"
    # A follow-up naming no model keeps the chat's (release tester C2: a reopened Gemma
    # chat was answered by the default, Claude).
    [gemmas] = [c for c in http.get("/v1/chats", headers=owner).json()
                if http.get(f"/v1/chats/{c['id']}", headers=owner).json()["messages"][-1]
                ["data"].get("chat_model") == "gemma"]  # fmt: skip
    again = http.post(f"/v1/chats/{gemmas['id']}/messages", headers=owner, json={"text": "yes"})
    assert again.status_code == 202, again.text
    for _ in range(200):
        full = http.get(f"/v1/chats/{gemmas['id']}", headers=owner).json()
        if not full["busy"] and len(full["messages"]) == 4:
            break
        time.sleep(0.05)
    assert full["messages"][3]["data"]["chat_model"] == "gemma"
    # A guest's first model (beam, with no replies) is unavailable: the next answers.
    guests = _answered(http, guest, {"text": "hi"})
    assert guests["data"]["chat_model"] == "gemma"
    assert guests["content"].startswith("answer")

    chat = http.post("/v1/chats", headers=guest, json={}).json()
    refused = http.post(f"/v1/chats/{chat['id']}/messages", headers=guest,
                        json={"text": "hi", "model": "claude"})  # fmt: skip
    assert refused.status_code == 403  # not on alice's list
    chat = http.post("/v1/chats", headers=owner, json={}).json()
    unknown = http.post(f"/v1/chats/{chat['id']}/messages", headers=owner,
                        json={"text": "hi", "model": "absent"})  # fmt: skip
    assert unknown.status_code == 403


def test_without_models_every_chat_keeps_the_one_model(tmp_path: Path) -> None:
    root = tmp_path / "service"
    stub = StubProvider([_answer("one model")])
    http = TestClient(create_app(root, chat_provider=lambda: stub))
    owner = {"Authorization": f"Bearer {api_token(root)}"}
    assert http.get("/v1/chats/models", headers=owner).json() == {
        "models": [], "default": None, "choosable": False, "note": None,
    }  # fmt: skip
    reply = _answered(http, owner, {"text": "hi"})
    assert reply["content"] == "one model"
    chat = http.post("/v1/chats", headers=owner, json={}).json()
    named = http.post(f"/v1/chats/{chat['id']}/messages", headers=owner,
                      json={"text": "hi", "model": "claude"})  # fmt: skip
    assert named.status_code == 422


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


def test_a_guests_agents_answer_with_the_guests_models(
    tmp_path: Path, linear_case: Callable[..., Case]
) -> None:
    case = linear_case(name="alpha", ensemble_size=13)
    root = tmp_path / "root"
    root.mkdir()
    (root / "service.toml").write_text(
        f'[workspace]\nroots = ["{tmp_path}"]\n'
        '[chat.models.claude]\nprovider = "stub"\noptions = { replies = [] }\n'
    )
    (root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[agents]\nnames = ["monitor"]\n'
        '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nlocalizations = ["*"]\n'
        'simulators = ["*"]\nexecutors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
        "[actions]\nsubmit = true\n"
    )
    service = ApexService(root)
    http = TestClient(create_app(root))
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='alice')}"}
    handle = http.post("/v1/runs", headers=guest, json={"config_file": str(case.config)}).json()
    assert service.wait(handle["id"], timeout_s=300, poll_s=0.2).state == "complete"
    run = handle["run_ids"][0]

    # No model listed for guests: their agents answer by the rules, and say why; the owner's
    # [agents.llm] (no model set here) is never theirs.
    asked = http.post("/v1/agents/monitor", headers=guest, json={"run": run, "model": True})
    assert asked.status_code == 200, asked.text
    assert "no assistant model is available" in asked.json()["reason"]


def test_a_guest_with_no_listed_model_is_told_so(tmp_path: Path) -> None:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(SERVICE.format(claude="", gemma=""))
    (root / "ui_access.toml").write_text("[workspace]\nfolders = []\n")
    http = TestClient(create_app(root))
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='bob')}"}
    offered = http.get("/v1/chats/models", headers=guest).json()
    assert offered["models"] == []
    assert "No assistant model" in offered["note"]
    reply = _answered(http, guest, {"text": "hi"})
    assert reply["data"]["decided_by"] == "fallback"
    assert "no assistant model is available" in reply["data"]["reason"]


def test_which_models_keep_a_chat_on_this_machine() -> None:
    """Release tester C2 (2026-10-24): the web UI warns on a switch from a model on this
    machine to a hosted one; unsure counts as hosted."""
    from apex.services.chat_models import on_machine

    chat = ChatSection.model_validate({"models": {
        "local": {"provider": "ollama", "model": "gemma4:31b"},
        "cloud": {"provider": "ollama", "model": "gemma4:31b-cloud"},
        "remote": {"provider": "ollama", "model": "gemma4:31b", "base_url": "http://gpu:11434"},
        "vllm": {"provider": "openai_compatible", "model": "m",
                 "base_url": "http://127.0.0.1:8000/v1"},
        "default": {"provider": "openai_compatible", "model": "m"},
        "openai": {"provider": "openai_compatible", "model": "m",
                   "base_url": "https://api.openai.com/v1"},
        "bare": {"provider": "ollama", "model": "m", "base_url": "127.0.0.2:11434"},
        "claude": {"provider": "claude_code", "model": "sonnet"},
    }})  # fmt: skip
    assert {n: on_machine(chat, n) for n in chat.models} == {
        "local": True, "cloud": False, "remote": False, "vllm": True, "default": True,
        "openai": False, "bare": True, "claude": False,
    }  # fmt: skip
