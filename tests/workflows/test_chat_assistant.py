"""The chat assistant, on scripted models (no live model) and linear runs.

Covered: the chat store (turns, events, one turn at a time), the history's limits
(retention and size, traces deleted with their chat), the agent loop's local tools,
history and events, the assistant's turns (agents as tools, read-only, a zero budget,
the fallback without a model, the turn limit), and the REST routes.
"""

import json
import shutil
import sqlite3
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest
from pydantic import BaseModel, ConfigDict

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from apex.agents.catalogue import CORE_AGENTS
from apex.agents.chat import READ_TOOLS, SYSTEM, UNPROPOSED, ChatAssistant, public_agents
from apex.agents.loop import AgentLoop, Fallback, LocalTool
from apex.agents.providers.stub import StubProvider
from apex.agents.trace import CallRecorder
from apex.api.app import create_app
from apex.core.errors import ConfigError
from apex.core.llm import Completion, LLMProvider, Message, ToolCall
from apex.services import composition
from apex.services.api import ApexService
from apex.services.auth import ui_token
from apex.services.chat_followup import after_launch
from apex.services.daemon import prune_chats
from apex.services.settings import ChatSection, load_settings
from apex.services.tools import TOOLS, ToolSession
from apex.storage.chat_store import ChatBusy, ChatStore

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _answer(text: str) -> str:
    return json.dumps({"answer": text})


# ---------------------------------------------------------------------- the store


def test_the_store_keeps_turns_and_events_and_one_turn_at_a_time(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    chat = store.create("First", {"page": "run", "id": "abc"})

    store.add_turn(chat.id, "user", "hello")
    store.add_turn(chat.id, "assistant", "hi", {"decided_by": "model"})
    store.add_event(chat.id, "model_call", {"turn": 1})
    store.add_event(chat.id, "answer", {"text": "hi"})

    assert [(t.n, t.role, t.content) for t in store.turns(chat.id)] == [
        (0, "user", "hello"), (1, "assistant", "hi"),
    ]  # fmt: skip
    assert [e.kind for e in store.events(chat.id, since=1)] == ["answer"]
    assert store.get(chat.id).turns == 2
    assert store.get(chat.id).context == {"page": "run", "id": "abc"}
    store.acquire(chat.id)
    with pytest.raises(ChatBusy):
        store.acquire(chat.id)
    store.release(chat.id)
    store.acquire(chat.id)
    store.release_all()
    assert not store.get(chat.id).busy
    store.traces(chat.id).mkdir(parents=True)
    store.delete(chat.id)
    assert store.chats() == []
    assert not store.traces(chat.id).exists()
    with pytest.raises(KeyError):
        store.get(chat.id)


def _age(store: ChatStore, chat_id: str, days: float) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - days * 86400))
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE chats SET updated = ? WHERE id = ?", (stamp, chat_id))


def test_the_history_size_survives_a_write_ahead_log_that_vanishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (2026-10-12): SQLite may remove chats.db-wal at a checkpoint between the
    size check's listing and its stat, which ended a chat turn with FileNotFoundError."""
    store = ChatStore(tmp_path)
    store.create("a")
    wal = store.path.with_name(store.path.name + "-wal")
    real_stat = Path.stat

    def stat(self: Path, *args: object, **kwargs: object) -> object:
        if self == wal:  # listed a moment ago, gone now
            raise FileNotFoundError(str(self))
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: True)
    monkeypatch.setattr(Path, "stat", stat)

    assert store.size() == real_stat(store.path).st_size


def test_chats_inactive_for_three_months_are_deleted(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    old, recent = store.create("old"), store.create("recent")
    store.traces(old.id).mkdir(parents=True)
    _age(store, old.id, 93)
    _age(store, recent.id, 91)

    deleted = store.prune(retention_days=ChatSection().retention_days, max_bytes=10**9)

    assert deleted == [old.id]
    assert [c.id for c in store.chats()] == [recent.id]
    assert not store.traces(old.id).exists()


def test_beyond_the_size_limit_the_oldest_chats_go_first(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    ids = []
    for i in range(6):
        chat = store.create(f"chat {i}")
        for _ in range(20):
            store.add_turn(chat.id, "user", "x" * 5000)
        _age(store, chat.id, 10 - i)  # chat 0 is the oldest
        ids.append(chat.id)
    busy = store.create("busy")
    store.add_turn(busy.id, "user", "y" * 5000)
    _age(store, busy.id, 20)  # the oldest of all, but answering
    store.acquire(busy.id)
    before = store.size()
    limit = before // 2

    deleted = store.prune(retention_days=92, max_bytes=limit)

    assert store.size() <= limit * 0.9
    assert deleted == ids[: len(deleted)]  # oldest first
    assert busy.id not in deleted
    assert {c.id for c in store.chats()} == {busy.id, *ids[len(deleted) :]}


def test_the_service_prunes_with_the_configured_limits(tmp_path: Path) -> None:
    (tmp_path / "service.toml").write_text("[chat]\nretention_days = 30\n")
    assert prune_chats(tmp_path, load_settings(tmp_path)) == []  # no chats.db yet: nothing
    assert not (tmp_path / "chats.db").exists()
    store = ChatStore(tmp_path)
    chat = store.create("a")
    _age(store, chat.id, 31)
    store.acquire(chat.id)  # left busy by a service that stopped mid-turn

    deleted = prune_chats(tmp_path, load_settings(tmp_path), first=True)

    assert deleted == [chat.id]


# ---------------------------------------------------------------------- the loop


class Echo(BaseModel):
    """Arguments of the test's local tool."""

    model_config = ConfigDict(extra="forbid")

    word: str


class Reply(BaseModel):
    """The test's answer."""

    answer: str


def test_the_loop_runs_local_tools_with_history_and_events(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, object]]] = []
    echo = LocalTool("echo", "Echo a word.", Echo, lambda a: {"echoed": a.word.upper()})
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "echo", {"word": "apex"}),)),
        Completion(tool_calls=(ToolCall("c2", "echo", {"wrong": 1}),)),
        '{"answer": "done"}',
    ])  # fmt: skip
    loop = AgentLoop(
        stub, CallRecorder(tmp_path, "t"),
        session=ToolSession(ApexService(tmp_path / "s"), client="t"),
        tools=["list_plugins"], local_tools=[echo],
        on_event=lambda kind, fields: events.append((kind, dict(fields))),
    )  # fmt: skip
    history = [Message("user", "earlier question"), Message("assistant", "earlier answer")]

    outcome = loop.run("now", Reply, Fallback("x", lambda: Reply(answer="fallback")),
                       history=history)  # fmt: skip

    assert outcome.decision.answer == "done"
    first, second, _ = stub.requests
    assert [m.content for m in first.messages[1:]] == ["earlier question", "earlier answer", "now"]
    assert [t.name for t in first.tools] == ["list_plugins", "echo"]
    assert "APEX" in second.messages[-1].content
    assert [k for k, _ in events] == [
        "model_call", "tool_call", "tool_result", "model_call", "tool_call", "tool_result",
        "model_call",
    ]  # fmt: skip
    # The result travels as an excerpt, so a reader of the events can hold an answer to it.
    assert events[2][1] == {
        "tool": "echo",
        "ok": True,
        "result": {"result": {"echoed": "APEX"}},
    }
    assert events[5][1]["ok"] is False  # invalid arguments: an error result, with its reason
    assert str(events[5][1]["error"]).startswith("arguments of echo")
    with pytest.raises(ValueError, match="shadow"):
        AgentLoop(stub, CallRecorder(tmp_path, "t"), local_tools=[
            LocalTool("validate", "x", Echo, lambda a: a)
        ])  # fmt: skip


# ---------------------------------------------------------------------- the assistant


def _assistant(tmp_path: Path, provider: LLMProvider | None, **chat: Any) -> ChatAssistant:  # noqa: ANN401
    service = ApexService(tmp_path / "service")
    store = ChatStore(service.root)
    return ChatAssistant(
        service, store, (lambda: provider) if provider is not None else None,
        settings=ChatSection(**chat),
    )  # fmt: skip


def test_the_assistant_uses_an_agent_and_keeps_the_conversation(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    run_dir = str(case.directory / "runs" / "case")
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "diagnose_run", {"run": run_dir}),)),
        _answer("The run is healthy."),
        _answer("You asked about the run."),
    ])  # fmt: skip
    assistant = _assistant(tmp_path, stub)
    chat = assistant.store.create("t", {"page": "run"})

    first = assistant.run(chat.id, "Is my run healthy?", {"page": "run", "id": "abc"})
    second = assistant.run(chat.id, "What did I ask?")

    assert (first, second) == ("The run is healthy.", "You asked about the run.")
    request = stub.requests[0]
    offered = [t.name for t in request.tools]
    assert offered == [
        *READ_TOOLS, "list_agents", "find_cases", "case_info", "monitor_run", "diagnose_run",
        "write_report", "plot_run", "figure_status",
        "plan_study", "draft_optimization", "draft_run", "convert_case", "create_case",
        "draft_prior", "read_draft", "revise_draft", "propose_action",
    ]  # fmt: skip
    assert not {"submit_run", "submit_study", "stop_job", "resume_job", "validate"} & set(offered)
    assert "The user is on the UI page: page run, id abc" in request.messages[-1].content
    diagnosis = json.loads(stub.requests[1].messages[-1].content)
    assert diagnosis["ok"]
    assert diagnosis["result"]["diagnosis"]["verdict"] in ("healthy", "problems")
    third = stub.requests[2].messages
    assert [m.content for m in third[1:3]] == ["Is my run healthy?", "The run is healthy."]
    turns = assistant.store.turns(chat.id)
    assert [t.role for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[1].data["tools"] == ["diagnose_run"]
    kinds = [e.kind for e in assistant.store.events(chat.id)]
    assert kinds[:5] == ["turn_started", "model_call", "tool_call", "tool_result", "model_call"]
    assert kinds.count("answer") == 2
    traces = assistant.store.traces(chat.id) / "agents"
    assert list(traces.glob("chat-*-call.json"))


def test_the_chat_session_cannot_spend(tmp_path: Path) -> None:
    stub = StubProvider([_answer("ok")])
    assistant = _assistant(tmp_path, stub)
    chat = assistant.store.create("t")
    assistant.run(chat.id, "hi")
    assert set(READ_TOOLS) <= set(TOOLS)
    assert all(not TOOLS[name].spends_compute for name in READ_TOOLS)


def test_without_a_model_the_chat_says_so_and_names_the_commands(tmp_path: Path) -> None:
    def broken() -> LLMProvider:
        raise ConfigError("provider 'claude_code' is not installed")

    service = ApexService(tmp_path / "service")
    store = ChatStore(service.root)
    assistant = ChatAssistant(service, store, broken, settings=ChatSection())
    chat = store.create("t")

    answer = assistant.run(chat.id, "Diagnose my run")

    assert "not installed" in answer
    assert all(a.command in answer for a in public_agents())
    from apex.agents.catalogue import AGENTS

    private = [a.command for a in AGENTS if a.visibility == "private"]
    assert not any(command in answer for command in private)  # private agents are never named
    assert store.turns(chat.id)[-1].data["decided_by"] == "fallback"


def test_a_chat_closes_at_its_turn_limit(tmp_path: Path) -> None:
    assistant = _assistant(tmp_path, StubProvider([_answer("one"), _answer("two")]),
                           max_turns_per_chat=4)  # fmt: skip
    chat = assistant.store.create("t")
    assistant.run(chat.id, "1")
    assistant.run(chat.id, "2")

    assert assistant.store.get(chat.id).closed is not None
    with pytest.raises(ValueError, match="closed"):
        assistant.run(chat.id, "3")


# ---------------------------------------------------------------------- REST


def test_the_chat_over_http(tmp_path: Path) -> None:
    stub = StubProvider([_answer("Hello from APEX.")])
    root = tmp_path / "service"
    http = TestClient(create_app(root, token=TOKEN, chat_provider=lambda: stub))

    assert http.get("/v1/chats").status_code == 401
    agents = http.get("/v1/agents", headers=AUTH).json()
    visibility = {a["name"]: a["visibility"] for a in agents}
    # Core's agents as listed; any an installed workspace member contributes is private.
    core = {a.name for a in CORE_AGENTS}
    assert all(v == "private" for k, v in visibility.items() if k not in core)
    assert {k: v for k, v in visibility.items() if k in core} == {
        "monitor": "public", "diagnosis": "public", "planner": "public", "report": "public",
        "plotter": "public", "case-converter": "public", "geomodeler": "public",
    }  # fmt: skip
    chat = http.post("/v1/chats", headers=AUTH, json={"title": "Try"}).json()
    sent = http.post(f"/v1/chats/{chat['id']}/messages", headers=AUTH,
                     json={"text": "hi", "context": {"page": "jobs"}})  # fmt: skip
    assert sent.status_code == 202
    for _ in range(100):
        page = http.get(f"/v1/chats/{chat['id']}/events", headers=AUTH).json()
        if not page["busy"] and any(e["kind"] == "answer" for e in page["events"]):
            break
        time.sleep(0.05)
    stream = http.get(
        f"/v1/chats/{chat['id']}/events", headers={**AUTH, "Accept": "text/event-stream"}
    )
    assert "event: answer" in stream.text
    full = http.get(f"/v1/chats/{chat['id']}", headers=AUTH).json()
    assert [m["content"] for m in full["messages"]] == ["hi", "Hello from APEX."]
    assert [c["id"] for c in http.get("/v1/chats", headers=AUTH).json()] == [chat["id"]]
    assert http.delete(f"/v1/chats/{chat['id']}", headers=AUTH).status_code == 204
    assert http.get(f"/v1/chats/{chat['id']}", headers=AUTH).status_code == 404
    assert (root / "chats.db").exists()


def test_a_second_message_while_answering_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "service"
    http = TestClient(create_app(root, token=TOKEN, chat_provider=lambda: StubProvider([])))
    chat = http.post("/v1/chats", headers=AUTH, json={}).json()
    ChatStore(root).acquire(chat["id"])

    busy = http.post(f"/v1/chats/{chat['id']}/messages", headers=AUTH, json={"text": "hi"})

    assert busy.status_code == 409


# ---------------------------------------------------------------------- phase 2: planning


def _planning(
    tmp_path: Path, linear_case: CaseFactory, provider: LLMProvider | None
) -> tuple[Case, ApexService, ChatStore, ChatAssistant]:
    case = linear_case(name="alpha", ensemble_size=20)
    root = tmp_path / "service"
    root.mkdir(exist_ok=True)
    (root / "service.toml").write_text(
        f'[workspace]\nroots = ["{tmp_path}"]\n[approval]\nmax_simulator_runs = 150\n'
    )
    service = ApexService(root)
    store = ChatStore(root)
    assistant = ChatAssistant(service, store, (lambda: provider) if provider else None)
    return case, service, store, assistant


def test_the_assistant_plans_a_draft_and_proposes_it_but_never_submits(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"},
                {"name": "loc", "patch": {"method": {"localization": {"name": "adaptive"}}},
                 "rationale": "localized"}]  # fmt: skip
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "does localization help", "budget": 200,
            "name": "loc_test", "variants": variants, "replicates": 3}),)),
        Completion(tool_calls=(ToolCall("c2", "plan_study", {
            "case": "alpha/config.toml", "goal": "does localization help", "budget": 200,
            "name": "loc_test", "variants": variants, "replicates": 1}),)),
        Completion(tool_calls=(ToolCall("c3", "propose_action", {
            "draft": "PLACEHOLDER", "summary": "Launch the two-variant study."}),)),
        _answer("I planned the study and proposed it."),
    ])  # fmt: skip
    _, service, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")
    shown, folder = assistant.drafts(chat.id)
    draft = f"{shown}/study_loc_test.toml"
    replies = list(stub.replies)
    replies[2] = Completion(tool_calls=(ToolCall("c3", "propose_action", {
        "draft": draft, "summary": "Launch the two-variant study."}),))  # fmt: skip
    stub.replies = replies

    answer = assistant.run(chat.id, "Plan a localization study on alpha")

    assert answer == "I planned the study and proposed it."
    over = json.loads(stub.requests[1].messages[-1].content)
    assert not over["ok"]
    assert "over its budget of 200" in over["error"]  # 2 x 3 x 100 = 600 > 200: revised
    assert "To fit: run 1 replicate (200 runs)" in over["error"]  # a reported issue, #248
    planned = json.loads(stub.requests[2].messages[-1].content)["result"]
    assert planned["cost"] == 2 * 1 * 100
    # What the arms share and how they differ, from the cells (a reported issue).
    assert planned["held_fixed"]["prior.ensemble_size"] == 20
    assert planned["differs"]["alpha | loc"]["method.localization.name"] == "adaptive"
    # No localization in the baseline: "none", not a default that would localize.
    assert planned["differs"]["alpha | baseline"]["method.localization.name"] == "none"
    assert list(planned["seeds"]) == ["r0"]
    # Whether rmse can compare, for the answer to state (a reported issue, #247).
    assert planned["true_model"].startswith("yes: the case is synthetic")
    assert (folder / "study_loc_test.toml").exists()
    [proposal] = store.proposals(chat.id)
    assert (proposal.state, proposal.draft, proposal.data["cost"]) == ("pending", draft, 200)
    assert service.jobs.all_records() == []  # nothing ran: only a person launches
    assert shown.startswith("UI/owner/drafts/")  # the owner's web-UI folder


def test_a_draft_the_model_does_not_propose_gets_no_card_and_is_named(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#273 (the user, 2026-11-02): a card appeared for a plan the answer had said does not
    answer the request; only propose_action shows one now, and the answer names the saved
    draft (it was given a card anyway since a reported issue of 2026-10-24)."""
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "baseline only", "budget": 200,
            "name": "no_card", "variants": variants, "replicates": 1}),)),
        _answer("This plan does not answer your request, so I am not proposing it."),
    ])  # fmt: skip
    _, service, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")
    _, folder = assistant.drafts(chat.id)

    answer = assistant.run(chat.id, "Plan a study on alpha so I can review and launch it")

    assert answer == (
        "This plan does not answer your request, so I am not proposing it.\n\n"
        + UNPROPOSED.format(drafts="`study_no_card.toml`")
    )
    assert store.turns(chat.id)[-1].content == answer
    assert store.proposals(chat.id) == []
    assert (folder / "study_no_card.toml").exists()  # saved, without a card
    assert service.jobs.all_records() == []


def test_asked_to_propose_a_study_the_assistant_proposes_it_at_once() -> None:
    """A reported issue (2026-10-24): asked to set a study up and propose it, the model planned
    it and asked questions instead, holding the card back for its unequal arms and for the
    approval launching needs; the card states both, and approval applies only at launch."""
    assert "call propose_action in that answer once its draft keeps to the rules" in SYSTEM
    assert "approval applies only at launch" in SYSTEM
    assert "ask any other questions after it" in SYSTEM
    # After the sentence that introduces the card, and before the rules it yields to.
    rule = SYSTEM.index("call propose_action in that answer")
    assert SYSTEM.index("shows the user a card") < rule < SYSTEM.index("never propose a card over")


def test_a_draft_saved_before_a_fallback_answer_is_named_in_it(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A turn that planned a draft and then gave no usable answer names the saved draft (no
    card, #273); the stored turn and the answer event agree."""
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "baseline only", "budget": 200,
            "name": "fell_back", "variants": variants, "replicates": 1}),)),
    ])  # fmt: skip
    _, _, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")

    answer = assistant.run(chat.id, "Set up a baseline study on alpha and propose it")

    assert store.turns(chat.id)[-1].data["decided_by"] == "fallback"
    assert answer.endswith(UNPROPOSED.format(drafts="`study_fell_back.toml`"))
    assert store.turns(chat.id)[-1].content == answer
    [event] = [e for e in store.events(chat.id) if e.kind == "answer"]
    assert event.data["text"] == answer
    assert store.proposals(chat.id) == []


def test_a_timed_out_model_is_said_in_plain_words_and_the_draft_still_named(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """issue #239: the answer was the provider's error ("no answer in 120 s (2
    attempt(s))"); a timeout is now said plainly, with what was done and the card named."""
    from apex.core.llm import LLMTimeout

    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "baseline only", "budget": 200,
            "name": "slow", "variants": variants, "replicates": 1}),)),
        LLMTimeout("claude_code: no answer in 120 s (2 attempt(s))"),
    ])  # fmt: skip
    _, _, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")

    answer = assistant.run(chat.id, "Set up a baseline study on alpha and propose it")

    assert answer.startswith("The model took too long to answer")
    assert "120 s" not in answer
    assert "attempt" not in answer
    assert "plan_study (done)" in answer
    assert answer.endswith(UNPROPOSED.format(drafts="`study_slow.toml`"))
    assert "no answer in 120 s" in store.turns(chat.id)[-1].data["reason"]
    [shown] = [e for e in store.events(chat.id) if e.kind == "fallback"]
    assert shown.data["reason"] == "the model took too long to answer"
    # #290: the answer says the provider failed, not the assistant, for clients judging it.
    [said] = [e for e in store.events(chat.id) if e.kind == "answer"]
    assert said.data["unavailable"] is True
    assert said.data["timed_out"] is True


def test_the_closing_sentence_names_every_unproposed_draft() -> None:
    """None, one, several (#273: saved drafts named, no cards)."""
    from apex.agents.chat import _unproposed_note

    assert _unproposed_note([]) == ""
    one = _unproposed_note(["UI/owner/drafts/c/study_a.toml"])
    assert one == "\n\n" + UNPROPOSED.format(drafts="`study_a.toml`")
    two = _unproposed_note(["d/study_a.toml", "d/study_b.toml"])
    assert two == "\n\n" + UNPROPOSED.format(drafts="`study_a.toml`, `study_b.toml`")


def test_a_stopped_turn_names_none_of_its_drafts(tmp_path: Path, linear_case: CaseFactory) -> None:
    """After Stop (or once another turn is the chat's), no draft is named: its event would be
    dropped, or it is a later turn's draft."""
    import threading

    from apex.agents.chat import _Turn

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    plan.handler(plan.arguments(case="alpha/config.toml", goal="baseline only", budget=200,
                                name="stopped", variants=variants, replicates=1))  # fmt: skip
    stopped = threading.Event()
    stopped.set()
    assistant._running[chat.id] = 7

    assert assistant._unproposed_drafts(chat.id, 0, _Turn(7, stopped, [])) == []
    other = _Turn(6, threading.Event(), [])  # no longer the chat's turn
    assert assistant._unproposed_drafts(chat.id, 0, other) == []
    assert store.proposals(chat.id) == []
    current = _Turn(7, threading.Event(), [])
    assert assistant._unproposed_drafts(chat.id, 0, current) == [
        f"{assistant.drafts(chat.id)[0]}/study_stopped.toml"
    ]


def test_a_draft_the_model_proposed_gets_no_second_card(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The model's own propose_action is kept as it is: no extra card for the same draft."""
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "baseline only", "budget": 200,
            "name": "one_card", "variants": variants, "replicates": 1}),)),
        Completion(tool_calls=(ToolCall("c2", "propose_action", {
            "draft": "PLACEHOLDER", "summary": "Launch the baseline study."}),)),
        _answer("Planned and proposed."),
    ])  # fmt: skip
    _, _, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")
    shown, _ = assistant.drafts(chat.id)
    replies = list(stub.replies)
    proposed = {"draft": f"{shown}/study_one_card.toml", "summary": "Launch the baseline study."}
    replies[1] = Completion(tool_calls=(ToolCall("c2", "propose_action", proposed),))
    stub.replies = replies

    answer = assistant.run(chat.id, "Plan a study on alpha")

    assert answer == "Planned and proposed."  # its own card: no closing sentence
    [proposal] = store.proposals(chat.id)
    assert proposal.summary == "Launch the baseline study."  # the model's, not the fallback's
    assert [e.kind for e in store.events(chat.id)].count("proposal") == 1


def test_a_proposal_is_launched_or_discarded_by_the_user_over_http(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan = tools["plan_study"]
    result = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))
    draft = result["draft"]
    revise = tools["revise_draft"]
    with pytest.raises(ValueError, match="not a valid draft"):
        revise.handler(revise.arguments(draft=draft, text="[study]\nname = 'x'\n" * 2))
    text = (tmp_path / draft).read_text()
    revised = revise.handler(revise.arguments(draft=draft, text=text + "\n# reviewed\n"))
    assert "+# reviewed" in revised["diff"]
    with pytest.raises(ValueError, match="not a draft of this chat"):
        revise.handler(revise.arguments(draft="alpha/config.toml", text=text))
    propose = tools["propose_action"]
    first = propose.handler(propose.arguments(draft=draft, summary="launch it"))["proposal"]
    second = propose.handler(propose.arguments(draft=draft, summary="launch it now"))["proposal"]
    assert [p.state for p in store.proposals(chat.id)] == ["replaced", "pending"]

    http = TestClient(create_app(service.root, token=TOKEN, chat_provider=lambda: StubProvider([])))
    job = service.submit_study(tmp_path / draft)
    launched = http.post(f"/v1/chats/{chat.id}/proposals/{second}:launched", headers=AUTH,
                         json={"job": job.id})  # fmt: skip
    assert launched.status_code == 200
    assert launched.json()["state"] == "launched"
    again = http.post(f"/v1/chats/{chat.id}/proposals/{second}:discard", headers=AUTH)
    assert again.status_code == 409
    assert http.post(f"/v1/chats/{chat.id}/proposals/{first}:discard", headers=AUTH
                     ).status_code == 409  # replaced, not pending  # fmt: skip
    full = http.get(f"/v1/chats/{chat.id}", headers=AUTH).json()
    assert full["proposals"][1]["job"] == job.id
    # The launch message, followed by the study's end once it ends: on a loaded machine the
    # study may end before this request, so the launch message need not be the last one.
    launches = [m["content"] for m in full["messages"] if m["content"].startswith("You launched")]
    assert len(launches) == 1

    # A launch answered with another submission's job (not the user's to see): settled
    # with no job, said in the chat, and not launched twice (release tester U9).
    third = propose.handler(propose.arguments(draft=draft, summary="once more"))["proposal"]
    missing = http.post(f"/v1/chats/{chat.id}/proposals/{third}:launched", headers=AUTH, json={})
    assert missing.status_code == 422
    elsewhere = http.post(f"/v1/chats/{chat.id}/proposals/{third}:launched", headers=AUTH,
                          json={"elsewhere": True})  # fmt: skip
    assert elsewhere.status_code == 200, elsewhere.json()
    assert (elsewhere.json()["state"], elsewhere.json()["job"]) == ("launched", None)
    assert http.post(f"/v1/chats/{chat.id}/proposals/{third}:launched", headers=AUTH,
                     json={"elsewhere": True}).status_code == 409  # fmt: skip
    # Not necessarily the last message: the study launched above may end meanwhile.
    told = [m["content"] for m in http.get(f"/v1/chats/{chat.id}", headers=AUTH).json()["messages"]]
    assert sum("outside your web-UI folder: nothing new was submitted" in m for m in told) == 1


def test_ui_users_see_their_own_drafts_and_plan_only_if_listed(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    _, service, _, assistant = _planning(tmp_path, linear_case, None)
    ui = {"Authorization": f"Bearer {ui_token(service.root, create=True)}"}
    http = TestClient(create_app(service.root, chat_provider=lambda: StubProvider([])))
    theirs = http.post("/v1/chats", headers=ui, json={}).json()
    (service.root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[agents]\nnames = ["diagnosis"]\n'
    )
    from apex.services.access import Access, load_access

    access = Access(load_access(service.root), service.jobs.all_records())
    assert assistant._planning(theirs["id"], access) == []  # the planner is not listed
    (service.root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[agents]\nnames = ["planner"]\n'
        '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nlocalizations = ["*"]\n'
        'simulators = ["*"]\nexecutors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
    )
    access = Access(load_access(service.root), service.jobs.all_records())
    tools = {t.name: t for t in assistant._planning(theirs["id"], access)}
    plan = tools["plan_study"]
    draft = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))[
        "draft"
    ]

    listing = [e["path"] for e in http.get("/v1/files", headers=ui).json()]
    assert draft.startswith(f"UI/guests/guest/drafts/{theirs['id']}/")
    assert set(listing) == {"alpha", "UI"}
    inside = [e["path"] for e in http.get("/v1/files", headers=ui, params={"path": "UI"}).json()]
    assert inside == ["UI/guests"]  # not UI/owner
    assert http.get("/v1/files/content", headers=ui, params={"path": draft}).status_code == 200
    owners = http.post("/v1/chats", headers=AUTH_OWNER(service.root), json={}).json()
    owners_shown, owners_folder = assistant.drafts(owners["id"])
    assert owners_shown.startswith("UI/owner/drafts/")
    owners_folder.mkdir(parents=True)
    (owners_folder / "x.toml").write_text("# the owner's\n")
    other_draft = f"{owners_shown}/x.toml"
    assert (
        http.get("/v1/files/content", headers=ui, params={"path": other_draft}).status_code == 404
    )
    current = http.get("/v1/files/content", headers=ui, params={"path": draft}).json()
    body = {"path": draft, "text": current["text"] + "#\n", "sha256": current["sha256"]}
    saved = http.put("/v1/files/content", headers=ui, json=body)
    assert saved.status_code == 200  # their own draft, although [actions] edit is off


def AUTH_OWNER(root: Path) -> dict[str, str]:
    from apex.services.auth import api_token

    return {"Authorization": f"Bearer {api_token(root)}"}


def test_algorithms_are_planned_as_methods_and_the_draft_is_what_was_validated(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The 2026-09-30 fault: ES-MDA planned as a variant of a GIES case, with
    ``max_trials_per_iteration = null`` dropped from the TOML draft (TOML has no null), so the
    launch check refused the draft that plan_study had passed."""
    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="gies5", ensemble_size=20,
        method='name = "gies"\noptions = { max_trials_per_iteration = 4 }\n'
        'strategy = { name = "rlmmac", options = { energy = 0.99 } }\nmax_iterations = 2',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]

    def call(**arguments: Any) -> dict[str, Any]:  # noqa: ANN401
        result: dict[str, Any] = plan.handler(
            plan.arguments(
                case="gies5/config.toml", goal="compare algorithms", budget=5000, **arguments
            )
        )
        return result  # fmt: skip

    def switched(options: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"name": "esmda", "rationale": "ES-MDA",
                 "patch": {"method": {"name": "esmda", "options": options}}}]  # fmt: skip

    with pytest.raises(ValueError, match="null"):
        call(variants=switched({"inflation": [4, 4, 4, 4], "max_trials_per_iteration": None}))
    with pytest.raises(ValueError, match=r"study\.methods"):
        call(variants=switched({"inflation": [4, 4, 4, 4]}))

    result = call(
        name="esmda_vs_gies",
        methods={
            "esmda": {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]},
                      "strategy": {"name": "approx"}},
            "gies": {"name": "gies", "options": {"max_trials_per_iteration": 4},
                     "strategy": {"name": "rlmmac", "options": {"energy": 0.99}},
                     "max_iterations": 2},
        },
        variants=[
            {"name": "plain", "rationale": "no localization"},
            {"name": "adaptive", "rationale": "adaptive localization",
             "patch": {"method": {"localization": {"name": "adaptive"}}}},
        ],
    )  # fmt: skip

    assert set(result["methods"]) == {"esmda", "gies"}
    draft = tmp_path / result["draft"]
    validation = service.validate_study(draft)  # the launch check reads the file
    assert validation.valid, validation.errors
    assert len(validation.plans) == 4
    spec = tomllib.loads(draft.read_text())["study"]
    assert "max_trials_per_iteration" not in spec["methods"]["esmda"].get("options", {})
    # The case as the UI shows it, no server path; the linear model runs no deck,
    # so the study keeps its own simulator.
    assert spec["cases"] == {"gies5": "gies5/config.toml"}
    assert "simulator" not in spec


def test_without_the_web_uis_folder_the_assistant_writes_no_drafts(tmp_path: Path) -> None:
    root = tmp_path / "service"
    root.mkdir()
    (root / "service.toml").write_text(f'[workspace]\nroots = ["{tmp_path}"]\nui_folder = ""\n')
    service = ApexService(root)
    store = ChatStore(root)
    assistant = ChatAssistant(service, store, None)
    chat = store.create("plan")

    with pytest.raises(ValueError, match="set \\[workspace\\] ui_folder"):
        assistant.drafts(chat.id)
    assert not (tmp_path / "apex_drafts").exists()


def test_case_info_reads_a_cases_configuration_and_data(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The 2026-10-01 question ("get the grid and position information of the 5spots
    model"): the assistant had no tool that reads a case."""
    _, _, _, assistant = _planning(tmp_path, linear_case, None)
    tool = {t.name: t for t in assistant.local_tools()}["case_info"]

    info = tool.handler(tool.arguments(case="alpha/config.toml"))

    assert info["case"] == "alpha/config.toml"
    assert info["method"]["name"] == "esmda"
    assert [v["name"] for v in info["state_variables"]] == ["x"]
    assert info["data"]["n_data"] == 8
    assert "deck" not in info  # the linear model has no deck (tests/unit/simulation)


def test_case_info_gives_the_prior_and_the_observation_errors(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (#176/#177): the assistant ranked "a tight prior" and "tight data errors"
    as causes without any tool giving it either."""
    _, _, _, assistant = _planning(tmp_path, linear_case, None)
    tool = {t.name: t for t in assistant.local_tools()}["case_info"]

    info = tool.handler(tool.arguments(case="alpha/config.toml"))

    assert info["prior"]["x"]["name"] == "gaussian"
    errors = info["data"]["errors"]
    assert sum(e["count"] for e in errors.values()) == 8
    for each in errors.values():
        assert 0 < each["sigma_min"] <= each["sigma_mean"] <= each["sigma_max"]


def test_a_refused_tool_call_tells_the_user_why(tmp_path: Path, linear_case: CaseFactory) -> None:
    """A planner refusing a design showed only "plan_study failed" in the web UI."""
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "compare", "budget": 1}),)),
        _answer("Over budget; I would need more simulator runs."),
    ])  # fmt: skip
    _, _, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")

    assistant.run(chat.id, "Plan a study")

    [result] = [e for e in store.events(chat.id) if e.kind == "tool_result"]
    assert result.data["ok"] is False
    assert "over its budget of 1. Runs per replicate" in result.data["error"]


def test_a_streaming_model_writes_its_answer_as_it_goes(tmp_path: Path) -> None:
    """The answer's text arrives as answer_delta events, a few words at a
    time (not one row per token), and adds up to the answer."""
    answer = "The run converged after six iterations, and its RMSE fell by about 40 percent."
    stub = StubProvider([_answer(answer)], streaming=True, chunk=5)
    assistant = _assistant(tmp_path, stub)
    chat = assistant.store.create("streaming")

    assert assistant.run(chat.id, "How did it go?") == answer

    deltas = [e.data["text"] for e in assistant.store.events(chat.id) if e.kind == "answer_delta"]
    assert "".join(deltas) == answer
    assert 1 < len(deltas) < len(answer) // 5  # batched


def test_the_chat_hears_when_the_study_it_launched_ends(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The chat gets a message when its launched study ends, and the job's
    notification (e.g. a Teams card) names the study and the chat."""
    import sys

    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    record = tmp_path / "notified.jsonl"
    script = tmp_path / "notify.py"
    script.write_text(
        f"import json, sys\nopen({str(record)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    toml = (service.root / "service.toml").read_text()
    (service.root / "service.toml").write_text(
        toml + f'[notifications]\ncommand = "{sys.executable} {script} {{event}} {{job}} '
        '{state} {summary}"\n'
    )
    service = ApexService(service.root)
    assistant = ChatAssistant(service, store, None)
    chat = store.create("Compare localizations")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    draft = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))
    propose = {t.name: t for t in assistant.planning_tools(chat.id, None)}["propose_action"]
    n = propose.handler(propose.arguments(draft=draft["draft"], summary="launch it"))["proposal"]
    job = service.submit_study(tmp_path / draft["draft"])
    store.settle(chat.id, n, "launched", job.id)
    # As the API records a launch: the study may have ended, and been notified finding no
    # chat to tell, before the launch was recorded (a test process stalled for the study's
    # two seconds on a loaded host: the CI flakes of 2026-10-14 and 2026-10-18).
    at_launch = service.jobs.record(job.id)
    after_launch(service.root, at_launch)
    # Notified before the launch was recorded: its notice may not name the chat.
    late = bool(at_launch.get("notified"))

    assert service.wait(job.id, timeout_s=300, poll_s=0.2).state == "complete"

    # The study is marked complete, then notified (and its chat told), possibly by another
    # process (the last cell's worker): wait for the chat's turn and then its event, which
    # is written in a transaction of its own after the turn.
    def told() -> bool:
        return bool(store.turns(chat.id)) and any(
            e.kind == "job_ended" for e in store.events(chat.id)
        )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not told():
        time.sleep(0.2)

    # Should it fail again: say why.
    logs = "\n".join(f"{log.parent.name}: {log.read_text()[-2000:]}"
                     for log in sorted((service.root / "jobs").glob("*/worker.log")))  # fmt: skip
    assert told(), f"the chat was not told; job {service.jobs.record(job.id)}; workers:\n{logs}"
    last = store.turns(chat.id)[-1]
    assert last.role == "assistant"
    assert "has ended: **complete**" in last.content
    assert last.data["job"] == job.id
    assert [e for e in store.events(chat.id) if e.kind == "job_ended"]

    def chat_notices() -> list[list[str]]:
        lines = record.read_text().splitlines() if record.exists() else []
        notices = [json.loads(line) for line in lines]
        return [n for n in notices if n[1] == job.id and n[0] == "chat_job_ended"]

    if late:  # (rare: the job's plain notice went out; the chat was told above)
        return

    # The job's own notifications may be written first: wait for the chat's.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not chat_notices():
        time.sleep(0.2)
    mine = chat_notices()
    assert mine
    assert mine[0][0] == "chat_job_ended"
    assert "launched from the chat \u201cCompare localizations\u201d" in mine[0][3]


def test_a_launch_recorded_after_the_study_ended_still_tells_the_chat_once(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The web UI submits a study, then records its launch: a quick study can end, and be
    notified, in between. Recording the launch then tells the chat, and only once."""
    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("Compare localizations")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    draft = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))
    propose = {t.name: t for t in assistant.planning_tools(chat.id, None)}["propose_action"]
    n = propose.handler(propose.arguments(draft=draft["draft"], summary="launch it"))["proposal"]
    job = service.submit_study(tmp_path / draft["draft"])
    assert service.wait(job.id, timeout_s=300, poll_s=0.2).state == "complete"
    record = service.jobs.record(job.id)
    assert record.get("notified")  # notified with no chat to tell
    assert not [e for e in store.events(chat.id) if e.kind == "job_ended"]

    store.settle(chat.id, n, "launched", job.id)
    after_launch(service.root, record)
    after_launch(service.root, record)  # a second time (the notification racing it)

    told = [e for e in store.events(chat.id) if e.kind == "job_ended"]
    assert len(told) == 1
    assert "has ended: **complete**" in store.turns(chat.id)[-1].content


def test_a_chat_is_told_its_job_ended_once_when_two_tell_it_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding (2026-10-18): the "told already" check read the events outside the
    writes, so the notification and a late launch's follow-up, in two processes, could
    both pass it and tell the chat twice. Both here pass a stale check (no events seen)."""
    from apex.services.chat_followup import chat_followup

    store = ChatStore(tmp_path)
    chat = store.create("Compare localizations")
    proposal = store.add_proposal(chat.id, "launch_study", "drafts/x.toml", "launch it", {})
    store.settle(chat.id, proposal.n, "launched", "job-1")
    record = {"id": "job-1", "state": "complete", "spec": {}}
    monkeypatch.setattr(ChatStore, "events", lambda self, chat_id, since=0: [])

    assert chat_followup(tmp_path, record) is not None
    # The other teller, its check stale: it tells nothing, nor names the chat in its notice.
    assert chat_followup(tmp_path, record) is None
    assert store.tell_job_ended(chat.id, "job-1", "again", {}, "complete") is False

    monkeypatch.undo()
    assert [e.kind for e in store.events(chat.id)].count("job_ended") == 1
    assert sum("has ended" in t.content for t in store.turns(chat.id)) == 1


def test_every_plugin_names_its_workflow() -> None:
    from apex.services.plugins import all_plugins

    workflows = {p.registry: p.workflow for p in all_plugins()}
    assert workflows["methods"] == "history_matching"
    assert workflows["forecast_scores"] == "prediction"
    assert workflows["response_objectives"] == "optimization"
    assert workflows["simulators"] == "simulation"
    assert None not in workflows.values()


def test_apex_against_erts_localization_is_planned_within_the_turns(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The 2026-10-04 chat ("plan a study on 5spots to compare adaptive localization in APEX
    and ERT"): lookups used half the turns, ERT-engine methods with a strategy were refused
    without a trace, and nothing was planned. Now the refusal is recorded and names the
    design, it adds a turn, and the rules plan ES-MDA with adaptive_rows against
    ert_adaptive, and approx and ert_esmda as references."""
    pytest.importorskip("iterative_ensemble_smoother")
    goal = "compare adaptive localization in APEX and ERT"
    lookups = [("case_info", {"case": "alpha/config.toml"})] + [
        ("list_plugins", {"registry": r}) for r in
        ("methods", "update_strategies", "localizations", "metrics", "simulators", "priors")
    ] + [("list_agents", {})]  # fmt: skip
    wrong = {"ert_adaptive": {"engine": "ert", "name": "es_mda", "options": {"weights": [4, 4]},
                              "strategy": {"name": "ert_adaptive"}}}  # fmt: skip
    stub = StubProvider([
        *(Completion(tool_calls=(ToolCall(f"l{i}", n, a),)) for i, (n, a) in enumerate(lookups)),
        Completion(tool_calls=(ToolCall("p1", "plan_study", {
            "case": "alpha/config.toml", "goal": goal, "budget": 2000, "methods": wrong}),)),
        Completion(tool_calls=(ToolCall("p2", "plan_study", {
            "case": "alpha/config.toml", "goal": goal, "budget": 2000}),)),
        _answer("I planned ES-MDA with adaptive_rows against ert_adaptive."),
    ])  # fmt: skip
    # Ten turns of lookups and two plans fill the stub's 32k context to its edge; this
    # test checks the turns, not the context (the figure_status tool tipped it, 2026-10-31).
    stub.max_context = 40_000
    _, service, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("ert")

    answer = assistant.run(chat.id, "plan a study on alpha to " + goal)

    # The model did not propose its draft: no card (#273), and the answer names the draft.
    assert answer.startswith("I planned ES-MDA with adaptive_rows against ert_adaptive.\n\n")
    assert "Saved without a launch card: `study_" in answer
    assert len(stub.requests) == 11  # 10 turns, and one more for the refused plan
    results = [e.data for e in store.events(chat.id) if e.kind == "tool_result"]
    refused = results[len(lookups)]
    assert (refused["tool"], refused["ok"]) == ("plan_study", False)
    assert "takes name and options only" in refused["error"]
    assert "ert_adaptive" in refused["error"]
    [*_, last_plan] = [m for m in stub.requests[-1].messages if m.role == "tool"]
    planned = json.loads(last_plan.content)["result"]
    strategies = {k: m["strategy"]["name"] for k, m in planned["methods"].items()}
    assert strategies == {"esmda_approx": "approx", "esmda_ert_esmda": "ert_esmda",
                          "esmda_adaptive_rows": "adaptive_rows",
                          "esmda_ert_adaptive": "ert_adaptive"}  # fmt: skip
    assert all(m.get("engine", "apex") == "apex" for m in planned["methods"].values())
    # A reported issue: energy and truncation were called different settings.
    [equivalent] = planned["equivalent"]
    assert "same setting under two names" in equivalent
    assert "equal in every arm" in equivalent
    validation = service.validate_study(tmp_path / planned["draft"])
    assert validation.valid, validation.errors


def test_the_equivalent_note_names_arms_left_at_their_default() -> None:
    """Second review: approx without energy runs at its default 0.98, so 0.99 elsewhere is
    not 'equal in every arm'."""
    from types import SimpleNamespace

    from apex.agents.chat import _comparison

    def cell(method: str, strategy: dict[str, object]) -> SimpleNamespace:
        config = SimpleNamespace(
            run=SimpleNamespace(seed=1),
            model_dump=lambda mode: {"method": {"name": "esmda", "strategy": strategy}},
        )
        return SimpleNamespace(config=config, replicate=0, case="c", method=method, variant="")

    cells = [cell("a", {"name": "approx"}),
             cell("r", {"name": "adaptive_rows", "options": {"energy": 0.99}}),
             cell("e", {"name": "ert_adaptive", "options": {"truncation": 0.99}})]  # fmt: skip
    [note] = _comparison(cells)["equivalent"]
    assert "not set (the plugin's default) in: c | a" in note
    [note] = _comparison(cells[1:])["equivalent"]
    assert note.endswith("equal in every arm, so the arms are comparable")


def test_the_comparison_gives_the_defaults_an_arm_leaves_out() -> None:
    """A reported issue: chop's options were "(not set)", so the assistant stated its
    defaults (component length_scale, mode data) without having seen them."""
    from types import SimpleNamespace

    from apex.agents.chat import _comparison

    def cell(method: str, dumped: dict[str, object]) -> SimpleNamespace:
        config = SimpleNamespace(run=SimpleNamespace(seed=1), model_dump=lambda mode: dumped)
        return SimpleNamespace(config=config, replicate=0, case="c", method=method, variant="")

    cells = [
        cell("esmda", {"method": {"name": "esmda", "options": {"inflation": [4, 4]},
                                  "strategy": {"name": "approx", "options": {"energy": 0.99}}}}),
        cell("chop", {"method": {"name": "chop", "options": {}}}),
    ]  # fmt: skip
    cells.append(cell("chop2", {"method": {"name": "chop", "options": {}}}))
    cells.append(cell("gn", {"method": {"name": "gn_enrml", "options": {}}}))
    groups = _comparison(cells)["defaults"]
    by_arm = {arm: g["defaults"] for g in groups for arm in g["arms"]}

    chop = by_arm["c | chop"]
    assert chop["method.options.components"] == [{"name": "length_scale", "mode": "data"}]
    assert chop["method.options.hyper_localization"] == "same"
    # The wrapper runs GIES unless told otherwise, with GIES's own options' defaults.
    assert chop["method.options.inner"] == "gies"
    assert chop["method.options.max_trials_per_iteration"] is None
    # The default named, with the values it runs with (#260).
    assert chop["method.strategy"].startswith("not set: the method's own default, rlmmac (")
    assert chop["method.damping"].startswith(
        "not set: the method's own default, rlmmac (initial = 1.0, factor = 2.0, "
    )
    esmda = by_arm["c | esmda"]
    assert "method.options.inflation" not in esmda  # set, so no default
    assert "method.strategy.options.energy" not in esmda
    assert "method.damping" not in esmda  # ES-MDA has no damping
    assert "method.damping" not in by_arm["c | gn"]  # GN-EnRML's schedule is no damping
    # Arms with the same defaults share one entry.
    assert [g["arms"] for g in groups if "c | chop" in g["arms"]] == [["c | chop", "c | chop2"]]


def test_the_comparison_tells_another_methods_option_from_a_default() -> None:
    """A reported issue: against ES-MDA, a CHOP arm's ``inflation`` read "(not set: the
    plugin's default)", though CHOP has no such option, and its strategy "none" (not
    used), though it runs GIES's own RLM-MAC."""
    from types import SimpleNamespace

    from apex.agents.chat import _comparison

    def cell(method: str, dumped: dict[str, object]) -> SimpleNamespace:
        config = SimpleNamespace(run=SimpleNamespace(seed=1), model_dump=lambda mode: dumped)
        return SimpleNamespace(config=config, replicate=0, case="c", method=method, variant="")

    cells = [
        cell("esmda", {"method": {"name": "esmda", "options": {"inflation": [4, 4]},
                                  "strategy": {"name": "approx", "options": {"energy": 0.99}},
                                  "localization": None}}),
        cell("chop", {"method": {"name": "chop", "options": {}, "strategy": None,
                                 "localization": None}}),
        cell("gies", {"method": {"name": "gies", "options": {}, "strategy": None,
                                 "localization": None}}),
    ]  # fmt: skip
    differs = _comparison(cells)["differs"]
    chop, gies, esmda = differs["c | chop"], differs["c | gies"], differs["c | esmda"]
    assert chop["method.options.inflation"] == "not an option of chop"
    assert gies["method.options.inflation"] == "not an option of gies"
    own = chop["method.strategy"]
    assert own.startswith("not set: the method's own default, rlmmac (energy = ")
    assert gies["method.strategy"] == own
    # In full on the section's key only; its entries point there (the review of #260).
    entry = "not set: the method's own default (see method.strategy)"
    assert chop["method.strategy.name"] == chop["method.strategy.options.energy"] == entry
    assert esmda["method.strategy"] == "(set: see its entries)"
    assert esmda["method.strategy.name"] == "approx"
    assert esmda["method.options.inflation"] == [4, 4]
    # A section no arm uses stays "none", and an option the method has but the arm leaves
    # out is still the plugin's default.
    assert "method.localization" not in differs["c | chop"]
    trials = {"max_trials_per_iteration": 3}
    with_option = [cell("g1", {"method": {"name": "gies", "options": trials}}),
                   cell("g2", {"method": {"name": "gies", "options": {}}})]  # fmt: skip
    g2 = _comparison(with_option)["differs"]["c | g2"]
    assert g2["method.options.max_trials_per_iteration"] == "(not set: the plugin's default)"
    # The rule holds for every plugin section: approx's option on an rlmmac arm too.
    strategies = [cell("a", {"method": {"name": "gies", "options": {},
                                        "strategy": {"name": "rlmmac", "options": {}}}}),
                  cell("b", {"method": {"name": "gies", "options": {}, "strategy": {
                      "name": "approx", "options": {"empirical_data_error": True}}}})]  # fmt: skip
    a = _comparison(strategies)["differs"]["c | a"]
    assert a["method.strategy.options.empirical_data_error"] == "not an option of rlmmac"
    # CHOP on ES-MDA has ES-MDA's inflation; an unknown method gets no such label.
    on_esmda = [cell("e", {"method": {"name": "esmda", "options": {"inflation": [4]}}}),
                cell("w", {"method": {"name": "chop", "options": {"inner": "esmda"}}}),
                cell("u", {"method": {"name": "unknown_method", "options": {}}})]  # fmt: skip
    shown = _comparison(on_esmda)["differs"]
    assert shown["c | w"]["method.options.inflation"] == "(not set: the plugin's default)"
    assert shown["c | u"]["method.options.inflation"] == "(not set: the plugin's default)"


def test_the_prompt_describes_es_mda_and_keeps_the_users_length() -> None:
    """A reported issue: "in one sentence" got several, and ES-MDA was called a
    "non-iterative-gradient ensemble smoother"."""
    from apex.agents.chat import SYSTEM

    assert "fixed number of times" in SYSTEM
    assert "no convergence test" in SYSTEM
    assert "in one sentence" in SYSTEM
    assert "same setting under two names" in SYSTEM
    # A reported issue: asked where its draft was, the assistant said it had no path.
    assert "give its path from the result's 'draft' field" in SYSTEM
    # A reported issue: energy 0.99 in both arms was reported as a shared strategy.
    assert "does not make the strategy shared" in SYSTEM


def test_the_prompt_keeps_the_users_arms_and_states_inherited_limits() -> None:
    """Reported issues #244-#246: asked for a baseline against adaptive_per_update, the
    assistant proposed six arms; asked an open question, it tested two localizations at a
    smaller ensemble only; and it compared damping schemes over the case's 2 iterations
    without saying that this limits the comparison."""
    from apex.agents.chat import SYSTEM

    assert "the arms the user names are the study: plan exactly those" in SYSTEM
    assert "never drop an arm the user named" in SYSTEM
    assert "when a localization question names no arms" in SYSTEM
    assert "at the case's own ensemble size" in SYSTEM
    assert "joins only when every arm fits at both sizes" in SYSTEM
    assert "never some arms at one size only" in SYSTEM
    assert (
        "is stated as a limit with an offer to raise it: never raise it unasked in the draft"
        in SYSTEM
    )
    # Its review: the "too small" rule must not override it, and the study rules are not
    # inside the optimization sentence.
    assert "a setting the arms inherit from the case is no reason to revise" in SYSTEM
    assert "which an inherited setting you state as a limit does not make it" in SYSTEM
    general = SYSTEM.index("For every study you plan (history matching or optimization): ")
    assert general < SYSTEM.index("the arms the user names") < SYSTEM.index("is stated as a limit")
    assert SYSTEM.index("When the user asks to optimize") < general
    # The old rule made every localization study add a smaller ensemble.
    assert "smaller ensemble (prior.ensemble_size as a variant" not in SYSTEM


def test_erts_localization_is_planned_only_on_a_case_that_can_carry_it(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A study cannot remove a case's own localization (or [validation]): such a case is
    refused with the reason. On a GIES case the four ES-MDA methods are costed as ES-MDA
    runs, not as the case's."""
    pytest.importorskip("iterative_ensemble_smoother")
    from apex.agents.planner import plan_facts
    from apex.agents.planner import rules as plan_rules

    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(name="localized", ensemble_size=20, method=(
        'name = "esmda"\noptions = { inflation = [2, 2] }\nstrategy = { name = "approx" }\n'
        'localization = { name = "adaptive" }'
    ))  # fmt: skip
    linear_case(name="gies5", ensemble_size=20, method=(
        'name = "gies"\noptions = { max_trials_per_iteration = 4 }\n'
        'strategy = { name = "rlmmac", options = { energy = 0.99 } }\nmax_iterations = 2'
    ))  # fmt: skip
    plan = {t.name: t for t in assistant.planning_tools(store.create("p").id, None)}["plan_study"]
    goal = "compare adaptive localization in APEX and ERT"

    with pytest.raises(ValueError, match=r"has \[method.localization\] \(adaptive\)"):
        plan.handler(plan.arguments(case="localized/config.toml", goal=goal, budget=5000))

    linear_case(name="damped", ensemble_size=20, method=(
        'name = "gies"\noptions = { max_trials_per_iteration = 4 }\n'
        'strategy = { name = "rlmmac", options = { energy = 0.99 } }\nmax_iterations = 2\n'
        'damping = { name = "lm" }'
    ))  # fmt: skip
    with pytest.raises(ValueError, match=r"has \[method.damping\] \(lm\)"):
        plan.handler(plan.arguments(case="damped/config.toml", goal=goal, budget=5000))

    facts = plan_facts(service, tmp_path / "gies5" / "config.toml", budget=1000)
    proposal = plan_rules(facts, goal)
    assert {m.strategy.name for m in proposal.methods.values() if m.strategy} == {
        "approx", "ert_esmda", "adaptive_rows", "ert_adaptive"}  # fmt: skip
    # Four ES-MDA methods of 20 members x (4 steps + the prior): 400 runs a replicate.
    assert proposal.replicates == 1000 // 400


def test_the_assistant_reads_its_draft_and_one_schema_section(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A test session through the web UI (2026-10-05): the assistant could not read its own draft
    back, and the whole schema was cut off before [prior], so it guessed its layout."""
    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan = tools["plan_study"]
    draft = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))[
        "draft"
    ]

    read = tools["read_draft"]
    text = read.handler(read.arguments(draft=draft))
    assert text["text"].startswith("#") or "[study]" in text["text"]
    with pytest.raises(ValueError, match="not a draft of this chat"):
        read.handler(read.arguments(draft="alpha/config.toml"))

    from apex.services.tools import call

    session = ToolSession(service, client="t")
    prior = call("get_schema", {"kind": "history_match", "section": "prior"}, session)
    assert prior["ok"], prior
    assert "ensemble_size" in json.dumps(prior["result"]["schema"])
    assert "$ref" not in json.dumps(prior["result"])
    wrong = call("get_schema", {"kind": "history_match", "section": "nope"}, session)
    assert not wrong["ok"]
    assert "sections:" in str(wrong.get("hint", ""))


def test_a_draft_read_back_keeps_its_inherited_settings_and_says_when_it_needs_approval(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue: reading the study's TOML, the assistant took back the ensemble
    size it had reported (the draft does not spell out what its cells inherit from the
    case), and it proposed 10 runs without saying that launching needs approval (8 runs then;
    20 since the user raised it, 2026-10-11)."""
    from apex.services.settings import ApprovalSection

    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    assert ApprovalSection().max_runs == 20
    # The runs alone decide here: the test service's simulator-run limit is far lower.
    service.approvals.settings = ApprovalSection(max_simulator_runs=10**6)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan, read = tools["plan_study"], tools["read_draft"]

    small = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))
    limits = service.approvals.settings
    assert small["approval"]["limits"] == {
        "runs": limits.max_runs,
        "simulator_runs": limits.max_simulator_runs,
    }
    assert (small["approval"]["runs"], small["approval"]["needed"]) == (1, False)

    variants = [{"name": f"v{k}", "patch": {}, "rationale": "a repeat"} for k in range(3)]
    large = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=5000,
                                        name="many", variants=variants, replicates=7))  # fmt: skip
    assert large["approval"]["runs"] == 21  # 3 variants x 7 replicates, over 20 runs
    assert large["approval"]["needed"] is True
    assert "approval" in large["approval"]["note"]

    back = read.handler(read.arguments(draft=large["draft"]))
    assert "ensemble_size" not in back["text"]  # inherited from the case, not written
    assert back["held_fixed"]["prior.ensemble_size"] == 20
    assert "inherit" in back["inherited"]
    assert back["approval"]["needed"] is True


def test_a_drafts_approval_counts_its_predictions_as_the_service_does(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The service's gate counts a study's prediction runs (``StudyPlan``); the assistant's
    approval must too, or it says no approval is needed for a launch that is refused."""
    from apex.services.settings import ApprovalSection

    _, service, store, assistant = _planning(tmp_path, linear_case, None)
    service.approvals.settings = ApprovalSection(max_simulator_runs=120, max_runs=8)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan, read, revise = tools["plan_study"], tools["read_draft"], tools["revise_draft"]
    planned = plan.handler(plan.arguments(case="alpha/config.toml", goal="compare", budget=100))
    assert (planned["cost"], planned["approval"]["needed"]) == (100, False)

    text = read.handler(read.arguments(draft=planned["draft"]))["text"]
    assert "predictions = []" in text
    assert "max_simulator_runs = 100" in text
    text = text.replace("max_simulator_runs = 100", "max_simulator_runs = 500")
    text = text.replace("predictions = []", 'predictions = [{ prediction = { name = "future", '
                        'ensembles = ["posterior", "prior"] } }]')  # fmt: skip
    revised = revise.handler(revise.arguments(draft=planned["draft"], text=text))

    assert revised["approval"]["simulator_runs"] > 120  # the predictions' members too
    assert revised["approval"]["needed"] is True


def test_a_diagnosis_patch_becomes_a_run_draft_to_launch(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (2026-10-05): the assistant could not turn a diagnosis's
    recommended change into a configuration to run; draft_run writes it, validated."""
    case, service, store, assistant = _planning(tmp_path, linear_case, None)
    composition.start(case.config)
    chat = store.create("revise")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    run = str(case.directory / "runs" / "alpha")
    patch = {"method": {"options": {"inflation": [2, 2]}}}

    made = tools["draft_run"].handler(tools["draft_run"].arguments(run=run, patch=patch))

    draft = tmp_path / made["draft"]
    text = draft.read_text()
    assert made["kind"] == "run"
    assert tomllib.loads(text)["method"]["options"]["inflation"] == [2, 2]
    assert str(tmp_path) not in text  # no server path in the draft
    assert service.validate(draft).valid
    propose = tools["propose_action"]
    propose.handler(propose.arguments(draft=made["draft"], summary="run it"))
    assert store.proposals(chat.id)[-1].kind == "launch_run"
    with pytest.raises(ValueError, match="does not apply"):
        tools["draft_run"].handler(tools["draft_run"].arguments(
            run=run, patch={"method": {"options": {"inflation": [5, 5]}}}))  # fmt: skip


def test_healthy_needs_a_diagnosis_and_a_report_can_be_saved(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issues #7 and #8 (2026-10-05): the monitor said healthy without a diagnosis; a
    report could not be kept."""
    case, service, store, assistant = _planning(tmp_path, linear_case, None)
    composition.start(case.config)
    chat = store.create("check")
    tools = {t.name: t for t in assistant.local_tools(None, chat.id)}
    run = str(case.directory / "runs" / "alpha")

    checked = tools["monitor_run"].handler(tools["monitor_run"].arguments(run=run))
    saved = tools["write_report"].handler(tools["write_report"].arguments(run=run, save=True))

    assert "diagnosis" in checked
    assert checked["healthy"] == (checked["report"]["status"] == "healthy"
                                  and checked["diagnosis"]["verdict"] == "healthy")  # fmt: skip
    assert saved["saved"].endswith(".md")
    shown = service.workspace().read(saved["saved"])
    assert shown.text == saved["markdown"]
    folder = saved["saved"].rsplit("/", 1)[0]
    listed = [e.path for e in service.workspace().list(folder)]
    assert saved["saved"] in listed


def test_no_answer_speaks_to_the_request_not_a_command_list() -> None:
    """G6(d), 2026-10-05: a timeout's answer listed commands."""
    from apex.agents.chat import _no_answer

    text = _no_answer(
        "the provider failed: timeout", "Plan a study comparing ssim",
        [{"tool": "plan_study", "ok": True}], ["UI/owner/drafts/c/study_x.toml"],
    )  # fmt: skip

    assert "Plan a study comparing ssim" in text
    assert "plan_study (done)" in text
    assert "study_x.toml" in text
    assert "apex agent" not in text


def test_a_guests_run_draft_stays_in_the_workspace_and_their_own_runs(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """draft_run for a web-UI user: never another user's run; a path outside the workspace
    is refused by the option's name, never shown (review of 2026-10-05)."""
    from apex.services.access import Access, load_access

    case, service, store, assistant = _planning(tmp_path, linear_case, None)
    composition.start(case.config)
    (service.root / "ui_access.toml").write_text(
        '[workspace]\nfolders = ["alpha"]\n[agents]\nnames = ["planner"]\n'
        '[plugins]\nmethods = ["*"]\nupdate_strategies = ["*"]\nsimulators = ["*"]\n'
        'executors = ["*"]\npriors = ["*"]\nmetrics = ["*"]\n'
    )
    access = Access(load_access(service.root), service.jobs.all_records())
    chat = store.create("g", owner="guest:guest")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, access)}
    owners = str(case.directory / "runs" / "alpha")

    with pytest.raises(KeyError, match="no run"):  # the owner's run is not theirs
        tools["draft_run"].handler(tools["draft_run"].arguments(run=owners, patch={}))

    from apex.agents.chat import _relative_paths

    roots = [r.resolve() for r in service.workspace().roots]
    inside = str(case.directory / "observations.csv")
    folder = case.directory / "UI"
    assert _relative_paths({"f": inside}, folder, roots, ui=True) == {
        "f": "../observations.csv"}  # fmt: skip
    with pytest.raises(ValueError, match=r"simulator\.options\.forward_file names a file outside"):
        _relative_paths({"simulator": {"options": {"forward_file": "/etc/passwd"}}}, folder,
                        roots, ui=True)  # fmt: skip
    kept = _relative_paths({"f": "/etc/passwd"}, folder, roots, ui=False)
    assert kept == {"f": "/etc/passwd"}  # the owner's own server


def test_the_assistant_finds_a_cases_configurations_and_drafts_a_run_from_one(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The user's 2026-10-06 chat: "create a config file to run history matching in the
    Norne case, using rlmmac and adaptive_prior". The assistant could not find
    Norne/configs/*.toml (no tool lists the workspace) and draft_run took only a run."""
    case, service, store, assistant = _planning(tmp_path, linear_case, None)
    configs = tmp_path / "Norne" / "configs"
    shutil.copytree(case.directory, configs)
    (configs / "config.toml").rename(configs / "gies.toml")
    text = case.config.read_text()
    (configs / "study.toml").write_text('[study]\nname = "s"\n[study.cases]\nn = "gies.toml"\n')
    composition.start(case.config)  # alpha/runs/alpha: run output, not a case
    other = tmp_path / "UI" / "guests" / "bob"
    other.mkdir(parents=True)
    (other / "norne_copy.toml").write_text(text)  # another web-UI user's: never listed
    chat = store.create("norne")
    tools = {t.name: t for t in assistant.local_tools(None, chat.id)}
    find = tools["find_cases"]

    found = find.handler(find.arguments(query="NORNE"))["configurations"]

    assert {c["path"]: c["kind"] for c in found} == {
        "Norne/configs/gies.toml": "run", "Norne/configs/study.toml": "study"}  # fmt: skip
    assert found[0]["method"] == "esmda"
    everything = [c["path"] for c in find.handler(find.arguments())["configurations"]]
    assert not [p for p in everything if "/runs/" in p or p.startswith("UI/")]
    with pytest.raises(KeyError, match="no folder"):
        find.handler(find.arguments(folder="nowhere"))
    with pytest.raises(KeyError, match="no folder"):  # another web-UI user's, by its name
        find.handler(find.arguments(folder="UI/guests/bob"))
    with pytest.raises(KeyError, match="no case"):
        tools["case_info"].handler(
            tools["case_info"].arguments(case="UI/guests/bob/norne_copy.toml")
        )

    planning = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    patch = {"method": {"localization": {"name": "adaptive_prior"}}}
    made = planning["draft_run"].handler(
        planning["draft_run"].arguments(case="Norne/configs/gies.toml", patch=patch, name="n1")
    )
    draft = tmp_path / made["draft"]
    assert tomllib.loads(draft.read_text())["method"]["localization"]["name"] == "adaptive_prior"
    assert str(tmp_path) not in draft.read_text()
    assert service.validate(draft).valid
    with pytest.raises(ValueError, match="not a history-matching run"):
        planning["draft_run"].handler(
            planning["draft_run"].arguments(case="Norne/configs/study.toml", patch={}))  # fmt: skip
    with pytest.raises(ValueError, match="either run or case"):
        planning["draft_run"].arguments(patch={})


def test_a_guest_finds_only_the_cases_of_their_folders(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    from apex.services.access import Access, load_access

    case, service, store, assistant = _planning(tmp_path, linear_case, None)
    hidden = tmp_path / "Norne" / "configs"
    hidden.mkdir(parents=True)
    (hidden / "gies.toml").write_text(case.config.read_text())
    (service.root / "ui_access.toml").write_text('[workspace]\nfolders = ["alpha"]\n')
    access = Access(load_access(service.root), service.jobs.all_records())
    chat = store.create("g", owner="guest:guest")
    tools = {t.name: t for t in assistant.local_tools(access, chat.id)}
    find = tools["find_cases"]

    paths = [c["path"] for c in find.handler(find.arguments())["configurations"]]

    assert paths == ["alpha/config.toml"]
    with pytest.raises(KeyError, match="no folder"):
        find.handler(find.arguments(folder="Norne"))
    planning = {t.name: t for t in assistant.planning_tools(chat.id, access)}
    with pytest.raises(KeyError, match="no case"):
        planning["draft_run"].handler(
            planning["draft_run"].arguments(case="Norne/configs/gies.toml", patch={}))  # fmt: skip


# ---------------------------------------------------------------------- the Stop button


def _join_turns() -> None:
    """Wait for the chat's background turns (daemon threads) to end."""
    import threading

    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.daemon:
            thread.join(10)


def test_stop_ends_a_turn_at_once_and_drops_what_the_stuck_turn_does_later(
    tmp_path: Path,
) -> None:
    """A model call that cannot be cut short (an HTTP request; here a call that waits):
    the stop ends the turn and releases the chat at once. The call then returns a tool call;
    the stopped turn runs nothing more and writes nothing more (no fallback, tool or answer
    event after "stopped"), while a new message, asked meanwhile, is answered."""
    import threading

    from apex.agents.chat import STOPPED_ANSWER
    from apex.agents.providers.stub import StubRequest

    called, release = threading.Event(), threading.Event()

    def stuck(_: StubRequest) -> Completion:
        called.set()
        release.wait(30)
        return Completion(tool_calls=(ToolCall("c1", "list_agents", {}),))

    stub = StubProvider([stuck, _answer("Here again.")])
    assistant = _assistant(tmp_path, stub)
    chat = assistant.store.create("t")
    assistant.start(chat.id, "Plan a big study")
    assert called.wait(30)
    assert assistant.stop(chat.id) is True
    assert not assistant.store.get(chat.id).busy  # released at once
    assert [(t.role, t.content) for t in assistant.store.turns(chat.id)] == [
        ("user", "Plan a big study"),
        ("assistant", STOPPED_ANSWER),
    ]
    assert assistant.stop(chat.id) is False  # nothing runs any more
    stopped_at = len(assistant.store.events(chat.id))
    # A new message while the stopped turn's thread is still waiting.
    assert assistant.run(chat.id, "Hello?") == "Here again."
    release.set()  # the stuck call returns a tool call now
    _join_turns()
    late = [e.kind for e in assistant.store.events(chat.id)][stopped_at:]
    assert "list_agents" not in str(assistant.store.events(chat.id)[stopped_at:])
    assert late.count("answer") == 1  # only the new message's
    assert "fallback" not in late
    assert [t.content for t in assistant.store.turns(chat.id)][-1] == "Here again."
    assert not assistant.store.get(chat.id).busy


def test_a_stop_pressed_before_the_turn_starts_is_kept(tmp_path: Path) -> None:
    """Review: a Stop pressed before the turn's thread had begun was lost, and the whole
    turn ran (model calls, tools), writing the user's message after "Stopped"."""
    import threading

    from apex.agents.chat import STOPPED_ANSWER, ChatAssistant

    gate = threading.Event()
    stub = StubProvider([_answer("This should never be asked for.")])

    def provider() -> StubProvider:
        gate.wait(30)  # the turn's thread waits here, before any model call
        return stub

    service = ApexService(tmp_path / "service")
    assistant = ChatAssistant(service, ChatStore(service.root), provider)
    chat = assistant.store.create("t")
    assistant.start(chat.id, "Plan a study")
    assert assistant.stop(chat.id) is True
    gate.set()
    _join_turns()
    assert stub.requests == []  # no model call after the stop
    assert [(t.role, t.content) for t in assistant.store.turns(chat.id)] == [
        ("user", "Plan a study"),
        ("assistant", STOPPED_ANSWER),
    ]


def test_stop_without_a_running_turn_does_nothing(tmp_path: Path) -> None:
    assistant = _assistant(tmp_path, StubProvider([]))
    chat = assistant.store.create("t")
    assert assistant.stop(chat.id) is False
    assert assistant.store.turns(chat.id) == []


def test_a_turn_that_cannot_start_leaves_the_chat_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: a failure between taking the chat and starting its thread left it busy."""
    import threading

    assistant = _assistant(tmp_path, StubProvider([_answer("ok")]))
    chat = assistant.store.create("t")

    def broken(self: threading.Thread) -> None:
        raise RuntimeError("no thread")

    monkeypatch.setattr(threading.Thread, "start", broken)
    with pytest.raises(RuntimeError, match="no thread"):
        assistant.start(chat.id, "Hello")
    monkeypatch.undo()
    assert not assistant.store.get(chat.id).busy
    assert assistant.run(chat.id, "Hello again") == "ok"


def test_compared_algorithms_keep_their_own_base_and_say_what_differs(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue: chop against an ES-MDA case ran up to 10 iterations on its own
    defaults against ES-MDA's 4 steps and the case's strategy, and was proposed without a
    word. Each algorithm stays its own base (the user's rule): what differs comes
    back, for the assistant to state, propose, and offer to match (a reported issue)."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="esmda4", ensemble_size=20,
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "approx", options = { energy = 0.98 } }',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    result: dict[str, Any] = plan.handler(
        plan.arguments(
            case="esmda4/config.toml", goal="compare chop with the case's algorithm",
            budget=20000, name="chop_vs_esmda",
            methods={"esmda": {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]},
                               "strategy": {"name": "approx", "options": {"energy": 0.98}}},
                     "chop": {"name": "chop"}},
        )
    )  # fmt: skip
    # Each algorithm is its own base: chop keeps its defaults; how it differs
    # comes back, for the assistant to state, propose, and offer to match (a reported issue).
    assert "max_iterations" not in result["methods"]["chop"]
    # Its own strategy, with only the setting both strategies take made the case's (#256).
    assert result["methods"]["chop"]["strategy"] == {"name": "rlmmac", "options": {"energy": 0.98}}
    assert any(m.startswith("chop: rlmmac energy")
               and m.endswith("-> 0.98, as the case's algorithm")
               for m in result["matched"])  # fmt: skip
    assert "chop runs to its own iteration limit, the case 4 steps" in result["differences"]
    assert any(d.startswith("chop's strategy is rlmmac; the case's is approx")
               for d in result["differences"])  # fmt: skip
    # A reported issue: asked to propose, the assistant proposes the card and states what
    # differs; matching is an offer, not a stop.
    assert "propose_action(draft=...) as usual" in result["next"]
    assert "offer to match" in result["next"]
    assert "state each to the user" in result["next"]
    assert "before propose_action" not in result["next"]  # the old stop (#174)
    assert result["methods"]["esmda"]["strategy"]["name"] == "approx"
    assert not any(d.startswith("esmda") for d in result["differences"])


def test_arms_the_model_set_stay_and_a_non_iterative_arms_schedule_is_named(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue, the review's cases: an iteration limit the model gave is kept (and
    named as a difference), and ES-MDA's three steps against a GIES case of 2 iterations
    are named too, rather than passing silently."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="gies2", ensemble_size=20,
        method='name = "gies"\noptions = { max_trials_per_iteration = 4 }\nmax_iterations = 2',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    result: dict[str, Any] = plan.handler(
        plan.arguments(
            case="gies2/config.toml", goal="compare algorithms", budget=20000, name="algos",
            methods={"gies": {"name": "gies", "options": {"max_trials_per_iteration": 4},
                              "max_iterations": 2},
                     "chop": {"name": "chop", "max_iterations": 3},
                     "esmda": {"name": "esmda", "options": {"inflation": [3, 3, 3]}}},
        )
    )  # fmt: skip
    assert result["methods"]["chop"]["max_iterations"] == 3
    assert "chop runs up to 3 iterations, the case 2 iterations" in result["differences"]
    assert "esmda runs 3 steps, the case 2 iterations" in result["differences"]


def test_a_compared_arm_takes_the_cases_energy_unless_it_sets_its_own(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue, #256: ES-MDA at the case's energy 0.99 was compared with GN-EnRML
    at its default 0.95, an avoidable confound. An arm whose strategy takes ``energy`` and
    leaves it unset runs at the case's, and says so ('matched'); an arm that sets its own
    keeps it, and every arm keeps its own strategy. #252: the cost comes as a
    line to quote, its arithmetic done."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="esmda99", ensemble_size=20,
        method='name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    result: dict[str, Any] = plan.handler(
        plan.arguments(
            case="esmda99/config.toml", goal="compare gn_enrml with the case's algorithm",
            budget=20000, name="gn_vs_esmda", replicates=2,
            methods={"esmda": {"name": "esmda", "options": {"inflation": [2, 2]},
                               "strategy": {"name": "approx", "options": {"energy": 0.99}}},
                     "gn": {"name": "gn_enrml", "max_iterations": 2},
                     "lm": {"name": "lm_enrml", "max_iterations": 2,
                            "strategy": {"name": "approx", "options": {"energy": 0.9}}}},
        )
    )  # fmt: skip
    assert result["methods"]["gn"]["strategy"] == {"name": "approx", "options": {"energy": 0.99}}
    assert "gn: approx energy 0.95 -> 0.99, as the case's algorithm" in result["matched"]
    assert result["methods"]["lm"]["strategy"]["options"] == {"energy": 0.9}  # its own
    assert not any(m.startswith(("lm:", "esmda:")) for m in result["matched"])
    assert not any(d.startswith("gn's") for d in result.get("differences", []))
    assert "'matched' lists" in result["next"]
    # #252: the total is the plan's, its arithmetic spelled out per arm.
    assert result["cost_line"].endswith(f"in total {result['cost']} simulator runs at most")
    assert "esmda | baseline: 2 x " in result["cost_line"]
    assert "quote 'cost_line'" in result["next"]


def test_the_energy_matched_is_what_the_cases_own_arm_runs(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #256: an arm of the case's method given without a strategy runs the
    method's default (an entry replaces every algorithm field), so the others
    match that, not the case file's 0.99; a strategy without energy is left alone."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="esmda99b", ensemble_size=20,
        method='name = "esmda"\noptions = { inflation = [2, 2] }\n'
        'strategy = { name = "approx", options = { energy = 0.99 } }',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    result: dict[str, Any] = plan.handler(
        plan.arguments(
            case="esmda99b/config.toml", goal="compare gn_enrml with the case's algorithm",
            budget=20000, name="gn_vs_esmda_default",
            methods={"esmda": {"name": "esmda", "options": {"inflation": [2, 2]}},
                     "gn": {"name": "gn_enrml", "max_iterations": 2},
                     "sub": {"name": "lm_enrml", "max_iterations": 2,
                             "strategy": {"name": "subspace2"}}},
        )
    )  # fmt: skip
    # The case's arm runs approx at its default 0.98: gn (0.95) is matched to that.
    assert result["methods"]["gn"]["strategy"]["options"] == {"energy": 0.98}
    assert result["methods"]["sub"]["strategy"] == {"name": "subspace2"}
    assert not any(m.startswith("sub:") for m in result["matched"])


def test_a_cost_line_multiplies_equal_arms(tmp_path: Path, linear_case: CaseFactory) -> None:
    """#252: arms of one cost are said as arms x replicates x runs = the plan's total."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"},
                {"name": "loc", "patch": {"method": {"localization": {"name": "adaptive"}}},
                 "rationale": "localized"}]  # fmt: skip
    result: dict[str, Any] = plan.handler(
        plan.arguments(case="alpha/config.toml", goal="does localization help", budget=2000,
                       name="loc", variants=variants, replicates=3)
    )  # fmt: skip
    per_run = result["cost"] // 6
    assert result["cost_line"] == (f"2 arms x 3 replicates x {per_run} simulator runs = "
                                   f"{result['cost']} simulator runs at most")  # fmt: skip


def test_a_comparison_with_the_cases_own_algorithm_is_planned_not_asked() -> None:
    """Reported issues #254/#255: asked to compare ES-MDA with the case's own algorithm,
    also ES-MDA, the assistant stopped after case_info and asked, so nothing was costed or
    checked against the cap. The prompt makes plan_study come first."""
    assert "always ends in a plan_study call before you ask anything" in SYSTEM
    assert "unless a rule here says to ask first" in SYSTEM
    assert "still plan it" in SYSTEM
    assert "seed replicates" in SYSTEM
    assert "never instead of planning" in SYSTEM


def test_a_configuration_with_plugin_names_as_strings_is_described(tmp_path: Path) -> None:
    """``find_cases`` lists a folder whatever its files hold (the release tester's review).

    A section written as a bare name (``method = "es"``, ``optimizer = "enopt"``) made
    ``_describe_config`` call ``.get`` on a string, which broke the whole listing.
    """
    from apex.agents.chat import _describe_config

    run = tmp_path / "run.toml"
    run.write_text('[run]\nname = "x"\nmethod = "es"\nsimulator = "linear"\n')
    # (method inside [run] is not a top-level method; the next file has one.)
    flat = tmp_path / "flat.toml"
    flat.write_text('method = "es"\nsimulator = ["linear"]\n[run]\nname = "x"\n')
    optimization = tmp_path / "opt.toml"
    optimization.write_text('optimizer = "enopt"\n')

    assert _describe_config(run)["kind"] == "other"
    assert _describe_config(flat) == {"kind": "run", "method": "es", "simulator": None}
    assert _describe_config(optimization) == {"kind": "optimization", "optimizer": "enopt"}


def test_a_wrappers_inner_method_is_not_reported_as_left_at_its_default() -> None:
    """The review of release-tester 41: ``inner`` in CHOP's option model must not make the
    assistant say a given inner method is the default ``gies``."""
    from apex.agents.chat import _method_defaults

    given = _method_defaults({"name": "chop", "options": {"inner": "lm_enrml"}})
    assert "method.options.inner" not in given
    left_out = _method_defaults({"name": "chop"})
    assert left_out.get("method.options.inner") == "gies"


def _opt_cell(method: str, optimizer: dict[str, object], variant: str = "") -> object:
    """A study cell whose configuration dumps to ``optimizer`` (as the comparison reads it)."""
    from types import SimpleNamespace

    dumped = {"optimizer": optimizer}
    config = SimpleNamespace(optimization=SimpleNamespace(seed=1), model_dump=lambda mode: dumped)
    return SimpleNamespace(config=config, replicate=0, case="c", method=method, variant=variant)


def test_unmatched_budgets_are_reported_beyond_the_tolerance() -> None:
    """Reported issues (2026-10-20): optimizers compared on 12 and 29 evaluations; the arms'
    costs and the mismatch are said by code, not left to the model."""
    from apex.agents.chat import _comparison, _Matching

    cells = [_opt_cell("enopt", {"name": "enopt", "options": {"ne": 4}}),
             _opt_cell("pso", {"name": "pso", "options": {"pop_size": 4}})]  # fmt: skip
    costs: list[dict[str, int | None] | None] = [
        {"simulator_runs": 12, "evaluations": 12},
        {"simulator_runs": 29, "evaluations": 29},
    ]
    result = _comparison(cells, _Matching(costs, {}, {}))
    assert result["arm_costs"]["c | enopt"]["evaluations"] == 12
    [budget] = [u for u in result["unmatched"] if u.startswith("budget")]
    assert "evaluations differ (c | enopt: 12, c | pso: 29)" in budget
    close: list[dict[str, int | None] | None] = [
        {"simulator_runs": 100, "evaluations": 100},
        {"simulator_runs": 90, "evaluations": 90},
    ]
    assert not [u for u in _comparison(cells, _Matching(close, {}, {})).get("unmatched", [])
                if u.startswith("budget")]  # fmt: skip


def test_a_setting_both_arms_take_but_set_differently_is_reported() -> None:
    """A reported issue (2026-10-20): alpha set for enopt only, smcopt at its own default; an
    option no arm sets (both at their plugins' defaults) is not reported. Set to smcopt's
    own default (0.1), alpha is the same in both arms (compared by value, a reported issue)."""
    from apex.agents.chat import _comparison, _Matching

    cells = [_opt_cell("enopt", {"name": "enopt", "options": {"alpha": 0.3}}),
             _opt_cell("smcopt", {"name": "smcopt", "options": {}})]  # fmt: skip
    unmatched = _comparison(cells, _Matching([None, None], {}, {}))["unmatched"]
    [alpha] = [u for u in unmatched if "optimizer.options.alpha" in u]
    assert "c | enopt: 0.3" in alpha
    assert "c | smcopt: 0.1" in alpha
    assert not [u for u in unmatched if "optimizer.options.ftol" in u]  # nobody set it

    cells[0] = _opt_cell("enopt", {"name": "enopt", "options": {"alpha": 0.1}})
    result = _comparison(cells, _Matching([None, None], {}, {}))
    assert not [u for u in result.get("unmatched", []) if "optimizer.options.alpha" in u]


def test_the_case_s_own_algorithm_changed_from_the_case_is_reported_unless_a_variant_does_it() -> (
    None
):
    """A reported issue (2026-10-20): the user's EnOpt went from ne 4, maxiter 3 to 6 and 4
    without a word. A variant's own change (its patch) is intended, so not reported."""
    from apex.agents.chat import _comparison, _Matching

    case = {"optimizer": {"name": "enopt", "options": {"ne": 4, "maxiter": 3}}}
    cells = [_opt_cell("enopt", {"name": "enopt", "options": {"ne": 6, "maxiter": 4}}),
             _opt_cell("ga", {"name": "ga", "options": {"maxiter": 6}})]  # fmt: skip
    unmatched = _comparison(cells, _Matching([None, None], {"c": case}, {}))["unmatched"]
    [changed] = [u for u in unmatched if u.startswith("changed from the case")]
    assert "optimizer.options.ne 4 -> 6" in changed
    assert "optimizer.options.maxiter 3 -> 4" in changed
    assert "c | ga" not in changed  # another algorithm: its settings are its own
    # #234: the case's values are the user's baseline, changed only on their word.
    assert "ask the user before changing them" in changed
    from apex.agents.chat import SYSTEM

    assert "keep the case's own optimizer as the case configures it (the user's baseline" in (
        SYSTEM
    )

    varied = [_opt_cell("enopt", {"name": "enopt", "options": {"ne": 6, "maxiter": 3}}, "big"),
              _opt_cell("ga", {"name": "ga", "options": {}}, "big")]  # fmt: skip
    intended = {"big": frozenset({"optimizer.options.ne"})}
    result = _comparison(varied, _Matching([None, None], {"c": case}, intended))
    assert not [u for u in result.get("unmatched", []) if u.startswith("changed from the case")]


def test_a_variant_s_own_budget_is_not_an_unmatched_budget() -> None:
    """A variant changing the ensemble size (a smaller ensemble) costs less by design: budgets
    are compared between the methods of one case and variant only."""
    from apex.agents.chat import _comparison, _Matching

    cells = [_opt_cell("enopt", {"name": "enopt", "options": {"ne": 4}}, "full"),
             _opt_cell("enopt", {"name": "enopt", "options": {"ne": 2}}, "half")]  # fmt: skip
    costs: list[dict[str, int | None] | None] = [
        {"simulator_runs": 40, "evaluations": 40},
        {"simulator_runs": 20, "evaluations": 20},
    ]
    intended = {"half": frozenset({"optimizer.options.ne"})}
    result = _comparison(cells, _Matching(costs, {}, intended))
    assert not [u for u in result.get("unmatched", []) if u.startswith("budget")]


def test_plan_study_reports_unequal_history_matching_budgets_but_not_a_variant_s(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Through the real plan_study: two methods with different numbers of steps cost
    differently (reported); a variant halving the ensemble does so by design (not)."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    methods = {"four": {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]}},
               "one": {"name": "es", "options": {}}}  # fmt: skip
    result = plan.handler(plan.arguments(case="alpha/config.toml", goal="steps", budget=2000,
                                         methods=methods, name="steps"))  # fmt: skip
    assert all(isinstance(c["simulator_runs"], int) for c in result["arm_costs"].values())
    assert [u for u in result["unmatched"] if u.startswith("budget")]

    half = [
        {"name": "baseline", "patch": {}, "rationale": "as configured"},
        {"name": "half", "patch": {"prior": {"ensemble_size": 10}}, "rationale": "smaller"},
    ]
    varied = plan.handler(plan.arguments(case="alpha/config.toml", goal="size", budget=2000,
                                         name="size", variants=half))  # fmt: skip
    assert not [u for u in varied.get("unmatched", []) if u.startswith("budget")]


def test_the_assistant_converts_a_deck_and_gives_it_a_prior(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """In the chat: a deck imported as a run draft to launch, its prior
    redrawn, and a synthetic case created (its truth run left to the user)."""
    from apex.testing import write_small_deck

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    write_small_deck(tmp_path / "decks")
    chat = store.create("bring a deck")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}

    made = tools["convert_case"].handler(tools["convert_case"].arguments(
        source="decks/CASE.DATA", name="small"))  # fmt: skip
    prior = tools["draft_prior"].handler(tools["draft_prior"].arguments(draft=made["draft"]))
    created = tools["create_case"].handler(tools["create_case"].arguments(
        description={}, name="box"))  # fmt: skip
    propose = tools["propose_action"]
    propose.handler(propose.arguments(draft=made["draft"], summary="run the deck"))

    assert made["valid"], made["errors"]
    assert made["state"] == ["PERMX (log)", "PORO (none)"]
    assert str(tmp_path) not in (tmp_path / made["draft"]).read_text()
    assert prior["valid"], (prior["errors"], prior["qc"])
    assert prior["draft"].endswith("config_prior.toml")
    assert not created["valid"]
    assert "truth" in created["next"]
    assert store.proposals(chat.id)[-1].kind == "launch_run"


def test_the_converters_tools_are_offered_only_with_their_agents(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of 2026-10-24: a guest whose list names the planner, not the
    case-converter or the geomodeler, is not offered their tools in the chat."""
    from apex.services.access import Access, UIAccess

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("guest")
    planner_only = Access(UIAccess.model_validate({"agents": {"names": ["planner"]}}), [])

    names = {t.name for t in assistant.planning_tools(chat.id, planner_only)}

    assert {"plan_study", "draft_run"} <= names
    assert not {"convert_case", "create_case", "draft_prior"} & names


def test_the_assistant_never_copies_files_a_deck_includes_from_elsewhere(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of 2026-10-24: convert_case picks its skill itself; the deck's includes
    are held to the chat user's folders all the same."""
    from apex.services.access import Access, UIAccess
    from apex.testing import write_small_deck

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    deck = write_small_deck(tmp_path / "decks")
    secret = tmp_path / "private" / "secret.inc"
    secret.parent.mkdir()
    secret.write_text("-- not yours\n")
    deck.write_text(deck.read_text().replace("PROPS\n", f"PROPS\nINCLUDE\n '{secret}' /\n", 1))
    agents = {"names": ["planner", "case-converter"]}
    rules = UIAccess.model_validate({"workspace": {"folders": ["decks"]}, "agents": agents})
    chat = store.create("bring a deck")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, Access(rules, []))}

    with pytest.raises(ValueError, match="not a file of your folders"):
        tools["convert_case"].handler(tools["convert_case"].arguments(
            source="decks/CASE.DATA", name="evil"))  # fmt: skip
    assert not list(tmp_path.rglob("drafts/**/secret.inc"))


def test_an_unset_damping_names_the_methods_default_scheme() -> None:
    """A reported issue (#260): asked to compare LM damping against the GIES case's own, the
    plan said only "not set: the method's own default", and nobody could tell what LM was
    compared with. The default scheme is named, with the values it runs with."""
    from types import SimpleNamespace

    from apex.agents.chat import _comparison

    def cell(variant: str, method: dict[str, object]) -> SimpleNamespace:
        dumped = {"method": method}
        config = SimpleNamespace(run=SimpleNamespace(seed=1), model_dump=lambda mode: dumped)
        return SimpleNamespace(config=config, replicate=0, case="c", method=method["name"],
                               variant=variant)  # fmt: skip

    cells = [
        cell("", {"name": "gies", "options": {}, "damping": None}),
        cell("lm", {"name": "gies", "options": {}, "damping": {"name": "lm", "options": {}}}),
    ]
    comparison = _comparison(cells)
    differs = comparison["differs"]

    base = differs["c | gies"]["method.damping"]
    assert base == (
        "not set: the method's own default, rlmmac (initial = 1.0, factor = 2.0, minimum = "
        "none, data_mismatch_tolerance = 0.01, reduction = 0.9, converge_on_rejection = "
        "false, max_stalled_rejections = 3, confirm_convergence = false)"
    )
    assert differs["c | gies | lm"]["method.damping.name"] == "lm"
    # Its entries point to it, and defaults does not repeat it (the review of #260).
    assert differs["c | gies"]["method.damping.name"] == (
        "not set: the method's own default (see method.damping)"
    )
    shown_defaults = [g["defaults"] for g in comparison.get("defaults", [])]
    assert not any("method.damping" in d for d in shown_defaults)
    # The case's [method.schedule] options are the default scheme's, and shown as such.
    scheduled = [cell("", {"name": "gies", "options": {}, "damping": None,
                           "schedule": {"reduction": 0.5}}),
                 cell("lm", {"name": "gies", "options": {},
                             "damping": {"name": "lm", "options": {}}})]  # fmt: skip
    shown = _comparison(scheduled)["differs"]["c | gies"]["method.damping"]
    assert "reduction = 0.5" in shown
    # LM-EnRML's own is lm; a method with no damping has no such entry (ES-MDA).
    lm = [cell("", {"name": "lm_enrml", "options": {}, "damping": None}),
          cell("r", {"name": "lm_enrml", "options": {},
                     "damping": {"name": "rlmmac", "options": {}}})]  # fmt: skip
    assert _comparison(lm)["differs"]["c | lm_enrml"]["method.damping"].startswith(
        "not set: the method's own default, lm (initial = 100.0, factor = 5.0, "
    )


def test_the_assistant_is_told_to_name_the_methods_default() -> None:
    """#260: the prompt asked the assistant not to name a value for a method's own default."""
    from apex.agents.chat import SYSTEM

    assert "so do not name a value for it" not in SYSTEM
    assert "name them when you say what an arm is compared with" in SYSTEM


def test_a_strategy_that_localizes_by_itself_shows_its_localization() -> None:
    """A reported issue (#261): in an approx against adaptive_rows study, both arms showed
    no localization, and the assistant's caveat about it was judged wrong, though
    adaptive_rows is ERT's per-row adaptive localization."""
    from types import SimpleNamespace

    from apex.agents.chat import SYSTEM, _comparison

    def cell(method: str, strategy: str, localization: object = None) -> SimpleNamespace:
        dumped = {"method": {"name": "esmda", "options": {"inflation": [4, 4]},
                             "strategy": {"name": strategy, "options": {}},
                             "localization": localization}}  # fmt: skip
        config = SimpleNamespace(run=SimpleNamespace(seed=1), model_dump=lambda mode: dumped)
        return SimpleNamespace(config=config, replicate=0, case="c", method=method, variant="")

    differs = _comparison([cell("approx", "approx"), cell("rows", "adaptive_rows")])["differs"]

    assert differs["c | approx"]["method.localization"] == "none"
    within = "within the strategy adaptive_rows (it localizes by itself)"
    assert differs["c | rows"]["method.localization"] == within
    # An entry of the section reads the same (the review of #261), against an arm that sets
    # one.
    against = _comparison(
        [
            cell("rows", "adaptive_rows"),
            cell("dist", "local_analysis", {"name": "distance", "options": {}}),
        ]
    )
    assert against["differs"]["c | rows"]["method.localization.name"] == within  # fmt: skip
    # Every local analysis and ERT's localizing smoothers say so; a given section is kept.
    for strategy in ("local_analysis", "local_rlmmac", "ert_adaptive", "ert_localized"):
        shown = _comparison([cell("a", "approx"), cell("s", strategy)])["differs"]["c | s"]
        assert (
            shown["method.localization"]
            == f"within the strategy {strategy} (it localizes by itself)"
        )
    given = {"name": "distance", "options": {}}
    held = _comparison([cell("a", "local_analysis", given), cell("b", "local_analysis", given)])
    assert held["held_fixed"]["method.localization.name"] == "distance"
    assert "caveat you add about a plan concerns a setting the plan shows" in SYSTEM


def test_the_assistant_takes_a_runs_stop_reason_from_the_tools() -> None:
    """#259: the assistant stated a cause of non-convergence no tool result gave."""
    from apex.agents.chat import SYSTEM

    assert "Why a run stopped or did not converge comes only from a tool's result" in SYSTEM


def test_a_cut_section_key_leaves_its_entries_the_full_default() -> None:
    """The re-review of #260: when MAX_DIFFERENCES cuts the section's own key, its entries
    give the full default rather than point to a key the plan does not show."""
    from apex.agents.chat import _shown_value

    flat = {"method.damping": None}
    defaults = {"method.damping": "not set: the method's own default, rlmmac (factor = 2.0)"}
    shown = _shown_value(flat, "method.damping.name", defaults, None, ["method.damping.name"])
    assert shown == defaults["method.damping"]
    pointed = _shown_value(flat, "method.damping.name", defaults, None,
                           ["method.damping", "method.damping.name"])  # fmt: skip
    assert pointed == "not set: the method's own default (see method.damping)"


def test_a_method_the_goal_names_is_planned_or_the_draft_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issue #266 (2026-11-02): "the assimilated method" became a localization study."""
    from apex.agents.planner import named_methods

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"},
                {"name": "loc", "patch": {"method": {"localization": {"name": "adaptive"}}},
                 "rationale": "localized"}]  # fmt: skip

    def call(goal: str) -> dict[str, Any]:
        result: dict[str, Any] = plan.handler(plan.arguments(
            case="alpha/config.toml", budget=5000, variants=variants, goal=goal))  # fmt: skip
        return result

    with pytest.raises(ValueError, match=r"names the method.*gies.*never substitute"):
        call("compare GIES against the case's own algorithm")
    # The case's own method named: planned as before.
    planned = call("does localization help ES-MDA")
    assert planned["draft"].endswith(".toml")
    names = ["assimilated", "chop", "es", "esmda", "gies"]
    assert named_methods("compare chop with ES-MDA", names) == {"chop", "esmda"}
    assert named_methods("compare the assimilated method", names) == {"assimilated"}
    # An everyday word is no method's name unless "method" is beside it.
    assert named_methods("how well are the data assimilated by GIES", names) == {"gies"}


def test_a_card_proposes_the_chats_latest_draft_of_its_kind(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issue #263 (2026-11-02): the card named the first plan, the answer its revision."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]

    def plan(name: str) -> str:
        draft: str = tools["plan_study"].handler(tools["plan_study"].arguments(
            case="alpha/config.toml", goal="a baseline", budget=5000, name=name,
            variants=variants))["draft"]  # fmt: skip
        return draft

    def propose(draft: str) -> dict[str, Any]:
        result: dict[str, Any] = tools["propose_action"].handler(
            tools["propose_action"].arguments(draft=draft, summary="Launch it.")
        )
        return result  # fmt: skip

    first, second = plan("first"), plan("second")
    with pytest.raises(ValueError, match=f"replaced by a later draft of this chat, {second}"):
        propose(first)
    propose(second)
    # Revised after the other, the first is the latest again.
    text = tools["read_draft"].handler(tools["read_draft"].arguments(draft=first))
    tools["revise_draft"].handler(tools["revise_draft"].arguments(draft=first, text=text["text"]))
    propose(first)
    assert [p.draft for p in store.proposals(chat.id)] == [second, first]


def test_a_report_in_a_chat_is_saved_unless_the_model_says_not(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issue #262 (2026-11-02): a write-up came back as chat text only."""
    case, _, store, assistant = _planning(tmp_path, linear_case, None)
    composition.start(case.config)
    chat = store.create("report")
    tools = {t.name: t for t in assistant.local_tools(None, chat.id)}
    run = str(case.directory / "runs" / "alpha")

    written = tools["write_report"].handler(tools["write_report"].arguments(run=run))
    quick = tools["write_report"].handler(tools["write_report"].arguments(run=run, save=False))
    outside = {t.name: t for t in assistant.local_tools(None, None)}["write_report"]

    assert written["saved"].endswith(".md")
    assert "saved" not in quick
    assert "saved" not in outside.handler(outside.arguments(run=run))  # no chat: no folder


def test_a_follow_up_report_updates_the_chat_s_report_in_place(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issue #277 (2026-11-03): a section asked for went into report_x-2.md, not the report."""
    case, _, store, assistant = _planning(tmp_path, linear_case, None)
    composition.start(case.config)
    chat = store.create("report")
    report = {t.name: t for t in assistant.local_tools(None, chat.id)}["write_report"]
    run = str(case.directory / "runs" / "alpha")

    first = report.handler(report.arguments(run=run))
    compared = report.handler(report.arguments(run=run, compare_with=run))

    assert compared["saved"] == first["saved"]
    assert compared["replaced"]
    assert not first["replaced"]
    assert "whole report" in compared["complete"]
    assert (tmp_path / compared["saved"]).read_text() == compared["markdown"]
    assert compared["markdown"] != first["markdown"]
    assert not list((tmp_path / compared["saved"]).parent.glob("report_*-2.md"))
    record = (tmp_path / compared["saved"]).parent / ".reports.json"
    record.write_text("[]")  # a damaged record: the report counts as edited, kept
    again = report.handler(report.arguments(run=run))
    assert again["saved"].endswith("-2.md")


def test_named_methods_reads_multiword_names_and_a_wrappers_inner_counts(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #266: "multilevel ES-MDA", "ES MDA", and CHOP on GIES (gies is run)."""
    from apex.agents.planner import named_methods

    names = ["es", "esmda", "multilevel_esmda", "chop", "gies", "lm_enrml"]
    assert named_methods("multilevel ES-MDA", names) == {"multilevel_esmda"}
    assert named_methods("try ES MDA", names) == {"esmda"}
    assert named_methods("LM-EnRML", names) == {"lm_enrml"}
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    linear_case(
        name="gies5", ensemble_size=20,
        method='name = "gies"\noptions = { max_trials_per_iteration = 4 }\n'
        'strategy = { name = "rlmmac", options = { energy = 0.99 } }\nmax_iterations = 2',
    )  # fmt: skip
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    planned = plan.handler(plan.arguments(
        case="gies5/config.toml", goal="CHOP on GIES", budget=5000,
        methods={"chop": {"name": "chop", "max_iterations": 2}},
        variants=[{"name": "plain", "rationale": "as configured"}],
    ))  # fmt: skip
    assert set(planned["methods"]) == {"chop"}


def test_two_run_drafts_can_both_be_proposed_and_a_saved_report_is_kept(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #263 and #262: several runs on purpose; an edited report not overwritten."""
    case, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("runs")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}

    def draft(name: str) -> str:
        made: str = tools["draft_run"].handler(tools["draft_run"].arguments(
            case="alpha/config.toml", patch={"run": {"name": name}}))["draft"]  # fmt: skip
        return made

    first, second = draft("one"), draft("two")
    for made in (first, second):
        tools["propose_action"].handler(
            tools["propose_action"].arguments(draft=made, summary="Launch it."))  # fmt: skip
    assert [p.draft for p in store.proposals(chat.id)] == [first, second]

    composition.start(case.config)
    report = {t.name: t for t in assistant.local_tools(None, chat.id)}["write_report"]
    run = str(case.directory / "runs" / "alpha")
    saved = report.handler(report.arguments(run=run))["saved"]
    path = tmp_path / saved
    path.write_text("my edits\n")
    again = report.handler(report.arguments(run=run))["saved"]
    assert again != saved
    assert again.endswith("-2.md")
    assert path.read_text() == "my edits\n"
    assert report.handler(report.arguments(run=run))["saved"] == again  # same text, same file
    # #277: a follow-up replaces the assistant's -2, never adds a -3, and names the kept file.
    compared = report.handler(report.arguments(run=run, compare_with=run))
    assert compared["saved"] == again
    assert compared["replaced"]
    assert path.name in compared["complete"]
    assert (tmp_path / again).read_text() == compared["markdown"]
    assert path.read_text() == "my edits\n"


def test_a_comparisons_card_states_each_arms_runs_and_flags_a_fifth(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Issues #264 and #267 (the user, 2026-11-02): state the budgets on the card, flag a
    difference beyond about 20 %, cap nothing."""
    from apex.agents.chat import BUDGET_TOLERANCE, _comparison, _Matching

    assert BUDGET_TOLERANCE == 0.20
    cells = [_opt_cell("a", {"name": "enopt"}), _opt_cell("b", {"name": "pso"})]
    near = [{"simulator_runs": 100}, {"simulator_runs": 78}]  # 22 % apart: flagged now
    [flagged] = [u for u in _comparison(cells, _Matching(near, {}, {}))["unmatched"]
                 if u.startswith("budget")]  # fmt: skip
    assert "equal-budget variant can be drafted on request" in flagged

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    planned = tools["plan_study"].handler(tools["plan_study"].arguments(
        case="alpha/config.toml", goal="compare ES against ES-MDA", budget=5000,
        methods={"es": {"name": "es", "strategy": {"name": "approx"}},
                 "esmda": {"name": "esmda", "options": {"inflation": [4, 4, 4, 4]},
                           "strategy": {"name": "approx"}}},
        variants=[{"name": "plain", "rationale": "as configured"}]))  # fmt: skip
    propose = tools["propose_action"]
    propose.handler(propose.arguments(draft=planned["draft"], summary="Launch it."))
    [card] = store.proposals(chat.id)
    assert set(card.data["arms"].values()) == {40, 100}  # as the plan costs them
    assert any(u.startswith("budget") for u in card.data["unmatched"])
    # Nothing capped: ES-MDA keeps its four steps.
    written = tomllib.loads((tmp_path / planned["draft"]).read_text())["study"]["methods"]
    assert written["esmda"]["options"]["inflation"] == [4, 4, 4, 4]
    assert "max_iterations" not in written["esmda"]


def test_arms_that_need_a_missing_extra_are_named_so_the_others_are_kept(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#271 (2026-11-02): ert_adaptive (no ert extra) was refused, and the assistant fell back
    to a default draft without the adaptive_rows arm the user asked for."""
    from apex.services import plugins

    def missing(factory: object) -> tuple[str, str] | None:
        requires = getattr(factory, "requires", None) or {}
        return next(iter(requires.items()), None)

    monkeypatch.setattr(plugins, "missing_extra", missing)
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    rows = {
        "name": "esmda",
        "options": {"inflation": [2, 2]},
        "strategy": {"name": "adaptive_rows", "options": {"threshold": 0.5}},
    }
    ert = {"name": "esmda", "options": {"inflation": [2, 2]}, "strategy": {"name": "ert_adaptive"}}

    def call(methods: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = plan.handler(plan.arguments(
            case="alpha/config.toml", goal="adaptive localization in APEX and ERT",
            budget=5000, methods=methods,
            variants=[{"name": "plain", "rationale": "as configured"}]))  # fmt: skip
        return result

    with pytest.raises(ValueError, match="need an extra") as refused:
        call({"esmda_rows": rows, "esmda_ert": ert})
    message = str(refused.value)
    assert "method esmda_ert: update_strategies 'ert_adaptive' needs" in message
    assert "esmda_rows" not in message
    assert "keeping every other arm" in message
    assert set(call({"esmda_rows": rows})["methods"]) == {"esmda_rows"}  # the rest is planned


def test_a_weak_comparison_is_warned_on_the_card_not_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#276 (2026-11-02): one seed and two GIES iterations compared LM damping with the case's
    rlmmac, whose defaults also differ; the card says so, and the user decides."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    method = 'name = "gies"\nstrategy = { name = "rlmmac" }\nmax_iterations = 2'
    linear_case(name="gies2", ensemble_size=20, method=method)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    planned = tools["plan_study"].handler(tools["plan_study"].arguments(
        case="gies2/config.toml", goal="LM damping against the case's", budget=5000,
        replicates=1, variants=[
            {"name": "baseline", "rationale": "as configured"},
            {"name": "lm", "rationale": "LM damping",
             "patch": {"method": {"damping": {"name": "lm"}}}},
        ]))  # fmt: skip
    warnings = planned["design_warnings"]
    assert warnings[0].startswith("1 replicate(s): a difference between the arms may be seed")
    assert any("2 iteration(s) of gies" in w for w in warnings)
    [damping] = [w for w in warnings if w.startswith("damping:")]
    for option in ("initial", "factor", "minimum", "confirm_convergence"):
        assert option in damping
    assert "rlmmac (" in damping
    assert "lm (" in damping

    propose = tools["propose_action"]
    propose.handler(propose.arguments(draft=planned["draft"], summary="Launch it."))
    [card] = store.proposals(chat.id)
    assert card.data["warnings"] == warnings  # shown on the card; nothing refused
    # Three replicates and enough iterations: those warnings go, the defaults' stays.
    three = tools["plan_study"].handler(tools["plan_study"].arguments(
        case="gies2/config.toml", goal="LM damping against the case's", budget=5000,
        replicates=3, name="three", variants=[
            {"name": "baseline", "rationale": "as configured",
             "patch": {"method": {"max_iterations": 3}}},
            {"name": "lm", "rationale": "LM damping",
             "patch": {"method": {"max_iterations": 3, "damping": {"name": "lm"}}}},
        ]))  # fmt: skip
    assert [w.split(":")[0] for w in three["design_warnings"]] == ["damping"]


def test_a_revision_asks_only_for_what_the_budget_holds(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#293 (2026-11-08): the revision asked for 3 replicates at 700 runs each under a cap
    of 1900, so the alternative it named was refused. It now names only what the revised
    design, costed as it would run, keeps within the budget, and the alternative it names
    is accepted; what is left out is said with its numbers, beside the card too."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    method = 'name = "gies"\nstrategy = { name = "rlmmac" }\nmax_iterations = 2'
    linear_case(name="gies2", ensemble_size=20, method=method)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan, propose = tools["plan_study"], tools["propose_action"]

    def variants(iterations: int | None = None) -> list[dict[str, Any]]:
        more = {} if iterations is None else {"max_iterations": iterations}
        return [
            {"name": "baseline", "rationale": "as configured", "patch": {"method": {**more}}},
            {"name": "lm", "rationale": "LM damping",
             "patch": {"method": {**more, "damping": {"name": "lm"}}}},
        ]  # fmt: skip

    def planned(
        name: str,
        budget: int,
        replicates: int = 1,
        iterations: int | None = None,
        alternative_to: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = plan.handler(plan.arguments(
            case="gies2/config.toml", goal="LM damping against the case's", budget=budget,
            name=name, replicates=replicates, variants=variants(iterations),
            alternative_to=alternative_to))  # fmt: skip
        return result

    roomy = planned("roomy", 5000)
    assert roomy["design_revision"]["replicates"] == 3
    assert "design_unfit" not in roomy
    one = roomy["cost"]  # one replicate at 2 iterations
    raised = planned("raised", 5000, iterations=3)["cost"]  # one at 3
    assert raised > one

    # Room for 2 replicates as planned (and 1 at 3 iterations), not for 3.
    budget = max(2 * one, raised) + 1
    tight = planned("tight", budget)
    revision = tight.get("design_revision", {})
    assert "replicates" not in revision  # never an alternative the budget refuses
    assert f"one replicate costs {one} ({raised} with its iterations)" in tight["design_unfit"]
    # The raised iterations fit at 1 replicate: still offered, and the budget accepts them.
    assert revision["max_iterations"]
    assert planned("tight_alt", budget, iterations=3, alternative_to=tight["draft"])["draft"]
    card = propose.handler(propose.arguments(draft=tight["draft"], summary="As planned."))
    assert card["design_unfit"] == tight["design_unfit"]
    assert "design_unfit" in card["next"]


def test_fitting_a_revision_to_the_budget() -> None:
    """#293 and its review: the revised design is costed with its iterations; what fits is
    kept, replicates first; nothing known, nothing changed."""
    from apex.agents.chat import _fit_revision

    def cost(planned: int, raised: int) -> Any:  # noqa: ANN401 (a ReplicateCost)
        return lambda iterations: raised if iterations else planned

    both = {"replicates": 3, "max_iterations": {"a": 3}}
    assert _fit_revision({"replicates": 3}, 1, cost(600, 600), 1900) == ({"replicates": 3}, "")
    assert _fit_revision(both, 1, None, 1900) == (both, "")
    assert _fit_revision(both, 1, cost(600, 850), None) == (both, "")
    # 3 x 850 does not fit; 3 replicates as planned (1800) do, and so do the raised
    # iterations at 1 replicate (850): both are said, the user chooses (review).
    kept, why = _fit_revision(both, 1, cost(600, 850), 1900)
    assert kept == {"replicates": 3}
    assert "either fits alone, not both: 3 replicates (1800)" in why
    assert "the iterations at 1 replicate(s) (850)" in why
    assert "offer the other too" in why
    assert "do not fit" not in why
    # Only iterations revised (3 replicates already): no replicates named.
    kept, why = _fit_revision({"max_iterations": {"a": 3}}, 3, cost(600, 700), 1900)
    assert kept == {}
    assert why.startswith("raised iterations would cost 2100 runs")
    assert "the iterations do not fit even at 3 replicate(s) (2100)" in why
    # #293's numbers: 3 x 700 does not fit; the iterations at the 2 replicates planned do.
    kept, why = _fit_revision(both, 2, cost(700, 900), 1900)
    assert kept == {"max_iterations": {"a": 3}}
    assert why.startswith("3 replicates and raised iterations would cost 2700 runs")
    assert "over the budget of 1900" in why
    assert "one replicate costs 700 (900 with its iterations), so 2 fit as planned" in why
    assert "3 replicates need at most 633 each" in why
    assert why.endswith("the user decides")
    # Nothing fits: nothing is named.
    assert _fit_revision(both, 2, cost(700, 1000), 1900)[0] == {}


def test_a_warned_plan_offers_an_alternative_beside_its_card(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """#288 (2026-11-07): plan_study warned that 2 GIES iterations and 1 replicate were too
    few to compare LM damping with the case's, and only a card with those settings went up.
    The result names the revision; the card's result asks for the alternative; the
    alternative (a draft of its own) replaces neither, and the original stays launchable."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    method = 'name = "gies"\nstrategy = { name = "rlmmac" }\nmax_iterations = 2'
    linear_case(name="gies2", ensemble_size=20, method=method)
    chat = store.create("plan")
    tools = {t.name: t for t in assistant.planning_tools(chat.id, None)}
    plan, propose = tools["plan_study"], tools["propose_action"]

    def variants(iterations: int | None) -> list[dict[str, Any]]:
        more = {} if iterations is None else {"max_iterations": iterations}
        return [
            {"name": "baseline", "rationale": "as configured",
             **({"patch": {"method": more}} if more else {})},
            {"name": "lm", "rationale": "LM damping",
             "patch": {"method": {**more, "damping": {"name": "lm"}}}},
        ]  # fmt: skip

    first = plan.handler(plan.arguments(
        case="gies2/config.toml", goal="LM damping against the case's", budget=5000,
        name="damping", replicates=1, variants=variants(None)))  # fmt: skip
    revision = first["design_revision"]
    assert revision["replicates"] == 3
    assert sorted(revision["max_iterations"].values()) == [3, 3]
    assert "alternative_to" in first["design_warnings_note"]

    card = propose.handler(propose.arguments(draft=first["draft"], summary="As planned."))
    assert card["design_warnings_open"] == first["design_warnings"]
    assert "alternative_to" in card["next"]

    with pytest.raises(ValueError, match="name of its own"):  # it would overwrite the first
        plan.handler(plan.arguments(
            case="gies2/config.toml", goal="LM damping against the case's", budget=5000,
            name="damping", replicates=3, variants=variants(3),
            alternative_to=first["draft"]))  # fmt: skip
    second = plan.handler(plan.arguments(
        case="gies2/config.toml", goal="LM damping against the case's", budget=5000,
        name="damping_revised", replicates=3, variants=variants(3),
        alternative_to=first["draft"]))  # fmt: skip
    assert second["draft"] != first["draft"]
    assert second["alternative_to"] == first["draft"]
    assert "design_revision" not in second  # only the plugins' defaults are left
    assert "alternative to" in second["design_warnings_note"]

    other = propose.handler(propose.arguments(draft=second["draft"], summary="Revised."))
    assert "design_warnings_open" not in other
    # The original stays the user's choice: not refused as replaced by the alternative.
    again = propose.handler(propose.arguments(draft=first["draft"], summary="As planned."))
    assert "design_warnings_open" not in again  # its alternative exists now
    assert [p.draft for p in store.proposals(chat.id)] == [
        first["draft"], second["draft"], first["draft"]]  # fmt: skip


def test_the_prompt_offers_an_alternative_to_a_warned_plan_and_never_a_cut_list() -> None:
    """#288 and #289: the rules behind the tools' fields."""
    assert "propose the draft as planned and, beside it, an alternative" in SYSTEM
    assert "which to launch (or neither) is the user's choice" in SYSTEM
    assert "never present a partial or truncated list" in SYSTEM
    assert "say how many were shown and how many matched" in SYSTEM
    # The review of #288 (M2): the older rules no longer forbid the alternative.
    assert "never raise it unasked; " not in SYSTEM
    assert "a study you judge too small to answer the question is revised" not in SYSTEM
    assert "a gap you point out in your own draft is fixed" not in SYSTEM
    assert "never too small, which an inherited" not in SYSTEM
    assert "that offer is the alternative draft of the next rule" in SYSTEM
    assert "plan_study's design_warnings have their own rule" in SYSTEM
    assert "(not a plan_study design warning, which the alternative draft answers)" in SYSTEM
    # The re-review (m-c, m-d): only replicates and the revision's iterations are traded;
    # cut fields are read before their values are stated.
    assert "trade only its replicates and the revision's iterations, never members" in SYSTEM
    assert "before stating values from them, read the draft (read_draft)" in SYSTEM


def test_find_cases_says_when_its_list_is_cut(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#289 (2026-11-07): find_cases listed its first 100 and the assistant called them all
    the configurations with 'config' in their name. The result counts every match (up to a
    bound) and says, in fields and words, when the list is partial."""
    import apex.agents.chat as chat_module

    case, _, store, assistant = _planning(tmp_path, linear_case, None)
    many = tmp_path / "many"
    many.mkdir()
    for k in range(4):
        (many / f"config_{k}.toml").write_text(case.config.read_text())
    monkeypatch.setattr(chat_module, "FIND_LIMIT", 3)
    monkeypatch.setattr(chat_module, "FIND_COUNT_LIMIT", 5)
    chat = store.create("find")
    find = {t.name: t for t in assistant.local_tools(None, chat.id)}["find_cases"]

    cut = find.handler(find.arguments(query="config", folder="many"))
    assert (cut["shown"], cut["total"], cut["complete"]) == (3, 4, False)
    assert len(cut["configurations"]) == 3
    assert cut["truncated"].startswith("showing 3 of 4 matching configurations")

    whole = find.handler(find.arguments(query="config_1", folder="many"))
    assert (whole["shown"], whole["total"], whole["complete"]) == (1, 1, True)
    assert "truncated" not in whole

    for k in range(4, 10):
        (many / f"config_{k}.toml").write_text(case.config.read_text())
    beyond = find.handler(find.arguments(query="config", folder="many"))
    assert (beyond["shown"], beyond["total"], beyond["complete"]) == (3, "more than 5", False)
    assert beyond["truncated"].startswith("showing 3 of more than 5")


def _gies2_planning(
    tmp_path: Path, linear_case: CaseFactory
) -> tuple[ChatStore, str, dict[str, LocalTool]]:
    """A GIES case of 2 iterations, its chat and planning tools (#288)."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    method = 'name = "gies"\nstrategy = { name = "rlmmac" }\nmax_iterations = 2'
    linear_case(name="gies2", ensemble_size=20, method=method)
    chat = store.create("plan")
    return store, chat.id, {t.name: t for t in assistant.planning_tools(chat.id, None)}


def _damping_variants(iterations: int | None) -> list[dict[str, Any]]:
    more = {} if iterations is None else {"max_iterations": iterations}
    return [
        {"name": "baseline", "rationale": "as configured",
         **({"patch": {"method": more}} if more else {})},
        {"name": "lm", "rationale": "LM damping",
         "patch": {"method": {**more, "damping": {"name": "lm"}}}},
    ]  # fmt: skip


def _plan_damping(tools: dict[str, LocalTool], **given: object) -> dict[str, Any]:
    plan = tools["plan_study"]
    arguments = {"case": "gies2/config.toml", "goal": "LM damping against the case's",
                 "budget": 5000, **given}  # fmt: skip
    result: dict[str, Any] = plan.handler(plan.arguments(**arguments))
    return result


def test_an_alternatives_link_holds_for_any_spelling_and_only_from_its_latest_plan(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #288: alternative_to and propose's draft spelt absolute or with './'
    still link the pair (m1); a later plan under the alternative's name without
    alternative_to ends the link (m2); a run draft is no study to be an alternative to
    (m3); an alternative's note names remaining warnings only (m5)."""
    store, chat_id, tools = _gies2_planning(tmp_path, linear_case)
    propose = tools["propose_action"]
    first = _plan_damping(tools, name="damping", replicates=1,
                          variants=_damping_variants(None))  # fmt: skip

    run = tools["draft_run"].handler(tools["draft_run"].arguments(
        case="gies2/config.toml", patch={}, name="as_run"))  # fmt: skip
    with pytest.raises(ValueError, match="study draft of this chat of the same kind"):
        _plan_damping(tools, name="other", replicates=3, variants=_damping_variants(3),
                      alternative_to=run["draft"])  # fmt: skip

    second = _plan_damping(tools, name="damping_revised", replicates=3,
                           variants=_damping_variants(3),
                           alternative_to=str(tmp_path / first["draft"]))  # fmt: skip
    assert second["alternative_to"] == first["draft"]  # recorded as the chat names it
    note = second["design_warnings_note"]
    assert "design warnings remain" in note  # the plugins' defaults still differ
    assert "design_revision of its own" not in note
    propose.handler(propose.arguments(draft="./" + first["draft"], summary="As planned."))
    propose.handler(propose.arguments(draft=second["draft"], summary="Revised."))
    assert [p.draft for p in store.proposals(chat_id)] == [first["draft"], second["draft"]]

    # A new design under the alternative's name, not offered as one: the link ends.
    _plan_damping(tools, name="damping_revised", replicates=4, variants=_damping_variants(3))
    with pytest.raises(ValueError, match="replaced by a later draft"):
        propose.handler(propose.arguments(draft=first["draft"], summary="As planned."))


def test_a_draft_revised_under_another_spelling_is_still_the_same_draft(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The re-review of #288 (m-b): revise_draft recorded the draft as typed ('./...'), so
    proposing it under the same spelling was refused as replaced by its own revision."""
    store, chat_id, tools = _gies2_planning(tmp_path, linear_case)
    first = _plan_damping(tools, name="damping", replicates=1,
                          variants=_damping_variants(None))  # fmt: skip
    text = (tmp_path / first["draft"]).read_text() + "\n# revised\n"
    revise = tools["revise_draft"]
    revised = revise.handler(revise.arguments(draft="./" + first["draft"], text=text))
    assert revised["draft"] == first["draft"]
    [event] = [e for e in store.events(chat_id) if e.kind == "draft_revised"]
    assert event.data["draft"] == first["draft"]
    propose = tools["propose_action"]
    propose.handler(propose.arguments(draft="./" + first["draft"], summary="Revised."))
    assert [p.draft for p in store.proposals(chat_id)] == [first["draft"]]


def test_an_alternative_the_budget_cannot_hold_stops_the_reminder(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #288 (m4): the alternative refused on budget is recorded, and the
    original's card no longer asks for one but quotes the refusal."""
    _, _, tools = _gies2_planning(tmp_path, linear_case)
    propose = tools["propose_action"]
    first = _plan_damping(tools, name="damping", replicates=1,
                          variants=_damping_variants(None))  # fmt: skip
    with pytest.raises(ValueError, match="budget") as refused:
        _plan_damping(tools, name="damping_revised", replicates=3, budget=first["cost"],
                      variants=_damping_variants(3), alternative_to=first["draft"])  # fmt: skip
    card = propose.handler(propose.arguments(draft=first["draft"], summary="As planned."))
    assert "design_warnings_open" not in card
    assert card["alternative_refused"] == str(refused.value)[:2000]
    assert "quote the refusal's cost" in card["next"]


def test_a_plans_result_states_its_cost_and_warnings_first_and_fits_the_limit(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #288 (m6) and the release tester's H33: a plan's cost line, design
    warnings and revision, approval and arm costs come before its long blocks, so that the
    events' excerpt and the tools' bound keep them; over the limit, the long blocks are left
    out whole, never the cost or the warnings."""
    import apex.agents.chat as chat_module
    from apex.agents.loop import result_excerpt
    from apex.services.tools import BOUND_OMIT_MIN, bound

    _, _, tools = _gies2_planning(tmp_path, linear_case)
    first = _plan_damping(tools, name="damping", replicates=1,
                          variants=_damping_variants(None))  # fmt: skip
    keys = list(first)
    for key in ("cost", "cost_line", "design_warnings", "design_revision", "approval",
                "arm_costs", "next"):  # fmt: skip
        if key in first:
            assert keys.index(key) < keys.index("differs") < len(keys), key
            assert keys.index(key) < keys.index("held_fixed"), key
    excerpt = result_excerpt(first)
    for key in ("cost", "design_revision", "approval"):
        assert key in excerpt, key

    last = [k for k in chat_module.PLAN_LAST if k in first]
    assert keys[-len(last) :] == last
    # The tools' bound on a long plan cuts the long blocks from the end, never the rest.
    head = {k: v for k, v in first.items() if k not in chat_module.PLAN_LAST}
    whole = len(json.dumps(first, default=str))
    room = len(json.dumps(head, default=str)) + len(chat_module.PLAN_LAST) * BOUND_OMIT_MIN + 300
    assert room < whole - 100
    for limit in (whole - 100, room):
        small = bound(first, limit=limit)
        assert len(json.dumps(small, default=str)) <= limit
        assert {k: small[k] for k in head} == head  # only the long blocks were cut
        assert small["truncated"] is True


def test_a_plans_arms_are_cut_last(tmp_path: Path, linear_case: CaseFactory) -> None:
    """The re-review's suite (2026-11-07): a plan of ES-MDA's four localization arms reached
    the model with two, since bound() cut its methods before its comparison blocks. The
    arms are cut last; the comparison's note and the defaults (read_draft gives them) first."""
    from apex.services.tools import bound

    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("arms")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]
    base = {"name": "esmda", "options": {"inflation": [2, 2]}}
    methods = {
        "esmda_approx": {**base, "strategy": {"name": "approx", "options": {"energy": 0.99}}},
        "esmda_rows": {**base, "strategy": {"name": "adaptive_rows", "options": {"energy": 0.99}}},
        "esmda_three": {"name": "esmda", "options": {"inflation": [3, 3, 3]},
                        "strategy": {"name": "approx", "options": {"energy": 0.99}}},
        "esmda_rows3": {"name": "esmda", "options": {"inflation": [3, 3, 3]},
                        "strategy": {"name": "adaptive_rows", "options": {"energy": 0.99}}},
    }  # fmt: skip
    first: dict[str, Any] = plan.handler(plan.arguments(
        case="alpha/config.toml", goal="adaptive localization", budget=5000, methods=methods,
        variants=[{"name": "plain", "rationale": "as configured"}]))  # fmt: skip
    assert set(first["methods"]) == set(methods)
    whole = len(json.dumps(first, default=str))
    lighter = {k: v for k, v in first.items()
               if k not in ("note", "defaults", "held_fixed", "differs")}  # fmt: skip
    limit = len(json.dumps(lighter, default=str)) + 1000  # their previews, bound's marks
    assert limit < whole
    view = bound(first, limit=limit)
    assert len(json.dumps(view, default=str)) <= limit
    assert view["methods"] == first["methods"]


def test_find_cases_fits_the_tools_bound_and_counts_what_it_keeps(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The review of #289 (M1): 60 matches, all under FIND_LIMIT, are over the tools'
    result limit; bound() halved the list and left shown 60 and complete true. find_cases
    now keeps what fits, and its counts and note say so, through bound() as the loop
    applies it."""
    from apex.services.tools import RESULT_LIMIT, bound

    case, _, store, assistant = _planning(tmp_path, linear_case, None)
    many = tmp_path / "many_configurations_of_the_field_model"
    many.mkdir()
    for k in range(60):
        (many / f"config_history_match_variant_{k:02d}.toml").write_text(case.config.read_text())
    chat = store.create("find")
    find = {t.name: t for t in assistant.local_tools(None, chat.id)}["find_cases"]

    view = bound(find.handler(find.arguments(query="config_history", folder=many.name)))

    assert len(json.dumps(view, default=str)) <= RESULT_LIMIT
    assert view["total"] == 60
    assert view["shown"] == len(view["configurations"]) < 60
    assert view["complete"] is False
    assert view["truncated"].startswith(f"showing {view['shown']} of 60 matching")


def test_bound_keeps_a_results_own_truncation_note_and_its_short_fields() -> None:
    """The review of #289 and #288: bound() kept no note of a result's own (it wrote True
    over find_cases' 'showing N of M'); a plan's result (CutFromEnd) is cut from its end,
    its long strings to previews, never left out whole, its short fields never; any other
    result keeps the old order (the re-review: read_draft lost its text)."""
    from apex.services.tools import BOUND_CUT, CutFromEnd, bound

    rows = {"truncated": "showing 3 of 9", "rows": [{"i": i, "t": "x" * 50} for i in range(80)]}
    assert bound(rows, limit=2000)["truncated"].startswith("showing 3 of 9; ")

    fields = {"draft": "d.toml", "cost_line": "120 simulator runs", "approval": "not needed",
              "held_fixed": "h" * 3000, "differs": "d" * 3000}  # fmt: skip
    view = bound(CutFromEnd(fields), limit=2000)
    assert len(json.dumps(view)) <= 2000
    assert (view["cost_line"], view["approval"]) == ("120 simulator runs", "not needed")
    assert set(view["shortened"]) == {"held_fixed", "differs"}
    for key in ("held_fixed", "differs"):
        assert view[key].endswith(BOUND_CUT)
        assert view[key].startswith(fields[key][0] * 10)  # its start, not left out
    assert bound(fields, limit=2000) == _old_bound(fields, 2000)  # a plain dict: as before


def _old_bound(result: Any, limit: int) -> Any:  # noqa: ANN401
    """bound() as it was before #288's review (services/tools.py at 707e2c23)."""
    text = json.dumps(result, default=str)
    if len(text) <= limit:
        return result
    if isinstance(result, dict):
        shortened = dict(result)
        for key, value in sorted(
            result.items(), key=lambda kv: -len(json.dumps(kv[1], default=str))
        ):
            if isinstance(value, list) and len(value) > 1:
                shortened[key] = value[: max(1, len(value) // 2)]
                shortened["truncated"] = True
                return _old_bound(shortened, limit)
            if isinstance(value, dict) and len(value) > 1:
                items = list(value.items())
                shortened[key] = dict(items[: max(1, len(items) // 2)])
                shortened["truncated"] = True
                return _old_bound(shortened, limit)
    return {"truncated": True, "preview": text[: limit - 100]}


def test_bound_keeps_its_order_for_drafts_and_run_summaries() -> None:
    """The re-review of #288 (M1): cutting every result from its end deleted read_draft's
    11k-character text whole and halved a run summary's posterior metrics before its
    per-iteration ones. Results that do not ask for it (CutFromEnd) are cut as before."""
    from apex.services.tools import RESULT_LIMIT, bound

    text = "".join(f'[study.variants.v{k}]\nrationale = "variant {k} of the design"\n'
                   f'patch = {{ method = {{ max_iterations = {k} }} }}\n\n'
                   for k in range(120))[:11000]  # fmt: skip
    draft = {
        "draft": "UI/owner/drafts/c/study_x.toml", "text": text, "truncated": False,
        "held_fixed": {f"method.option_{k}": k for k in range(40)},
        "differs": {f"arm{a}": {"method.damping.name": "lm" if a else "rlmmac"}
                    for a in range(2)},
        "inherited": "a study's draft names only what it sets; " * 5,
        "approval": {"needed": False, "runs": 6},
    }  # fmt: skip
    assert len(json.dumps(draft)) > RESULT_LIMIT
    view = bound(draft)
    assert view == _old_bound(draft, RESULT_LIMIT)
    assert json.dumps(text[:200])[1:-1] in view["preview"]  # the text's start, as before

    labels = ("normalized_data_mismatch", "rmse", "ssim", "model_change", "data_mismatch",
              "cv_normalized_data_mismatch")  # fmt: skip
    summary = {
        "run_id": "abc", "path": "runs/x", "name": "x", "method": "gies", "state": "complete",
        "converged": True, "iterations": list(range(15)), "posterior_iteration": 14,
        "members": 100,
        "data_mismatch": {str(k): [100.0 / (k + 1), 3.0] for k in range(15)},
        "metrics": {str(k): {m: {f"layer_{j}": 0.1 * k + j for j in range(6)}
                             for m in labels} for k in range(15)},
        "posterior_metrics": {m: {"PERMX": 0.11, "PORO": 0.22, "all": 0.33} for m in labels},
        "simulator_runs": 1500, "annotations": {"experiment": "e"},
    }  # fmt: skip
    assert len(json.dumps(summary)) > RESULT_LIMIT
    view = bound(summary)
    assert view == _old_bound(summary, RESULT_LIMIT)
    assert view["posterior_metrics"] == summary["posterior_metrics"]


def test_design_warnings_spare_es_mda_iterations_and_alike_defaults() -> None:
    """The review of #276: max_iterations below 3 on ES-MDA (not iterative) warned; a set
    section and the same plugin left to the method compared as different."""
    from types import SimpleNamespace

    from apex.agents.chat import _design_warnings

    cells = [SimpleNamespace(replicate=r) for r in range(3)]
    arms = {"a": {"method.name": "esmda", "method.max_iterations": 2},
            "b": {"method.name": "esmda", "method.max_iterations": 2}}  # fmt: skip
    methods: dict[str, dict[str, Any]] = {
        "a": {"name": "esmda", "max_iterations": 2, "strategy": {"name": "approx"}},
        "b": {"name": "esmda", "max_iterations": 2},
    }
    assert _design_warnings(cells, arms, methods) == []
    gies: dict[str, dict[str, Any]] = {"a": {"name": "gies", "damping": {"name": "rlmmac"}},
                                       "b": {"name": "gies"}}  # fmt: skip
    flat = {k: {"method.name": "gies"} for k in gies}
    assert _design_warnings(cells, flat, gies) == []  # rlmmac set, and rlmmac by default


def test_a_goal_only_comparison_reports_its_differences_and_matches(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The final review of H32's fix (M1): the rules' path drafts "Compare ES-MDA with GIES"
    as a comparison, so the answer must carry how the arms differ (GIES's own iteration
    limit) and the energy the rules matched, as a design given with methods does."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]

    result = plan.handler(plan.arguments(case="alpha/config.toml",
                                         goal="Compare ES-MDA with GIES on this case",
                                         budget=5000))  # fmt: skip

    assert set(result["methods"]) == {"esmda", "gies"}
    assert any(d.startswith("gies runs to its own iteration limit")
               for d in result["differences"])  # fmt: skip
    assert any("gies: rlmmac energy 0.95 -> 1.0" in m for m in result["matched"])
    assert result["next"].startswith("the compared algorithms differ")


def test_a_refused_goal_only_comparison_names_no_server_path(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The final review's m6: the refusal reaches the chat model without server paths."""
    _, _, store, assistant = _planning(tmp_path, linear_case, None)
    chat = store.create("plan")
    plan = {t.name: t for t in assistant.planning_tools(chat.id, None)}["plan_study"]

    with pytest.raises(ValueError, match="within the budget of 300") as refused:
        plan.handler(plan.arguments(case="alpha/config.toml",
                                    goal="Compare ES-MDA with GIES on this case",
                                    budget=300))  # fmt: skip
    assert str(tmp_path) not in str(refused.value)
