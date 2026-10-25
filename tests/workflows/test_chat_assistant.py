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
from apex.agents.chat import AUTO_PROPOSED, READ_TOOLS, SYSTEM, ChatAssistant, public_agents
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
        "write_report", "plot_run",
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
    assert "over the budget" in over["error"]  # 2 x 3 x 100 = 600 > 200: revised
    planned = json.loads(stub.requests[2].messages[-1].content)["result"]
    assert planned["cost"] == 2 * 1 * 100
    # What the arms share and how they differ, from the cells (a reported issue).
    assert planned["held_fixed"]["prior.ensemble_size"] == 20
    assert planned["differs"]["alpha | loc"]["method.localization.name"] == "adaptive"
    # No localization in the baseline: "none", not a default that would localize.
    assert planned["differs"]["alpha | baseline"]["method.localization.name"] == "none"
    assert list(planned["seeds"]) == ["r0"]
    assert (folder / "study_loc_test.toml").exists()
    [proposal] = store.proposals(chat.id)
    assert (proposal.state, proposal.draft, proposal.data["cost"]) == ("pending", draft, 200)
    assert service.jobs.all_records() == []  # nothing ran: only a person launches
    assert shown.startswith("UI/owner/drafts/")  # the owner's web-UI folder


def test_a_planned_draft_gets_its_card_when_the_model_does_not_propose_it(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A reported issue (2026-10-24): a valid plan, and an answer with no propose_action; the card
    is shown anyway (launched only on the user's click), and the answer is the model's."""
    variants = [{"name": "baseline", "patch": {}, "rationale": "as configured"}]
    stub = StubProvider([
        Completion(tool_calls=(ToolCall("c1", "plan_study", {
            "case": "alpha/config.toml", "goal": "baseline only", "budget": 200,
            "name": "no_card", "variants": variants, "replicates": 1}),)),
        _answer("Here is the plan; it costs 100 simulator runs."),
    ])  # fmt: skip
    _, service, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("plan")
    shown, _ = assistant.drafts(chat.id)

    answer = assistant.run(chat.id, "Plan a study on alpha so I can review and launch it")

    # The model's answer, closed by the card it did not propose (a reported issue,
    # 2026-10-24: "planned but not proposed yet" above the card).
    assert answer == (
        "Here is the plan; it costs 100 simulator runs.\n\n"
        + AUTO_PROPOSED.format(drafts="`study_no_card.toml`")
    )
    assert store.turns(chat.id)[-1].content == answer
    [proposal] = store.proposals(chat.id)
    assert (proposal.state, proposal.draft) == ("pending", f"{shown}/study_no_card.toml")
    assert proposal.kind == "launch_study"
    assert proposal.data["cost"] == 100
    assert [e.kind for e in store.events(chat.id)].count("proposal") == 1
    assert service.jobs.all_records() == []  # a card, not a launch


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


def test_a_card_shown_after_a_fallback_answer_is_named_in_it(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """A turn that planned a draft and then gave no usable answer still gets the draft's card,
    and the fallback answer says so; the stored turn and the answer event agree."""
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
    assert answer.endswith(AUTO_PROPOSED.format(drafts="`study_fell_back.toml`"))
    assert store.turns(chat.id)[-1].content == answer
    [event] = [e for e in store.events(chat.id) if e.kind == "answer"]
    assert event.data["text"] == answer
    assert len(store.proposals(chat.id)) == 1


def test_a_timed_out_model_is_said_in_plain_words_and_the_card_still_named(
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
    assert answer.endswith(AUTO_PROPOSED.format(drafts="`study_slow.toml`"))
    assert "no answer in 120 s" in store.turns(chat.id)[-1].data["reason"]
    [shown] = [e for e in store.events(chat.id) if e.kind == "fallback"]
    assert shown.data["reason"] == "the model took too long to answer"


def test_the_closing_sentence_names_every_card_and_who_may_launch() -> None:
    """One card, several, and a sign-in that may not launch (its card says so)."""
    from types import SimpleNamespace

    from apex.agents.chat import AUTO_PROPOSED_MANY, AUTO_PROPOSED_NO_LAUNCH, _auto_proposed_note

    assert _auto_proposed_note([], None) == ""
    one = _auto_proposed_note(["UI/owner/drafts/c/study_a.toml"], None)
    assert one == "\n\n" + AUTO_PROPOSED.format(drafts="`study_a.toml`")
    two = _auto_proposed_note(["d/study_a.toml", "d/study_b.toml"], None)
    assert two == "\n\n" + AUTO_PROPOSED_MANY.format(drafts="`study_a.toml`, `study_b.toml`")

    def access(submit: bool) -> Any:  # noqa: ANN401 (a stand-in for Access)
        return SimpleNamespace(rules=SimpleNamespace(actions=SimpleNamespace(submit=submit)))

    guest = _auto_proposed_note(["d/study_a.toml"], access(False))
    assert guest == "\n\n" + AUTO_PROPOSED_NO_LAUNCH.format(drafts="`study_a.toml`")
    assert _auto_proposed_note(["d/study_a.toml"], access(True)) == one


def test_a_stopped_turn_proposes_none_of_its_drafts(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """After Stop (or once another turn is the chat's), a draft gets no card: its event
    would be dropped, leaving a card the user never saw, or a later turn's draft."""
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

    assert assistant._propose_drafts(chat.id, None, 0, _Turn(7, stopped, [])) == []
    other = _Turn(6, threading.Event(), [])  # no longer the chat's turn
    assert assistant._propose_drafts(chat.id, None, 0, other) == []
    assert store.proposals(chat.id) == []
    current = _Turn(7, threading.Event(), [])
    assert assistant._propose_drafts(chat.id, None, 0, current) == [
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
    told = http.get(f"/v1/chats/{chat.id}", headers=AUTH).json()["messages"][-1]["content"]
    assert "outside your web-UI folder: nothing new was submitted" in told


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
    assert "over the budget" in result.data["error"]


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
    _, service, store, assistant = _planning(tmp_path, linear_case, stub)
    chat = store.create("ert")

    answer = assistant.run(chat.id, "plan a study on alpha to " + goal)

    # The model did not propose its draft: the card was added, and the answer says so.
    assert answer.startswith("I planned ES-MDA with adaptive_rows against ert_adaptive.\n\n")
    assert answer.endswith("launched from the card.")
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
    assert chop["method.strategy"] == chop["method.damping"] == "not set: the method's own default"
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
    own = "not set: the method's own default"
    assert chop["method.strategy"] == chop["method.strategy.name"] == own
    assert chop["method.strategy.options.energy"] == own
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
    assert "strategy" not in result["methods"]["chop"]
    assert "chop runs to its own iteration limit, the case 4 steps" in result["differences"]
    assert any(d.startswith("chop's strategy is rlmmac (its default); the case's is approx")
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
