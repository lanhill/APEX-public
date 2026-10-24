"""Advice to the agents from installed packages."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from apex.core import advice


class _Section(BaseModel):
    limit: int = 3


class _Advisor:
    def __init__(self, word: str, ident: int) -> None:
        self.word, self.ident = word, ident

    def prompt(self, agent: str, system: str, root: Path) -> tuple[str, list[int]]:
        return f"{system} {self.word}({agent})", [self.ident]

    def thresholds[S: BaseModel](
        self, section_name: str, section: S, root: Path
    ) -> tuple[S, list[int]]:
        return section.model_copy(update={"limit": self.ident}), [self.ident]


def test_without_advisors_an_agent_gets_its_own_prompt_and_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(advice, "advisors", lambda: ())
    assert advice.advise_prompt("You judge.", "monitor", tmp_path) == ("You judge.", [])
    section = _Section()
    assert advice.advise_thresholds("agents.monitor", section, tmp_path) == (section, [])


def test_advisors_apply_in_order_and_their_ids_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        advice, "advisors", lambda: (("a", _Advisor("a", 2)), ("b", _Advisor("b", 1)))
    )
    # Each advisor's ids carry its name: one's 2 never reads as another's.
    assert advice.advise_prompt("You", "x", tmp_path) == ("You a(x) b(x)", ["a:2", "b:1"])
    updated, ids = advice.advise_thresholds("agents.x", _Section(), tmp_path)
    assert (updated.limit, ids) == (1, ["a:2", "b:1"])


class _EntryPoint:
    def __init__(self, value: str, load: object) -> None:
        self.name = self.value = value
        self._load = load

    def load(self) -> object:
        if isinstance(self._load, Exception):
            raise self._load
        return self._load


def test_a_broken_or_wrong_advisor_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        _EntryPoint("broken", ImportError("no such module")),
        _EntryPoint("wrong", dict),  # gives no prompt and no thresholds
        _EntryPoint("good", lambda: _Advisor("g", 7)),
    ]
    monkeypatch.setattr(advice, "entry_points", lambda group: entries)
    advice.advisors.cache_clear()
    try:
        found = advice.advisors()
    finally:
        advice.advisors.cache_clear()
    assert [(name, type(a).__name__) for name, a in found] == [("good", "_Advisor")]


class _Failing:
    def prompt(self, agent: str, system: str, root: Path) -> tuple[str, list[int]]:
        raise RuntimeError("its store is locked")

    def thresholds[S: BaseModel](
        self, section_name: str, section: S, root: Path
    ) -> tuple[S, list[int]]:
        raise RuntimeError("its store is locked")


def test_an_advisor_failing_on_a_call_never_stops_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = (("failing", _Failing()), ("good", _Advisor("g", 4)))
    monkeypatch.setattr(advice, "advisors", lambda: pair)
    assert advice.advise_prompt("You", "x", tmp_path) == ("You g(x)", ["good:4"])
    updated, ids = advice.advise_thresholds("agents.x", _Section(), tmp_path)
    assert (updated.limit, ids) == (4, ["good:4"])


class _Skilled:
    def __init__(self) -> None:
        self.events: list[tuple[str, advice.AgentEvent]] = []

    def skill(self, agent: str, skill: str, root: Path) -> tuple[list[str], list[int]]:
        return [f"{agent} on {skill}"], [3, 3]

    def observe(self, agent: str, event: advice.AgentEvent, root: Path) -> None:
        self.events.append((agent, event))


class _FailingSkilled:
    def skill(self, agent: str, skill: str, root: Path) -> tuple[list[str], list[int]]:
        raise RuntimeError("its store is locked")

    def observe(self, agent: str, event: advice.AgentEvent, root: Path) -> None:
        raise RuntimeError("its store is locked")


def test_without_advisors_a_skill_gets_no_advice_and_events_go_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(advice, "advisors", lambda: ())
    assert advice.advise_skill("converter", "opm_flow", tmp_path) == ([], [])
    advice.report_event("converter", advice.AgentEvent("k", "s"), tmp_path)


def test_skill_advice_and_events_skip_advisors_without_them_or_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skilled = _Skilled()
    pairs = (("plain", _Advisor("p", 1)), ("failing", _FailingSkilled()), ("s", skilled))
    monkeypatch.setattr(advice, "advisors", lambda: pairs)
    event = advice.AgentEvent("unread:X", "unread: X", skill="opm_flow")

    assert advice.advise_skill("converter", "opm_flow", tmp_path) == (
        ["converter on opm_flow"],
        ["s:3"],
    )
    advice.report_event("converter", event, tmp_path)
    assert skilled.events == [("converter", event)]
