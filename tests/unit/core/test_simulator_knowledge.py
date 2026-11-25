"""Core's model of a simulator's knowledge (``apex.core.simulator_knowledge``): no simulator."""

from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from apex.core.simulation_skills import Finding
from apex.core.simulator_knowledge import (
    KnowledgeFact,
    SimulatorKnowledge,
    knowledge_findings,
    read_knowledge,
)

FILE = """\
installed = "Sim 1.0"
sources = ["1.0: its manual"]

[[facts]]
topic = "pvt"
title = "Tables"
keywords = ["PVTO"]
read_from = "1.0"
confirmed_installed = true
statement = "how it reads PVTO"
source = "manual p. 1"
verified = "a run"
consequence = "read PVTO alike"
differs = "the same in 0.9"

[[facts]]
topic = "wells"
title = "Every deck"
read_from = "1.0"
confirmed_installed = false
statement = "a general fact"
source = "manual p. 2"
verified = "read"
consequence = "know it"
differs = "not read"
"""


class _Fact(KnowledgeFact):
    differs: str = Field(min_length=1)


class _Knowledge(SimulatorKnowledge[_Fact]):
    pass


def _write(tmp_path: Path, text: str = FILE) -> Path:
    path = tmp_path / "sim.toml"
    path.write_text(text)
    return path


def test_a_file_is_read_into_the_simulator_s_own_model(tmp_path: Path) -> None:
    knowledge = read_knowledge(_write(tmp_path), _Knowledge)
    assert isinstance(knowledge.facts[0], _Fact)
    assert knowledge.facts[0].differs == "the same in 0.9"
    with pytest.raises(ValueError, match="not valid TOML"):
        read_knowledge(_write(tmp_path, "installed = "), _Knowledge)
    with pytest.raises(ValidationError):  # an unknown field is refused
        read_knowledge(_write(tmp_path, FILE + 'extra = "x"\n'), _Knowledge)
    with pytest.raises(ValidationError, match="upper case"):
        read_knowledge(_write(tmp_path, FILE.replace('["PVTO"]', '["pvto"]')), _Knowledge)


def test_facts_apply_by_keywords_or_all_when_unknown(tmp_path: Path) -> None:
    knowledge = read_knowledge(_write(tmp_path), _Knowledge)
    assert [f.title for f in knowledge.applicable(["pvto"])] == ["Tables", "Every deck"]
    assert [f.title for f in knowledge.applicable([])] == ["Every deck"]
    assert [f.title for f in knowledge.applicable(None)] == ["Tables", "Every deck"]


def test_findings_are_info_with_version_and_evidence(tmp_path: Path) -> None:
    path = _write(tmp_path)
    found = knowledge_findings(
        lambda: read_knowledge(path, _Knowledge), ["PVTO"], "sim_runs",
        label="Sim", file_name=path.name, unconfirmed=", unconfirmed", evidence="evidence",
    )  # fmt: skip
    assert [f.message for f in found] == [
        "Sim knowledge (1.0): Tables: how it reads PVTO",
        "Sim knowledge (1.0, unconfirmed): Every deck: a general fact",
    ]
    assert found[0].advice == "read PVTO alike (evidence: manual p. 1)"
    assert all(f.severity == "info" and f.skill == "sim_runs" for f in found)


def test_an_unreadable_file_is_an_info_note_naming_the_file_only(tmp_path: Path) -> None:
    def missing() -> _Knowledge:
        raise FileNotFoundError(2, "No such file or directory", str(tmp_path / "sim.toml"))

    (note,) = knowledge_findings(missing, ["PVTO"], "sim_runs", label="Sim", file_name="sim.toml")
    assert note == Finding(
        skill="sim_runs", severity="info",
        message="the Sim knowledge is unreadable: sim.toml: No such file or directory",
    )  # fmt: skip

    def broken() -> _Knowledge:
        raise ValueError("bad")

    (note,) = knowledge_findings(broken, None, "sim_runs", label="Sim", file_name="sim.toml")
    assert note.message == "the Sim knowledge is unreadable: bad"
