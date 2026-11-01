"""Simulation skills' interface (core): refusals by default, failures found by their patterns."""

from pathlib import Path
from typing import ClassVar

import pytest

from apex.core.errors import ConfigError
from apex.core.plugins import simulation_skills
from apex.core.simulation_skills import FailurePattern, Finding, SimulationSkill


class _Bare(SimulationSkill):
    name = "bare"
    kind = "technique"


class _Patterned(SimulationSkill):
    name = "patterned"
    kind = "simulator"
    simulator = "toy"
    failure_patterns: ClassVar[tuple[FailurePattern, ...]] = (
        FailurePattern(pattern=r"^\s*diverged", meaning="the solver diverged", advice="cut dt"),
        FailurePattern(pattern=r"slow step", meaning="a slow step", advice="none", severity="info"),
    )


def test_a_skill_refuses_what_it_does_not_override(tmp_path: Path) -> None:
    skill = _Bare()
    assert not skill.accepts(tmp_path / "deck.DATA")
    with pytest.raises(ConfigError, match="'bare' cannot give facts"):
        skill.facts(tmp_path / "deck.DATA")
    with pytest.raises(ConfigError, match="'bare' cannot give command"):
        skill.command(tmp_path / "deck.DATA", threads=2)
    assert skill.diagnose("anything at all") == []
    # No licences by default (open-source simulators).
    assert skill.licence_features(tmp_path / "deck.DATA", ranks=4) == {}


def test_diagnose_reports_each_matching_pattern_once() -> None:
    found = _Patterned().diagnose("step 1\n  DIVERGED at step 2\ndiverged again\n")
    assert found == [
        Finding(skill="patterned", severity="error", message="the solver diverged", advice="cut dt")
    ]
    assert _Patterned().diagnose("a slow step")[0].severity == "info"


def test_the_registry_takes_skills_by_name() -> None:
    simulation_skills.register("test_patterned")(_Patterned)
    try:
        assert simulation_skills.create("test_patterned").simulator == "toy"
        assert simulation_skills.entry_point_group == "apex.simulation_skills"
    finally:
        simulation_skills.remove("test_patterned")
