"""Converter-skill options: declared, set, refused, and given.

A skill declares its options (``SkillOption``), each an attribute holding its default;
``configure`` sets them on one instance. The case-converter applies them, and the command
line gives them as ``--option NAME=VALUE``.
"""

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from apex.agents.converter import CaseConverter
from apex.cli.main import app
from apex.core.conversion import (
    Capability,
    ConversionResult,
    ConverterSkill,
    SkillOption,
    parse_option_pairs,
)
from apex.core.errors import ConfigError
from apex.core.plugins import converter_skills
from apex.services.api import ApexService

NAME = "test_options"


class _Skill(ConverterSkill):
    """A skill that writes its options' values (a test stand-in)."""

    name = NAME
    formats: ClassVar[tuple[str, ...]] = ("text", "apex_config")
    capabilities: ClassVar[frozenset[Capability]] = frozenset({"convert_to"})
    states: bool = False
    level: int = 1
    mode: str = "fast"
    options: ClassVar[Mapping[str, SkillOption]] = MappingProxyType(
        {
            "states": SkillOption("write the states"),
            "level": SkillOption("how much", int),
            "mode": SkillOption("how", str, ("fast", "slow")),
        }
    )

    def convert_to(self, config: Path, output: Path) -> ConversionResult:
        output.mkdir(parents=True)
        path = output / "out.txt"
        path.write_text(f"{self.states} {self.level} {self.mode}")
        return ConversionResult(path)


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    converter_skills.add(NAME, _Skill)
    try:
        yield
    finally:
        converter_skills.remove(NAME)


def test_options_are_parsed_by_kind_and_refused_by_name() -> None:
    skill = _Skill()
    skill.configure({"states": "yes", "level": "3", "mode": "slow"})
    assert (skill.states, skill.level, skill.mode) == (True, 3, "slow")
    assert _Skill().states is False  # set on the instance only
    skill.configure({"states": False, "level": 2})
    assert (skill.states, skill.level) == (False, 2)
    with pytest.raises(ConfigError, match=r"takes no option colour \(its options: level, mode"):
        skill.configure({"colour": "red"})
    with pytest.raises(ConfigError, match="states='maybe': give true or false"):
        skill.configure({"states": "maybe"})
    with pytest.raises(ConfigError, match="give a int"):
        skill.configure({"level": "2.5"})
    with pytest.raises(ConfigError, match="give one of fast, slow"):
        skill.configure({"mode": "/etc/passwd"})
    with pytest.raises(ValueError, match="needs its choices"):
        SkillOption("free text", str)
    with pytest.raises(ConfigError, match=r"takes no option vtk_states .*its options: none"):
        ConverterSkill().configure({"vtk_states": True})


def test_a_date_option_takes_an_iso_date_only() -> None:
    """Follow-up 10: a ``datetime.date`` option, held as its ISO text."""
    import datetime as dt

    option = SkillOption("the run's start", dt.date)
    assert option.parse("start", " 2030-01-01 ") == "2030-01-01"
    assert option.parse("start", dt.date(2030, 2, 3)) == "2030-02-03"
    assert option.describe() == "date: the run's start"
    for bad in ("20300101", "2030-02-30", "tomorrow", "../x", dt.datetime(2030, 1, 1), 2030):
        with pytest.raises(ConfigError, match=r"give an ISO date \(YYYY-MM-DD\)"):
            option.parse("start", bad)


def test_option_pairs_from_the_command_line() -> None:
    assert parse_option_pairs(["a=1", "b=x=y", "c="]) == {"a": "1", "b": "x=y", "c": ""}
    with pytest.raises(ConfigError, match="give NAME=VALUE"):
        parse_option_pairs(["flag"])
    with pytest.raises(ConfigError, match="given twice"):
        parse_option_pairs(["a=1", "a=2"])


def test_the_converter_and_the_command_apply_them(tmp_path: Path) -> None:
    source = tmp_path / "case.toml"
    source.write_text("")
    converter = CaseConverter(ApexService(tmp_path / "svc"), None)
    result = converter.translate(NAME, "to", source, tmp_path / "a", options={"states": "true"})
    assert result.path.read_text() == "True 1 fast"
    with pytest.raises(ConfigError, match="takes no option"):
        converter.translate(NAME, "to", source, tmp_path / "b", options={"colour": "red"})
    assert not (tmp_path / "b").exists()

    runner = CliRunner()
    given = runner.invoke(app, ["agent", "convert", "to", NAME, str(source), str(tmp_path / "c"),
                                "--option", "level=4", "--option", "mode=slow",
                                "--root", str(tmp_path / "svc")])  # fmt: skip
    assert given.exit_code == 0, given.output
    assert (tmp_path / "c" / "out.txt").read_text() == "False 4 slow"
    refused = runner.invoke(app, ["agent", "convert", "to", NAME, str(source),
                                  str(tmp_path / "d"), "--option", "colour=red",
                                  "--root", str(tmp_path / "svc")])  # fmt: skip
    assert refused.exit_code != 0
    assert "takes no option colour" in refused.output
    listed = runner.invoke(app, ["agent", "convert", "skills"], terminal_width=200)
    assert "--option mode=... (fast|slow: how; default fast)" in listed.output
