"""Agents workspace members contribute are listed, configured and kept private.

The end-to-end test installs a fake distribution (``fake-lab``, audience ``lab``) on the
path of a fresh interpreter, since the catalogue and the settings read the entry points when
first imported; the others call the loader and the merge directly.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from apex.agents import catalogue
from apex.agents.catalogue import CORE_AGENTS, merged
from apex.core import agents as core_agents
from apex.core import audience
from apex.core.agents import (
    AgentContribution,
    AgentInfo,
    _clear_for_tests,
    contributed_agents,
)
from apex.core.errors import ConfigError
from apex.services.settings import _agents_section, _CoreAgentsSection

_AGENT = """
import typer
from pydantic import BaseModel, ConfigDict, Field

from apex.core.agents import AgentContribution, AgentInfo


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colour: str = "default"
    max_seconds: float = Field(default=60.0, gt=0)


def contribution():
    info = AgentInfo("fake-demo", "Greets a plugin.", "apex agent fake-demo PLUGIN",
                     None, "0064", "public")
    return AgentContribution(info, Section)


def _greet(plugin: str) -> None:
    typer.echo(f"greeting {plugin}")


def cli():
    return ("agent", "fake-demo", _greet, False, True)
"""

_ENTRY_POINTS = """[apex.agents]
fake-demo = fake_lab_agent:contribution

[apex.audience]
lab = fake_lab_agent

[apex.cli]
fake-demo = fake_lab_agent:cli
"""

_CHECK = """
import json

from typer.testing import CliRunner

from apex.agents.catalogue import AGENTS, agent_catalogue, public_agents
from apex.cli.main import app
from apex.services.settings import ServiceSettings
from fake_lab_agent import Section

found = {}
agent = next(a for a in AGENTS if a.name == "fake-demo")
found["visibility"] = agent.visibility
found["override"] = next(
    a for a in agent_catalogue({"fake-demo": "public"}) if a.name == "fake-demo"
).visibility
found["public"] = "fake-demo" in {a.name for a in public_agents({"fake-demo": "public"})}
settings = ServiceSettings.model_validate({"agents": {"fake_demo": {"colour": "amber"}}})
found["colour"] = settings.agents.contributed("fake-demo", Section).colour
try:
    ServiceSettings.model_validate({"agents": {"fake_demo": {"pitch": 1}}})
    found["unknown_refused"] = False
except ValueError:
    found["unknown_refused"] = True
result = CliRunner().invoke(app, ["agent", "fake-demo", "rlmmac"])
found["cli_output"] = result.output.strip()
print(json.dumps(found))
"""


def _install(site: Path) -> None:
    (site / "fake_lab_agent.py").write_text(_AGENT)
    info = site / "fake_lab-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-lab\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text(_ENTRY_POINTS)


def test_a_contributed_agent_is_listed_configured_and_private(tmp_path: Path) -> None:
    """A lab package's agent: in the catalogue, its section strict, its command under
    ``apex agent``; ``[agents.visibility]`` cannot make it public."""
    site = tmp_path / "site"
    site.mkdir()
    _install(site)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), *sys.path])}

    done = subprocess.run(
        [sys.executable, "-c", _CHECK], env=env, capture_output=True, text=True, check=False,
        cwd=tmp_path, timeout=300,
    )  # fmt: skip

    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found == {
        "visibility": "private", "override": "private", "public": False,
        "colour": "amber", "unknown_refused": True, "cli_output": "greeting rlmmac",
    }  # fmt: skip


@dataclass
class _Dist:
    name: str


@dataclass
class _EntryPoint:
    name: str
    target: Any
    dist: _Dist | None = None
    module: str = "fake_lab_module"

    def load(self) -> Any:  # noqa: ANN401 (an entry point's object)
        return self.target


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: float = Field(default=7.0, gt=0)


class _NoDefault(BaseModel):
    colour: str


def _info(name: str, tool: str | None = None) -> AgentInfo:
    return AgentInfo(name, "An agent.", f"apex agent {name}", tool, "0064")


@pytest.fixture
def entry_points(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_EntryPoint]]:
    declared: list[_EntryPoint] = []

    def found(group: str) -> list[_EntryPoint]:
        return declared if group == core_agents.AGENTS_GROUP else []

    monkeypatch.setattr(core_agents, "entry_points", found)
    _clear_for_tests()
    yield declared
    _clear_for_tests()


def _broken() -> AgentContribution:
    raise RuntimeError("cannot")


def test_the_loader_skips_broken_contributions(
    entry_points: list[_EntryPoint], caplog: pytest.LogCaptureFixture
) -> None:
    """A tool, a section without defaults, a wrong object, an error, a repeated name or
    section (``a_b`` after ``a-b``) is logged and skipped; the rest load with their
    distribution."""
    entry_points += [
        _EntryPoint(
            "good", lambda: AgentContribution(_info("good"), _Section), _Dist("apex-member")
        ),
        _EntryPoint("tool", lambda: AgentContribution(_info("tool", tool="run_tool"))),
        _EntryPoint("nodefault", lambda: AgentContribution(_info("nodefault"), _NoDefault)),
        _EntryPoint("wrong", lambda: _info("wrong")),
        _EntryPoint("broken", _broken),
        _EntryPoint("again", lambda: AgentContribution(_info("good"))),
        _EntryPoint("a-b", lambda: AgentContribution(_info("a-b"))),
        _EntryPoint("same-section", lambda: AgentContribution(_info("a_b"))),
        _EntryPoint("reserved", lambda: AgentContribution(_info("llm"))),
    ]

    found = contributed_agents()

    assert [(c.info.name, c.distribution) for c in found] == [
        ("good", "apex-member"),
        ("a-b", None),
    ]
    for name in ("tool", "nodefault", "wrong", "broken", "again", "same-section", "reserved"):
        assert f"entry point {name}" in caplog.text


def test_the_merge_keeps_core_names_and_follows_the_audience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contributed agent never takes core's name or section; a public package's agent
    keeps its own visibility (overridable), a lab package's is private and locked."""
    monkeypatch.setattr(audience, "_audiences", lambda: {"apex-member": "lab"})
    contributions = [
        AgentContribution(_info("monitor"), distribution="apex-member"),
        AgentContribution(_info("core_agent"), distribution="apex-member"),  # core-agent's section
        AgentContribution(_info("lab-agent"), distribution="apex-member"),
        AgentContribution(_info("public-agent"), distribution="apex-extra"),
    ]

    core = (*CORE_AGENTS, _info("core-agent"))
    agents, locked = merged(core, contributions)

    assert [a.name for a in agents] == [*(a.name for a in core), "lab-agent",
                                        "public-agent"]  # fmt: skip
    visibility = {a.name: a.visibility for a in agents}
    assert (visibility["lab-agent"], visibility["public-agent"]) == ("private", "public")
    assert locked == {"lab-agent"}
    monkeypatch.setattr(catalogue, "AGENTS", agents)
    monkeypatch.setattr(catalogue, "_AUDIENCE_PRIVATE", locked)
    shown = {a.name: a.visibility for a in catalogue.agent_catalogue(
        {"lab-agent": "public", "public-agent": "private"})}  # fmt: skip
    assert (shown["lab-agent"], shown["public-agent"]) == ("private", "private")


def test_a_contributed_section_never_replaces_cores() -> None:
    """``[agents.monitor]`` stays core's; a contributed agent without settings adds none."""
    section = _agents_section([
        AgentContribution(_info("monitor"), _Section),
        AgentContribution(_info("no-settings")),
        AgentContribution(_info("extra-agent"), _Section),
    ])  # fmt: skip

    added = set(section.model_fields) - set(_CoreAgentsSection.model_fields)
    assert added == {"extra_agent"}
    assert section().contributed("extra-agent", _Section).width == 7.0
    with pytest.raises(ConfigError, match="installed"):
        section().contributed("no-settings", _Section)


def test_an_uninstalled_agents_section_is_refused_with_a_hint(tmp_path: Path) -> None:
    """The service refuses ``[agents.<name>]`` of an agent no installed package contributes,
    naming the cure; a pinned worker's lenient reading drops it."""
    from apex.services.settings import load_settings

    (tmp_path / "service.toml").write_text("[agents.not_installed]\ncolour = 'amber'\n")

    with pytest.raises(ConfigError, match=r"agents\.not_installed.*all-packages"):
        load_settings(tmp_path)
    assert load_settings(tmp_path, lenient=True).agents.visibility == {}


def test_a_typo_in_a_core_agents_section_gets_no_install_hint(tmp_path: Path) -> None:
    """The hint is for a whole unknown section, not a misspelt key of a known one."""
    from apex.services.settings import load_settings

    (tmp_path / "service.toml").write_text("[agents.monitor]\ntypo = 1\n")

    with pytest.raises(ConfigError) as refused:
        load_settings(tmp_path)
    assert "installed" not in str(refused.value)


def test_without_contributions_the_agents_schema_is_cores() -> None:
    """The schema keeps its name and description and gains nothing: the documentation
    inventory's fingerprints of ``service.toml`` do not change with the hook."""
    schema = _agents_section([]).model_json_schema()
    core = _CoreAgentsSection.model_json_schema()

    assert (schema["title"], schema["description"]) == ("AgentsSection", "``[agents]``.")
    assert schema["properties"] == core["properties"]


_CYCLIC = """
{imports}
from apex.core.agents import AgentContribution, AgentInfo


def contribution():
    return AgentContribution(AgentInfo("cyclic", "Imports too much.", "apex agent cyclic", None,
                                       "0064"))
"""

_IMPORT_FIRST = """
import json
import sys

first = sys.argv[1]
try:
    __import__(first)
except ImportError as error:
    print(json.dumps({{"error": str(error)}}))
    raise SystemExit(0)
from apex.agents.catalogue import AGENTS
from apex.services.settings import AgentsSection

print(json.dumps({{"listed": "cyclic" in {{a.name for a in AGENTS}},
                  "section": "cyclic" in AgentsSection.model_fields}}))
"""


@pytest.mark.parametrize(
    ("imports", "first", "expected"),
    [
        # The member's own import comes first and reaches the loader: refused, loudly.
        ("from apex.services.settings import ServiceSettings", "fake_cyclic", "error"),
        # APEX comes first and the member reaches back into the catalogue: the agent is
        # skipped everywhere alike (logged), never listed without its settings.
        ("import apex.agents.catalogue", "apex.services.settings", {"listed": False,
                                                                  "section": False}),
        # Reaching only the settings (whose LLMSection is there while they are imported):
        # skipped whichever of APEX's modules comes first, so no process lists an agent
        # another leaves out (the re-review's order dependence).
        ("from apex.services.settings import LLMSection", "apex.services.settings",
         {"listed": False, "section": False}),
        ("from apex.services.settings import LLMSection", "apex.agents.catalogue",
         {"listed": False, "section": False}),
        ("from apex import services", "apex.agents.catalogue",
         {"listed": False, "section": False}),
        # What the hook allows: core and domain.
        ("from apex.core.llm import LLMSection", "apex.agents.catalogue",
         {"listed": True, "section": False}),
    ],
)  # fmt: skip
def test_an_entry_module_importing_apex_services_is_never_half_loaded(
    tmp_path: Path, imports: str, first: str, expected: object
) -> None:
    """The reviewer's cycle: a cached empty or partial result made the catalogue and the
    settings disagree."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "fake_cyclic.py").write_text(_CYCLIC.format(imports=imports))
    info = site / "fake_cyclic-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-cyclic\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text("[apex.agents]\ncyclic = fake_cyclic:contribution\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), *sys.path])}

    done = subprocess.run(
        [sys.executable, "-c", _IMPORT_FIRST.format(), first], env=env, capture_output=True,
        text=True, check=False, cwd=tmp_path, timeout=300,
    )  # fmt: skip

    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    if expected == "error":
        assert "fake_cyclic" in found["error"]
        assert "apex.core" in found["error"]
    else:
        assert found == expected


def test_a_first_call_from_two_threads_loads_once(entry_points: list[_EntryPoint]) -> None:
    """The second thread waits for the first's load; it is not refused as a cycle."""
    import threading
    import time

    calls: list[int] = []

    def slow() -> AgentContribution:
        calls.append(1)
        time.sleep(0.2)
        return AgentContribution(_info("slow"))

    entry_points.append(_EntryPoint("slow", slow))
    found: list[object] = []
    threads = [threading.Thread(target=lambda: found.append(contributed_agents()))
               for _ in range(2)]  # fmt: skip
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == [1]
    assert [[c.info.name for c in f] for f in found] == [["slow"], ["slow"]]  # type: ignore[attr-defined]


_PACKAGE_CHECK = """
import json
import sys

for first in sys.argv[1:]:
    __import__(first)
from apex.agents.catalogue import AGENTS
from apex.services.settings import AgentsSection

print(json.dumps({"listed": "packaged" in {a.name for a in AGENTS},
                  "section": "packaged" in AgentsSection.model_fields}))
"""

_PACKAGED_ENTRY = """
from pydantic import BaseModel

from apex.core.agents import AgentContribution, AgentInfo

from .helper import VALUE


class Section(BaseModel):
    value: int = VALUE


def contribution():
    return AgentContribution(AgentInfo("packaged", "In a package.", "apex agent packaged", None,
                                       "0064"), Section)
"""


@pytest.mark.parametrize(
    ("helper", "first", "listed"),
    [
        # The third review's case: the helper (imported relatively) reaches the workflows.
        # Whether the helper was imported before the contributions load or not, the agent
        # is skipped alike: its imports are read from the sources.
        ("import apex.workflows.history_matching\nVALUE = 1\n", [], False),
        ("import apex.workflows.history_matching\nVALUE = 1\n", ["fakepkg.helper"], False),
        ("import apex.workflows.history_matching\nVALUE = 1\n", ["apex.agents.catalogue"],
         False),
        # A helper using core only, and one importing the services only inside a function
        # (run when called, not on import): both allowed.
        ("from apex.core.llm import LLMSection\nVALUE = 1\n", ["fakepkg.helper"], True),
        ("VALUE = 1\n\n\ndef later():\n    import apex.services.settings\n", [], True),
    ],
)  # fmt: skip
def test_a_member_package_is_judged_by_its_sources_whatever_was_imported_first(
    tmp_path: Path, helper: str, first: list[str], listed: bool
) -> None:
    """A module of the member imported before the contributions load is never run again,
    so its imports are read from its source (the third review's order dependence)."""
    site = tmp_path / "site"
    package = site / "fakepkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text(_PACKAGED_ENTRY)
    (package / "helper.py").write_text(helper)
    info = site / "fakepkg-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakepkg\nVersion: 0.1\n")
    (info / "entry_points.txt").write_text("[apex.agents]\npackaged = fakepkg.entry:contribution\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), *sys.path])}

    done = subprocess.run(
        [sys.executable, "-c", _PACKAGE_CHECK, *first], env=env, capture_output=True,
        text=True, check=False, cwd=tmp_path, timeout=300,
    )  # fmt: skip

    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.strip().splitlines()[-1])
    assert found == {"listed": listed, "section": listed}
    if not listed:
        assert "fakepkg.helper imports apex.workflows.history_matching" in done.stderr


def test_a_forbidden_import_is_refused_before_it_runs() -> None:
    """Refused before it runs, so a misbehaving contributor never waits on the services'
    import lock (the third review's hang); the original ``__import__`` is restored."""
    import builtins

    from apex.core.import_rules import refusing_imports

    original = builtins.__import__

    def load() -> None:
        import apex.not_a_module_at_all  # type: ignore[import-not-found]  # noqa: F401

    with pytest.raises(ImportError, match=r"imports apex\.not_a_module_at_all"):
        refusing_imports(load, ("apex.core", "apex.domain"))
    assert builtins.__import__ is original

    def relative() -> None:
        exec("from . import services", {"__name__": "apex.x", "__package__": "apex"})

    with pytest.raises(ImportError, match=r"imports apex\.services"):
        refusing_imports(relative, ("apex.core", "apex.domain"))


_ALLOWED = ("apex.core", "apex.domain")


def _module(root: Path, name: str, source: str) -> None:
    path = root.joinpath(*name.split(".")).with_suffix(".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    for parent in path.parents:
        if parent == root:
            break
        (parent / "__init__.py").touch()
    path.write_text(source)


def test_the_source_check_skips_names_and_blocks_that_never_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``from apex import __version__`` imports a name, not a module; ``if TYPE_CHECKING:``
    and ``if __name__ == "__main__":`` never run on import, their ``else`` does; a function
    body runs when called (the fourth review's false positives)."""
    from apex.core.import_rules import module_level_imports

    _module(tmp_path, "fakesrc.entry", (
        "from typing import TYPE_CHECKING\n"
        "from apex import __version__\n"
        "from apex import core\n"
        "if TYPE_CHECKING:\n    from apex.services.settings import ServiceSettings\n"
        "else:\n    import apex.workflows\n"
        "if __name__ == '__main__':\n    import apex.cli.main\n"
        "def later():\n    import apex.agents.chat\n"
        "class Kept:\n    from apex import services\n"
    ))  # fmt: skip
    monkeypatch.syspath_prepend(str(tmp_path))

    edges = module_level_imports(["fakesrc.entry"], _ALLOWED)

    assert edges == [("fakesrc.entry", "apex.services"), ("fakesrc.entry", "apex.workflows")]


def test_the_source_check_never_exhausts_the_recursion_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deeply nested expression: statements are walked with a stack, so the import after
    it is found, and a parse that fails is taken as importing nothing, never raised (the
    fourth review's crash)."""
    import ast

    from apex.core import import_rules

    deep = "x = " + "+".join(["1"] * 5000) + "\n"
    _module(tmp_path, "fakedeep.entry", deep + "import apex.workflows\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    edges = import_rules.module_level_imports(["fakedeep.entry"], _ALLOWED)

    assert edges == [("fakedeep.entry", "apex.workflows")]

    def too_deep(*args: object, **kwargs: object) -> ast.Module:
        raise RecursionError

    monkeypatch.setattr(ast, "parse", too_deep)
    assert import_rules.module_level_imports(["fakedeep.entry"], _ALLOWED) == []


def test_the_checks_never_import_a_package_unchecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether a name is a module is found without importing (the fifth review's blocker):
    the source check leaves ``sys.modules`` as it was, and a package's ``__init__`` reached
    while loading goes through the hook, which refuses its import."""
    from apex.core.import_rules import module_level_imports, refusing_imports

    _module(tmp_path, "fakeinit.helper", "VALUE = 1\n")
    (tmp_path / "fakeinit" / "__init__.py").write_text("import apex.cli.main\n")
    _module(tmp_path, "fakeuser.entry", "from fakeinit import helper\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(sys.modules)

    module_level_imports(["fakeuser.entry"], _ALLOWED)

    assert set(sys.modules) == before

    def load() -> None:
        from fakeinit import helper  # type: ignore[import-not-found]  # noqa: F401

    with pytest.raises(ImportError, match=r"fakeinit imports apex\.cli"):
        refusing_imports(load, _ALLOWED)
    for name in ("fakeinit", "fakeinit.helper"):
        sys.modules.pop(name, None)


def test_a_hook_left_installed_by_someone_else_passes_through() -> None:
    """Someone who patched ``__import__`` during the load and restores ours afterwards
    leaves it installed: it is off, so later imports pass (the fourth review's leak)."""
    import builtins

    from apex.core.import_rules import refusing_imports

    original = builtins.__import__
    theirs: list[object] = []

    def load() -> None:
        saved = builtins.__import__
        theirs.append(saved)
        builtins.__import__ = lambda *a, **k: saved(*a, **k)

    try:
        refusing_imports(load, _ALLOWED)
        builtins.__import__ = theirs[0]  # type: ignore[assignment]  # restores ours
        import apex.services.settings  # noqa: F401  (passes: the hook is off)
    finally:
        builtins.__import__ = original


_DEPENDENT = """
import apex.workflows.history_matching

VALUE = 1
"""

_USES_DEPENDENT = """
from apex.core.agents import AgentContribution, AgentInfo


def contribution():
    from fakedep.mod import VALUE

    return AgentContribution(AgentInfo("crossing", f"Uses {VALUE}.", "apex agent crossing",
                                       None, "0064"))
"""


def test_a_module_of_another_package_built_on_apex_is_read_too(tmp_path: Path) -> None:
    """A member's function imports another APEX-based package's module, imported before
    the contributions load: its source is read, so the agent is skipped alike."""
    site = tmp_path / "site"
    _module(site, "fakedep.mod", _DEPENDENT)
    _module(site, "fakecross.entry", _USES_DEPENDENT)
    for name, requires, entry in (
        ("fakedep", "Requires-Dist: apex\n", "[apex.audience]\nlab = fakedep\n"),
        ("fakecross", "Requires-Dist: apex\n",
         "[apex.agents]\ncrossing = fakecross.entry:contribution\n"),
    ):  # fmt: skip
        info = site / f"{name}-0.1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.1\n{requires}"
        )
        (info / "entry_points.txt").write_text(entry)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), *sys.path])}
    check = (
        "import sys\n"
        "import fakedep.mod\n"
        "assert 'apex.core.agents' not in sys.modules  # loaded before the contributions\n"
        "from apex.agents.catalogue import AGENTS\n"
        "print('crossing' in {a.name for a in AGENTS})\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", check], env=env, capture_output=True, text=True, check=False,
        cwd=tmp_path, timeout=300,
    )  # fmt: skip

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False"
    assert "fakedep.mod imports apex.workflows.history_matching" in done.stderr
