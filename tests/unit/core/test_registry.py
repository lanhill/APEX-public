import pytest

from apex.core.errors import DuplicatePluginError, UnknownPluginError
from apex.core.registry import Registry


class Method:
    def __init__(self, step: float = 1.0) -> None:
        self.step = step


def test_register_decorator_creates_instances_with_arguments() -> None:
    registry: Registry[Method] = Registry("test_methods")

    @registry.register("scaled")
    class Scaled(Method):
        pass

    method = registry.create("scaled", step=0.5)

    assert isinstance(method, Scaled)
    assert method.step == 0.5
    assert registry.names() == ["scaled"]


def test_duplicate_name_is_rejected() -> None:
    registry: Registry[Method] = Registry("test_methods")
    registry.add("plain", Method)

    with pytest.raises(DuplicatePluginError, match="plain"):
        registry.add("plain", Method)


def test_unknown_name_lists_available_plugins() -> None:
    registry: Registry[Method] = Registry("test_methods")
    registry.add("plain", Method)

    with pytest.raises(UnknownPluginError, match="available: plain"):
        registry.create("missing")


def test_entry_point_group_follows_kind() -> None:
    assert Registry[Method]("methods").entry_point_group == "apex.methods"


class _Dist:
    def __init__(self, name: str) -> None:
        self.name = name


class _EntryPoint:
    """An installed entry point, as importlib.metadata gives it."""

    def __init__(self, name: str, value: object, dist: str | None) -> None:
        self.name = name
        self._value = value
        self.dist = None if dist is None else _Dist(dist)

    def load(self) -> object:
        return self._value


def test_a_plugin_from_another_package_records_its_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin loaded by entry point records the package that provides it, so its
    audience (public, NORCE, a project) follows the package; one registered in code has
    none (APEX's own)."""
    import apex.core.registry as registry_module

    monkeypatch.setattr(
        registry_module, "entry_points",
        lambda group: [_EntryPoint("remote", dict, "apex-member")]
        if group == "apex.things" else [],
    )  # fmt: skip
    things: Registry[object] = Registry("things")
    things.add("local", dict)
    assert things.origin("remote") == "apex-member"
    assert things.origin("local") is None
    assert things.origin("unknown") is None


def test_a_package_declares_its_audience_and_its_plugins_follow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package's audience is declared by an entry point of ``apex.audience``
    (installed metadata carries no [tool] tables); its plugins are private unless public."""
    import importlib.metadata

    from apex.core import audience
    from apex.services import plugins

    declared = [_EntryPoint("norce", "apex_member", "apex_member")]
    monkeypatch.setattr(
        importlib.metadata, "entry_points",
        lambda group: declared if group == audience.AUDIENCE_GROUP else [],
    )  # fmt: skip
    audience._audiences.cache_clear()
    try:
        assert audience.audience("apex-member") == "norce"  # names compare as PEP 503 says
        assert audience.audience("apex") == "public"
        executors = plugins.REGISTRIES["executors"]
        # Loaded first: entry points load lazily, and their origins must land in the real
        # table, not in the copy the test puts in place for a while.
        executors.names()
        monkeypatch.setattr(executors, "_origins", {**executors._origins, "serial": "apex-member"})
        assert plugins.is_private("executors", "serial")
        assert plugins.plugin_info("executors", "serial").visibility == "private"
        assert not plugins.is_private("executors", "process")
    finally:
        audience._audiences.cache_clear()


def test_an_entry_point_whose_module_is_gone_leaves_the_others_usable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An environment installed before a plugin moved still names its
    old module; that plugin is unknown (validation refuses it), the others work."""
    from importlib.metadata import EntryPoint

    from apex.core import registry as module

    gone = EntryPoint("gone", "apex.nowhere.moved:Executor", "apex.widgets")
    here = EntryPoint("here", "collections:OrderedDict", "apex.widgets")
    monkeypatch.setattr(module, "entry_points", lambda group: [gone, here])
    widgets: Registry[object] = Registry("widgets")
    with caplog.at_level("ERROR", logger="apex.core.registry"):
        assert widgets.names() == ["here"]
    assert "uv sync --inexact --all-packages" in caplog.text
    with pytest.raises(UnknownPluginError):
        widgets.factory("gone")


def test_an_entry_point_failing_on_anything_but_its_own_module_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review: only a module gone is an environment older than its
    code; a plugin whose own import fails (a dependency, a renamed core name) is a bug."""
    from importlib.metadata import EntryPoint

    from apex.core import registry as module

    broken = EntryPoint("broken", "json:dumps", "apex.widgets")

    def load(self: EntryPoint) -> object:
        raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")

    monkeypatch.setattr(EntryPoint, "load", load)
    monkeypatch.setattr(module, "entry_points", lambda group: [broken])
    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        Registry[object]("widgets").names()
