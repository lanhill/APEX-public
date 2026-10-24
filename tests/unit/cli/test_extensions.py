"""Commands and maintenance tasks other packages contribute."""

from pathlib import Path
from typing import cast

import pytest
import typer
from typer.testing import CliRunner


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self._value = value

    def load(self) -> object:
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


def _hello() -> None:
    """Say hello."""
    typer.echo("hello")


def test_contributed_commands_join_their_groups_and_broken_ones_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group or a command joins the group it names; a contributor that fails to load,
    names no group, gives no command, or would replace a command is skipped, never
    stopping apex; a private one is listed for the documentation inventory."""
    from apex.cli import extensions

    sub = typer.Typer()
    sub.command("ping")(_hello)
    contributions = [
        _EntryPoint("group", lambda: ("tools", "extra", sub, False, True)),
        _EntryPoint("command", lambda: ("", "hello", _hello, True, False)),
        _EntryPoint("broken", ImportError("no such module")),
        _EntryPoint("lost", lambda: ("nowhere", "x", _hello, False, False)),
        _EntryPoint("empty", lambda: ("", "nothing", 42, False, False)),
        _EntryPoint("shadow", lambda: ("", "version", _hello, False, False)),
    ]
    monkeypatch.setattr(extensions, "entry_points", lambda group: contributions)
    monkeypatch.setattr(extensions, "PRIVATE", set())
    app = typer.Typer()

    @app.command()
    def version() -> None:
        """The core's own command."""
        typer.echo("1.0")

    tools = typer.Typer()
    app.add_typer(tools, name="tools")
    added = extensions.install({"": app, "tools": tools})

    assert added == ["apex tools extra", "apex hello"]
    assert sorted(extensions.PRIVATE) == ["apex tools extra"]
    assert CliRunner().invoke(app, ["tools", "extra", "ping"]).output == "hello\n"
    assert CliRunner().invoke(app, ["hello"]).output == "hello\n"
    assert CliRunner().invoke(app, ["version"]).output == "1.0\n"  # not replaced


def test_a_failing_maintenance_task_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.core.maintenance import FrozenCopies, MaintenanceContext
    from apex.core.registry import Registry
    from apex.services import daemon

    class Works:
        def run(self, context: MaintenanceContext, dry_run: bool) -> list[str]:
            return [f"kept {context.keep_days:g} days, dry run {dry_run}"]

    class Fails:
        def run(self, context: MaintenanceContext, dry_run: bool) -> list[str]:
            raise OSError("host unreachable")

    tasks: Registry[object] = Registry("maintenance-test")
    tasks.add("a-fails", Fails)
    tasks.add("b-works", Works)
    monkeypatch.setattr(daemon, "maintenance_tasks", tasks)
    context = MaintenanceContext(root=tmp_path, code=cast("FrozenCopies", None), jobs=list,
                                 keep_days=2.0)  # fmt: skip

    reports = daemon.run_maintenance(context, dry_run=True)

    assert reports == {"a-fails": ["failed: host unreachable"],
                       "b-works": ["kept 2 days, dry run True"]}  # fmt: skip


def test_entry_points_the_checkout_declares_but_the_environment_lacks_are_reported(
    tmp_path: Path,
) -> None:
    """The review: plugins register by entry point, which only an install writes; the
    service warns when a checkout declares one its environment lacks (uv sync not run)."""
    from apex.services.daemon import stale_entry_points

    (tmp_path / "pyproject.toml").write_text(
        '[project.entry-points."apex.engines"]\nert = "apex.integrations.ert.engine:ErtEngine"\n'
        'newer = "apex.execution.newer:Newer"\n'
    )
    assert stale_entry_points(tmp_path) == ["apex.engines:newer"]
    assert stale_entry_points(tmp_path / "absent") == []


def test_an_entry_point_installed_with_another_target_is_reported(tmp_path: Path) -> None:
    """A plugin that moved is installed under its old module until the
    environment is synced again; the service reports it."""
    from apex.services.daemon import stale_entry_points

    (tmp_path / "pyproject.toml").write_text(
        '[project.entry-points."apex.executors"]\nssh = "apex.execution.ssh:SshExecutor"\n'
    )
    assert stale_entry_points(tmp_path) == ["apex.executors:ssh"]


def test_an_old_distributions_entry_point_next_to_the_new_one_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's review: stale metadata of apex (ssh in
    apex.execution) next to the member's: whichever is listed last, it is reported."""
    import importlib.metadata
    from importlib.metadata import EntryPoint

    from apex.services.daemon import stale_entry_points

    new = "apex_member.execution.ssh:SshExecutor"
    (tmp_path / "pyproject.toml").write_text(
        f'[project.entry-points."apex.executors"]\nssh = "{new}"\n'
    )
    old = EntryPoint("ssh", "apex.execution.ssh:SshExecutor", "apex.executors")
    for listed in ([EntryPoint("ssh", new, "apex.executors"), old],
                   [old, EntryPoint("ssh", new, "apex.executors")]):  # fmt: skip
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group, found=listed: found)
        assert stale_entry_points(tmp_path) == ["apex.executors:ssh"]


def test_a_workspace_members_entry_points_count_as_declared(tmp_path: Path) -> None:
    """A plain ``uv sync`` installs only the root; the service warns about
    a member's entry points too, read generically from the workspace's members."""
    from apex.services.daemon import stale_entry_points

    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\n'
        '[project.entry-points."apex.engines"]\nert = "apex.integrations.ert.engine:ErtEngine"\n'
    )
    member = tmp_path / "packages" / "apex-other"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text(
        '[project.entry-points."apex.audience"]\nother = "apex_other"\n'
    )
    assert stale_entry_points(tmp_path) == ["apex.audience:other"]


def test_excluded_members_and_nested_globs_are_not_reported_twice(tmp_path: Path) -> None:
    """An excluded member is no package of the workspace, and ``**``, which
    also yields the folders above a member, reports each entry point once."""
    from apex.services.daemon import stale_entry_points

    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/**", "packages/*", ""]\n'
        'exclude = ["packages/old"]\n'
    )
    for name in ("new", "old"):
        (tmp_path / "packages" / name).mkdir(parents=True)
        (tmp_path / "packages" / name / "pyproject.toml").write_text(
            f'[project.entry-points."apex.audience"]\n{name} = "apex_{name}"\n'
        )
    assert stale_entry_points(tmp_path) == ["apex.audience:new"]


def test_gc_options_go_without_a_contributed_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """The review: ``apex gc --code SUB`` would silently ignore ``--code``; core keeps the
    guard with no package installed (a stand-in subcommand here)."""
    from apex.cli.main import app
    from apex.cli.services_commands import gc_app

    sub = typer.Typer()
    sub.command("run")(_hello)
    monkeypatch.setattr(gc_app, "registered_groups", list(gc_app.registered_groups))
    gc_app.add_typer(sub, name="standin")
    result = CliRunner().invoke(app, ["gc", "--code", "standin", "--dry-run"],
                                standalone_mode=False)  # fmt: skip
    assert isinstance(result.exception, typer.BadParameter)
    assert "without a subcommand" in str(result.exception)
