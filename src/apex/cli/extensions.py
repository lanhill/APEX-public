"""``apex`` commands other packages contribute by entry point.

An entry point of the group ``apex.cli`` names a function returning ``(parent, name, item,
hidden, private)``: ``parent`` is ``""`` (the top level) or an existing group (``"agent"``
for a contributed agent's command, ``"code"``, ``"gc"``), ``item`` a Typer app (a group)
or a function (a command), ``hidden`` keeps it out of ``--help``, and ``private`` out of the
public documentation (:data:`PRIVATE`). The core CLI names no contributor; a broken one is
reported and skipped, never stopping ``apex``, and none may replace a command already there.
"""

import logging
from collections.abc import Mapping
from importlib.metadata import entry_points

import typer

#: The entry-point group of contributed commands.
CLI_GROUP = "apex.cli"

#: The full names (``"apex gc hosts"``) of the contributed commands that are private: the
#: documentation inventory lists them as such.
PRIVATE: set[str] = set()

logger = logging.getLogger(__name__)


def install(groups: Mapping[str, typer.Typer]) -> list[str]:
    """Add every contributed command to its group; return their full names."""
    added = []
    for entry_point in entry_points(group=CLI_GROUP):
        try:
            parent, name, item, hidden, private = entry_point.load()()
            target = groups[parent]
        except Exception as error:  # noqa: BLE001  (a broken contributor never stops apex)
            logger.warning("apex.cli entry point %s cannot be used: %r", entry_point.name, error)
            continue
        full = " ".join(word for word in ("apex", parent, name) if word)
        if _taken(target, name):
            logger.warning("apex.cli entry point %s would replace %r; skipped",
                           entry_point.name, full)  # fmt: skip
            continue
        if isinstance(item, typer.Typer):
            target.add_typer(item, name=name, hidden=hidden)
        elif callable(item):
            target.command(name, hidden=hidden)(item)
        else:
            logger.warning("apex.cli entry point %s gives no command", entry_point.name)
            continue
        if private:
            PRIVATE.add(full)
        added.append(full)
    return added


def _taken(group: typer.Typer, name: str) -> bool:
    commands = {c.name or getattr(c.callback, "__name__", "").replace("_", "-")
                for c in group.registered_commands}  # fmt: skip
    groups = {g.name for g in group.registered_groups}
    return name in commands or name in groups
