"""Routines installed packages run in the job service.

An entry point of the group ``apex.service_routines`` names a factory ``(root) -> routine |
None``: ``None`` when the routine's own settings switch it off. The service calls each
routine after every scheduling pass with its :class:`~apex.services.jobs.LocalJobRunner`;
a routine returns quickly (its work runs in a thread of its own) and keeps its own timing
and records. ``close()``, if it has one, is called when the service stops.

A routine that fails to load, or fails on a tick, is logged and never stops the service.
"""

import logging
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, runtime_checkable

from apex.services.jobs import LocalJobRunner

#: The entry-point group of the service's routines.
ROUTINES_GROUP = "apex.service_routines"

logger = logging.getLogger("apex.service")


@runtime_checkable
class ServiceRoutine(Protocol):
    """What the service calls after each scheduling pass."""

    def __call__(self, runner: LocalJobRunner) -> None:
        """Do (or start in the background) what is due; return quickly."""
        ...


def routines(root: Path) -> list[ServiceRoutine]:
    """Return the installed routines that their settings switch on for ``root``.

    None when ``[service] routines`` is off.
    """
    from apex.services.settings import load_settings

    if not load_settings(root).service.routines:
        return []
    found: list[ServiceRoutine] = []
    for entry_point in entry_points(group=ROUTINES_GROUP):
        try:
            factory: Callable[[Path], object] = entry_point.load()
            routine = factory(root)
        except Exception as error:  # noqa: BLE001 (a package's routine must not stop the service)
            extra = {"event": "routine_unloadable", "routine": entry_point.name}
            logger.error("the service routine %s cannot start: %s", entry_point.value, error,
                         extra=extra)  # fmt: skip
            continue
        if routine is None:
            continue
        if not isinstance(routine, ServiceRoutine):
            extra = {"event": "routine_unloadable", "routine": entry_point.name}
            logger.error("%s gives no service routine", entry_point.value, extra=extra)
            continue
        found.append(routine)
    return found


class ServiceHooks:
    """Several routines as the service's one hook: called in turn, closed together."""

    def __init__(self, hooks: Sequence[Callable[[LocalJobRunner], None]]) -> None:
        self.hooks = list(hooks)

    def __call__(self, runner: LocalJobRunner) -> None:
        """Call every routine; one that fails is logged and the others still run."""
        for hook in self.hooks:
            try:
                hook(runner)
            except Exception:  # (one routine must not stop the others)
                logger.exception("service routine failed", extra={"event": "hook_failed"})

    def close(self) -> None:
        """Close every routine that can be closed."""
        for hook in self.hooks:
            close = getattr(hook, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # (closing one must not keep the others open)
                    logger.exception("closing a service routine failed",
                                     extra={"event": "hook_close_failed"})  # fmt: skip
