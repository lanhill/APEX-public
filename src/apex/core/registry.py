"""Generic plugin registry.

Every extensible subsystem (methods, simulators, executors, ...) owns one `Registry`.
Plugins register either in code with the `Registry.register` decorator or from
another installed package through the entry-point group ``apex.<kind>``:

.. code-block:: toml

    [project.entry-points."apex.methods"]
    my_method = "my_package.methods:MyMethod"
"""

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import cast

from apex.core.errors import DuplicatePluginError, UnknownPluginError

_logger = logging.getLogger(__name__)


def _is_or_contains(missing: str | None, module: str) -> bool:
    """Whether ``missing`` is ``module`` or one of its parent packages."""
    return missing is not None and (module == missing or module.startswith(missing + "."))


class Registry[T]:
    """Name-to-factory mapping for one kind of plugin.

    Parameters
    ----------
    kind
        Plural plugin kind, e.g. ``"methods"``. It names the entry-point group
        ``apex.<kind>`` and appears in error messages.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}
        self._origins: dict[str, str] = {}
        self._entry_points_loaded = False

    @property
    def entry_point_group(self) -> str:
        """Entry-point group scanned for externally installed plugins."""
        return f"apex.{self.kind}"

    def add(self, name: str, factory: Callable[..., T], origin: str | None = None) -> None:
        """Register ``factory`` under ``name``; ``origin``: the distribution providing it.

        Raises
        ------
        DuplicatePluginError
            If ``name`` is already registered. Silent overrides would make the
            selected implementation depend on import order.
        """
        if name in self._factories:
            raise DuplicatePluginError(f"{self.kind} plugin {name!r} is already registered")
        self._factories[name] = factory
        if origin is not None:
            self._origins[name] = origin

    def remove(self, name: str) -> None:
        """Unregister ``name`` (for tests that add a plugin of their own; unknown: no-op).

        An entry-point plugin removed so stays removed: entry points load once.
        """
        self._factories.pop(name, None)
        self._origins.pop(name, None)

    def register[F: Callable[..., object]](self, name: str) -> Callable[[F], F]:
        """Return a decorator that registers a class or factory function under ``name``."""

        def decorator(factory: F) -> F:
            # Callers are trusted to register factories that produce `T`; the decorator
            # returns `factory` unchanged so decorated classes keep their own type.
            self.add(name, cast("Callable[..., T]", factory))
            return factory

        return decorator

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Instantiate the plugin registered under ``name``.

        Raises
        ------
        UnknownPluginError
            If no plugin is registered under ``name``; the message lists the
            available names.
        """
        self._load_entry_points()
        try:
            factory = self._factories[name]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise UnknownPluginError(
                f"unknown {self.kind} plugin {name!r}; available: {available}"
            ) from None
        return factory(*args, **kwargs)

    def factory(self, name: str) -> Callable[..., T]:
        """Return the factory registered under ``name`` (without calling it).

        Raises
        ------
        UnknownPluginError
            If no plugin is registered under ``name``.
        """
        self._load_entry_points()
        try:
            return self._factories[name]
        except KeyError:
            available = ", ".join(self.names()) or "none"
            raise UnknownPluginError(
                f"unknown {self.kind} plugin {name!r}; available: {available}"
            ) from None

    def origin(self, name: str) -> str | None:
        """Return the distribution that provides ``name`` by entry point.

        ``None`` for a plugin registered in code, or unknown.
        """
        self._load_entry_points()
        return self._origins.get(name)

    def names(self) -> list[str]:
        """Return the sorted names of all registered plugins."""
        self._load_entry_points()
        return sorted(self._factories)

    def registered(self) -> list[str]:
        """Return the names in the order they were registered (an alias after its plugin)."""
        self._load_entry_points()
        return list(self._factories)

    def _load_entry_points(self) -> None:
        # Loaded lazily, once, so importing a registry never triggers plugin imports.
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        for entry_point in entry_points(group=self.entry_point_group):
            dist = entry_point.dist
            try:
                factory = entry_point.load()
            except ModuleNotFoundError as error:
                # An environment installed before its plugin moved: every other
                # plugin stays usable, and this one is unknown, so a configuration naming it
                # is refused by validation rather than every command failing. Only its own
                # module gone: any other import error is a bug, raised as before.
                if not _is_or_contains(error.name, entry_point.module):
                    raise
                _logger.error(
                    "the plugin %s of %s cannot be loaded (%s): this environment is older "
                    "than its code; run uv sync --inexact --all-packages",
                    entry_point.name, self.entry_point_group, error,
                    extra={"event": "plugin_unloadable", "entry_point": entry_point.value},
                )  # fmt: skip
                continue
            self.add(entry_point.name, factory, None if dist is None else dist.name)
