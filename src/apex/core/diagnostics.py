"""Optional diagnostics of algorithms, update strategies and localizations.

Plugins may produce internals worth inspecting, such as a taper, a Kalman-gain-like
matrix or retained singular values. Some of these are ``n_state x n_data`` and too large
to keep by default, so **nothing is saved unless the user asks for it**, at two levels:

- run-wide, in ``[output] diagnostics`` (a bare name applies to every plugin that
  produces it; ``"localization.taper"`` to one plugin role), with ``diagnostics_max_mb``
  and ``diagnostics_every``;
- per plugin, in the ``diagnostics`` table of the plugin's own section, e.g.
  ``localization = { name = "adaptive_prior", diagnostics = { taper = true, max_mb = 2000 } }``.

**A plugin's own switch overrides** ``[output]``, both ways (``name = true`` saves an item
``[output]`` does not list; ``name = false`` suppresses one it does). Precedence: the
plugin's switch, then a qualified ``[output]`` name, then a bare one, then off; a plugin's
``max_mb`` and ``every`` override the run-wide ones. The run logs the resulting plan
(event ``diagnostics_plan``), warns when a plugin switch contradicts ``[output]``, and
writes the plan to ``checkpoints/diagnostics_plan.json``.

Plugins declare what they can produce (``available_diagnostics``) and call
:meth:`Diagnostics.wants` before forming an item, so an unrequested matrix is never
computed; a callable item is only evaluated when it is wanted and fits the size limit.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

#: An item, or a callable producing it (evaluated only when the item is wanted).
Item = NDArray[Any] | Callable[[], NDArray[Any]]

#: Keys of a plugin's ``diagnostics`` table that are settings, not item names.
SETTINGS = ("max_mb", "every")


class Diagnostics(Protocol):
    """Where plugins put their diagnostics for the current step."""

    def wants(self, role: str, name: str) -> bool:
        """Whether item ``name`` of the plugin in ``role`` (e.g. ``"localization"``) is saved."""
        ...

    def add(self, role: str, name: str, item: Item, *, nbytes: int | None = None) -> None:
        """Offer an item (``nbytes``: its size, if known before forming it)."""
        ...


class NoDiagnostics:
    """Saves nothing (the default)."""

    def wants(self, role: str, name: str) -> bool:
        """Return ``False``: nothing is requested."""
        return False

    def add(self, role: str, name: str, item: Item, *, nbytes: int | None = None) -> None:
        """Discard ``item``."""


class EmitsDiagnostics:
    """Mixin for plugins that can produce diagnostics.

    Subclasses list what they can produce in ``available_diagnostics`` (name to
    description); the algorithm that owns them attaches the run's sink.
    """

    #: Diagnostics this plugin can produce, name -> description (shapes included).
    available_diagnostics: ClassVar[Mapping[str, str]] = {}
    _diagnostics: Diagnostics = NoDiagnostics()

    def attach_diagnostics(self, sink: Diagnostics) -> None:
        """Use ``sink`` for this plugin's diagnostics."""
        self._diagnostics = sink


@dataclass(frozen=True)
class PlanItem:
    """One diagnostic the run saves.

    ``source`` says which setting switched it on: ``"plugin"`` (the plugin's own
    ``diagnostics`` table) or ``"[output]"``.
    """

    role: str
    name: str
    description: str
    max_mb: float
    every: int
    source: Literal["plugin", "[output]"]


@dataclass(frozen=True)
class DiagnosticsPlan:
    """What a run saves, resolved from ``[output]`` and the plugins' own switches.

    Build it with :meth:`build`; ``warnings`` lists plugin switches that contradict
    ``[output]`` (the run logs them).
    """

    items: Mapping[tuple[str, str], PlanItem]
    warnings: Sequence[str] = ()

    @classmethod
    def build(
        cls,
        available: Mapping[str, Mapping[str, str]],
        requested: Sequence[str] = (),
        overrides: Mapping[str, Mapping[str, bool | float]] | None = None,
        *,
        max_mb: float = 200.0,
        every: int = 1,
    ) -> "DiagnosticsPlan":
        """Resolve the plan.

        Parameters
        ----------
        available
            Per plugin role (``"method"``, ``"strategy"``, ``"localization"``), the
            diagnostics its plugin can produce (name to description).
        requested
            ``[output] diagnostics``: bare names or ``"role.name"``.
        overrides
            Per role, the plugin's own ``diagnostics`` table: ``name = true | false``,
            and ``max_mb`` / ``every``.
        max_mb, every
            Run-wide size limit per item and step, and saving interval.

        Raises
        ------
        ValueError
            For a name no configured plugin produces, a role without a plugin, or
            invalid settings.
        """
        overrides = dict(overrides or {})
        if not math.isfinite(max_mb) or max_mb <= 0:
            raise ValueError(f"diagnostics_max_mb must be positive, got {max_mb}")
        if every < 1:
            raise ValueError(f"diagnostics_every must be at least 1, got {every}")
        bare: set[str] = set()
        qualified: set[tuple[str, str]] = set()
        for entry in requested:
            role, _, name = entry.rpartition(".")
            if role:
                if name not in available.get(role, {}):
                    raise ValueError(
                        f"[output] diagnostics: the {role!r} plugin produces no {name!r} "
                        f"(it produces {sorted(available.get(role, {})) or 'none'})"
                    )
                qualified.add((role, name))
            else:
                if not any(name in names for names in available.values()):
                    raise ValueError(
                        f"[output] diagnostics: no configured plugin produces {name!r}; "
                        f"available: {_listing(available)}"
                    )
                bare.add(name)
        items: dict[tuple[str, str], PlanItem] = {}
        warnings: list[str] = []
        for role, names in available.items():
            table = dict(overrides.pop(role, {}))
            role_max = float(table.pop("max_mb", max_mb))
            role_every = int(table.pop("every", every))
            if role_max <= 0 or role_every < 1:
                raise ValueError(f"{role} diagnostics: max_mb must be > 0 and every >= 1")
            for name, switch in table.items():
                if name not in names:
                    raise ValueError(
                        f"{role} diagnostics: the plugin produces no {name!r} "
                        f"(it produces {sorted(names) or 'none'})"
                    )
                if not isinstance(switch, bool):
                    raise ValueError(f"{role} diagnostics: {name} must be true or false")
            for name, description in names.items():
                from_output = (role, name) in qualified or name in bare
                if name in table:
                    on = bool(table[name])
                    if from_output and not on:
                        warnings.append(
                            f"{role} diagnostics: {name} = false overrides [output] "
                            f"diagnostics, so {role}.{name} is not saved"
                        )
                    source: Literal["plugin", "[output]"] = "plugin"
                else:
                    on, source = from_output, "[output]"
                if on:
                    items[(role, name)] = PlanItem(
                        role, name, description, role_max, role_every, source
                    )
        if overrides:
            raise ValueError(
                f"diagnostics set for {', '.join(sorted(overrides))}, which produce none"
            )
        return cls(items, tuple(warnings))

    def describe(self) -> list[dict[str, Any]]:
        """Return the plan as rows (for the log and ``diagnostics_plan.json``)."""
        return [
            {
                "item": f"{i.role}.{i.name}",
                "description": i.description,
                "max_mb": i.max_mb,
                "every": i.every,
                "source": i.source,
            }
            for i in self.items.values()
        ]


class RecordingDiagnostics:
    """Keeps the planned items of the current iteration, within their size limits.

    The workflow sets :attr:`iteration` (the iteration the next update produces) before
    each step and collects the items with :meth:`take`.
    """

    def __init__(self, plan: DiagnosticsPlan) -> None:
        self.plan = plan
        self.iteration = 0
        self._items: dict[str, NDArray[Any]] = {}
        self._index: dict[str, str] = {}

    def wants(self, role: str, name: str) -> bool:
        """Whether ``role.name`` is planned and due at this iteration."""
        item = self.plan.items.get((role, name))
        return item is not None and self.iteration % item.every == 0

    def add(self, role: str, name: str, item: Item, *, nbytes: int | None = None) -> None:
        """Keep ``item`` if wanted and within the size limit; note why it was skipped."""
        if not self.wants(role, name):
            return
        planned = self.plan.items[(role, name)]
        key, limit = f"{role}.{name}", planned.max_mb * 1e6
        if nbytes is not None and nbytes > limit:
            self._index[key] = f"skipped: {nbytes / 1e6:.1f} MB exceeds max_mb = {planned.max_mb}"
            return
        array = np.asarray(item() if callable(item) else item)
        if array.nbytes > limit:
            self._index[key] = (
                f"skipped: {array.nbytes / 1e6:.1f} MB exceeds max_mb = {planned.max_mb}"
            )
            return
        self._items[key] = array
        self._index[key] = f"saved ({planned.source})"

    def take(self) -> tuple[dict[str, NDArray[Any]], dict[str, str]]:
        """Return and clear the items kept since the last call, and the index of decisions."""
        items, index = self._items, self._index
        self._items, self._index = {}, {}
        return items, index


def _listing(available: Mapping[str, Mapping[str, str]]) -> str:
    names = [f"{role}.{name}" for role, names in available.items() for name in names]
    return ", ".join(sorted(names)) or "none"
