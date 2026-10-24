"""Run engines: what turns a history-matching configuration into a run.

APEX's own loop is the implicit engine. Another engine (ERT, ``apex.integrations.ert``)
runs its own algorithm on the same case and writes the result as an APEX run directory, so
studies, comparisons, reports and agents read it like any run. Engines are plugins
(registry ``engines``, entry-point group ``apex.engines``): the services choose them by
name and never import them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from apex.core.progress import ProgressReporter
from apex.domain.config import HistoryMatchConfig
from apex.domain.results import HistoryMatchResult

#: APEX's own engine: the default, and never a plugin.
APEX_ENGINE = "apex"


@dataclass(frozen=True)
class EngineMethod:
    """An engine's own algorithm: e.g. ERT's ``es_mda`` with ``weights``.

    Attributes
    ----------
    engine
        The engine's registry name (``"ert"``).
    name
        The algorithm the engine runs.
    options
        Its options, checked by the engine.
    """

    engine: str
    name: str
    options: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """Return the method as JSON (job specifications, run identity)."""
        return {"engine": self.engine, "name": self.name, "options": dict(self.options)}


@dataclass
class EngineContext:
    """What a run gets from the service that runs it.

    Attributes
    ----------
    run_dir
        Where the run goes (APEX's run layout).
    progress
        Where progress is reported, if anywhere.
    """

    run_dir: Path
    progress: ProgressReporter | None = None


class RunEngine(Protocol):
    """An engine that runs a configuration's case with an algorithm of its own."""

    name: str

    def check(self, config: HistoryMatchConfig, method: EngineMethod) -> list[str]:
        """Return why the engine cannot run ``method`` on the case (empty if it can)."""
        ...

    def simulator_runs(self, method: EngineMethod, members: int | None) -> int | None:
        """Return the simulator runs of ``members`` members (``None`` if unknown)."""
        ...

    def run(
        self, config: HistoryMatchConfig, method: EngineMethod, context: EngineContext
    ) -> HistoryMatchResult:
        """Run ``method`` on the case into ``context.run_dir``; return the result."""
        ...
