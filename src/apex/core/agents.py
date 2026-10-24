"""Runtime agents other packages contribute.

An entry point of the group ``apex.agents`` names a function returning an
:class:`AgentContribution`: the agent as the catalogue lists it, and its ``[agents.<name>]``
section of ``service.toml`` (a pydantic model whose fields all have defaults). A contributed
agent is listed as core's are: the catalogue (``agents.catalogue.AGENTS``) names it, the
settings validate its section, the documentation inventory lists it. Its visibility
follows its package's audience: an agent of a package that is not public is private, and
``[agents.visibility]`` cannot make it public.

The entry point's module is loaded while ``services.settings`` is imported, so it imports
nothing of ``apex`` but this module, ``apex.core`` and ``apex.domain`` (the agent itself
lives elsewhere, imported by its command). A broken contribution is logged and skipped,
never stopping APEX.
"""

import functools
import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint, entry_points
from typing import Literal

from pydantic import BaseModel, ValidationError

from apex.core import import_rules

#: The entry-point group of contributed agents.
AGENTS_GROUP = "apex.agents"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentInfo:
    """One of APEX's runtime agents, as the chat and the UI list it.

    ``visibility``: ``public`` agents are shown in the web UI and named by the assistant;
    ``private`` ones are not (``[agents.visibility]`` in ``service.toml`` overrides it).
    """

    name: str
    summary: str
    command: str
    tool: str | None
    adr: str
    visibility: Literal["public", "private"] = "public"


@dataclass(frozen=True)
class AgentContribution:
    """An agent a package contributes.

    Attributes
    ----------
    info
        The agent as the catalogue lists it. ``tool`` must be ``None``: the assistant's
        tools are core's.
    settings
        Its ``[agents.<name>]`` section (``-`` in the name read as ``_``), if any; every
        field has a default.
    distribution
        The distribution that contributed it (set by :func:`contributed_agents`).
    """

    info: AgentInfo
    settings: type[BaseModel] | None = None
    distribution: str | None = None

    @property
    def section(self) -> str:
        """The agent's field of ``[agents]`` (``an-agent``: ``an_agent``)."""
        return self.info.name.replace("-", "_")


#: What an entry module may import of APEX, directly or not: the rest imports the settings,
#: which load the contributions.
ALLOWED_IMPORTS = ("apex.core", "apex.domain")

#: ``[agents]`` keys that are not agents' sections: no contribution may take them.
RESERVED_SECTIONS = frozenset({"llm", "visibility"})


class _Contributions:
    """The contributions, loaded once: a cache, as ``functools.cache`` would be.

    Not ``functools.cache``: a contribution's module that imports what calls back here
    (``services.settings``, ``agents.catalogue``) would see an empty or partial result, and
    the catalogue and the settings would then disagree. Such a cycle is refused instead, and
    a first call from two threads loads once (the second waits).
    """

    def __init__(self) -> None:
        self.found: tuple[AgentContribution, ...] | None = None
        self.loader: int | None = None
        self.lock = threading.RLock()

    def get(self) -> tuple[AgentContribution, ...]:
        with self.lock:
            if self.found is not None:
                return self.found
            if self.loader == threading.get_ident():
                raise ImportError(
                    "a contributed agent's entry module imports apex.services or apex.agents "
                    "while the contributions load: it may import only apex.core and "
                    "apex.domain"
                )
            self.loader = threading.get_ident()
            try:
                self.found = _load()
            finally:
                self.loader = None
            return self.found


#: The process's contributions (a cache; the entry points do not change while it runs).
_CONTRIBUTIONS = _Contributions()


def contributed_agents() -> tuple[AgentContribution, ...]:
    """Return every contributed agent, each once, in the entry points' order.

    Raises
    ------
    ImportError
        If an entry module is being imported (it imported, directly or not, what loads the
        contributions): its agent would be missing from some of APEX's views.
    """
    return _CONTRIBUTIONS.get()


def _clear_for_tests() -> None:
    # The catalogue's AGENTS and the settings' AgentsSection, built at their import, stay.
    with _CONTRIBUTIONS.lock:
        _CONTRIBUTIONS.found = None


def _load() -> tuple[AgentContribution, ...]:
    found: dict[str, AgentContribution] = {}
    sections: set[str] = set(RESERVED_SECTIONS)
    for entry_point in entry_points(group=AGENTS_GROUP):
        _refuse_cycle(entry_point.module)
        contribution, problem = _contribution_of(entry_point)
        if contribution is None:
            logger.warning("apex.agents entry point %s cannot be used: %s", entry_point.name,
                           problem)  # fmt: skip
            continue
        # One agent per name and per section (``a-b`` and ``a_b`` share [agents.a_b]), and
        # none on [agents]' own keys.
        if contribution.info.name in found or contribution.section in sections:
            logger.warning("apex.agents entry point %s repeats an agent or takes [agents.%s]; "
                           "skipped", entry_point.name, contribution.section)  # fmt: skip
            continue
        sections.add(contribution.section)
        dist = entry_point.dist
        found[contribution.info.name] = replace(
            contribution, distribution=None if dist is None else dist.name
        )
    return tuple(found.values())


def _contribution_of(entry_point: EntryPoint) -> tuple[AgentContribution | None, str]:
    """Return the entry point's contribution, or ``None`` and why it cannot be used.

    Its imports are checked from the sources first (whatever is already imported), then
    while it loads and its function runs, where a forbidden one is refused before it runs;
    the modules of its package that the function imported are read from their sources too.
    """
    rule = f"an entry module may import only {' and '.join(ALLOWED_IMPORTS)}"
    try:
        edges = import_rules.module_level_imports([entry_point.module], ALLOWED_IMPORTS)
        if edges:
            return None, f"{_edges(edges)}: {rule}"
        try:
            contribution, seen = import_rules.refusing_imports(
                functools.partial(_call, entry_point), ALLOWED_IMPORTS
            )
        except ImportError as error:
            return None, f"{error}: {rule}"
        # What the function imported of packages built on APEX, loaded before or not.
        followed = import_rules.built_on_apex() | {entry_point.module.partition(".")[0]}
        edges = import_rules.module_level_imports(
            sorted(name for name in seen if name.partition(".")[0] in followed),
            ALLOWED_IMPORTS,
        )
    except Exception as error:  # noqa: BLE001  (a broken contributor never stops apex)
        return None, repr(error)
    if edges:
        return None, f"{_edges(edges)}: {rule}"
    problem = _problem(contribution)
    if problem is not None or not isinstance(contribution, AgentContribution):
        return None, problem or "no contribution"
    return contribution, ""


def _call(entry_point: EntryPoint) -> object:
    return entry_point.load()()


def _edges(edges: Sequence[tuple[str, str]]) -> str:
    return "; ".join(f"{importer} imports {imported}" for importer, imported in edges)


def _refuse_cycle(module: str) -> None:
    # The entry module (or a package above it) is being imported, and its import reached
    # here: loading it now would give a partial module, so its agent would silently miss
    # from the settings built now. Refused, loudly: it is the contributor's import to fix.
    parts = module.split(".")
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        spec = getattr(sys.modules.get(name), "__spec__", None)
        if getattr(spec, "_initializing", False):
            raise ImportError(
                f"{name} imports what loads APEX's contributed agents while it is itself "
                f"being imported (the apex.agents entry point {module}): its entry module may "
                "import only apex.core and apex.domain, and its package nothing that imports "
                "apex.services or apex.agents"
            )


def _problem(contribution: object) -> str | None:
    if not isinstance(contribution, AgentContribution):
        return f"gives {type(contribution).__name__}, not an AgentContribution"
    if contribution.info.tool is not None:
        return "names a tool: the assistant's tools are core's"
    if contribution.settings is not None:
        try:
            contribution.settings()
        except (ValidationError, TypeError) as error:
            return f"its settings need a default for every field ({error})"
    return None
