"""REST routes and web-UI pages installed packages add.

An entry point of the group ``apex.api_routes`` names a function ``(app, context)`` that
adds its routes to the FastAPI app, with the :class:`RouteContext`: the service, the
owner-only check, the problem a route raises for a client's error, and the model a route
may ask (replaced in tests). A route takes the request's guard as core's do
(``who: apex.api.app.Who``). A package that fails to add its routes is logged and skipped;
the API serves the rest.

An entry point of the group ``apex.ui_pages`` names a function returning a :class:`UiPage`:
a page built on its own (a folder with ``index.html``), served at ``/ui/pages/<name>/``
next to the web UI, which lists it (``GET /v1/ui/pages``) to whom may see it. Its data
come from the REST API with the web UI's sign-in, as the web UI's own pages do.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path

from fastapi import FastAPI

from apex.api.guard import Denied, Guard
from apex.core.llm import LLMProvider
from apex.services.api import ApexService
from apex.services.settings import LLMSection

#: The entry-point group of the routes installed packages add.
ROUTES_GROUP = "apex.api_routes"

logger = logging.getLogger("apex.service")


@dataclass(frozen=True)
class RouteContext:
    """What a package's routes get from the API.

    Attributes
    ----------
    service
        The service the API serves.
    problem
        ``problem(status, title, detail)``: the exception a route raises for a client's
        error, answered as an RFC 9457 problem.
    provider
        The model the API's tests put in place of the configured ones, or ``None``.
    """

    service: ApexService
    problem: Callable[[int, str, str], Exception]
    provider: Callable[[], LLMProvider] | None = None

    def owner(self, who: Guard) -> None:
        """Refuse anyone but the owner outside the web UI (guests see nothing: a 404)."""
        if who.ui:
            raise Denied(404, "not found")

    def model(self, wanted: bool, settings: LLMSection) -> LLMProvider | None:
        """Return the model a route asks for: none, the tests', or the one ``settings`` name."""
        if not wanted:
            return None
        if self.provider is not None:
            return self.provider()
        from apex.agents.providers import provider_or_rules

        return provider_or_rules(settings)


def install_routes(app: FastAPI, context: RouteContext) -> list[str]:
    """Add every installed package's routes; return the names of those added."""
    added = []
    for entry_point in entry_points(group=ROUTES_GROUP):
        try:
            register: Callable[[FastAPI, RouteContext], None] = entry_point.load()
            register(app, context)
        except Exception as error:  # noqa: BLE001 (a package's routes must not stop the API)
            extra = {"event": "routes_unloadable", "routes": entry_point.name}
            logger.error("the routes %s cannot be added: %s", entry_point.value, error,
                         extra=extra)  # fmt: skip
            continue
        added.append(entry_point.name)
    return added


#: The entry-point group of the web-UI pages installed packages add.
UI_PAGES_GROUP = "apex.ui_pages"


@dataclass(frozen=True)
class UiPage:
    """A web-UI page a package adds.

    Attributes
    ----------
    name
        Its path: ``/ui/pages/<name>/`` (letters, digits, ``-`` and ``_``).
    title
        What the web UI's navigation calls it.
    folder
        The built page: ``index.html`` and its assets.
    owner_only
        Listed to the owner only (never to guests); its routes check it too.
    listed
        Whether the navigation lists it, by the service (e.g. a setting of its own); a page
        not listed is still served at its address.
    """

    name: str
    title: str
    folder: Path
    owner_only: bool = True
    listed: Callable[[ApexService], bool] | bool = True

    def shown(self, service: ApexService) -> bool:
        """Whether the navigation lists the page: built, and listed."""
        if not (self.folder / "index.html").is_file():
            return False
        return self.listed(service) if callable(self.listed) else self.listed


def ui_pages() -> dict[str, UiPage]:
    """Return the installed packages' web-UI pages by name (a broken one is logged)."""
    found: dict[str, UiPage] = {}
    for entry_point in entry_points(group=UI_PAGES_GROUP):
        try:
            page = entry_point.load()()
        except Exception as error:  # noqa: BLE001 (a package's page must not stop the API)
            logger.error("the web-UI page %s cannot be added: %s", entry_point.value, error,
                         extra={"event": "ui_page_unloadable"})  # fmt: skip
            continue
        if not isinstance(page, UiPage) or not _NAME.fullmatch(page.name) or page.name in found:
            logger.error("%s gives no usable web-UI page", entry_point.value,
                         extra={"event": "ui_page_unloadable"})  # fmt: skip
            continue
        found[page.name] = page
    return found


_NAME = re.compile(r"[A-Za-z0-9_-]+")
