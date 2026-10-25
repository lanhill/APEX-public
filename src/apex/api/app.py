"""The REST API of APEX: the service calls over HTTP, one to one.

Every ``/v1`` route but ``/v1/health`` needs ``Authorization: Bearer <token>`` (the root's
``api_token``, :func:`apex.services.auth.api_token`). Submissions return ``202 Accepted``
with the job's URL; ``POST /v1/runs`` is idempotent by run id. Errors are RFC
9457 problem documents (``application/problem+json``); a submission that needs approval
is ``403`` with the pending ``token``, which a person approves with ``apex approve``.
Events stream as Server-Sent Events (``Accept: text/event-stream``, resumable with
``Last-Event-ID``) or come as JSON pages (``?since=``). Lists are paged with cursors.

With the ``mcp`` extra, the app also serves APEX's MCP server over Streamable HTTP at
``/mcp``, behind the same bearer token; each MCP session gets its own
tool session (:class:`apex.services.mcp.HttpSessions`), named by ``X-APEX-Client``.
"""

import base64
import dataclasses
import datetime as dt
import functools
import json
import logging
import os
import time
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apex.api.extensions import RouteContext, UiPage, install_routes, ui_pages
from apex.api.guard import Denied, Guard
from apex.api.scrub import STATE_KEY, GuestPaths
from apex.core.engines import EngineMethod
from apex.core.errors import ApexError, ConfigError, validation_problems
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.domain.optimization_study import OptimizationStudySpec
from apex.domain.prediction import PredictionConfig
from apex.domain.services import (
    Comparison,
    JobHandle,
    JobStatus,
    OptimizationSummary,
    PluginInfo,
    PredictionSummary,
    RunSteps,
    RunSummary,
    Validation,
)
from apex.domain.study import StudySpec
from apex.services import policy
from apex.services.access import UI_CLIENT
from apex.services.api import ApexService, SchemaKind
from apex.services.approval import ApprovalRequired
from apex.services.auth import (
    OWNER_UI_FILE,
    api_token,
    expires_at,
    guest_token_files,
    token_matches,
)
from apex.services.identities import ui_identities
from apex.services.jobs import FINAL
from apex.services.optimization_study import is_optimization_study
from apex.services.workspace import WorkspaceError, WriteConflict

if TYPE_CHECKING:
    from apex.core.llm import LLMProvider
    from apex.services.chat_models import Asker, ModelChoice
    from apex.storage.chat_store import ChatStore

PROBLEM = "application/problem+json"


def _mcp_endpoint(
    service: ApexService, expected: Callable[[], str], host: str
) -> tuple[Any, Callable[[Any], AbstractAsyncContextManager[None]]] | None:
    """Return the ``/mcp`` endpoint (ASGI, behind the token) and its lifespan, or ``None``.

    ``None`` without the ``mcp`` extra: the REST API is served alone.
    """
    try:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager  # noqa: F401
    except ImportError:
        return None
    from apex.services.mcp import UNAUTHORISED, HttpSessions, bearer_of, build_server

    server = build_server(HttpSessions(service))
    # Builds the session manager; its own Starlette app is not used, only its handler.
    inner = server.streamable_http_app(streamable_http_path="/mcp", host=host).routes[0].endpoint
    manager = server.session_manager

    class Endpoint:
        """The MCP handler behind the bearer token.

        A class, so that Starlette routes every method to it as an ASGI app: the
        transport uses POST, GET and DELETE.
        """

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
            if not token_matches(bearer_of(scope), expected()):
                response = _problem(401, "Unauthorised", UNAUTHORISED)
                await response(scope, receive, send)
                return
            await inner(scope, receive, send)

    endpoint = Endpoint()

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:  # noqa: ANN401 (FastAPI)
        async with manager.run():
            yield

    return endpoint, lifespan


Priority = Literal["high", "normal", "low"]


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfigBody(_Body):
    """A configuration: a file on the server, or the document itself."""

    config_file: str | None = Field(default=None, description="A TOML file on the server.")
    config: dict[str, Any] | None = Field(default=None, description="The configuration itself.")
    config_text: str | None = Field(
        default=None, description="The configuration as TOML text (validation only)."
    )
    base_dir: str | None = Field(
        default=None,
        description="Where relative paths of `config` or `config_text` resolve (on the server).",
    )


#: The code a job runs: live, frozen, or a commit; only the owner chooses.
CodeChoice = Annotated[
    str | None,
    Field(
        pattern=r"^[0-9A-Za-z][0-9A-Za-z._/-]{0,99}$",
        description="The code the job runs: 'live' (default), 'frozen' (the last commit; a "
        "resume or prediction: the run's own) or a commit. The owner only.",
    ),
]


class RunBody(ConfigBody):
    """``POST /v1/runs``."""

    annotations: dict[str, str] = Field(default_factory=dict)
    max_steps: int | None = None
    force_new: bool = False
    priority: Priority = "normal"
    client: str = "api"
    approval_token: str | None = None
    code: CodeChoice = None


class StudyBody(_Body):
    """``POST /v1/studies``."""

    study_file: str | None = None
    study: dict[str, Any] | None = None
    base_dir: str | None = None
    priority: Priority = "normal"
    client: str = "api"
    approval_token: str | None = None
    code: CodeChoice = None


class PredictionBody(_Body):
    """``POST /v1/predictions``."""

    prediction_file: str | None = None
    config: dict[str, Any] | None = None
    base_dir: str | None = None
    annotations: dict[str, str] = Field(default_factory=dict)
    priority: Priority = "normal"
    client: str = "api"
    code: CodeChoice = None


class OptimizationBody(_Body):
    """``POST /v1/optimizations``."""

    optimization_file: str | None = None
    config: dict[str, Any] | None = None
    base_dir: str | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    annotations: dict[str, str] = Field(default_factory=dict)
    priority: Priority = "normal"
    client: str = "api"
    approval_token: str | None = None
    code: CodeChoice = None


class ResumeBody(_Body):
    """``POST /v1/jobs/{id}:resume``."""

    max_steps: int | None = None
    code: CodeChoice = None


class ExampleCopyBody(_Body):
    """``POST /v1/examples:copy``: which example to copy into one's own folder."""

    id: str = Field(min_length=1, description="The example's id, as GET /v1/examples lists it.")


class FileBody(_Body):
    """``PUT /v1/files/content``: a workspace file's new text."""

    path: str
    text: str
    sha256: str | None = Field(
        description="The SHA-256 the editor read; null creates a new file (which must not exist)."
    )


class Page(BaseModel):
    """A page of a list, with the cursor of the next one (``None``: the last page)."""

    items: list[dict[str, Any]]
    next_cursor: str | None


class _Problem(Exception):
    def __init__(self, status: int, title: str, detail: str, **extra: Any) -> None:  # noqa: ANN401
        super().__init__(detail)
        self.status, self.title, self.detail, self.extra = status, title, detail, extra


def _problem(status: int, title: str, detail: str, **extra: Any) -> JSONResponse:  # noqa: ANN401
    slug = title.lower().replace(" ", "-")
    body = {
        "type": f"https://apex/problems/{slug}",
        "title": title,
        "status": status,
        "detail": detail,
    }
    return JSONResponse({**body, **extra}, status_code=status, media_type=PROBLEM)


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode()).decode()


def _offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (ValueError, UnicodeDecodeError) as error:
        raise _Problem(400, "Bad cursor", f"cursor {cursor!r} is not one this API gave") from error


def _page(items: list[dict[str, Any]], cursor: str | None, limit: int) -> Page:
    start = _offset(cursor)
    chunk = items[start : start + limit]
    more = start + limit < len(items)
    return Page(items=chunk, next_cursor=_cursor(start + limit) if more else None)


def _kind(path: Path) -> str:
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise _Problem(422, "Unreadable configuration", str(error)) from error
    if "optimization" in document and "optimizer" in document:
        return "optimization"
    first = next(iter(document), "")
    return {"study": "study", "prediction": "prediction"}.get(first, "history_match")


_log = logging.getLogger("apex.service")


def _authorised(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_apex_client: Annotated[str | None, Header()] = None,
) -> Guard:
    """Return the request's guard: the owner's token, or the UI token.

    A UI user's answers then go out without server paths (:class:`GuestPaths`).
    """
    who = _guard(request, authorization, x_apex_client)
    if who.ui:
        setattr(request.state, STATE_KEY, who.scrubber())
    return who


def _guard(request: Request, authorization: str | None, x_apex_client: str | None) -> Guard:
    """Return the guard of a request's token (see :func:`_authorised`).

    The web UI sends ``X-APEX-Client: ui`` and signs in with a web-UI token, which expires:
    the owner's web-UI token sees only the owner's own work, not the guests',
    and a guest's only theirs. The owner's API token, which does not
    expire, is for other clients, and is refused from the web UI.
    """
    service: ApexService = request.app.state.service
    given = authorization.removeprefix("Bearer ").strip() if authorization else None
    days = service.jobs.settings.ui.token_days
    if token_matches(given, request.app.state.expected()):
        if x_apex_client == UI_CLIENT:
            raise _Problem(
                401, "Web-UI token required",
                "the web UI signs in with a web-UI token, which expires; the owner's is "
                "printed by `apex service token --ui --owner` on the service's machine",
            )  # fmt: skip
        return Guard(service, "owner")
    # Web-UI tokens, read at each request: created, rotated or revoked at any time.
    owner_file = service.root / OWNER_UI_FILE
    if owner_file.exists() and token_matches(given, owner_file.read_text().strip()):
        expires = _unexpired(
            owner_file, days, "apex service token --ui --owner --rotate", guest=False
        )
        return Guard(service, "owner", web_ui=True, expires=expires)
    for guest, path in guest_token_files(service.root).items():
        if token_matches(given, path.read_text().strip()):
            command = f"apex service token --ui --guest {guest} --rotate"
            return Guard(service, "ui", guest, expires=_unexpired(path, days, command, guest=True))
    # The kinds of web-UI user other packages add (their tokens issued by those packages).
    for identity in ui_identities():
        for name, path in identity.token_files(service.root).items():
            if token_matches(given, path.read_text().strip()):
                expires = _unexpired(path, days, None, guest=False)
                return Guard(service, "ui", name, identity=identity, expires=expires)
    raise _Problem(401, "Unauthorised", "send Authorization: Bearer <apex service token>")


def _unexpired(token_file: Path, days: float, command: str | None, *, guest: bool) -> dt.datetime:
    """Return when a web-UI token expires, or refuse it if it has.

    A guest is told to ask the owner; the command that issues a new token goes to the log
    only, since a guest can't run it (the sign-in page is open to anyone).
    """
    expires = expires_at(token_file, days)
    if dt.datetime.now(dt.UTC) >= expires:
        when = expires.isoformat(timespec="seconds")
        if guest:
            _log.info("expired web-UI token refused; a new one: %s", command)
            detail = f"your web-UI token expired on {when}; ask the service's owner for a new one"
        elif command is None:  # a user of an added kind: its package issues a new one
            detail = f"your web-UI token expired on {when}"
        else:
            detail = f"your web-UI token expired on {when}; issue a new one with {command}"
        raise _Problem(401, "Token expired", detail, expires=when)
    return expires


#: The guard of a request, as a route parameter.
Who = Annotated[Guard, Depends(_authorised)]


def _checked(
    who: Guard,
    validation: Validation,
    configs: "Callable[[], Sequence[HistoryMatchConfig | OptimizationConfig]]",
) -> Validation:
    """Add what UI users may not use to a validation's errors.

    Plugins, and files and programs.
    """
    if not who.ui or not validation.valid:
        return validation
    loaded = configs()
    runs = [c for c in loaded if isinstance(c, HistoryMatchConfig)]
    hidden = sorted({p for c in runs for p in who.access.hidden_plugins(c)})
    if hidden:
        errors = [f"plugins not available in the UI: {', '.join(hidden)}"]
    else:
        paths = who.paths()
        errors = sorted({r for c in runs for r in who.access.refused_names(c, paths)})
    for optimization in (c for c in loaded if not isinstance(c, HistoryMatchConfig)):
        try:
            who.optimization_config(optimization)  # its plugins, files and programs
        except Denied as error:
            errors.append(str(error))
    if not errors:
        return validation
    return validation.model_copy(update={"valid": False, "errors": errors})


def _checked_optimizations(
    who: Guard,
    service: ApexService,
    validation: Validation,
    study: "tuple[OptimizationStudySpec, Path] | None",
) -> Validation:
    """Add the cells UI users may not run to a study of optimizations' errors."""
    if not who.ui or not validation.valid or study is None:
        return validation
    from apex.services.optimization_study import expand

    refused: list[str] = []
    for cell in expand(service, *study):
        try:
            who.optimization_config(cell.config)
        except Denied as error:
            refused.append(str(error))
    if not refused:
        return validation
    return validation.model_copy(update={"valid": False, "errors": sorted(set(refused))})


def _read_study(
    who: Guard, service: ApexService, path: Path
) -> tuple[StudySpec | OptimizationStudySpec, Path]:
    """Read a web-UI user's study file once: its cases checked, annotated with its submitter.

    What is checked is what is submitted: the file is not read again (a study file, or
    which cases it names, changed in between would otherwise run unchecked).
    """
    policy.check_file(path, service.jobs.settings.policy, service.root)  # before it is opened
    try:
        document = tomllib.loads(path.read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise _Problem(422, "Invalid study", f"cannot read {who.relative(path)}") from error
    base = path.resolve().parent
    _check_cases(who, service, document, base)
    model: type[StudySpec | OptimizationStudySpec] = (
        OptimizationStudySpec if is_optimization_study(document) else StudySpec
    )
    try:
        spec = model.model_validate(document)
    except ValidationError as error:
        raise _Problem(422, "Invalid study", validation_problems(error)) from error
    study = spec.study.model_copy(
        update={"annotations": who.annotations(dict(spec.study.annotations))}
    )
    return spec.model_copy(update={"study": study}), base


def _own_output(who: Guard, folder: Path | None, config: PredictionConfig) -> None:
    """Refuse a web-UI user's prediction that writes outside their own folder."""
    output = config.prediction.output_dir
    if output is None:
        return  # beside its run, in the user's folder
    if folder is None or not output.resolve().is_relative_to(folder.resolve()):
        raise Denied(403, "a guest's predictions are written in their own folder: leave "
                     "[prediction] output_dir out")  # fmt: skip


def _check_snapshot(who: Guard, record: Mapping[str, Any]) -> None:
    """Refuse a web-UI user's resume of work whose saved configuration they may not run.

    A resume runs the configuration saved with the work, not a file: work submitted before
    a rule existed (the file and program checks of 2026-10-05) is held to it now.
    """
    dirs = record.get("run_dirs") or []
    if not dirs:
        return
    run_dir = Path(dirs[0])
    try:
        if record.get("kind") in ("optimization", "optimization_resume"):
            from apex.storage.optimization_directory import read_config

            saved: HistoryMatchConfig | OptimizationConfig = OptimizationConfig.model_validate(
                read_config(run_dir)
            )
        elif record.get("kind") in ("run", "resume"):
            from apex.services.composition import snapshot_config
            from apex.storage.run_directory import RunDirectory

            saved = snapshot_config(RunDirectory(run_dir))
        else:
            return
    except (OSError, ValueError, ApexError) as error:  # ValidationError is a ValueError
        # Its message would name the server's paths: the job, not the folder.
        raise _Problem(422, "Cannot resume", "its saved configuration cannot be read") from error
    if isinstance(saved, OptimizationConfig):
        who.optimization_config(saved)
    else:
        who.config(saved)


def _check_cases(who: Guard, service: ApexService, document: dict[str, Any], base: Path) -> None:
    """Refuse a web-UI user's study whose cases lie outside their folders.

    Checked before anything reads the cases; a refusal names the case as written, not the
    server path it resolves to.
    """
    from apex.services.study import resolve_case

    study = document.get("study")
    cases = study.get("cases") if isinstance(study, dict) else None
    if not who.ui or not isinstance(cases, dict):
        return
    space = service.workspace()
    for name, written in cases.items():
        try:
            who.file(resolve_case(Path(str(written)), base, space))
        except Denied as error:
            raise Denied(error.status, f"case {name!r} ({written}) is not available") from None


def _scrubbed(who: Guard, service: ApexService, validation: Validation) -> Validation:
    """Return a validation without server paths for web-UI users.

    Errors and notes name files as the workspace shows them; the service's root as
    ``<service>`` (:meth:`Guard.scrubber`). Other clients get the paths as they are.
    """
    if not who.ui:
        return validation
    scrub = who.scrubber()
    return validation.model_copy(update={
        "errors": [scrub.text(e) for e in validation.errors],
        "notes": [scrub.text(n) for n in validation.notes],
    })  # fmt: skip


def _shown(service: ApexService, path: Path) -> str:
    """Return a path as the UI shows it: workspace-relative, else its name (no server path)."""
    try:
        return service.workspace().relative(path)
    except WorkspaceError:
        return path.name


def _as_submitted(who: Guard) -> "dict[str, Any]":
    """Return :meth:`ApexService.validate`'s keywords to judge a run as its submission would.

    The web UI's runs go to its user's folder: there, for a web-UI request. The
    folder is not created (a guest's may not exist yet: then nothing is there to collide).
    """
    folder: Path | None = None
    if who.ui:
        folder = who.own_folder(create=False)
    elif who.web_ui:
        folder = who.service.workspace().ui_user_folder(None, create=False)
    return {"check_output": True, "place_in": folder, "annotations": who.annotations({})}


def _file_configs(service: ApexService, path: Path) -> list[HistoryMatchConfig]:
    """Return the configurations a run or study file makes (a study: every cell)."""
    if _kind(path) == "study":
        from apex.services.study import plan_study

        spec, base = service.checked_study(path)
        return [cell.config for cell in plan_study(service, spec, base).cells]
    return [service.resolve(path)]


def _toml_or_none(text: str) -> dict[str, Any] | None:
    """Return parsed TOML, or ``None`` if it does not parse (the validation says why)."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None


def _text_configs(
    service: ApexService, text: str, base: Path
) -> "list[HistoryMatchConfig | OptimizationConfig]":
    """Return the configurations TOML text makes (a run's, an optimization's; a study's cells)."""
    document = tomllib.loads(text)
    if "optimization" in document and "optimizer" in document:
        return [service.resolve_optimization(document, base)]
    if "study" in document:
        from apex.services.study import plan_study

        spec = StudySpec.model_validate(document)
        return [cell.config for cell in plan_study(service, spec, base).cells]
    return [service.resolve(document, base)]


def _chat_provider(service: ApexService) -> "Callable[[], LLMProvider]":
    """Return the chat's model factory: ``[chat.llm]``, else ``[agents.llm]``."""

    def make() -> "LLMProvider":
        from apex.agents.providers import create_provider

        settings = service.jobs.settings
        return create_provider(settings.chat.llm or settings.agents.llm)

    return make


def create_app(
    root: Path | None = None,
    *,
    token: str | None = None,
    host: str = "127.0.0.1",
    chat_provider: "Callable[[], LLMProvider] | None" = None,
    agent_provider: "Callable[[], LLMProvider] | None" = None,
) -> FastAPI:
    """Build the API over the services of ``root`` (default ``$APEX_HOME``).

    ``chat_provider`` and ``agent_provider`` replace the configured models (tests).

    ``host`` is the address served: on a loopback address the MCP endpoint also refuses
    other ``Host`` headers (the MCP SDK's DNS-rebinding protection).
    """
    service = ApexService(root)
    api_token(service.root)  # create it on first use

    def expected() -> str:
        # Read at each request, so that `apex service token --rotate` takes effect at once.
        return token if token is not None else api_token(service.root)

    mcp = _mcp_endpoint(service, expected, host)
    app = FastAPI(
        title="APEX",
        version="1",
        summary="Ensemble history matching, predictions and studies.",
        lifespan=None if mcp is None else mcp[1],
    )
    if mcp is not None:
        from starlette.routing import Route

        app.router.routes.append(Route("/mcp", endpoint=mcp[0]))

    app.state.service = service
    app.state.expected = expected
    # A guest's answers carry no server paths (finding 31; set by _authorised).
    app.add_middleware(GuestPaths)

    @app.exception_handler(_Problem)
    async def on_problem(request: Request, error: _Problem) -> JSONResponse:
        return _problem(error.status, error.title, error.detail, **error.extra)

    @app.exception_handler(ApprovalRequired)
    async def on_approval(request: Request, error: ApprovalRequired) -> JSONResponse:
        return _problem(403, "Approval required", str(error), token=error.token)

    @app.exception_handler(RequestValidationError)
    async def on_request(request: Request, error: RequestValidationError) -> JSONResponse:
        errors = [
            {"location": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
            for e in error.errors()
        ]
        return _problem(
            422, "Invalid request", "the request does not match its schema", errors=errors
        )

    @app.exception_handler(ConfigError)
    async def on_config(request: Request, error: ConfigError) -> JSONResponse:
        return _problem(422, "Invalid configuration", str(error))

    @app.exception_handler(ApexError)
    async def on_apex(request: Request, error: ApexError) -> JSONResponse:
        return _problem(400, "Request failed", str(error))

    @app.exception_handler(KeyError)
    async def on_missing(request: Request, error: KeyError) -> JSONResponse:
        return _problem(404, "Not found", str(error.args[0]) if error.args else "not found")

    @app.exception_handler(Denied)
    async def on_denied(request: Request, error: Denied) -> JSONResponse:
        title = "Not found" if error.status == 404 else "Not allowed"
        return _problem(error.status, title, str(error))

    @app.exception_handler(WorkspaceError)
    async def on_workspace(request: Request, error: WorkspaceError) -> JSONResponse:
        return _problem(403, "Not editable", str(error))

    @app.exception_handler(WriteConflict)
    async def on_conflict(request: Request, error: WriteConflict) -> JSONResponse:
        return _problem(409, "Conflict", str(error), sha256=error.current)

    def config_source(body: ConfigBody, who: Guard) -> tuple[Path | dict[str, Any], Path | None]:
        # Paths absolute, or relative to the workspace (finding 31): the web UI sends those.
        base = who.optional_path(body.base_dir)
        if body.config_file:
            return who.server_path(body.config_file), base
        if body.config is not None:
            return body.config, base
        raise _Problem(422, "Missing configuration", "give config_file or config")

    # ------------------------------------------------------------------ discovery

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/whoami")
    def whoami(who: Who) -> dict[str, Any]:
        """Return the token's role, and what a UI user may do."""
        # Named, not created: asking who one is writes nothing.
        folder = who.own_folder(create=False)
        ui_folder = None if folder is None else _shown(service, folder)
        expires = None if who.expires is None else who.expires.isoformat(timespec="seconds")
        if not who.ui:
            return {"role": "owner",
                    "actions": ["submit", "stop", "resume", "edit", "delete", "examples"],
                    "ui_folder": ui_folder, "expires": expires}  # fmt: skip
        actions = who.access.rules.actions
        return {
            "role": "ui", "guest": who.guest,
            "actions": [a for a, on in actions.model_dump().items() if on],
            "ui_folder": ui_folder, "expires": expires,
        }  # fmt: skip

    @app.get("/v1/plugins")
    def plugins(who: Who) -> list[PluginInfo]:
        return who.plugins(service.plugins())

    @app.get("/v1/schemas/{kind}", dependencies=[Depends(_authorised)])
    def schema(kind: SchemaKind) -> dict[str, Any]:
        return service.schema(kind)

    @app.post("/v1/configs:validate")
    def validate(body: ConfigBody, who: Who) -> Validation:
        # A run is judged as its submission would be: placed in the web UI's folder for a
        # web-UI request, its folder holding another run an error (Submit agrees).
        as_submitted = _as_submitted(who)
        if body.config_text is not None:
            if not body.base_dir:
                raise _Problem(422, "Missing base_dir", "config_text needs base_dir")
            base_dir = who.server_path(body.base_dir)
            who.folder(base_dir)
            document = _toml_or_none(body.config_text)
            if document is not None:
                _check_cases(who, service, document, base_dir)
            text_validation = service.validate_text(body.config_text, base_dir, **as_submitted)
            if document is not None and "study" in document and is_optimization_study(document):
                if not text_validation.valid:
                    return _scrubbed(who, service, text_validation)
                spec = OptimizationStudySpec.model_validate(document)
                found = service.checked_optimization_study(spec, base_dir)
                return _scrubbed(
                    who, service, _checked_optimizations(who, service, text_validation, found)
                )
            return _scrubbed(who, service, _checked(
                who, text_validation,
                lambda: _text_configs(service, body.config_text or "", base_dir),
            ))  # fmt: skip
        source, base = config_source(body, who)
        if who.ui:
            if not isinstance(source, Path):
                raise Denied(403, "UI users validate files of the workspace (config_file)")
            who.file(source)
            if _kind(source) == "optimization":
                from apex.services.optimization import load_optimization_config

                return _scrubbed(who, service, _checked(
                    who, service.validate_optimization(source),
                    lambda: [load_optimization_config(source, place=False)],
                ))  # fmt: skip
            if _kind(source) == "study":
                document = _toml_or_none(source.read_text("utf-8")) or {}
                _check_cases(who, service, document, source.resolve().parent)
                found = service.checked_optimization_study(source)
                if found is not None:
                    return _scrubbed(who, service, _checked_optimizations(
                        who, service, service.validate_study(source), found
                    ))  # fmt: skip
            validation = (
                service.validate_study(source) if _kind(source) == "study"
                else service.validate(source, **as_submitted)
            )  # fmt: skip
            return _scrubbed(
                who, service, _checked(who, validation, lambda: _file_configs(service, source))
            )
        if isinstance(source, Path):
            # Before the file is opened to see what it holds.
            policy.check_file(source, service.jobs.settings.policy, service.root)
            if _kind(source) == "study":
                return service.validate_study(source)
            if _kind(source) == "optimization":
                return service.validate_optimization(source)
            return service.validate(source, **as_submitted)
        if "study" in source:
            try:
                given: StudySpec | OptimizationStudySpec = (
                    OptimizationStudySpec.model_validate(source)
                    if is_optimization_study(source)
                    else StudySpec.model_validate(source)
                )
            except ValidationError as error:
                raise _Problem(422, "Invalid study", validation_problems(error)) from error
            return service.validate_study(given, base_dir=base)
        if "optimization" in source and "optimizer" in source:
            return service.validate_optimization(source, base_dir=base)
        return service.validate(source, base_dir=base, **as_submitted)

    # ------------------------------------------------------------------ submissions

    def accepted(handle: JobHandle) -> JSONResponse:
        return JSONResponse(
            handle.model_dump(mode="json"),
            status_code=202,
            headers={"Location": f"/v1/jobs/{handle.id}"},
        )

    @app.post("/v1/runs", status_code=202, response_model=JobHandle)
    def submit_run(body: RunBody, who: Who) -> JSONResponse:
        source, base = config_source(body, who)
        checked: HistoryMatchConfig | None = None
        if who.ui:
            who.action("submit")
            if not isinstance(source, Path):
                raise Denied(403, "UI users submit files of the workspace (config_file)")
            who.file(source)
            # Read once: what is checked is what is submitted (a file changed in between
            # is not read again).
            checked = service.resolve(source, base)
            who.config(checked)
        submitted: Path | dict[str, Any] | HistoryMatchConfig = checked or source
        folder = who.ui_folder(body.client)
        if folder is not None:
            # The web UI's runs go to its user's folder, whatever the file says.
            resolved = checked or service.resolve(source, base)
            run = resolved.run  # its name is one folder name (RunSection)
            submitted = resolved.model_copy(
                update={"run": run.model_copy(update={"output_dir": folder / "runs" / run.name})}
            )
        handle = service.submit_run(
            submitted,
            annotations=who.annotations(body.annotations),
            base_dir=base,
            max_steps=body.max_steps,
            force_new=body.force_new,
            priority=body.priority,
            client=who.client(body.client),
            approval_token=body.approval_token,
            code=who.code(body.code),
        )
        return accepted(_handle_as_seen(handle, who))

    @app.post("/v1/studies", status_code=202, response_model=JobHandle)
    def submit_study(body: StudyBody, who: Who) -> JSONResponse:
        base = who.optional_path(body.base_dir)
        if body.study_file:
            spec: StudySpec | Path = who.server_path(body.study_file)
        elif body.study is not None:
            spec = StudySpec.model_validate(body.study)
        else:
            raise _Problem(422, "Missing study", "give study_file or study")
        submitted: StudySpec | OptimizationStudySpec | Path = spec
        if who.ui:
            who.action("submit")
            if not isinstance(spec, Path):
                raise Denied(403, "UI users submit files of the workspace (study_file)")
            who.file(spec)
            # Read once: the cases checked here are the ones submitted, named as
            # written if refused, and the study carries who submitted it.
            submitted, base = _read_study(who, service, spec)
        folder = who.ui_folder(body.client)

        def check(item: object) -> None:
            # Everything that will run, as the service plans it for submission: the cells,
            # an engine cell's method, and each prediction as it runs.
            if isinstance(item, HistoryMatchConfig):
                who.config(item)
            elif isinstance(item, OptimizationConfig):
                who.optimization_config(item)
            elif isinstance(item, PredictionConfig):
                who.names(item)
                _own_output(who, folder, item)
                # Its own simulator and executor, as a standalone prediction's check sees
                # them (the cells' runs are checked as configurations above).
                overrides = [
                    ("simulators", item.simulator.name),
                    ("executors", item.execution.name if item.execution is not None else None),
                ]
                hidden = [
                    f"{r}:{n}"
                    for r, n in overrides
                    if n is not None and not who.access.plugin(r, n)
                ]
                if hidden:
                    raise Denied(403, f"plugins not available in the UI: {', '.join(hidden)}")
            elif isinstance(item, EngineMethod) and not who.access.plugin("engines", item.engine):
                raise Denied(403, f"plugins not available in the UI: engines:{item.engine}")

        return accepted(
            service.submit_study(
                submitted,
                base_dir=base,
                approval_token=body.approval_token,
                priority=body.priority,
                client=who.client(body.client),
                output_root=None if folder is None else folder / "studies",
                check=check if who.ui else None,
                code=who.code(body.code),
            )
        )

    @app.post("/v1/predictions", status_code=202, response_model=JobHandle)
    def submit_prediction(body: PredictionBody, who: Who) -> JSONResponse:
        from apex.services.prediction import load_prediction_config, resolve_prediction_config

        if body.prediction_file:
            config: PredictionConfig | Path = who.server_path(body.prediction_file)
        elif body.config is not None:
            base = who.optional_path(body.base_dir) or Path.cwd()
            config = resolve_prediction_config(PredictionConfig.model_validate(body.config), base)
        else:
            raise _Problem(422, "Missing prediction", "give prediction_file or config")
        if who.ui:
            who.action("submit")
            if not isinstance(config, Path):
                raise Denied(403, "UI users submit files of the workspace (prediction_file)")
            who.file(config)
            loaded = load_prediction_config(config)
            who.prediction(loaded.prediction.run_dir)
            who.names(loaded)  # its simulator's overrides and files, as for runs
            # What runs: the run's saved configuration (work from before a rule too) with the
            # prediction's simulator and executor, which may replace the run's.
            from apex.services.composition import snapshot_config
            from apex.services.prediction import as_predicted
            from apex.storage.run_directory import RunDirectory

            who.config(
                as_predicted(loaded, snapshot_config(RunDirectory(loaded.prediction.run_dir)))
            )
            config = loaded  # what was checked is what is submitted
            # A guest's prediction writes only in their own web-UI folder: in
            # their run's folder, or, for a run they reused, in UI/guests/<name>/predictions.
            folder = who.ui_folder(body.client)
            section = loaded.prediction
            output = section.output_dir or section.run_dir / "predictions" / section.name
            if folder is not None and not output.resolve().is_relative_to(folder.resolve()):
                if output != section.run_dir / "predictions" / section.name:
                    raise Denied(403, "a guest's predictions are written in their own folder: "
                                 "leave [prediction] output_dir out")  # fmt: skip
                moved = folder / "predictions" / section.run_dir.name / section.name
                config = loaded.model_copy(
                    update={"prediction": section.model_copy(update={"output_dir": moved})}
                )
        return accepted(
            service.submit_prediction(
                config, annotations=who.annotations(body.annotations), priority=body.priority,
                client=who.client(body.client), code=who.code(body.code),
            )
        )  # fmt: skip

    # ------------------------------------------------------------------ jobs

    @app.get("/v1/jobs")
    def jobs(
        who: Who,
        state: str | None = None,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> Page:
        records = [
            {
                k: r.get(k)
                for k in ("id", "kind", "state", "created", "finished", "client", "priority")
            }
            for r in who.records()
            if state is None or r["state"] == state
        ]
        return _page(records, cursor, limit)

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str, who: Who) -> JobStatus:
        who.job(job_id)
        return _reused_as_seen(service.status(job_id), who)

    @app.get("/v1/jobs/{job_id}/events", response_model=None)
    def job_events(
        job_id: str,
        who: Who,
        since: int = 0,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        accept: Annotated[str | None, Header()] = None,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> Response | dict[str, Any]:
        who.job(job_id)
        if last_event_id is not None and last_event_id.isdigit():
            since = int(last_event_id) + 1
        if accept and "text/event-stream" in accept:
            return StreamingResponse(
                _stream(service, job_id, since), media_type="text/event-stream"
            )
        events = service.events(job_id, since=since)[:limit]
        return {
            "events": [e.model_dump(mode="json") for e in events],
            "next_since": events[-1].index + 1 if events else since,
        }

    @app.post("/v1/jobs/{job_id}:stop")
    def stop(job_id: str, who: Who) -> JobStatus:
        who.job(job_id)
        who.action("stop")
        service.stop(job_id)
        return service.status(job_id)

    @app.post("/v1/jobs/{job_id}:resume", status_code=202, response_model=JobHandle)
    def resume(job_id: str, who: Who, body: ResumeBody | None = None) -> JSONResponse:
        who.job(job_id)
        who.action("resume")
        if who.ui:
            _check_snapshot(who, service.jobs.record(job_id))
        return accepted(
            service.resume(
                job_id,
                max_steps=body.max_steps if body else None,
                code=who.code(body.code if body else None),
            )
        )

    @app.get("/v1/events", response_model=None)
    def feed(
        who: Who,
        since: int = 0,
        accept: Annotated[str | None, Header()] = None,
    ) -> Response | dict[str, Any]:
        visible = who.visible_jobs()
        if accept and "text/event-stream" in accept:
            return StreamingResponse(
                _feed_stream(service, since, visible=who if who.filtered else None),
                media_type="text/event-stream",
            )
        rows = [r for r in service.jobs.store.feed(since) if visible is None or r[1] in visible]
        return {
            "events": [{"id": i, "job": job, **event} for i, job, event in rows],
            "next_since": rows[-1][0] if rows else since,
        }

    # ------------------------------------------------------------------ results

    @app.get("/v1/runs")
    def runs(
        who: Who,
        annotation: Annotated[list[str] | None, Query(description="key=value")] = None,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> Page:
        wanted = dict(a.split("=", 1) for a in annotation or [] if "=" in a)
        rows = [
            {"run_id": e.run_id, "name": e.name, "created": e.created, "path": str(e.path),
             "annotations": e.annotations}
            for e in who.entries()
            if all(e.annotations.get(k) == v for k, v in wanted.items())
        ]  # fmt: skip
        return _page(rows, cursor, limit)

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str, who: Who) -> RunSummary:
        who.run(run_id)
        return service.run_summary(run_id)

    @app.get("/v1/runs/{run_id}/steps")
    def run_steps(run_id: str, who: Who) -> RunSteps:
        who.run(run_id)
        return service.run_steps(run_id)

    @app.get("/v1/runs/{run_id}/figures")
    def figures(run_id: str, who: Who) -> list[str]:
        """List the figures ``apex report --plots`` wrote (PNG, under ``report/``)."""
        who.run(run_id)
        entry = service.store.find(run_id)
        if entry is None:
            raise _Problem(404, "Not found", f"no run {run_id!r}")
        base = entry.path.resolve()
        report = base / "report"
        if not report.is_dir():
            return []
        return sorted(
            str(f.relative_to(base))
            for f in report.rglob("*.png")
            if f.is_file() and f.resolve().is_relative_to(base)
        )

    @app.get("/v1/runs/{run_id}/artifacts/{path:path}", response_model=None)
    def artifact(run_id: str, path: str, who: Who) -> FileResponse:
        who.run(run_id)
        entry = service.store.find(run_id)
        if entry is None:
            raise _Problem(404, "Not found", f"no run {run_id!r}")
        base = entry.path.resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise _Problem(404, "Not found", f"no file {path!r} in run {run_id[:12]}")
        return FileResponse(target)

    @app.get("/v1/studies/{name}")
    def study(name: str, who: Who) -> JobStatus:
        who.study(name)
        return service.study_status(name)

    @app.get("/v1/studies/{name}/comparison")
    def comparison(
        name: str, who: Who, metric: Annotated[list[str] | None, Query()] = None
    ) -> Comparison:
        who.study(name)
        return service.compare(name, metrics=metric)

    @app.get("/v1/predictions/summary")
    def prediction_summary(path: str, who: Who) -> PredictionSummary:
        where = who.server_path(path)
        who.prediction(where)
        return service.prediction_summary(where)

    # ------------------------------------------------------------ optimization

    @app.post("/v1/optimizations", status_code=202, response_model=JobHandle)
    def submit_optimization(body: OptimizationBody, who: Who) -> JSONResponse:

        if body.optimization_file:
            source: Path | dict[str, Any] = who.server_path(body.optimization_file)
        elif body.config is not None:
            source = body.config
        else:
            raise _Problem(422, "Missing optimization", "give optimization_file or config")
        base = who.optional_path(body.base_dir)
        folder = who.ui_folder(body.client)
        if who.ui:
            who.action("submit")
            if not isinstance(source, Path):
                raise Denied(403, "UI users submit files of the workspace (optimization_file)")
            who.file(source)
        resolved = service.resolve_optimization(source, base)
        if who.ui:
            who.optimization_config(resolved)  # read once: checked as submitted
        if folder is not None:
            # Web-UI submissions write in the user's own folder, placed as the
            # service places them: optimizations/<name>/<id>.
            section = resolved.optimization
            placed = service.root / "optimizations" / section.name
            output = section.output_dir
            if output is not None and not output.resolve().is_relative_to(folder.resolve()):
                if not output.resolve().is_relative_to(placed.resolve()):
                    raise Denied(403, "web-UI optimizations are written in your own folder: "
                                 "leave [optimization] output_dir out")  # fmt: skip
                moved = folder / "optimizations" / section.name / output.name
                resolved = resolved.model_copy(
                    update={"optimization": section.model_copy(update={"output_dir": moved})}
                )
        return accepted(
            service.submit_optimization(
                resolved, max_iterations=body.max_iterations,
                annotations=who.annotations(body.annotations), priority=body.priority,
                client=who.client(body.client), approval_token=body.approval_token,
                code=who.code(body.code),
            )
        )  # fmt: skip

    @app.get("/v1/optimizations")
    def optimizations(who: Who) -> list[dict[str, Any]]:
        visible = who.visible_jobs()
        return [r for r in service.optimizations() if visible is None or r["job"] in visible]

    @app.get("/v1/optimizations/summary")
    def optimization_summary(path: str, who: Who) -> OptimizationSummary:
        where = who.server_path(path)
        who.optimization(where)
        return service.optimization_summary(where)

    # ------------------------------------------------------------------ workspace

    # ------------------------------------------------------------------ examples

    @app.get("/v1/examples", dependencies=[Depends(_authorised)])
    def examples() -> dict[str, Any]:
        """Return the catalogue of examples and toy problems, each with its workflows."""
        from apex.services.examples import examples_root, find_examples

        root = examples_root(service.jobs.settings.examples)
        if root is None:
            return {"examples": []}
        return {"examples": [
            {"id": e.id, **e.manifest.example.model_dump(mode="json")}
            for e in find_examples(root)
        ]}  # fmt: skip

    @app.post("/v1/examples:copy")
    def copy_example(body: ExampleCopyBody, who: Who) -> dict[str, Any]:
        """Copy an example into one's own UI folder; return where its configurations are."""
        from apex.services import examples

        root = examples.examples_root(service.jobs.settings.examples)
        if root is None:
            raise Denied(404, "this service offers no examples ([examples] root)")
        if who.ui:
            who.action("examples")
        folder = who.ui_folder(UI_CLIENT)  # the owner's: UI/owner; a guest's: their own
        if folder is None:
            raise Denied(403, "examples are copied into the web UI's folder: set [workspace] "
                         "ui_folder")  # fmt: skip
        try:
            example = examples.get_example(root, body.id)
        except KeyError:
            raise Denied(404, f"no example {body.id!r}") from None
        destination = folder / "examples"
        if who.ui:
            # Copies are files a UI user adds without submitting anything: they count against
            # the guest's storage, together with what is already in their folder.
            limit = int(service.jobs.settings.guests.max_storage_mb * 1e6)
            used = (
                sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
                if folder.is_dir()
                else 0
            )
            if used + examples.example_size(example) > limit:
                raise Denied(
                    403,
                    f"copying this example would exceed your storage ({limit / 1e6:.0f} MB): "
                    "delete earlier copies or work first",
                )
        # Its runs would go to the user's runs/: names taken there are not reused.
        copied = examples.copy_example(example, destination, runs_dir=folder / "runs")
        space = service.workspace()
        return {
            "folder": space.relative(copied.folder),
            "configs": [
                {"workflow": workflow, "title": title, "path": space.relative(path)}
                for workflow, title, path in copied.configs
            ],
        }

    @app.get("/v1/workspace")
    def workspace(who: Who) -> dict[str, Any]:
        """Return the roots: each one's name (the prefix of relative paths) and folder.

        UI users get the names only: every route takes relative paths.
        """
        return {"roots": service.workspace().describe(paths=not who.ui)}

    @app.get("/v1/files")
    def files(who: Who, path: str | None = None) -> list[dict[str, Any]]:
        if path and who.ui:
            relative = who.relative(path)
            # A folder on the way to a visible one is listed (its visible entries only),
            # unless it is a link: then it is judged by where it leads.
            if not (who.on_the_way(relative) and who.real(path) == relative.strip("/")):
                who.folder(path)
        return [
            dataclasses.asdict(e)
            for e in service.workspace().list(path)
            if who.listed(e.path, e.kind)
        ]

    @app.delete("/v1/files/content")
    def delete_file(path: str, sha256: str, who: Who) -> dict[str, str]:
        """Delete a file of one's own web-UI folder, if unchanged since read."""
        who.folder(path)
        if who.ui:
            who.action("delete")
        folder = who.ui_folder(UI_CLIENT)  # the owner's: UI/owner; a guest's: their own
        if folder is None:
            raise Denied(403, "the web UI's folder is off ([workspace] ui_folder)")
        return {"deleted": service.workspace().delete(path, sha256, folder)}

    # ------------------------------------------------------------------ deletion

    def mine(who: Guard) -> str:
        from apex.services import deletion

        return deletion.OWNER if not who.ui else who.submitter

    def deletion_job(
        who: Guard, job: str | None, run: str | None, study: str | None, prediction: str | None
    ) -> str:
        from apex.services import deletion

        given = [x for x in (job, run, study, prediction) if x]
        if len(given) != 1:
            raise _Problem(422, "Missing target", "give one of job, run, study or prediction")
        try:
            if job:
                who.job(job)
                return str(deletion.top_level(service, job)["id"])
            if run:
                who.run(run)
                return deletion.job_of_run(service, run, mine(who))  # theirs, not the creator's
            if prediction:
                where = who.server_path(prediction)
                who.prediction(where)
                return deletion.job_of_prediction(service, where, mine(who))
            if who.filtered and not who.view.study(study or ""):
                raise KeyError(study)
            return deletion.job_of_study(service, study or "")
        except KeyError:
            # Named by what was asked for: no other person's job id comes back.
            raise Denied(404, f"nothing of yours to delete for {given[0]!r}") from None

    def deletion_plan(who: Guard, job_id: str, asked: str) -> dict[str, Any]:
        """Return the plan, if it is this token's own work (guests: 404 otherwise)."""
        from apex.services import deletion

        if who.ui:
            who.action("delete")
        found = deletion.plan(service, job_id)
        if found.owner != mine(who):
            if who.ui:
                raise Denied(404, f"nothing of yours to delete for {asked!r}")
            raise Denied(403, f"job {job_id} is {found.owner}'s: only its submitter deletes it "
                         "here (apex gc --guests prunes guests' work)")  # fmt: skip
        document = found.document()
        document["dirs"] = [_shown(service, Path(d)) for d in document["dirs"]]
        document["owner"] = found.owner
        return document

    @app.get("/v1/deletions")
    def deletion_preview(
        who: Who,
        job: str | None = None,
        run: str | None = None,
        study: str | None = None,
        prediction: str | None = None,
    ) -> dict[str, Any]:
        """Return what deleting a job (or a run's, study's or prediction's submission) removes."""
        asked = job or run or study or prediction or ""
        return deletion_plan(who, deletion_job(who, job, run, study, prediction), asked)

    @app.post("/v1/deletions")
    def deletion_confirmed(body: DeletionBody, who: Who) -> dict[str, Any]:
        """Delete one's own submission, as the preview's token describes it."""
        from apex.services import deletion

        who.job(body.job)
        preview = deletion_plan(who, body.job, body.job)  # ownership and the action, as above
        try:
            done = deletion.delete(service, body.job, body.token, str(preview["owner"]))
        except deletion.DeletionRefused as error:
            raise _Problem(409, "Not deleted", str(error)) from error
        return {"deleted": done.job, "jobs": len(done.jobs), "runs": len(done.runs),
                "bytes": done.size}  # fmt: skip

    @app.get("/v1/files/content")
    def file_content(path: str, who: Who) -> dict[str, Any]:
        who.folder(path)
        space = service.workspace()

        def allowed(real: Path) -> bool:
            try:
                return not who.ui or who.visible(space.relative(real))
            except WorkspaceError:
                return False

        try:
            return dataclasses.asdict(space.read(path, allowed))
        except KeyError:
            raise Denied(404, f"no file {path}") from None

    @app.put("/v1/files/content")
    def save_file(body: FileBody, who: Who) -> dict[str, Any]:
        who.folder(body.path)
        relative = who.relative(body.path)
        if not (who.ui and who.own_ui_draft(relative)):
            who.action("edit")  # UI users may always revise their own chats' drafts
        saved = service.workspace().write(body.path, body.text, body.sha256)
        return {"path": saved.path, "sha256": saved.sha256}

    # The tests' models replace the configured ones, the chooser's too.
    _chat_routes(app, service, chat_provider or _chat_provider(service),
                 chooser=chat_provider is None)  # fmt: skip
    _agent_routes(app, service, agent_provider)
    install_routes(app, RouteContext(service, _Problem, agent_provider))
    pages = ui_pages()

    @app.get("/v1/ui/pages")
    def contributed_pages(who: Who) -> dict[str, Any]:
        """Return the web-UI pages installed packages add that the caller may see."""
        if not service.jobs.settings.ui.contributed_pages:
            return {"pages": []}
        return {"pages": [{"name": p.name, "title": p.title, "url": f"/ui/pages/{p.name}/"}
                          for p in pages.values()
                          if not (p.owner_only and who.ui) and p.shown(service)]}  # fmt: skip

    _serve_ui(app, ui_dist(service), pages)
    return app


class DeletionBody(_Body):
    """``POST /v1/deletions``: the submission, and the token its preview gave."""

    job: str
    token: str = Field(min_length=8, max_length=64)


class ChatBody(_Body):
    """``POST /v1/chats``."""

    title: str = Field(default="Chat", max_length=200)
    context: dict[str, str] = Field(default_factory=dict)


class LaunchedBody(_Body):
    """``POST /v1/chats/{id}/proposals/{n}:launched``: the job the user's launch started."""

    job: str | None = Field(default=None, description="The job (needed unless elsewhere).")
    elsewhere: bool = Field(
        default=False,
        description="The launch was answered with another submission's job, not the user's "
        "to see (an identical run already queued or running outside their web-UI folder): "
        "nothing new was submitted. The proposal is settled with no job.",
    )


class MessageBody(_Body):
    """``POST /v1/chats/{id}/messages``: a user message, with the page it was sent from."""

    text: str = Field(min_length=1, max_length=20_000)
    context: dict[str, str] = Field(default_factory=dict)
    model: str | None = Field(
        default=None, max_length=32,
        description="A model of GET /v1/chats/models for this answer (the owner's choice).",
    )  # fmt: skip


class AgentPlotBody(_Body):
    """``POST /v1/agents/plot``: a run's figures."""

    run: str = Field(description="A run id (from GET /v1/runs).")
    request: str | None = Field(
        default=None, max_length=2000,
        description="What to draw, in words; default: the classic set (brugge_classic).",
    )  # fmt: skip
    preset: str | None = Field(
        default=None, description="Draw this figure preset as it is (no model is asked)."
    )
    preset_options: dict[str, bool] = Field(
        default_factory=dict,
        description="The preset's switches, e.g. {\"model_change\": true} for brugge_classic.",
    )  # fmt: skip
    model: bool = Field(
        default=False,
        description="Ask the model of [agents.llm] to choose the figures (slower), not only "
        "the plotter's rules.",
    )


class AgentRunBody(_Body):
    """``POST /v1/agents/{monitor,diagnose,report}``: a run (or, to report, a study)."""

    run: str | None = Field(default=None, description="A run id (from GET /v1/runs).")
    study: str | None = Field(default=None, description="A study's name (report only).")
    optimization: str | None = Field(
        default=None,
        description="An optimization's directory (from GET /v1/optimizations; monitor and report).",
    )
    model: bool = Field(
        default=False,
        description="Ask the model of [agents.llm] (slower), not only the agent's rules.",
    )


class AgentPlanBody(_Body):
    """``POST /v1/agents/plan``: a case, a goal and a budget."""

    case: str = Field(description="The case's configuration, as the workspace shows it.")
    goal: str = Field(min_length=3, max_length=2000)
    budget: int | None = Field(default=None, ge=1)
    model: bool = False


class AgentConvertBody(_Body):
    """``POST /v1/agents/convert``: a deck (and its history) to import as an APEX case."""

    source: str = Field(description="The case to convert, e.g. a .DATA deck, as the "
                        "workspace shows it.")  # fmt: skip
    history: str | None = Field(default=None, description="Its observed data: a CSV "
                                "(time,datatype,value) or a summary file, as the workspace "
                                "shows it; default the deck's own history.")  # fmt: skip
    request: str | None = Field(default=None, max_length=2000)
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    skill: str | None = None
    model: bool = False


class AgentCreateBody(_Body):
    """``POST /v1/agents/convert/create``: a synthetic case from a description."""

    request: str | None = Field(default=None, max_length=2000)
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    skill: str | None = None
    observe: bool = Field(
        default=False,
        description="Run the truth once now to make the observations (the owner only; the "
        "caller has asked the user).",
    )
    model: bool = False


class AgentGeomodelBody(_Body):
    """``POST /v1/agents/geomodel``: a prior for a configuration's fields."""

    config: str = Field(description="The configuration, as the workspace shows it.")
    request: str | None = Field(default=None, max_length=2000)
    reference: str | None = Field(default=None, description="An .npz of reference fields in "
                                  "state space, as the workspace shows it: the prior shares "
                                  "their statistics and variogram.")  # fmt: skip
    data: str | None = Field(default=None, description="Hard data to estimate from and "
                             "condition to: a CSV of i,j,k,property,value, as the workspace "
                             "shows it.")  # fmt: skip
    model: bool = False


def _agent_routes(
    app: FastAPI, service: ApexService, provider: "Callable[[], LLMProvider] | None"
) -> None:
    """Serve the agents as REST routes.

    Monitor, diagnose, report and plan; the case-converter and the geomodeler.

    Each runs its rules, or with ``model`` also the model of ``[agents.llm]`` (a model that
    is unavailable falls back to the rules, as on the command line). They read, and
    ``plan`` writes a draft in the user's web-UI folder; none submits, stops or resumes.
    UI users see only the agents their list names, on their own runs and studies.
    """
    from apex.agents.loop import LoopSettings

    def model_for(who: Guard, agent: str, wanted: bool) -> tuple[Any, LoopSettings, str | None]:
        if who.ui and not who.agent(agent):
            raise Denied(404, f"no agent {agent!r}")
        llm = service.jobs.settings.agents.llm
        settings = LoopSettings(max_turns=6, max_tokens=max(llm.max_tokens, 2048),
                                temperature=llm.temperature)  # fmt: skip
        if not wanted:
            return None, LoopSettings(max_turns=6), None
        make: Callable[[], Any] = functools.partial(_agent_provider, service)
        if provider is not None:
            make = provider
        elif _asker(who) == "guest" and service.jobs.settings.chat.models:
            # A guest's agents answer with the guest's chat models.
            make = _chosen_provider(service, _model_choice(service, who).names)
        try:
            chosen = make()
        except ApexError as error:
            return None, LoopSettings(max_turns=6), f"no model ({error}): the rules answered"
        return chosen, settings, None

    def run_dir(who: Guard, run: str | None) -> Path:
        if not run:
            raise _Problem(422, "Missing run", "give run (a run id)")
        who.run(run)
        entry = service.store.find(run)
        if entry is None:
            raise Denied(404, f"no run {run!r}")
        return Path(entry.path)

    @app.post("/v1/agents/monitor")
    def monitor(body: AgentRunBody, who: Who) -> dict[str, Any]:
        """Return the run monitor's report on a run (advisory: it never stops anything)."""
        from apex.agents.monitor import RunMonitor

        if body.optimization is not None:
            from apex.agents.optimization import OptimizationMonitor

            who.optimization(who.server_path(body.optimization))
            chosen, settings, note = model_for(who, "monitor", body.model)
            check = OptimizationMonitor(service, chosen, settings=settings).check(
                who.server_path(body.optimization)
            )
            return {"report": check.report.model_dump(mode="json"),
                    "decided_by": check.decided_by, "reason": note or check.reason}  # fmt: skip
        path = run_dir(who, body.run)
        chosen, settings, note = model_for(who, "monitor", body.model)
        result = RunMonitor(service, chosen, settings=settings).check(path, act=False)
        return {"report": result.report.model_dump(mode="json"),
                "decided_by": result.decided_by, "reason": note or result.reason}  # fmt: skip

    @app.post("/v1/agents/diagnose")
    def diagnose(body: AgentRunBody, who: Who) -> dict[str, Any]:
        """Return a diagnosis of a run: causes, with evidence, and recommended changes."""
        from apex.agents.diagnosis import Diagnostician

        if body.optimization is not None:  # 2026-10-04
            from apex.agents.optimization_advice import OptimizationDiagnostician

            who.optimization(who.server_path(body.optimization))
            chosen, settings, note = model_for(who, "diagnosis", body.model)
            found = OptimizationDiagnostician(service, chosen, settings=settings).diagnose(
                who.server_path(body.optimization)
            )
            return {"diagnosis": found.diagnosis.model_dump(mode="json"),
                    "decided_by": found.decided_by, "reason": note or found.reason}  # fmt: skip
        path = run_dir(who, body.run)
        chosen, settings, note = model_for(who, "diagnosis", body.model)
        result = Diagnostician(service, chosen, settings=settings).diagnose(str(path))
        return {"diagnosis": result.diagnosis.model_dump(mode="json"),
                "decided_by": result.decided_by, "reason": note or result.reason}  # fmt: skip

    @app.post("/v1/agents/report")
    def report(body: AgentRunBody, who: Who) -> dict[str, Any]:
        """Return the narrative report of a run or a study (Markdown; numbers checked)."""
        from apex.agents.reporter import Reporter

        given = [x for x in (body.run, body.study, body.optimization) if x is not None]
        if len(given) != 1:
            raise _Problem(422, "Missing target", "give one of run, study or optimization")
        chosen, settings, note = model_for(who, "report", body.model)
        reporter = Reporter(service, chosen, settings=settings)
        if body.optimization is not None:
            from apex.agents.optimization import optimization_report

            who.optimization(who.server_path(body.optimization))
            result = optimization_report(service, chosen, who.server_path(body.optimization))
        elif body.study is not None:
            who.study(body.study)
            result = reporter.study_report(body.study)
        else:
            result = reporter.run_report(str(run_dir(who, body.run)))
        return {"markdown": result.text, "decided_by": result.decided_by,
                "reason": note or result.reason}  # fmt: skip

    @app.post("/v1/agents/plot")
    def plot(body: AgentPlotBody, who: Who) -> dict[str, Any]:
        """Choose a run's figures and draw them in the background into <run>/report/plots.

        A classic set takes minutes: the answer comes once the figures are chosen, and
        they appear under GET /v1/runs/{id}/figures as they are drawn (``summary.json``
        says when all are).
        """
        from apex.agents.plotter import Plotter

        # A preset is drawn as it is: no model to ask.
        chosen, settings, note = model_for(who, "plotter", body.model and body.preset is None)
        started = Plotter(service, chosen, loop_settings=settings).start(
            str(run_dir(who, body.run)),
            request=body.request,
            preset=body.preset,
            preset_options=body.preset_options,
        )
        return {
            "state": "drawing",
            "requests": started.requests,
            "figures": [{"figure": c.figure, "subdir": c.subdir} for c in started.plan.figures],
            "preset": started.plan.preset,
            "decided_by": started.decided_by,
            "reason": note or started.reason or started.plan.reason,
        }

    @app.get("/v1/agents/plot/{run_id}")
    def plot_state(run_id: str, who: Who) -> dict[str, Any]:
        """Return how a run's figure drawing stands: none, drawing, done or failed."""
        from apex.services.figures import DEFAULT_SUBDIR, drawing_state

        if who.ui and not who.agent("plotter"):
            raise Denied(404, "no agent 'plotter'")
        state = drawing_state(run_dir(who, run_id) / DEFAULT_SUBDIR)
        return {k: v for k, v in state.items() if k in ("state", "requests", "written",
                                                        "skipped", "error")}  # fmt: skip

    def convert_folder(who: Guard, name: str) -> tuple[Path, str]:
        """Return a new folder for a converted or created case in the user's drafts."""
        folder = who.ui_folder(UI_CLIENT)
        if folder is None:
            raise Denied(403, "conversions are drafts in the web UI's folder ([workspace] "
                         "ui_folder)")  # fmt: skip
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        target = folder / "drafts" / "case-converter" / f"{name}-{stamp}"
        return target, service.workspace().relative(target)

    def existing(who: Guard, space: Any, path: str) -> Path:  # noqa: ANN401 (a Workspace)
        """Return a file of the workspace the caller may read: a deck, history or reference.

        One answer whether it is hidden or missing.
        """
        try:
            found: Path = space.real(path)
            who.file(found)
        except (Denied, ApexError):
            raise Denied(404, f"no file {path!r}") from None
        if not found.is_file():
            raise Denied(404, f"no file {path!r}")
        return found

    def readable(who: Guard) -> Callable[[Path], bool] | None:
        """Return what a web-UI user's deck may include: their visible folders."""
        if not who.ui:
            return None

        def allowed(path: Path) -> bool:
            try:
                service.workspace().real(path)
                who.file(path)
            except (Denied, ApexError):
                return False
            return True

        return allowed

    def skill_allowed(who: Guard, skill: str) -> None:
        from apex.agents.converter import visible_skills

        if _asker(who) != "owner" and skill not in visible_skills(public_only=True):
            raise Denied(404, f"no converter skill {skill!r}")

    @app.get("/v1/agents/convert/skills")
    def convert_skills(who: Who) -> list[dict[str, Any]]:
        """Return the converter skills the caller may use, and what each can do."""
        from apex.agents.converter import visible_skills
        from apex.core.plugins import converter_skills

        if who.ui and not who.agent("case-converter"):
            raise Denied(404, "no agent 'case-converter'")
        names = visible_skills(public_only=_asker(who) != "owner")
        found = []
        for name in names:
            skill = converter_skills.create(name)
            summary = (type(skill).__doc__ or "").strip().splitlines()[0]
            found.append({"name": name, "capabilities": sorted(skill.capabilities),
                          "formats": list(skill.formats), "simulator": skill.simulator,
                          "summary": summary})  # fmt: skip
        return found

    @app.post("/v1/agents/convert")
    def convert(body: AgentConvertBody, who: Who) -> dict[str, Any]:
        """Import a deck as an APEX case: a draft in the user's web-UI folder, validated."""
        from apex.agents.converter import CaseConverter, case_name

        chosen, settings, note = model_for(who, "case-converter", body.model)  # 404 first
        if body.skill is not None:
            skill_allowed(who, body.skill)
        space = service.workspace()
        source = existing(who, space, body.source)  # a deck: any file, not the editor's TOML
        history = None if body.history is None else existing(who, space, body.history)
        converter = CaseConverter(service, chosen, loop_settings=settings)
        converter.allowed = readable(who)
        if who.ui:
            converter.max_cells = service.jobs.settings.agents.case_converter.ui_max_cells
        skill = converter.skill_for(source, body.skill)
        skill_allowed(who, skill.name)
        target, shown = convert_folder(who, body.name or case_name(source))
        outcome = converter.convert(source, target, history=history, request=body.request,
                                    name=body.name, skill=skill.name)  # fmt: skip
        return {"draft": f"{shown}/config.toml", "skill": outcome.skill,
                "valid": outcome.valid, "errors": outcome.validation.errors,
                "plan": None if outcome.validation.plan is None
                else outcome.validation.plan.model_dump(mode="json"),
                "choices": outcome.choices.model_dump(mode="json"), "notes": outcome.notes,
                "decided_by": outcome.decided_by, "reason": note or outcome.reason}  # fmt: skip

    @app.post("/v1/agents/convert/create")
    def convert_create(body: AgentCreateBody, who: Who) -> dict[str, Any]:
        """Create a synthetic case (a draft); ``observe`` runs its truth once (owner only)."""
        from apex.agents.converter import CaseConverter

        chosen, settings, note = model_for(who, "case-converter", body.model)  # 404 first
        if body.observe and _asker(who) != "owner":
            raise Denied(403, "only the owner runs a created case's truth")
        if body.skill is not None:
            skill_allowed(who, body.skill)
        converter = CaseConverter(service, chosen, loop_settings=settings)
        if who.ui:
            converter.max_cells = service.jobs.settings.agents.case_converter.ui_max_cells
        target, shown = convert_folder(who, body.name or "synthetic")
        outcome = converter.create(target, request=body.request, name=body.name,
                                   skill=body.skill, simulate=body.observe)  # fmt: skip
        return {"draft": f"{shown}/config.toml", "valid": outcome.valid,
                "errors": outcome.validation.errors, "spec": outcome.spec.model_dump(mode="json"),
                "notes": outcome.notes, "decided_by": outcome.decided_by,
                "reason": note or outcome.reason}  # fmt: skip

    @app.post("/v1/agents/geomodel")
    def geomodel(body: AgentGeomodelBody, who: Who) -> dict[str, Any]:
        """Give a configuration's fields a prior: a new configuration beside it."""
        from apex.agents.geomodeler import Geomodeler

        chosen, settings, note = model_for(who, "geomodeler", body.model)  # 404 first
        space = service.workspace()
        config = space.file(body.config)
        who.file(config)
        own = who.ui_folder(UI_CLIENT)
        if who.ui and (own is None or own.resolve() not in config.resolve().parents):
            raise Denied(403, "the geomodeler writes beside the configuration: give one in "
                         "your own web-UI folder")  # fmt: skip
        reference = None if body.reference is None else existing(who, space, body.reference)
        data = None if body.data is None else existing(who, space, body.data)
        geomodeler = Geomodeler(service, chosen, loop_settings=settings)
        if who.ui:
            geomodeler.max_cells = service.jobs.settings.agents.geomodeler.ui_max_cells
        outcome = geomodeler.model(config, request=body.request, reference=reference,
                                   data=data)  # fmt: skip
        return {"config": space.relative(outcome.config), "skill": outcome.plan.skill,
                "valid": outcome.valid, "errors": outcome.validation.errors, "qc": outcome.qc,
                "notes": outcome.notes, "decided_by": outcome.decided_by,
                "reason": note or outcome.reason}  # fmt: skip

    @app.post("/v1/agents/plan")
    def plan(body: AgentPlanBody, who: Who) -> dict[str, Any]:
        """Plan a study for a case and a goal, as a draft in the user's web-UI folder."""
        from apex.agents.optimization_advice import (
            OptimizationPlanner,
            OptimizationPlanResult,
            is_optimization_config,
        )
        from apex.agents.planner import Planner, PlanResult

        space = service.workspace()
        case = space.file(body.case)
        who.file(case)
        optimizing = is_optimization_config(case)
        if who.ui and optimizing:
            from apex.services.optimization import load_optimization_config

            who.optimization_config(load_optimization_config(case, place=False))
        elif who.ui:
            who.config(service.resolve(case))
        folder = who.ui_folder(UI_CLIENT)
        if folder is None:
            raise Denied(403, "plans are drafts in the web UI's folder ([workspace] ui_folder)")
        chosen, settings, note = model_for(who, "planner", body.model)
        result: PlanResult | OptimizationPlanResult
        if optimizing:
            result = OptimizationPlanner(service, chosen, settings=settings).plan(
                case, body.goal, budget=body.budget
            )
        else:
            result = Planner(service, chosen, settings=settings).plan(
                case, body.goal, budget=body.budget
            )
        if who.ui and isinstance(result, OptimizationPlanResult):
            # Every cell, as the history-matching branch: a variant may switch the
            # simulator, objective, executor or a parameterization, not only the optimizer.
            from apex.services.optimization_study import expand

            for cell in expand(service, result.spec, case.parent):
                who.optimization_config(cell.config)
        elif who.ui and isinstance(result, PlanResult):
            from apex.agents.chat import _cell_configs

            hidden = sorted({p for c in _cell_configs(service, result.spec)
                             for p in who.access.hidden_plugins(c)})  # fmt: skip
            if hidden:
                raise Denied(403, f"the plan uses plugins not available: {', '.join(hidden)}")
        target = folder / "drafts" / "planner" / f"study_{result.spec.study.name}.toml"
        shown = space.relative(target)
        try:
            current: str | None = space.read(shown).sha256
        except KeyError:
            current = None
        space.write(shown, result.toml(space, target.parent), current)
        p = result.proposal
        return {
            "draft": shown, "study": p.name, "hypothesis": p.hypothesis,
            "variants": [v.model_dump() for v in p.variants], "replicates": p.replicates,
            "cost": result.cost, "budget": result.facts.budget_simulator_runs,
            "within_budget": result.within_budget, "decided_by": result.decided_by,
            "reason": note or result.reason, "revisions": list(result.revisions),
            # An optimization study's plan compares no algorithms (OptimizationPlanResult).
            "differences": list(getattr(result, "differences", ())),
        }  # fmt: skip


def _asker(who: Guard) -> "Asker":
    """Return who asks for a model: the owner, a guest, or another kind of web-UI user."""
    if not who.ui:
        return "owner"
    return "guest" if who.identity is None else "other"


def _model_choice(service: ApexService, who: Guard) -> "ModelChoice":
    """Return the models ``who``'s assistant may answer with."""
    from apex.services.access import load_access
    from apex.services.chat_models import choice

    chat = service.jobs.settings.chat
    asker = _asker(who)
    listed = load_access(service.root).chat.models if asker == "guest" and chat.models else None
    return choice(chat, asker, listed)


def _chosen_provider(service: ApexService, names: Sequence[str]) -> "Callable[[], LLMProvider]":
    """Return a factory of the models ``names``, tried in order."""

    def make() -> "LLMProvider":
        from apex.agents.providers import create_provider
        from apex.agents.providers.fallback import FallbackProvider
        from apex.core.llm import LLMUnavailable
        from apex.services.chat_models import section

        if not names:
            raise ConfigError("no assistant model is available for this account")
        chat = service.jobs.settings.chat
        chosen = FallbackProvider([(n, functools.partial(create_provider, section(chat, n)))
                                   for n in names])  # fmt: skip
        try:
            chosen.model_id  # noqa: B018 (makes the first model that can be made)
        except LLMUnavailable as error:  # none can: the caller's fallback, with the reasons
            raise ConfigError(str(error)) from error
        return chosen

    return make


def _agent_provider(service: ApexService) -> "LLMProvider":
    """Return the agents' model, ``[agents.llm]``."""
    from apex.agents.providers import create_provider

    return create_provider(service.jobs.settings.agents.llm)


def _chat_routes(
    app: FastAPI,
    service: ApexService,
    provider: "Callable[[], LLMProvider]",
    *,
    chooser: bool = True,
) -> None:
    """Add the chat assistant's routes.

    A UI user sees only the chats UI users started, and their assistant only the
    agents and the runs, jobs, studies and plugins the list allows.
    """
    from apex.agents.chat import ChatAssistant, agent_catalogue
    from apex.storage.chat_store import ChatBusy, ChatStore

    store = ChatStore(service.root)
    assistant = ChatAssistant(service, store, provider)

    @app.exception_handler(ChatBusy)
    async def on_busy(request: Request, error: ChatBusy) -> JSONResponse:
        return _problem(409, "Busy", str(error))

    def mine(chat_id: str, who: Guard) -> None:
        if not who.owns_chat(store.get(chat_id).owner):
            raise Denied(404, f"no chat {chat_id!r}")

    @app.get("/v1/agents")
    def agents(who: Who) -> list[dict[str, Any]]:
        """Every agent with its visibility; a guest gets the public ones its list allows."""
        catalogue = agent_catalogue(service.jobs.settings.agents.visibility)
        return [
            dataclasses.asdict(a)
            for a in catalogue
            if who.agent(a.name) and not (who.ui and a.visibility == "private")
        ]

    @app.post("/v1/chats", status_code=201)
    def create_chat(body: ChatBody, who: Who) -> dict[str, Any]:
        return dataclasses.asdict(store.create(body.title, body.context, owner=who.chat_owner))

    @app.get("/v1/chats/models")
    def chat_models(who: Who) -> dict[str, Any]:
        """Return the models this user's assistant may answer with.

        ``choosable``: the owner picks one per message; a guest's are tried in order, with
        ``note`` saying so. No ``models``: the service's one model answers every chat.
        """
        from apex.services.chat_models import label, on_machine

        picked = _model_choice(service, who)
        chat = service.jobs.settings.chat
        return {
            # ``local``: the chat's text stays on this machine (the UI warns on a switch
            # from such a model to a hosted one; release tester C2).
            "models": [
                {"name": n, "label": label(chat, n), "local": on_machine(chat, n)}
                for n in picked.names
            ],
            "default": picked.default,
            "choosable": picked.choosable,
            "note": picked.note,
        }

    @app.get("/v1/chats")
    def chats(who: Who) -> list[dict[str, Any]]:
        return [dataclasses.asdict(c) for c in store.chats() if who.owns_chat(c.owner)]

    @app.get("/v1/chats/{chat_id}")
    def chat(chat_id: str, who: Who) -> dict[str, Any]:
        mine(chat_id, who)
        return {
            **dataclasses.asdict(store.get(chat_id)),
            "messages": [dataclasses.asdict(t) for t in store.turns(chat_id)],
            "proposals": [dataclasses.asdict(p) for p in store.proposals(chat_id)],
        }

    @app.post("/v1/chats/{chat_id}/proposals/{n}:launched")
    def launched(chat_id: str, n: int, body: LaunchedBody, who: Who) -> dict[str, Any]:
        """Record that the user launched a proposal (the UI submitted it) as a job."""
        mine(chat_id, who)
        if body.elsewhere:
            # No job is recorded (nor followed): it is not this user's. Settling it only
            # stops a second launch, as discarding it would.
            try:
                proposal = store.settle(chat_id, n, "launched", None)
            except ValueError as error:
                raise _Problem(409, "Not pending", str(error)) from error
            store.add_turn(
                chat_id, "assistant",
                f"An identical run of {proposal.draft} is already queued or running outside "
                "your web-UI folder: nothing new was submitted.",
                {"note": "proposal", "proposal": n},
            )  # fmt: skip
            store.add_event(chat_id, "proposal_launched", {"n": n, "elsewhere": True})
            return dataclasses.asdict(proposal)
        if body.job is None:
            raise _Problem(422, "Missing job", "give the job the launch started, or elsewhere")
        who.job(body.job)
        try:
            proposal = store.settle(chat_id, n, "launched", body.job)
        except ValueError as error:
            raise _Problem(409, "Not pending", str(error)) from error
        store.add_turn(chat_id, "assistant", f"You launched {proposal.draft} as job {body.job}.",
                       {"note": "proposal", "proposal": n, "job": body.job})  # fmt: skip
        store.add_event(chat_id, "proposal_launched", {"n": n, "job": body.job})
        # A quick study may have ended (and been notified, finding no chat) already.
        from apex.services.chat_followup import after_launch

        after_launch(service.root, service.jobs.record(body.job))
        return dataclasses.asdict(proposal)

    @app.post("/v1/chats/{chat_id}/proposals/{n}:discard")
    def discard(chat_id: str, n: int, who: Who) -> dict[str, Any]:
        mine(chat_id, who)
        try:
            proposal = store.settle(chat_id, n, "discarded")
        except ValueError as error:
            raise _Problem(409, "Not pending", str(error)) from error
        store.add_event(chat_id, "proposal_discarded", {"n": n})
        return dataclasses.asdict(proposal)

    @app.delete("/v1/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str, who: Who) -> Response:
        mine(chat_id, who)
        store.delete(chat_id)
        return Response(status_code=204)

    @app.post("/v1/chats/{chat_id}/messages", status_code=202)
    def message(chat_id: str, body: MessageBody, who: Who) -> dict[str, Any]:
        mine(chat_id, who)
        model = None
        picked = _model_choice(service, who)
        if chooser and picked.configured:
            requested = body.model
            if requested is None and picked.choosable:
                from apex.services.chat_models import last_model

                # The chat's own model, not the default (release tester C2, 2026-10-24).
                requested = last_model(store.turns(chat_id), picked.names)
            try:
                model = _chosen_provider(service, picked.ordered(requested))
            except PermissionError as error:
                raise Denied(403, str(error)) from error
        elif body.model is not None:
            raise _Problem(422, "No model choice", "this service's chat has no model to choose")
        try:
            assistant.start(chat_id, body.text, body.context,
                            access=who.view if who.filtered else None,
                            provider=model)  # fmt: skip
        except ValueError as error:
            raise _Problem(400, "Chat closed", str(error)) from error
        return {"chat": chat_id, "events": f"/v1/chats/{chat_id}/events"}

    @app.post("/v1/chats/{chat_id}:stop")
    def stop_chat(chat_id: str, who: Who) -> dict[str, Any]:
        """Stop the chat's running answer at once; the user's own chat.

        ``stopped`` is false when nothing was running (it had just finished). Nothing done
        before the stop is undone, and the chat takes a new message at once.
        """
        mine(chat_id, who)
        return {"chat": chat_id, "stopped": assistant.stop(chat_id)}

    @app.get("/v1/chats/{chat_id}/events", response_model=None)
    def chat_events(
        chat_id: str,
        who: Who,
        since: int = 0,
        accept: Annotated[str | None, Header()] = None,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> Response | dict[str, Any]:
        mine(chat_id, who)
        if last_event_id is not None and last_event_id.isdigit():
            since = int(last_event_id) + 1
        if accept and "text/event-stream" in accept:
            return StreamingResponse(
                _chat_stream(store, chat_id, since), media_type="text/event-stream"
            )
        events = store.events(chat_id, since)
        return {
            "events": [dataclasses.asdict(e) for e in events],
            "next_since": events[-1].n + 1 if events else since,
            "busy": store.get(chat_id).busy,
        }


#: The UI's pages: only its own scripts, styles and images, and API calls to this origin.
CSP = (
    "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
BUILD_HINT = "npm --prefix ui ci && npm --prefix ui run build"


def ui_dist(service: ApexService) -> Path | None:
    """Return the built UI's folder: ``[ui] dist_dir``, ``$APEX_UI_DIST`` or ``ui/dist``.

    The last is beside the source tree (an editable install); ``None`` if none is set and
    that folder does not exist.
    """
    configured = service.jobs.settings.ui.dist_dir or os.environ.get("APEX_UI_DIST")
    if configured:
        return Path(configured)
    beside = Path(__file__).resolve().parents[3] / "ui" / "dist"
    return beside if beside.is_dir() else None


def _serve_ui(app: FastAPI, dist: Path | None, pages: dict[str, UiPage] | None = None) -> None:
    """Serve the built UI at ``/ui/``, with its Content-Security-Policy.

    A page an installed package adds is served at ``/ui/pages/<name>/`` from its own folder.
    """
    # Said once, at start: a missing build otherwise shows only as a 404 at /ui/ (a
    # service, 2026-10-01: its service.toml had no [ui] dist_dir).
    if dist is not None and (dist / "index.html").is_file():
        _log.info("the web UI is served from %s", dist, extra={"event": "ui_served"})
    else:
        _log.warning(
            "no web UI build%s: /ui/ answers 404; set [ui] dist_dir in service.toml (or "
            "$APEX_UI_DIST) to a built UI, or build it: %s",
            "" if dist is None else f" in {dist}", BUILD_HINT,
            extra={"event": "ui_missing"},
        )  # fmt: skip

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/ui", include_in_schema=False)
    def ui_root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.get("/ui/{path:path}", include_in_schema=False, response_model=None)
    def ui(path: str) -> Response:
        top, _, rest = path.partition("/")
        name, slash, path_in_page = rest.partition("/")
        if top == "pages" and pages and name in pages:
            if not slash:  # its assets are relative to the page's folder
                return RedirectResponse(f"/ui/pages/{name}/")
            return _static(pages[name].folder, path_in_page)
        if dist is None or not (dist / "index.html").is_file():
            return _problem(404, "No UI build", f"build the UI first: {BUILD_HINT}")
        return _static(dist, path)


def _static(folder: Path, path: str) -> Response:
    """Return a built page's file (``index.html`` for a hash route), with the UI's policy."""
    if not (folder / "index.html").is_file():
        return _problem(404, "No build", f"no built page in {folder.name}")
    base = folder.resolve()
    target = (base / (path or "index.html")).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        if "." in Path(path).name:
            return _problem(404, "Not found", f"no file {path!r} in the UI")
        target = base / "index.html"  # hash routes: every page is index.html
    headers = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff"}
    if target.name == "index.html":
        headers["Cache-Control"] = "no-cache"
    return FileResponse(target, headers=headers)


def _chat_stream(
    store: "ChatStore", chat_id: str, since: int, poll_s: float = 0.5
) -> Iterator[str]:
    """Stream a chat's events until no turn is running and nothing more is coming."""
    while True:
        try:
            busy = store.get(chat_id).busy
        except KeyError:
            return  # deleted while streaming
        events = store.events(chat_id, since)
        for event in events:
            yield _sse(event.n, event.kind, dataclasses.asdict(event))
            since = event.n + 1
        if not busy and not events:
            return
        if not events:
            yield ": keep-alive\n\n"
        time.sleep(poll_s)


def _sse(index: int, event: str | None, data: dict[str, Any]) -> str:
    return f"id: {index}\nevent: {event or 'message'}\ndata: {json.dumps(data, default=str)}\n\n"


def _handle_as_seen(handle: JobHandle, who: Guard) -> JobHandle:
    """Return a submission's answer with whether this web-UI request may open its job.

    A reused identical run may be answered with another submission's job (still queued or
    running) that the web UI's user may not see: the web UI then says so instead of opening
    a page that would be "not found". Other clients see every job: nothing is added.
    """
    if not who.filtered:
        return handle
    record = who.service.jobs.record(handle.id)
    # A fresh view: the one this request read before submitting lacks the new job.
    visible = who.again().view.job(record, who.service.store.entries())
    return handle.model_copy(update={"job_visible": visible, "job_state": record["state"]})


def _reused_as_seen(status: JobStatus, who: Guard) -> JobStatus:
    """Return a job's status as this request sees the run it reused, if it reused one.

    ``reused_visible`` says whether the request may open that run. A hidden run's id is
    left out (its page may not be opened). A web-UI guest gets no path of a run that is not
    theirs; the owner, in the web UI, gets the path of a run outside their web-UI folder
    (shown workspace-relative), but no run page.
    """
    if status.reused is None or not who.filtered:
        return status
    entries = {e.run_id: e for e in who.service.store.entries()}
    visible = any(
        (entry := entries.get(run_id)) is not None
        and who.view.run(entry.run_id, entry.name, entry.annotations, Path(entry.path))
        for run_id in status.run_ids
    )
    if visible:
        return status.model_copy(update={"reused_visible": True})
    # Hidden: not its id (a run page this request may not open); for a guest, no path.
    hidden: dict[str, Any] = {"reused_visible": False, "run_ids": []}
    if who.ui:
        hidden |= {"reused": None, "run_dirs": []}
    return status.model_copy(update=hidden)


def _stream(service: ApexService, job_id: str, since: int, poll_s: float = 0.5) -> Iterator[str]:
    """Stream a job's events until it ends and nothing more is coming."""
    while True:
        # The state first: a job that had ended before its events were read has no more.
        ended = service.jobs.record(job_id)["state"] in FINAL
        events = service.events(job_id, since=since)
        for event in events:
            yield _sse(event.index, event.event, event.model_dump(mode="json"))
            since = event.index + 1
        if ended and not events:
            return
        if not events:
            yield ": keep-alive\n\n"
        time.sleep(poll_s)


def _feed_stream(
    service: ApexService, since: int, poll_s: float = 1.0, visible: Guard | None = None
) -> Iterator[str]:
    """Stream every job's events (the client closes the connection to stop).

    With ``visible`` (a UI user's guard), only the jobs it may see, re-read as jobs start.
    """
    while True:
        rows = service.jobs.store.feed(since)
        allowed = None if visible is None or not rows else visible.again().visible_jobs()
        for index, job, event in rows:
            if allowed is None or job in allowed:
                yield _sse(index, event.get("event"), {"job": job, **event})
            since = index
        if not rows:
            yield ": keep-alive\n\n"
        time.sleep(poll_s)
