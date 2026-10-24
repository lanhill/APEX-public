"""The Python client of APEX's services.

``Client`` is the service interface for programs such as an experiment planner. ``Client()``
runs in-process (the same calls as :class:`apex.services.api.ApexService`), and
``Client.connect(url, token)`` returns a :class:`RemoteClient` with the same methods over the
REST API, so a program written against one moves to the other unchanged::

    from apex.client import Client

    client = Client()                                  # root: $APEX_HOME or ~/.apex/service
    # client = Client.connect("http://127.0.0.1:8765")  # token: $APEX_API_TOKEN
    print([p.name for p in client.plugins() if p.registry == "localizations"])
    check = client.validate(Path("configs/gies_opm.toml"))
    job = client.submit_study(Path("study.toml"))
    client.wait(job.id)
    comparison = client.compare("localization_m5spots")

Remote paths are the **service's**: a ``Path`` given to a remote client (a configuration, a
study, a prediction) names a file on the service's machine, so it is sent as an absolute
path. A configuration given as a document is sent as it is, with ``base_dir`` for its
relative paths (again on the service's machine); ``base_dir`` and the summaries' paths are
sent absolute too (resolved here), since the service reads a relative path as one under its
workspace (finding 31 of the release tester, 2026-10-19). Errors are the in-process ones:
``ConfigError`` for invalid input, ``ApprovalRequired`` (with its ``token``) when a person
must approve, and ``KeyError`` for an unknown job.
"""

import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, NoReturn, Protocol

from pydantic import BaseModel, TypeAdapter

from apex.core.engines import EngineMethod
from apex.core.errors import ApexError, ConfigError
from apex.domain.config import HistoryMatchConfig
from apex.domain.optimization import OptimizationConfig
from apex.domain.prediction import PredictionConfig
from apex.domain.services import (
    Annotations,
    Comparison,
    Event,
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
from apex.services.api import ApexService, SchemaKind
from apex.services.approval import ApprovalRequired
from apex.services.jobs import FINAL

TOKEN_VARIABLE = "APEX_API_TOKEN"


def _absolute(path: Path) -> str:
    """Return a path as sent: absolute, resolved where the client runs (see the docstring)."""
    return str(Path(path).resolve())


class HTTPClient(Protocol):
    """What :class:`RemoteClient` uses of an HTTP client: ``httpx.Client`` or alike.

    Starlette's ``TestClient`` (built on ``httpx2`` when that is installed) fits too.
    """

    @property
    def headers(self) -> Any: ...  # noqa: ANN401, D102 (httpx.Headers or httpx2.Headers)

    def request(  # noqa: D102
        self,
        method: str,
        url: str,
        *,
        json: Any = None,  # noqa: ANN401
        params: Any = None,  # noqa: ANN401
    ) -> Any: ...  # noqa: ANN401

    def stream(  # noqa: D102
        self,
        method: str,
        url: str,
        *,
        headers: Any = None,  # noqa: ANN401
    ) -> AbstractContextManager[Any]: ...

    def get(self, url: str) -> Any: ...  # noqa: ANN401, D102

    def close(self) -> None: ...  # noqa: D102


class RemoteError(ApexError):
    """The REST API answered with an error that has no in-process equivalent."""

    def __init__(self, status: int, problem: Mapping[str, Any]) -> None:
        super().__init__(f"HTTP {status}: {problem.get('title')}: {problem.get('detail')}")
        self.status = status
        self.problem = dict(problem)


class RemoteClient:
    """APEX's services over the REST API (see the module docstring).

    Parameters
    ----------
    url
        The service's address, e.g. ``http://127.0.0.1:8765``.
    token
        The bearer token (``apex service token`` on the service's machine); default
        ``$APEX_API_TOKEN``.
    http
        An ``httpx.Client`` to use in place of a new one (tests pass a Starlette
        ``TestClient``); its base URL is then its own.
    timeout_s
        Seconds per request.
    """

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        http: HTTPClient | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        import httpx

        token = token if token is not None else os.environ.get(TOKEN_VARIABLE)
        if not token:
            raise ConfigError(
                f"no API token for {url}: pass token=..., or set {TOKEN_VARIABLE} to the "
                "output of `apex service token` on the service's machine"
            )
        self.url = url.rstrip("/")
        self.http: HTTPClient = (
            http if http is not None else httpx.Client(base_url=self.url, timeout=timeout_s)
        )
        self.http.headers["Authorization"] = f"Bearer {token}"

    def close(self) -> None:
        """Close the HTTP connection."""
        self.http.close()

    def __enter__(self) -> "RemoteClient":
        """Use the client in a ``with`` block, which closes it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client."""
        self.close()

    # ------------------------------------------------------------------ transport

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401 (JSON)
        response = self.http.request(method, f"/v1{path}", json=body, params=params)
        if response.status_code < 400:
            return response.json()
        self._raise_for(response)

    @staticmethod
    def _raise_for(response: Any) -> NoReturn:  # noqa: ANN401 (an httpx response)
        """Raise the error an HTTP error response means (the problem document's detail)."""
        try:
            problem: dict[str, Any] = response.json()
        except ValueError:
            problem = {"title": response.reason_phrase, "detail": response.text}
        detail = str(problem.get("detail", ""))
        errors = problem.get("errors")
        if errors:
            detail += ": " + "; ".join(f"{e['location']}: {e['message']}" for e in errors)
        if response.status_code == 403 and "token" in problem:
            raise ApprovalRequired(detail, str(problem["token"]))
        if response.status_code in (400, 422):
            raise ConfigError(detail)
        if response.status_code == 404:
            raise KeyError(detail)
        raise RemoteError(response.status_code, problem)

    @staticmethod
    def _config(
        config: HistoryMatchConfig
        | PredictionConfig
        | OptimizationConfig
        | Mapping[str, Any]
        | Path,
        key: str,
    ) -> dict[str, Any]:
        if isinstance(config, Path):
            return {f"{key}_file": str(config.resolve())}
        if isinstance(config, BaseModel):
            return {"config": config.model_dump(mode="json")}
        return {"config": dict(config)}

    # ------------------------------------------------------------------ capabilities

    def plugins(self) -> list[PluginInfo]:
        """Return every registered plugin, with its status and option schema."""
        return TypeAdapter(list[PluginInfo]).validate_python(self._call("GET", "/plugins"))

    def schema(self, kind: SchemaKind) -> dict[str, Any]:
        """Return the JSON Schema of a configuration kind."""
        return dict(self._call("GET", f"/schemas/{kind}"))

    def validate(
        self, config: HistoryMatchConfig | Mapping[str, Any] | Path, base_dir: Path | None = None
    ) -> Validation:
        """Validate a configuration and plan its cost; nothing is simulated."""
        body = self._config(config, "config")
        if base_dir is not None:
            body["base_dir"] = _absolute(base_dir)
        return Validation.model_validate(self._call("POST", "/configs:validate", body=body))

    def validate_study(self, spec: StudySpec | Path, base_dir: Path | None = None) -> Validation:
        """Expand a study and validate every run; check its budget."""
        if isinstance(spec, Path):
            body: dict[str, Any] = {"config_file": str(spec.resolve())}
        else:
            body = {"config": spec.model_dump(mode="json")}
            if base_dir is not None:
                body["base_dir"] = _absolute(base_dir)
        return Validation.model_validate(self._call("POST", "/configs:validate", body=body))

    # ------------------------------------------------------------------ work

    def submit_run(
        self,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        *,
        annotations: Annotations | None = None,
        base_dir: Path | None = None,
        max_steps: int | None = None,
        force_new: bool = False,
        priority: str = "normal",
        client: str = "api",
        approval_token: str | None = None,
        code: str | None = None,
    ) -> JobHandle:
        """Run a configuration in the background; an identical run is reused.

        ``code``: ``live``, ``frozen`` or a commit (the owner only).
        """
        body = self._config(config, "config") | {
            "code": code,
            "annotations": dict(annotations or {}),
            "base_dir": None if base_dir is None else _absolute(base_dir),
            "max_steps": max_steps,
            "force_new": force_new,
            "priority": priority,
            "client": client,
            "approval_token": approval_token,
        }
        return JobHandle.model_validate(self._call("POST", "/runs", body=body))

    def submit_study(
        self,
        spec: StudySpec | Path,
        *,
        base_dir: Path | None = None,
        approval_token: str | None = None,
        priority: str = "normal",
        client: str = "api",
        code: str | None = None,
    ) -> JobHandle:
        """Run a study's cells (and predictions) as jobs; identical runs are reused."""
        body: dict[str, Any] = {
            "code": code,
            "base_dir": None if base_dir is None else _absolute(base_dir),
            "approval_token": approval_token,
            "priority": priority,
            "client": client,
        }
        if isinstance(spec, Path):
            body["study_file"] = str(spec.resolve())
        else:
            body["study"] = spec.model_dump(mode="json")
        return JobHandle.model_validate(self._call("POST", "/studies", body=body))

    def submit_prediction(
        self,
        config: PredictionConfig | Path,
        *,
        annotations: Annotations | None = None,
        priority: str = "normal",
        client: str = "api",
        code: str | None = None,
    ) -> JobHandle:
        """Forecast a run beyond history in the background."""
        body = self._config(config, "prediction") | {
            "code": code,
            "annotations": dict(annotations or {}),
            "priority": priority,
            "client": client,
        }
        return JobHandle.model_validate(self._call("POST", "/predictions", body=body))

    def status(self, job: str) -> JobStatus:
        """Return where a job is."""
        return JobStatus.model_validate(self._call("GET", f"/jobs/{job}"))

    def events(self, job: str, *, since: int = 0) -> list[Event]:
        """Return a job's events from number ``since`` on."""
        events: list[Event] = []
        while True:
            page = self._call("GET", f"/jobs/{job}/events", params={"since": since, "limit": 1000})
            batch = [Event.model_validate(e) for e in page["events"]]
            events.extend(batch)
            if len(batch) < 1000:
                return events
            since = int(page["next_since"])

    def follow(self, job: str, *, since: int = 0) -> Iterator[Event]:
        """Yield a job's events as they happen (Server-Sent Events), until it ends."""
        import json

        headers = {"Accept": "text/event-stream", "Last-Event-ID": str(since - 1)}
        with self.http.stream("GET", f"/v1/jobs/{job}/events", headers=headers) as response:
            if response.status_code >= 400:
                response.read()
                self._call("GET", f"/jobs/{job}")  # raises the matching error
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield Event.model_validate(json.loads(line.removeprefix("data: ")))

    def stop(self, job: str) -> None:
        """Stop a job: queued ones are cancelled, running runs stop at their next checkpoint."""
        self._call("POST", f"/jobs/{job}:stop")

    def resume(
        self, job: str, *, max_steps: int | None = None, code: str | None = None
    ) -> JobHandle:
        """Continue a stopped run job."""
        body = {"max_steps": max_steps, "code": code}
        return JobHandle.model_validate(self._call("POST", f"/jobs/{job}:resume", body=body))

    def wait(self, job: str, *, timeout_s: float | None = None, poll_s: float = 1.0) -> JobStatus:
        """Block until a job ends (for scripts and tests); return its final status."""
        began = time.monotonic()
        while True:
            status = self.status(job)
            if status.state in FINAL:
                return status
            if timeout_s is not None and time.monotonic() - began > timeout_s:
                return status
            time.sleep(poll_s)

    # ------------------------------------------------------------------ studies and results

    def study_status(self, name: str) -> JobStatus:
        """Return a study's status: its job, with every cell's state."""
        return JobStatus.model_validate(self._call("GET", f"/studies/{name}"))

    def compare(self, study: str, *, metrics: Sequence[str] | None = None) -> Comparison:
        """Compare a study's variants at each run's posterior."""
        params = {"metric": list(metrics)} if metrics else None
        return Comparison.model_validate(
            self._call("GET", f"/studies/{study}/comparison", params=params)
        )

    def run_summary(self, run: str) -> RunSummary:
        """Return a run's results, by run id (a directory is a local-only lookup)."""
        return RunSummary.model_validate(self._call("GET", f"/runs/{run}"))

    def run_steps(self, run: str) -> RunSteps:
        """Return a run's performance metrics step by step, by run id."""
        return RunSteps.model_validate(self._call("GET", f"/runs/{run}/steps"))

    def prediction_summary(self, path: Path) -> PredictionSummary:
        """Return a finished prediction's results (``path`` on the service's machine)."""
        return PredictionSummary.model_validate(
            self._call("GET", "/predictions/summary", params={"path": _absolute(path)})
        )

    def validate_optimization(
        self, config: OptimizationConfig | Mapping[str, Any] | Path, base_dir: Path | None = None
    ) -> Validation:
        """Validate an optimization configuration and plan its cost."""
        body = self._config(config, "config") | (
            {"base_dir": _absolute(base_dir)} if base_dir is not None else {}
        )
        return Validation.model_validate(self._call("POST", "/configs:validate", body=body))

    def submit_optimization(
        self,
        config: OptimizationConfig | Mapping[str, Any] | Path,
        *,
        base_dir: Path | None = None,
        max_iterations: int | None = None,
        annotations: Annotations | None = None,
        priority: str = "normal",
        client: str = "api",
        approval_token: str | None = None,
        code: str | None = None,
    ) -> JobHandle:
        """Optimize in the background."""
        body = self._config(config, "optimization") | {
            "code": code,
            "max_iterations": max_iterations,
            "annotations": dict(annotations or {}),
            "priority": priority,
            "client": client,
            "approval_token": approval_token,
        }
        if base_dir is not None:
            body["base_dir"] = _absolute(base_dir)
        return JobHandle.model_validate(self._call("POST", "/optimizations", body=body))

    def optimizations(self) -> list[dict[str, Any]]:
        """Return the optimization jobs this token may see."""
        rows: list[dict[str, Any]] = self._call("GET", "/optimizations")
        return rows

    def optimization_summary(self, path: Path) -> OptimizationSummary:
        """Return an optimization's progress and result (``path`` on the service's machine)."""
        return OptimizationSummary.model_validate(
            self._call("GET", "/optimizations/summary", params={"path": _absolute(path)})
        )

    def runs(self, annotations: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        """Return the catalogue's runs, optionally those with all of ``annotations``."""
        params: dict[str, Any] = {"limit": 500}
        if annotations:
            params["annotation"] = [f"{k}={v}" for k, v in annotations.items()]
        rows: list[dict[str, Any]] = []
        while True:
            page = self._call("GET", "/runs", params=params)
            rows.extend(page["items"])
            if page["next_cursor"] is None:
                return rows
            params["cursor"] = page["next_cursor"]

    def artifact(self, run: str, path: str) -> bytes:
        """Return a file of a run, e.g. ``report/metrics_box.png``."""
        response = self.http.get(f"/v1/runs/{run}/artifacts/{path}")
        if response.status_code == 404:
            raise KeyError(f"no file {path!r} in run {run}")
        if response.status_code >= 400:
            # A wrong token, the policy or a service fault is not a missing file.
            self._raise_for(response)
        return bytes(response.content)


class Client(ApexService):
    """APEX's services, in this process (see the module docstring)."""

    def validate(
        self,
        config: HistoryMatchConfig | Mapping[str, Any] | Path,
        base_dir: Path | None = None,
        engine: EngineMethod | None = None,
        *,
        check_output: bool = True,
        place_in: Path | None = None,
        annotations: Annotations | None = None,
    ) -> Validation:
        """Validate a configuration and plan its cost, as submitting it would judge it.

        :meth:`ApexService.validate`, with ``check_output`` on by default: a run folder
        holding another run is an error, as the REST API's validation says.
        """
        return super().validate(config, base_dir, engine, check_output=check_output,
                                place_in=place_in, annotations=annotations)  # fmt: skip

    @classmethod
    def connect(cls, url: str, token: str | None = None) -> RemoteClient:
        """Connect to a remote APEX service over its REST API (needs the ``api`` extra)."""
        return RemoteClient(url, token)
