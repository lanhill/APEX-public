"""One tool registry for any LLM client.

Each tool is defined once (:class:`Tool`: a name, a short description with an example,
a flat argument model, whether it spends compute, and a handler over the services); the
MCP server (``apex mcp``), the function-calling exports (``apex tools export``) and the
dispatcher (:func:`call`, ``apex tools call``) are generated from :data:`TOOLS`, so
Claude Code, other MCP hosts and OpenAI-compatible agent loops over local models (Ollama,
vLLM, llama.cpp, Kimi, GLM) use the same tools.

- **Errors are results**, never exceptions: ``{"ok": false, "error": ..., "hint": ...}``
  (unknown names with "did you mean", invalid arguments with what is allowed), so a
  model can correct itself.
- **Results are bounded** (8 kB of JSON): lists are shortened and marked ``truncated``.
- **Spending needs a token** from ``validate``, bound to what was validated. Below the
  approval limits (``service.toml``) it is approved at once (and recorded); above them it
  is pending until a person runs ``apex approve TOKEN``. The service enforces this, not
  the host or the model.
- **Session budgets** cap the simulator runs one client session may submit.
- **Every call is recorded** in the store (``tool_calls``): client, model, tool,
  arguments, the outcome.
"""

import difflib
import hashlib
import json
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apex.core.errors import ApexError, ConfigError
from apex.domain.config import HistoryMatchConfig
from apex.domain.prediction import PredictionConfig
from apex.services import composition, policy, prediction
from apex.services.api import ApexService, SchemaKind
from apex.services.approval import fingerprint
from apex.services.identity import run_id
from apex.storage.catalogue import now

if TYPE_CHECKING:
    from apex.services.access import Access

#: Largest result, in characters of JSON: small models have small contexts.
RESULT_LIMIT = 8000

#: Every registry of ``services.plugins.REGISTRIES`` (a test keeps the two equal).
Registry = Literal[
    "methods", "update_strategies", "localizations", "damping_schemes", "gmac_norms",
    "gmac_transforms", "gmac_constraints", "gmac_constraint_metrics", "priors",
    "parameterizations", "model_error_models", "simulators", "executors", "data_transforms",
    "metrics", "stopping_rules", "forecast_scores", "optimizers", "objectives",
    "response_objectives", "step_rules", "figures", "figure_presets", "petro_elastic_models",
    "reflectivity", "avo_attributes", "displacement_kernels", "chop_components",
    "converter_skills", "geomodeling_skills", "variogram_estimators",
]  # fmt: skip


class _Arguments(BaseModel):
    # Unknown arguments are errors, as in every APEX configuration.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------- arguments


class ListPlugins(_Arguments):
    """Arguments of ``list_plugins``."""

    registry: Registry | None = Field(default=None, description="Only this registry.")
    details: bool = Field(default=False, description="Include each plugin's option types.")


class GetSchema(_Arguments):
    """Arguments of ``get_schema``."""

    kind: SchemaKind = "history_match"
    section: str | None = Field(
        default=None,
        description="Only one top-level section, e.g. 'prior' or 'method', with its "
        "definitions inlined: the whole schema is long and may be cut off.",
    )


class Validate(_Arguments):
    """Arguments of ``validate``: a configuration file, or a job to resume."""

    config_file: str | None = Field(
        default=None, description="A run, study, prediction or optimization file (TOML)."
    )
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Changes to a run's configuration, by dotted key, e.g. "
        '{"method.max_iterations": 10}.',
    )
    resume_job: str | None = Field(default=None, description="A stopped job to resume.")
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description='The submission\'s annotations: {"experiment": NAME} lets a grant of '
        "this client for that experiment approve it (submit with the same annotations).",
    )


class SubmitRun(_Arguments):
    """Arguments of ``submit_run``."""

    config_file: str
    approval_token: str = Field(description="From validate, for the same file and overrides.")
    overrides: dict[str, Any] = Field(default_factory=dict)
    priority: Literal["high", "normal", "low"] = "normal"
    annotations: dict[str, str] = Field(
        default_factory=dict, description='Why, e.g. {"hypothesis": "H3"}.'
    )


class SubmitStudy(_Arguments):
    """Arguments of ``submit_study``."""

    study_file: str
    approval_token: str
    priority: Literal["high", "normal", "low"] = "normal"
    annotations: dict[str, str] = Field(default_factory=dict)


class SubmitPrediction(_Arguments):
    """Arguments of ``submit_prediction``."""

    prediction_file: str
    approval_token: str
    annotations: dict[str, str] = Field(default_factory=dict)


class SubmitOptimization(_Arguments):
    """Arguments of ``submit_optimization``."""

    optimization_file: str
    approval_token: str = Field(description="From validate, for the same file.")
    max_iterations: int | None = Field(
        default=None, ge=1, description="Stop after this many accepted steps (resumable)."
    )
    priority: Literal["high", "normal", "low"] = "normal"
    annotations: dict[str, str] = Field(default_factory=dict)


class OptimizationPath(_Arguments):
    """Arguments of ``optimization_summary``."""

    path: str = Field(description="The optimization's directory (from job_status's run_dirs).")


class JobId(_Arguments):
    """A job."""

    job_id: str


class JobEvents(_Arguments):
    """Arguments of ``job_events``."""

    job_id: str
    since: int = Field(default=0, ge=0, description="Events numbered from this one.")
    limit: int = Field(default=50, ge=1, le=500)


class ResumeJob(_Arguments):
    """Arguments of ``resume_job``."""

    job_id: str
    approval_token: str = Field(description="From validate(resume_job=...).")
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description='As given to validate: {"experiment": NAME} lets a grant pay for it.',
    )


class RunId(_Arguments):
    """A run (its id, or its first characters)."""

    run_id: str


class CompareStudy(_Arguments):
    """Arguments of ``compare_study``."""

    study: str
    metrics: list[str] | None = Field(default=None, description='e.g. ["rmse:all"].')


class PredictionPath(_Arguments):
    """Arguments of ``prediction_summary``."""

    path: str


class ListRuns(_Arguments):
    """Arguments of ``list_runs``."""

    annotations: dict[str, str] = Field(default_factory=dict, description="Only runs with these.")
    limit: int = Field(default=50, ge=1, le=500, description="At most this many, newest first.")
    offset: int = Field(default=0, ge=0, description="Skip this many newest runs (next_offset).")


# ---------------------------------------------------------------------- the session


@dataclass
class ToolSession:
    """One client's session: who calls, and what they may still spend.

    Attributes
    ----------
    service
        The services.
    client
        The client's name (quotas, approvals and the call log use it).
    model, host
        The model and host the client declares, for the record.
    budget_runs
        Simulator runs this session may submit (default: the client's daily quota).
    """

    service: ApexService
    client: str = "llm"
    model: str | None = None
    host: str | None = None
    budget_runs: int | None = None
    used_runs: int = 0
    #: The web UI's allowlist for a UI user's session (the assistant's): the
    #: read tools see only what it allows. ``None``: everything.
    access: "Access | None" = None
    #: Whether the files the client names must lie in the folders it may read
    #: (:meth:`readable`); off only where the owner named the file (``apex agent plan``).
    confined: bool = True

    def __post_init__(self) -> None:
        """Default the budget to the client's daily quota."""
        if self.budget_runs is None:
            quota = self.service.jobs.settings.quotas.get(self.client)
            self.budget_runs = None if quota is None else quota.max_simulator_runs_per_day

    def readable(self) -> list[str] | None:
        """Return the folders whose files this client may name, or ``None`` for any.

        Under a ``[policy]`` the policy decides (``policy.check_file``), so ``None`` here.
        Otherwise the workspace's roots, ``[tools] roots`` and the service's own ``runs/``,
        ``studies/`` and ``optimizations/``: a tool client acts as the service's owner,
        and without this it could read any file the owner can (release tester N3).
        """
        settings = self.service.jobs.settings
        if not self.confined or policy.active(settings.policy):
            return None
        roots = [str(r) for r in self.service.workspace().roots]
        return [*roots, *settings.tools.roots, *policy.service_roots(self.service.root)]


class ToolFailure(Exception):
    """A tool's error, returned to the model as a result with a hint."""

    def __init__(self, error: str, hint: str = "") -> None:
        super().__init__(error)
        self.error, self.hint = error, hint


Handler = Callable[[ToolSession, Any], Any]


@dataclass(frozen=True)
class Tool:
    """One tool: its name, description, argument model and handler."""

    name: str
    description: str
    arguments: type[_Arguments]
    spends_compute: bool
    handler: Handler = field(repr=False)

    def schema(self) -> dict[str, Any]:
        """Return the JSON Schema of the arguments (self-contained, no ``$ref``)."""
        schema = self.arguments.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)
        return _inline(schema)


def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``s into the schema: some function-calling servers reject them."""
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:  # noqa: ANN401 (JSON)
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(definitions[node["$ref"].split("/")[-1]])
            options = node.get("anyOf")
            if isinstance(options, list) and len(options) == 2 and {"type": "null"} in options:
                # "X or null" with a default: plain X, optional as it is not required. Small
                # models follow flat types more reliably than anyOf.
                other = next(o for o in options if o != {"type": "null"})
                merged = {k: v for k, v in node.items() if k not in ("anyOf", "default")}
                return resolve({**other, **merged})
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    result: dict[str, Any] = resolve(schema)
    return result


# ---------------------------------------------------------------------- helpers


def _path(session: ToolSession, value: str, what: str) -> Path:
    path = Path(value).expanduser()
    service = session.service
    try:
        # The service's [policy] before the file is opened, or even looked for:
        # otherwise its values come back in the validation errors, and "not found" tells
        # whether a file exists (as ApexService.resolve checks a path it is given).
        policy.check_file(path, service.jobs.settings.policy, service.root)
    except policy.PolicyViolation as error:
        raise ToolFailure(
            str(error), f"give a {what} file under the service's input roots"
        ) from error
    _confine(session, path, what)
    if not path.is_file():
        raise ToolFailure(f"{what} {value!r} not found", "give an absolute path to a TOML file")
    return path.resolve()


def _outside(session: ToolSession, roots: list[str], hint: str) -> ToolFailure:
    """Return the refusal of a path outside ``roots``.

    The roots are listed to the owner's own clients only: a web-UI user's session (the
    assistant's, ``access`` set) gets no server path (finding 31).
    """
    where = f": {', '.join(roots)}" if session.access is None else ""
    return ToolFailure(f"outside the folders tool clients may read{where}", hint)


def _inside(value: str | Path, roots: list[str]) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() and any(policy.under(path.resolve(), root) for root in roots)


#: What a tool client's ``overrides`` may not set: what runs, not only what is read.
UNSAFE_OVERRIDES = frozenset(
    {"code", "executable", "arguments", "undeclared", "host", "remote_program"}
)


def _confine(session: ToolSession, path: Path, what: str) -> None:
    """Refuse a path outside the folders the session may read (:meth:`ToolSession.readable`).

    Checked before the file is opened or looked for, as the policy is: "not found" would
    tell whether a file exists.
    """
    roots = session.readable()
    if roots is None:
        return
    absolute = Path(path).expanduser().absolute()
    if not _inside(absolute, roots):
        folder = absolute.parent
        refusal = _outside(
            session, roots,
            f"give a {what} file under one of them, or ask the service's owner to add its "
            "folder to [tools] roots in service.toml"
            + (f' ([tools]\nroots = ["{folder}"])' if session.access is None else ""),
        )  # fmt: skip
        raise ToolFailure(f"{path} is {refusal.error}", refusal.hint)


def _readable_item(
    kind: str, value: str, roots: list[str], runs: list[str], *, trusted: bool = True
) -> bool:
    """Whether one thing a configuration names lies where a confined client may name it.

    Files read (``input``) must lie in the folders; templates and programs (``code``,
    ``executable``) written in the owner's own configuration file are the owner's choice
    (a GEOS build, an MPSlib binary in a home folder), and only ``overrides`` are refused
    them (:data:`UNSAFE_OVERRIDES`). Not ``trusted``: the configuration, or the template
    or program, lies in another web-UI user's folder (a guest's, an agent's), so running
    it would run their code as the owner; then they are refused, unless a ``[policy]``
    decides (and this check is off).
    """
    if kind == "input":
        return _inside(value, roots)
    if kind in ("code", "executable"):
        return trusted
    if kind in ("output", "source"):
        return _inside(value, [*roots, *runs])
    return True


def _confine_config(
    session: ToolSession,
    config: BaseModel,
    before: BaseModel | None = None,
    sources: Sequence[Path] = (),
) -> None:
    """Refuse a resolved configuration that names anything outside the readable folders.

    Every path, template and program it names (APEX's fields by name, plugin options by
    their declarations, :func:`policy.named`), checked before anything reads,
    renders or runs them, as a ``[policy]`` would be with its roots the session's. Without
    this, a configuration inside the folders, or ``overrides`` of one, could name any file,
    and the readers' errors show what it holds (review of release tester N3). ``before`` is
    the configuration without the overrides: what the overrides change may not be a
    template, a program, its arguments, a host or an undeclared option; a ``[policy]``
    decides those (and then this check is off: the policy's own runs instead).
    """
    roots = session.readable()
    if roots is None:
        return
    runs = [str(e.path) for e in session.service.store.entries()]
    if isinstance(config, PredictionConfig) and not _inside(
        config.prediction.run_dir, [*roots, *runs]
    ):
        # Before policy.named, which reads the run's snapshot to judge its simulator.
        refusal = _outside(session, roots, "give a run of the catalogue or of those folders")
        raise ToolFailure(f"prediction.run_dir: {refusal.error}", refusal.hint)
    known = set(policy.named(before)) if before is not None else None
    space = session.service.workspace()
    # For the owner's clients (MCP, the CLI's tools, agent loops); a web-UI user's session
    # (the assistant's, ``access`` set) is held to its own allowlist.
    owners = session.access is None
    written_by_others = owners and any(space.in_other_ui_user_folder(f) for f in sources)
    problems = []
    for kind, key, value in policy.named(config):
        if kind in ("code", "executable") and written_by_others:
            problems.append(f"{key}: a template or program in a configuration another web-UI "
                            "user wrote is refused (a [policy] decides those)")  # fmt: skip
        elif known is not None and kind in UNSAFE_OVERRIDES and (kind, key, value) not in known:
            problems.append(f"{key}: overrides may not set templates, programs, arguments or "
                            "hosts (a [policy] decides those)")  # fmt: skip
        elif not _readable_item(
            kind, value, roots, runs,
            trusted=not (owners and space.in_other_ui_user_folder(value)),
        ):  # fmt: skip
            problems.append(f"{key} = {value!r}")
    if problems:
        refusal = _outside(
            session, roots,
            "use files under those folders, or ask the service's owner to add theirs to "
            "[tools] roots in service.toml",
        )  # fmt: skip
        raise ToolFailure(f"{'; '.join(problems[:5])}: {refusal.error}", refusal.hint)


def _confine_optimization(session: ToolSession, path: Path) -> None:
    """Refuse an optimization file whose configuration names files outside the folders."""
    if session.readable() is None:
        return
    from apex.services.optimization import load_optimization_config

    try:
        config = load_optimization_config(path)
    except ApexError as error:
        # Refused here: validating it would read what it names, unchecked.
        raise ToolFailure(str(error), "fix the optimization file") from error
    _confine_config(session, config, sources=[path])


def _kind(path: Path) -> str:
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ToolFailure(f"cannot read {path}: {error}") from error
    if "optimization" in document and "optimizer" in document:
        return "optimization"
    first = next(iter(document), "")
    return {"study": "study", "prediction": "prediction"}.get(first, "history_match")


def _set(document: dict[str, Any], dotted: str, value: Any) -> None:  # noqa: ANN401
    keys = dotted.split(".")
    node = document
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ToolFailure(f"override {dotted!r}: {key!r} is not a table")
        node = child
    node[keys[-1]] = value


def _run_config(
    session: ToolSession, file: str, overrides: Mapping[str, Any]
) -> HistoryMatchConfig:
    path = _path(session, file, "configuration")
    base = composition.load_config(path)
    _confine_config(session, base, sources=[path])
    if not overrides:
        return base
    document = tomllib.loads(path.read_text())
    for dotted, value in overrides.items():
        _set(document, dotted, value)
    if session.readable() is not None:
        # Checked before ApexService.resolve, which builds the simulator (and may run it).
        changed = composition._resolve_paths(
            HistoryMatchConfig.model_validate(document), path.parent
        )
        _confine_config(session, changed, before=base, sources=[path])
    return session.service.resolve(document, base_dir=path.parent)


def reuse_fingerprint(run: str) -> str:
    """Return the fingerprint of a token for reusing the run ``run`` as it is.

    Never a submission's (``fingerprint([run])``), so such a token pays for no simulation.
    """
    return fingerprint([f"reuse:{run}"])


def _prediction_print(config_json: str) -> str:
    return fingerprint(["prediction:" + hashlib.sha256(config_json.encode()).hexdigest()])


def _plan_summary(plan: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "run_id": plan.run_id[:12],
        "method": plan.method,
        "members": plan.members,
        "data": plan.data,
        "max_forecasts": plan.max_forecasts,
        "max_simulator_runs": plan.max_simulator_runs,
        "simulator": plan.simulator,
        "estimated_hours": plan.estimated_hours,
        "checkpoint_mb": plan.checkpoint_mb,
        "forecast_memory_mb": plan.forecast_memory_mb,
        "warnings": plan.warnings,
    }


def _issue_token(
    session: ToolSession,
    print_: str,
    simulator_runs: int | None,
    runs: int,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Return an approval token for what was validated, approved at once below the limits.

    Above them it is pending, unless a grant of the session's client for
    ``experiment`` covers it: then the submission with the same ``experiment`` annotation
    charges the grant (:func:`_spend`). Checked, not charged, here: a validation that is
    never submitted spends nothing.
    """
    service = session.service
    approvals = service.approvals
    cost = {"simulator_runs": simulator_runs, "runs": runs}
    token = approvals.request(print_, cost, requested_by=session.client)
    if approvals.needed(simulator_runs, runs):
        grant = service.grants.covering(session.client, experiment, simulator_runs)
        if grant is not None:
            left = grant["max_simulator_runs"] - grant["used_simulator_runs"]
            return {
                "approval_token": token,
                "approval": "granted",
                "grant": grant["id"],
                "next": f"submit with this token and annotations {{'experiment': "
                f"{experiment!r}}}: grant {grant['id']} pays for it ({left} simulator runs "
                "left)",
            }
        why = service.grants.why_not(session.client, experiment, simulator_runs)
        return {
            "approval_token": token,
            "approval": "pending",
            "next": f"ask a person to run `apex approve {token}`, then submit with this token"
            + (f" (no grant applies: {why})" if why else ""),
        }
    approvals.approve(token, by="auto: below the approval limits")
    return {"approval_token": token, "approval": "approved", "next": "submit with this token"}


@dataclass
class Spending:
    """A token used (and maybe a grant charged) for one submission, until it is made.

    :meth:`rollback` takes it back when the submission then fails, so that neither the
    grant nor the token pays for a job that does not exist (review).
    """

    session: ToolSession
    token: str
    simulator_runs: int | None
    grant: str | None = None

    def rollback(self) -> None:
        """Release the token, refund the grant and the session's budget."""
        service = self.session.service
        if self.grant is not None:
            service.grants.refund(self.grant, self.token)
        service.approvals.release(self.token, unapprove=self.grant is not None)
        self.session.used_runs -= self.simulator_runs or 0


def _spend(
    session: ToolSession,
    token: str,
    print_: str,
    simulator_runs: int | None,
    experiment: str | None = None,
) -> Spending:
    """Check the session budget, then use the token (both before anything is submitted).

    A token still pending is approved by a grant of the client for ``experiment`` that
    covers the cost, if there is one: the token is checked again, approved, used and the
    grant charged in one store transaction (two submissions of one token cannot both
    charge it), as at the service's gate. Call
    :meth:`Spending.rollback` if the submission then fails.
    """
    budget = session.budget_runs
    if budget is not None and (
        simulator_runs is None or session.used_runs + simulator_runs > budget
    ):
        raise ToolFailure(
            f"this session's budget is {budget} simulator runs ({session.used_runs} used); "
            f"this submission may take {simulator_runs}",
            "submit something smaller, or ask a person to raise the client's quota",
        )
    approvals = session.service.approvals
    record = approvals.store.approval(token)
    grant: dict[str, Any] | None = None
    if record is not None and not record["approved_by"] and not record["used"]:

        def use(grant_id: str) -> str:
            # Inside the grant's transaction: the token as it is now, not as read above.
            approvals.approve(token, by=f"grant {grant_id}")
            approvals.consume(token, print_)
            return token

        try:
            grant = session.service.grants.charge(
                session.client, experiment, simulator_runs, record=use
            )
        except ConfigError as error:
            raise ToolFailure(str(error), "call validate again for this submission") from error
    if grant is None:
        try:
            approvals.consume(token, print_)
        except ConfigError as error:
            raise ToolFailure(
                str(error), "call validate again for this exact submission"
            ) from error
    session.used_runs += simulator_runs or 0
    return Spending(session, token, simulator_runs, None if grant is None else grant["id"])


def _submitted(spending: Spending, submit: Callable[[], Any]) -> Any:  # noqa: ANN401
    """Return ``submit()``; on failure take back what ``spending`` used, then re-raise.

    Only when nothing was submitted: a study queues its jobs before it records itself, so
    a failure after its first job leaves work that runs, which the grant and the token
    have paid for. Any job queued meanwhile counts (another client's too): keeping a
    charge is the safe side, refunding running work is not.
    """
    jobs = spending.session.service.jobs
    before = {r["id"] for r in jobs.all_records()}
    try:
        return submit()
    except BaseException:
        if {r["id"] for r in jobs.all_records()} <= before:
            spending.rollback()
        raise


# ---------------------------------------------------------------------- handlers


def _list_plugins(session: ToolSession, args: ListPlugins) -> Any:  # noqa: ANN401
    rows = []
    for info in session.service.plugins():
        if args.registry and info.registry != args.registry:
            continue
        if session.access is not None and not session.access.plugin(info.registry, info.name):
            continue
        row: dict[str, Any] = {"registry": info.registry, "name": info.name}
        if info.status:
            row["status"] = info.status
        row["summary"] = info.summary
        properties = info.options.get("properties", {})
        if args.details:
            row["options"] = {
                k: {key: v[key] for key in ("type", "default", "enum", "anyOf") if key in v}
                for k, v in properties.items()
            }
        else:
            row["options"] = list(properties)
        rows.append(row)
    return {"plugins": rows}


def _get_schema(session: ToolSession, args: GetSchema) -> Any:  # noqa: ANN401
    schema = session.service.schema(args.kind)
    if args.section is None:
        return schema
    # One section, self-contained: a model learns a section's layout (e.g. [prior]) without
    # the rest of a schema that the tool layer's bound would cut short (issue #6).
    sections = list(schema.get("properties", {}))
    if args.section not in sections:
        raise ToolFailure(f"no section {args.section!r} in the {args.kind} schema",
                          f"sections: {', '.join(sections)}")  # fmt: skip
    whole = _inline(schema)
    return {"kind": args.kind, "section": args.section,
            "schema": whole["properties"][args.section],
            "required": args.section in whole.get("required", [])}  # fmt: skip


def _validate(session: ToolSession, args: Validate) -> Any:  # noqa: ANN401
    service = session.service
    experiment = args.annotations.get("experiment")
    if args.resume_job:
        record = _job(service, args.resume_job)
        runs = int(record.get("planned_runs") or 0) or None
        print_ = fingerprint([*record["run_ids"], f"resume:{record['id']}"])
        return {"valid": True, **_issue_token(session, print_, runs, 1, experiment)}
    if not args.config_file:
        raise ToolFailure(
            "give config_file, or resume_job", 'e.g. validate(config_file="/path/run.toml")'
        )
    path = _path(session, args.config_file, "configuration")
    kind = _kind(path)
    if kind == "study":
        if args.overrides:
            raise ToolFailure("overrides apply to runs only", "change the study's variants instead")
        return _validate_study(session, path, experiment)
    if kind == "prediction":
        return _validate_prediction(session, path)
    if kind == "optimization":
        if args.overrides:
            raise ToolFailure("overrides apply to runs only", "edit the optimization file")
        return _validate_optimization(session, path, experiment)
    try:
        config = _run_config(session, args.config_file, args.overrides)
    except ApexError as error:
        return {"valid": False, "errors": [str(error)]}
    # As submitting it would judge it: a run folder holding another run is an error.
    validation = service.validate(config, check_output=True)
    if not validation.valid or validation.plan is None:
        return {"valid": False, "errors": validation.errors[:20]}
    plan = validation.plan
    print_ = fingerprint([plan.run_id])
    if service.free_reuse(config):
        # An identical run is complete, queued or running: submitting it reuses it and runs
        # nothing, so the token is approved at once, whatever the cost (no grant is used).
        # Its fingerprint is a reuse's, not the run's: should the run no longer be reusable
        # (its folder pruned, its job stopped), the token cannot pay for running it.
        token = service.approvals.request(
            reuse_fingerprint(plan.run_id), {"simulator_runs": 0, "runs": 0},
            requested_by=session.client,
        )  # fmt: skip
        service.approvals.approve(token, by="auto: an identical run is reused")
        return {
            "valid": True,
            "plan": _plan_summary(plan),
            "reused": True,
            "approval_token": token,
            "approval": "approved",
            "next": "submit with this token: the identical run is reused, nothing runs",
        }
    return {
        "valid": True,
        "plan": _plan_summary(plan),
        **_issue_token(session, print_, plan.max_simulator_runs, 1, experiment),
    }


def _checked_study(session: ToolSession, path: Path) -> tuple[Any, Path]:
    # The study's case files too, before they are opened: the study file alone
    # being inside the roots says nothing about the cases it names.
    try:
        spec, base = session.service.checked_study(path)
    except policy.PolicyViolation as error:
        raise ToolFailure(str(error), "give case files under the service's input roots") from error
    for case in spec.study.cases.values():
        _confine(session, Path(case), "case")
    if session.readable() is not None:
        # Each cell as it would run: a variant's patch or a method entry may name files too.
        from apex.services.prediction import simulator_section
        from apex.services.study import _prediction_config, expand

        try:
            cells = expand(spec, session.service.root, base)
        except ConfigError as error:
            # Refused here: planning would validate the cells, reading what they name.
            raise ToolFailure(str(error), "fix the study or its cases") from error
        # A cell is written by the study file and its case file together.
        sources = [path, *(Path(c) for c in spec.study.cases.values())]
        for cell in cells:
            _confine_config(session, cell.config, sources=sources)
            assert cell.config.run.output_dir is not None
            for template in spec.study.predictions:
                # Each prediction as it would run, and the cell with its simulator.
                try:
                    forecast = _prediction_config(template, cell.config.run.output_dir, base)
                except ConfigError as error:
                    raise ToolFailure(str(error), "fix the study's predictions") from error
                _confine_config(session, forecast, sources=sources)
                simulator = simulator_section(forecast, cell.config)
                _confine_config(
                    session,
                    cell.config.model_copy(update={"simulator": simulator}),
                    sources=sources,
                )
    return spec, base


def _validate_study(
    session: ToolSession, path: Path, experiment: str | None = None
) -> dict[str, Any]:
    from apex.services.study import plan_study

    spec, base = _checked_study(session, path)
    experiment = experiment or spec.study.annotations.get("experiment")
    study_plan = plan_study(session.service, spec, base)
    checked = study_plan.validation
    if not checked.valid:
        return {"valid": False, "errors": checked.errors[:20]}
    total = study_plan.max_simulator_runs
    return {
        "valid": True,
        "runs": len(checked.plans),
        "max_simulator_runs": total,
        **_issue_token(
            session,
            fingerprint([p.run_id for p in checked.plans]),
            total,
            len(checked.plans),
            experiment,
        ),
    }


def _validate_prediction(session: ToolSession, path: Path) -> dict[str, Any]:
    try:
        forecast = prediction.load_prediction_config(path)
    except ApexError as error:
        return {"valid": False, "errors": [str(error)]}
    _confine_config(session, forecast, sources=[path])
    print_ = _prediction_print(forecast.model_dump_json())
    return {"valid": True, **_issue_token(session, print_, None, 1)}


def _validate_optimization(
    session: ToolSession, path: Path, experiment: str | None = None
) -> dict[str, Any]:
    _confine_optimization(session, path)
    validation = session.service.validate_optimization(path)
    plan = validation.optimization
    if not validation.valid or plan is None:
        return {"valid": False, "errors": validation.errors[:20]}
    return {
        "valid": True,
        "plan": {
            "optimization_id": plan.optimization_id[:12],
            "optimizer": plan.optimizer,
            "objective": plan.objective,
            "n_controls": plan.n_controls,
            "max_evaluations": plan.max_evaluations,
            "max_simulator_runs": plan.max_simulator_runs,
            "simulator": plan.simulator,
            "warnings": plan.warnings,
        },
        **_issue_token(
            session, fingerprint([plan.optimization_id]), plan.max_simulator_runs, 1, experiment
        ),
    }


def _submit_optimization(session: ToolSession, args: SubmitOptimization) -> Any:  # noqa: ANN401
    path = _path(session, args.optimization_file, "optimization")
    _confine_optimization(session, path)
    validation = session.service.validate_optimization(path)
    plan = validation.optimization
    if not validation.valid or plan is None:
        raise ToolFailure("; ".join(validation.errors[:5]), "fix the file and validate again")
    spending = _spend(session, args.approval_token, fingerprint([plan.optimization_id]),
                      plan.max_simulator_runs, args.annotations.get("experiment"))  # fmt: skip
    handle = _submitted(
        spending,
        lambda: session.service.submit_optimization(
            path,
            max_iterations=args.max_iterations,
            annotations=_annotations(session, args.annotations),
            priority=args.priority,
            client=session.client,
            approved=True,
        ),
    )
    return {"job_id": handle.id, "optimization_id": plan.optimization_id[:12]}


def _optimization_summary(session: ToolSession, args: OptimizationPath) -> Any:  # noqa: ANN401
    path = Path(args.path)
    _confine_result(session, path, "optimization")
    if session.access is not None:
        target = path.resolve()
        own = [
            r for r in session.service.jobs.all_records()
            if r["kind"] in ("optimization", "optimization_resume")
            and any(Path(d).resolve() == target for d in r.get("run_dirs") or [])
            and session.access.job(r)
        ]  # fmt: skip
        if not own:
            raise ToolFailure(f"no optimization at {args.path}", "see job_status's run_dirs")
    try:
        summary = session.service.optimization_summary(path)
    except (ApexError, policy.PolicyViolation) as error:
        raise ToolFailure(str(error), "give an optimization's directory") from error
    data = summary.model_dump(mode="json")
    data["history"] = data["history"][-20:]  # the last rows: small models, small results
    return data


def _submit_run(session: ToolSession, args: SubmitRun) -> Any:  # noqa: ANN401
    config = _run_config(session, args.config_file, args.overrides)
    validation = session.service.validate(config)
    if not validation.valid or validation.plan is None:
        raise ToolFailure("invalid configuration: " + "; ".join(validation.errors[:5]))
    service = session.service
    annotations = _annotations(session, args.annotations)
    if service.free_reuse(config):
        # Reused as it is: nothing runs, so no token is used and no grant is charged. The
        # service's own gate still applies should the reuse turn into a run (its job ended),
        # without the client's token: that run needs an approval of its own.
        before = {r["id"] for r in service.jobs.all_records()}
        handle = service.submit_run(
            config, annotations=annotations, priority=args.priority, client=session.client
        )
        new = handle.id not in before
        if new and (handle.kind == "resume" or not handle.reused):
            # It turned into a run (approved by the gate or a grant): the session spent it.
            # An existing live job handed back (even a resume) runs nothing new.
            session.used_runs += validation.plan.max_simulator_runs or 0
    else:
        spending = _spend(
            session,
            args.approval_token,
            fingerprint([run_id(config)]),
            validation.plan.max_simulator_runs,
            args.annotations.get("experiment"),
        )
        handle = _submitted(
            spending,
            lambda: service.submit_run(
                config,
                annotations=annotations,
                priority=args.priority,
                client=session.client,
                approved=True,
            ),
        )
    return {"job_id": handle.id, "run_id": handle.run_ids[0][:12], "reused": handle.reused}


def _submit_study(session: ToolSession, args: SubmitStudy) -> Any:  # noqa: ANN401
    from apex.services.study import plan_study

    path = _path(session, args.study_file, "study")
    spec, base = _checked_study(session, path)
    plan = plan_study(session.service, spec, base)
    if not plan.validation.valid:
        raise ToolFailure("invalid study: " + "; ".join(plan.validation.errors[:5]))
    print_ = fingerprint([p.run_id for p in plan.validation.plans])
    experiment = args.annotations.get("experiment") or spec.study.annotations.get("experiment")
    spending = _spend(session, args.approval_token, print_, plan.max_simulator_runs, experiment)
    spec = spec.model_copy(
        update={
            "study": spec.study.model_copy(
                update={
                    "annotations": {
                        **spec.study.annotations,
                        **_annotations(session, args.annotations),
                    }
                }
            )
        }
    )
    submitted = spec
    handle = _submitted(
        spending,
        lambda: session.service.submit_study(
            submitted, base_dir=base, priority=args.priority, client=session.client,
            approved=True,
        ),
    )  # fmt: skip
    return {"job_id": handle.id, "runs": len(handle.run_ids), "study": spec.study.name}


def _submit_prediction(session: ToolSession, args: SubmitPrediction) -> Any:  # noqa: ANN401
    path = _path(session, args.prediction_file, "prediction")
    config = prediction.load_prediction_config(path)
    _confine_config(session, config, sources=[path])
    spending = _spend(
        session,
        args.approval_token,
        _prediction_print(config.model_dump_json()),
        None if session.budget_runs else 0,
    )
    handle = _submitted(
        spending,
        lambda: session.service.submit_prediction(
            config, annotations=_annotations(session, args.annotations), client=session.client
        ),
    )
    return {"job_id": handle.id}


def _entries(session: ToolSession) -> list[Any]:
    """Return the catalogue's runs the session may see."""
    entries = list(session.service.store.entries())
    access = session.access
    if access is None:
        return entries
    return [e for e in entries if access.run(e.run_id, e.name, e.annotations, e.path)]


def _seen_job(session: ToolSession, job_id: str) -> dict[str, Any]:
    """Return a job's record, if the session may see it (as :func:`_job` otherwise)."""
    record = _job(session.service, job_id)
    if session.access is not None and not session.access.job(
        record, session.service.store.entries()
    ):
        raise ToolFailure(f"no job {job_id!r}", "see list_runs")
    return record


def _job(service: ApexService, job_id: str) -> dict[str, Any]:
    try:
        return service.jobs.record(job_id)
    except KeyError:
        ids = [r["id"] for r in service.jobs.all_records()]
        close = difflib.get_close_matches(job_id, ids, n=3)
        raise ToolFailure(
            f"no job {job_id!r}", f"did you mean {', '.join(close)}?" if close else "see list_runs"
        ) from None


def _job_status(session: ToolSession, args: JobId) -> Any:  # noqa: ANN401
    _seen_job(session, args.job_id)
    return session.service.status(args.job_id).model_dump(mode="json", exclude={"schema_version"})


def _job_events(session: ToolSession, args: JobEvents) -> Any:  # noqa: ANN401
    _seen_job(session, args.job_id)
    events = session.service.events(args.job_id, since=args.since)
    shown = events[: args.limit]
    return {
        "events": [
            {"index": e.index, "time": e.time, "event": e.event, "message": e.message[:200]}
            for e in shown
        ],
        "more": len(events) > len(shown),
        "next_since": (shown[-1].index + 1) if shown else args.since,
    }


def _stop_job(session: ToolSession, args: JobId) -> Any:  # noqa: ANN401
    _seen_job(session, args.job_id)
    session.service.stop(args.job_id)
    return {"stopping": args.job_id, "note": "a running run stops at its next checkpoint"}


def _resume_job(session: ToolSession, args: ResumeJob) -> Any:  # noqa: ANN401
    record = _seen_job(session, args.job_id)
    print_ = fingerprint([*record["run_ids"], f"resume:{record['id']}"])
    spending = _spend(session, args.approval_token, print_,
                      int(record.get("planned_runs") or 0) or None,
                      args.annotations.get("experiment"))  # fmt: skip
    return {"job_id": _submitted(spending, lambda: session.service.resume(args.job_id)).id}


def _run_summary(session: ToolSession, args: RunId) -> Any:  # noqa: ANN401
    matches = [e for e in _entries(session) if e.run_id.startswith(args.run_id)]
    if len(matches) != 1:
        raise ToolFailure(
            f"{len(matches)} runs match {args.run_id!r}", "use list_runs to find the id"
        )
    summary = session.service.run_summary(matches[0].run_id)
    return summary.model_dump(mode="json", exclude={"schema_version", "provenance"})


def _compare_study(session: ToolSession, args: CompareStudy) -> Any:  # noqa: ANN401
    if session.access is not None and not session.access.study(args.study):
        raise ToolFailure(f"no study {args.study!r}", "see list_runs")
    return session.service.compare(args.study, metrics=args.metrics).model_dump(mode="json")


def _confine_result(session: ToolSession, path: Path, what: str) -> None:
    """Refuse a result directory outside the readable folders and the catalogue's runs."""
    roots = session.readable()
    if roots is None:
        return
    runs = [str(e.path) for e in session.service.store.entries()]
    target = Path(path).expanduser().absolute().resolve()
    if not any(policy.under(target, root) for root in [*roots, *runs]):
        raise ToolFailure(
            f"{path} is outside the folders tool clients may read",
            f"give the {what}'s directory as job_status's run_dirs name it",
        )


def _prediction_summary(session: ToolSession, args: PredictionPath) -> Any:  # noqa: ANN401
    _confine_result(session, Path(args.path), "prediction")
    if session.access is not None and not session.access.run_dir(
        Path(args.path), session.service.store.entries()
    ):
        raise ToolFailure(f"no prediction at {args.path}", "see list_runs")
    return session.service.prediction_summary(Path(args.path)).model_dump(mode="json")


def _list_runs(session: ToolSession, args: ListRuns) -> Any:  # noqa: ANN401
    rows = [
        {"run_id": e.run_id[:12], "name": e.name, "created": e.created, "path": str(e.path)}
        # The catalogue is oldest first; its order breaks ties of the one-second ``created``.
        for _, e in sorted(
            enumerate(_entries(session)), key=lambda p: (p[1].created, p[0]), reverse=True
        )
        if all(e.annotations.get(k) == v for k, v in args.annotations.items())
    ]
    # Newest first, and the page cut here to what fits a result: bound() would halve it
    # from the end, and a model then could not tell whether a newer run was cut (a
    # reported issue, the latest run guessed from 20 of 39). next_offset continues it.
    page = rows[args.offset : args.offset + args.limit]
    result: dict[str, Any] = {"runs": page, "total": len(rows), "order": "newest first"}
    while len(page) > 1 and len(json.dumps(result, default=str)) > RESULT_LIMIT - 100:
        page = page[: len(page) // 2]
        result["runs"] = page
    if args.offset + len(page) < len(rows):
        result["next_offset"] = args.offset + len(page)
    return result


def _annotations(session: ToolSession, given: Mapping[str, str]) -> dict[str, str]:
    notes = {**given, "via": "tools", "client": session.client}
    if session.model:
        notes["model"] = session.model
    return notes


# ---------------------------------------------------------------------- the registry

#: What each tool does, in a line with an example: short, for small models.
DESCRIPTIONS = {
    "list_plugins": "List methods, localizations, simulators and other plugins. "
    'Example: list_plugins(registry="localizations").',
    "get_schema": "Get the JSON Schema of a configuration, or one section of it. Examples: "
    'get_schema(kind="study"), get_schema(kind="history_match", section="prior").',
    "validate": "Check a run, study, prediction or optimization file without running it; "
    "returns its cost "
    'and an approval token. Example: validate(config_file="/cases/m5/configs/run.toml", '
    'overrides={"method.max_iterations": 10}).',
    "submit_run": "Run a history-matching configuration in the background. Needs the token "
    "from validate for the same file and overrides.",
    "submit_study": "Run a study (cases x methods x variants x seeds) in the background. Needs the "
    "token from validate.",
    "submit_prediction": "Forecast a finished run beyond history. Needs the token from validate.",
    "job_status": 'Get a job\'s state. Example: job_status(job_id="20260928T160000-ab12cd34").',
    "job_events": "Get a job's events from number since (use next_since to continue).",
    "stop_job": "Stop a job: queued jobs are cancelled, running runs stop at their next "
    "checkpoint.",
    "resume_job": "Continue a stopped job. Needs the token from validate(resume_job=...).",
    "run_summary": "Get a run's results: iterations, data mismatch mean and std, metrics.",
    "compare_study": "Compare a study's methods and variants at each run's posterior.",
    "prediction_summary": "Get a finished prediction's statistics and scores.",
    "list_runs": "List runs, newest first (the first is the latest), optionally by "
    "annotation; next_offset gives the next page. "
    'Example: list_runs(annotations={"study": "loc"}).',
    "submit_optimization": "Run an optimization (e.g. maximise NPV) in the background. Needs "
    "the token from validate for the same file.",
    "optimization_summary": "Get an optimization's state, objective by iteration and result. "
    'Example: optimization_summary(path="/service/optimizations/3well/ab12cd34ef56").',
}

_SPECS: list[tuple[str, type[_Arguments], bool, Handler]] = [
    ("list_plugins", ListPlugins, False, _list_plugins),
    ("get_schema", GetSchema, False, _get_schema),
    ("validate", Validate, False, _validate),
    ("submit_run", SubmitRun, True, _submit_run),
    ("submit_study", SubmitStudy, True, _submit_study),
    ("submit_prediction", SubmitPrediction, True, _submit_prediction),
    ("job_status", JobId, False, _job_status),
    ("job_events", JobEvents, False, _job_events),
    ("stop_job", JobId, False, _stop_job),
    ("resume_job", ResumeJob, True, _resume_job),
    ("run_summary", RunId, False, _run_summary),
    ("compare_study", CompareStudy, False, _compare_study),
    ("prediction_summary", PredictionPath, False, _prediction_summary),
    ("list_runs", ListRuns, False, _list_runs),
    ("submit_optimization", SubmitOptimization, True, _submit_optimization),
    ("optimization_summary", OptimizationPath, False, _optimization_summary),
]

#: Every tool, by name.
TOOLS: dict[str, Tool] = {
    name: Tool(name, DESCRIPTIONS[name], arguments, spends, handler)
    for name, arguments, spends, handler in _SPECS
}


# ---------------------------------------------------------------------- dispatch


def bound(result: Any, limit: int = RESULT_LIMIT) -> Any:  # noqa: ANN401
    """Shorten a result to ``limit`` characters of JSON, marking what was cut."""
    text = json.dumps(result, default=str)
    if len(text) <= limit:
        return result
    if isinstance(result, dict):
        shortened = dict(result)
        for key, value in sorted(
            result.items(), key=lambda kv: -len(json.dumps(kv[1], default=str))
        ):
            if isinstance(value, list) and len(value) > 1:
                shortened[key] = value[: max(1, len(value) // 2)]
                shortened["truncated"] = True
                return bound(shortened, limit)
            if isinstance(value, dict) and len(value) > 1:
                items = list(value.items())
                shortened[key] = dict(items[: max(1, len(items) // 2)])
                shortened["truncated"] = True
                return bound(shortened, limit)
    return {"truncated": True, "preview": text[: limit - 100]}


def call(name: str, arguments: Mapping[str, Any] | None, session: ToolSession) -> dict[str, Any]:
    """Run one tool call; always return a result (``ok`` false with a hint on errors)."""
    tool = TOOLS.get(name)
    result: dict[str, Any]
    if tool is None:
        close = difflib.get_close_matches(name, TOOLS, n=3)
        hint = f"did you mean {', '.join(close)}?" if close else f"tools: {', '.join(TOOLS)}"
        result = {"ok": False, "error": f"no tool {name!r}", "hint": hint}
    else:
        try:
            parsed = tool.arguments.model_validate(dict(arguments or {}))
            result = {"ok": True, "result": bound(tool.handler(session, parsed))}
        except ValidationError as error:
            problems = [
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in error.errors()
            ]
            allowed = ", ".join(tool.arguments.model_fields)
            result = {
                "ok": False,
                "error": "; ".join(problems[:5]),
                "hint": f"arguments: {allowed}",
            }
        except ToolFailure as failure:
            result = {"ok": False, "error": failure.error, "hint": failure.hint}
        except ApexError as error:
            result = {"ok": False, "error": str(error), "hint": ""}
    session.service.jobs.store.add_tool_call(
        {
            "time": now(),
            "client": session.client,
            "model": session.model,
            "host": session.host,
            "tool": name,
            "arguments": dict(arguments or {}),
            "ok": result["ok"],
            "outcome": json.dumps(result, default=str)[:500],
        }
    )
    return result


def export(format: Literal["openai", "anthropic", "json-schema"]) -> list[dict[str, Any]]:
    """Return the tools as function-calling definitions of an API format."""
    if format == "openai":
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema(),
                },
            }
            for t in TOOLS.values()
        ]
    if format == "anthropic":
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema()}
            for t in TOOLS.values()
        ]
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema(),
            "spends_compute": t.spends_compute,
        }
        for t in TOOLS.values()
    ]
