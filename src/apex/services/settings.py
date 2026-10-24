"""Settings of a services root: ``<root>/service.toml``.

Every key is optional; the defaults are below. Example::

    [service]
    max_concurrent_jobs = 2       # jobs running at once (the pools share out their runs)
    poll_s = 1.0                  # how often the service looks at the queue
    max_retries = 2               # automatic resumes of a job whose worker died

    [jobs]                        # which code jobs run
    cli_code = "frozen"           # apex submit / apex study run: pin to HEAD; or "live"
    code_dir = "/cases/.apex-code"    # frozen copies (default <root>/code)
    code_extras = ["reservoir", "plots"]   # default: the extras installed here
    keep_code_days = 2

    [approval]
    # Ask a person before a submission that may take more than this many simulator runs...
    max_simulator_runs = 20000
    # ...or that starts more than this many runs (a study with many variants).
    max_runs = 20

    [notifications]               # optional, when a job ends
    command = "notify-send 'APEX {kind} {job}' {state}"
    webhook = "https://example.org/hook"

    [quotas.planner-x]            # per client (the submission's client field)
    max_concurrent_jobs = 1
    max_simulator_runs_per_day = 50000

    [policy]                      # what submissions may read, execute and write
    input_roots = ["/cases", "/projects/study/experiments"]
    code_roots = ["/cases/*/model", "/opt/apex/examples"]
    output_roots = ["/cases/*/runs/planner"]
    executables = ["flow", "eclrun"]

    [workspace]                   # where the UI's editor may write .toml files
    roots = ["/cases"]            # default <root>/configs; required under a [policy]

    [ui]                          # the built web UI
    dist_dir = "/opt/apex/ui/dist"    # default $APEX_UI_DIST, else ui/dist of the source tree

    [agents.diagnosis]            # the diagnosis agent's thresholds
    mahalanobis_limit = 10.83

    [agents.planner]              # the planning agent's bounds
    max_revisions = 2

    [chat]                        # the chat assistant of the web UI
    max_history_mb = 200              # size limit of <root>/chats.db
    retention_days = 92               # delete chats inactive for more than 3 months
    default_model = "claude"          # the owner's model; guests: ui_access.toml [chat] models

    [chat.models.claude]          # models of the web UI's model chooser
    label = "Claude"
    provider = "claude_code"
    model = "sonnet"

    [guests]                      # the web UI's guests' work
    max_storage_mb = 500              # per guest; their oldest finished work goes first
    retention_days = 30               # deleted this long after submission

    [agents.llm]                  # the model of APEX's runtime agents
    provider = "openai_compatible"    # or "anthropic", "ollama", "stub"
    model = "qwen3:32b"               # or "glm-4.6", "kimi-k2", "claude-opus-5-5", ...
    base_url = "http://localhost:11434/v1"
    api_key_env = "OLLAMA_API_KEY"    # the variable holding the key, if the server needs one
    temperature = 0.0
    timeout_s = 120
"""

import logging
import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from apex.core.agents import AgentContribution, contributed_agents
from apex.core.errors import ConfigError
from apex.core.llm import LLMSection as LLMSection  # moved to core
from apex.domain.config import _Strict

logger = logging.getLogger(__name__)

_Section = TypeVar("_Section", bound=BaseModel)


class ServiceSection(_Strict):
    """``[service]``."""

    max_concurrent_jobs: int = Field(default=2, ge=1)
    poll_s: float = Field(default=1.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # Run the routines installed packages add to the service; off for a
    # service that must do nothing on its own (a demonstration).
    routines: bool = True


class JobsSection(_Strict):
    """``[jobs]``: which code jobs run."""

    cli_code: Literal["live", "frozen"] = Field(
        default="frozen",
        description="Code of jobs submitted from the command line (apex submit, apex study "
        "run): 'frozen' pins each to the repository's HEAD at submission, 'live' runs the "
        "checkout as it is when the worker starts ($APEX_CLI_CODE overrides it). Other "
        "clients run live code unless they ask.",
    )
    repository: Path | None = Field(
        default=None,
        description="The git repository frozen copies are made from; default: the checkout "
        "APEX runs from.",
    )
    code_dir: Path | None = Field(
        default=None, description="Where frozen copies live; default <root>/code."
    )
    code_extras: list[str] | None = Field(
        default=None,
        description="Extras synced into each frozen copy's environment; default: those "
        "installed in the environment APEX runs in (torch never).",
    )
    keep_code_days: float = Field(
        default=2.0,
        ge=0,
        description="apex gc --code (and the service, daily) removes frozen copies that no "
        "job or process uses and that were last pinned longer ago than this; never the "
        "newest copy of master.",
    )


class ApprovalSection(_Strict):
    """``[approval]``: when a person must approve a submission."""

    # About a day of a 100-member OPM case at 18 runs at once.
    max_simulator_runs: int = Field(default=20_000, ge=1)
    # Many cheap runs (e.g. a study with many variants) deserve a look too.
    # Raised from 8 to 20 (the user, 2026-10-11): a two-method study with 5 replicates (10
    # runs) needed approval.
    max_runs: int = Field(default=20, ge=1)


class NotificationSection(_Strict):
    """``[notifications]``: optional calls when a job ends."""

    command: str | None = None
    webhook: str | None = None


class QuotaSection(_Strict):
    """``[quotas.<client>]``."""

    max_concurrent_jobs: int | None = Field(default=None, ge=1)
    max_simulator_runs_per_day: int | None = Field(default=None, ge=1)


class MonitorSection(_Strict):
    """``[agents.monitor]``: the run monitor's thresholds."""

    # The job service checks every running history-matching job after each scored
    # forecast (advisory only: it never stops a job), and notifies on these statuses.
    auto: bool = False
    notify_on: tuple[Literal["warning", "critical"], ...] = ("warning", "critical")

    # A variable's spread below 5% of the prior's: the ensemble no longer spans its
    # uncertainty (the runtime-agent rules' own example).
    collapse_spread_ratio: float = Field(default=0.05, gt=0, lt=1)
    # More than a fifth of the members failed: the posterior may be biased toward the
    # members that simulate.
    max_failed_fraction: float = Field(default=0.2, gt=0, lt=1)
    # Rejected trials in a row: the damping has grown by factor**n (2**5 = 32 by default).
    max_consecutive_rejections: int = Field(default=5, ge=1)
    # The accepted data mismatch changed by less than 1% over 3 accepted iterates.
    stagnation_tolerance: float = Field(default=0.01, gt=0)
    stagnation_window: int = Field(default=3, ge=1)
    # Mean-square normalized data mismatch (about 1 at the noise level) below 0.5: noise is fitted.
    overfit_mismatch: float = Field(default=0.5, gt=0)
    # Still four orders of magnitude above the noise level at the end: the model, the prior
    # or the data errors are likely wrong (the user's setting, 2026-09-29; data mismatches of real
    # cases often stay well above 1 without anything being wrong).
    poor_fit_mismatch: float = Field(default=10_000.0, gt=1)
    # Accepted iterates without a CV improvement (the default patience).
    cv_patience: int = Field(default=2, ge=1)
    # A finished run whose data mismatch fell by less than 10% from the prior's: it stopped
    # (e.g. "converged") without leaving the prior (a heavily damped first step, a release
    # tester's finding 34).
    min_mismatch_reduction: float = Field(default=0.1, gt=0, lt=1)
    # Optimizations: the objective improved by less than 0.1% (relative to its
    # size) over the last 3 accepted steps, so the step size or the perturbations are
    # likely too small to make progress.
    optimization_stall_tolerance: float = Field(default=1e-3, gt=0)
    optimization_stall_window: int = Field(default=3, ge=1)
    # Multi-objective runs: no feasible member after this many generations
    # (the constraints may be too tight for the variation to reach them) ...
    optimization_infeasible_generations: int = Field(default=10, ge=1)
    # ... and a front with fewer members than this share of the population after
    # optimization_min_front_generations generations (the population lost its diversity).
    optimization_min_front_share: float = Field(default=0.1, gt=0, le=1)
    optimization_min_front_generations: int = Field(default=10, ge=1)
    # A hypervolume more than 5% below its best under one penalty factor: the front lost
    # ground (crowding truncation causes small dips, which are normal).
    optimization_hypervolume_drop: float = Field(default=0.05, gt=0, lt=1)


class DiagnosisSection(_Strict):
    """``[agents.diagnosis]``: the diagnosis agent's thresholds."""

    # More than a fifth of a data type's observations outside the forecast range: the
    # ensemble cannot reach them (a biased prior, too small data errors, or the model).
    coverage_outside_fraction: float = Field(default=0.2, gt=0, lt=1)
    # And a binomial test, so that a few data outside by chance are not a cause (U1).
    coverage_significance: float = Field(default=0.01, gt=0, lt=1)
    # The 99.9th percentile of chi-squared with one degree of freedom: a datum (or a data
    # type's leading component) this far from its perturbed forecast is unlikely when the
    # model and the data are consistent (Oliver 2020).
    mahalanobis_limit: float = Field(default=10.83, gt=0)
    # More than 10% of a group's parameters moved beyond 3 prior standard deviations: the
    # update over-corrects, typically through spurious correlations of a small ensemble.
    overcorrection_percent: float = Field(default=10.0, gt=0, le=100)
    # The same failure reason for at least 3 members: a systematic, not a random, failure.
    recurring_failure_members: int = Field(default=3, ge=2)


class PlotterSection(_Strict):
    """``[agents.plotter]``: the plotter agent's bounds."""

    # Figure requests one plot may draw at most (a preset counts as its requests): the
    # model's plan is refused beyond it.
    max_figures: int = Field(default=40, ge=1)
    # The figures' resolution relative to their plugin's own (1: the reference's, e.g. 500
    # dpi box plots); lower draws faster and smaller files.
    dpi_scale: float = Field(default=1.0, gt=0, le=2)
    # Background drawings (REST, web UI, assistant) running at once on the machine, over
    # every run and user: each loads iterates and renders large figures, beside the
    # simulations. A drawing beyond it is refused, not queued.
    max_drawings: int = Field(default=2, ge=1, le=16)


class GeomodelerSection(_Strict):
    """``[agents.geomodeler]``: the geomodeler's bounds and defaults."""

    # Drafts sent back to the model with their problems (validation, QC) before the rules.
    max_revisions: int = Field(default=3, ge=0)
    # The rules' correlation range, as a fraction of the grid's larger side (cells): no
    # data says otherwise, and a quarter keeps several correlated patches in the field.
    default_range_fraction: float = Field(default=0.25, gt=0, le=2)
    # Members the QC draws, and its tolerance: the drawn mean's error in stds and the
    # std's relative error, and a facies' proportion's absolute error (a correlated sample
    # has few independent values, so the tolerance is loose; it catches clipping and
    # misread units, not sampling noise).
    qc_members: int = Field(default=20, ge=2, le=200)
    # The QC draws at most this many values (cells x members); a larger prior, or one
    # drawn by SGSIM (pure Python, per layer and member), is not drawn, and says so.
    qc_max_values: int = Field(default=5_000_000, ge=1)
    qc_tolerance: float = Field(default=0.5, gt=0)
    qc_facies_tolerance: float = Field(default=0.15, gt=0, le=1)
    # The ensemble size a configuration without one gets (APEX's default case study).
    ensemble_size: int = Field(default=100, ge=2)
    # The largest prior a web-UI user may have modelled (cells over every variable): its QC
    # draw runs in the service's request, which this bounds.
    ui_max_cells: int = Field(default=200_000, ge=1)


class CaseConverterSection(_Strict):
    """``[agents.case_converter]``: the case-converter's bounds and defaults."""

    # Drafts sent back to the model with their validation errors before the rules convert.
    max_revisions: int = Field(default=3, ge=0)
    # The skill a deck goes to when several read it (.DATA: opm_flow and eclipse); OPM Flow
    # unless the user needs Eclipse, which uses licences.
    default_skill: str = "opm_flow"
    # The rules' data errors: a standard deviation of relative_error x |value|, at least
    # floor_fraction of the vector's largest value (zero rates still get an error), the
    # examples' choice (3dBox: 10 %).
    relative_error: float = Field(default=0.1, gt=0, le=1)
    floor_fraction: float = Field(default=0.01, gt=0, le=1)
    # APEX's default case study (GIES, rlmmac, adaptive_prior): its members and iterations.
    ensemble_size: int = Field(default=100, ge=10)
    iterations: int = Field(default=10, ge=1)
    # Observation candidates the model is shown at most (each a vector of a well).
    max_candidates: int = Field(default=400, ge=1)
    # The largest case a web-UI user may convert or create (cells): the work (reading the
    # deck, drawing the truth) runs in the service's request, which this bounds.
    ui_max_cells: int = Field(default=200_000, ge=1)


class PlannerSection(_Strict):
    """``[agents.planner]``: the planning agent's bounds."""

    # Proposals sent back to the model with their validation errors before the rules plan.
    max_revisions: int = Field(default=2, ge=0)
    # Replicates the rules' plan uses at most, however large the budget.
    max_replicates: int = Field(default=5, ge=1)
    # Variants a proposal may have (each costs a run per case and replicate).
    max_variants: int = Field(default=6, ge=1)


class PolicySection(_Strict):
    """``[policy]``: roots a submission may read, execute and write.

    Off by default (all lists empty); once one list is set, all three must be. Roots are
    absolute paths, optionally with glob components (``/cases/*/runs/planner``).
    """

    input_roots: list[str] = Field(default_factory=list)
    code_roots: list[str] = Field(default_factory=list)
    output_roots: list[str] = Field(default_factory=list)
    executables: list[str] = Field(
        default_factory=lambda: ["flow", "eclrun"],
        description="Program names (found on the service's PATH) allowed under code_roots.",
    )
    hosts: list[str] = Field(
        default_factory=list,
        description="Other machines that members may run on; none by default.",
    )

    @model_validator(mode="after")
    def _complete(self) -> "PolicySection":
        names = ("input_roots", "code_roots", "output_roots")
        for name in names:
            for root in getattr(self, name):
                if not Path(root).is_absolute():
                    raise ValueError(f"[policy] {name}: {root!r} must be an absolute path")
                if ".." in Path(root).parts:
                    raise ValueError(f"[policy] {name}: {root!r} must not contain '..'")
        # A partial policy would leave something open (e.g. code execution without
        # code_roots): once one list is set, all three must be.
        given = [name for name in names if getattr(self, name)]
        if given and len(given) < len(names):
            missing = ", ".join(name for name in names if name not in given)
            raise ValueError(f"[policy] sets {', '.join(given)} but not {missing}: set all three")
        return self


class WorkspaceSection(_Strict):
    """``[workspace]``: folders whose ``.toml`` files the UI's editor may write.

    Empty means ``<root>/configs``, except under a ``[policy]``, where the editor is off
    until roots are set (each under the policy's ``input_roots``).
    """

    roots: list[str] = Field(default_factory=list)
    ui_folder: str | None = Field(
        default="UI",
        description="Folder of the first root where the web UI's submissions write their "
        'runs, studies and drafts: UI/owner/ and UI/guests/<name>/; "" turns '
        "it off (their configurations say where).",
    )

    @field_validator("ui_folder")
    @classmethod
    def _off(cls, value: str | None) -> str | None:
        return value or None  # TOML has no null: "" turns the folder off

    @model_validator(mode="after")
    def _absolute(self) -> "WorkspaceSection":
        folder = self.ui_folder
        if folder is not None and (Path(folder).name != folder or folder in (".", "..")):
            raise ValueError(f"[workspace] ui_folder: {folder!r} must be one folder name")
        for root in self.roots:
            if not Path(root).is_absolute() or ".." in Path(root).parts:
                raise ValueError(f"[workspace] roots: {root!r} must be absolute, without '..'")
            if any(c in root for c in "*?["):
                raise ValueError(f"[workspace] roots: {root!r} must be a plain folder, no globs")
        return self


class UISection(_Strict):
    """``[ui]``: the built web UI the service serves at ``/ui/``."""

    dist_dir: str | None = None
    # Web-UI tokens (the guests' and the owner's web-UI token) expire this long after they
    # were issued: a sign-in to the web UI does not last for ever. The owner's
    # API token (CLI, MCP, scripts) does not expire.
    token_days: float = Field(default=7.0, gt=0)
    # List the web-UI pages installed packages add; off for a service
    # whose pages must show only core's (a demonstration).
    contributed_pages: bool = True


class ChatModelSection(LLMSection):
    """``[chat.models.<name>]``: a model the web UI's assistant may answer with.

    The keys of ``[agents.llm]``, plus the name the web UI shows.
    """

    label: str | None = Field(default=None, max_length=60)


#: A chat model's name: what ``ui_access.toml`` and the web UI use.
_MODEL_NAME = r"^[a-z][a-z0-9_-]{0,31}$"


class ChatSection(_Strict):
    """``[chat]``: the chat assistant of the web UI."""

    # The size limit of <root>/chats.db: beyond it the chats of oldest activity are
    # deleted until the file is below 90 % of it.
    max_history_mb: float = Field(default=200.0, gt=0)
    # Chats inactive for more than this are deleted (3 months).
    retention_days: float = Field(default=92.0, gt=0)
    # A chat is closed after this many messages (both sides), so no single chat grows
    # without bound.
    max_turns_per_chat: int = Field(default=500, ge=2)
    # Earlier messages the model sees with each new one (older ones stay in the history).
    context_turns: int = Field(default=20, ge=0)
    # Model calls per answer, tool rounds included; each failed tool call (a refused plan,
    # invalid arguments) adds one, up to refusal_turns, so that looking things up and one
    # refusal cannot use up an answer before the model revises (the 2026-10-04 chat).
    max_model_calls: int = Field(default=10, ge=2)
    refusal_turns: int = Field(default=4, ge=0)
    # A tool that runs longer than this is abandoned and reported to the model as timed out:
    # a safety net beside the Stop button (Brugge's deck once took
    # minutes to read). What it gives is dropped; a tool in a worker process (isolate_tools)
    # is killed, another runs on in the background. 0: no limit.
    tool_timeout_s: float = Field(default=120.0, ge=0)
    # Run the slow read-only tools (case_info, monitor_run, diagnose_run, write_report) in a
    # worker process each, killed when the tool is abandoned (a stop or tool_timeout_s), so
    # that it stops using CPU and memory. false: in a thread, which
    # runs on until it returns.
    isolate_tools: bool = True
    # The model; default [agents.llm].
    llm: LLMSection | None = None
    # Models to choose: the owner picks any in the web UI, a guest
    # those ui_access.toml's [chat] models lists, in that order (the next when one fails);
    # other kinds of web-UI user, and the agents' buttons of the owner, default_model.
    # Without models, everyone's chat uses llm (else [agents.llm]), as before.
    models: dict[str, ChatModelSection] = Field(default_factory=dict)
    default_model: str | None = None
    # Show the answer as it is written, with providers that stream (anthropic,
    # openai_compatible). Others answer at once, after their steps.
    stream: bool = True

    @model_validator(mode="after")
    def _models(self) -> "ChatSection":
        bad = [n for n in self.models if not re.fullmatch(_MODEL_NAME, n)]
        if bad:
            raise ValueError(f"[chat.models]: {bad} must be lower-case names (a-z, 0-9, _ -)")
        if self.default_model is not None and self.default_model not in self.models:
            raise ValueError(f"[chat] default_model {self.default_model!r} is not one of "
                             f"[chat.models]: {sorted(self.models)}")  # fmt: skip
        if self.models and self.llm is not None:
            raise ValueError("[chat] llm and [chat.models] exclude each other: name the "
                             "model in [chat.models] instead")  # fmt: skip
        return self

    @property
    def default(self) -> str | None:
        """The owner's model: ``default_model``, else the first of ``models`` (none: ``llm``)."""
        return self.default_model or next(iter(self.models), None)


class GuestsSection(_Strict):
    """``[guests]``: how long, and how much, the web UI's guests' work is kept.

    Applies only to work a guest submitted (``submitted_by = guest:<name>``); the
    owner's, and other local programs' (the CLI, the Python client), is never pruned.
    """

    # Each guest's jobs, runs, studies and predictions take at most this much disk: beyond
    # it, their oldest finished work is deleted first.
    max_storage_mb: float = Field(default=500.0, gt=0)
    # A guest's finished work is deleted this long after it was submitted.
    retention_days: float = Field(default=30.0, gt=0)


class _CoreAgentsSection(_Strict):
    """``[agents]``: core's agents (:data:`AgentsSection` adds the contributed ones)."""

    llm: LLMSection = Field(default_factory=LLMSection)
    monitor: MonitorSection = Field(default_factory=MonitorSection)
    diagnosis: DiagnosisSection = Field(default_factory=DiagnosisSection)
    planner: PlannerSection = Field(default_factory=PlannerSection)
    plotter: PlotterSection = Field(default_factory=PlotterSection)
    case_converter: CaseConverterSection = Field(default_factory=CaseConverterSection)
    geomodeler: GeomodelerSection = Field(default_factory=GeomodelerSection)
    # Which agents the web UI and the assistant show: each agent's default, overridden here
    # by name, e.g. { plotter = "private" } (an agent of a package that is not public stays
    # private).
    visibility: dict[str, Literal["public", "private"]] = Field(default_factory=dict)

    def contributed(self, name: str, model: type[_Section]) -> _Section:
        """Return the section of an agent a workspace member contributes (``core.agents``).

        Raises
        ------
        ConfigError
            If no installed package contributes an agent of that name with that section.
        """
        value = getattr(self, name.replace("-", "_"), None)
        if not isinstance(value, model):
            raise ConfigError(
                f"[agents.{name}] is not a contributed agent's section here: is its package "
                "installed? (uv sync --inexact --all-packages)"
            )
        return value


def _agents_section(contributions: Sequence[AgentContribution]) -> type[_CoreAgentsSection]:
    """Return ``[agents]`` with a section per contributed agent that has settings.

    Strict as every section: a section of an agent no installed package contributes is
    refused (a worker's lenient reading drops it).
    """
    fields: dict[str, Any] = {}
    for contribution in contributions:
        section = contribution.section
        if contribution.settings is None:
            continue
        if section in _CoreAgentsSection.model_fields or section in fields:
            logger.warning("the contributed agent %r would replace [agents.%s]; skipped",
                           contribution.info.name, section)  # fmt: skip
            continue
        fields[section] = (contribution.settings, Field(default_factory=contribution.settings))
    # Always a subclass of its own name and doc, so the JSON Schema is the same with no
    # contributions as before the hook (the documentation inventory fingerprints it).
    return create_model(
        "AgentsSection",
        __base__=_CoreAgentsSection,
        __module__=__name__,
        __doc__="``[agents]``.",
        **fields,
    )


if TYPE_CHECKING:
    AgentsSection = _CoreAgentsSection
else:
    #: ``[agents]``: core's agents and the contributed ones.
    AgentsSection = _agents_section(contributed_agents())


class ExamplesSection(_Strict):
    """``[examples]``: the catalogue of examples and toy problems (``services/examples.py``)."""

    root: str | None = Field(
        default=None,
        description="Folder holding the examples (each with an example.toml); default: the "
        "repository's examples/ when APEX runs from a checkout. Its files are trusted as the "
        "server's: a copied example's deck templates are read from here.",
    )
    enabled: bool = Field(default=True, description="Offer the catalogue (the Examples page).")


class ToolsSection(_Strict):
    """``[tools]``: what tool clients (MCP, ``apex tools call``, agent loops) may read.

    Without a ``[policy]``, a tool client names files only under the workspace's roots
    (``[workspace] roots``, by default ``<root>/configs``), these ``roots`` and the
    service's ``runs/``, ``studies/`` and ``optimizations/``. With a ``[policy]``, its
    readable roots decide instead.
    """

    # A tool client acts as the owner: unconfined, it could read any file the owner can
    # (release tester N3, 2026-10-20).

    roots: list[str] = Field(
        default_factory=list,
        description="More folders whose files tool clients may name (absolute; no [policy]).",
    )

    @field_validator("roots")
    @classmethod
    def _absolute(cls, roots: list[str]) -> list[str]:
        for root in roots:
            if not Path(root).is_absolute() or ".." in Path(root).parts:
                raise ValueError(f"[tools] roots: {root!r} must be absolute, without '..'")
        return roots


class ServiceSettings(_Strict):
    """The settings of a services root."""

    service: ServiceSection = Field(default_factory=ServiceSection)
    jobs: JobsSection = Field(default_factory=JobsSection)
    approval: ApprovalSection = Field(default_factory=ApprovalSection)
    notifications: NotificationSection = Field(default_factory=NotificationSection)
    quotas: dict[str, QuotaSection] = Field(default_factory=dict)
    agents: AgentsSection = Field(default_factory=AgentsSection)
    policy: PolicySection = Field(default_factory=PolicySection)
    workspace: WorkspaceSection = Field(default_factory=WorkspaceSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    chat: ChatSection = Field(default_factory=ChatSection)
    guests: GuestsSection = Field(default_factory=GuestsSection)
    ui: UISection = Field(default_factory=UISection)
    examples: ExamplesSection = Field(default_factory=ExamplesSection)

    @model_validator(mode="after")
    def _workspace_readable(self) -> "ServiceSettings":
        # Under a policy, what the editor saves must be readable by submissions, and the
        # editor must not make code writable.
        from apex.services.policy import under

        rules = self.policy
        if not (rules.input_roots or rules.code_roots or rules.output_roots):
            return self
        for root in self.workspace.roots:
            path = Path(root).resolve()
            if not any(under(path, allowed) for allowed in rules.input_roots):
                raise ValueError(f"[workspace] root {root} is outside [policy] input_roots")
            if any(under(path, code) for code in rules.code_roots):
                raise ValueError(f"[workspace] root {root} lies under [policy] code_roots")
        return self


def load_settings(root: Path, *, lenient: bool = False) -> ServiceSettings:
    """Read ``<root>/service.toml`` (defaults when it is missing).

    ``lenient`` drops keys this code does not know, instead of refusing them: a job's
    worker pinned to an older commit reads a ``service.toml`` that newer code
    may have extended. Everything else must still match the schema.

    Raises
    ------
    ConfigError
        If it cannot be read or does not match the schema.
    """
    path = Path(root) / "service.toml"
    if not path.exists():
        return ServiceSettings()
    try:
        data = tomllib.loads(path.read_text())
        for _ in range(100):
            try:
                return ServiceSettings.model_validate(data)
            except ValidationError as error:
                unknown = [e["loc"] for e in error.errors() if e["type"] == "extra_forbidden"]
                # A whole section under [agents] (not a key of a known section's).
                agents = [str(loc[1]) for loc in unknown if len(loc) == 2 and loc[0] == "agents"]
                if not lenient and agents:
                    # Perhaps an agent's section whose package is not installed.
                    sections = ", ".join(f"[agents.{name}]" for name in agents)
                    raise ConfigError(
                        f"{path} has {sections}, which no installed agent has: a misspelt "
                        "name, or the settings of agents another APEX installation has. "
                        "Remove or correct those sections, or move the file away (or set "
                        "APEX_HOME to another folder); in a development checkout, install the "
                        "packages that provide them (uv sync --inexact --all-packages).\n"
                        f"Details: {error}"
                    ) from error
                if not lenient or not unknown:
                    raise
                for loc in unknown:
                    _drop(data, loc)
        raise ConfigError(f"invalid {path}: too many unknown keys")
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise ConfigError(f"invalid {path}: {error}") from error


def _drop(data: dict[str, Any], loc: tuple[int | str, ...]) -> None:
    """Remove the key at ``loc`` from nested tables (a missing one is fine)."""
    node: Any = data
    for part in loc[:-1]:
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            return
    if isinstance(node, dict):
        node.pop(loc[-1], None)
