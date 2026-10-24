"""Models of the application services: what clients send and receive.

Every model is a strict pydantic model, serialisable to JSON, with a ``schema_version``,
so the Python client, the CLI and (later) the REST API and MCP server exchange the same
documents.
"""

from typing import Any, Literal

from pydantic import Field

from apex.domain.config import _Strict

#: Version of the service documents; bumped on a breaking change (with an ADR).
SCHEMA_VERSION = 1

#: Free-form labels a client attaches to its work (hypothesis, rationale, model id).
Annotations = dict[str, str]

#: How a job ends, or where it is.
JobState = Literal["queued", "running", "complete", "stopped", "failed", "cancelled"]


class _Document(_Strict):
    schema_version: int = SCHEMA_VERSION


class PluginInfo(_Document):
    """One registered plugin."""

    registry: str
    name: str
    workflow: (
        Literal["history_matching", "prediction", "optimization", "simulation", "reporting"] | None
    ) = Field(
        default=None,
        description="The workflow its registry serves (simulation: shared by all; "
        "reporting: figures); None for any other.",
    )
    status: str | None = Field(
        default=None, description="'parity', 'validated' or 'experimental', where declared."
    )
    summary: str = Field(default="", description="The first line of its docstring.")
    options: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema of the options a configuration may set."
    )
    supplied: list[str] = Field(
        default_factory=list,
        description="Parameters APEX supplies itself (e.g. state bounds), not configurable.",
    )
    visibility: Literal["public", "private"] = Field(
        default="public",
        description="'private': left out of the public documentation (a plugin built for "
        "one site, e.g. the ssh executor).",
    )
    unavailable: str | None = Field(
        default=None,
        description="Why it cannot be used here, e.g. a dependency of an extra that is not "
        "installed; None when it can.",
    )


class RunPlan(_Document):
    """What a history-matching run will cost, before it runs."""

    run_id: str
    method: str
    engine: str = Field(default="apex", description="Where it runs.")
    members: int | None = Field(description="None when a stored prior's size is unknown.")
    n_state: int
    data: dict[str, int] = Field(description="Data per role: history_matching, cv, test.")
    max_forecasts: int | None = Field(
        description="Ensemble forecasts the budget allows (None if the method cannot say)."
    )
    max_simulator_runs: int | None
    simulator: str
    licences: int | None = None
    parallel_runs: int | None = None
    estimated_hours: float | None = Field(
        default=None, description="From earlier runs of the same case, if any."
    )
    checkpoint_mb: float | None = Field(
        default=None,
        description="About this much per checkpointed iterate (MB): the ensemble, its "
        "forecast (transformed) and auxiliary outputs, a fixed schedule's perturbed "
        "observations, and raw forecasts if they are kept.",
    )
    forecast_memory_mb: float | None = Field(
        default=None,
        description="About this much memory for one ensemble forecast (MB): the raw "
        "forecasts twice while they are stacked and, with data transforms, the transformed "
        "copy.",
    )
    warnings: list[str] = Field(default_factory=list)


class OptimizationPlan(_Document):
    """What an optimization will cost, before it runs."""

    optimization_id: str = Field(description="Hash of the configuration without its place.")
    name: str
    optimizer: str
    objective: str
    n_controls: int
    n_realizations: int
    max_evaluations: int | None = Field(
        description="Upper bound on the objective's evaluations (None: the optimizer cannot say)."
    )
    max_simulator_runs: int | None = Field(
        description="One per evaluation for a simulated objective; 0 for an analytic one."
    )
    simulator: str | None = None
    parallel_runs: int | None = None
    output_dir: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Validation(_Document):
    """The outcome of validating a configuration (nothing is simulated)."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    plan: RunPlan | None = None
    optimization: OptimizationPlan | None = Field(
        default=None, description="The plan of an optimization configuration."
    )
    plans: list[RunPlan] = Field(default_factory=list, description="Per run, for a study.")
    notes: list[str] = Field(
        default_factory=list,
        description="What a study will not run, and why (e.g. a localization "
        "variant of an ERT method).",
    )


class JobHandle(_Document):
    """A submitted job."""

    id: str
    kind: Literal["run", "resume", "prediction", "study", "optimization", "optimization_resume"]
    run_ids: list[str] = Field(default_factory=list)
    reused: bool = Field(default=False, description="An identical run existed: no new run.")
    job_visible: bool | None = Field(
        default=None,
        description="Whether this request may open the job's page: false when the answer is "
        "another submission's job (an identical run already queued or running) that the web "
        "UI's user may not see; null when not said (not a web-UI request).",
    )
    job_state: JobState | None = Field(
        default=None, description="The job's state when answered (with job_visible)."
    )


class JobStatus(_Document):
    """Where a job is."""

    id: str
    kind: str
    state: JobState
    created: str
    started: str | None = None
    finished: str | None = None
    error: str | None = None
    run_dirs: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list, description="The runs the job ran or used.")
    children: dict[str, JobState] = Field(
        default_factory=dict, description="A study's cells: job id -> state."
    )
    annotations: Annotations = Field(default_factory=dict)
    reused: str | None = Field(
        default=None,
        description="An identical run this job reused instead of running (its directory); "
        "null otherwise, and for a web-UI guest when the run is not theirs.",
    )
    reused_visible: bool | None = Field(
        default=None,
        description="Whether this request may open the reused run (its run page); null when "
        "nothing was reused.",
    )


class Event(_Document):
    """One event of a job, numbered from 0."""

    index: int
    time: str
    event: str | None = None
    level: str = "INFO"
    message: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)


class RunSummary(_Document):
    """A history-matching run's results, as data."""

    run_id: str | None
    path: str
    name: str
    method: str
    state: Literal["complete", "stopped", "not started"]
    converged: bool | None
    iterations: list[int]
    accepted_iterations: list[int] = Field(
        default_factory=list,
        description="The accepted iterates' records, in order: step n is the n-th (0: the prior).",
    )
    posterior_iteration: int | None
    selected_iteration: int | None = Field(description="With cross validation.")
    members: int | None
    data_mismatch: dict[int, tuple[float, float]] = Field(description="Iteration -> (mean, std).")
    metrics: dict[int, dict[str, dict[str, float]]] = Field(
        description="Iteration -> metric -> label -> member mean."
    )
    posterior_metrics: dict[str, dict[str, float]]
    simulator_runs: int
    annotations: Annotations = Field(default_factory=dict)
    provenance: dict[str, Any] | None = None


class StepInfo(_Document):
    """One step of a run's step summary (see :class:`RunSteps`).

    ``step`` numbers the steps 0, 1, 2, ... (0: the prior; rejected trials not counted);
    ``iteration`` is the history record (``iter_<k>``), which counts every trial and keys
    the metrics.
    """

    iteration: int
    step: int = 0
    n_members: int
    rejected_before: int = 0
    control: float | None = None
    label: str | None = None


class StepStatistics(_Document):
    """One metric and label at one step: statistics over the members.

    ``None`` where a value is not a statistic over members (a forecast score has a mean
    only), or is not finite.
    """

    metric: str
    label: str
    iteration: int
    mean: float | None
    std: float | None
    median: float | None
    p10: float | None
    p90: float | None


class RunSteps(_Document):
    """A run's performance metrics step by step (``apex summary``, as data)."""

    workflow: str
    name: str
    path: str
    method: str
    status: str
    steps: list[StepInfo]
    metrics: list[StepStatistics]
    selected_iteration: int | None = None
    selected_step: int | None = Field(default=None, description="The result's step number.")
    forecasts: int = 0
    notes: list[str] = Field(default_factory=list)


class PredictionSummary(_Document):
    """A prediction's results, as data."""

    path: str
    name: str
    run_dir: str
    ensembles: list[str]
    vectors: list[str]
    times: list[str]
    first_prediction_time: int
    final: dict[str, dict[str, dict[str, float]]] = Field(
        description="Ensemble -> vector -> {mean, std, p10, p50, p90} at the last time."
    )
    scores: dict[str, dict[str, dict[str, float]]]


class VariantStats(_Document):
    """One variant's metric over the replicates (``std`` is ``None`` for one replicate).

    ``method`` is the study's method entry; ``None`` when the study has none.
    """

    method: str | None = None
    engine: str = Field(default="apex", description="Where the method ran.")
    variant: str
    n: int
    mean: float
    std: float | None
    minimum: float
    maximum: float
    values: list[float]

    def label(self) -> str:
        """``method/variant``, or the variant alone without methods."""
        return group_label(self.method, self.variant)


class PairedDifference(_Document):
    """``(method, variant) - (reference_method, reference)`` over the shared replicates.

    The methods are ``None`` when the study has no method entries; otherwise one
    of the two pairs is the same (a variant's effect within a method, or a method's effect
    with the same variant).
    """

    method: str | None = None
    variant: str
    reference_method: str | None = None
    reference: str
    n: int
    mean_difference: float
    standard_error: float | None
    p_value: float | None = Field(description="Wilcoxon signed-rank, two-sided; None for n = 1.")
    note: str = ""

    def label(self) -> str:
        """``method/variant``, or the variant alone without methods."""
        return group_label(self.method, self.variant)

    def reference_label(self) -> str:
        """``reference_method/reference``, or the reference variant alone without methods."""
        return group_label(self.reference_method, self.reference)


def group_label(method: str | None, variant: str) -> str:
    """Return ``method/variant`` (a study's cell group), or ``variant`` alone."""
    return f"{method}/{variant}" if method else variant


class MetricComparison(_Document):
    """One metric of one case, compared across methods and variants."""

    case: str
    metric: str
    variants: list[VariantStats]
    differences: list[PairedDifference]


class Comparison(_Document):
    """A study's comparison of its variants, at each run's posterior."""

    study: str
    replicates: int
    reference: str = Field(description="The variant differences are taken against (the first).")
    reference_method: str | None = Field(
        default=None, description="The method differences are taken against (the first)."
    )
    metrics: list[MetricComparison]
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list, description="Cells without results yet.")
    skipped: list[dict[str, str]] = Field(
        default_factory=list,
        description="Cells the study does not run, each {label, reason} (e.g. a "
        "localization variant of an ERT method).",
    )


class OptimizationSummary(_Document):
    """An optimization run's progress and result."""

    path: str
    name: str
    state: Literal["new", "running", "stopped", "finished"]
    optimizer: str
    objective: str
    controls: list[str] = Field(description="The name of each control.")
    history: list[dict[str, Any]] = Field(
        description="One row per start, accepted step and EPF update (history.jsonl)."
    )
    result: dict[str, Any] | None = Field(description="result.json, once finished.")
    pareto: dict[str, Any] | None = Field(
        default=None,
        description="Several objectives: the front at the last checkpoint "
        "(pareto.json): x, objectives, violation, representative, hypervolume.",
    )
    evaluations: int = Field(description="Simulated evaluations in the journal (0: analytic).")
    failed_evaluations: int = 0
    simulated: bool = Field(
        default=False, description="Whether the objective is computed from simulations."
    )
    penalised: bool = Field(
        default=False,
        description="Whether the external penalty (epf) is configured: the objective values "
        "then include it, from the first outer iteration.",
    )
    objective_scaling: float | None = Field(
        default=None,
        description="The objective's obj_scaling option, if any: every optimizer minimises, "
        "so a negative scaling means the quantity (e.g. NPV) is maximised.",
    )
    multi_objective: bool = Field(
        default=False,
        description="Whether the run has several objectives: its optimizer is a "
        "multi-objective one. Known from the start, before any front.",
    )
    objective_names: list[str] | None = Field(
        default=None,
        description="Several objectives: their names, as results show them (null: one "
        "objective, or not known).",
    )
    objective_scalings: list[float | None] | None = Field(
        default=None,
        description="[[objectives]]: each one's obj_scaling option, if any (null where it has "
        "none); a negative scaling means that quantity is maximised as its negative.",
    )
