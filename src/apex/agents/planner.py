"""The planning agent: an experiment for a case and a goal.

1. :func:`plan_facts` gathers :class:`PlanFacts` from code: the case's validated plan (what
   one run costs), its method, strategy and localization, whether it is synthetic, the
   plugins available, and the budget.
2. The model proposes a :class:`StudyProposal`: optionally the algorithms to compare, each
   a complete method entry (the algorithm is the base and is never derived from
   another algorithm's settings), variants for the auxiliary techniques (JSON merge
   patches, as the study variants), replicates and metrics.
3. Code holds it to the facts (:meth:`Planner.check`): it builds the
   :class:`~apex.domain.study.StudySpec`, validates it with the service (every run is
   validated and costed; nothing is simulated) and checks its cost against the budget. An
   invalid or over-budget proposal goes back to the model with the errors, up to
   ``[agents.planner] max_revisions`` times; then :func:`rules` plan. The draft written
   (:func:`study_toml`) must reload to the study validated: TOML has no null, so a patch
   that removes a key with ``null`` is refused.

Planning spends nothing. :func:`submit` spends, and only through the tool dispatcher
(``validate`` issues an approval token, pending above the service's limits until a person
runs ``apex approve``; ``submit_study`` consumes it), after a person has confirmed.
"""

import json
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from apex.agents.common import Decision, decide, default_localization
from apex.agents.loop import LoopSettings
from apex.agents.trace import CallRecorder
from apex.algorithms.data_assimilation.registries import methods
from apex.core.advice import advise_prompt, advise_thresholds
from apex.core.errors import ApexError, ConfigError, UnknownPluginError
from apex.core.llm import LLMProvider
from apex.domain.config import ALGORITHM_FIELDS, HistoryMatchConfig
from apex.domain.services import Validation
from apex.domain.study import MethodEntry, StudySpec
from apex.services.api import ApexService
from apex.services.settings import PlannerSection
from apex.services.study import OVER_BUDGET
from apex.services.tools import ToolSession, call

if TYPE_CHECKING:
    from apex.services.workspace import Workspace

#: Registries whose plugins a proposal may name (listed in the facts).
PLUGIN_REGISTRIES = ("methods", "update_strategies", "localizations", "metrics", "simulators")
#: The simulator planned studies run decks on unless asked otherwise (the user's
#: choice: OPM Flow by default, Eclipse only on request).
DEFAULT_SIMULATOR = "opm_flow"


class PlanFacts(BaseModel):
    """What the planner knows of the case (see :func:`plan_facts`)."""

    model_config = ConfigDict(extra="forbid")

    case: str = Field(description="The case's name in the study.")
    config_file: str
    method: str
    iterative: bool
    strategy: str | None
    localization: str | None
    synthetic: bool = Field(description="A true model exists: rmse can be compared.")
    has_validation: bool
    members: int | None
    simulator: str
    runs_per_cell: int | None = Field(
        description="Simulator runs one run of the case may take (its plan's upper bound)."
    )
    budget_simulator_runs: int = Field(description="The whole study's budget.")
    max_cells: int | None = Field(
        description="Runs (methods x variants x replicates) the budget allows at this cost."
    )
    max_variants: int = Field(description="At most this many methods x variants.")
    case_method: dict[str, Any] = Field(
        description="The case's own algorithm as a method entry (its [method] without the "
        "auxiliary techniques): reuse it for this algorithm, never for another."
    )
    plugins: dict[str, list[str]] = Field(description="Registry -> plugin names.")
    configuration: dict[str, Any] = Field(description="The case's configuration.")
    engines: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Other engines that can run this case -> their algorithms, "
        'e.g. {"ert": ["es_mda", "ensemble_smoother"]}: a method entry with engine = "ert" '
        "runs ERT's own smoother on the case.",
    )


class VariantProposal(BaseModel):
    """One variant of the proposed study."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", description="e.g. baseline, adaptive_loc.")
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description='A JSON merge patch over the case, e.g. {"method": {"localization": '
        '{"name": "adaptive"}}}; {} for the baseline. Never patch run.name/seed/output_dir '
        "or method.name (algorithms are methods), and never use null (TOML has none).",
    )
    rationale: str = Field(min_length=3)


class StudyProposal(BaseModel):
    """A proposed experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    hypothesis: str = Field(min_length=3, description="What the study tests, for a person.")
    simulator: str | None = Field(
        default=None,
        description="The simulator every case's deck runs on: opm_flow (the "
        "default for deck cases), eclipse (only if the user asks), or 'case' to keep each "
        "case's own.",
    )
    methods: dict[str, MethodEntry] = Field(
        default_factory=dict,
        description="Algorithms to compare, name -> a complete method entry "
        "(name, options, strategy, schedule, max_iterations; unset ones take the method's "
        "defaults); {} runs the case's own. Copy case_method only for the case's algorithm.",
    )
    variants: list[VariantProposal] = Field(min_length=1)
    replicates: int = Field(ge=1, description="Seeds per variant (paired across variants).")
    metrics: list[str] = Field(
        min_length=1, description="'metric:label' to compare, e.g. 'rmse:all'."
    )
    reasoning: str = Field(min_length=3, description="Why this design answers the goal.")


def plan_facts(
    service: ApexService,
    config_file: Path,
    *,
    budget: int | None = None,
    thresholds: PlannerSection | None = None,
) -> PlanFacts:
    """Return what the planner knows of a case (validates it; nothing is simulated).

    ``budget`` defaults to the service's approval limit on simulator runs.

    Raises
    ------
    ConfigError
        If the case's configuration is invalid.
    """
    t = thresholds or PlannerSection()
    config_file = Path(config_file).resolve()
    validation = service.validate(config_file, check_output=False)  # the case, not a run
    if not validation.valid or validation.plan is None:
        raise ConfigError(f"{config_file} is not a valid case: " + "; ".join(validation.errors))
    config = service.resolve(config_file)
    plan = validation.plan
    total = budget if budget is not None else service.approvals.settings.max_simulator_runs
    per_cell = plan.max_simulator_runs
    plugins: dict[str, list[str]] = {}
    for info in service.plugins():
        if info.registry in PLUGIN_REGISTRIES:
            plugins.setdefault(info.registry, []).append(info.name)
    return PlanFacts(
        case=_case_name(config),
        config_file=str(config_file),
        method=config.method.name,
        iterative=bool(getattr(methods.factory(config.method.name), "iterative", False)),
        strategy=config.method.strategy.name if config.method.strategy else None,
        localization=config.method.localization.name if config.method.localization else None,
        synthetic=config.case.kind == "synthetic",
        has_validation=config.validation is not None,
        members=plan.members,
        simulator=plan.simulator,
        runs_per_cell=per_cell,
        budget_simulator_runs=total,
        max_cells=None if not per_cell else total // per_cell,
        max_variants=t.max_variants,
        plugins=plugins,
        configuration=json.loads(config.model_dump_json(exclude_none=True)),
        case_method=case_method_entry(config),
        engines=_engines(config),
    )


def _engines(config: HistoryMatchConfig) -> dict[str, list[str]]:
    """Return the other engines that can run the case, and their algorithms."""
    from apex.core.engines import EngineMethod
    from apex.services.api import engine_plugin

    try:
        engine = engine_plugin("ert")
    except ApexError:
        return {}
    if engine.check(config, EngineMethod("ert", "es_mda", {"weights": [4, 4, 4, 4]})):
        return {}
    return {"ert": ["es_mda", "ensemble_smoother"]}


def _wants_ert(goal: str) -> bool:
    """Whether a goal asks about ERT (the word, any case)."""
    import re

    return re.search(r"\bert\b", goal, re.IGNORECASE) is not None


def _wants_localization(goal: str) -> bool:
    """Whether a goal asks about localization (adaptive or by distance)."""
    import re

    return re.search(r"locali[sz]|adaptive|taper", goal, re.IGNORECASE) is not None


#: ERT's localization is in its smoother library, not in ERT's own engine (which takes
#: name and options only): APEX runs it as ES-MDA strategies, with
#: adaptive_rows its native counterpart (equal to ert_adaptive untruncated).
ERT_LOCALIZATION_STRATEGIES = ("ert_esmda", "ert_adaptive")
#: The adaptive strategies' default threshold (3/sqrt(N)) is refused below this ensemble.
ERT_LOCALIZATION_MIN_MEMBERS = 10


def ert_localization_goal(facts: PlanFacts, goal: str) -> bool:
    """Whether a goal compares localization in APEX and ERT, and the strategies can run.

    ERT's strategies are registered by entry points whether or not their library is
    installed, so the library is looked for too.
    """
    import importlib.util

    strategies = facts.plugins.get("update_strategies", [])
    return (
        _wants_ert(goal)
        and _wants_localization(goal)
        and all(s in strategies for s in (*ERT_LOCALIZATION_STRATEGIES, "adaptive_rows"))
        and importlib.util.find_spec("iterative_ensemble_smoother") is not None
    )


def ert_localization_obstacle(facts: PlanFacts) -> str | None:
    """Return why a case cannot carry the ES-MDA comparison of APEX's and ERT's localization.

    A study cannot remove a case's section (a patch has no null in TOML): a case's
    ``[validation]`` (cross validation needs an iterative method, which ES-MDA is not) or
    its own localization (adaptive_rows refuses one, and the references would be localized)
    would be in every cell.
    """
    if facts.has_validation:
        return ("the case has [validation]: cross validation needs an iterative method, and "
                "ERT's strategies run in ES-MDA, which is not")  # fmt: skip
    if facts.localization is not None:
        return (f"the case has [method.localization] ({facts.localization}): adaptive_rows "
                "refuses one, and the approx/ert_esmda references would be localized")  # fmt: skip
    damping = facts.configuration.get("method", {}).get("damping")
    if damping:
        return (f"the case has [method.damping] ({damping.get('name')}): ES-MDA adapts no "
                "damping")  # fmt: skip
    if facts.members is not None and facts.members < ERT_LOCALIZATION_MIN_MEMBERS:
        return (f"the case has {facts.members} members: the adaptive strategies need at least "
                f"{ERT_LOCALIZATION_MIN_MEMBERS}")  # fmt: skip
    return None


def study_simulator(requested: str | None, case_simulator: str) -> str | None:
    """Return a planned study's ``[study] simulator``: as asked, else OPM Flow for decks.

    ``"case"`` keeps each case's own (``None``); a case that runs no deck keeps its own.
    """
    from apex.simulation.deck import DECK_SIMULATORS

    if requested == "case":
        return None
    if requested:
        return requested
    return DEFAULT_SIMULATOR if case_simulator in DECK_SIMULATORS else None


def case_method_entry(config: HistoryMatchConfig) -> dict[str, Any]:
    """Return a case's algorithm as a study method entry: its ``[method]``'s algorithm fields."""
    section = json.loads(config.method.model_dump_json(exclude_none=True))
    return {k: v for k, v in section.items() if k in ALGORITHM_FIELDS}


def _case_name(config: HistoryMatchConfig) -> str:
    # ASCII only, and never "." or "..", as the study's names are (domain/study.py
    # _folder_name): a run named "méthode" must not make a case name the study refuses.
    name = "".join(
        c if c.isascii() and (c.isalnum() or c in "_-.") else "_" for c in config.run.name
    )
    return name if name not in ("", ".", "..") else "case"


def case_steps(facts: PlanFacts) -> int | None:
    """Return the case's own budget in updates (``None``: not known).

    An iterative case's ``max_iterations``; otherwise ES-MDA's (and multilevel ES-MDA's)
    steps, ``len(inflation)``, or ES's one update.
    """
    entry = facts.case_method
    if facts.iterative:
        steps = entry.get("max_iterations")
        return int(steps) if isinstance(steps, int) else None
    return arm_steps(facts.method, entry.get("options", {}))


def arm_steps(name: str, options: Mapping[str, Any]) -> int | None:
    """Return a non-iterative method's updates: ES-MDA's inflations, ES's one."""
    inflation = options.get("inflation")
    if isinstance(inflation, list):
        return len(inflation)
    return 1 if name == "es" else None


def _iterative(name: str) -> bool | None:
    """Whether a method plugin iterates (``None``: not known here, e.g. a misspelt name)."""
    try:
        flag = getattr(methods.factory(name), "iterative", None)
    except (UnknownPluginError, KeyError):
        return None
    return flag if isinstance(flag, bool) else None


def method_differences(facts: PlanFacts, proposal: StudyProposal) -> list[str]:
    """Return how the compared algorithms' arms differ from the case besides the algorithm.

    Each algorithm is its own base (the user's rule of 2026-10-01): an arm is
    never given another algorithm's settings. But what an arm runs differently from the
    case (its budget in updates, its strategy) confounds the comparison (a reported issue: chop
    up to 10 iterations on its own strategy against the case's ES-MDA with 4 steps and a
    localized strategy). It is returned, for the assistant to state with the proposal and
    to offer to match (revise_draft; a reported issue); the arms are left as given.
    Localization is every arm's (an auxiliary field, the case's).
    """
    steps = case_steps(facts)
    strategy = facts.case_method.get("strategy")
    unit = "iteration" if facts.iterative else "step"
    notes: list[str] = []

    def count(n: int, what: str) -> str:
        return f"{n} {what}{'' if n == 1 else 's'}"

    for label, entry in proposal.methods.items():
        if entry.external:
            # An engine's arm takes no strategy or iteration limit; its schedule can differ.
            weights = (entry.options or {}).get("weights")
            if steps is not None and isinstance(weights, list) and len(weights) != steps:
                notes.append(f"{label} runs {count(len(weights), 'step')}, the case "
                             f"{count(steps, unit)}")  # fmt: skip
            continue
        iterative = _iterative(entry.name)
        if steps is not None and iterative and entry.max_iterations not in (None, steps):
            notes.append(f"{label} runs up to {count(entry.max_iterations, 'iteration')}, the "
                         f"case {count(steps, unit)}")  # fmt: skip
        elif steps is not None and iterative and entry.max_iterations is None:
            notes.append(f"{label} runs to its own iteration limit, the case "
                         f"{count(steps, unit)}")  # fmt: skip
        elif steps is not None and iterative is False:
            own = arm_steps(entry.name, entry.options or {})
            if own != steps:
                runs = "its own schedule of steps" if own is None else count(own, "step")
                notes.append(f"{label} runs {runs}, the case {count(steps, unit)}")
        from apex.services.compatibility import default_strategy

        # A strategy left unset is resolved to its method's default, options included, as
        # the draft names it (#225, #231: the energy an unset arm ran was not the case's).
        theirs = (
            default_strategy(entry.name, entry.options)
            if entry.strategy is None
            else {"name": entry.strategy.name, "options": entry.strategy.options}
        )
        ours = strategy if isinstance(strategy, Mapping) else None
        if ours is None and facts.method is not None:
            ours = default_strategy(facts.method, facts.case_method.get("options"))
        their_name = None if theirs is None else theirs.get("name")
        our_name = None if ours is None else ours.get("name")
        default = " (its default)" if entry.strategy is None else ""
        if their_name != our_name:
            # Information, not a proposal: a method may not take the case's strategy.
            notes.append(f"{label}'s strategy is {their_name or 'its default'}{default}; the "
                         f"case's is {our_name or 'its default'} (the case's localization, "
                         f"{facts.localization or 'none'}, is every arm's)")  # fmt: skip
        elif theirs is not None and ours is not None and (
            (theirs.get("options") or {}) != (ours.get("options") or {})
        ):  # fmt: skip
            notes.append(f"{label}'s {their_name}{default} options differ from the case's: "
                         f"{theirs.get('options') or 'its defaults'} against "
                         f"{ours.get('options') or 'its defaults'}")  # fmt: skip
    return notes


def _with_default_strategy(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a method entry that names its update strategy, its method's default if unset.

    A study's method entry replaces the case's algorithm fields, so an arm with
    no strategy runs its method's default, which a draft did not show: the assistant then
    misdescribed it as the case's (reported issues #225, #231). The default is named, never
    the case's strategy put in (each algorithm is its own base, the user's rule of
    2026-10-01); ``method_differences`` says where it differs from the case's. Another
    engine's arm takes no strategy.
    """
    if entry.get("strategy") is not None or entry.get("engine", "apex") != "apex":
        return entry
    from apex.services.compatibility import default_strategy

    section = default_strategy(str(entry["name"]), entry.get("options"))
    return entry if section is None else {**entry, "strategy": section}


# ---------------------------------------------------------------------- rules


def rules(facts: PlanFacts, goal: str, thresholds: PlannerSection | None = None) -> StudyProposal:
    """Return the rules' plan: a baseline, and localization if the case has none.

    Replicates are as many as the budget allows, capped by ``max_replicates``, never fewer
    than one (a plan over budget is then flagged by :meth:`Planner.check`).
    """
    t = thresholds or PlannerSection()
    variants = [VariantProposal(name="baseline", patch={}, rationale="the case as configured")]
    if ert_localization_goal(facts, goal) and ert_localization_obstacle(facts) is None:
        return _ert_localization_rules(facts, goal, t)
    if "ert" in facts.engines and _wants_ert(goal):
        return _ert_rules(facts, goal, t)
    small: list[VariantProposal] = []
    if facts.localization is None:
        variants += _localization_variants(facts, goal)
        small = _smaller_ensemble(facts, goal, variants)
    both = len(variants) + len(small)
    if (
        small
        and both <= facts.max_variants
        and facts.max_cells is not None
        and facts.max_cells >= 2 * both
    ):
        # Every arm at both sizes, with paired replicates, or the case's size alone: some
        # arms at one size only answer neither question (a reported issue, #246).
        variants += small
    cells = facts.max_cells if facts.max_cells is not None else len(variants)
    if facts.max_cells is not None and len(variants) > 2 and cells < 2 * len(variants):
        # Paired replicates before breadth: drop the later localizations first (with an
        # unknown cost there is no budget to trim to).
        variants = variants[: max(2, cells // 2)]
    replicates = max(1, min(t.max_replicates, cells // len(variants)))
    if cells < len(variants):
        variants = variants[:1]  # not even one replicate of each: the baseline alone
        replicates = 1
    metrics = (["rmse:all"] if facts.synthetic else []) + ["normalized_data_mismatch:all"]
    if facts.has_validation:
        metrics.append("cv_normalized_data_mismatch:all")
    return StudyProposal(
        name=f"{facts.case}_plan",
        hypothesis=f"Which set-up best serves the goal: {goal}"[:300],
        variants=variants,
        replicates=replicates,
        metrics=metrics,
        reasoning=(
            f"rules: the baseline{' and adaptive localization' if len(variants) > 1 else ''}"
            f"{' at two ensemble sizes' if any(_SMALL in v.name for v in variants) else ''}, "
            f"with {replicates} paired replicate(s) within the budget of "
            f"{facts.budget_simulator_runs} simulator runs ({facts.runs_per_cell} per run)"
        ),
    )


#: The adaptive localizations a localization study compares (each where the method takes
#: it); the method's default comes first.
ADAPTIVE_LOCALIZATIONS = ("adaptive_prior", "adaptive_per_update", "adaptive_projected")


def _localization_variants(facts: PlanFacts, goal: str) -> list[VariantProposal]:
    """Return the localization variants of a case without one.

    The method's adaptive localization; for a goal about localization also the other
    adaptive ones the method takes (a reported issue, #201: a localization study compared
    one default variant). A localization needs an explicit strategy: a case without one
    gets its method's default named in the patch (its own).
    """
    from apex.services.compatibility import accepts_localization, default_strategy

    method = dict(facts.case_method)
    if method.get("strategy") is None:
        strategy = default_strategy(facts.method, method.get("options"))
        if strategy is not None:
            method["strategy"] = strategy
    from apex.services.plugins import REGISTRIES

    first = default_localization(facts.method)
    names = [first]
    if _wants_localization(goal):
        same = REGISTRIES["localizations"].factory  # an alias is its plugin: adaptive, _prior
        names += [n for n in ADAPTIVE_LOCALIZATIONS
                  if same(n) is not same(first) and accepts_localization(method, n)]  # fmt: skip
    variants = []
    for name in names:
        patch: dict[str, Any] = {"localization": {"name": name}}
        if facts.case_method.get("strategy") is None and method.get("strategy") is not None:
            patch["strategy"] = method["strategy"]
        label = "adaptive_localization" if name == first else name
        variants.append(VariantProposal(
            name=label, patch={"method": patch},
            rationale=f"adaptive localization ({name}) against spurious correlations",
        ))  # fmt: skip
    return variants


#: The name ending of the smaller-ensemble variants of a localization study.
_SMALL = "_small_ensemble"
#: The smallest ensemble the smaller-ensemble variants use.
MIN_SMALL_ENSEMBLE = 10


def _smaller_ensemble(
    facts: PlanFacts, goal: str, arms: list[VariantProposal]
) -> list[VariantProposal]:
    """Return every arm again at half the ensemble, for a localization goal.

    Localization matters most where the ensemble is small (a reported issue, 2026-10-24);
    the caller adds these only when every arm fits at both sizes (#246). None when the goal
    is not about localization, the half would be below :data:`MIN_SMALL_ENSEMBLE`, a
    stored prior fixes the size, or the method's levels do (multilevel ES-MDA's members are
    its levels').
    """
    if not _wants_localization(goal) or facts.members is None:
        return []
    small = facts.members // 2
    prior = facts.configuration.get("prior", {})
    sections = [prior, *(v for v in (prior.get("variables") or {}).values() if isinstance(v, dict))]
    generators = {_generator_name(p.get("generator")) for p in sections}
    levels = (facts.case_method.get("options") or {}).get("levels")
    if small < MIN_SMALL_ENSEMBLE or "file" in generators or prior.get("groups") or levels:
        return []
    size = {"prior": {"ensemble_size": small}}
    return [
        VariantProposal(name=f"{arm.name}{_SMALL}", patch={**arm.patch, **size},
                        rationale=f"{arm.rationale}, at {small} members, where spurious "
                        "correlations are stronger")
        for arm in arms
    ]  # fmt: skip


def _generator_name(generator: Any) -> str | None:  # noqa: ANN401 (JSON values)
    """Return a prior generator's name, given as a name or as a table with ``name``."""
    if isinstance(generator, dict):
        name = generator.get("name")
        return name if isinstance(name, str) else None
    return generator if isinstance(generator, str) else None


def _ert_localization_rules(facts: PlanFacts, goal: str, t: PlannerSection) -> StudyProposal:
    """Return the rules' plan for comparing APEX's and ERT's localization.

    Four ES-MDA methods on APEX's engine, which differ only in the analysis: APEX's approx
    and ERT's ``ert_esmda`` (the references, without localization), APEX's ``adaptive_rows``
    and ERT's ``ert_adaptive`` (the 3/sqrt(N) per-row adaptive localization).
    """
    options = facts.case_method.get("options", {})
    inflation = options.get("inflation") if facts.method == "esmda" else None
    schedule = {"inflation": list(inflation) if isinstance(inflation, list) else [4, 4, 4, 4]}
    analyses = {
        "esmda_approx": ("approx", {"energy": 0.99}),
        "esmda_ert_esmda": ("ert_esmda", {"truncation": 0.99}),
        "esmda_adaptive_rows": ("adaptive_rows", {"energy": 0.99}),
        "esmda_ert_adaptive": ("ert_adaptive", {"truncation": 0.99}),
    }
    entries = {
        label: MethodEntry.model_validate(
            {"name": "esmda", "options": schedule,
             "strategy": {"name": strategy, "options": strategy_options}}
        )
        for label, (strategy, strategy_options) in analyses.items()
    }  # fmt: skip
    # One run is the schedule's forecasts on every member, whatever the case's own method
    # costs (a GIES case's runs_per_cell is not ES-MDA's).
    if facts.members:
        per_run = facts.members * (len(schedule["inflation"]) + 1)
        affordable = facts.budget_simulator_runs // (per_run * len(entries))
    else:  # members unknown: the case's own cost per run
        affordable = (facts.max_cells or len(entries)) // len(entries)
    replicates = max(1, min(t.max_replicates, affordable))
    metrics = (["rmse:all"] if facts.synthetic else []) + ["normalized_data_mismatch:all"]
    return StudyProposal(
        name=f"{facts.case}_apex_vs_ert_localization",
        hypothesis=f"APEX's adaptive_rows against ERT's adaptive localization: {goal}"[:300],
        methods=entries,
        variants=[VariantProposal(name="base", patch={}, rationale="each analysis as given")],
        replicates=replicates,
        metrics=metrics,
        reasoning=(
            "rules: ERT's adaptive localization runs as the ES-MDA strategy ert_adaptive (its "
            "own engine has no localization); APEX's counterpart is adaptive_rows, and "
            "approx/ert_esmda are the unlocalized references; energy (APEX) and truncation "
            "(ERT) are the same setting, 0.99 in all four, "
            f"{replicates} paired replicate(s) within the budget of "
            f"{facts.budget_simulator_runs} simulator runs"
        ),
    )


def _ert_rules(facts: PlanFacts, goal: str, t: PlannerSection) -> StudyProposal:
    """Return the rules' plan for a goal about ERT: the case's method against ERT's ES-MDA."""
    options = facts.case_method.get("options", {})
    inflation = options.get("inflation") if facts.method == "esmda" else None
    weights = list(inflation) if isinstance(inflation, list) else [4, 4, 4, 4]
    entries = {
        facts.method: MethodEntry.model_validate(facts.case_method),
        "ert_es_mda": MethodEntry(engine="ert", name="es_mda", options={"weights": weights}),
    }
    cells = facts.max_cells if facts.max_cells is not None else len(entries)
    replicates = max(1, min(t.max_replicates, cells // len(entries)))
    metrics = (["rmse:all"] if facts.synthetic else []) + ["normalized_data_mismatch:all"]
    return StudyProposal(
        name=f"{facts.case}_apex_vs_ert",
        hypothesis=f"APEX's {facts.method} against ERT's ES-MDA on the case: {goal}"[:300],
        methods=entries,
        variants=[VariantProposal(name="base", patch={}, rationale="the case as configured")],
        replicates=replicates,
        metrics=metrics,
        reasoning=(
            f"rules: the goal asks about ERT, so APEX's {facts.method} and ERT's ES-MDA "
            f"(weights {weights}) on the same priors, {replicates} paired replicate(s) within "
            f"the budget of {facts.budget_simulator_runs} simulator runs"
        ),
    )


# ---------------------------------------------------------------------- the agent

SYSTEM = (
    "You plan history-matching experiments with APEX. You get facts about one case (its "
    "configuration, what one run costs, the budget, the plugins) and a goal. Propose a study: "
    "variants of the case as JSON merge patches over its configuration (the baseline has "
    "an empty patch), the replicates (seeds per variant; variants are compared paired by "
    "replicate, so two or more give a significance test) and the metrics to compare. Stay "
    "within the budget: methods x variants x replicates x the cost of one run. Use only "
    "plugin names from the facts and option names from get_schema. Never patch run.name, "
    "run.seed or run.output_dir: the study sets them. The algorithm is the base and auxiliary "
    "techniques (localization, the damping scheme [method.damping]: rlmmac or lm) are "
    "variants: to compare algorithms (e.g. esmda and gies), "
    "give each as a complete entry in methods, written from its own schema (case_method only "
    "for the case's own algorithm), and never switch method.name in a variant. Never use "
    "null in a patch. When the goal asks about ERT and facts.engines lists ert, ERT's own "
    'smoothers can be compared as methods with engine = "ert" (name es_mda with options '
    "weights, or ensemble_smoother), beside APEX's; variants that patch [method] do not "
    "apply to them. ERT's engine has no localization: to compare localization in APEX and "
    "ERT, use ES-MDA methods on APEX's engine that differ in strategy: adaptive_rows "
    "(APEX's per-row adaptive localization) against ert_adaptive (ERT's), with approx and "
    "ert_esmda as the unlocalized references (ert_localized is ERT's distance taper)."
)
_RULE = "the planner's rules: baseline and localization within the budget"


@dataclass(frozen=True)
class PlanResult:
    """A planned study.

    Attributes
    ----------
    facts
        What the planner knew.
    proposal
        The accepted proposal (the model's, or the rules').
    spec
        The study it makes.
    validation
        The service's validation of the study (its runs and their plans).
    cost
        Simulator runs the study may take (the sum of its runs' upper bounds).
    within_budget
        Whether ``cost`` fits the budget.
    decided_by, reason, trace
        See :class:`~apex.agents.common.Decision`; ``reason`` also says why proposals
        were refused.
    revisions
        The errors of each refused proposal, in order.
    differences
        What the compared algorithms' arms still differ in besides the algorithm, after
        :func:`method_differences` (a reported issue).
    """

    facts: PlanFacts
    proposal: StudyProposal
    spec: StudySpec
    validation: Validation
    cost: int | None
    within_budget: bool
    decided_by: str
    reason: str | None
    trace: Path
    revisions: tuple[str, ...] = ()
    differences: tuple[str, ...] = ()

    def toml(self, workspace: "Workspace | None" = None, study_dir: Path | None = None) -> str:
        """Return the study as TOML, with the proposal's hypothesis and reasoning.

        With ``workspace`` and ``study_dir`` (the file's folder), its cases are named
        without server paths (:func:`place_cases`).
        """
        spec = self.spec
        if workspace is not None and study_dir is not None:
            spec = place_cases(spec, workspace, study_dir)
        return study_toml(
            spec, self.proposal, self.cost, self.facts.budget_simulator_runs,
            self.decided_by, self.facts.plugins.get("simulators", []),
        )  # fmt: skip


def study_toml(
    spec: StudySpec,
    proposal: StudyProposal,
    cost: int | None,
    budget: int,
    decided_by: str,
    simulators: Sequence[str] = (),
) -> str:
    """Return a planned study as TOML, with the proposal's hypothesis and reasoning.

    ``simulators`` (the registered ones) are listed in the header, for a person to choose.
    """
    from apex.simulation.deck import DECK_SIMULATORS

    document = spec.model_dump(mode="json", exclude_defaults=False)
    decks = [s for s in simulators if s in DECK_SIMULATORS]
    choice = (
        f'# Simulator: [study] simulator = "{spec.study.simulator}"; the deck simulators are '
        f"{', '.join(decks)} (OPM Flow by default; remove the key to run each case's own).\n"
        if spec.study.simulator is not None and decks
        else ""
    )
    header = (
        f"# Planned by APEX's planning agent ({decided_by}).\n"
        f"# Hypothesis: {proposal.hypothesis}\n"
        f"# Reasoning: {proposal.reasoning}\n"
        + "".join(f"# {v.name}: {v.rationale}\n" for v in proposal.variants)
        + choice
        + f"# Cost: at most {cost} simulator runs (budget {budget}).\n\n"
    )
    text = header + tomli_w.dumps(_drop_none(document))
    # TOML has no null: the file must say what was validated, or the launch check (which
    # reads the file) would run something else.
    if StudySpec.model_validate(tomllib.loads(text)) != StudySpec.model_validate(document):
        raise ConfigError(
            f"study {spec.study.name}: its TOML would differ from the validated study "
            "(a null in a patch?)"
        )
    return text


def place_cases(spec: StudySpec, workspace: "Workspace", study_dir: Path) -> StudySpec:
    """Return the study with its cases named as a person reads them, no server paths.

    A case in the workspace is named as the UI shows it (``5Spots/configs/gies.toml``);
    another is named relative to the study file's folder. Both resolve back to the same
    file (:func:`apex.services.study.resolve_case`).
    """
    cases: dict[str, Path] = {}
    for name, path in spec.study.cases.items():
        try:
            cases[name] = Path(workspace.relative(path))
        except ConfigError:  # outside the workspace (WorkspaceError)
            cases[name] = Path(os.path.relpath(path, study_dir))
    return spec.model_copy(update={"study": spec.study.model_copy(update={"cases": cases})})


def null_paths(value: Any, prefix: str = "") -> list[str]:  # noqa: ANN401 (JSON values)
    """Return the dotted paths of ``None`` values in a JSON document."""
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found += [path] if item is None else null_paths(item, path)
        return found
    if isinstance(value, list):
        return [p for i, item in enumerate(value) for p in null_paths(item, f"{prefix}[{i}]")]
    return []


def _drop_none(value: Any) -> Any:  # noqa: ANN401 (JSON values)
    """Remove ``None`` values (TOML has no null)."""
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


class Planner:
    """Plans studies (see the module docstring).

    Parameters
    ----------
    service
        The services (to validate and cost proposals).
    provider
        The model; ``None`` for rules only.
    thresholds
        ``[agents.planner]`` of ``service.toml`` by default.
    settings
        Bounds of each of the model's tasks.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        thresholds: PlannerSection | None = None,
        settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        # The installed advice applies unless thresholds are given.
        self.threshold_advice: list[str] = []
        if thresholds is None:
            thresholds, self.threshold_advice = advise_thresholds(
                "agents.planner", service.jobs.settings.agents.planner, service.root
            )
        self.thresholds = thresholds
        self.settings = settings or LoopSettings(max_turns=6)

    def spec(self, facts: PlanFacts, proposal: StudyProposal) -> StudySpec:
        """Return the study a proposal makes (its budget is the facts').

        Raises
        ------
        pydantic.ValidationError
            If the proposal does not make a valid study (e.g. a patch of run.seed).
        """
        return StudySpec.model_validate({
            "study": {
                "name": proposal.name,
                "replicates": proposal.replicates,
                "metrics": proposal.metrics,
                "budget": {"max_simulator_runs": facts.budget_simulator_runs},
                "annotations": {"planned_by": "apex-planner"},
                "cases": {facts.case: facts.config_file},
                "simulator": study_simulator(proposal.simulator, facts.simulator),
                "methods": {
                    k: _with_default_strategy(v.model_dump(mode="json", exclude_unset=True))
                    for k, v in proposal.methods.items()
                },
                "variants": [{"name": v.name, "patch": v.patch} for v in proposal.variants],
            }
        })  # fmt: skip

    def check(
        self, facts: PlanFacts, proposal: StudyProposal
    ) -> tuple[StudySpec | None, Validation | None, int | None, list[str]]:
        """Hold a proposal to the facts: return its study, validation, cost and errors."""
        errors: list[str] = []
        groups = max(len(proposal.methods), 1) * len(proposal.variants)
        if groups > facts.max_variants:
            errors.append(
                f"{groups} methods x variants; at most {facts.max_variants} "
                "([agents.planner] max_variants)"
            )
        nulls_found = False
        for v in proposal.variants:
            nulls = null_paths(v.patch)
            if nulls:
                nulls_found = True
                errors.append(
                    f"variant {v.name!r} sets {', '.join(nulls)} to null, which a TOML study "
                    "cannot hold: to compare algorithms use methods (complete entries), "
                    "otherwise leave the key out"
                )
        if nulls_found:
            return None, None, None, errors
        try:
            spec = self.spec(facts, proposal)
        except ValidationError as error:
            errors.extend(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            return None, None, None, errors
        validation = self.service.validate_study(spec)
        errors.extend(validation.errors)
        costs = [p.max_simulator_runs for p in validation.plans]
        cost = None if not costs or any(c is None for c in costs) else sum(c or 0 for c in costs)
        # The study's own refusal (its budget is this one) says each arm's cost and what
        # fits (a reported issue, #248); this one only when another error hid it.
        refused = any(e.startswith(OVER_BUDGET) for e in validation.errors)
        if cost is not None and cost > facts.budget_simulator_runs and not refused:
            errors.append(
                f"the study may take {cost} simulator runs, over the budget of "
                f"{facts.budget_simulator_runs}: use fewer variants or replicates"
            )
        return spec, validation, cost, errors

    def plan(self, config_file: Path, goal: str, *, budget: int | None = None) -> PlanResult:
        """Plan a study for a case and a goal (see the module docstring)."""
        facts = plan_facts(self.service, config_file, budget=budget, thresholds=self.thresholds)
        recorder = CallRecorder(self.service.root, "planner")
        system, prompt_advice = advise_prompt(SYSTEM, "planner", self.service.root)
        recorder.advice = sorted([*prompt_advice, *self.threshold_advice])
        rule_plan = rules(facts, goal, self.thresholds)
        thresholds = self.thresholds.model_dump(mode="json")
        revisions: list[str] = []
        base_task = (
            f"Goal: {goal}\n\nFacts of the case:\n{facts.model_dump_json(indent=1)}\n\n"
            f"The rules' plan, for reference:\n{rule_plan.model_dump_json(indent=1)}\n\n"
            "Propose the study."
        )
        decision: Decision[StudyProposal] | None = None
        attempts = 1 + self.thresholds.max_revisions if self.provider is not None else 1
        for attempt in range(attempts):
            task = base_task
            if revisions:
                task += (
                    f"\n\nYour previous proposal was refused:\n{revisions[-1]}\n"
                    "Correct it and propose again."
                )
            decision = decide(
                self.service, self.provider, recorder, task=task,
                response_model=StudyProposal, rules=rule_plan, rule=_RULE,
                thresholds=thresholds, client="planner", settings=self.settings, system=system,
            )  # fmt: skip
            if decision.decided_by != "model":
                break
            spec, validation, cost, errors = self.check(facts, decision.value)
            if not errors and spec is not None and validation is not None:
                differences = self._differences(facts, decision, recorder)
                return self._result(facts, decision, spec, validation, cost, revisions,
                                    differences)  # fmt: skip
            revisions.append("; ".join(errors))
            recorder.record_action(
                f"refused proposal {attempt + 1}: " + revisions[-1], proposal=decision.value
            )
        assert decision is not None
        reason = decision.reason
        if decision.decided_by == "model":  # every revision refused: the rules plan
            reason = f"{len(revisions)} proposal(s) refused; the rules planned"
            trace = recorder.record_decision(
                self.provider, task=base_task, decision=rule_plan, fell_back=True,
                reason=reason, rule=_RULE, thresholds=thresholds, tool_calls=[],
            )  # fmt: skip
            decision = Decision(rule_plan, "fallback", reason, trace)
        spec, validation, cost, errors = self.check(facts, decision.value)
        if spec is None or validation is None:
            raise ConfigError("the rules' plan is not a valid study: " + "; ".join(errors))
        return self._result(facts, decision, spec, validation, cost, revisions)

    def _differences(
        self, facts: PlanFacts, decision: Decision[StudyProposal], recorder: CallRecorder
    ) -> tuple[str, ...]:
        """Record how the model's compared algorithms differ from the case (a reported issue)."""
        differences = method_differences(facts, decision.value) if decision.value.methods else []
        if differences:
            recorder.record_action(
                "the compared algorithms differ from the case besides the algorithm",
                differences=differences,
            )
        return tuple(differences)

    def _result(
        self,
        facts: PlanFacts,
        decision: Decision[StudyProposal],
        spec: StudySpec,
        validation: Validation,
        cost: int | None,
        revisions: list[str],
        differences: tuple[str, ...] = (),
    ) -> PlanResult:
        within = cost is not None and cost <= facts.budget_simulator_runs and validation.valid
        return PlanResult(
            facts, decision.value, spec, validation, cost, within, decision.decided_by,
            decision.reason, decision.trace, tuple(revisions), differences,
        )  # fmt: skip


def submit(
    service: ApexService,
    study_file: Path,
    *,
    confirm: Callable[[str], bool] | None,
    client: str = "planner",
    model: str | None = None,
    approval_token: str | None = None,
) -> dict[str, Any]:
    """Submit a planned study through the tools, once a person agrees.

    Returns what happened: ``submitted`` (with the job), ``pending`` (the approval token a
    person must approve with ``apex approve``, then submit again with ``approval_token``),
    ``declined`` or ``refused`` (with the tools' error).
    """
    # Not confined to the tool clients' folders: the owner named the study at the terminal
    # (`apex agent plan CONFIG --submit`) and confirms the submission.
    session = ToolSession(service, client=client, model=model, confined=False)
    recorder = CallRecorder(service.root, "planner")
    if approval_token is not None:
        # An approved token from an earlier, pending submission; submit_study checks it
        # against this exact study (and refuses it if still pending or used).
        if confirm is None or not confirm(f"Submit study {study_file} with the approved token?"):
            recorder.record_action("declined: not submitted", study_file=str(study_file))
            return {"outcome": "declined"}
        return _submit(session, recorder, study_file, approval_token)
    checked = call("validate", {"config_file": str(study_file)}, session)
    if not checked["ok"]:
        recorder.record_action("refused: validation failed", result=checked)
        return {"outcome": "refused", "error": checked.get("error"), "hint": checked.get("hint")}
    result = checked["result"]
    token = result.get("approval_token")
    if not result.get("valid", True) or token is None:
        recorder.record_action("refused: the study is not valid", result=result)
        return {"outcome": "refused", "error": result.get("errors")}
    if result.get("approval") == "pending":
        recorder.record_action("pending approval", token=token)
        return {"outcome": "pending", "approval_token": token, "next": result.get("next")}
    runs = result.get("max_simulator_runs")
    question = f"Submit study {study_file} (at most {runs} simulator runs)?"
    if confirm is None or not confirm(question):
        recorder.record_action("declined: not submitted", study_file=str(study_file))
        return {"outcome": "declined"}
    return _submit(session, recorder, study_file, token)


def _submit(
    session: ToolSession, recorder: CallRecorder, study_file: Path, token: str
) -> dict[str, Any]:
    submitted = call(
        "submit_study", {"study_file": str(study_file), "approval_token": token}, session
    )
    if not submitted["ok"]:
        recorder.record_action("refused by submit_study", result=submitted)
        return {"outcome": "refused", "error": submitted.get("error")}
    recorder.record_action("submitted", result=submitted["result"])
    return {"outcome": "submitted", **submitted["result"]}
