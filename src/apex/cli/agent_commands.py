"""``apex agent``: APEX's own agent loop on any model, and the run monitor.

``apex agent ask QUESTION`` runs the loop of :mod:`apex.agents.loop` with the model of
``[agents.llm]`` (``<root>/service.toml``), or the one given by options, and APEX's tools.
Its calls and its answer are traced in ``<root>/agents/``. Without a usable model, the
deterministic fallback answers, and says why. ``apex agent monitor RUN`` checks a run's
health (``--watch`` to follow it, ``--act`` to stop it on a stop recommendation).
``apex agent diagnose RUN`` explains what went wrong and what to change, ``apex agent plan
CONFIG --goal TEXT`` proposes a study, and ``apex agent report RUN`` (or ``--study NAME``)
writes a narrative report. ``apex agent providers`` lists the providers.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import BaseModel, Field

from apex.core.errors import ApexError
from apex.storage.run_directory import RunDirectory

if TYPE_CHECKING:
    from apex.agents.converter import CreateOutcome
    from apex.agents.loop import LoopSettings
    from apex.agents.monitor import MonitorResult
    from apex.core.llm import LLMProvider
    from apex.services.api import ApexService
    from apex.services.settings import LLMSection

agent_app = typer.Typer(no_args_is_help=True, help="APEX's agent loop on any LLM.")

Root = Annotated[
    Path | None, typer.Option(help="Services root (default: $APEX_HOME, else ~/.apex/service).")
]
Provider = Annotated[str | None, typer.Option(help="Provider (default: \\[agents.llm] provider).")]
Model = Annotated[str | None, typer.Option(help="Model (default: \\[agents.llm] model).")]
BaseURL = Annotated[str | None, typer.Option(help="Server address.")]
KeyEnv = Annotated[
    str | None, typer.Option(help="Variable holding the key (never the key itself).")
]
# Flags without a "--no-no-..." form (a reported issue, 2026-10-24).
NoToolCalls = Annotated[
    bool,
    typer.Option("--no-tool-calls", help="The model has no native tool calling: prompt for JSON."),
]


def _llm(
    service: "ApexService",
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    no_tool_calls: bool,
) -> tuple["LLMProvider", "LLMSection"]:
    """Return the provider of ``[agents.llm]``, with the command's overrides."""
    from apex.agents.providers import create_provider

    overrides = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "tool_calls": False if no_tool_calls else None,
    }
    llm = service.jobs.settings.agents.llm
    llm = llm.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    try:
        return create_provider(llm), llm
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


class Answer(BaseModel):
    """What ``apex agent ask`` expects from the model."""

    answer: str = Field(description="The answer to the question, for a person to read.")


@agent_app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="The question or task.")],
    root: Root = None,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
    tools: Annotated[bool, typer.Option(help="Offer APEX's tools.")] = True,
    max_turns: Annotated[int, typer.Option(min=1, help="Model calls at most.")] = 8,
    client: Annotated[str, typer.Option(help="Client name for quotas and the call log.")] = "agent",
) -> None:
    """Ask APEX's agent a question; it may use APEX's tools to answer."""
    from apex.agents.loop import AgentLoop, Fallback, LoopSettings
    from apex.agents.trace import CallRecorder
    from apex.services.api import ApexService
    from apex.services.tools import ToolSession

    service = ApexService(root)
    chosen, llm = _llm(service, provider, model, base_url, api_key_env, no_tool_calls)
    session = ToolSession(service, client=client, model=chosen.model_id, host=llm.base_url)
    loop = AgentLoop(
        chosen,
        CallRecorder(service.root, "ask"),
        session=session if tools else None,
        settings=LoopSettings(
            max_turns=max_turns, max_tokens=llm.max_tokens, temperature=llm.temperature
        ),
    )
    fallback = Fallback(
        "no model answer: say so, and point to the trace",
        lambda: Answer(answer="(no answer: the model was unavailable or its answers were invalid)"),
    )
    outcome = loop.run(question, Answer, fallback)
    typer.echo(outcome.decision.answer)
    if outcome.fell_back:
        typer.secho(f"fallback: {outcome.reason}", err=True, fg=typer.colors.YELLOW)
    typer.secho(f"trace: {outcome.trace}", err=True, dim=True)


@agent_app.command("monitor")
def monitor(
    target: Annotated[str, typer.Argument(help="A run directory, a run id or a job id.")],
    root: Root = None,
    watch: Annotated[
        bool, typer.Option(help="Check whenever a new forecast is scored, until the job ends.")
    ] = False,
    interval: Annotated[
        float, typer.Option(min=1, help="Seconds between looks, with --watch.")
    ] = 60,
    act: Annotated[
        bool,
        typer.Option(
            help="Offer to stop the job when a stop is recommended (running iterative runs "
            "only); you are asked to confirm first."
        ),
    ] = False,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Rules only: no model is asked.")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Check a run's health: rules on its facts, and a model's judgement."""
    import time

    from apex.agents.loop import LoopSettings
    from apex.agents.monitor import RunMonitor
    from apex.services.api import ApexService
    from apex.services.jobs import FINAL

    service = ApexService(root)
    chosen = None
    settings = LoopSettings(max_turns=6)
    if not no_llm:
        chosen, llm = _model_or_rules(
            service, provider, model, base_url, api_key_env, no_tool_calls
        )
        if llm is not None:
            settings = LoopSettings(
                max_turns=6, max_tokens=llm.max_tokens, temperature=llm.temperature
            )
    if _is_optimization(service, target):
        _monitor_optimization(service, chosen, settings, target, act, json_output)
        return
    agent = RunMonitor(service, chosen, settings=settings)
    try:
        run_path, job = agent.resolve(target)
    except ApexError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    seen = -1
    while True:
        scored = len(RunDirectory(run_path).history_dicts())
        state = None if job is None else service.jobs.record(job)["state"]
        ended = job is None or state in FINAL
        if scored != seen and scored > 0:
            seen = scored
            try:
                result = agent.check(run_path, act=act, confirm=_confirm)
            except ApexError as error:
                typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
                raise typer.Exit(1) from error
            _show(result, json_output)
        if not watch or ended:
            return
        time.sleep(interval)


def _is_optimization(service: "ApexService", target: str) -> bool:
    """Whether ``target`` is an optimization's directory or job."""
    from apex.agents.optimization import is_optimization_dir

    if is_optimization_dir(Path(target)):
        return True
    try:
        record = service.jobs.record(target)
    except KeyError:
        return False
    return record["kind"] in ("optimization", "optimization_resume")


def _monitor_optimization(
    service: "ApexService",
    provider: "LLMProvider | None",
    settings: "LoopSettings",
    target: str,
    act: bool,
    json_output: bool,
) -> None:
    from apex.agents.optimization import OptimizationMonitor

    if act:
        typer.secho("--act does not apply to optimizations: the monitor only advises",
                    err=True, fg=typer.colors.YELLOW)  # fmt: skip
    try:
        check = OptimizationMonitor(service, provider, settings=settings).check(target)
    except ApexError as error:
        raise _fail(error) from error
    if json_output:
        typer.echo(json.dumps({
            "decided_by": check.decided_by,
            "fallback_reason": check.reason,
            "report": check.report.model_dump(),
            "rules_report": check.rules_report.model_dump(),
            "facts": check.facts.model_dump(),
            "corrections": list(check.corrections),
            "trace": str(check.trace),
        }))  # fmt: skip
        return
    report = check.report
    colour = {"healthy": typer.colors.GREEN, "warning": typer.colors.YELLOW}.get(
        report.status, typer.colors.RED
    )
    typer.secho(f"{report.status}: {_ADVICE[report.recommendation]}", fg=colour, bold=True)
    typer.echo(report.summary)
    for finding in report.findings:
        typer.echo(f"  [{finding.severity}] {finding.code}: {finding.evidence}")


def _confirm(question: str) -> bool:
    """Ask the person at the terminal; no answer (e.g. under nohup) is a no."""
    try:
        return typer.confirm(question, default=False, err=True)
    except typer.Abort:
        typer.secho("no answer: the job is not stopped", err=True, fg=typer.colors.YELLOW)
        return False


def _show(result: "MonitorResult", json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps({
            "decided_by": result.decided_by,
            "fallback_reason": result.reason,
            "report": result.report.model_dump(),
            "rules_report": result.rules_report.model_dump(),
            "observation": result.observation.model_dump(),
            "action": result.action,
            "corrections": list(result.corrections),
            "trace": str(result.trace),
        }))  # fmt: skip
        return
    report = result.report
    colour = {"healthy": typer.colors.GREEN, "warning": typer.colors.YELLOW}.get(
        report.status, typer.colors.RED
    )
    typer.secho(f"{report.status}: {_ADVICE[report.recommendation]}", fg=colour, bold=True)
    typer.echo(report.summary)
    for finding in report.findings:
        typer.echo(f"  [{finding.severity}] {finding.code}: {finding.evidence}")
    for correction in result.corrections:
        typer.secho(f"corrected: {correction}", fg=typer.colors.YELLOW)
    if result.decided_by != "rules":
        rules = ", ".join(f.code for f in result.rules_report.findings) or "nothing"
        typer.echo(f"(rules: {result.rules_report.recommendation}; noticed {rules})")
    if result.decided_by == "fallback":
        typer.secho(f"fallback: {result.reason}", err=True, fg=typer.colors.YELLOW)
    if result.action != "none":
        typer.secho(f"action: {result.action}", bold=True)
    typer.secho(f"trace: {result.trace}", err=True, dim=True)


def _model_or_rules(
    service: "ApexService",
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    no_tool_calls: bool,
) -> tuple["LLMProvider | None", "LLMSection | None"]:
    """Return the model for an agent with rules, or ``(None, None)``: the rules answer.

    A model asked for on the command line must work (an error otherwise); one that
    ``[agents.llm]`` cannot set up leaves the answer to the agent's rules, with a warning,
    as the REST routes do (docs/api.md; the release tester, 2026-10-19).
    """
    if not any((provider, model, base_url, api_key_env)):
        from apex.agents.providers import create_provider

        try:
            create_provider(service.jobs.settings.agents.llm)
        except ApexError as error:
            typer.secho(
                f"warning: no model ({error}); the agent's rules answer. Set [agents.llm] in "
                "service.toml, or pass --no-llm to say so.",
                err=True, fg=typer.colors.YELLOW,
            )  # fmt: skip
            return None, None
    return _llm(service, provider, model, base_url, api_key_env, no_tool_calls)


def _provider_and_settings(
    service: "ApexService",
    no_llm: bool,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    no_tool_calls: bool,
    max_turns: int,
) -> tuple["LLMProvider | None", "LoopSettings"]:
    """Return the model (``None`` with ``--no-llm``) and the loop's bounds."""
    from apex.agents.loop import LoopSettings

    if no_llm:
        return None, LoopSettings(max_turns=max_turns)
    chosen, llm = _model_or_rules(service, provider, model, base_url, api_key_env, no_tool_calls)
    if chosen is None or llm is None:
        return None, LoopSettings(max_turns=max_turns)
    return chosen, LoopSettings(
        max_turns=max_turns, max_tokens=max(llm.max_tokens, 2048), temperature=llm.temperature
    )


#: The monitor's recommendations, in words ("healthy: none" read as nothing found).
_ADVICE = {
    "continue": "let it continue",
    "stop": "stop it",
    "investigate": "investigate",
    "none": "nothing to do",
}


def _fail(error: Exception) -> typer.Exit:
    typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
    return typer.Exit(1)


def _footer(decided_by: str, reason: str | None, corrections: tuple[str, ...], trace: Path) -> None:
    for correction in corrections:
        typer.secho(f"corrected: {correction}", fg=typer.colors.YELLOW)
    if decided_by == "fallback":
        typer.secho(f"fallback: {reason}", err=True, fg=typer.colors.YELLOW)
    typer.secho(f"decided by: {decided_by}; trace: {trace}", err=True, dim=True)


NoLLM = Annotated[bool, typer.Option("--no-llm", help="Rules only: no model is asked.")]
JsonOutput = Annotated[bool, typer.Option("--json", help="Print the result as JSON.")]


@agent_app.command("diagnose")
def diagnose(
    target: Annotated[str, typer.Argument(help="A run directory, a run id or a job id.")],
    root: Root = None,
    no_llm: NoLLM = False,
    json_output: JsonOutput = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Explain what went wrong in a run and what to change (advisory)."""
    from apex.agents.diagnosis import Diagnostician
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 6
    )
    try:
        from apex.agents.diagnosis import DiagnosisResult
        from apex.agents.optimization_advice import (
            OptimizationDiagnosis,
            OptimizationDiagnostician,
        )

        result: DiagnosisResult | OptimizationDiagnosis
        if _is_optimization(service, target):  # 2026-10-04
            result = OptimizationDiagnostician(service, chosen, settings=settings).diagnose(target)
        else:
            result = Diagnostician(service, chosen, settings=settings).diagnose(target)
    except ApexError as error:
        raise _fail(error) from error
    d = result.diagnosis
    if json_output:
        typer.echo(json.dumps({
            "decided_by": result.decided_by, "fallback_reason": result.reason,
            "diagnosis": d.model_dump(), "rules_diagnosis": result.rules_diagnosis.model_dump(),
            "facts": result.facts.model_dump(mode="json"),
            "corrections": list(result.corrections), "trace": str(result.trace),
        }))  # fmt: skip
        return
    colour = {"healthy": typer.colors.GREEN, "problems": typer.colors.YELLOW}.get(
        d.verdict, typer.colors.RED
    )
    typer.secho(d.verdict, fg=colour, bold=True)
    typer.echo(d.summary)
    for cause in d.causes:
        typer.echo(f"  [{cause.severity}] {cause.code}: {cause.evidence}")
    for i, rec in enumerate(d.recommendations, 1):
        typer.echo(f"{i}. {rec.change}\n   why: {rec.rationale}")
        if rec.patch:
            typer.echo(f"   patch (validated): {json.dumps(rec.patch)}")
    for note in result.facts.notes:
        typer.secho(f"note: {note}", dim=True)
    _footer(result.decided_by, result.reason, result.corrections, result.trace)


@agent_app.command("plan")
def plan(
    config: Annotated[
        Path, typer.Argument(help="The case: a history-matching or optimization configuration.")
    ],
    goal: Annotated[str, typer.Option(help="What the experiment should find out.")],
    root: Root = None,
    budget: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Simulator runs for the whole study (default: the approval "
            "limit of service.toml).",
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Where to write the study (default: beside the case).")
    ] = None,
    submit: Annotated[
        bool,
        typer.Option(
            help="Submit the study after you confirm (through the tools, with an approval token)."
        ),
    ] = False,
    approval_token: Annotated[
        str | None,
        typer.Option(
            help="Submit with this token, approved by `apex approve` after an "
            "earlier --submit was pending (implies --submit)."
        ),
    ] = None,
    no_llm: NoLLM = False,
    json_output: JsonOutput = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Propose a study for a case and a goal; submit it only if you confirm."""
    from apex.agents.planner import Planner
    from apex.agents.planner import submit as submit_study
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 6
    )
    from apex.agents.optimization_advice import (
        OptimizationPlanner,
        OptimizationPlanResult,
        is_optimization_config,
    )
    from apex.agents.planner import PlanResult

    result: PlanResult | OptimizationPlanResult
    try:
        if is_optimization_config(config):  # a study of optimizations
            result = OptimizationPlanner(service, chosen, settings=settings).plan(
                config, goal, budget=budget
            )
        else:
            result = Planner(service, chosen, settings=settings).plan(config, goal, budget=budget)
    except ApexError as error:
        raise _fail(error) from error
    target = out or config.resolve().parent / f"study_{result.spec.study.name}.toml"
    # Cases named as the workspace shows them, or relative to the file.
    target.write_text(result.toml(service.workspace(), target.resolve().parent))
    # An optimization study's plan compares no algorithms (OptimizationPlanResult).
    differences: tuple[str, ...] = getattr(result, "differences", ())
    if differences and not json_output:
        # Before any confirmation: the compared arms differ in more than the algorithm
        # (a reported issue).
        typer.secho("the compared algorithms also differ in:", fg=typer.colors.YELLOW, err=True)
        for difference in differences:
            typer.secho(f"  {difference}", fg=typer.colors.YELLOW, err=True)
    outcome = None
    if submit or approval_token:
        outcome = submit_study(
            service, target, confirm=_confirm_submit, model=None if chosen is None else
            chosen.model_id, approval_token=approval_token,
        )  # fmt: skip
    if json_output:
        typer.echo(json.dumps({
            "decided_by": result.decided_by, "fallback_reason": result.reason,
            "proposal": result.proposal.model_dump(), "study_file": str(target),
            "cost": result.cost, "within_budget": result.within_budget,
            "revisions": list(result.revisions), "differences": list(differences),
            "submission": outcome,
            "trace": str(result.trace),
        }))  # fmt: skip
        return
    p = result.proposal
    typer.secho(f"study {p.name}: {len(p.variants)} variant(s) x {p.replicates} replicate(s)",
                bold=True)  # fmt: skip
    typer.echo(f"hypothesis: {p.hypothesis}")
    for v in p.variants:
        typer.echo(f"  {v.name}: {v.rationale}" + (f"  {json.dumps(v.patch)}" if v.patch else ""))
    colour = typer.colors.GREEN if result.within_budget else typer.colors.RED
    typer.secho(
        f"cost: at most {result.cost} simulator runs (budget "
        f"{result.facts.budget_simulator_runs})", fg=colour,
    )  # fmt: skip
    for i, revision in enumerate(result.revisions, 1):
        typer.secho(f"refused proposal {i}: {revision}", fg=typer.colors.YELLOW)
    typer.echo(f"written: {target}")
    if outcome is not None:
        if outcome["outcome"] == "pending":
            typer.secho(
                f"approval needed: run `apex approve {outcome['approval_token']}`, then "
                f"`apex agent plan ... --approval-token {outcome['approval_token']}` or "
                "`apex study run` on the file",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.echo(f"submission: {json.dumps(outcome)}")
    else:
        typer.echo(f"to run it: apex study run {target}")
    _footer(result.decided_by, result.reason, (), result.trace)


def _confirm_submit(question: str) -> bool:
    """Ask the person at the terminal; no answer is a no."""
    try:
        return typer.confirm(question, default=False, err=True)
    except typer.Abort:
        typer.secho("no answer: nothing is submitted", err=True, fg=typer.colors.YELLOW)
        return False


@agent_app.command("draft-optimization")
def draft_optimization(
    case: Annotated[Path, typer.Argument(help="The history-matching case (a deck simulator).")],
    forecast_end: Annotated[
        str, typer.Option(help="The forecast's last date (ISO), after the history.")
    ],
    wop: Annotated[float | None, typer.Option(help="Oil price per unit of FOPT.")] = None,
    wwp: Annotated[float | None, typer.Option(help="Cost per unit of produced water.")] = None,
    wwi: Annotated[float | None, typer.Option(help="Cost per unit of injected water.")] = None,
    disc: Annotated[float | None, typer.Option(help="Yearly discount rate.")] = None,
    wgp: Annotated[float | None, typer.Option(help="Gas price (default: not priced).")] = None,
    obj_scaling: Annotated[
        float | None, typer.Option(help="Divides NPV (default -1e6: millions, maximised).")
    ] = None,
    producer_bhp: Annotated[
        str | None, typer.Option(help="Producers' pressure bounds LOW,HIGH (with bhp control).")
    ] = None,
    producer_control: Annotated[
        str, typer.Option(help="Producers on 'bhp' or 'oil_rate'.")
    ] = "bhp",
    control_periods: Annotated[int, typer.Option(min=1, max=60)] = 4,
    run: Annotated[
        str | None, typer.Option(help="The run whose posterior is the model (id, name or path).")
    ] = None,
    members: Annotated[int, typer.Option(min=1, max=200, help="Posterior members used.")] = 10,
    optimizer: Annotated[str, typer.Option(help="The optimizer plugin.")] = "enopt",
    name: Annotated[str | None, typer.Option(help="The draft's name (default <run>_npv).")] = None,
    out: Annotated[
        Path | None, typer.Option(help="The folder to write it in (default: beside the case).")
    ] = None,
    root: Root = None,
    json_output: JsonOutput = False,
) -> None:
    """Draft an optimization of a history-matched case's forecast (no model).

    The economics are yours: without the prices, the discount and (on bhp) the producers'
    pressure bounds, the draft is refused, naming what is missing. Run it with
    ``apex optimize``.
    """
    from apex.services.api import ApexService
    from apex.services.optimization_draft import DraftRequest, draft_optimization

    bounds: tuple[float, float] | None = None
    if producer_bhp is not None:
        try:
            low, high = (float(v) for v in producer_bhp.split(","))
        except ValueError as error:
            raise _fail(ValueError("--producer-bhp is LOW,HIGH, e.g. 50,300")) from error
        bounds = (low, high)
    service = ApexService(root)
    try:
        request = DraftRequest.model_validate({
            "case": str(case.resolve()), "forecast_end": forecast_end, "wop": wop, "wwp": wwp,
            "wwi": wwi, "disc": disc, "wgp": wgp, "obj_scaling": obj_scaling,
            "producer_bhp": bounds, "producer_control": producer_control,
            "control_periods": control_periods, "run": run, "members": members,
            "optimizer": optimizer, "name": name,
        })  # fmt: skip
        made = draft_optimization(service, request, (out or case.resolve().parent).resolve())
    except (ApexError, ValueError) as error:  # a DraftRefused names what is missing
        raise _fail(error) from error
    plan = made.validation.optimization
    cost = plan.max_simulator_runs if plan is not None else None
    if json_output:
        typer.echo(json.dumps({
            "config": str(made.config), "deck": str(made.deck), "model": str(made.model),
            "wells": made.wells, "assumptions": made.assumptions, "cost": cost,
        }))  # fmt: skip
        return
    typer.secho(f"draft: {made.config}", bold=True)
    for well, control in made.wells.items():
        typer.echo(f"  {well}: {control}")
    for assumption in made.assumptions:
        typer.secho(f"assumed: {assumption}", fg=typer.colors.YELLOW)
    typer.echo(f"cost: at most {cost} simulator runs" if cost is not None else "cost: unbounded")
    typer.echo(f"review it, then: apex optimize {made.config}")


@agent_app.command("report")
def report(
    target: Annotated[
        str | None, typer.Argument(help="A run directory, a run id or a job id.")
    ] = None,
    study: Annotated[str | None, typer.Option(help="Report on this study instead.")] = None,
    root: Root = None,
    out: Annotated[Path | None, typer.Option(help="Where to write the Markdown.")] = None,
    no_llm: NoLLM = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Write a narrative report of a run or a study."""
    from apex.agents.reporter import Reporter
    from apex.services.api import ApexService

    if (target is None) == (study is None):
        raise _fail(ValueError("give a run, or --study NAME"))
    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 4
    )
    reporter = Reporter(service, chosen, settings=settings)
    try:
        if target is not None and _is_optimization(service, target):
            from apex.agents.optimization import optimization_report

            result = optimization_report(service, chosen, target)
            default = result.trace.parent.parent / "report" / "narrative.md"
        elif study is not None:
            result = reporter.study_report(study)
            default = service.root / "studies" / study / "narrative.md"
        else:
            assert target is not None
            result = reporter.run_report(target)
            default = result.trace.parent.parent / "report" / "narrative.md"
    except ApexError as error:
        raise _fail(error) from error
    path = out or default
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.text)
    typer.echo(f"written: {path}")
    _footer(result.decided_by, result.reason, result.corrections, result.trace)


@agent_app.command("plot")
def plot(
    target: Annotated[str, typer.Argument(help="A run directory, a run id or a job id.")],
    request: Annotated[
        str | None, typer.Option(help="What to draw, in words (default: the classic set).")
    ] = None,
    preset: Annotated[
        str | None, typer.Option(help="Draw this preset (e.g. brugge_classic); no model.")
    ] = None,
    with_: Annotated[
        list[str] | None,
        typer.Option(
            "--with",
            help="Switch on a preset's option, e.g. --with model_change "
            "--with normalized (brugge_classic).",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Output folder (default <run>/report/plots).")
    ] = None,
    root: Root = None,
    no_llm: NoLLM = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Draw a run's result figures: a preset, or what --request asks for."""
    from apex.agents.plotter import Plotter
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = (None, None) if preset is not None else _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 4
    )  # fmt: skip
    try:
        outcome = Plotter(service, chosen, loop_settings=settings).plot(
            target,
            request=request,
            preset=preset,
            preset_options=dict.fromkeys(with_ or [], True),
            output_dir=output,
        )
    except ApexError as error:
        raise _fail(error) from error
    result = outcome.result
    typer.echo(f"written: {len(result.files)} figures (PNG and PDF) in {result.output}")
    for drawn in result.drawn:
        if drawn.skipped:
            typer.secho(f"skipped {drawn.figure} ({drawn.subdir}): {drawn.skipped}",
                        fg=typer.colors.YELLOW)  # fmt: skip
    if outcome.trace is not None:
        _footer(outcome.decided_by, outcome.reason, (), outcome.trace)


convert_app = typer.Typer(
    no_args_is_help=True,
    help="The case-converter: a simulator's case as an APEX case, and back.",
)
agent_app.add_typer(convert_app, name="convert")


@convert_app.command("import")
def convert_import(
    case: Annotated[Path, typer.Argument(help="The case, e.g. an OPM Flow or Eclipse .DATA deck.")],
    output: Annotated[Path, typer.Option("--out", help="A new folder for the APEX case.")],
    history: Annotated[
        Path | None,
        typer.Option(
            help="Observed data: a CSV (time,datatype,value) or a summary file "
            "(default: the deck's WCONHIST/WCONINJH)."
        ),
    ] = None,
    request: Annotated[
        str | None, typer.Option(help="What to match and how, in words (for the model).")
    ] = None,
    name: Annotated[str | None, typer.Option(help="The run's name (default: the case's).")] = None,
    skill: Annotated[
        str | None, typer.Option(help="The converter skill (default: the one that reads it).")
    ] = None,
    root: Root = None,
    no_llm: NoLLM = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
    json_output: JsonOutput = False,
) -> None:
    """Convert a case into an APEX history-matching case that validates; nothing runs."""
    from apex.agents.converter import CaseConverter
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 4
    )
    try:
        outcome = CaseConverter(service, chosen, loop_settings=settings).convert(
            case, output, history=history, request=request, name=name, skill=skill
        )
    except ApexError as error:
        raise _fail(error) from error
    if json_output:
        typer.echo(json.dumps({
            "output": str(outcome.output), "skill": outcome.skill, "valid": outcome.valid,
            "errors": outcome.validation.errors, "choices": outcome.choices.model_dump(mode="json"),
            "notes": outcome.notes, "decided_by": outcome.decided_by,
        }, indent=2))  # fmt: skip
    else:
        choices = outcome.choices
        typer.echo(f"written: {outcome.output}/config.toml (skill {outcome.skill})")
        typer.echo(
            "state: " + ", ".join(f"{v.property} ({v.transform})" for v in choices.variables)
        )
        typer.echo(f"observations: {len(choices.observations)} vectors; method {choices.method}, "
                   f"{choices.ensemble_size} members")  # fmt: skip
        typer.echo(f"notes and assumed values: {outcome.output}/CONVERSION.md")
        plan = outcome.validation.plan
        if outcome.valid:
            typer.secho("valid", fg=typer.colors.GREEN)
            if plan is not None:
                typer.echo(f"plan: {plan.model_dump_json()}")
        else:
            for problem in outcome.validation.errors:
                typer.secho(f"invalid: {problem}", fg=typer.colors.RED)
    if outcome.traces:
        _footer(outcome.decided_by, outcome.reason, (), outcome.traces[-1])
    if not outcome.valid:
        raise typer.Exit(1)


Yes = Annotated[bool, typer.Option("--yes", help="Run the truth without asking.")]


def _ask(question: str) -> bool:
    """Ask the person at the terminal; no answer (e.g. under nohup) is a no."""
    try:
        return typer.confirm(question, default=False, err=True)
    except typer.Abort:
        typer.secho("no answer: nothing is run", err=True, fg=typer.colors.YELLOW)
        return False


def _created(outcome: "CreateOutcome") -> None:
    typer.echo(f"written: {outcome.output}/config.toml ({outcome.spec.name})")
    for note in outcome.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    if outcome.valid:
        typer.secho("valid", fg=typer.colors.GREEN)
    else:
        for problem in outcome.validation.errors:
            typer.secho(f"not valid yet: {problem}", fg=typer.colors.RED)


@convert_app.command("create")
def convert_create(
    output: Annotated[Path, typer.Option("--out", help="A new folder for the synthetic case.")],
    request: Annotated[
        str | None, typer.Option(help="The case in words (default: a five-spot).")
    ] = None,
    name: Annotated[str | None, typer.Option(help="The case's name.")] = None,
    skill: Annotated[
        str | None, typer.Option(help="The simulator's skill (default opm_flow).")
    ] = None,
    observe: Annotated[
        bool, typer.Option(help="Run the truth once now to make the observations.")
    ] = True,
    yes: Yes = False,
    root: Root = None,
    no_llm: NoLLM = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Create a synthetic case: deck, truth, prior, and observations from one truth run."""
    from apex.agents.converter import CaseConverter
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 4
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise _fail(ValueError(f"{output} is not an empty folder: give a new one"))
    simulate = observe and (yes or _ask(
        "This runs the truth once on the case's simulator, to make its observations. Continue?"
    ))  # fmt: skip
    try:
        outcome = CaseConverter(service, chosen, loop_settings=settings).create(
            output, request=request, name=name, skill=skill, simulate=simulate
        )
    except ApexError as error:
        raise _fail(error) from error
    _created(outcome)
    if outcome.trace is not None:
        _footer(outcome.decided_by, outcome.reason, (), outcome.trace)
    if simulate and not outcome.valid:
        raise typer.Exit(1)


@convert_app.command("observe")
def convert_observe(
    case: Annotated[Path, typer.Argument(help="A case `convert create` wrote.")],
    yes: Yes = False,
    root: Root = None,
) -> None:
    """Run a created case's truth once, and write its observations."""
    from apex.agents.converter import CaseConverter
    from apex.services.api import ApexService

    if not yes and not _ask("This runs the truth once on the case's simulator. Continue?"):
        raise typer.Exit(1)
    try:
        outcome = CaseConverter(ApexService(root), None).observe(case)
    except ApexError as error:
        raise _fail(error) from error
    _created(outcome)
    if not outcome.valid:
        raise typer.Exit(1)


def _translate(skill: str, direction: str, source: Path, output: Path, root: Path | None) -> None:
    from apex.agents.converter import CaseConverter
    from apex.services.api import ApexService

    try:
        result = CaseConverter(ApexService(root), None).translate(
            skill, "to" if direction == "to" else "from", source, output
        )
    except ApexError as error:
        raise _fail(error) from error
    typer.echo(f"written: {result.path}")
    for note in result.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)


@convert_app.command("to")
def convert_to(
    skill: Annotated[
        str,
        typer.Argument(help="The converter skill, e.g. pet, ert (`apex agent convert skills`)."),
    ],
    source: Annotated[
        Path,
        typer.Argument(help="The source, in what the skill takes (`apex agent convert skills`)."),
    ],
    output: Annotated[Path, typer.Argument(help="A new folder for the translated case.")],
    root: Root = None,
) -> None:
    """Translate a case into a skill's format (see `apex agent convert skills`); nothing runs."""
    _translate(skill, "to", source, output, root)


@convert_app.command("from")
def convert_from(
    skill: Annotated[
        str,
        typer.Argument(help="The converter skill, e.g. pet, ert (`apex agent convert skills`)."),
    ],
    source: Annotated[Path, typer.Argument(help="The case in the skill's format.")],
    output: Annotated[Path, typer.Argument(help="A new folder for the translated case.")],
    root: Root = None,
) -> None:
    """Translate a case from a skill's format; nothing runs."""
    _translate(skill, "from", source, output, root)


@convert_app.command("to-pet")
def convert_to_pet(
    config: Annotated[Path, typer.Argument(help="The APEX configuration (.toml).")],
    output: Annotated[Path, typer.Argument(help="A new folder for the PET case.")],
    root: Root = None,
) -> None:
    """Write an APEX case as a PET case (pet.toml, data, runfile, prior); nothing runs."""
    _translate("pet", "to", config, output, root)


@convert_app.command("from-pet")
def convert_from_pet(
    pet_config: Annotated[Path, typer.Argument(help="The PET configuration (.toml).")],
    output: Annotated[Path, typer.Argument(help="A new folder for the APEX case.")],
    root: Root = None,
) -> None:
    """Write a PET case as an APEX case (as apex pet-import); nothing runs."""
    _translate("pet", "from", pet_config, output, root)


@convert_app.command("skills")
def convert_skills() -> None:
    """List the converter skills and what they can do."""
    import apex.services.composition  # noqa: F401  (registers core's skills)
    from apex.core.plugins import converter_skills

    for name in converter_skills.names():
        skill = converter_skills.create(name)
        doc = (type(skill).__doc__ or "").replace("``", "").strip().splitlines()
        capabilities = ", ".join(sorted(skill.capabilities))
        typer.echo(f"{name:<12} {doc[0] if doc else ''} [{capabilities}]")


@agent_app.command("geomodel")
def geomodel(
    config: Annotated[Path, typer.Argument(help="The APEX configuration (.toml).")],
    output: Annotated[
        Path | None,
        typer.Option("--out", help="The new configuration (default <config>_prior.toml)."),
    ] = None,
    request: Annotated[
        str | None, typer.Option(help="The prior you want, in words (for the model).")
    ] = None,
    reference: Annotated[
        Path | None,
        typer.Option(
            help="An .npz of reference fields in state space (e.g. the deck's values): the "
            "prior shares their mean, std and variogram, without reproducing them."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(
            help="Hard data to estimate from and condition to: a CSV of i,j,k,property,value "
            "(1-based cells; values in the deck's units; property facies: codes)."
        ),
    ] = None,
    root: Root = None,
    no_llm: NoLLM = False,
    provider: Provider = None,
    model: Model = None,
    base_url: BaseURL = None,
    api_key_env: KeyEnv = None,
    no_tool_calls: NoToolCalls = False,
) -> None:
    """Give a case's state variables a prior, checked on a drawn sample."""
    from apex.agents.geomodeler import Geomodeler
    from apex.services.api import ApexService

    service = ApexService(root)
    chosen, settings = _provider_and_settings(
        service, no_llm, provider, model, base_url, api_key_env, no_tool_calls, 4
    )
    try:
        outcome = Geomodeler(service, chosen, loop_settings=settings).model(
            config, output, request=request, reference=reference, data=data
        )
    except ApexError as error:
        raise _fail(error) from error
    typer.echo(f"written: {outcome.config} (skill {outcome.plan.skill})")
    for note in outcome.notes:
        typer.secho(f"note: {note}", fg=typer.colors.YELLOW)
    for problem in [*outcome.validation.errors, *outcome.qc]:
        typer.secho(f"problem: {problem}", fg=typer.colors.RED)
    if outcome.valid:
        typer.secho("valid, and the drawn sample matches", fg=typer.colors.GREEN)
    if outcome.traces:
        _footer(outcome.decided_by, outcome.reason, (), outcome.traces[-1])
    if not outcome.valid:
        raise typer.Exit(1)


@agent_app.command("providers")
def providers() -> None:
    """List the LLM providers."""
    from apex.agents.providers import DESCRIPTIONS  # also registers them
    from apex.core.plugins import llm_providers

    for name in llm_providers.names():
        doc = (llm_providers.factory(name).__doc__ or "").strip().splitlines()
        typer.echo(f"{name:<18} {DESCRIPTIONS.get(name, doc[0] if doc else '')}")


def register(app: typer.Typer) -> None:
    """Add ``apex agent`` to the CLI."""
    app.add_typer(agent_app, name="agent")
