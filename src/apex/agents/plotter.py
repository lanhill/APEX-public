"""The plotter agent: a run's result figures, chosen from a request.

1. **Facts from code** (:func:`plot_facts`): what the run holds (its metrics and labels,
   variables with grids, data types, auxiliary forecasts, iterations, truth), and every
   figure plugin's and preset's option schema with whether it applies to this run.
2. **The model chooses** a :class:`PlotPlan`: a preset and/or a list of figures with their
   options. The plan is validated against the run before it is accepted (unknown figures,
   options the plugin refuses, figures that do not apply, more than
   ``[agents.plotter] max_figures``); a refused plan goes back to the model with the
   reasons, and the rules decide if it cannot repair it.
3. **The rules** (:func:`rules`) match the request's words onto the figures and presets;
   with no request, or none matched, they draw the ``brugge_classic`` preset.
4. **Code draws** (:func:`apex.services.figures.draw`) into ``<run>/report/plots`` by
   default, where the run's page in the web UI shows them, with ``summary.json``.

The plotter reads the run and writes only figures.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from apex.agents.base import AgentContext
from apex.agents.common import decide
from apex.agents.loop import LoopSettings
from apex.agents.monitor import RunMonitor
from apex.core.errors import ConfigError
from apex.core.figures import FigureRequest, RunView
from apex.core.llm import LLMProvider
from apex.core.plugins import figure_presets, figures
from apex.services.api import ApexService
from apex.services.figure_view import StoredRunView, run_view
from apex.services.figures import (
    DrawResult,
    check_request,
    draw,
    preset_requests,
    start_drawing,
)
from apex.services.settings import PlotterSection

NAME = "plotter"
#: The preset the rules draw when the request names nothing more specific.
DEFAULT_PRESET = "brugge_classic"
#: At most this many labels per metric in the facts (Brugge has 70 data types).
FACT_LABELS = 12

SYSTEM = (
    "You are APEX's plotter. You choose which figures to draw of a history-matching run, "
    "from the figure plugins and presets in the facts, for the user's request. Answer with "
    "a plot plan: a preset (e.g. brugge_classic, the whole classic set) and/or a list of "
    "figures, each with options from its schema and a subfolder. Use only the figures, "
    "options, metrics, labels and variables the facts list, and only figures whose "
    "'applies' is null (the others cannot be drawn for this run). Draw what the request "
    "asks for, no more; with no specific request, choose the preset. The data mismatch "
    "is data_mismatch (the objective); draw normalized_data_mismatch only when the request "
    "names it (the preset's option normalized), and model_change only when the request asks "
    "for the model change (the preset's option model_change). Give a one-sentence reason."
)


class FigureChoice(BaseModel):
    """One figure plugin to draw."""

    model_config = ConfigDict(extra="forbid")

    figure: str = Field(description="A figure plugin's name from the facts.")
    options: dict[str, Any] = Field(
        default_factory=dict, description="Options from the plugin's schema."
    )
    subdir: str = Field(default="", description="Subfolder of the output, e.g. 'rmse'.")


class PlotPlan(BaseModel):
    """What to draw: a preset, figures, or both."""

    model_config = ConfigDict(extra="forbid")

    preset: str | None = Field(default=None, description="A preset's name, or null.")
    preset_options: dict[str, Any] = Field(
        default_factory=dict, description="The preset's options from its schema, if any."
    )
    figures: list[FigureChoice] = Field(default_factory=list)
    reason: str = Field(default="", max_length=500, description="Why, in one sentence.")

    @model_validator(mode="after")
    def _something(self) -> "PlotPlan":
        if self.preset is None and not self.figures:
            raise ValueError("choose a preset or at least one figure")
        if self.preset is None and self.preset_options:
            raise ValueError("preset_options need a preset")
        return self


def plan_model(view: RunView, max_figures: int) -> type[PlotPlan]:
    """Return :class:`PlotPlan` checked against ``view``: what the model's answer must pass.

    A refusal names each problem, so the model can revise its plan.
    """

    class CheckedPlotPlan(PlotPlan):
        @model_validator(mode="after")
        def _against_the_run(self) -> "CheckedPlotPlan":
            problems = plan_problems(self, view, max_figures)
            if problems:
                raise ValueError("; ".join(problems))
            return self

    CheckedPlotPlan.__name__ = "PlotPlan"
    return CheckedPlotPlan


def plan_problems(plan: PlotPlan, view: RunView, max_figures: int) -> list[str]:
    """Return why ``plan`` cannot be drawn on this run (empty: it can)."""
    problems: list[str] = []
    requests = [FigureRequest(c.figure, c.options, c.subdir) for c in plan.figures]
    for request in requests:
        reason = check_request(request, view)
        if reason is not None:
            problems.append(reason)
    if plan.preset is not None:
        try:
            preset = preset_requests(plan.preset, view, plan.preset_options)
        except ConfigError as error:
            problems.append(str(error))
        else:
            # A preset asks for its whole set: what the run cannot give is drawn as skipped,
            # with the reason, so only the options are checked here.
            problems += [r for r in map(check_request, preset) if r is not None]
            requests += preset
    if len(requests) > max_figures:
        problems.append(f"{len(requests)} figure requests: at most {max_figures}")
    return problems


def requests_of(plan: PlotPlan, view: RunView) -> list[FigureRequest]:
    """Return the figure requests a (checked) plan draws."""
    chosen = [FigureRequest(c.figure, c.options, c.subdir) for c in plan.figures]
    if plan.preset is not None:
        chosen += preset_requests(plan.preset, view, plan.preset_options)
    return chosen


def plot_facts(view: RunView) -> dict[str, Any]:
    """Return what the plotter's model sees: the run's contents and the figures' schemas."""
    from apex.services.plugins import plugin_info

    labels = view.metric_labels()
    observations = view.observations()
    datatypes = sorted({k.datatype for k in observations.keys if k.index is None})
    prior = view.iterate(0)
    run = {
        "iterations": len(view.iterations),
        "accepted_iterations": len(view.accepted_iterations),
        "metrics": {
            name: {"labels": found[:FACT_LABELS], "n_labels": len(found)}
            for name, found in labels.items()
        },
        "variables": {
            name: {"grid": None if (g := view.grid(name)) is None else list(g.dims)}
            for name in view.variables()
        },
        "data_types": datatypes[:FACT_LABELS],
        "n_data_types": len(datatypes),
        "auxiliary_forecasts": prior.auxiliary_predicted is not None,
        "truth": view.truth() is not None,
    }
    plugins: dict[str, Any] = {}
    for name in figures.names():
        info = plugin_info("figures", name)
        figure = figures.create(name, **_defaults_for(name))
        plugins[name] = {
            "summary": info.summary,
            "options": info.options.get("properties", {}),
            "required": info.options.get("required", []),
            # With its default options: options may change it (another metric, variable).
            "applies": figure.applies(view),
        }
    presets = {}
    for name in figure_presets.names():
        info = plugin_info("figure_presets", name)
        presets[name] = {"summary": info.summary,
                         "options": info.options.get("properties", {})}  # fmt: skip
    return {"run": run, "figures": plugins, "presets": presets}


def _defaults_for(name: str) -> dict[str, Any]:
    """Return options that let a plugin with required options be created to ask ``applies``."""
    required = {"metric_distribution": {"metric": "data_mismatch"},
                "state_histogram": {"variable": "nn_params"}}  # fmt: skip
    return required.get(name, {})


#: Request words and what they draw: (words, figure, options, subdir).
_KEYWORDS: tuple[tuple[tuple[str, ...], str, dict[str, Any], str], ...] = (
    (("objective", "obj"), "objective_distribution", {}, "obj"),
    (("normalized objective", "normalised objective"), "objective_distribution",
     {"metric": "normalized_data_mismatch"}, "obj_normalized"),
    (("mismatch",), "metric_distribution",
     {"metric": "data_mismatch", "prefix": "data_mismatch"}, "data_mismatch"),
    (("normalized", "normalised"), "metric_distribution",
     {"metric": "normalized_data_mismatch", "prefix": "normalized_data_mismatch"},
     "normalized_data_mismatch"),
    (("rmse",), "metric_distribution", {"metric": "rmse"}, "rmse"),
    (("ssim",), "metric_distribution", {"metric": "ssim", "prefix": "mssim"}, "mssim"),
    (("model change", "model_change", "model-change"), "metric_distribution",
     {"metric": "model_change"}, "model_change"),
    (("production", "forecast", "band", "simdata", "wells", "per well"), "data_bands",
     {"source": "predicted"}, "simData"),
    (("residual", "mec"), "data_bands", {"source": "auxiliary"}, "residual"),
    (("map", "layer", "field"), "field_layers", {}, "models"),
    (("seismic", "avo", "gravity", "displacement", "subsidence", "time shift", "time-shift",
      "timeshift", "4d", "4-d"),
     "field_maps", {}, "geophysics"),
    (("histogram", "hist", "nn_params", "neural"), "state_histogram",
     {"variable": "nn_params"}, "nn_hist"),
)  # fmt: skip


#: Words naming the whole classic set, and the preset's options that words switch on.
_CLASSIC = ("classic", "preset", "brugge_classic", "all figures", "every figure", "full set",
            "whole set")  # fmt: skip
_PRESET_EXTRAS = (("normalized", ("normalized", "normalised")),
                  ("model_change", ("model change", "model_change", "model-change")))  # fmt: skip


def _mentions(text: str, words: tuple[str, ...]) -> bool:
    """Whether ``text`` has one of ``words`` as a whole word, or its plural.

    Whole words: "hist" is not "history", "map" not "mapping", "band" not "bandwidth".
    """
    return any(re.search(rf"\b{re.escape(word)}(?:s|es)?\b", text) for word in words)


def rules(request: str | None, view: RunView) -> PlotPlan:
    """Return the rules' plan: the request's keywords, else the default preset.

    A matched figure that does not apply to the run is left out; if none is left, the
    preset is drawn.
    """
    text = f" {(request or '').lower()} "
    if _mentions(text, _CLASSIC):
        # The whole set, with the groups drawn only on request when the words name them.
        options = {name: True for name, words in _PRESET_EXTRAS if _mentions(text, words)}
        reason = f"rules: the classic set ({DEFAULT_PRESET}) the request names"
        return PlotPlan(preset=DEFAULT_PRESET, preset_options=options, reason=reason)
    chosen: list[FigureChoice] = []
    left_out: list[str] = []  # named but not drawable, with why: said in the reason
    for words, figure, options, subdir in _KEYWORDS:
        if not _mentions(text, words):
            continue
        request_ = FigureRequest(figure, options, subdir)
        problem = check_request(request_, view)
        if problem is None:
            chosen.append(FigureChoice(figure=figure, options=options, subdir=subdir))
        else:
            # check_request may name the figure already ("field_maps: ...").
            said = problem if problem.startswith(f"{figure}:") else f"{figure}: {problem}"
            if said not in left_out:
                left_out.append(said)
    named = re.search(r"\bnormali[sz]ed (?:data )?mismatch", text)
    if named and len(re.findall(r"\bmismatch", text)) == 1:
        # "the normalized data mismatch" names the normalized one only; "the data mismatch
        # and the normalized one" names both.
        chosen = [c for c in chosen if c.subdir != "data_mismatch"]
    if any(c.subdir == "obj_normalized" for c in chosen):
        # "the normalized objective": that one figure, not the objective nor the
        # normalized mismatch per data type as well.
        chosen = [c for c in chosen if c.subdir != "obj"
                  and (c.subdir != "normalized_data_mismatch" or named)]  # fmt: skip
    skipped = f"; left out: {'; '.join(left_out)}" if left_out else ""
    if chosen:
        reason = f"rules: the figures the request's words name{skipped}"
        return PlotPlan(figures=chosen, reason=reason[:500])
    if left_out:
        # The fallback first: the truncation may cut the list, not what is drawn.
        reason = (f"rules: the classic set ({DEFAULT_PRESET}); the figures the request names "
                  f"cannot be drawn for this run{skipped}")  # fmt: skip
        return PlotPlan(preset=DEFAULT_PRESET, reason=reason[:500])
    return PlotPlan(preset=DEFAULT_PRESET, reason="rules: no specific figure asked for; the "
                    f"classic set ({DEFAULT_PRESET})")  # fmt: skip


@dataclass
class PlotChoice:
    """What to draw on a run, and who chose it."""

    run_path: Path
    view: StoredRunView
    plan: PlotPlan
    requests: list[FigureRequest]
    dpi_scale: float
    decided_by: str
    reason: str | None
    trace: Path | None


@dataclass
class PlotOutcome:
    """A plot's figures and who chose them."""

    result: DrawResult
    plan: PlotPlan
    decided_by: str
    reason: str | None
    trace: Path | None


@dataclass
class PlotStarted:
    """A plot drawn in the background: where, how many requests, and who chose them."""

    output: Path
    requests: int
    plan: PlotPlan
    decided_by: str
    reason: str | None
    trace: Path | None


class Plotter:
    """The plotter agent.

    Parameters
    ----------
    service
        The service (to resolve runs and record decisions).
    provider
        The model, or ``None`` for the rules.
    settings
        ``[agents.plotter]`` (default: the service's).
    loop_settings
        The agent loop's settings.
    """

    def __init__(
        self,
        service: ApexService,
        provider: LLMProvider | None,
        *,
        settings: PlotterSection | None = None,
        loop_settings: LoopSettings | None = None,
    ) -> None:
        self.service = service
        self.provider = provider
        self.settings = settings or service.jobs.settings.agents.plotter
        self.loop_settings = loop_settings

    def choose(
        self,
        run: str | Path,
        *,
        request: str | None = None,
        preset: str | None = None,
        preset_options: Mapping[str, Any] | None = None,
    ) -> PlotChoice:
        """Choose a run's figures: ``preset`` as given, else what ``request`` asks for.

        Raises
        ------
        ConfigError
            If the run cannot be read, or the preset is unknown or too large.
        """
        run_path, _ = RunMonitor(self.service, None).resolve(run)
        try:
            view = run_view(run_path)
        except ConfigError as error:  # its message names the server's path
            raise ConfigError(str(error).replace(str(run_path), "<run>")) from None
        supervised = AgentContext(self.service.root, NAME, run_path)
        settings = supervised.thresholds(self.settings)
        if preset is None and preset_options:
            raise ConfigError("preset options need a preset (e.g. brugge_classic)")
        if preset is not None:
            plan = PlotPlan(preset=preset, preset_options=dict(preset_options or {}),
                            reason="the preset asked for")  # fmt: skip
            problems = plan_problems(plan, view, settings.max_figures)
            if problems:
                raise ConfigError("; ".join(problems))
            decided_by, reason, trace = "request", None, None
        else:
            fallback = rules(request, view)
            task = json.dumps({"request": request or "(none: the classic set)",
                               "facts": plot_facts(view),
                               "max_figures": settings.max_figures}, default=str)  # fmt: skip
            decision = decide(
                self.service, self.provider, supervised.recorder, task=task,
                response_model=plan_model(view, settings.max_figures), rules=fallback,
                rule="plot_rules", thresholds=settings.model_dump(mode="json"),
                client="agent:plotter", settings=self.loop_settings, tools=(),
                system=supervised.system(SYSTEM),
            )  # fmt: skip
            plan = PlotPlan.model_validate(decision.value.model_dump())
            decided_by, reason, trace = decision.decided_by, decision.reason, decision.trace
        # The rules' plan is bounded too (advice may lower max_figures).
        requests = requests_of(plan, view)[: settings.max_figures]
        return PlotChoice(run_path, view, plan, requests, settings.dpi_scale, decided_by,
                          reason, trace)  # fmt: skip

    def plot(
        self,
        run: str | Path,
        *,
        request: str | None = None,
        preset: str | None = None,
        preset_options: Mapping[str, Any] | None = None,
        output_dir: Path | None = None,
    ) -> PlotOutcome:
        """Choose and draw a run's figures, waiting for them.

        Raises
        ------
        ConfigError
            If the run cannot be read, or the preset is unknown or too large.
        """
        choice = self.choose(run, request=request, preset=preset, preset_options=preset_options)
        result = draw(choice.run_path, choice.requests, output_dir=output_dir,
                      dpi_scale=choice.dpi_scale, view=choice.view)  # fmt: skip
        return PlotOutcome(result, choice.plan, choice.decided_by, choice.reason, choice.trace)

    def start(
        self,
        run: str | Path,
        *,
        request: str | None = None,
        preset: str | None = None,
        preset_options: Mapping[str, Any] | None = None,
    ) -> PlotStarted:
        """Choose a run's figures, and draw them in a background process (the web UI's).

        Raises
        ------
        ConfigError
            If the run cannot be read, the preset is unknown or too large, or its figures
            are already being drawn.
        """
        choice = self.choose(run, request=request, preset=preset, preset_options=preset_options)
        output = start_drawing(choice.run_path, choice.requests, dpi_scale=choice.dpi_scale,
                               max_drawings=self.settings.max_drawings)  # fmt: skip
        return PlotStarted(output, len(choice.requests), choice.plan, choice.decided_by,
                           choice.reason, choice.trace)  # fmt: skip
