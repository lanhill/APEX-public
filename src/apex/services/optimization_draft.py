"""An optimization drafted from a history-matching case (part B).

The history-matched model is optimized over a forecast after its history (the user's
choice, B2):

- **The deck.** A copy of the case's deck template continues past the history: at each of
  ``control_periods`` periods up to ``forecast_end``, ``WCONPROD`` (producers on bottom-hole
  pressure, or oil rate) and ``WCONINJE`` (injectors on their history's type and rate) take
  the period's control, then ``DATES`` the period's end. Its ``${case_dir}`` includes point
  at the case's folder. The case's own files are not changed.
- **Controls.** One block per active well (shut or idle wells are left as the history left
  them), a value per period; bounds from the history (rates from 0 to twice the largest) or
  given (pressures, which a history of rates does not have).
- **The objective.** ``npv`` from the deck's ``START``, counted from the history's last
  date (``from_time``), with the prices the user gave: never invented; what was assumed
  (an unpriced gas, the scaling) is said in the draft's header.
- **The model.** Robust optimization over ``members`` members of a finished run of the case:
  its posterior's state arrays as ``[model] realizations_file``, and the case's
  ``[[parameterizations]]``. Never a synthetic case's truth.

The draft is written (deck, model, configuration) into ``out_dir`` and validated; a draft
that does not validate is removed and refused with the service's errors.
"""

import datetime as dt
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from apex.core.errors import ConfigError
from apex.domain.config import HistoryMatchConfig
from apex.domain.services import Validation
from apex.simulation.deck import DECK_SIMULATORS, case_folder, schedule_summary

if TYPE_CHECKING:
    from apex.core.storage import CatalogueEntry
    from apex.services.api import ApexService

#: Where a finished run keeps its posterior's state arrays (``(n_values, n_members)``).
POSTERIOR = Path("checkpoints") / "posterior.npz"
#: The scaling assumed when none is given: NPV in millions, maximised (PET's tutorial).
DEFAULT_SCALING = -1.0e6


class DraftRequest(BaseModel):
    """What an optimization draft is made of (the assistant's or the CLI's arguments)."""

    model_config = ConfigDict(extra="forbid")

    case: str = Field(description="The history-matching case's configuration.")
    forecast_end: str = Field(description="The forecast's last date (ISO), after the history.")
    control_periods: int = Field(default=4, ge=1, le=60, description="Control periods.")
    wop: float | None = Field(default=None, description="Oil price per unit of FOPT.")
    wgp: float | None = Field(default=None, description="Gas price per unit of FGPT.")
    wwp: float | None = Field(default=None, description="Cost per unit of produced water.")
    wwi: float | None = Field(default=None, description="Cost per unit of injected water.")
    disc: float | None = Field(default=None, description="Yearly discount rate.")
    obj_scaling: float | None = Field(
        default=None, description="Divides NPV (default -1e6: millions, maximised)."
    )
    producer_control: Literal["bhp", "oil_rate"] = "bhp"
    producer_bhp: tuple[float, float] | None = Field(
        default=None, description="Producers' bottom-hole pressure bounds (with bhp)."
    )
    run: str | None = Field(default=None, description="The run whose posterior is the model.")
    members: int = Field(default=10, ge=1, le=200)
    optimizer: str = "enopt"
    optimizer_options: dict[str, Any] | None = None
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")


class DraftRefused(ConfigError):
    """A draft that cannot be made as asked; ``missing`` lists what the user must give."""

    def __init__(self, message: str, missing: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing = missing or []


@dataclass
class OptimizationDraft:
    """A written, validated optimization draft."""

    config: Path
    deck: Path
    model: Path
    validation: Validation
    assumptions: list[str] = field(default_factory=list)
    wells: dict[str, str] = field(default_factory=dict)


def _identifier(prefix: str, well: str) -> str:
    """Return a Mako-safe control name for a well: ``bhp_PROD1``, ``rate_INJ_1``."""
    return f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', well)}"


def _period_ends(start: dt.date, end: dt.date, periods: int) -> list[dt.date]:
    """Return ``periods`` dates splitting ``(start, end]`` evenly by days (distinct)."""
    days = (end - start).days
    if days < periods:
        raise DraftRefused(f"the forecast has {days} days, fewer than its {periods} periods")
    return [start + dt.timedelta(days=round(days * k / periods)) for k in range(1, periods + 1)]


def _eclipse(date: dt.date) -> str:
    return f"{date.day} '{date.strftime('%b').upper()}' {date.year}"


def _posterior(
    service: "ApexService",
    config: HistoryMatchConfig,
    run: str | None,
    visible: "Callable[[CatalogueEntry], bool] | None",
) -> Path:
    """Return the run directory whose posterior is the model (see the module docstring).

    With ``visible`` (a web-UI user's view), only catalogued runs it shows are
    candidates: never the case's own ``output_dir`` nor a path, which may be another
    user's run.
    """
    candidates: list[Path] = []
    entries = [e for e in service.store.entries() if visible is None or visible(e)]
    if run is not None:
        found = [e for e in entries if run in (e.run_id, e.name)]
        if found:
            candidates = [Path(e.path) for e in sorted(found, key=lambda e: e.created,
                                                       reverse=True)]  # fmt: skip
        elif visible is None:
            candidates = [Path(run)]
    else:
        if config.run.output_dir is not None and visible is None:
            candidates.append(config.run.output_dir)
        named = [e for e in entries if e.name == config.run.name]
        candidates += [Path(e.path) for e in sorted(named, key=lambda e: e.created, reverse=True)]
    for path in candidates:
        if (path / POSTERIOR).is_file():
            return path
    tried = ", ".join(p.name for p in candidates) or "none"
    raise DraftRefused(
        f"no finished run of {config.run.name!r} with a posterior (tried: {tried}); run the "
        "history match first, or name a run"
    )


def draft_optimization(
    service: "ApexService",
    request: DraftRequest,
    out_dir: Path,
    visible: "Callable[[CatalogueEntry], bool] | None" = None,
) -> OptimizationDraft:
    """Write and validate an optimization drafted from a history-matching case.

    ``visible``, a web-UI user's view of the run catalogue, limits the runs whose posterior
    may be the model to theirs.

    Raises
    ------
    DraftRefused
        With the reason (and ``missing``: what the user must give) when the case cannot
        support it, or the draft does not validate.
    """
    case = Path(request.case)
    config = service.resolve(case)
    simulator = config.simulator
    template_value = simulator.options.get("deck_template_file") if simulator else None
    if simulator is None or simulator.name not in DECK_SIMULATORS or not template_value:
        raise DraftRefused(
            "the case has no deck simulator (opm_flow, eclipse): an optimization drafted from "
            "a case runs its deck"
        )
    template = Path(str(template_value))
    includes = case_folder(simulator.options) or template.resolve().parent
    summary = schedule_summary(template, includes)
    start, dates = summary["start"], summary["dates"]
    if not isinstance(start, dt.date) or not isinstance(dates, list) or not dates:
        raise DraftRefused("the deck's START and DATES cannot be read: no forecast can follow it")
    if summary["tstep"]:
        # TSTEP advances the schedule past the last DATES, so its end is not known here.
        raise DraftRefused("the deck's schedule uses TSTEP: its end cannot be read from DATES")
    if dates != sorted(dates):
        raise DraftRefused("the deck's DATES (in the order its files are read) are not in order")
    if summary["unreadable"]:
        raise DraftRefused(
            "the controls of " + ", ".join(cast(list[str], summary["unreadable"]))
            + " are template expressions: their history cannot be read"
        )  # fmt: skip
    history_end: dt.date = max(dates)
    forecast_end = dt.date.fromisoformat(request.forecast_end)
    if forecast_end <= history_end:
        raise DraftRefused(f"forecast_end must be after the history's end, {history_end}")

    # Economics: never invented. The gas price and the scaling may be assumed, and say so.
    missing = [k for k in ("wop", "wwp", "wwi", "disc") if getattr(request, k) is None]
    if request.producer_control == "bhp" and request.producer_bhp is None:
        missing.append("producer_bhp (the producers' pressure bounds: the history has none)")
    if missing:
        raise DraftRefused(
            "give " + ", ".join(missing) + " (ask the user; do not invent them)", missing
        )
    assumptions: list[str] = []
    wgp = request.wgp if request.wgp is not None else 0.0
    if request.wgp is None:
        assumptions.append("gas is not priced (wgp = 0)")
    scaling = request.obj_scaling if request.obj_scaling is not None else DEFAULT_SCALING
    if request.obj_scaling is None:
        assumptions.append("NPV is scaled to millions and maximised (obj_scaling = -1e6)")

    producers = summary["producers"]
    injectors = summary["injectors"]
    shut = set(summary["shut"]) if isinstance(summary["shut"], list) else set()
    assert isinstance(producers, dict)
    assert isinstance(injectors, dict)
    active_producers = {n: i for n, i in producers.items() if n not in shut}
    active_injectors = {n: i for n, i in injectors.items() if n not in shut}
    # Only water injection is priced (FWIT): other injectors keep their last history control.
    for name in [n for n, i in active_injectors.items() if i["type"] != "WATER"]:
        del active_injectors[name]
        assumptions.append(f"{name} ({injectors[name]['type']} injector, not priced) keeps its "
                           "last history control")  # fmt: skip
    if not active_producers:
        raise DraftRefused("the deck has no active producer to control")
    ends = _period_ends(history_end, forecast_end, request.control_periods)
    controls: list[dict[str, Any]] = []
    wells: dict[str, str] = {}
    for name, info in active_producers.items():
        if request.producer_control == "bhp":
            assert request.producer_bhp is not None
            low, high = request.producer_bhp
            if not low < high:
                raise DraftRefused("producer_bhp must be (lower, upper) with lower < upper")
            variable, initial = _identifier("bhp", name), (low + high) / 2
        else:
            if float(info["max_oil_rate"]) <= 0.0:
                raise DraftRefused(f"{name} has no oil rate in its history to bound an oil "
                                   "rate control; use producer_control = 'bhp'")  # fmt: skip
            low, high = 0.0, 2.0 * float(info["max_oil_rate"])
            variable, initial = _identifier("orat", name), float(info["max_oil_rate"])
            assumptions.append(f"{name}'s oil rate bounds 0 to twice its history's largest")
        wells[name] = variable
        controls.append({"name": variable, "initial": [initial] * len(ends), "lower": low,
                         "upper": high, "variance": ((high - low) / 5.0) ** 2})  # fmt: skip
    for name, info in active_injectors.items():
        high = 2.0 * float(info["max_rate"])
        variable = _identifier("rate", name)
        wells[name] = variable
        controls.append({"name": variable, "initial": [float(info["max_rate"])] * len(ends),
                         "lower": 0.0, "upper": high, "variance": (high / 5.0) ** 2})  # fmt: skip
        assumptions.append(f"{name}'s injection rate bounds 0 to twice its history's largest")

    # The model: the posterior of a finished run of the case.
    run_dir = _posterior(service, config, request.run, visible)
    deck_text = _forecast_deck(template, includes, out_dir, active_producers, active_injectors,
                               wells, ends, request.producer_control)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    name = _free_name(out_dir, request.name, f"{config.run.name}_npv")
    with np.load(run_dir / POSTERIOR) as stored:
        names = [v.name for v in config.state.variables]
        absent = [n for n in names if n not in stored.files]
        if absent:
            raise DraftRefused(f"run {run_dir.name}'s posterior lacks {absent}")
        n_members = min(request.members, int(stored[names[0]].shape[1]))
        model = out_dir / f"{name}_model.npz"
        arrays = {n: np.ascontiguousarray(stored[n][:, :n_members]) for n in names}
        np.savez(model, **arrays)  # type: ignore[arg-type]

    deck = out_dir / f"{name}_deck.mako"
    deck.write_text(deck_text)
    times = [history_end.isoformat(), *(d.isoformat() for d in ends)]
    from apex.algorithms.optimization.population import DRAFT_OPTIONS, is_population_optimizer

    # A population optimizer has no ensemble or step: its own defaults.
    defaults = (dict(DRAFT_OPTIONS) if is_population_optimizer(request.optimizer)
                else {"ne": 10, "maxiter": 5, "alpha": 0.2, "transform": True})  # fmt: skip
    options = request.optimizer_options or defaults
    # The draft's deck lies in out_dir and reaches the case's includes from there (no case_dir).
    kept = {k: v for k, v in simulator.options.items() if k != "case_dir"}
    simulator_options = {**_relative(kept, out_dir), "deck_template_file": deck.name}
    document: dict[str, Any] = {
        "optimization": {"name": name, "seed": config.run.seed},
        "controls": {"realizations": n_members, "variables": controls},
        "objective": {"name": "npv", "options": {
            "start": start.isoformat(), "times": times, "from_time": times[0],
            "wop": request.wop, "wgp": wgp, "wwp": request.wwp, "wwi": request.wwi,
            "disc": request.disc, "obj_scaling": scaling,
        }},
        "optimizer": {"name": request.optimizer, "options": options},
        "simulator": {"name": simulator.name, "options": _jsonable(simulator_options)},
        "execution": {**(executed := config.execution.model_dump(mode="json", exclude_none=True)),
                      "options": _relative(dict(executed.get("options") or {}), out_dir)},
        "model": {"realizations_file": model.name},
        "parameterizations": [p.model_dump(mode="json", exclude_none=True)
                              for p in config.parameterizations],
        "output": {"keep_simulations": "failed"},
    }  # fmt: skip
    if not document["parameterizations"]:
        del document["parameterizations"]
    header = (
        f"# Drafted from {case.name} by APEX's assistant: a forecast from "
        f"{history_end} to {forecast_end} in {len(ends)} control periods,\n"
        f"# robust over {n_members} members of run {run_dir.name}'s posterior. Review it.\n"
        + "".join(f"# Assumed: {a}.\n" for a in assumptions)
        + "\n"
    )
    path = out_dir / f"{name}.toml"
    path.write_text(header + tomli_w.dumps(document), encoding="utf-8")
    validation = service.validate_optimization(path)
    if not validation.valid:
        for written in (path, deck, model):
            written.unlink(missing_ok=True)
        raise DraftRefused("the draft does not validate: " + "; ".join(validation.errors))
    return OptimizationDraft(path, deck, model, validation, assumptions, wells)


def _jsonable(options: dict[str, Any]) -> dict[str, Any]:
    """Options as TOML can hold them (paths as text)."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}


def _relative(options: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Return path options relative to the draft's folder: no server path in a draft."""
    return {
        k: os.path.relpath(v, out_dir) if isinstance(v, Path) or (
            isinstance(v, str) and k.endswith(("_file", "_dir", "_path")) and Path(v).is_absolute()
        ) else v
        for k, v in options.items()
    }  # fmt: skip


def _free_name(out_dir: Path, given: str | None, default: str) -> str:
    """Return the draft's name: ``given`` if free (else refused), or ``default`` made free.

    A second draft in the same chat must not overwrite the first, which a proposal may
    name.
    """

    def taken(name: str) -> bool:
        return any((out_dir / f"{name}{s}").exists() for s in (".toml", "_deck.mako",
                                                                "_model.npz"))  # fmt: skip

    if given is not None:
        if taken(given):
            raise DraftRefused(f"a draft named {given!r} exists already: choose another name")
        return given
    name, k = default, 1
    while taken(name):
        k += 1
        name = f"{default}_{k}"
    return name


FIELD_TOTALS = ("FOPT", "FGPT", "FWPT", "FWIT")


def _with_field_totals(text: str) -> str:
    """Return the deck with the field totals ``npv`` reads added to its SUMMARY section.

    A history-matching deck asks only for what its observations need (5Spots has no
    ``FWIT``), and a summary without a vector fails every member's objective.
    """
    # A section keyword may have text after it on its line ("SUMMARY ===="), as decks
    # write them; simulators read only the keyword.
    section = re.search(r"(?m)^SUMMARY\b[^\n]*$", text)
    if section is None:
        raise DraftRefused("the deck has no SUMMARY section to read the field totals from")
    rest = text[section.end() :]
    following = re.search(r"(?m)^SCHEDULE\b", rest)
    body = rest[: following.start()] if following else rest
    present = {line.split()[0] for line in body.splitlines() if line.strip()}
    missing = [v for v in FIELD_TOTALS if v not in present]
    if not missing:
        return text
    added = "\n-- Added by APEX for the objective.\n" + "\n".join(missing) + "\n"
    return text[: section.end()] + added + rest


def _forecast_deck(
    template: Path,
    includes: Path,
    out_dir: Path,
    producers: dict[str, Any],
    injectors: dict[str, Any],
    wells: dict[str, str],
    ends: list[dt.date],
    producer_control: str,
) -> str:
    """Return the case's deck template continued by the forecast's controlled periods."""
    # The copy renders with ``case_dir`` its own folder, so the case's includes (``includes``:
    # its case_dir, or its template's folder) are reached from there, by a relative path (no
    # server path in the draft).
    to_case = Path(os.path.relpath(includes.resolve(), out_dir.resolve())).as_posix()
    text = _with_field_totals(
        template.read_text(errors="replace").replace("${case_dir}", "${case_dir}/" + to_case)
    )
    lines = [f"-- The forecast drafted by APEX: {len(ends)} control periods."]
    for k, end in enumerate(ends):
        lines.append("WCONPROD")
        for name in producers:
            value = '${"%.6f" % ' + f"{wells[name]}[{k}]" + "}"
            mode = "BHP 5* " if producer_control == "bhp" else "ORAT "
            lines.append(f" '{name}' 'OPEN' {mode}{value} /")
        lines.append("/")
        if injectors:
            lines.append("WCONINJE")
            for name, info in injectors.items():
                value = '${"%.6f" % ' + f"{wells[name]}[{k}]" + "}"
                lines.append(f" '{name}' {info['type']} 'OPEN' RATE {value} /")
            lines.append("/")
        lines += ["DATES", f" {_eclipse(end)} /", "/"]
    block = "\n".join(lines) + "\n"
    end_line = re.search(r"(?m)^END\s*$", text)
    if end_line is None:
        return text.rstrip("\n") + "\n\n" + block + "\nEND\n"
    return text[: end_line.start()] + block + "\n" + text[end_line.start() :]
