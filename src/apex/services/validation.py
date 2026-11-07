"""Validate a configuration and plan what it will cost, without simulating."""

import hashlib
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from apex.algorithms.data_assimilation.gies import UNSCALED_DATA_ANOMALIES_WARNING
from apex.core.errors import ApexError
from apex.core.plugins import priors
from apex.core.storage import RunStore
from apex.domain.config import HistoryMatchConfig
from apex.domain.services import RunPlan, Validation
from apex.execution.resources import ECLIPSE_LICENCES
from apex.services import composition
from apex.services.identity import run_id
from apex.services.plugins import check_options, plugin_sections, unavailable_errors


def case_key(config: HistoryMatchConfig) -> str:
    """Return what makes two runs cost alike: their simulator and observations sections.

    Used to estimate a new run's duration from earlier runs of the same case.
    """
    document = json.loads(config.model_dump_json(include={"simulator", "observations"}))
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]


def _stored_members(path: Path) -> int | None:
    """Members of a stored prior (``.npz``), read from the array headers only."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".npy"))
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                read = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, _, _ = read(handle)
    except (OSError, StopIteration, ValueError, zipfile.BadZipFile):
        return None
    return int(shape[-1]) if shape else None


def _sized_priors(config: HistoryMatchConfig) -> dict[str, int | None]:
    """Members of each variable's generator that gives its own size, in the state's order.

    As ``draw_prior_ensemble``: without [prior] ensemble_size, the generators whose class sets
    ``needs_ensemble_size = False`` (a stored prior, the default generator's or a variable's
    own) give the size; variables drawn by [prior.groups] are not theirs. A size is None
    when it cannot be read from the generator's ``ensemble_file``.
    """
    grouped = {name for group in config.prior.groups.values() for name in group.variables}
    sizes: dict[str, int | None] = {}
    for variable in config.state.variables:
        if variable.name in grouped:
            continue
        section = config.prior.generator_for(variable.name)
        if section.name not in priors.names():
            continue
        if getattr(priors.factory(section.name), "needs_ensemble_size", True):
            continue
        stored = section.options.get("ensemble_file")
        sizes[variable.name] = None if stored is None else _stored_members(Path(str(stored)))
    return sizes


def _plugin_errors(config: HistoryMatchConfig) -> list[str]:
    errors: list[str] = []
    for registry, section in plugin_sections(config):
        errors += unavailable_errors(registry, section.name)
        errors += check_options(registry, section.name, section.options)
    return errors


def validate_history_match(
    config: HistoryMatchConfig | Mapping[str, Any] | Path,
    *,
    base_dir: Path | None = None,
    store: RunStore | None = None,
) -> Validation:
    """Validate a history-matching configuration and plan its cost.

    Parameters
    ----------
    config
        A configuration file, a resolved model, or a mapping (e.g. JSON from a client),
        whose relative paths resolve against ``base_dir`` (default: the working
        directory).
    store
        Earlier runs, to estimate the duration from runs of the same case.
    """
    try:
        if isinstance(config, Path):
            resolved = composition.load_config(config)
        elif isinstance(config, HistoryMatchConfig):
            resolved = config
        else:
            model = HistoryMatchConfig.model_validate(dict(config))
            resolved = composition._resolve_paths(model, (base_dir or Path.cwd()).resolve())
    except ValidationError as error:
        return Validation(
            valid=False,
            errors=[f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()],
        )
    except ApexError as error:
        return Validation(valid=False, errors=[str(error)])
    errors = _plugin_errors(resolved)
    if resolved.surrogate is not None:  # "Refused": said first, whatever else is wrong
        from apex.services import surrogates

        errors = surrogates.refusals(resolved) + errors
    if errors:
        return Validation(valid=False, errors=errors)
    try:
        run = composition.build_run(resolved)
    except (ApexError, OSError, ValueError) as error:
        return Validation(valid=False, errors=[str(error)])
    if resolved.prior.ensemble_size is None:
        sizes = {k: n for k, n in _sized_priors(resolved).items() if n is not None}
        if len(set(sizes.values())) > 1:
            return Validation(
                valid=False, errors=[f"prior variables have different ensemble sizes: {sizes}"]
            )
    planned = plan(resolved, run, store)
    section = resolved.surrogate
    if (
        section is not None
        and planned.surrogate_evaluations is not None
        and planned.surrogate_evaluations > section.max_evaluations
    ):
        return Validation(
            valid=False,
            errors=[
                f"[surrogate]: the run may take {planned.surrogate_evaluations:,} surrogate "
                f"evaluations, more than max_evaluations = {section.max_evaluations:,}"
            ],
            plan=planned,
        )
    return Validation(valid=True, plan=planned)


def plan(config: HistoryMatchConfig, run: "composition.Run", store: RunStore | None) -> RunPlan:
    """Return the cost of a built run (see :class:`~apex.domain.services.RunPlan`)."""
    warnings: list[str] = []
    members = config.prior.ensemble_size
    generator = config.prior.generator
    if members is None:
        members = next(iter(_sized_priors(config).values()), None)
    if generator.name == "file" and not config.prior.variables:
        warnings.append(
            "the prior is stored (file): runs with another seed share it and differ only in "
            "their random perturbations"
        )
    # The method's own strategy: CHOP (which delegates .strategy) refuses unscaled anomalies,
    # and no other wrapper holds an RLMMACUpdate; a new one should be checked here too.
    if getattr(getattr(run.method, "strategy", None), "scale_data_anomalies", True) is False:
        warnings.append(UNSCALED_DATA_ANOMALIES_WARNING)
    workflow = run.workflow
    data = {"history_matching": workflow.observations.n_data}
    data.update({role: o.n_data for role, o in workflow.held_out_observations.items()})
    budget = getattr(run.method, "max_forecasts", None)
    ensemble, single = budget() if budget is not None else (None, None)
    runs = None if ensemble is None or members is None else ensemble * members + (single or 0)
    evaluations, surrogate = None, None
    if config.surrogate is not None:
        # The forecasts are the surrogate's; only the simulator's runs count.
        from apex.services import surrogates

        evaluations = surrogates.planned_evaluations(
            config.surrogate, ensemble, single, members,
            iterative=bool(getattr(run.method, "iterative", False)),
        )  # fmt: skip
        runs, notes = surrogates.planned_simulator_runs(config.surrogate, ensemble, members)
        warnings += notes
        forecasting = run.workflow.surrogate
        if forecasting is not None and forecasting.coarse is not None:
            # Every surrogate evaluation runs its member's coarse stack first.
            if runs is not None and evaluations is not None:
                runs += evaluations
            warnings.append(
                "the surrogate corrects a coarse simulation: each of its evaluations is also "
                "a run of the coarse stack, counted with the simulator runs"
            )
        if forecasting is not None and forecasting.early_until is not None:
            # Every surrogate evaluation first runs its member up to the
            # model's early date.
            if runs is not None and evaluations is not None:
                runs += evaluations
            warnings.append(
                "the surrogate continues each member's run from "
                f"{forecasting.early_until}: each of its evaluations is also a simulator run "
                "up to that date, counted with the simulator runs (each a share of a full run; "
                "members outside the domain or whose early run fails make one too, uncounted)"
            )
        if forecasting is not None:
            original = forecasting.original
            surrogate = f"{original.manifest.plugin} {original.identity[:12]}"
    options = config.simulator.options
    licences = options.get("licenses") if config.simulator.name == "eclipse" else None
    if licences is not None and int(licences) > ECLIPSE_LICENCES:
        warnings.append(
            f"licenses = {licences} exceeds the site's {ECLIPSE_LICENCES}: the machine-wide "
            "pool will hold the runs back"
        )
    warnings += run.warnings  # [geophysics]: what the deck's text cannot confirm
    checkpoint_mb, forecast_mb = _sizes(run, members)
    workers = config.execution.options.get("workers")
    parallel = min(int(workers), int(licences)) if workers and licences else workers
    return RunPlan(
        run_id=run_id(config),
        method=config.method.name,
        members=members,
        n_state=run.workflow.layout.n_state,
        data=data,
        max_forecasts=ensemble,
        max_simulator_runs=runs,
        simulator=config.simulator.name,
        licences=None if licences is None else int(licences),
        parallel_runs=None if parallel is None else int(parallel),
        estimated_hours=_estimate_hours(case_key(config), ensemble, store),
        checkpoint_mb=checkpoint_mb,
        forecast_memory_mb=forecast_mb,
        surrogate_evaluations=evaluations,
        surrogate=surrogate,
        warnings=warnings,
    )


def _sizes(run: "composition.Run", members: int | None) -> tuple[float | None, float | None]:
    """Return about how large a checkpointed iterate and one forecast are (MB, float64).

    From the data's counts. An iterate holds the ensemble, its forecast of
    every observed datum (transformed, held-out ones included), the simulator's auxiliary
    outputs (e.g. model-error residuals), the raw forecasts with ``[output]
    raw_forecasts``, and for a fixed schedule the step's perturbed observations (an
    iterative method saves them once). A forecast holds the members' raw outputs while they
    are stacked (twice the raw forecasts) and, with transforms, the transformed copy.
    Method-specific extras (e.g. GIES's mean-model forecast) are not counted.
    """
    if members is None:
        return None, None
    workflow = run.workflow
    n_raw = workflow.raw_observations.n_data
    n_data = workflow.observations.n_data + sum(
        o.n_data for o in workflow.held_out_observations.values()
    )
    perturbed = 0 if getattr(run.method, "iterative", False) else workflow.observations.n_data
    raw_kept = n_raw if workflow.save_raw_forecasts else 0
    rows = workflow.layout.n_state + n_data + len(workflow.auxiliary_keys) + raw_kept + perturbed
    forecast = 2 * n_raw + (n_data if workflow.transforms else 0)
    mb = 8.0 * members / 1e6
    return round(rows * mb, 1), round(forecast * mb, 1)


def _estimate_hours(key: str, ensemble: int | None, store: RunStore | None) -> float | None:
    """Mean duration of an ensemble forecast in earlier runs of the case, times the budget."""
    if store is None or ensemble is None:
        return None
    seconds: list[float] = []
    for entry in store.entries():
        if entry.annotations.get("case_key") != key:
            continue
        events = entry.path / "events.jsonl"
        if not events.exists():
            continue
        for line in events.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "forecast_done" and record.get("label") == "ensemble":
                seconds.append(float(record["seconds"]))
    if not seconds:
        return None
    return float(np.mean(seconds)) * ensemble / 3600.0
