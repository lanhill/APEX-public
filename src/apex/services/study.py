"""Studies: expand, validate, submit and compare.

A study's cells (case x method x variant x replicate) are ordinary run
jobs of the job runner, children of one study job; its predictions are prediction jobs
that wait for their run.
``<root>/studies/<study>/study.json`` records the specification and every cell, for
status and comparisons. Identical runs are shared (content addressing): a cell whose
configuration was already run reuses that run.
"""

import json
import logging
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import TypeAdapter, ValidationError

from apex.algorithms.metrics.comparison import paired, smallest_p_value, summarise
from apex.core.engines import EngineMethod
from apex.core.errors import ConfigError, validation_problems
from apex.domain.config import HistoryMatchConfig
from apex.domain.prediction import PredictionConfig
from apex.domain.services import (
    Comparison,
    JobHandle,
    MetricComparison,
    PairedDifference,
    RunPlan,
    Validation,
    VariantStats,
)
from apex.domain.study import MethodEntry, SkippedCell, StudyCell, StudySpec, merge_patch
from apex.services import composition

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from apex.services.api import ApexService
    from apex.services.workspace import Workspace

#: Decides whether an expensive study may run: gets the total plan, returns yes or no.
ApprovalHook = Callable[["StudyPlan"], bool]


@dataclass(frozen=True)
class PlannedCell:
    """One cell of an expanded study, before submission."""

    case: str
    method: str | None
    variant: str
    replicate: int
    config: HistoryMatchConfig
    engine: EngineMethod | None = None

    @property
    def label(self) -> str:
        """``case/[method/]variant/r<k>``, as errors and status name the cell."""
        return cell_label(self.case, self.method, self.variant, self.replicate)


def cell_label(case: str, method: str | None, variant: str, replicate: int) -> str:
    """Return a cell's label: ``case/method/variant/r<k>``, without the method if none."""
    return "/".join([case, *([method] if method else []), variant, f"r{replicate}"])


#: How a study's refusal over its budget begins, for planners that would say it again.
OVER_BUDGET = "the study may take"


def over_budget(total: int | None, limit: int, costs: Sequence[tuple[str, int, int | None]]) -> str:
    """Return why a study is over its budget, with each arm's cost and what would fit.

    ``costs`` holds ``(arm, replicate, simulator runs)`` per cell (``None``: unbounded), the
    arm being ``case/[method/]variant``. A reported issue (#248): the bare total left the
    assistant to guess which cells to shrink, and it handed the choice back to the user.
    """
    head = (
        f"{OVER_BUDGET} {total if total is not None else 'an unknown number of'} "
        f"simulator runs, over its budget of {limit}"
    )
    unbounded = sorted({arm for arm, _, runs in costs if runs is None})
    if unbounded:
        return f"{head}: the runs of {', '.join(unbounded)} cannot be bounded"
    first = {arm: runs for arm, replicate, runs in costs if replicate == 0 and runs is not None}
    if not first:
        return head
    replicates = max(replicate for _, replicate, _ in costs) + 1
    per_replicate = sum(first.values())
    arms = ", ".join(f"{arm} {runs}" for arm, runs in first.items())
    fits = limit // per_replicate if per_replicate else replicates
    costliest = max(first, key=lambda arm: first[arm])
    shrink = (
        f"lower the cost of the costliest arm, {costliest} ({first[costliest]} runs): "
        "fewer iterations, members or a smaller population"
    )
    if 1 <= fits < replicates:
        plural = "s" if fits > 1 else ""
        advice = f"run {fits} replicate{plural} ({fits * per_replicate} runs), or {shrink}"
    elif fits < 1:
        advice = f"even one replicate is over it; {shrink}, or drop an arm"
    else:  # (the replicates' costs differ: a seed changed a bound)
        advice = shrink
    return (
        f"{head}. Runs per replicate: {arms} ({per_replicate} in all, "
        f"{replicates} replicate{'s' if replicates > 1 else ''}). To fit: {advice}"
    )


@dataclass(frozen=True)
class StudyPlan:
    """An expanded, validated study."""

    spec: StudySpec
    cells: list[PlannedCell]
    validation: Validation
    skipped: tuple[SkippedCell, ...] = ()

    @property
    def max_simulator_runs(self) -> int | None:
        """Total over the cells, their predictions included (``None`` if one cannot say)."""
        total = 0
        for plan in self.validation.plans:
            predictions = prediction_runs(self.spec, plan.members)
            if plan.max_simulator_runs is None or predictions is None:
                return None
            total += plan.max_simulator_runs + predictions
        return total


def prediction_runs(spec: StudySpec, members: int | None) -> int | None:
    """Return the simulator runs of one cell's predictions: each ensemble's members.

    Plus the true model and, with ``include_mean_model``, one mean model per ensemble.
    The pilot is one of the members. ``None`` when the members are unknown.
    """
    total = 0
    for template in spec.study.predictions:
        section = template.get("prediction", {})
        ensembles = list(section.get("ensembles", ["posterior", "prior"]))
        forecast = [e for e in ensembles if e != "truth"]
        if forecast and members is None:
            return None
        total += len(forecast) * (members or 0) + ("truth" in ensembles)
        total += len(forecast) if section.get("include_mean_model") else 0
    return total


def resolve_case(case: Path, base: Path, workspace: "Workspace | None" = None) -> Path:
    """Return a study's case file as an absolute path.

    A relative path is taken against the study's directory, and, if no file is there,
    against the workspace, as the UI shows its paths (``5Spots/configs/gies.toml``; ADR
    0032). Otherwise the study's directory is kept, and the error names that path.
    """
    if case.is_absolute():
        return case.resolve()
    beside = (base / case).resolve()
    if beside.exists() or workspace is None or not workspace.roots:
        return beside
    try:
        found = workspace.file(case)
    except ConfigError:  # not in the workspace (WorkspaceError), or not a .toml
        return beside
    return found if found.exists() else beside


def load_study(path: Path, workspace: "Workspace | None" = None) -> StudySpec:
    """Read a study file, resolving its case paths (:func:`resolve_case`).

    Raises
    ------
    ConfigError
        If it cannot be read or does not match the schema.
    """
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        spec = StudySpec.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read study {path}: {error}") from error
    except ValidationError as error:
        raise ConfigError(f"invalid study {path}:\n{validation_problems(error)}") from error
    if any(m.split(":", 1)[0] == "data_mismatch" for m in spec.study.metrics):
        logger.warning(
            # The name's meaning changed on 2026-09-30: older studies meant the
            # mean square, which is now normalized_data_mismatch.
            "study %s compares data_mismatch, the objective the methods minimise (the "
            "error-weighted sum of squares per member); for the mean square per datum, "
            "compare normalized_data_mismatch",
            path,
            extra={"event": "metric_meaning_changed", "metric": "data_mismatch"},
        )
    base = path.resolve().parent
    cases = {name: resolve_case(p, base, workspace) for name, p in spec.study.cases.items()}
    return spec.model_copy(update={"study": spec.study.model_copy(update={"cases": cases})})


def replicate_seeds(case_seed: int, study_seed: int, replicates: int) -> list[int]:
    """Return the seeds of replicates 0, 1, ...: the case's own, then derived from the study's."""
    derived = np.random.SeedSequence(study_seed).spawn(max(replicates - 1, 0))
    return [case_seed, *(int(s.generate_state(1)[0]) for s in derived)]


def expand(
    spec: StudySpec, root: Path, base_dir: Path, output_root: Path | None = None
) -> list[PlannedCell]:
    """Expand a study into its cells (case x method x variant x replicate).

    Cells go to ``<output_root>/<study>/<case>/[<method>/]<variant>/r<k>``; ``output_root``
    defaults to ``<root>/studies`` (the web UI's users have their own).

    A method entry replaces the case's algorithm fields before the variant's patch applies;
    without methods, each case runs its own.

    Raises
    ------
    ConfigError
        If a case or a patched configuration is invalid.
    """
    study = spec.study
    methods: list[tuple[str | None, MethodEntry | None]] = list(study.methods.items()) or [
        (None, None)
    ]
    studies = output_root if output_root is not None else root / "studies"
    cells = []
    for case, path in study.cases.items():
        base = json.loads(composition.load_config(path).model_dump_json())
        if study.simulator is not None and base["simulator"]["name"] != study.simulator:
            try:
                base["simulator"] = switch_simulator(base["simulator"], study.simulator)
            except ConfigError as error:
                raise ConfigError(f"study {study.name}: case {case!r}: {error}") from error
        seeds = replicate_seeds(int(base["run"]["seed"]), study.seed, study.replicates)
        for method, entry in methods:
            external = entry is not None and entry.external
            # Another engine's method runs the case with the engine's own
            # algorithm; the case's [method] stays only to validate the configuration.
            algorithm = (
                base
                if entry is None or external
                else {**base, "method": entry.method_section(base["method"])}
            )
            engine = (
                EngineMethod(entry.engine, entry.name, dict(entry.options))
                if entry is not None and external
                else None
            )
            for variant in study.variants:
                if engine is not None and "method" in variant.patch:
                    continue  # not applicable: see skipped_cells
                patched = merge_patch(algorithm, variant.patch)
                parts = [case, *([method] if method else []), variant.name]
                for replicate, seed in enumerate(seeds):
                    document = merge_patch(
                        patched,
                        {
                            "run": {
                                "name": "-".join([study.name, *parts, f"r{replicate}"]),
                                "seed": seed,
                                "output_dir": str(
                                    studies.joinpath(study.name, *parts, f"r{replicate}")
                                ),
                            }
                        },
                    )
                    label = cell_label(case, method, variant.name, replicate)
                    try:
                        model = HistoryMatchConfig.model_validate(document)
                    except ValidationError as error:
                        raise ConfigError(
                            f"study {study.name}: {label}:\n{validation_problems(error)}"
                        ) from error
                    # Paths a patch adds may be relative: to the study file, like its cases.
                    cells.append(
                        PlannedCell(
                            case,
                            method,
                            variant.name,
                            replicate,
                            composition._resolve_paths(model, base_dir),
                            engine,
                        )
                    )
    return cells


def skipped_cells(spec: StudySpec) -> list[SkippedCell]:
    """Return the cells a study does not run.

    These are the variants that patch ``[method]`` (e.g. a localization) of another
    engine's method, which has no ``[method]``.
    """
    study = spec.study
    skipped = []
    for method, entry in study.methods.items():
        if not entry.external:
            continue
        for variant in study.variants:
            if "method" not in variant.patch:
                continue
            for case in study.cases:
                for replicate in range(study.replicates):
                    skipped.append(
                        SkippedCell(
                            label=cell_label(case, method, variant.name, replicate),
                            reason=f"variant {variant.name!r} patches [method], which the "
                            f"{entry.engine} engine's {entry.name} does not have",
                        )
                    )
    return skipped


def switch_simulator(section: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a case's ``[simulator]`` (as JSON) moved to another deck simulator.

    Only the options both simulators mean the same by are kept
    (:data:`~apex.simulation.deck.PORTABLE_OPTIONS`); the target's own take its defaults.

    Raises
    ------
    ConfigError
        If either simulator does not run decks (e.g. the analytic ``linear``).
    """
    from apex.simulation.deck import DECK_SIMULATORS, PORTABLE_OPTIONS

    current = str(section.get("name"))
    if current not in DECK_SIMULATORS or name not in DECK_SIMULATORS:
        raise ConfigError(
            f"[study] simulator = {name!r} cannot replace {current!r}: only the deck "
            f"simulators ({', '.join(sorted(DECK_SIMULATORS))}) run the same case"
        )
    options = {k: v for k, v in section.get("options", {}).items() if k in PORTABLE_OPTIONS}
    return {**section, "name": name, "options": options}


def plan_study(
    service: "ApexService", spec: StudySpec, base_dir: Path, output_root: Path | None = None
) -> StudyPlan:
    """Expand and validate every cell; check the budget (``output_root``: see :func:`expand`)."""
    try:
        cells = expand(spec, service.root, base_dir, output_root)
    except ConfigError as error:
        return StudyPlan(spec, [], Validation(valid=False, errors=[str(error)]))
    errors: list[str] = []
    plans: list[RunPlan] = []
    costs: list[tuple[str, int, int | None]] = []
    valid_cells: list[PlannedCell] = []
    skipped = tuple(skipped_cells(spec))
    for cell in cells:
        # Not check_output: a cell is placed (and an identical one reused) by the study.
        validation = service.validate(cell.config, engine=cell.engine, check_output=False)
        label = cell.label
        errors += [f"{label}: {e}" for e in validation.errors]
        if validation.plan is not None:
            plans.append(validation.plan)
            runs, predictions = (
                validation.plan.max_simulator_runs,
                prediction_runs(spec, validation.plan.members),
            )
            arm = label.rsplit("/", 1)[0]
            costs.append((arm, cell.replicate, None if runs is None or predictions is None
                          else runs + predictions))  # fmt: skip
        errors += [f"{label}: {e}" for e in _prediction_violations(service, spec, cell, base_dir)]
        if not validation.errors:
            valid_cells.append(cell)
    errors += _unrecorded_metrics(spec, valid_cells)
    errors += _unknown_metric_labels(spec, valid_cells)
    plan = StudyPlan(spec, cells, Validation(valid=not errors, errors=errors, plans=plans), skipped)
    limit = spec.study.budget.max_simulator_runs
    total = plan.max_simulator_runs
    if not errors and limit is not None and (total is None or total > limit):
        errors.append(over_budget(total, limit, costs))
        plan = StudyPlan(spec, cells, Validation(valid=False, errors=errors, plans=plans), skipped)
    return plan


def _unrecorded_metrics(spec: StudySpec, cells: Sequence[PlannedCell]) -> list[str]:
    """Return why a study metric names a performance metric none of its cells records.

    A reported issue (#192): a study compared ``facies_brier`` on a case that records no facies
    score, and would have finished with no values to compare. A metric some cells record
    (e.g. ``rmse`` in a study of a synthetic and a real case) is compared where it is.
    Only names of the ``metrics`` registry are checked; the objective and the held-out
    data's scores (``cv_``/``test_``) are recorded by the workflow itself.
    """
    from apex.core.plugins import metrics

    wanted = {m.split(":", 1)[0] for m in spec.study.metrics}
    asked = sorted(k for k in wanted if k in metrics.names())
    if not asked or not cells:
        return []
    recorded: set[str] = set()
    truthless = False
    for cell in cells:
        if cell.replicate != 0:
            continue  # replicates differ in their seeds only
        recorded |= set(composition.build_metrics(cell.config))
        kind = cell.config.case
        truthless |= kind.kind == "real" or kind.truth_file is None
    return [
        f"study metric {key!r} would have no values: the cells record "
        f"{', '.join(sorted(recorded))}"
        + (" (a case without a true model has no model-error metrics)" if truthless else "")
        for key in asked
        if key not in recorded
    ]


#: Metrics the workflow records itself, labelled ``all`` and by data type.
_DATA_METRICS = frozenset({
    "data_mismatch", "normalized_data_mismatch",
    "cv_normalized_data_mismatch", "test_normalized_data_mismatch",
})  # fmt: skip
#: Metrics whose labels are their own, not the state's variables.
_OWN_LABELS = {"constraint_cost": frozenset({"cost", "violations"})}


def _unknown_metric_labels(spec: StudySpec, cells: Sequence[PlannedCell]) -> list[str]:
    """Return why a study metric's name or label cannot be recorded by any of its cells.

    A reported issue (2026-10-24): ``rmse:xx``, a typo for the state variable ``x``,
    validated, ran, and was left out of the comparison without a word. The labels are
    known from each case: ``all``, its state variables for the metrics of the state, and
    its data types for the data metrics. Where a case does not fix them before it runs
    (data transforms rename its data types, parameterizations give the metrics properties,
    data read from another program), its labels are not checked.
    """
    from apex.core.plugins import metrics

    if not cells:
        return []
    known = set(metrics.names()) | _DATA_METRICS
    variables: set[str] = set()
    datatypes: set[str] = set()
    open_variables = open_datatypes = False
    for cell in cells:
        if cell.replicate != 0:
            continue  # replicates differ in their seeds only
        config = cell.config
        variables |= {v.name for v in config.state.variables}
        open_variables |= bool(config.parameterizations)
        found = _datatypes(config)
        open_datatypes |= found is None
        datatypes |= found or set()
    errors = []
    for metric in spec.study.metrics:
        key, _, label = metric.partition(":")
        if key not in known:
            errors.append(f"study metric {metric!r}: no metric {key!r}; the metrics are "
                          f"{', '.join(sorted(known))}")  # fmt: skip
            continue
        if key in _OWN_LABELS:
            labels, kind, label = set(_OWN_LABELS[key]), "labels", label or "all"
        elif not label or label == "all" or (key in _DATA_METRICS and open_datatypes):
            continue
        elif key in _DATA_METRICS:
            labels, kind = datatypes, "data types"
        elif open_variables:
            continue
        else:
            # Facies scores add their own parts after the variable ("facies:skill").
            labels, kind, label = variables, "state variables", label.split(":", 1)[0]
        if label not in labels:
            given = "'all' or " if key not in _OWN_LABELS else ""
            errors.append(f"study metric {metric!r}: no {label!r} in the cases; give {given}"
                          f"one of their {kind}: {', '.join(sorted(labels))}")  # fmt: skip
    return errors


def _datatypes(config: HistoryMatchConfig) -> set[str] | None:
    """Return a case's data types as its metrics label them, or None if not known yet."""
    from apex.storage.inputs import read_observations

    section = config.observations
    if config.data_transforms or (section.file is None and not section.fields):
        return None
    found = {entry.datatype for entry in section.fields}
    if config.geophysics is not None:
        found |= {name for datum in config.geophysics.data for name in datum.datatypes()}
    if section.file is not None:
        try:
            found |= {key.datatype for key in read_observations(section.file).keys}
        except ConfigError:
            return None  # the cell's own validation reports it
    return found


def _prediction_violations(
    service: "ApexService", spec: StudySpec, cell: PlannedCell, base_dir: Path
) -> list[str]:
    """Return what the ``[policy]`` refuses in a cell's predictions.

    Checked when the study is planned, so a refused prediction stops the study before any
    cell is queued (it was found only after, and the cells rolled back): the prediction as
    configured, and what would run (the cell's configuration with the prediction's
    simulator).
    """
    from apex.services import policy
    from apex.services.prediction import simulator_section

    rules = service.jobs.settings.policy
    if not policy.active(rules) or not spec.study.predictions:
        return []
    assert cell.config.run.output_dir is not None
    problems: list[str] = []
    for template in spec.study.predictions:
        try:
            config = _prediction_config(template, cell.config.run.output_dir, base_dir)
            policy.check(config, rules, service.root)
            simulator = simulator_section(config, cell.config)
            policy.check(cell.config.model_copy(update={"simulator": simulator}), rules,
                         service.root)  # fmt: skip
        except ConfigError as error:  # PolicyViolation, or an invalid template
            problems.append(f"prediction {template.get('prediction', {}).get('name')!r}: {error}")
    return problems


def submit_study(
    service: "ApexService",
    spec: StudySpec,
    base_dir: Path,
    *,
    approve: ApprovalHook | None = None,
    approval_token: str | None = None,
    priority: str = "normal",
    client: str = "user",
    approved: bool = False,
    output_root: Path | None = None,
    check: Callable[[Any], None] | None = None,
    code: str | None = None,
) -> JobHandle:
    """Validate, approve and submit a study; its cells run as child jobs.

    ``code`` is the code every cell and prediction runs (``ApexService.study_code``),
    frozen once the study is approved.

    ``output_root`` holds the cells' runs (default ``<root>/studies``, :func:`expand`).
    ``check`` sees, before anything is submitted, everything that will run as it will run:
    each cell's configuration, its engine's method (``EngineMethod``) for an engine cell,
    and each of its predictions (``PredictionConfig``) with the cell's configuration as the
    prediction runs it.

    Approval covers the whole study: its simulator runs, predictions
    included, and its number of runs (a study with many variants needs a look too).

    Raises
    ------
    ConfigError
        If the study is invalid or over budget.
    ApprovalRequired
        If it needs a person's approval first.
    """
    check_study_name(service, spec.study.name, spec.study.annotations)
    plan = plan_study(service, spec, base_dir, output_root)
    # The caller's check first: what a user may not use is refused as such, before whether
    # it would be valid is said (a hidden plugin is not described).
    if check is not None:
        from apex.services.prediction import as_predicted

        for cell in plan.cells:
            check(cell.config)
            if cell.engine is not None:
                check(cell.engine)
            for template in spec.study.predictions:
                # The cell's planned run folder; a reused identical run may lie elsewhere
                # (the catalogue's), but it is the same user's (owned run ids) and its
                # configuration the one checked here.
                where = cell.config.run.output_dir or base_dir
                prediction = _prediction_config(template, Path(where), base_dir)
                check(prediction)
                check(as_predicted(prediction, cell.config))
    if not plan.validation.valid:
        raise ConfigError("invalid study: " + "; ".join(plan.validation.errors))
    if not approved:
        service.gate(
            [p.run_id for p in plan.validation.plans],
            plan.max_simulator_runs,
            approve=None if approve is None else (lambda cost: approve(plan)),
            approval_token=approval_token,
            client=client,
            annotations=spec.study.annotations,
        )
    commit = service.study_code(code)
    study = spec.study
    # The study's own job carries the submitting client too, as its cells do (the web UI's
    # users see the studies they launched by it).
    study_job = service.jobs.submit(
        "study", {"study": spec.model_dump(mode="json")}, annotations=study.annotations,
        start=False, client=client,
    )  # fmt: skip
    records: list[StudyCell] = []
    try:
        for cell in plan.cells:
            annotations = {
                **study.annotations,
                "study": study.name,
                "case": cell.case,
                **({"method": cell.method} if cell.method else {}),
                "variant": cell.variant,
                "replicate": str(cell.replicate),
            }
            handle = service.submit_run(
                cell.config,
                annotations=annotations,
                parent=study_job,
                approved=True,  # the study as a whole was
                priority=priority,
                client=client,
                engine=cell.engine,
                code=commit,
            )
            found = service.store.find(handle.run_ids[0])
            run_dir = found.path if found is not None else cell.config.run.output_dir
            assert run_dir is not None
            predictions = []
            for template in study.predictions:
                config = _prediction_config(template, Path(run_dir), base_dir)
                predictions.append(
                    service.submit_prediction(
                        config,
                        annotations=annotations,
                        after=[handle.id],
                        parent=study_job,
                        priority=priority,
                        client=client,
                        code=commit,
                    ).id
                )
            records.append(
                StudyCell(
                    case=cell.case,
                    method=cell.method,
                    engine="apex" if cell.engine is None else cell.engine.engine,
                    variant=cell.variant,
                    replicate=cell.replicate,
                    seed=cell.config.run.seed,
                    run_id=handle.run_ids[0],
                    run_dir=Path(run_dir),
                    job=handle.id,
                    prediction_jobs=predictions,
                )
            )
    except Exception:
        # Nothing half-submitted keeps running: stop the cells already queued.
        service.jobs.stop(study_job)
        service.jobs.update(study_job, state="failed", error="submission failed", sealed=True)
        raise
    directory = service.root / "studies" / study.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "study.json").write_text(
        json.dumps(
            {
                "job": study_job,
                "spec": spec.model_dump(mode="json"),
                "cells": [c.model_dump(mode="json") for c in records],
                "skipped": [c.model_dump(mode="json") for c in plan.skipped],
            },
            indent=2,
        )
    )
    # Every cell is submitted: the study may now finish when its cells have.
    service.jobs.update(study_job, sealed=True)
    service.jobs.schedule()
    return JobHandle(id=study_job, kind="study", run_ids=[c.run_id for c in records])


def _prediction_config(template: dict[str, Any], run_dir: Path, base_dir: Path) -> PredictionConfig:
    """Return a study's prediction of one run; its paths resolve against the study file."""
    from apex.services.prediction import resolve_prediction_config

    document = merge_patch(template, {"prediction": {"run_dir": str(run_dir)}})
    try:
        config = PredictionConfig.model_validate(document)
    except ValidationError as error:
        raise ConfigError(f"invalid study prediction:\n{validation_problems(error)}") from error
    return resolve_prediction_config(config, base_dir)


def _difference_note(
    n: int, nonzero: int, reference: Mapping[int, float], variant: Mapping[int, float]
) -> str:
    """Return the caveats of one paired difference, from the pairs actually used."""
    notes = []
    if n == 1:
        notes.append("one replicate: no test")
    elif nonzero < 6:
        notes.append(
            f"{nonzero} non-zero pairs: p cannot fall below {smallest_p_value(nonzero):.3g}"
        )
    if set(reference) != set(variant):
        notes.append(f"over the {n} replicates both have (the variant means use all of theirs)")
    return "; ".join(notes)


def check_study_name(service: "ApexService", name: str, annotations: Mapping[str, str]) -> None:
    """Refuse a study name another submitter's study already has.

    A study's record is ``<root>/studies/<name>/study.json``, which status, comparisons and
    deletion read by name: two people's studies of one name would share it, and one's
    deletion would remove the other's record. The submitter is the
    ``submitted_by`` annotation (a web-UI guest's), none for the owner and local programs.

    Raises
    ------
    ConfigError
        If a study of that name belongs to someone else.
    """
    from apex.services.access import SUBMITTED_BY

    path = service.root / "studies" / name / "study.json"
    if not path.exists():
        return
    try:
        record = json.loads(path.read_text())
        theirs = str(((record.get("spec") or {}).get("study") or {}).get("annotations", {})
                     .get(SUBMITTED_BY, ""))  # fmt: skip
    except (OSError, ValueError, AttributeError):
        theirs = "?"  # unreadable: not shared
    if theirs != str(annotations.get(SUBMITTED_BY, "")):
        raise ConfigError(f"a study named {name!r} exists already: give yours another name")


def study_record(service: "ApexService", name: str) -> dict[str, Any]:
    """Return the study's ``study.json``.

    Raises
    ------
    ConfigError
        If there is no such study.
    """
    # One path segment: a name such as "../../x" (from a tool or an MCP resource) would
    # otherwise open a study.json anywhere. The root is not named in the error.
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name:
        raise ConfigError(f"no study {name!r}")
    path = service.root / "studies" / name / "study.json"
    if not path.exists():
        raise ConfigError(f"no study {name!r}")
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"study {name!r}: its record cannot be read: {error}") from error
    # A damaged record is refused clearly here, not as a KeyError or TypeError (an HTTP 500)
    # wherever it is used.
    if (
        not isinstance(record, dict)
        or not isinstance(record.get("spec"), dict)
        or not isinstance(record.get("cells"), list)
    ):
        raise ConfigError(
            f"study {name!r}: its record is damaged (it needs a spec object and a cells list)"
        )
    return record


#: A study record's cells, validated together so that an error names its cell's index.
_CELLS: TypeAdapter[list[StudyCell]] = TypeAdapter(list[StudyCell])


def _stored[T](name: str, part: str, validate: Callable[[], T]) -> T:
    """Validate one part of a stored study; refuse clearly, naming the field.

    A study stored before a rule was tightened (e.g. a variant named "."), or a damaged
    study.json, gives a refusal such as ``cells.1.variant: Field required``, not a raw
    validation error (an HTTP 500 through the API).
    """
    try:
        return validate()
    except ValidationError as error:
        problems = error.errors()
        if problems:
            field = ".".join([part, *(str(p) for p in problems[0]["loc"])])
            first = f"{field}: {problems[0]['msg']}"
        else:
            first = f"{part}: invalid"
        raise ConfigError(
            f"study {name!r}: its stored configuration is not valid now: {first}"
        ) from error


def compare_study(
    service: "ApexService", name: str, metrics: Sequence[str] | None = None
) -> Comparison:
    """Compare a study's cells at each run's posterior, per case and metric.

    Differences are paired by replicate. Without methods, each variant is taken against the
    first. With methods there are two kinds: each variant against the first
    variant of the same method (the auxiliary technique's effect), and each method against
    the first method with the same variant (the algorithm's effect).
    """
    from apex.services.api import summarise_run

    record = study_record(service, name)
    spec = _stored(name, "spec", lambda: StudySpec.model_validate(record["spec"]))
    cells = _stored(name, "cells", lambda: _CELLS.validate_python(record["cells"]))
    wanted = list(metrics or spec.study.metrics)
    variants = [v.name for v in spec.study.variants]
    methods: list[str | None] = list(spec.study.methods) or [None]
    values: dict[tuple[str, str, str | None, str], dict[int, float]] = {}
    missing = []
    for cell in cells:
        summary = None
        if (cell.run_dir / "history.json").exists():
            summary = summarise_run(cell.run_dir)
        if summary is None or summary.state != "complete":
            missing.append(cell_label(cell.case, cell.method, cell.variant, cell.replicate))
            continue
        for metric in wanted:
            key, _, label = metric.partition(":")
            value = summary.posterior_metrics.get(key, {}).get(label or "all")
            if value is not None:
                where = (cell.case, metric, cell.method, cell.variant)
                values.setdefault(where, {})[cell.replicate] = value
    engines = {name: entry.engine for name, entry in spec.study.methods.items()}
    return comparison_from_values(
        service.root / "studies" / name,
        name,
        cases=list(spec.study.cases),
        metrics=wanted,
        methods=methods,
        variants=variants,
        replicates=spec.study.replicates,
        values=values,
        missing=missing,
        skipped=[
            {"label": str(c["label"]), "reason": str(c["reason"])}
            for c in record.get("skipped", [])
        ],
        engines=engines,
    )


def comparison_from_values(
    directory: Path,
    name: str,
    *,
    cases: Sequence[str],
    metrics: Sequence[str],
    methods: Sequence[str | None],
    variants: Sequence[str],
    replicates: int,
    values: Mapping[tuple[str, str, str | None, str], Mapping[int, float]],
    missing: list[str],
    skipped: list[dict[str, str]],
    engines: Mapping[str, str] | None = None,
) -> Comparison:
    """Return (and write to ``directory``) a study's comparison from its cells' values.

    ``values[(case, metric, method, variant)][replicate]`` is one cell's value. Shared by
    history-matching studies and studies of optimizations.
    """
    engines = engines or {}
    wanted = list(metrics)
    groups = [(m, v) for m in methods for v in variants]
    notes = []
    if replicates == 1:
        notes.append("one replicate: values and differences only, no spread over seeds, no test")
    elif replicates < 6:
        notes.append(
            f"{replicates} replicates: the Wilcoxon test cannot reach p < 0.05 "
            f"(smallest p = {smallest_p_value(replicates):.3g})"
        )
    # (method, variant) against (reference method, reference variant).
    pairs = [((m, v), (m, variants[0])) for m in methods for v in variants[1:]]
    pairs += [((m, v), (methods[0], v)) for v in variants for m in methods[1:]]
    if len(pairs) > 1 and replicates > 1:
        notes.append(
            f"{len(pairs)} paired tests per case and metric; the p-values are not corrected "
            "for multiple comparisons"
        )
    comparisons = []
    for case in cases:
        for metric in wanted:
            per_group = {g: values.get((case, metric, *g), {}) for g in groups}
            if not any(per_group.values()):
                continue
            stats = []
            for (method, variant), by_replicate in per_group.items():
                if by_replicate:
                    summary_stats = summarise(list(by_replicate.values()))
                    stats.append(
                        VariantStats(
                            method=method,
                            engine=engines.get(method or "", "apex"),
                            variant=variant,
                            n=summary_stats.n,
                            mean=summary_stats.mean,
                            std=summary_stats.std,
                            minimum=summary_stats.minimum,
                            maximum=summary_stats.maximum,
                            values=list(by_replicate.values()),
                        )
                    )
            differences = []
            for group, reference_group in pairs:
                base, other = per_group[reference_group], per_group[group]
                shared = sorted(set(base) & set(other))
                if not shared:
                    continue
                result = paired([other[r] for r in shared], [base[r] for r in shared])
                differences.append(
                    PairedDifference(
                        method=group[0],
                        variant=group[1],
                        reference_method=reference_group[0],
                        reference=reference_group[1],
                        n=result.n,
                        mean_difference=result.mean_difference,
                        standard_error=result.standard_error,
                        p_value=result.p_value,
                        note=_difference_note(result.n, result.n_nonzero, base, other),
                    )
                )
            comparisons.append(
                MetricComparison(case=case, metric=metric, variants=stats, differences=differences)
            )
    if values:
        # A metric no finished cell recorded (a misspelt label in an older study) is said,
        # not left out without a word.
        compared = {m.metric for m in comparisons}
        notes += [
            f"{metric}: no values in any finished cell; check the metric's name and label"
            for metric in wanted
            if metric not in compared
        ]
    comparison = Comparison(
        study=name,
        replicates=replicates,
        reference=variants[0],
        reference_method=methods[0],
        metrics=comparisons,
        notes=notes,
        missing=missing,
        skipped=skipped,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(comparison.model_dump_json(indent=2))
    rows = [
        {
            "case": m.case,
            "metric": m.metric,
            **({"method": v.method, "engine": v.engine} if methods != [None] else {}),
            "variant": v.variant,
            "n": v.n,
            "mean": v.mean,
            "std": "" if v.std is None else v.std,
            "min": v.minimum,
            "max": v.maximum,
        }
        for m in comparisons
        for v in m.variants
    ]
    if rows:
        import csv

        with (directory / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return comparison
