"""Studies of optimizations: expand, validate, submit and compare.

A cell is a case's optimization configuration with a method's ``[optimizer]`` (or the
case's own), a variant's patch and a replicate's seed (the case's own first, then seeds
derived from the study's, as history-matching studies do). Each cell runs as a child
``optimization`` job of the study's job; the record is ``<root>/studies/<name>/study.json``
with ``"kind": "optimization"``, and the comparison reuses the history-matching studies'
statistics (:func:`apex.services.study.comparison_from_values`).
"""

import json
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import TypeAdapter, ValidationError

from apex.algorithms.optimization.multiobjective import MAX_HYPERVOLUME_OBJECTIVES, hypervolume
from apex.algorithms.optimization.protocols import FloatArray
from apex.core.errors import ApexError, ConfigError, validation_problems
from apex.domain.optimization import OptimizationConfig
from apex.domain.optimization_study import (
    METRIC_KEYS,
    OptimizationStudyCell,
    OptimizationStudySpec,
    metric_key,
)
from apex.domain.services import Comparison, JobHandle, OptimizationPlan, Validation
from apex.domain.study import merge_patch
from apex.services.optimization import optimization_summary, resolve_optimization_config
from apex.services.study import (
    cell_label,
    comparison_from_values,
    over_budget,
    replicate_seeds,
    resolve_case,
    study_record,
)
from apex.storage import optimization_directory as layout

if TYPE_CHECKING:
    from apex.services.api import ApexService
    from apex.services.workspace import Workspace

#: The ``kind`` of an optimization study's ``[study]`` table and record.
KIND = "optimization"

_CELLS: TypeAdapter[list[OptimizationStudyCell]] = TypeAdapter(list[OptimizationStudyCell])


def is_optimization_study(document: dict[str, Any]) -> bool:
    """Return whether a parsed study file is a study of optimizations."""
    study = document.get("study")
    return isinstance(study, dict) and study.get("kind") == KIND


def load_optimization_study(
    path: Path, workspace: "Workspace | None" = None
) -> OptimizationStudySpec:
    """Read a study of optimizations, resolving its case paths.

    Raises
    ------
    ConfigError
        If it cannot be read or does not match the schema.
    """
    try:
        spec = OptimizationStudySpec.model_validate(tomllib.loads(path.read_text("utf-8")))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read study {path}: {error}") from error
    except ValidationError as error:
        raise ConfigError(f"invalid study {path}:\n{validation_problems(error)}") from error
    base = path.resolve().parent
    cases = {n: resolve_case(p, base, workspace) for n, p in spec.study.cases.items()}
    return spec.model_copy(update={"study": spec.study.model_copy(update={"cases": cases})})


@dataclass
class OptimizationCell:
    """One planned optimization of a study."""

    case: str
    method: str | None
    variant: str
    replicate: int
    config: OptimizationConfig
    plan: OptimizationPlan | None = None


@dataclass
class OptimizationStudyPlan:
    """A study's cells and what they cost."""

    cells: list[OptimizationCell]
    errors: list[str] = field(default_factory=list)

    @property
    def max_simulator_runs(self) -> int | None:
        """Return the sum of the cells' bounds (``None`` if one cannot say)."""
        runs = [c.plan.max_simulator_runs if c.plan else None for c in self.cells]
        return None if any(r is None for r in runs) else sum(r or 0 for r in runs)


def expand(
    service: "ApexService",
    spec: OptimizationStudySpec,
    base_dir: Path,
    output_root: Path | None = None,
) -> list[OptimizationCell]:
    """Return every cell: case x method x variant x replicate, placed under the study.

    Cells go to ``<output_root>/<study>/<label>``; ``output_root`` defaults to
    ``<root>/studies`` (the web UI's users' folders, as history-matching studies).

    Raises
    ------
    ConfigError
        If a case cannot be read or a patched configuration is invalid.
    """
    study = spec.study
    cells = []
    for case_name, case_path in study.cases.items():
        path = case_path if case_path.is_absolute() else (base_dir / case_path).resolve()
        try:
            document = tomllib.loads(path.read_text("utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError(f"case {case_name!r}: cannot read {path}: {error}") from error
        case_seed = int(document.get("optimization", {}).get("seed", 0))
        seeds = replicate_seeds(case_seed, study.seed, study.replicates)
        methods: list[str | None] = list(study.methods) or [None]
        for method in methods:
            for variant in study.variants:
                for replicate, seed in enumerate(seeds):
                    patched = merge_patch(document, variant.patch)
                    if method is not None:
                        patched["optimizer"] = study.methods[method].model_dump(mode="json")
                    label = cell_label(case_name, method, variant.name, replicate)
                    studies = output_root if output_root is not None else service.root / "studies"
                    place = studies / study.name / label
                    patched["optimization"] = {
                        **patched.get("optimization", {}),
                        "name": f"{study.name}-{label.replace('/', '-')}",
                        "seed": seed,
                        "output_dir": str(place),
                    }
                    config = resolve_optimization_config(patched, path.parent)
                    cells.append(
                        OptimizationCell(case_name, method, variant.name, replicate, config)
                    )
    return cells


def plan_optimization_study(
    service: "ApexService",
    spec: OptimizationStudySpec,
    base_dir: Path,
    output_root: Path | None = None,
) -> OptimizationStudyPlan:
    """Expand a study and plan every cell; nothing is evaluated."""
    try:
        cells = expand(service, spec, base_dir, output_root)
    except ApexError as error:
        return OptimizationStudyPlan([], [str(error)])
    errors = []
    for cell in cells:
        label = cell_label(cell.case, cell.method, cell.variant, cell.replicate)
        validation = service.validate_optimization(cell.config)
        if not validation.valid:
            errors += [f"{label}: {e}" for e in validation.errors]
        cell.plan = validation.optimization
    plan = OptimizationStudyPlan(cells, errors)
    budget = spec.study.budget.max_simulator_runs
    total = plan.max_simulator_runs
    # Not over invalid cells: their cost is unknown, not unbounded (their errors say why).
    if not errors and budget is not None and (total is None or total > budget):
        costs = [
            (cell_label(c.case, c.method, c.variant, c.replicate).rsplit("/", 1)[0],
             c.replicate, c.plan.max_simulator_runs if c.plan else None)
            for c in plan.cells
        ]  # fmt: skip
        plan.errors.append(over_budget(total, budget, costs))
    return plan


def validate_optimization_study(
    service: "ApexService", spec: OptimizationStudySpec, base_dir: Path
) -> Validation:
    """Validate a study of optimizations (``notes`` give each cell's bound)."""
    plan = plan_optimization_study(service, spec, base_dir)
    if plan.errors:
        return Validation(valid=False, errors=plan.errors[:50])
    notes = [
        f"{cell_label(c.case, c.method, c.variant, c.replicate)}: at most "
        f"{c.plan.max_simulator_runs if c.plan else '?'} simulator runs"
        for c in plan.cells
    ]
    notes.append(f"study: {len(plan.cells)} optimizations, at most {plan.max_simulator_runs} "
                 "simulator runs")  # fmt: skip
    return Validation(valid=True, notes=notes)


def submit_optimization_study(
    service: "ApexService",
    spec: OptimizationStudySpec,
    base_dir: Path,
    *,
    approve: "Callable[[Any], bool] | None" = None,
    approval_token: str | None = None,
    priority: str = "normal",
    client: str = "user",
    approved: bool = False,
    output_root: Path | None = None,
    check: "Callable[[Any], None] | None" = None,
    code: str | None = None,
) -> JobHandle:
    """Validate, approve (as a whole) and submit a study; its cells run as child jobs.

    ``code`` is the code every cell runs (``ApexService.study_code``), frozen once the
    study is approved.

    ``output_root`` holds the cells (default ``<root>/studies``; :func:`expand`).
    ``check`` sees every cell's configuration as it will be submitted, before anything is.

    ``approve`` is asked with the :class:`OptimizationStudyPlan` (its ``cells`` and
    ``max_simulator_runs``, as a history-matching study's plan has).

    Raises
    ------
    ConfigError
        If the study is invalid or over its budget.
    ApprovalRequired
        If it needs a person's approval first.
    """
    from apex.services.study import check_study_name

    check_study_name(service, spec.study.name, spec.study.annotations)
    plan = plan_optimization_study(service, spec, base_dir, output_root)
    # The caller's check first: a plugin a user may not use is refused as such.
    if check is not None:
        for cell in plan.cells:
            check(cell.config)
    if plan.errors:
        raise ConfigError("invalid study: " + "; ".join(plan.errors[:10]))
    if not approved:
        service.gate(
            [c.plan.optimization_id for c in plan.cells if c.plan],
            plan.max_simulator_runs,
            approve=None if approve is None else (lambda cost: approve(plan)),
            approval_token=approval_token,
            client=client,
            annotations=spec.study.annotations,
        )
    commit = service.study_code(code)
    study = spec.study
    study_job = service.jobs.submit(
        "study", {"study": spec.model_dump(mode="json"), "kind": KIND},
        annotations=study.annotations, start=False, client=client,
    )  # fmt: skip
    records: list[OptimizationStudyCell] = []
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
            handle = service.submit_optimization(
                cell.config,
                annotations=annotations,
                parent=study_job,
                priority=priority,
                client=client,
                approved=True,  # the study as a whole was
                code=commit,
            )
            assert cell.config.optimization.output_dir is not None
            records.append(
                OptimizationStudyCell(
                    case=cell.case,
                    method=cell.method,
                    variant=cell.variant,
                    replicate=cell.replicate,
                    seed=cell.config.optimization.seed,
                    optimization_id=handle.run_ids[0],
                    run_dir=cell.config.optimization.output_dir,
                    job=handle.id,
                )
            )
    except Exception:
        service.jobs.stop(study_job)
        service.jobs.update(study_job, state="failed", error="submission failed", sealed=True)
        raise
    directory = service.root / "studies" / study.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "study.json").write_text(
        json.dumps(
            {
                "kind": KIND,
                "job": study_job,
                "spec": spec.model_dump(mode="json"),
                "cells": [c.model_dump(mode="json") for c in records],
            },
            indent=2,
        )
    )
    service.jobs.update(study_job, sealed=True)
    service.jobs.schedule()
    return JobHandle(id=study_job, kind="study", run_ids=[c.optimization_id for c in records])


def compare_optimization_study(
    service: "ApexService", name: str, metrics: list[str] | None = None
) -> Comparison:
    """Compare a study's optimizations: final and best objective, evaluations, iterations.

    ``objective:final`` is the result's objective (its mean over realizations),
    ``objective:best`` the lowest accepted value in the history, ``evaluations:all`` the
    journal's simulated evaluations and ``iterations:all`` the accepted steps. With several
    objectives, ``hypervolume:final`` is each cell's front measured against one
    reference point for the whole study (``[study] reference_point``, else the nadir of all
    the cells' fronts plus 10 % of each range, so that cells compare on one scale), and
    ``front_size:final`` the front's members. Only finished cells count, so a finished
    cell's value does not move while others run; ``objective:final`` of such a run is its
    representative's first objective.

    Raises
    ------
    ConfigError
        If the hypervolume is asked for cells with different numbers of objectives, or with
        a ``reference_point`` of the wrong length; or if ``metrics`` names one the
        comparison does not report (it would be an empty column).
    """
    unknown = [m for m in metrics or [] if metric_key(m) not in METRIC_KEYS]
    if unknown:
        raise ConfigError(f"unknown metrics {unknown}: a study of optimizations reports "
                          f"{', '.join(METRIC_KEYS)} (a bare name means name:all)")  # fmt: skip
    record = study_record(service, name)
    spec = OptimizationStudySpec.model_validate(record["spec"])
    cells = _CELLS.validate_python(record["cells"])
    wanted = list(metrics or spec.study.metrics)
    variants = [v.name for v in spec.study.variants]
    methods: list[str | None] = list(spec.study.methods) or [None]
    values: dict[tuple[str, str, str | None, str], dict[int, float]] = {}
    missing = []
    finished = [cell for cell in cells if (cell.run_dir / layout.RESULT).exists()]
    fronts = _study_fronts(finished)
    wants_hv = any(m.split(":")[0] == "hypervolume" for m in wanted)
    reference = _study_reference(fronts, spec.study.reference_point) if wants_hv else None
    for cell in cells:
        label = cell_label(cell.case, cell.method, cell.variant, cell.replicate)
        if not (cell.run_dir / layout.RESULT).exists():
            missing.append(label)
            continue
        summary = optimization_summary(cell.run_dir)
        assert summary.result is not None
        objectives = [float(r["objective"]) for r in summary.history if "objective" in r]
        found = {
            "objective:final": float(summary.result["objective"]),
            "objective:best": min(objectives) if objectives else None,
            "evaluations:all": float(summary.evaluations),
            "iterations:all": float(summary.result["iterations"]),
        }
        front = fronts.get(cell.run_dir)
        if front is not None:
            found["front_size:final"] = float(front.shape[1])
            if reference is not None and len(reference) <= MAX_HYPERVOLUME_OBJECTIVES:
                found["hypervolume:final"] = hypervolume(front, reference) if front.size else 0.0
        for metric in wanted:
            key = metric_key(metric)
            value = found.get(key)
            if value is not None:
                values.setdefault((cell.case, metric, cell.method, cell.variant), {})[
                    cell.replicate
                ] = value
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
        skipped=[],
    )


def _study_fronts(cells: Sequence[OptimizationStudyCell]) -> dict[Path, FloatArray]:
    """Return each multi-objective cell's feasible front (means) ``(n_obj, m)``."""
    fronts: dict[Path, FloatArray] = {}
    for cell in cells:
        pareto = layout.read_pareto(cell.run_dir)
        if pareto is None:
            continue
        objectives = np.asarray(pareto["objectives"], dtype=float)
        feasible = np.asarray(pareto["violation"], dtype=float) <= 0.0
        fronts[cell.run_dir] = np.ascontiguousarray(objectives[feasible].T) if objectives.size \
            else np.empty((0, 0))  # fmt: skip
    return fronts


def _study_reference(
    fronts: Mapping[Path, FloatArray], given: Sequence[float] | None
) -> FloatArray | None:
    """Return the study's reference point: given, else all fronts' nadir plus 10 % of range.

    ``None`` without fronts. Raises ``ConfigError`` if the cells' fronts have different
    numbers of objectives, or ``given`` has another.
    """
    # A front with no feasible member still has a row per objective: ``(n_obj, 0)``.
    counts = {int(front.shape[0]) for front in fronts.values() if front.shape[0]}
    if len(counts) > 1:
        raise ConfigError(
            f"hypervolume: the study's cells have {sorted(counts)} objectives; compare cells "
            "with the same objectives (or leave hypervolume out of [study] metrics)"
        )
    if given is not None:
        if counts and len(given) not in counts:
            raise ConfigError(
                f"[study] reference_point has {len(given)} values for {counts.pop()} objectives"
            )
        return np.asarray(given, dtype=float)
    together = [f for f in fronts.values() if f.size]
    if not together:
        return None
    union = np.hstack(together)
    ideal, nadir = np.min(union, axis=1), np.max(union, axis=1)
    span = np.where(nadir > ideal, nadir - ideal, np.maximum(np.abs(nadir), 1.0))
    return np.asarray(nadir + 0.1 * span, dtype=float)
