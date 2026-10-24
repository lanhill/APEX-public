"""A study of optimizations: cases x optimizers x variants x seeds.

Example::

    [study]
    kind = "optimization"
    name = "3well_optimizers"
    replicates = 3

    [study.cases]
    threewell = "3well/enopt.toml"

    [study.methods.enopt]          # a complete [optimizer] section (the rule)
    name = "enopt"
    options = { ne = 10, maxiter = 5, alpha = 0.2 }

    [study.methods.smcopt]
    name = "smcopt"
    options = { ne = 10, maxiter = 5 }

    [[study.variants]]
    name = "base"

    [[study.variants]]
    name = "cheap_water"           # patches anything but the optimizer and the identity
    patch = { objective = { options = { wwi = 6.0 } } }
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from apex.domain.config import PluginSection, _Strict
from apex.domain.study import BudgetSection, _folder_name

#: What a study of optimizations compares by default: ``metric:which``.
OPTIMIZATION_METRICS = ("objective:final", "objective:best", "evaluations:all")
#: Every ``metric:which`` a study of optimizations reports (the comparison finds these keys;
#: a bare name means ``name:all``).
METRIC_KEYS = (
    "objective:final", "objective:best", "evaluations:all", "iterations:all",
    "front_size:final", "hypervolume:final",
)  # fmt: skip
#: What a study of multi-objective optimizations compares by default: the fronts.
MULTI_OBJECTIVE_METRICS = ("hypervolume:final", "front_size:final", "evaluations:all")
#: The values an optimization study compares (the part before ``:``).
METRIC_NAMES = frozenset({"objective", "evaluations", "iterations", "hypervolume", "front_size"})


def metric_key(metric: str) -> str:
    """Return the key a study's ``metric`` is compared by: a bare name means ``name:all``."""
    return metric if ":" in metric else f"{metric}:all"


class OptimizationVariant(_Strict):
    """``[[study.variants]]`` of an optimization study: one change to every case."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON merge patch (RFC 7386) over each case's optimization configuration.",
    )

    @model_validator(mode="after")
    def _keeps_the_optimizer_and_identity(self) -> "OptimizationVariant":
        if "optimizer" in self.patch:
            raise ValueError(
                f"variant {self.name!r} patches [optimizer]: compare optimizers as "
                "[study.methods.<name>] entries (each complete)"
            )
        owned = sorted(set(self.patch.get("optimization", {})) & {"name", "seed", "output_dir"})
        if owned:
            raise ValueError(
                f"variant {self.name!r} patches [optimization] {', '.join(owned)}: the study "
                "sets each cell's name, seed and place"
            )
        return self

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        return _folder_name(value, "variant name")


class OptimizationStudySection(_Strict):
    """``[study]`` with ``kind = "optimization"``."""

    kind: Literal["optimization"]
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = Field(default=0, ge=0, description="Seeds of replicates 1, 2, ... derive from it.")
    replicates: int = Field(default=1, ge=1, description="Seeds per case, method and variant.")
    metrics: list[str] = Field(
        default_factory=lambda: list(OPTIMIZATION_METRICS),
        description="What to compare: objective:final, objective:best, evaluations:all, "
        "iterations:all; for several objectives also hypervolume:final and "
        "front_size:final.",
    )
    reference_point: list[float] | None = Field(
        default=None,
        description="The hypervolume's reference point for every cell (several objectives); "
        "default: the nadir of all cells' fronts plus 10 % of each objective's range.",
    )
    budget: BudgetSection = Field(default_factory=BudgetSection)
    annotations: dict[str, str] = Field(default_factory=dict)
    cases: dict[str, Path] = Field(min_length=1, description="Name -> optimization configuration.")
    methods: dict[str, PluginSection] = Field(
        default_factory=dict,
        description="Name -> a complete [optimizer] section; none: each case's own.",
    )
    variants: list[OptimizationVariant] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "OptimizationStudySection":
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names repeat: {names}")
        for method in self.methods:
            _folder_name(method, "method name")
        for case in self.cases:
            _folder_name(case, "case name")
        # A label the comparison does not report (``objective:mean``, a bare ``hypervolume``,
        # which means ``hypervolume:all``) would be an empty column, not an error.
        unknown = [m for m in self.metrics if metric_key(m) not in METRIC_KEYS]
        if unknown:
            raise ValueError(
                f"unknown metrics {unknown}: a study of optimizations reports "
                f"{', '.join(METRIC_KEYS)} (a bare name means name:all)"
            )
        return self

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        return _folder_name(value, "study name")


class OptimizationStudySpec(_Strict):
    """A complete study of optimizations."""

    study: OptimizationStudySection


class OptimizationStudyCell(_Strict):
    """One optimization of a study."""

    case: str
    method: str | None = None
    variant: str
    replicate: int
    seed: int
    optimization_id: str
    run_dir: Path
    job: str
