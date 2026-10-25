"""Studies: cases x methods x variants x replicates, run and compared by APEX.

Example (TOML; the same model in Python)::

    [study]
    name = "localization_m5spots"
    seed = 2026
    replicates = 1                    # the default; more when the case is cheap enough
    metrics = ["rmse:all", "normalized_data_mismatch:all"]
    budget = { max_simulator_runs = 60000 }
    annotations = { hypothesis = "H3" }

    [study.cases]
    m5spots = "multi_5spots/configs/gies_opm.toml"

    [study.methods.esmda]              # optional: the algorithms compared
    name = "esmda"
    options = { inflation = [4, 4, 4, 4] }

    [study.methods.gies]
    name = "gies"
    options = { max_trials_per_iteration = 4 }
    strategy = { name = "rlmmac", options = { energy = 0.99 } }

    [study.methods.ert_esmda]          # ERT's own ES-MDA on the same cases
    engine = "ert"
    name = "es_mda"
    options = { weights = [4, 4, 4, 4] }

    [[study.variants]]
    name = "no_localization"

    [[study.variants]]
    name = "adaptive"
    patch = { method = { localization = { name = "adaptive" } } }

    [[study.predictions]]              # optional, per finished run
    prediction = { name = "to_2030", outputs = { vectors = ["WOPR:*"] } }
    simulator = { options = { deck_template_file = "model/deck_prediction.mako" } }

A **method** entry is a complete algorithm configuration: it replaces every algorithm field
of the case's ``[method]`` (:data:`~apex.domain.config.ALGORITHM_FIELDS`; unset ones take
the defaults, never the case's values) and keeps its auxiliary techniques
(:data:`~apex.domain.config.AUXILIARY_FIELDS`, e.g. localization). Without
``[study.methods]`` each case runs its own method. The algorithm is the base; auxiliary
techniques are variants, so a variant may not change ``method.name``. An entry with another
``engine`` runs that engine's algorithm on the case; variants that patch
``[method]`` do not apply to it, and those cells are listed as not run.

``simulator`` runs every case's deck on ``opm_flow`` or ``eclipse``, with only
the options both share. Relative case paths resolve against the study's folder, else the
workspace (as the UI shows its paths).

A variant's ``patch`` is a JSON merge patch (RFC 7386) over the case's configuration (after
its method entry). Cells are case x method x variant x replicate.
Replicate 0 keeps each case's own seed, so a one-replicate study reuses existing runs of
unchanged configurations; replicates 1, 2, ... take seeds derived from ``seed``. Every
method and variant uses the same seeds (and so the same prior ensemble), so cells are
paired by replicate.
"""

import copy
import re
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from apex.domain.config import ALGORITHM_FIELDS, PluginSection, _Strict

_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _folder_name(value: str, what: str) -> str:
    """Return ``value`` if it can name one folder: a study, a case, a method or a variant.

    Each is a path component of the study's cells (``studies/<study>/<case>/<variant>``), so
    ``.`` and ``..`` (the folder itself, its parent) are refused, as are separators. Not a
    ``pattern``: pydantic's default regex engine has no lookahead.
    """
    if not _NAME.fullmatch(value) or value in (".", ".."):
        raise ValueError(
            f"{what} {value!r}: use letters, digits, '_', '-' and '.', and not '.' or '..' alone"
        )
    return value


def merge_patch(target: Any, patch: Any) -> Any:  # noqa: ANN401 (JSON values)
    """Apply a JSON merge patch (RFC 7386): objects merge, ``None`` removes, others replace."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


class MethodEntry(_Strict):
    """``[study.methods.<name>]``: one algorithm, complete.

    Its fields are the algorithm's fields of ``[method]`` (``name``, ``options``,
    ``diagnostics``, ``strategy``, ``schedule``, ``max_iterations``); those left out take
    ``[method]``'s defaults, not the case's values.
    """

    engine: str = Field(
        default="apex",
        description="Where it runs: 'apex', or another engine such as 'ert'.",
    )
    name: str = Field(
        min_length=1,
        description="The method plugin (esmda, gies, ...), or the engine's algorithm (ERT's "
        "es_mda or ensemble_smoother).",
    )
    options: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, bool | float] | None = None
    strategy: PluginSection | None = None
    schedule: dict[str, Any] | None = None
    max_iterations: int | None = Field(default=None, ge=1)

    @property
    def external(self) -> bool:
        """Whether another engine than APEX's runs this method."""
        return self.engine != "apex"

    @model_validator(mode="after")
    def _engine_fields(self) -> "MethodEntry":
        # An engine's algorithm is its name and options; APEX's strategy, schedule and
        # iteration cap mean nothing to it.
        if self.external:
            given = [f for f in ("strategy", "schedule", "max_iterations", "diagnostics")
                     if getattr(self, f) is not None]  # fmt: skip
            if given:
                hint = (
                    "; to compare analyses or localization with ERT's, run ES-MDA on APEX's "
                    "engine (no engine) with strategy ert_esmda, ert_adaptive or ert_localized "
                    "(ERT's smoother library), against APEX's approx or adaptive_rows"
                    if "strategy" in given
                    else ""
                )
                raise ValueError(
                    f"an engine={self.engine!r} method takes name and options only, not "
                    f"{', '.join(given)}{hint}"
                )
        return self

    def method_section(self, case_method: dict[str, Any]) -> dict[str, Any]:
        """Return the case's ``[method]`` (as JSON) with this entry's algorithm in place."""
        kept = {k: v for k, v in case_method.items() if k not in ALGORITHM_FIELDS}
        own = self.model_dump(mode="json", exclude_unset=True)
        own.pop("engine", None)
        return {**kept, **own}


class VariantSection(_Strict):
    """``[[study.variants]]``: one change to every case."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON merge patch (RFC 7386) over each case's configuration.",
    )

    @model_validator(mode="after")
    def _keeps_the_algorithm(self) -> "VariantSection":
        # Switching the algorithm by a patch inherits the other algorithm's options, strategy
        # and schedule (an ES-MDA "variant" of a GIES case ran GIES's RLM-MAC update).
        method = self.patch.get("method")
        # A bare string is a name too (PluginSection's shorthand).
        if isinstance(method, str) or (isinstance(method, dict) and "name" in method):
            raise ValueError(
                f"variant {self.name!r} patches method.name: compare algorithms as "
                "[study.methods.<name>] entries (each complete), and keep variants for "
                "auxiliary techniques such as localization"
            )
        return self

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        return _folder_name(value, "variant name")

    @model_validator(mode="after")
    def _study_owns_the_run_section(self) -> "VariantSection":
        # The study sets each cell's name, seed and place; a patch there would be lost.
        owned = sorted(set(self.patch.get("run", {})) & {"name", "seed", "output_dir"})
        if owned:
            raise ValueError(
                f"variant {self.name!r} patches run.{', run.'.join(owned)}, which the study sets "
                "(seeds come from [study] seed and replicates)"
            )
        return self


class BudgetSection(_Strict):
    """``[study] budget``: refused before anything runs when the plan exceeds it."""

    max_simulator_runs: int | None = Field(default=None, ge=1)


class StudySection(_Strict):
    """``[study]``."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = Field(default=0, ge=0, description="Seeds of replicates 1, 2, ... derive from it.")
    replicates: int = Field(default=1, ge=1, description="Seeds per case and variant.")
    metrics: list[str] = Field(
        default_factory=lambda: [
            "normalized_data_mismatch:all",
            "rmse:all",
            "cv_normalized_data_mismatch:all",
        ],
        description="'metric:label' to compare at each run's posterior. An unknown metric, or "
        "a label none of the cases has (a state variable or data type; not checked where a "
        "case's labels are known only when it runs), is refused at validation. A metric a "
        "cell does not record (e.g. rmse in a real case) is left out of its comparison.",
    )
    budget: BudgetSection = Field(default_factory=BudgetSection)
    annotations: dict[str, str] = Field(default_factory=dict)
    cases: dict[str, Path] = Field(
        min_length=1, description="Name -> history-matching configuration."
    )
    simulator: str | None = Field(
        default=None,
        description="Run every case's deck on this simulator (opm_flow or eclipse): it "
        "replaces the case's [simulator] name and keeps the options both share. "
        "Unset: each case's own.",
    )
    methods: dict[str, MethodEntry] = Field(
        default_factory=dict,
        description="Name -> algorithm, each complete; none: each case's own.",
    )
    variants: list[VariantSection] = Field(min_length=1)
    predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prediction configurations without run_dir, run after each run.",
    )

    @model_validator(mode="after")
    def _unique(self) -> "StudySection":
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names repeat: {names}")
        for method in self.methods:
            _folder_name(method, "method name")
        for case in self.cases:
            # The same rule as study, method and variant names (it was isalnum, which took
            # any Unicode letter, and refused "." and ".." only by accident).
            _folder_name(case, "case name")
        return self

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        return _folder_name(value, "study name")


class StudySpec(_Strict):
    """A complete study."""

    study: StudySection


class StudyCell(_Strict):
    """One run of a study: a case, a method (``None``: the case's own), variant and replicate."""

    case: str
    method: str | None = None
    engine: str = "apex"
    variant: str
    replicate: int
    seed: int
    run_id: str
    run_dir: Path
    job: str
    prediction_jobs: list[str] = Field(default_factory=list)


class SkippedCell(_Strict):
    """A cell a study does not run: e.g. a localization variant of an ERT method."""

    label: str
    reason: str
