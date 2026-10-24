"""Predictions: forecasting a history-matched ensemble beyond history.

A prediction has its own configuration (:class:`PredictionConfig`), which points at a
history-matching run directory and never changes it. Example::

    [prediction]
    name = "to_2030"
    run_dir = "../runs/gies_cv"
    ensembles = ["posterior", "prior", "truth"]   # also "iteration:<k>"
    include_mean_model = false
    check_history = true
    percentile_convention = "percentile"          # or "exceedance" (SPE-PRMS labels)

    [prediction.outputs]
    vectors = ["FOPT", "WOPR:*"]
    times = "report"                              # or a list of ISO dates

    [simulator.options]                           # overrides of the run's [simulator]
    deck_template_file = "../model/deck_prediction.mako"

Relative paths are resolved against the configuration file's directory.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from apex.domain.config import PluginSection, _Strict

#: Ensemble names of a prediction: ``posterior``, ``prior``, ``truth``, ``iteration:<k>``.
ENSEMBLE_PATTERN = re.compile(r"^(posterior|prior|truth|iteration:\d+)$")

#: Percentile labels and the percentile each one is, per convention.
PERCENTILE_LABELS: dict[str, tuple[tuple[str, float], ...]] = {
    "percentile": (("p10", 10.0), ("p50", 50.0), ("p90", 90.0)),
    # SPE-PRMS: P90 is exceeded with 90 % probability, i.e. the 10th percentile.
    "exceedance": (("P90", 10.0), ("P50", 50.0), ("P10", 90.0)),
}

#: The scores computed when ``[prediction] scores`` is not given.
DEFAULT_SCORES = ("coverage_p10_p90", "crps", "mean_error", "spread")


class PredictionOutputsSection(_Strict):
    """``[prediction.outputs]``: which series to forecast."""

    vectors: list[str] = Field(
        min_length=1, description="Summary vectors or shell-style patterns, e.g. 'WOPR:*'."
    )
    times: Literal["report"] | list[str] = Field(
        default="report", description="'report' (every report time) or a list of ISO dates."
    )


class PredictionSection(_Strict):
    """``[prediction]``."""

    name: str = Field(min_length=1)
    run_dir: Path = Field(description="The history-matching run to forecast from.")
    ensembles: list[str] = Field(
        default_factory=lambda: ["posterior", "prior"],
        min_length=1,
        description="'posterior', 'prior', 'iteration:<k>' and, in synthetic cases, 'truth'.",
    )
    include_mean_model: bool = Field(
        default=False, description="Also forecast each ensemble's mean model."
    )
    check_history: bool = Field(
        default=True,
        description="Fail unless the forecast reproduces the run's stored forecasts over "
        "the history period.",
    )
    percentile_convention: Literal["percentile", "exceedance"] = Field(
        default="percentile",
        description="'percentile': p10 is the 10th percentile; 'exceedance': SPE-PRMS labels "
        "(P90 is the value exceeded with 90 % probability, the 10th percentile).",
    )
    scores: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SCORES),
        description="Forecast scores against the truth (synthetic cases, with 'truth').",
    )
    output_dir: Path | None = Field(
        default=None, description="Default: <run_dir>/predictions/<name>."
    )
    outputs: PredictionOutputsSection

    @model_validator(mode="after")
    def _known_ensembles(self) -> Self:
        bad = [name for name in self.ensembles if not ENSEMBLE_PATTERN.match(name)]
        if bad:
            raise ValueError(
                f"unknown ensembles {bad}; use 'posterior', 'prior', 'truth' or 'iteration:<k>'"
            )
        if len(set(self.ensembles)) != len(self.ensembles):
            raise ValueError(f"ensembles repeat: {self.ensembles}")
        if self.ensembles == ["truth"]:
            raise ValueError("a prediction needs at least one ensemble besides 'truth'")
        return self


class SimulatorOverride(_Strict):
    """``[simulator]`` of a prediction: changes to the run's simulator.

    Without ``name``, ``options`` update the run's simulator options (e.g. another deck
    template). With another ``name``, the simulator is replaced and takes only ``options``.
    """

    name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PredictionConfig(_Strict):
    """A complete prediction."""

    prediction: PredictionSection
    simulator: SimulatorOverride = Field(default_factory=SimulatorOverride)
    execution: PluginSection | None = Field(
        default=None, description="Where members run; default: the run's [execution]."
    )


@dataclass(frozen=True, slots=True)
class EnsembleForecast:
    """One ensemble's forecast.

    Attributes
    ----------
    name
        The ensemble, e.g. ``"posterior"`` or ``"iteration:3"``.
    times, vectors
        The time axis ``(n_times,)`` and the vectors ``(n_vectors,)``.
    values
        ``(n_vectors, n_times, n_members)``, the successful members only.
    member_ids
        The run's id of each member, ``(n_members,)``.
    failed_member_ids
        Members whose forecast failed.
    mean_model
        The forecast of the ensemble-mean model, ``(n_vectors, n_times)``, if requested.
    """

    name: str
    times: tuple[str, ...]
    vectors: tuple[str, ...]
    values: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    failed_member_ids: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    mean_model: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """The outcome of a prediction: its forecast ensembles and scores.

    Attributes
    ----------
    forecasts
        One forecast per ensemble, by name (``truth`` excluded).
    truth
        The true model's forecast ``(n_vectors, n_times)``, if requested.
    first_prediction_time
        Index of the first time after the history period (the last observation time on
        the axis), 0 if the axis has no observation time.
    scores
        ``{ensemble: {vector: {score: value}}}`` over the prediction period; empty without
        a truth forecast or a prediction period.
    """

    forecasts: dict[str, EnsembleForecast]
    truth: NDArray[np.float64] | None
    first_prediction_time: int
    scores: dict[str, dict[str, dict[str, float]]]
