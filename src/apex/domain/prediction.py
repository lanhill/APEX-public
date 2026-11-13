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

    [surrogate]                                   # optional: forecast with a fitted model
    model_file = "../surrogates/to_2030/model/surrogate.json"

Relative paths are resolved against the configuration file's directory.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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
    truth_is_simulator: bool = Field(
        default=False,
        description="A run with a model-error model: its truth is this simulator's own (the "
        "observations came from it, without model error), so 'truth' is forecast by the "
        "simulator alone. Without it, 'truth' is refused: a model error means the data came "
        "from another model, which the simulator does not reproduce.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="The seed of the prediction's own draws: a model-error model that "
        "extrapolates its parameters past the history. Default: "
        "derived from the run's seed and the prediction's name. Refused when nothing in the "
        "prediction draws.",
    )

    @model_serializer(mode="wrap")
    def _without_unset_truth_rule(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``truth_is_simulator`` and ``seed`` out of the dump when unset.

        A plain prediction's ``config.json`` and fingerprint stay those of predictions made
        before the options existed, so their folders resume.
        """
        data: dict[str, Any] = handler(self)
        if not data.get("truth_is_simulator"):
            data.pop("truth_is_simulator", None)
        if data.get("seed") is None:
            data.pop("seed", None)
        return data

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

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        # The name places the prediction (RUN_DIR/predictions/<name>, a guest's
        # UI/guests/<name>/predictions/<run>/<name>): a separator or "." / ".." would put it
        # elsewhere, out of a guest's own folder (the review).
        if "/" in value or "\\" in value or value in (".", ".."):
            raise ValueError(
                f"prediction name {value!r}: one folder name, no '/' or '\\', not '.' or '..'"
            )
        return value


class SimulatorOverride(_Strict):
    """``[simulator]`` of a prediction: changes to the run's simulator.

    Without ``name``, ``options`` update the run's simulator options (e.g. another deck
    template). With another ``name``, the simulator is replaced and takes only ``options``.
    """

    name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class PredictionSurrogateSection(_Strict):
    """``[surrogate]`` of a prediction: members forecast by a model.

    The model's keys must reach the requested vectors and times (its horizon): the
    prediction's times default to the model's report times. The true model is always
    forecast by the simulator, and the history check compares the surrogate's members
    within the model's gate.
    """

    model_file: Path = Field(description="The surrogate's manifest (surrogate.json).")
    outside: Literal["stop", "simulator"] = Field(
        default="stop",
        description="A member outside the model's domain stops the prediction, or is run "
        "by the simulator.",
    )

    @field_validator("outside", mode="before")
    @classmethod
    def _no_extrapolation(cls, value: object) -> object:
        # outside = "surrogate" (extrapolation by the user's explicit choice) is a history
        # match's option only: said so, rather than pydantic's list of the allowed values.
        if value == "surrogate":
            raise ValueError(
                'outside = "surrogate" (extrapolating beyond the validated domain) is offered '
                'only in history matching; here give "stop" or "simulator"'
            )
        return value

    workers: int | None = Field(
        default=None, ge=1, description="Process workers for the surrogate (None: serial)."
    )


class PredictionControlVariable(_Strict):
    """``[[controls.variables]]`` of a prediction: one named block of fixed controls."""

    name: str = Field(min_length=1)
    values: float | list[float] = Field(description="The block's value(s), one per control.")
    lower: float | None = Field(default=None, description="Lower bound of every value.")
    upper: float | None = Field(default=None, description="Upper bound of every value.")

    @property
    def size(self) -> int:
        """Return the number of controls in the block."""
        return 1 if isinstance(self.values, float | int) else len(self.values)

    def array(self) -> NDArray[np.float64]:
        """Return the values, ``(size,)``."""
        return np.atleast_1d(np.asarray(self.values, dtype=np.float64)).copy()

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        values = self.array()
        if values.size == 0:
            raise ValueError(f"controls {self.name!r} have no value")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"controls {self.name!r} have non-finite values")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"controls {self.name!r}: lower must be below upper")
        outside = [
            float(v)
            for v in values
            if (self.lower is not None and v < self.lower)
            or (self.upper is not None and v > self.upper)
        ]
        if outside:
            raise ValueError(f"controls {self.name!r}: values outside their bounds: {outside}")
        return self


class PredictionControlsSection(_Strict):
    """``[controls]`` of a prediction (decision 6): a control scenario.

    Fixed values every member and the truth are given, after the run's own
    parameterizations: the deck template reads them (e.g. ``${prodbhp_steps[k]}``), and a
    control-aware surrogate gets them as inputs. They are never part of the state.
    ``parameterizations`` map them first, as an optimization's controls (e.g.
    ``control_schedule``).
    """

    variables: list[PredictionControlVariable] = Field(min_length=1)
    parameterizations: list[PluginSection] = Field(
        default_factory=list,
        description="Parameterizations of the controls only (e.g. control_schedule), in order.",
    )

    @model_validator(mode="after")
    def _unique(self) -> Self:
        names = [v.name for v in self.variables]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"controls repeat: {repeated}")
        return self


class PredictionModelErrorSection(_Strict):
    """``[model_error]`` of a prediction: the forecast's own options.

    Merged into the run's model-error options. Only the options the model declares in
    ``forecast_options`` (how its parameters past the history are built) may be set: the
    others shape the history, which the prediction reproduces.
    """

    options: dict[str, Any] = Field(min_length=1)


#: Optional sections left out of a prediction's dump when unset, so that a prediction
#: without them keeps the configuration and fingerprint of one made before they existed.
_DROPPED_WHEN_UNSET = ("surrogate", "controls", "model_error")


class PredictionConfig(_Strict):
    """A complete prediction."""

    prediction: PredictionSection
    simulator: SimulatorOverride = Field(default_factory=SimulatorOverride)
    execution: PluginSection | None = Field(
        default=None, description="Where members run; default: the run's [execution]."
    )
    surrogate: PredictionSurrogateSection | None = Field(
        default=None, description="Forecast the members with a fitted surrogate."
    )
    controls: PredictionControlsSection | None = Field(
        default=None, description="A control scenario every member is given."
    )
    model_error: PredictionModelErrorSection | None = Field(
        default=None,
        description="Options of the run's model-error model for the forecast past the history "
        "(only those the model declares in forecast_options).",
    )

    @model_serializer(mode="wrap")
    def _without_unset_sections(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``surrogate``, ``controls`` and ``model_error`` out of the dump when unset.

        A plain prediction's ``config.json`` and its fingerprint then stay those of
        predictions made before the sections existed, so their folders resume (the coder's
        review). The return annotation keeps the serialization schema's shape.
        """
        data: dict[str, Any] = handler(self)
        for name in _DROPPED_WHEN_UNSET:
            if data.get(name) is None:
                data.pop(name, None)
        return data


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
