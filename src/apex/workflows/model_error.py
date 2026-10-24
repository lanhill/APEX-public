"""The PG-LSTM model-error model ``pg_lstm``.

It joins the schedule and well-feature reading of :mod:`apex.simulation.model_error` with
the network of :mod:`apex.algorithms.model_error.lstm`; the effective simulator
(:class:`~apex.simulation.model_error.ModelErrorSimulator`) adds its residuals to a run.

Ported from PET-fork
-----------------------
PET module : Examples/.../Brugge/ml_model_error.py (``ModelErrorProd``:
             ``set_obs_ranges``, ``initialize_lstm_models``, ``predict_residuals_lstm``)
             and simulator/eclipse.py (``add_me_residual``, ``assign_nn_params``), the
             checkout's working copy
PET commit : 1348861b08314b8b3f3e1ccf886c7c095273e202 (not public)
Reference  : Luo, X. Model error characterization for ensemble-based history matching
             through a physics guided long short-term memory network (manuscript).
Differences: ``shut_wells`` chooses between the manuscript's zero residual for a shut well
             (``"zero"``, the default) and PET-fork's behaviour, which never zeroes it
             because its ``forward`` tests the truth of a non-empty list (``"ignore"``;
             recorded in docs/mec.md).
"""

import datetime as dt
import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.model_error.lstm import WellNetwork, well_residuals
from apex.core.errors import SimulationError
from apex.core.options import InputFile, Text, TextList
from apex.core.plugins import model_error_models
from apex.domain.observations import DataKey, ObservationSet
from apex.simulation.model_error import (
    RESIDUAL_PREFIX,
    CaseFiles,
    feature_names,
    well_features,
    well_schedules,
)

__all__ = ["PGLSTMModel"]

ShutWells = Literal["zero", "ignore"]


@model_error_models.register("pg_lstm")
class PGLSTMModel:
    """Physics-guided LSTM well residual models, one per well.

    Parameters
    ----------
    schedule_file
        The history schedule: one COMPDAT, WCONHIST and WCONINJH block per report step.
    report_steps
        1-based report steps of the network's report times (the observation times).
    producers, injectors
        Well names.
    producer_data, injector_data
        Their data types, e.g. ``["WOPR", "WBHP", "WWCT"]`` and ``["WBHP"]``.
    hidden_size
        LSTM hidden size.
    state_variable
        The state variable holding the parameters of every well's network, producers then
        injectors, each in torch's layout.
    shut_wells
        ``"zero"`` (the manuscript): no residual while a well is shut (WBHP excepted);
        ``"ignore"``: residuals at every report time (PET-fork's published runs).
    initial_schedule_file
        Schedule text read before ``schedule_file``: the deck's own SCHEDULE keywords
        before its included history (e.g. a first COMPDAT), if any.
    phases
        The deck's phases (oil required): they set the dynamic features, each phase's
        density and saturation (:func:`~apex.simulation.model_error.feature_names`).

    Raises
    ------
    ValueError
        If the schedule does not describe every well at every report step.
    """

    def __init__(
        self,
        schedule_file: InputFile,
        report_steps: Sequence[int],
        producers: TextList = (),
        injectors: TextList = (),
        producer_data: TextList = ("WOPR", "WBHP", "WWCT"),
        injector_data: TextList = ("WBHP",),
        hidden_size: int = 20,
        state_variable: Text = "nn_params",
        shut_wells: ShutWells = "zero",
        phases: TextList = ("oil", "water"),
        initial_schedule_file: InputFile | None = None,
    ) -> None:
        if not producers and not injectors:
            raise ValueError("pg_lstm needs at least one producer or injector")
        if shut_wells not in ("zero", "ignore"):
            raise ValueError(f"shut_wells must be 'zero' or 'ignore', got {shut_wells!r}")
        steps = [int(s) for s in report_steps]
        if not steps or min(steps) < 1 or steps != sorted(set(steps)):
            raise ValueError(f"report_steps must be increasing positive steps, got {steps}")
        self.state_variable = str(state_variable)
        self.report_steps = tuple(steps)
        self.shut_wells: ShutWells = shut_wells
        self.phases = tuple(str(p).lower() for p in phases)
        feature_names(self.phases)  # validates the phases
        wells = [(w, "producer", tuple(producer_data)) for w in producers] + [
            (w, "injector", tuple(injector_data)) for w in injectors
        ]
        files = [Path(schedule_file)]
        if initial_schedule_file is not None:
            files.insert(0, Path(initial_schedule_file))
        self.wells = well_schedules(files, steps, wells)
        self.networks = [
            WellNetwork(w.n_features(self.phases), int(hidden_size), len(w.outputs), len(steps))
            for w in self.wells
        ]
        offsets = np.cumsum([0] + [n.n_params for n in self.networks])
        self._slices = [slice(int(a), int(b)) for a, b in itertools.pairwise(offsets)]
        self.scales: dict[str, float] | None = None

    @property
    def summary_vectors(self) -> list[tuple[str, str]]:
        """Return the ``(keyword, well)`` vectors the deck's summary must hold for the inputs.

        Every well's ``WBHP``, for its feature ``PRESS_DIFF`` (the well's bottom-hole
        pressure minus each perforated block's pressure), whether or not it is observed.
        """
        return [("WBHP", well.name) for well in self.wells]

    @property
    def n_params(self) -> int:
        """Parameters of every network, ``sum 4H(I + H) + 8H + L(OH + O)``."""
        return sum(n.n_params for n in self.networks)

    def fit(self, observations: ObservationSet) -> None:
        """Set each output's scale ``u = 0.9999 max_t d_obs`` (PET-fork's upper bound).

        Over every observation of that well and type, held-out (cross-validation or test)
        data included, as PET-fork's: the scale only bounds the residual's range.

        Raises
        ------
        ValueError
            If an output has no observation.
        """
        scales: dict[str, float] = {}
        for well in self.wells:
            for output in well.outputs:
                datatype = f"{output}:{well.name}"
                values = [
                    v
                    for k, v in zip(observations.keys, observations.values, strict=True)
                    if k.datatype == datatype
                ]
                if not values:
                    raise ValueError(f"pg_lstm: no observation of {datatype}")
                scales[datatype] = 0.9999 * float(max(values))
                if scales[datatype] <= 0:
                    # u (2 s - 1) with u <= 0 would give no residual, or one of the wrong
                    # sign (the manuscript's |d_max| agrees with max d only for d >= 0).
                    raise ValueError(
                        f"pg_lstm: {datatype}'s observations are never positive, so its "
                        "residual scale 0.9999 max(d_obs) is not positive; leave it out"
                    )
        self.scales = scales

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        """Return the residual key ``MEC:<type>`` of each key a well's output covers."""
        modelled = {f"{o}:{w.name}" for w in self.wells for o in w.outputs}
        return [
            DataKey(k.time, RESIDUAL_PREFIX + k.datatype)
            for k in keys
            if k.index is None and k.datatype in modelled
        ]

    def features(self, files: CaseFiles) -> list[NDArray[np.float64]]:
        """Return each well's input sequence ``(n_steps, n_features)`` from a simulated case."""
        return well_features(files, self.wells, self.report_steps, self.phases)

    def well_residuals(
        self, sequences: Sequence[NDArray[np.float64]], theta: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        """Each output's residuals over the report times, ``{"WOPR:BR-P-5": (n_steps,)}``."""
        if self.scales is None:
            raise ValueError("pg_lstm: fit() must see the observations first")
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"pg_lstm: {self.state_variable} must have {self.n_params} values, "
                f"got {theta.shape}"
            )
        result = {}
        for well, network, part, x in zip(
            self.wells, self.networks, self._slices, sequences, strict=True
        ):
            scale = np.array([self.scales[f"{o}:{well.name}"] for o in well.outputs])
            open_mask = np.ones((len(self.report_steps), len(well.outputs)), dtype=bool)
            if self.shut_wells == "zero":
                for k, output in enumerate(well.outputs):
                    if output != "WBHP":  # BHP is measured even while the well is shut
                        open_mask[:, k] = well.open
            r = well_residuals(network, x[:, :, None], theta[part, None], scale, open_mask)[:, :, 0]
            for k, output in enumerate(well.outputs):
                result[f"{output}:{well.name}"] = r[:, k]
        return result

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        """Return r at ``keys`` for one simulated member (0 for keys no network covers).

        A ``MEC:<type>`` key gives the residual of ``<type>`` itself.

        Raises
        ------
        SimulationError
            If the case's files lack a feature, or a covered key is not at a report time.
        """
        files = CaseFiles(case)
        series = self.well_residuals(self.features(files), theta)
        dates = [dt.date.fromisoformat(d) for d in files.report_dates(self.report_steps)]
        if len(set(dates)) != len(dates):
            # Keys are dates: two report steps on one day could not be told apart.
            raise SimulationError(
                f"pg_lstm: report steps {list(self.report_steps)} fall on {len(set(dates))} "
                "distinct days; data keys need one report step per day"
            )
        times = {date: t for t, date in enumerate(dates)}
        values = np.zeros(len(keys))
        for i, key in enumerate(keys):
            datatype = key.datatype.removeprefix(RESIDUAL_PREFIX)
            if key.index is not None or datatype not in series:
                continue
            try:
                t = times[dt.date.fromisoformat(key.time)]
            except (ValueError, KeyError):
                raise SimulationError(
                    f"pg_lstm: {key} is not at one of the network's report times"
                ) from None
            values[i] = series[datatype][t]
        return values
