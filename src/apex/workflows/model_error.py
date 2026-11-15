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
             recorded in docs/mec.md). ``causal_mask``, ``save_inputs`` and ``report_dates``
             are APEX's options, not PET-fork's: off by default,
             and runs without them are unchanged.
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
from apex.core.simulation import BlockStates
from apex.domain.observations import DataKey, ObservationSet
from apex.simulation.model_error import (
    INPUT_PREFIX,
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
        The deck's phases, one or more of oil, water and gas: they set the dynamic features,
        each phase's density and saturation (:func:`~apex.simulation.model_error.feature_names`;
        without oil, its density and saturation are 0).
    report_dates
        The ISO date of each report step, in order. Needed by
        ``causal_mask`` and ``save_inputs``, which must place a datum at its report step
        before any run; every case's own report dates are checked against them.
    causal_mask
        Declare the network's structure to the run's localization (``MaskingModel``):
        a well's LSTM acts on its own outputs at every report time,
        report time ``t``'s output layer (each output's row and bias) on that output at
        ``t`` only. Needs ``adaptive_prior``; ``nn_params`` in its ``unlocalized`` list
        gives the mask alone.
    save_inputs
        Also answer the inputs, ``MEC:X<j>:<well>`` at each report date (feature ``j`` of
        the well's inputs), saved with the residuals: a run of the prior gives a per-step
        prior's sequential pass its inputs.

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
        report_dates: TextList = (),
        causal_mask: bool = False,
        save_inputs: bool = False,
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
        self.report_dates = _report_dates(report_dates, len(steps))
        self.causal_mask = bool(causal_mask)
        self.save_inputs = bool(save_inputs)
        for name, wanted in (("causal_mask", self.causal_mask), ("save_inputs", self.save_inputs)):
            if wanted and not self.report_dates:
                raise ValueError(
                    f"pg_lstm: {name} needs report_dates, the ISO date of each of the "
                    f"{len(steps)} report steps"
                )

    @property
    def masks_parameters(self) -> bool:
        """Whether the run's localization takes :meth:`parameter_mask` (``causal_mask``)."""
        return self.causal_mask

    def parameter_mask(self, keys: Sequence[DataKey]) -> NDArray[np.bool_]:
        """Return which data each parameter can affect, ``(n_params, len(keys))``.

        The network's structure, exact: a well's parameters act on
        its own outputs only; its LSTM (shared over the report times) on each of them at
        every report time; report time ``t``'s output layer, its row ``k`` of ``fc_t.weight``
        and ``fc_t.bias[k]``, on output ``k`` at ``t`` only. A key the model does not
        correct (another well's, a field datum) is no parameter's.

        Raises
        ------
        ValueError
            If a corrected key's date is not one of ``report_dates``.
        """
        if not self.report_dates:
            raise ValueError("pg_lstm: the parameter mask needs report_dates")
        position = {date: t for t, date in enumerate(self.report_dates)}
        mask = np.zeros((self.n_params, len(keys)), dtype=bool)
        for well, network, part in zip(self.wells, self.networks, self._slices, strict=True):
            h, o = network.hidden_size, network.output_size
            lstm = 4 * h * (network.input_size + h) + 8 * h
            for i, key in enumerate(keys):
                output, _, name = key.datatype.partition(":")
                if key.index is not None or name != well.name or output not in well.outputs:
                    continue
                t = position.get(key.time[:10])
                if t is None:
                    raise ValueError(
                        f"pg_lstm: {key} is not at one of the report dates {self.report_dates}"
                    )
                k = well.outputs.index(output)
                mask[part.start : part.start + lstm, i] = True
                head = part.start + lstm + t * (o * h + o)
                mask[head + k * h : head + (k + 1) * h, i] = True  # row k of fc_t.weight
                mask[head + o * h + k, i] = True  # fc_t.bias[k]
        return mask

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
        """Return the residual key ``MEC:<type>`` of each key a well's output covers.

        With ``save_inputs``, also each report date's inputs, ``MEC:X<j>:<well>``.
        """
        modelled = {f"{o}:{w.name}" for w in self.wells for o in w.outputs}
        found = [
            DataKey(k.time, RESIDUAL_PREFIX + k.datatype)
            for k in keys
            if k.index is None and k.datatype in modelled
        ]
        if self.save_inputs:
            found += [
                DataKey(date, f"{RESIDUAL_PREFIX}{INPUT_PREFIX}{j}:{well.name}")
                for well, network in zip(self.wells, self.networks, strict=True)
                for date in self.report_dates
                for j in range(network.input_size)
            ]
        return found

    def features(self, files: BlockStates) -> list[NDArray[np.float64]]:
        """Return each well's input sequence ``(n_steps, n_features)`` from a member's states."""
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
        return self.state_residuals(self.case_states(case), theta, keys)

    def case_states(self, case: Path) -> BlockStates:
        """Return a simulated case's block states: its INIT, restart and summary files."""
        return CaseFiles(case)

    def state_residuals(
        self, states: BlockStates, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        """Return :meth:`residuals` from a member's block states.

        The same features, from a simulated case's files or a surrogate's predicted states.

        Raises
        ------
        SimulationError
            If the states lack a feature, or a covered key is not at a report time.
        """
        sequences = self.features(states)
        series = self.well_residuals(sequences, theta)
        dates = [dt.date.fromisoformat(d) for d in states.report_dates(self.report_steps)]
        if self.report_dates and [d.isoformat() for d in dates] != list(self.report_dates):
            # The mask and the saved inputs place data by these dates: a case reporting
            # others would put each datum at another report time's parameters.
            raise SimulationError(
                f"pg_lstm: {states.label} reports {[d.isoformat() for d in dates]} at the "
                f"report steps {list(self.report_steps)}, not report_dates "
                f"{list(self.report_dates)}"
            )
        if len(set(dates)) != len(dates):
            # Keys are dates: two report steps on one day could not be told apart.
            raise SimulationError(
                f"pg_lstm: report steps {list(self.report_steps)} fall on {len(set(dates))} "
                "distinct days; data keys need one report step per day"
            )
        times = {date: t for t, date in enumerate(dates)}
        inputs = {w.name: x for w, x in zip(self.wells, sequences, strict=True)}
        values = np.zeros(len(keys))
        for i, key in enumerate(keys):
            datatype = key.datatype.removeprefix(RESIDUAL_PREFIX)
            feature = _input_feature(key) if self.save_inputs else None
            if feature is not None:
                j, name = feature
                if name in inputs:
                    try:
                        t = times[dt.date.fromisoformat(key.time)]
                    except (ValueError, KeyError):
                        raise SimulationError(
                            f"pg_lstm: {key} is not at one of the network's report times"
                        ) from None
                    values[i] = inputs[name][t, j]
                continue
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


def _report_dates(dates: Sequence[str], n_steps: int) -> tuple[str, ...]:
    """Return ``report_dates`` as ISO dates, one per report step, increasing (or none).

    Raises
    ------
    ValueError
        If they are not ``n_steps`` increasing ISO dates.
    """
    if not dates:
        return ()
    try:
        parsed = [dt.date.fromisoformat(str(d)) for d in dates]
    except ValueError:
        raise ValueError(f"pg_lstm: report_dates must be ISO dates, got {list(dates)}") from None
    if len(parsed) != n_steps or any(b <= a for a, b in itertools.pairwise(parsed)):
        raise ValueError(
            f"pg_lstm: report_dates must be {n_steps} increasing dates, one per report step, "
            f"got {list(dates)}"
        )
    return tuple(d.isoformat() for d in parsed)


def _input_feature(key: DataKey) -> tuple[int, str] | None:
    """Return ``(j, well)`` of a saved-input key ``MEC:X<j>:<well>``, else ``None``."""
    if key.index is not None or not key.datatype.startswith(RESIDUAL_PREFIX + INPUT_PREFIX):
        return None
    name, _, well = key.datatype.removeprefix(RESIDUAL_PREFIX + INPUT_PREFIX).partition(":")
    return (int(name), well) if name.isdigit() and well else None
