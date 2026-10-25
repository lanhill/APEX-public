"""NPV objectives with CO2 emissions from eCalc.

``ecalc_emissions`` is APEX's, not PET's: the emitted tonnes alone, a second
objective next to an NPV for ``nsga2``.

Ported from PET
---------------
PET module : popt/cost_functions/ecalc_npv.py (``ecalc_npv``),
             popt/cost_functions/ecalc_pareto_npv.py (``ecalc_pareto_npv``),
             popt/cost_functions/ren_npv_co2.py (``ren_npv_co2``)
PET commit : fd17979dde98928e9a7191d7a251be4ca59ed50a (removed before e65fb10)
Reference  : as ``npv`` (Brouwer & Jansen, 2004), with the cost of the CO2 emitted while
             producing, computed by Equinor's eCalc (``libecalc``) from the field's rates.
Differences: (common)
             - eCalc 15 replaces the 8.23 PET pins; its API changed, not the economics. The
               emissions are the asset's rate of the emission ``emission`` (default ``co2``;
               every fuel and venting emitter of the model) per report step, where PET
               summed the ``co2_fuel_gas`` rate of the components eCalc returned.
             - The rates are given to eCalc in memory, not through ``ecalc_input.csv`` in
               the working folder; the model's START and END must be the optimization's
               start and last report time, so eCalc's periods are the report steps (checked;
               PET assumed it).
             - The start is checked by the summary's TIME, as ``npv``.
             (``ecalc_pareto_npv``) one of ``wemc`` or ``w`` is required (PET added an empty
             array without either, which returns nothing); the per-objective values PET saved
             to ``pareto_values.npz`` are not saved.
             (``ren_npv_co2``) two PET bugs are not reproduced (docs/migration/pet_parity.md):
             PET discounts each step by its own length, ``(1 + d)^(Δdays_k / 365)``, here by
             the days since the start as every NPV; and PET dates each step's rates at the
             step's end, which eCalc reads as the next step's, here at its start. The
             injectors' maximum tubing-head pressure (THP_MAX) is the one reported at the
             step's end (a summary has none at the start; PET read the step's start). Wind
             power is one series for every simulation (``wind_power_file``), where PET read
             one per ensemble member from a file its renewable simulator wrote.

For report times ``t_1 < ... < t_K`` after the start ``t_0``, with ``ΔQ_k`` the increases of
the field totals over step ``k`` and ``Δd_k`` its days, eCalc gets the rates ``ΔQ_k / Δd_k``
(``OIL_PROD``, ``GAS_PROD``, ``WATER_INJ``; ``ren_npv_co2`` also ``THP_MAX`` and
``WIND_POWER``) and returns the CO2 rate ``e_k`` (t/d); the emitted ``E_k = e_k Δd_k`` (t)
cost ``r_em`` each:

    NPV = sum_k (ΔQ_op r_o + ΔQ_gp r_g - ΔQ_wp r_w - ΔQ_wi r_wi - E_k r_em) / (1 + d)^(T_k / 365)

with ``T_k`` the days from ``t_0`` to ``t_k``. The objective is ``NPV / obj_scaling``.
"""

import datetime as dt
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.optimization.economics import TIME, TIME_TOLERANCE_DAYS
from apex.algorithms.optimization.protocols import FloatArray
from apex.algorithms.optimization.registries import response_objectives
from apex.core.options import TEXT, InputFile, TextList
from apex.domain.observations import DataKey

#: The field totals the eCalc objectives read, in this order.
TOTALS = ("FOPT", "FGPT", "FWPT", "FWIT")


@dataclass(frozen=True)
class RateTable:
    """The rates eCalc gets for one simulation: one row per report step.

    Attributes
    ----------
    starts
        Each step's first day ``(n_steps,)``.
    end
        The last step's end.
    columns
        eCalc's time-series columns, each the step's rate per day ``(n_steps,)``.
    """

    starts: tuple[dt.date, ...]
    end: dt.date
    columns: Mapping[str, NDArray[np.float64]]


class EcalcEmissions:
    """CO2 rates (t/d) per report step from an eCalc model, given the step rates.

    Parameters
    ----------
    model_file
        The eCalc model (YAML). Its START and END must be the optimization's start and
        last report time.
    time_series
        The model's time-series file whose data the simulations replace (default: its only
        one).
    start, end
        The optimization's start and last report time.
    emission
        The emission priced: its name in the model's fuels and emitters (eCalc lower-cases
        it; PET's cases named it ``co2_fuel_gas``).

    Raises
    ------
    ValueError
        If ``libecalc`` is missing, the model does not load, or its time span or time
        series do not fit.
    """

    def __init__(
        self,
        model_file: Path,
        time_series: str | None,
        start: dt.date,
        end: dt.date,
        emission: str = "co2",
    ) -> None:
        if importlib.util.find_spec("libecalc") is None:
            raise ValueError("eCalc objectives need libecalc: install APEX's 'ecalc' extra")
        from libecalc.presentation.yaml.file_configuration_service import (
            FileConfigurationService,
        )

        self.model_file = Path(model_file)
        if not self.model_file.is_file():
            raise ValueError(f"no eCalc model at {self.model_file}")
        self._configuration = FileConfigurationService(self.model_file).get_configuration()
        names = list(self._configuration.timeseries_resource_names)
        if time_series is None:
            if len(names) != 1:
                raise ValueError(
                    f"the eCalc model has time series {names}: name the one the simulations "
                    "replace (time_series)"
                )
            time_series = names[0]
        elif time_series not in names:
            raise ValueError(f"the eCalc model has no time series {time_series!r} ({names})")
        self.time_series = time_series
        self.emission = emission.lower()
        span = (self._configuration.start, self._configuration.end)
        if span[0] is None or span[1] is None or (span[0].date(), span[1].date()) != (start, end):
            raise ValueError(
                f"the eCalc model's START and END ({span[0]}, {span[1]}) must be the "
                f"optimization's start and last report time ({start}, {end})"
            )

    def __call__(self, table: RateTable) -> NDArray[np.float64]:
        """Return eCalc's rate (t/d) of the emission, of each step ``(n_steps,)``.

        Raises
        ------
        ValueError
            If a rate is not finite, or the model does not emit ``emission``.
        RuntimeError
            If eCalc's periods are not the report steps.
        """
        for column, values in table.columns.items():
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                raise ValueError(f"{column} is not finite in report step(s) {bad.tolist()}")
        from ecalc_cli.infrastructure.file_resource_service import FileResourceService
        from libecalc.presentation.json_result.mapper import get_asset_result
        from libecalc.presentation.yaml.domain.time_series_resource import TimeSeriesResource
        from libecalc.presentation.yaml.model import YamlModel
        from libecalc.presentation.yaml.yaml_entities import MemoryResource

        headers = ["DATE", *table.columns]
        data: list[list[float | int | str]] = [[d.isoformat() for d in table.starts]]
        data += [[float(v) for v in values] for values in table.columns.values()]
        resource = TimeSeriesResource(MemoryResource(headers=headers, data=data)).validate()
        name = self.time_series

        class InMemory(FileResourceService):  # type: ignore[misc]
            """The model's files, with this simulation's rates as the replaced time series."""

            def get_time_series_resources(self) -> tuple[dict[str, Any], list[Any]]:
                resources, errors = super().get_time_series_resources()
                errors = [e for e in errors if getattr(e, "resource_name", None) != name]
                return {**resources, name: resource}, errors

        model = YamlModel(
            configuration=self._configuration,
            resource_service=InMemory(self.model_file.parent, configuration=self._configuration),
        ).validate_for_run()
        model.evaluate_energy_usage()
        emissions = get_asset_result(model).components[0].emissions
        if self.emission not in emissions:
            raise ValueError(
                f"the eCalc model emits {sorted(emissions)}, not {self.emission!r} (emission)"
            )
        rate = emissions[self.emission].rate
        periods = [(p.start.date(), p.end.date()) for p in rate.periods.periods]
        steps = list(zip(table.starts, [*table.starts[1:], table.end], strict=True))
        if periods != steps:
            raise RuntimeError(f"eCalc's periods {periods} are not the report steps {steps}")
        return np.asarray(rate.values, dtype=np.float64)


class _EcalcEconomics:
    """The common part: dates, responses, step rates and the eCalc model."""

    def __init__(
        self,
        start: str,
        times: Sequence[str],
        prices: Mapping[str, float],
        ecalc_model_file: str | Path,
        time_series: str | None,
        extra_vectors: Sequence[str] = (),
        emission: str = "co2",
    ) -> None:
        first = dt.date.fromisoformat(start)
        dates = [dt.date.fromisoformat(t) for t in times]
        if not dates:
            raise ValueError("at least one report time is needed")
        if any(b <= a for a, b in zip([first, *dates], dates, strict=False)):
            raise ValueError("the report times must increase, after the start")
        bad = [k for k, v in prices.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"{', '.join(bad)} must be finite")
        if prices["obj_scaling"] == 0:
            raise ValueError("obj_scaling must not be zero")
        if prices["disc"] <= -1:
            raise ValueError(f"disc must be above -1, got {prices['disc']}")
        self.times = tuple(times)
        self.prices = dict(prices)
        self.starts = (first, *dates[:-1])
        self.end = dates[-1]
        days = np.array([(d - first).days for d in dates], dtype=float)
        self.first_days = float(days[0])
        self.step_days = np.diff(days, prepend=0.0)  # Δd_k
        self.discount = (1.0 + prices["disc"]) ** (days / 365.0)  # (1 + d)^(T_k / 365)
        self.extra_vectors = tuple(extra_vectors)
        self.model = EcalcEmissions(Path(ecalc_model_file), time_series, first, self.end, emission)

    def response_keys(self) -> Sequence[DataKey]:
        """Return the four totals, then TIME, then any extra vector, at every report time."""
        totals = [DataKey(t, v) for v in TOTALS for t in self.times]
        extra = [DataKey(t, v) for v in self.extra_vectors for t in self.times]
        return [*totals, DataKey(self.times[0], TIME), *extra]

    def _steps(self, responses: FloatArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the totals' step increases ``(4, n_steps, n)`` and the extras."""
        n_times = len(self.times)
        n_totals = len(TOTALS) * n_times
        expected = n_totals + 1 + len(self.extra_vectors) * n_times
        if responses.ndim != 2 or responses.shape[0] != expected:
            raise ValueError(f"responses must be ({expected}, n), got {responses.shape}")
        elapsed = responses[n_totals]
        if np.any(np.abs(elapsed - self.first_days) > TIME_TOLERANCE_DAYS):
            raise ValueError(
                f"the start is not the simulation's start: {self.times[0]} is "
                f"{self.first_days:g} days after it, the simulation says "
                f"{sorted(set(elapsed.tolist()))} (set start to the deck's START)"
            )
        n = responses.shape[1]
        totals = responses[:n_totals].reshape(len(TOTALS), n_times, n)
        extras = responses[n_totals + 1 :].reshape(len(self.extra_vectors), n_times, n)
        return np.diff(totals, axis=1, prepend=0.0), extras

    def _emissions(self, table: RateTable) -> NDArray[np.float64]:
        """Return the CO2 rate (t/d) per step for one simulation (tests replace it)."""
        return self.model(table)

    def _table(
        self, steps: NDArray[np.float64], j: int, extra: Mapping[str, NDArray[np.float64]]
    ) -> RateTable:
        for i, vector in enumerate(TOTALS):
            falling = np.flatnonzero(steps[i, :, j] < 0)
            if falling.size:
                raise ValueError(f"{vector} decreases in report step(s) {falling.tolist()}")
        oil, gas, _, water_inj = (steps[i, :, j] / self.step_days for i in range(len(TOTALS)))
        columns = {"GAS_PROD": gas, "OIL_PROD": oil, "WATER_INJ": water_inj, **extra}
        return RateTable(self.starts, self.end, columns)

    def _value(self, steps: NDArray[np.float64], j: int, emitted: NDArray[np.float64]) -> float:
        """Return the discounted NPV of simulation ``j`` (unscaled)."""
        p = self.prices
        cash = (
            steps[0, :, j] * p["wop"] + steps[1, :, j] * p["wgp"] - steps[2, :, j] * p["wwp"]
            - steps[3, :, j] * p["wwi"] - emitted * p["wem"]
        )  # fmt: skip
        return float(np.sum(cash / self.discount))


@response_objectives.register("ecalc_npv")
class EcalcNetPresentValue(_EcalcEconomics):
    """NPV of oil, gas, water and the CO2 emitted, the emissions computed by eCalc.

    Parameters
    ----------
    start, times
        The simulation's start and the report dates (ISO), as ``npv``.
    ecalc_model_file
        The eCalc model (YAML), whose START and END are ``start`` and the last time and
        whose time series reads ``OIL_PROD``, ``GAS_PROD`` and ``WATER_INJ`` (per day).
    time_series
        The model's time-series file the simulations replace (default: its only one).
    emission
        The emission priced at ``wem`` (its name in the model; default ``co2``).
    wop, wgp, wwp, wwi
        Prices and costs per unit of FOPT, FGPT, FWPT and FWIT.
    wem
        Cost per tonne of CO2.
    disc, obj_scaling
        As ``npv``.
    """

    status = "parity"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        ecalc_model_file: InputFile,
        wop: float,
        wem: float,
        wgp: float = 0.0,
        wwp: float = 0.0,
        wwi: float = 0.0,
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        time_series: Annotated[str | None, TEXT] = None,
        emission: Annotated[str, TEXT] = "co2",
    ) -> None:
        prices = {"wop": wop, "wgp": wgp, "wwp": wwp, "wwi": wwi, "wem": wem, "disc": disc,
                  "obj_scaling": obj_scaling}  # fmt: skip
        super().__init__(start, times, prices, ecalc_model_file, time_series, emission=emission)

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``NPV / obj_scaling`` of each simulation ``(n,)``."""
        steps, _ = self._steps(responses)
        values = []
        for j in range(steps.shape[2]):
            emitted = self._emissions(self._table(steps, j, {})) * self.step_days
            values.append(self._value(steps, j, emitted))
        return np.asarray(values, dtype=np.float64) / self.prices["obj_scaling"]


@response_objectives.register("ecalc_emissions")
class EcalcEmissionTotal(_EcalcEconomics):
    """The emission eCalc computes, in tonnes over the steps: a second objective.

    With ``npv`` (or ``ecalc_npv``) in ``[[objectives]]``, ``nsga2`` finds the trade-off
    between value and emissions in one run, where ``ecalc_pareto_npv`` folds the two into one
    number with a weight. The value is ``sum_k E_k / (1 + d)^(T_k/365) / obj_scaling``, with
    ``E_k = e_k Δd_k`` the tonnes of step ``k`` (eCalc's rate times the step's days), as
    ``ecalc_npv`` prices them: ``ecalc_emissions`` equals ``-ecalc_npv`` with ``wem = 1`` and
    no other price. Not in PET (its emission objectives are weighted into the NPV).

    Parameters
    ----------
    start, times, ecalc_model_file, time_series, emission
        As ``ecalc_npv``.
    disc
        Discount rate per year (default 0: plain tonnes).
    obj_scaling
        Divides the tonnes (e.g. 1000 for kilotonnes); positive, as emissions are minimised.
    """

    status = "validated"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        ecalc_model_file: InputFile,
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        time_series: Annotated[str | None, TEXT] = None,
        emission: Annotated[str, TEXT] = "co2",
    ) -> None:
        if obj_scaling <= 0:
            # A negative scale (the NPV's habit, to maximise) would maximise the emissions.
            raise ValueError(
                f"obj_scaling must be positive (it divides the tonnes), got {obj_scaling}"
            )
        prices = {"disc": disc, "obj_scaling": obj_scaling}
        super().__init__(start, times, prices, ecalc_model_file, time_series, emission=emission)

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return the (discounted) tonnes emitted by each simulation ``(n,)``, scaled."""
        steps, _ = self._steps(responses)
        values = [
            float(np.sum(self._emissions(self._table(steps, j, {})) * self.step_days
                         / self.discount))
            for j in range(steps.shape[2])
        ]  # fmt: skip
        return np.asarray(values, dtype=np.float64) / self.prices["obj_scaling"]


@response_objectives.register("ecalc_pareto_npv")
class EcalcParetoNetPresentValue(_EcalcEconomics):
    """NPV with eCalc's CO2, combined with an emission objective (PET's two variants).

    ``wemc`` (a CO2 cost correction): the objective is ``NPV/s - sum_k E_k r_emc / (1 +
    d)^(T_k/365) / s`` with ``s = obj_scaling``. ``w`` (the emission intensity): it is
    ``(1 - w) NPV/s + w I``, with ``I = 1000 sum_k E_k / (sum_k (840 ΔQ_op,k + ΔQ_gp,k) /
    1000)`` in kg per tonne of oil equivalent (PET's densities: oil 840, gas 1).

    Parameters
    ----------
    start
        The simulation's start date (ISO, e.g. ``2025-01-01``): the deck's START, checked
        against the summary's TIME; discounting counts the days from it.
    times
        The report dates (ISO), increasing and after ``start``; each is the end of one
        step whose rates eCalc gets.
    ecalc_model_file
        The eCalc model (YAML), whose START and END are ``start`` and the last of
        ``times`` and whose time series reads the step rates (per day).
    time_series
        The name of the model's time series that the simulated rates replace; by default
        the model's only one (required when it has several).
    emission
        The emission priced at ``wem``, by its name in the model's fuels and emitters
        (case-insensitive; default ``co2``, PET's cases used ``co2_fuel_gas``).
    wop
        Oil price per unit of FOPT (PET's ``wop``).
    wgp
        Gas price per unit of FGPT; default 0 leaves gas out of the value.
    wwp
        Cost per unit of water produced (FWPT); default 0 makes it free.
    wwi
        Cost per unit of water injected (FWIT); default 0 makes it free.
    wem
        Cost per tonne of the emission eCalc computes.
    disc
        Discount rate per year (e.g. 0.08 for 8 %), applied by the days since ``start``;
        must be above -1, and the default 0 does not discount.
    obj_scaling
        Divides the NPV (e.g. 1e6 for millions); any nonzero number, default 1.
    wemc
        The CO2 cost correction per tonne (one of ``wemc`` and ``w``).
    w
        The weight of the emission intensity, in ``[0, 1]``.
    """

    status = "parity"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        ecalc_model_file: InputFile,
        wop: float,
        wem: float,
        wgp: float = 0.0,
        wwp: float = 0.0,
        wwi: float = 0.0,
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        wemc: float | None = None,
        w: float | None = None,
        time_series: Annotated[str | None, TEXT] = None,
        emission: Annotated[str, TEXT] = "co2",
    ) -> None:
        if (wemc is None) == (w is None):
            raise ValueError("ecalc_pareto_npv needs one of wemc (a CO2 cost) or w (a weight)")
        if w is not None and not 0.0 <= w <= 1.0:
            raise ValueError(f"w must be in [0, 1], got {w}")
        self.wemc, self.w = wemc, w
        prices = {"wop": wop, "wgp": wgp, "wwp": wwp, "wwi": wwi, "wem": wem, "disc": disc,
                  "obj_scaling": obj_scaling}  # fmt: skip
        super().__init__(start, times, prices, ecalc_model_file, time_series, emission=emission)

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return the combined objective of each simulation ``(n,)``."""
        steps, _ = self._steps(responses)
        scaling = self.prices["obj_scaling"]
        values = []
        for j in range(steps.shape[2]):
            emitted = self._emissions(self._table(steps, j, {})) * self.step_days
            npv = self._value(steps, j, emitted) / scaling
            if self.wemc is not None:
                values.append(npv - float(np.sum(emitted * self.wemc / self.discount)) / scaling)
            else:
                assert self.w is not None
                # PET: kg CO2 per tonne of oil equivalent (oil 840, gas 1 kg/Sm3).
                equivalent = np.sum(steps[0, :, j] * 840.0 + steps[1, :, j] * 1.0) / 1000.0
                if equivalent <= 0:
                    raise ValueError("the emission intensity needs oil or gas produced")
                intensity = float(np.sum(emitted * 1000.0) / equivalent)
                values.append((1.0 - self.w) * npv + self.w * intensity)
        return np.asarray(values, dtype=np.float64)


@response_objectives.register("ren_npv_co2")
class RenewableCO2NetPresentValue(_EcalcEconomics):
    """NPV with eCalc's CO2, where wind power and injection pressure enter eCalc.

    eCalc also gets ``THP_MAX``, the largest tubing-head pressure (WTHP) of ``injectors``
    at each step's end, and ``WIND_POWER``, minus the wind power of each step (PET's sign:
    power produced is negative consumption).

    Parameters
    ----------
    start
        The simulation's start date (ISO, e.g. ``2025-01-01``): the deck's START, checked
        against the summary's TIME; discounting counts the days from it.
    times
        The report dates (ISO), increasing and after ``start``; each is the end of one
        step whose rates eCalc gets.
    ecalc_model_file
        The eCalc model (YAML), whose START and END are ``start`` and the last of
        ``times`` and whose time series reads the step rates (per day).
    time_series
        The name of the model's time series that the simulated rates replace; by default
        the model's only one (required when it has several).
    emission
        The emission priced at ``wem``, by its name in the model's fuels and emitters
        (case-insensitive; default ``co2``, PET's cases used ``co2_fuel_gas``).
    wop
        Oil price per unit of FOPT (PET's ``wop``).
    wgp
        Gas price per unit of FGPT; default 0 leaves gas out of the value.
    wwp
        Cost per unit of water produced (FWPT); default 0 makes it free.
    wwi
        Cost per unit of water injected (FWIT); default 0 makes it free.
    wem
        Cost per tonne of the emission eCalc computes.
    disc
        Discount rate per year (e.g. 0.08 for 8 %), applied by the days since ``start``;
        must be above -1, and the default 0 does not discount.
    obj_scaling
        Divides the NPV (e.g. 1e6 for millions); any nonzero number, default 1.
    injectors
        The injectors whose WTHP gives ``THP_MAX``.
    wind_power_file
        ``.npy`` of the wind power available in each report step ``(n_steps,)``.
    """

    status = "parity"

    def __init__(
        self,
        start: Annotated[str, TEXT],
        times: TextList,
        ecalc_model_file: InputFile,
        injectors: TextList,
        wind_power_file: InputFile,
        wop: float,
        wem: float,
        wgp: float = 0.0,
        wwp: float = 0.0,
        wwi: float = 0.0,
        disc: float = 0.0,
        obj_scaling: float = 1.0,
        time_series: Annotated[str | None, TEXT] = None,
        emission: Annotated[str, TEXT] = "co2",
    ) -> None:
        if not injectors:
            raise ValueError("ren_npv_co2 needs the injectors whose WTHP gives THP_MAX")
        wind = np.asarray(np.load(wind_power_file), dtype=np.float64)
        if wind.shape != (len(times),) or not np.all(np.isfinite(wind)):
            raise ValueError(
                f"{wind_power_file}: wind power must be finite, one per report step "
                f"({len(times)},), got {wind.shape}"
            )
        self.wind = wind
        self.injectors = tuple(injectors)
        prices = {"wop": wop, "wgp": wgp, "wwp": wwp, "wwi": wwi, "wem": wem, "disc": disc,
                  "obj_scaling": obj_scaling}  # fmt: skip
        super().__init__(
            start, times, prices, ecalc_model_file, time_series,
            extra_vectors=[f"WTHP:{w}" for w in injectors], emission=emission,
        )  # fmt: skip

    def evaluate(self, responses: FloatArray) -> FloatArray:
        """Return ``NPV / obj_scaling`` of each simulation ``(n,)``."""
        steps, extras = self._steps(responses)  # extras: (n_injectors, n_steps, n)
        values = []
        for j in range(steps.shape[2]):
            extra = {"THP_MAX": extras[:, :, j].max(axis=0), "WIND_POWER": -self.wind}
            emitted = self._emissions(self._table(steps, j, extra)) * self.step_days
            values.append(self._value(steps, j, emitted))
        return np.asarray(values, dtype=np.float64) / self.prices["obj_scaling"]
