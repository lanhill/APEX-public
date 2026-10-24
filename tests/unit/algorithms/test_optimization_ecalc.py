"""NPV objectives with water priced per well and CO2 from eCalc.

Independent references: a hand calculation of NPV with the emissions of a one-consumer
eCalc model (fuel ``0.01 GAS_PROD + 2 WATER_INJ`` Sm3/d, CO2 2 kg/Sm3), through libecalc
itself; and, for ``ren_npv_co2``, the corrections of PET's two bugs (the discount from the
start, the rates dated at their step's start). Parity with PET is in tests/parity.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.optimization import (
    EcalcEmissionTotal,
    EcalcNetPresentValue,
    EcalcParetoNetPresentValue,
    RateTable,
    RenewableCO2NetPresentValue,
    RenewableNetPresentValue,
    response_objectives,
)

pytest.importorskip("libecalc")

START, TIMES = "2025-01-01", ["2026-01-01", "2027-01-01", "2028-01-01"]
DAYS = np.array([365.0, 365.0, 365.0])  # 2025, 2026, 2027
SINCE_START = np.cumsum(DAYS)


def _model(folder: Path, start: str = START, end: str = TIMES[-1]) -> str:
    (folder / "rates.csv").write_text(f"DATE,OIL_PROD,GAS_PROD,WATER_INJ\n{start},0,0,0\n")
    path = folder / "model.yaml"
    path.write_text(
        f"START: {start}\nEND: {end}\nTIME_SERIES:\n  - NAME: SIM\n    FILE: rates.csv\n"
        "    TYPE: DEFAULT\nFUEL_TYPES:\n  - NAME: fuel_gas\n    EMISSIONS:\n"
        "      - NAME: CO2\n        FACTOR: 2.0\nINSTALLATIONS:\n  - NAME: platform\n"
        "    HCEXPORT: SIM;OIL_PROD\n    FUEL: fuel_gas\n    CATEGORY: FIXED\n"
        "    FUELCONSUMERS:\n      - NAME: turbine\n        CATEGORY: TURBINE-GENERATOR\n"
        "        ENERGY_USAGE_MODEL:\n          TYPE: DIRECT\n"
        "          FUELRATE: SIM;GAS_PROD {*} 0.01 {+} SIM;WATER_INJ {*} 2\n"
    )
    return str(path)


def _responses(rng: np.random.Generator, n: int, extra: int = 0) -> NDArray[np.float64]:
    """FOPT, FGPT, FWPT, FWIT at the three times, TIME, then ``extra`` vectors' values."""
    totals = np.cumsum(rng.uniform(100.0, 1e4, (4, 3, n)), axis=1).reshape(12, n)
    time = np.full((1, n), DAYS[0])
    return np.vstack([totals, time, rng.uniform(100.0, 300.0, (extra * 3, n))])


def test_ecalc_npv_prices_the_co2_ecalc_computes(tmp_path: Path) -> None:
    objective = EcalcNetPresentValue(
        start=START, times=TIMES, ecalc_model_file=_model(tmp_path), obj_scaling=-1e6,
        wop=300.0, wgp=0.4, wwp=30.0, wwi=10.0, wem=80.0, disc=0.08,
    )  # fmt: skip
    responses = _responses(np.random.default_rng(0), 2)
    steps = np.diff(responses[:12].reshape(4, 3, 2), axis=1, prepend=0.0)  # (4, 3, n)
    gas, water_inj = steps[1] / DAYS[:, None], steps[3] / DAYS[:, None]
    co2 = (0.01 * gas + 2.0 * water_inj) * 2.0 / 1000.0 * DAYS[:, None]  # tonnes per step
    cash = (300.0 * steps[0] + 0.4 * steps[1] - 30.0 * steps[2] - 10.0 * steps[3] - 80.0 * co2) / (
        1.08 ** (SINCE_START / 365.0)
    )[:, None]
    np.testing.assert_allclose(objective.evaluate(responses), cash.sum(axis=0) / -1e6, rtol=1e-10)
    assert {"ecalc_npv", "ecalc_pareto_npv", "ren_npv", "ren_npv_co2"} <= set(
        response_objectives.names()
    )


def test_the_model_must_span_the_optimization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="START and END"):
        EcalcNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0,
            ecalc_model_file=_model(tmp_path, end="2030-01-01"),
        )  # fmt: skip
    with pytest.raises(ValueError, match="no time series"):
        EcalcNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=_model(tmp_path),
            time_series="other.csv",
        )  # fmt: skip
    with pytest.raises(ValueError, match="no eCalc model"):
        EcalcNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=str(tmp_path / "x.yaml")
        )


def test_the_start_is_checked_by_time(tmp_path: Path) -> None:
    objective = EcalcNetPresentValue(
        start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=_model(tmp_path)
    )
    responses = _responses(np.random.default_rng(1), 1)
    responses[12] = 100.0  # TIME: not 365 days after START
    with pytest.raises(ValueError, match="simulation's start"):
        objective.evaluate(responses)


def test_pareto_needs_exactly_one_of_wemc_and_w(tmp_path: Path) -> None:
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="one of wemc"):
        EcalcParetoNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=model
        )
    with pytest.raises(ValueError, match="one of wemc"):
        EcalcParetoNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=model, wemc=1.0, w=0.5
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EcalcParetoNetPresentValue(
            start=START, times=TIMES, wop=1.0, wem=1.0, ecalc_model_file=model, w=1.5
        )


class _Recorded(RenewableCO2NetPresentValue):
    """``ren_npv_co2`` with a CO2 rate of 1 t/d, keeping the tables it handed to eCalc."""

    tables: list[RateTable]

    def _emissions(self, table: RateTable) -> NDArray[np.float64]:
        self.tables = [*getattr(self, "tables", []), table]
        return np.ones(len(table.starts))


def test_ren_npv_co2_corrects_pets_discount_and_dates(tmp_path: Path) -> None:
    """Discounted by the days since the start (PET: each step's own length), the rates dated
    at their step's start (PET: its end), THP_MAX the injectors' largest WTHP at the step's
    end, and WIND_POWER minus the wind power.
    """
    np.save(tmp_path / "wind.npy", np.array([5.0, 6.0, 7.0]))
    objective = _Recorded(
        start=START, times=TIMES, ecalc_model_file=_model(tmp_path), injectors=["i1", "i2"],
        wind_power_file=str(tmp_path / "wind.npy"), wop=100.0, wem=50.0, disc=0.1,
    )  # fmt: skip
    responses = _responses(np.random.default_rng(2), 1, extra=2)
    value = objective.evaluate(responses)[0]
    steps = np.diff(responses[:12, 0].reshape(4, 3), axis=1, prepend=0.0)
    expected = np.sum((100.0 * steps[0] - 50.0 * DAYS) / 1.1 ** (SINCE_START / 365.0))
    assert value == pytest.approx(expected, rel=1e-12)
    table = objective.tables[0]
    assert table.starts == (dt.date(2025, 1, 1), dt.date(2026, 1, 1), dt.date(2027, 1, 1))
    wthp = responses[13:, 0].reshape(2, 3)
    np.testing.assert_array_equal(table.columns["THP_MAX"], wthp.max(axis=0))
    np.testing.assert_array_equal(table.columns["WIND_POWER"], [-5.0, -6.0, -7.0])


def test_ren_npv_co2_refuses_a_wrong_wind_series(tmp_path: Path) -> None:
    np.save(tmp_path / "wind.npy", np.array([5.0, 6.0]))
    with pytest.raises(ValueError, match="one per report step"):
        RenewableCO2NetPresentValue(
            start=START, times=TIMES, ecalc_model_file=_model(tmp_path), injectors=["i1"],
            wind_power_file=str(tmp_path / "wind.npy"), wop=1.0, wem=1.0,
        )  # fmt: skip


def test_ren_npv_prices_each_injector() -> None:
    objective = RenewableNetPresentValue(
        start=START, times=TIMES, wop=10.0, wwi=2.0, wrenwi=0.5,
        injectors=["a"], renewable_injectors=["b"], disc=0.0,
    )  # fmt: skip
    keys = [f"{k.datatype}@{k.time}" for k in objective.response_keys()]
    assert keys[:3] == [f"FOPT@{t}" for t in TIMES]
    assert keys[3] == f"TIME@{TIMES[0]}"
    assert keys[4:] == [f"WWIT:{w}@{t}" for w in ("a", "b") for t in TIMES]
    oil = np.array([[1.0], [3.0], [6.0]])  # totals: steps of 1, 2, 3
    responses = np.vstack([oil, [[DAYS[0]]], oil * 2, oil * 4])  # WWIT a, b
    # 10 * 6 (oil) - 2 * 12 (a) - 0.5 * 24 (b), undiscounted
    assert objective.evaluate(responses)[0] == pytest.approx(60.0 - 24.0 - 12.0)
    with pytest.raises(ValueError, match="both in"):
        RenewableNetPresentValue(
            start=START, times=TIMES, wop=1.0, injectors=["a"], renewable_injectors=["a"]
        )


def test_unequal_steps_and_an_affine_fuel_rate(tmp_path: Path) -> None:
    """A constant fuel rate makes the per-day division matter (review of 2026-10-04): with
    steps of 90, 275 and 366 days, CO2 = (5 + 0.01 GAS + 2 WATER_INJ) Sm3/d * 2 kg/Sm3.
    """
    times = ["2025-04-01", "2026-01-01", "2027-01-02"]
    days = np.array([90.0, 275.0, 366.0])
    path = Path(_model(tmp_path, end=times[-1]))
    path.write_text(path.read_text().replace("FUELRATE: SIM", "FUELRATE: 5 {+} SIM"))
    objective = EcalcNetPresentValue(
        start=START, times=times, ecalc_model_file=str(path), wop=300.0, wem=80.0, disc=0.08
    )
    responses = _responses(np.random.default_rng(3), 2)
    responses[12] = days[0]
    steps = np.diff(responses[:12].reshape(4, 3, 2), axis=1, prepend=0.0)
    rate = 5.0 + 0.01 * steps[1] / days[:, None] + 2.0 * steps[3] / days[:, None]
    co2 = rate * 2.0 / 1000.0 * days[:, None]
    cash = (300.0 * steps[0] - 80.0 * co2) / (1.08 ** (np.cumsum(days) / 365.0))[:, None]
    np.testing.assert_allclose(objective.evaluate(responses), cash.sum(axis=0), rtol=1e-10)


def test_a_second_time_series_and_the_emission_name(tmp_path: Path) -> None:
    """The model's other time series are read as usual; the emission is chosen by name."""
    # A row per step: eCalc does not extend a DEFAULT series past its last date.
    (tmp_path / "other.csv").write_text("DATE,POWER\n2025-01-01,7\n2026-01-01,7\n2027-01-01,7\n")
    path = Path(_model(tmp_path))
    text = (
        path.read_text()
        .replace(
            "    TYPE: DEFAULT\n",
            "    TYPE: DEFAULT\n  - NAME: OTH\n    FILE: other.csv\n    TYPE: DEFAULT\n",
            1,
        )
        .replace("FUELRATE: SIM;GAS_PROD", "FUELRATE: OTH;POWER {+} SIM;GAS_PROD")
    )
    path.write_text(text.replace("NAME: CO2", "NAME: co2_fuel_gas"))
    objective = EcalcNetPresentValue(
        start=START, times=TIMES, ecalc_model_file=str(path), wop=1.0, wem=1.0,
        time_series="rates.csv", emission="co2_fuel_gas",
    )  # fmt: skip
    responses = _responses(np.random.default_rng(4), 1)
    steps = np.diff(responses[:12].reshape(4, 3, 1), axis=1, prepend=0.0)
    rate = 7.0 + 0.01 * steps[1, :, 0] / DAYS + 2.0 * steps[3, :, 0] / DAYS
    co2 = rate * 2.0 / 1000.0 * DAYS
    cash = (steps[0, :, 0] - co2) / 1.0
    assert objective.evaluate(responses)[0] == pytest.approx(cash.sum(), rel=1e-10)
    wrong = EcalcNetPresentValue(
        start=START, times=TIMES, ecalc_model_file=str(path), wop=1.0, wem=1.0,
        time_series="rates.csv",
    )  # fmt: skip
    with pytest.raises(ValueError, match="co2_fuel_gas"):
        wrong.evaluate(responses)
    with pytest.raises(ValueError, match="name the one"):
        EcalcNetPresentValue(start=START, times=TIMES, ecalc_model_file=str(path), wop=1.0, wem=1.0)


def test_rates_must_be_finite_and_totals_must_not_fall(tmp_path: Path) -> None:
    objective = EcalcNetPresentValue(
        start=START, times=TIMES, ecalc_model_file=_model(tmp_path), wop=1.0, wem=1.0
    )
    responses = _responses(np.random.default_rng(5), 1)
    falling = responses.copy()
    falling[5] = falling[4] - 1.0  # FGPT falls in the last step
    with pytest.raises(ValueError, match="FGPT decreases"):
        objective.evaluate(falling)
    np.save(tmp_path / "wind.npy", np.array([1.0, 2.0, 3.0]))
    ren = RenewableCO2NetPresentValue(
        start=START, times=TIMES, ecalc_model_file=_model(tmp_path), injectors=["i1"],
        wind_power_file=str(tmp_path / "wind.npy"), wop=1.0, wem=1.0,
    )  # fmt: skip
    with_thp = _responses(np.random.default_rng(6), 1, extra=1)
    with_thp[14] = np.nan  # a WTHP missing
    with pytest.raises(ValueError, match="THP_MAX is not finite"):
        ren.evaluate(with_thp)
    with pytest.raises(ValueError, match="without wells"):
        RenewableNetPresentValue(start=START, times=TIMES, wop=1.0, wwi=2.0)


def test_ecalc_emissions_are_the_tonnes_ecalc_computes(tmp_path: Path) -> None:
    """The second objective next to an NPV. By hand: the turbine burns
    ``0.01 GAS_PROD + 2 WATER_INJ`` Sm3/d at 2 kg CO2 per Sm3, so a step emits
    ``2e-3 (0.01 g + 2 w) Δd`` t for its daily rates g and w; and the same tonnes, priced at
    ``wem = 1`` with no other price, are ``-ecalc_npv`` (discounted alike)."""
    responses = _responses(np.random.default_rng(7), 3)
    model = _model(tmp_path)
    tonnes = EcalcEmissionTotal(start=START, times=TIMES, ecalc_model_file=model)
    steps = np.diff(responses[:12].reshape(4, 3, 3), axis=1, prepend=0.0)
    gas, water_inj = steps[1] / DAYS[:, None], steps[3] / DAYS[:, None]
    by_hand = np.sum(2e-3 * (0.01 * gas + 2.0 * water_inj) * DAYS[:, None], axis=0)
    np.testing.assert_allclose(tonnes.evaluate(responses), by_hand, rtol=1e-9)
    discounted = EcalcEmissionTotal(start=START, times=TIMES, ecalc_model_file=model,
                                    disc=0.08, obj_scaling=1e3)  # fmt: skip
    priced = EcalcNetPresentValue(start=START, times=TIMES, ecalc_model_file=model, wop=0.0,
                                  wem=1.0, disc=0.08, obj_scaling=1e3)  # fmt: skip
    np.testing.assert_allclose(discounted.evaluate(responses), -priced.evaluate(responses),
                               rtol=1e-12)  # fmt: skip
    assert "ecalc_emissions" in response_objectives.names()
