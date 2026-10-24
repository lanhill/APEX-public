"""OPM Flow's CO2STORE as APEX reads it (opt-in: ``-m integration``).

Two small decks of APEX's own (``tests/integration/data/co2store``), one in each of
CO2STORE's forms, CO₂ injected into brine at 40 °C, with a capillary pressure. Measured
on OPM Flow 2026.04 (the assumptions of revision 1, pinned):

- the restarts hold what the forward model reads with ``RPTRST BASIC=2 DEN``: SGAS,
  PRESSURE, GAS_DEN and the brine's density, WAT_DEN in the gas-water form, OIL_DEN
  in the oil-gas form (whose "oil" is brine);
- OPM's CO₂ density is Span & Wagner's at the cell's CO₂ pressure (CO2STORE tabulates
  it): to 1e-5 at the reported pressure in the gas-water form, whose PRESSURE is the
  CO₂'s; in the oil-gas form PRESSURE is the brine's, and the CO₂'s is above it by the
  capillary pressure;
- soft_sand with CO₂'s equation of state forecasts both forms, its CO₂ consistent with
  the simulator's in the gas-water form.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.geophysics import FlowState, SoftSandModel
from apex.core.simulation import MemberInput
from apex.simulation.geophysics import CaseState
from apex.simulation.opm_flow import OPMFlowSimulator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]
CoolProp = pytest.importorskip("CoolProp.CoolProp")

DECKS = Path(__file__).resolve().parent / "data" / "co2store"
KELVIN = 313.15  # the decks' RTEMPVD, 40 C at every depth


@pytest.fixture(scope="module", params=["gas_water", "oil_gas"])
def run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    form: str = request.param
    simulator = OPMFlowSimulator(
        deck_template_file=str(DECKS / f"{form}.mako"), case_dir=str(DECKS)
    )
    case: Path = simulator.simulate(MemberInput(0, {}, tmp_path_factory.mktemp(form)))
    return case


def _co2_pressure(density: float) -> float:
    """The pressure (Pa) at which Span & Wagner's CO₂ has ``density`` at 40 C."""
    from scipy.optimize import brentq

    return float(
        brentq(lambda p: CoolProp.PropsSI("D", "T", KELVIN, "P", p, "CO2") - density, 5e6, 6e7)
    )


def test_the_restarts_hold_what_the_forward_model_reads(run: Path) -> None:
    state = CaseState(run)
    oil_gas = run.parent.name.startswith("oil_gas")
    assert state.phases == (("oil", "gas") if oil_gas else ("water", "gas"))
    brine = "OIL_DEN" if oil_gas else "WAT_DEN"
    arrays = state.restart(state.restart_dates[-1], ["PRESSURE", "SGAS", "GAS_DEN", brine])
    assert set(arrays) == {"PRESSURE", "SGAS", "GAS_DEN", brine}
    assert 1000.0 < arrays[brine].min() < arrays[brine].max() < 1100.0  # brine, not oil
    plume = arrays["SGAS"] > 0.05
    assert plume.sum() >= 3
    reported = arrays["PRESSURE"][plume] * 1e5
    co2 = np.array([_co2_pressure(d) for d in arrays["GAS_DEN"][plume]])
    if oil_gas:
        # The brine's pressure: the CO₂'s is above it by the capillary pressure, SGOF's
        # (0, 0.1, 0.8) -> (0, 5, 20) bar at the cell's SGAS.
        pc = np.interp(arrays["SGAS"][plume], [0.0, 0.1, 0.8], [0.0, 5e5, 20e5])
        np.testing.assert_allclose(co2 - reported, pc, rtol=0, atol=0.1e5)
    else:
        # The CO₂'s pressure: Span & Wagner's density there is OPM's.
        np.testing.assert_allclose(co2, reported, rtol=0, atol=0.05e5)


def test_soft_sand_with_co2s_equation_of_state_forecasts_both_forms(run: Path) -> None:
    state = CaseState(run)
    oil_gas = run.parent.name.startswith("oil_gas")
    date = state.restart_dates[-1]
    pressure, saturations, densities = state.flow_state(date, densities=True)
    if oil_gas:  # as GeophysicsModel's brine_as_oil maps it
        saturations = {("water" if p == "oil" else p): s for p, s in saturations.items()}
        densities = {("water" if p == "oil" else p): d for p, d in densities.items()}
    model = SoftSandModel(temperature_c=KELVIN - 273.15)
    flow = FlowState(
        porosity=state.static("PORO"),
        pressure=pressure,
        overburden_pressure=np.full(pressure.size, 22600.0 * 1100.0),
        saturations=saturations,
        densities=densities,
    )
    props = model.elastic(flow)
    assert np.all(np.isfinite(props.vp))
    assert props.counts["co2_cells"] > 0
    assert props.counts["co2_two_phase_snapped"] == 0
    if not oil_gas:
        assert props.counts["co2_pressure_inconsistent"] == 0


def test_the_forward_model_reads_both_forms(run: Path, tmp_path: Path) -> None:
    """AVO maps through GeophysicsModel, its brine_as_oil for the oil-gas form."""
    from apex.algorithms.geophysics import Zoeppritz, avo_attributes
    from apex.domain.config import GeophysicsSection
    from apex.domain.observations import DataKey
    from apex.workflows.geophysics import build_geophysics_model

    state = CaseState(run)
    oil_gas = run.parent.name.startswith("oil_gas")
    vintage = state.restart_dates[-1].isoformat()
    section = GeophysicsSection.model_validate(
        {
            "pem": {"name": "soft_sand", "options": {"temperature_c": KELVIN - 273.15}},
            "overburden_pressure": {"gradient": 22600.0},
            "grid_file": Path(f"{run}.EGRID"),
            "data": [
                {
                    "kind": "avo",
                    "datatype": "avo",
                    "vintages": [vintage],
                    "options": {
                        "angles": [10],
                        "wavelet": {"frequency": 45.0, "length": 0.05},
                        "overburden": {"vp": 2450.0, "vs": 849.0, "density": 2300.0},
                        "times": {"start": 1.0, "end": 1.2, "step": 0.002},
                    },
                }
            ],
        }
    )
    model = build_geophysics_model(
        section,
        SoftSandModel(temperature_c=KELVIN - 273.15),
        [Zoeppritz()],
        [avo_attributes.create("rms")],
        brine_as_oil=oil_gas,
    )
    n = model.field_sizes()["avo_10"]
    values = model.forecast(run, [DataKey(vintage, "avo_10", i) for i in range(n)])
    assert np.all(np.isfinite(values))
    assert np.ptp(values) > 0  # the plume shows


def test_a_temperature_gradient_is_opms_per_cell_and_forecast(tmp_path: Path) -> None:
    """With a varying RTEMPVD (35 to 45 °C over the box's 1000-1050 m),
    OPM's CO₂ density in each plume cell is Span & Wagner's at that cell's pressure and
    its depth's temperature (linear in depth, as APEX reads the table); and soft_sand with
    temperature = "deck" and Lebedev's dissolved CO₂ forecasts the AVO maps."""
    from apex.algorithms.geophysics import Zoeppritz, avo_attributes
    from apex.domain.config import GeophysicsSection
    from apex.domain.observations import DataKey
    from apex.workflows.geophysics import CellTemperature, build_geophysics_model

    text = (DECKS / "gas_water.mako").read_text()
    assert text.count("RTEMPVD\n 0    40\n 2000 40 /") == 1
    deck = tmp_path / "gradient.mako"
    deck.write_text(text.replace("RTEMPVD\n 0    40\n 2000 40 /", "RTEMPVD\n 1000 35\n 1050 45 /"))
    simulator = OPMFlowSimulator(deck_template_file=str(deck), case_dir=str(DECKS))
    (tmp_path / "run").mkdir()
    run: Path = simulator.simulate(MemberInput(0, {}, tmp_path / "run"))
    state = CaseState(run)
    date = state.restart_dates[-1]
    arrays = state.restart(date, ["PRESSURE", "SGAS", "GAS_DEN"])
    table = (((1000.0, 35.0), (1050.0, 45.0)),)
    temperature = CellTemperature(tables=table).cells(state, date)
    assert 35.0 < temperature.min() < temperature.max() < 45.0
    plume = arrays["SGAS"] > 0.05
    assert plume.sum() >= 3
    expected = [
        CoolProp.PropsSI("D", "T", t + 273.15, "P", p * 1e5, "CO2")
        for t, p in zip(temperature[plume], arrays["PRESSURE"][plume], strict=True)
    ]
    np.testing.assert_allclose(arrays["GAS_DEN"][plume], expected, rtol=1e-4)

    vintage = date.isoformat()
    section = GeophysicsSection.model_validate(
        {
            "pem": {"name": "soft_sand", "options": {"temperature": "deck"}},
            "overburden_pressure": {"gradient": 22600.0},
            "grid_file": Path(f"{run}.EGRID"),
            "data": [
                {
                    "kind": "avo",
                    "datatype": "avo",
                    "vintages": [vintage],
                    "options": {
                        "angles": [10],
                        "wavelet": {"frequency": 45.0, "length": 0.05},
                        "overburden": {"vp": 2450.0, "vs": 849.0, "density": 2300.0},
                        "times": {"start": 1.0, "end": 1.2, "step": 0.002},
                    },
                }
            ],
        }
    )
    pem = SoftSandModel(
        temperature="deck", temperature_range=(35.0, 45.0), dissolved_co2="lebedev2014"
    )
    model = build_geophysics_model(
        section, pem, [Zoeppritz()], [avo_attributes.create("rms")],
        temperature=CellTemperature(tables=table),
    )  # fmt: skip
    n = model.field_sizes()["avo_10"]
    values = model.forecast(run, [DataKey(vintage, "avo_10", i) for i in range(n)])
    assert np.all(np.isfinite(values))
    assert np.ptp(values) > 0


def test_a_thermal_runs_temp_is_read_per_cell_and_forecast(tmp_path: Path) -> None:
    """A thermal CO2STORE run (THERMAL, CO₂ injected at 20 °C) writes
    each cell's TEMP; APEX reads it per cell, OPM's CO₂ density in the plume is Span &
    Wagner's at that temperature and the cell's pressure, and soft_sand with
    temperature = "deck" forecasts the AVO maps over a given temperature_range."""
    from apex.algorithms.geophysics import Zoeppritz, avo_attributes
    from apex.domain.config import GeophysicsSection
    from apex.domain.observations import DataKey
    from apex.simulation.deck import deck_contents
    from apex.workflows.geophysics import CellTemperature, build_geophysics_model

    deck = DECKS / "thermal.mako"
    contents = deck_contents(deck, DECKS)
    assert contents.thermal
    simulator = OPMFlowSimulator(deck_template_file=str(deck), case_dir=str(DECKS))
    (tmp_path / "run").mkdir()
    run: Path = simulator.simulate(MemberInput(0, {}, tmp_path / "run"))
    state = CaseState(run)
    date = state.restart_dates[-1]
    temperature = CellTemperature(thermal=True).cells(state, date)
    assert np.ptp(temperature) > 1.0  # the cold injection shows
    arrays = state.restart(date, ["PRESSURE", "SGAS", "GAS_DEN"])
    plume = arrays["SGAS"] > 0.05
    assert plume.sum() >= 3
    expected = [
        CoolProp.PropsSI("D", "T", t + 273.15, "P", p * 1e5, "CO2")
        for t, p in zip(temperature[plume], arrays["PRESSURE"][plume], strict=True)
    ]
    np.testing.assert_allclose(arrays["GAS_DEN"][plume], expected, rtol=1e-3)

    vintage = date.isoformat()
    section = GeophysicsSection.model_validate(
        {
            "pem": {"name": "soft_sand", "options": {"temperature": "deck"}},
            "overburden_pressure": {"gradient": 22600.0},
            "grid_file": Path(f"{run}.EGRID"),
            "data": [
                {
                    "kind": "avo",
                    "datatype": "avo",
                    "vintages": [vintage],
                    "options": {
                        "angles": [10],
                        "wavelet": {"frequency": 45.0, "length": 0.05},
                        "overburden": {"vp": 2450.0, "vs": 849.0, "density": 2300.0},
                        "times": {"start": 1.0, "end": 1.2, "step": 0.002},
                    },
                }
            ],
        }
    )
    pem = SoftSandModel(temperature="deck", temperature_range=(15.0, 60.0))
    model = build_geophysics_model(
        section, pem, [Zoeppritz()], [avo_attributes.create("rms")],
        temperature=CellTemperature(thermal=True),
    )  # fmt: skip
    n = model.field_sizes()["avo_10"]
    values = model.forecast(run, [DataKey(vintage, "avo_10", i) for i in range(n)])
    assert np.all(np.isfinite(values))
    assert np.ptp(values) > 0


def test_each_eqlnum_region_takes_its_own_table(tmp_path: Path) -> None:
    """Two EQLNUM regions (the box's halves) at 40 and 50 °C: OPM writes EQLNUM to INIT,
    and its CO₂ density in each plume cell is Span & Wagner's at that cell's region's
    temperature."""
    from apex.workflows.geophysics import CellTemperature

    text = (DECKS / "gas_water.mako").read_text()
    two = (
        text.replace("EQLDIMS\n/", "EQLDIMS\n 2 /", 1)
        .replace("SOLUTION\n", "REGIONS\nEQLNUM\n 50*1 50*2 /\nSOLUTION\n", 1)
        .replace(
            "EQUIL\n 1000 100 900 0 900 0 /\n",
            "EQUIL\n 1000 100 900 0 900 0 /\n 1000 100 900 0 900 0 /\n",
            1,
        )
        .replace(
            "RTEMPVD\n 0    40\n 2000 40 /", "RTEMPVD\n 0 40\n 2000 40 /\n 0 50\n 2000 50 /", 1
        )
    )
    deck = tmp_path / "regions.mako"
    deck.write_text(two)
    (tmp_path / "run").mkdir()
    run: Path = OPMFlowSimulator(deck_template_file=str(deck), case_dir=str(DECKS)).simulate(
        MemberInput(0, {}, tmp_path / "run")
    )
    state = CaseState(run)
    assert "EQLNUM" in state.init
    date = state.restart_dates[-1]
    tables = (((0.0, 40.0), (2000.0, 40.0)), ((0.0, 50.0), (2000.0, 50.0)))
    temperature = CellTemperature(tables=tables).cells(state, date)
    assert set(np.unique(temperature)) == {40.0, 50.0}
    arrays = state.restart(date, ["PRESSURE", "SGAS", "GAS_DEN"])
    plume = arrays["SGAS"] > 0.05
    assert plume.sum() >= 3
    expected = [
        CoolProp.PropsSI("D", "T", t + 273.15, "P", p * 1e5, "CO2")
        for t, p in zip(temperature[plume], arrays["PRESSURE"][plume], strict=True)
    ]
    np.testing.assert_allclose(arrays["GAS_DEN"][plume], expected, rtol=1e-4)
