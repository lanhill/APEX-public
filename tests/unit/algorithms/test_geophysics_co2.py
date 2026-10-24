"""CO₂'s modulus from Span & Wagner's equation of state.

Independent reference: the NIST Chemistry WebBook's CO₂ tables
(``tests/data/reference/geophysics/nist_co2.csv``), which implement Span & Wagner (1996);
the table against direct CoolProp calls; ``soft_sand``'s use of it.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("CoolProp")

from apex.algorithms.geophysics import FlowState, SoftSandModel
from apex.algorithms.geophysics.co2 import SpanWagnerCO2
from apex.algorithms.geophysics.errors import GeophysicsModelError

NIST = Path(__file__).resolve().parents[2] / "data" / "reference" / "geophysics" / "nist_co2.csv"


def _nist(saturated: bool = False) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in NIST.read_text().splitlines():
        if line.startswith("#") or line.startswith("temperature"):
            continue
        t, p, d, c, phase = line.split(",")
        if phase.startswith("saturated") == saturated:
            rows.append((float(t), float(p), float(d), float(c)))
    return rows


def test_the_equation_of_state_is_nists() -> None:
    """CoolProp's speed of sound at NIST's density within 1e-7 (measured 2e-8), and the
    table's modulus within its 1e-4 check (rho c² from NIST's own values)."""
    import CoolProp.CoolProp as CP

    for t, _, d, c in _nist():
        assert CP.PropsSI("A", "T", t, "D", d, "CO2") == pytest.approx(c, rel=1e-7)
        k, snapped = SpanWagnerCO2(t - 273.15).modulus([d])
        assert not snapped[0]
        assert k[0] == pytest.approx(d * c * c, rel=1e-4)


@pytest.mark.parametrize("temperature", [20.0, 25.0, 30.9, 30.97, 31.0, 31.1, 31.5, 37.0, 100.0])
def test_the_table_meets_its_tolerance_across_the_critical_region(temperature: float) -> None:
    """The setup check (midpoints within 1e-4 of direct calls) passes from 20 to 100 °C,
    within 0.2 °C of the critical temperature, on both branches below it."""
    model = SpanWagnerCO2(temperature)
    assert (model.saturated is not None) == (temperature < 30.978)
    rho = np.linspace(2.0, 1190.0, 41)
    if model.saturated is not None:
        vapour, liquid = model.saturated
        rho = rho[(rho <= vapour) | (rho >= liquid)]
    k, _ = model.modulus(rho)
    exact = SpanWagnerCO2._direct(temperature + 273.15, rho)
    np.testing.assert_allclose(k, exact, rtol=1e-4)


def test_two_phase_densities_snap_to_the_nearer_branch() -> None:
    model = SpanWagnerCO2(25.0)
    assert model.saturated is not None
    vapour, liquid = model.saturated
    inside = [vapour + 1.0, liquid - 1.0]
    outside = [vapour - 1.0, liquid + 1.0]
    k, snapped = model.modulus(inside + outside)
    assert snapped.tolist() == [True, True, False, False]
    at_saturation, _ = model.modulus([vapour, liquid])
    np.testing.assert_allclose(k[:2], at_saturation, rtol=1e-12)
    np.testing.assert_allclose(k[2:], SpanWagnerCO2._direct(298.15, np.array(outside)), rtol=1e-4)


def test_a_density_outside_the_range_fails_the_member() -> None:
    with pytest.raises(GeophysicsModelError, match="outside eos_density_range"):
        SpanWagnerCO2(37.0).modulus([1300.0])


def _co2_state(gas: float, pressure: float = 10e6) -> FlowState:
    return FlowState(
        porosity=np.array([0.3, 0.3]),
        pressure=np.full(2, pressure),
        overburden_pressure=np.full(2, 30e6),
        saturations={"water": np.array([0.7, 1.0]), "gas": np.array([0.3, 0.0])},
        densities={"water": np.full(2, 1030.0), "gas": np.full(2, gas)},
    )


def test_soft_sand_takes_co2s_modulus_at_its_simulated_density() -> None:
    model = SoftSandModel()
    assert model.gas_model == "span_wagner"
    model.check_phases(["water", "gas"])  # builds the table at setup
    from apex.algorithms.geophysics.petro_elastic import _cells

    # CO₂ at 37 °C and 10 MPa is 683.40257 kg/m3 (NIST): the simulator's density and
    # pressure agree.
    k_gas = SpanWagnerCO2._direct(310.15, np.array([683.40257]))[0]
    c = _cells(_co2_state(683.40257), 1.0, model.min_effective_pressure, True)
    _, bulk, counts = model.fluid(c)
    assert counts == {"co2_cells": 1, "co2_two_phase_snapped": 0, "co2_pressure_inconsistent": 0}
    k_water = 1030.0 * model_brine(model) ** 2 / 1e6
    expected = brie(0.7, k_water, k_gas / 1e6, model.brie_exponent)
    assert bulk[0] == pytest.approx(expected, rel=1e-4)
    # At 20 MPa CO₂ is 855 kg/m3, not 683: counted as inconsistent.
    c = _cells(_co2_state(683.40257, pressure=20e6), 1.0, model.min_effective_pressure, True)
    assert model.fluid(c)[2]["co2_pressure_inconsistent"] == 1


def model_brine(model: SoftSandModel) -> float:
    from apex.algorithms.geophysics import rock_physics as rp

    return float(rp.brine_velocity(model.temperature_c, 10.0, model.salinity))


def brie(s_liquid: float, k_liquid: float, k_gas: float, e: float) -> float:
    from apex.algorithms.geophysics import rock_physics as rp

    return float(rp.brie(np.array([s_liquid]), k_liquid, np.array([k_gas]), e)[0])


@pytest.mark.parametrize(
    ("options", "gas_model"),
    [
        ({}, "span_wagner"),
        ({"extrapolate_gas_correlation": True}, "batzle_wang"),  # as before revision 1
        ({"gas": "methane"}, "batzle_wang"),
        ({"gas_model": "batzle_wang"}, "batzle_wang"),
    ],
)
def test_the_gas_model_defaults(options: dict[str, Any], gas_model: str) -> None:
    assert SoftSandModel(**options).gas_model == gas_model


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"gas": "methane", "gas_model": "span_wagner"}, "CO₂'s equation of state"),
        ({"gas_model": "span_wagner", "extrapolate_gas_correlation": True}, "batzle_wang"),
        ({"eos_density_range": (5.0, 1.0)}, "eos_density_range"),
    ],
)
def test_settings_that_hold_for_no_member_are_refused(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SoftSandModel(**options)


def test_the_saturated_branch_ends_are_nists() -> None:
    """The branch ends below the critical temperature, evaluated at quality 0 and 1: the
    saturated densities and their moduli, against NIST's saturation table."""
    rows = _nist(saturated=True)
    assert len(rows) == 4
    for (t, _, d_liquid, c_liquid), (_, _, d_vapour, c_vapour) in zip(
        rows[::2], rows[1::2], strict=True
    ):
        model = SpanWagnerCO2(t - 273.15)
        assert model.saturated is not None
        assert model.saturated[0] == pytest.approx(d_vapour, rel=1e-6)
        assert model.saturated[1] == pytest.approx(d_liquid, rel=1e-6)
        k, snapped = model.modulus(list(model.saturated))
        assert not snapped.any()
        assert k[0] == pytest.approx(d_vapour * c_vapour**2, rel=1e-5)
        assert k[1] == pytest.approx(d_liquid * c_liquid**2, rel=1e-5)


@pytest.mark.parametrize("temperature", [30.976, 30.977, 30.978, 30.9781, 30.9783])
def test_the_table_refines_itself_next_to_the_critical_point(temperature: float) -> None:
    """A regression (review of phase 2b step 1): the modulus bends so sharply next to a
    saturated density just below the critical temperature that a 0.5 kg/m3 table missed
    its check there, and the step was not an option: the missing intervals are halved."""
    SpanWagnerCO2(temperature)


def test_a_coarse_table_is_refined_to_the_tolerance() -> None:
    model = SpanWagnerCO2(31.05, step=20.0)
    rho = np.linspace(2.0, 1190.0, 301)
    k, _ = model.modulus(rho)
    np.testing.assert_allclose(k, SpanWagnerCO2._direct(304.2, rho), rtol=1e-4)


def test_a_pressure_without_a_state_counts_as_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CoolProp has no state above CO₂'s melting line: NaN, which the 5 % check counts,
    rather than failing every member over a diagnostic."""
    k = SpanWagnerCO2(37.0).modulus_at_pressure([10e6, 900e6])
    assert np.isfinite(k[0])
    assert np.isnan(k[1])
    model = SoftSandModel()
    from apex.algorithms.geophysics.petro_elastic import _cells

    eos = model._eos()
    monkeypatch.setattr(eos, "modulus_at_pressure", lambda p: np.full(np.shape(p), np.nan))
    c = _cells(_co2_state(683.40257), 1.0, model.min_effective_pressure, True)
    assert model.fluid(c)[2]["co2_pressure_inconsistent"] == 1


def test_cells_without_co2_need_no_density_in_the_tables_range() -> None:
    """A regression (review): absent CO₂'s placeholder density was evaluated too, and
    fell outside a narrowed range, failing every member."""
    model = SoftSandModel(eos_density_range=(150.0, 1200.0))
    from apex.algorithms.geophysics.petro_elastic import _cells

    state = _co2_state(683.40257)
    state = FlowState(
        porosity=state.porosity, pressure=state.pressure,
        overburden_pressure=state.overburden_pressure, saturations=state.saturations,
        densities={"water": np.full(2, 1030.0), "gas": np.array([683.40257, 0.0])},
    )  # fmt: skip
    counts = model.fluid(_cells(state, 1.0, model.min_effective_pressure, True))[2]
    assert counts["co2_cells"] == 1


@pytest.mark.parametrize(
    ("deck", "varies", "outcome"),
    [
        (37.0, False, None),
        (37.05, False, None),
        (38.0, False, "reservoir temperature is 38"),
        (None, False, "states no readable reservoir temperature"),
        (None, True, "varies the temperature"),
    ],
)
def test_the_temperature_must_be_the_decks(
    deck: float | None, varies: bool, outcome: str | None
) -> None:
    model = SoftSandModel()
    if outcome is None:
        assert model.check_temperature(deck, varies, ["water", "gas"]) == []
    elif outcome == "states no readable reservoir temperature":
        (warning,) = model.check_temperature(deck, varies, ["water", "gas"])
        assert outcome in warning
    else:
        with pytest.raises(ValueError, match=outcome):
            model.check_temperature(deck, varies, ["water", "gas"])
    # Without CO₂ (no gas phase, or the correlation) the temperature is not checked.
    assert model.check_temperature(38.0, True, ["oil", "water"]) == []
    assert SoftSandModel(gas_model="batzle_wang").check_temperature(38.0, True, None) == []
    # A templated RUNSPEC (phases unknown): the deck may have no gas, so a varying
    # temperature is noted, not refused (review of phase 2b).
    (warning,) = model.check_temperature(None, True, None)
    assert "not refused" in warning
