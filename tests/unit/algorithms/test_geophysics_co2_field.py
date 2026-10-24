"""CO₂'s modulus over temperature and density.

Independent reference: direct CoolProp calls (Span & Wagner 1996, checked against NIST in
``test_geophysics_co2.py``) on dense (T, rho) samples below, through and above CO₂'s
critical temperature, each branch.
"""

import importlib.util

import numpy as np
import pytest

from apex.algorithms.geophysics import FlowState, SoftSandModel
from apex.algorithms.geophysics.co2 import (
    CRITICAL_BAND_K,
    TABLE_TOLERANCE,
    SpanWagnerCO2,
    SpanWagnerCO2Field,
)
from apex.algorithms.geophysics.errors import GeophysicsModelError

#: The tests that call CoolProp (the co2 extra); the others run without it.
needs_coolprop = pytest.mark.skipif(
    importlib.util.find_spec("CoolProp") is None, reason="needs the co2 extra"
)


@pytest.fixture(scope="module")
def field() -> SpanWagnerCO2Field:
    pytest.importorskip("CoolProp")
    return SpanWagnerCO2Field((20.0, 45.0))


def _t_crit() -> float:
    import CoolProp.CoolProp as CP

    return float(CP.PropsSI("Tcrit", "CO2")) - 273.15


def _direct(t: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.array(
        [SpanWagnerCO2._direct(a + 273.15, np.array([b]))[0] for a, b in zip(t, rho, strict=True)]
    )


@needs_coolprop
@pytest.mark.parametrize(
    ("label", "t_offsets", "densities"),
    [
        ("below", (-10.9, -1.0), (1.0, 1200.0)),
        ("just below", (-1.0, -CRITICAL_BAND_K - 1e-3), (300.0, 650.0)),
        ("just above", (CRITICAL_BAND_K + 1e-3, 1.0), (300.0, 650.0)),
        ("above", (1.0, 14.0), (1.0, 1200.0)),
    ],
)
def test_the_table_meets_its_tolerance_on_each_side(
    field: SpanWagnerCO2Field,
    label: str,
    t_offsets: tuple[float, float],
    densities: tuple[float, float],
) -> None:
    """Interpolated in temperature and density, within 1e-4 of direct calls (outside the
    two-phase dome, whose densities are snapped and checked below)."""
    rng = np.random.default_rng(7)
    t = _t_crit() + rng.uniform(*t_offsets, 1500)
    rho = rng.uniform(*densities, 1500)
    k, snapped, band = field.modulus(t, rho)
    assert not band.any()
    ok = ~snapped
    assert ok.sum() > 500, label
    np.testing.assert_allclose(k[ok], _direct(t[ok], rho[ok]), rtol=TABLE_TOLERANCE)


@needs_coolprop
def test_densities_in_the_dome_snap_to_the_saturated_state_at_the_cells_temperature(
    field: SpanWagnerCO2Field,
) -> None:
    import CoolProp.CoolProp as CP

    t = np.array([25.0, 25.0, 30.0])
    vapour = [CP.PropsSI("D", "T", x + 273.15, "Q", 1, "CO2") for x in t]
    liquid = [CP.PropsSI("D", "T", x + 273.15, "Q", 0, "CO2") for x in t]
    rho = np.array([vapour[0] + 1.0, liquid[1] - 1.0, 0.5 * (vapour[2] + liquid[2]) - 1.0])
    k, snapped, _ = field.modulus(t, rho)
    assert snapped.all()
    expected = [
        SpanWagnerCO2._direct(t[0] + 273.15, np.array([vapour[0]]), (vapour[0], liquid[0]))[0],
        SpanWagnerCO2._direct(t[1] + 273.15, np.array([liquid[1]]), (vapour[1], liquid[1]))[0],
        SpanWagnerCO2._direct(t[2] + 273.15, np.array([vapour[2]]), (vapour[2], liquid[2]))[0],
    ]
    np.testing.assert_allclose(k, expected, rtol=TABLE_TOLERANCE)


@needs_coolprop
def test_the_critical_band_is_computed_by_direct_calls(field: SpanWagnerCO2Field) -> None:
    """Within ±0.05 K of T_c, the cusp (6.7 to 8.5 MPa between 30.99 and 31.1 °C at the
    critical density) is followed by direct calls at the rounded (T, rho)."""
    rng = np.random.default_rng(3)
    t = _t_crit() + rng.uniform(-0.049, 0.049, 400)
    rho = rng.uniform(300.0, 650.0, 400)
    k, snapped, band = field.modulus(t, rho)
    assert band.all()
    out = ~snapped
    rounded_t, rounded_rho = np.round(t, 3), np.round(rho, 2)
    np.testing.assert_allclose(k[out], _direct(rounded_t[out], rounded_rho[out]), rtol=1e-9)
    assert np.isfinite(k).all()


@needs_coolprop
def test_one_temperature_agrees_with_the_one_dimensional_table() -> None:
    field = SpanWagnerCO2Field((40.0, 40.0))
    rho = np.linspace(1.0, 1200.0, 500)
    k, _, _ = field.modulus(np.full(rho.size, 40.0), rho)
    one, _ = SpanWagnerCO2(40.0).modulus(rho)
    np.testing.assert_allclose(k, one, rtol=2 * TABLE_TOLERANCE)


@needs_coolprop
def test_cells_outside_the_ranges_fail_the_member(field: SpanWagnerCO2Field) -> None:
    with pytest.raises(GeophysicsModelError, match="temperature_range"):
        field.modulus(np.array([46.0]), np.array([500.0]))
    with pytest.raises(GeophysicsModelError, match="eos_density_range"):
        field.modulus(np.array([40.0]), np.array([1300.0]))
    with pytest.raises(ValueError, match="temperature_range"):
        SpanWagnerCO2Field((45.0, 20.0))


@needs_coolprop
def test_the_modulus_at_pressure_takes_each_cells_temperature() -> None:
    import CoolProp.CoolProp as CP

    field = SpanWagnerCO2Field((35.0, 60.0))
    p = np.array([10e6, 10e6, 20e6])
    t = np.array([35.0, 60.0, 50.0])
    k = field.modulus_at_pressure(p, t)
    expected = [
        CP.PropsSI("D", "P", a, "T", b + 273.15, "CO2")
        * CP.PropsSI("A", "P", a, "T", b + 273.15, "CO2") ** 2
        for a, b in zip(p, t, strict=True)
    ]
    np.testing.assert_allclose(k, expected, rtol=1e-9)
    assert k[0] != k[1]


@needs_coolprop
def test_soft_sand_takes_each_cells_temperature_from_the_deck() -> None:
    """``temperature = "deck"``: each cell's brine at its own temperature, as the scalar
    model at that temperature (to the rounding of the per-cell sum); CO₂ from the 2-D table
    with the critical band counted; a cell outside 0-100 °C refused."""
    from apex.algorithms.geophysics.petro_elastic import PetroElasticError

    def state(t: list[float] | None, gas: float = 0.0) -> FlowState:
        n = 2
        return FlowState(
            porosity=np.full(n, 0.3),
            pressure=np.full(n, 15e6),
            overburden_pressure=np.full(n, 40e6),
            saturations={"water": np.full(n, 1.0 - gas), "gas": np.full(n, gas)},
            densities={"water": np.full(n, 1030.0), "gas": np.full(n, 600.0)},
            temperature=None if t is None else np.array(t),
        )

    deck = SoftSandModel(temperature="deck", temperature_range=(30.0, 70.0))
    both = deck.elastic(state([40.0, 60.0]))
    for cell, t in enumerate((40.0, 60.0)):
        alone = SoftSandModel(temperature_c=t).elastic(state(None))
        assert both.vp[cell] == pytest.approx(alone.vp[0], rel=1e-12)
    with pytest.raises(PetroElasticError, match="temperature"):
        SoftSandModel(temperature="deck", temperature_range=(30.0, 70.0)).elastic(
            state([40.0, 101.0])
        )
    with pytest.raises(ValueError, match="no temperature per cell"):
        deck.elastic(state(None))
    co2 = SoftSandModel(temperature="deck", temperature_range=(30.0, 70.0))
    props = co2.elastic(state([31.0, 50.0], gas=0.4))
    assert props.counts["co2_cells"] == 2
    assert props.counts["co2_critical_band"] == 1
    with pytest.raises(ValueError, match="needs temperature_range"):
        SoftSandModel(temperature="deck").elastic(state([40.0, 50.0], gas=0.4))
    with pytest.raises(ValueError, match="temperature_range"):
        SoftSandModel(temperature="deck", temperature_range=(-5.0, 50.0))


def _brine_state(s_gas: float, pressure_mpa: float, n: int = 1) -> FlowState:
    return FlowState(
        porosity=np.full(n, 0.3),
        pressure=np.full(n, pressure_mpa * 1e6),
        overburden_pressure=np.full(n, 40e6),
        saturations={"water": np.full(n, 1.0 - s_gas), "gas": np.full(n, s_gas)},
        densities={"water": np.full(n, 1030.0), "gas": np.full(n, 700.0)},
    )


def test_dissolved_co2_follows_lebedev_et_al_2014() -> None:
    """Lebedev et al. (2014, Energy Procedia 63, 4273-4280; their eq. (1), §3.5, p. 4277, not
    their data, CC BY-NC-ND): V = V_BW + 20.42 - 0.08377 T - 0.18 P - 0.0004173 S,
    at hand-computed points; its sign changes near 40,300 ppm at 30 °C and 6 MPa; only cells
    with free CO₂ inside 2-38 MPa and 30-70 °C are corrected, the others with free CO₂
    counted; the modulus changes by a few per cent at most, as in the paper (2.3 %)."""
    from apex.algorithms.geophysics import rock_physics as rp

    def water_modulus(
        model: "SoftSandModel", s_gas: float, p: float
    ) -> tuple[float, dict[str, int]]:
        from apex.algorithms.geophysics.petro_elastic import _cells

        c = _cells(_brine_state(s_gas, p), 1.0, model.min_effective_pressure, True)
        _, _, k, counts = model._phase_moduli(c)
        return float(k["water"][0]), counts

    def velocity(k_mpa: float) -> float:
        return float(np.sqrt(k_mpa * 1e6 / 1030.0))

    for t, p, ppm in ((30.0, 6.0, 0.0), (50.0, 20.0, 35_000.0), (70.0, 38.0, 100_000.0)):
        on = SoftSandModel(temperature_c=t, salinity=ppm / 1e6, dissolved_co2="lebedev2014",
                           gas_model="batzle_wang", extrapolate_gas_correlation=True)  # fmt: skip
        k, counts = water_modulus(on, 0.2, p)
        v_bw = float(rp.brine_velocity(t, np.array([p]), ppm / 1e6)[0])
        expected = v_bw + 20.42 - 0.08377 * t - 0.18 * p - 0.0004173 * ppm
        assert velocity(k) == pytest.approx(expected, rel=1e-12)
        assert counts["co2_dissolved_corrected"] == 1
        off = SoftSandModel(temperature_c=t, salinity=ppm / 1e6, gas_model="batzle_wang",
                            extrapolate_gas_correlation=True)  # fmt: skip
        k_off, _ = water_modulus(off, 0.2, p)
        assert abs(k / k_off - 1.0) < 0.05
        # No free CO₂: not corrected (full saturation only).
        k_none, none = water_modulus(on, 0.0, p)
        assert k_none == pytest.approx(water_modulus(off, 0.0, p)[0], rel=1e-15)
        assert none["co2_dissolved_corrected"] == 0
    # The sign of the correction changes with salinity near 40,300 ppm (30 °C, 6 MPa).
    root = (20.42 - 0.08377 * 30.0 - 0.18 * 6.0) / 0.0004173
    assert root == pytest.approx(40_323, abs=1)
    # Outside the fitted pressures or temperatures: uncorrected, counted.
    on = SoftSandModel(temperature_c=25.0, salinity=0.0, dissolved_co2="lebedev2014",
                       gas_model="batzle_wang", extrapolate_gas_correlation=True)  # fmt: skip
    off = SoftSandModel(temperature_c=25.0, salinity=0.0, gas_model="batzle_wang",
                        extrapolate_gas_correlation=True)  # fmt: skip
    k, counts = water_modulus(on, 0.2, 10.0)
    assert k == water_modulus(off, 0.2, 10.0)[0]
    assert counts == {**counts, "co2_dissolved_corrected": 0, "co2_dissolved_out_of_range": 1}
    with pytest.raises(ValueError, match="100000 ppm"):
        SoftSandModel(salinity=0.12, dissolved_co2="lebedev2014")
    with pytest.raises(ValueError, match="gas = 'co2'"):
        SoftSandModel(gas="methane", dissolved_co2="lebedev2014")


def test_misspelt_or_misplaced_options_are_refused() -> None:
    options: list[tuple[dict[str, object], str]] = [
        ({"temperature": "Deck"}, "temperature must be one of"),
        ({"dissolved_co2": "lebedev"}, "dissolved_co2 must be one of"),
        ({"mixing": "Brie"}, "mixing must be one of"),
        ({"temperature_range": (30.0, 60.0)}, "is for temperature = 'deck'"),
        ({"temperature": "deck", "temperature_range": (30.0,)}, r"must be \[low, high\]"),
    ]
    for given, message in options:
        with pytest.raises(ValueError, match=message):
            SoftSandModel(**given)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs temperature_range"):
        SoftSandModel(temperature="deck").check_phases(["water", "gas"])


def test_a_cell_outside_temperature_range_fails_its_member() -> None:
    """A thermal run's TEMP outside the range given: refused in every cell, brine too."""
    from apex.algorithms.geophysics.petro_elastic import PetroElasticError

    state = FlowState(
        porosity=np.full(2, 0.3),
        pressure=np.full(2, 15e6),
        overburden_pressure=np.full(2, 40e6),
        saturations={"water": np.ones(2)},
        densities={"water": np.full(2, 1030.0)},
        temperature=np.array([40.0, 75.0]),
    )
    model = SoftSandModel(temperature="deck", temperature_range=(30.0, 70.0))
    with pytest.raises(PetroElasticError, match="outside temperature_range"):
        model.elastic(state)


@needs_coolprop
def test_the_uncertain_options_share_the_built_table() -> None:
    """A member's model (uncertain options) keeps the table built at setup."""
    import copy

    base = SoftSandModel(temperature="deck", temperature_range=(30.0, 40.0))
    base.check_phases(["water", "gas"])
    member = copy.copy(base)
    member.coordination_number = 8.0
    assert member._co2_field is base._co2_field


@needs_coolprop
def test_a_density_range_without_both_branches_is_refused() -> None:
    with pytest.raises(ValueError, match="both of CO₂'s branches"):
        SpanWagnerCO2Field((0.0, 10.0), density_range=(1.0, 900.0))
