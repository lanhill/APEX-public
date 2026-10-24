"""Rock-physics building blocks and petro-elastic models, independently.

References:

- bruges 0.5.4 (``tests/data/reference/geophysics/rock_physics_bruges.npz``):
  Hashin-Shtrikman bulk bounds (its shear bounds have a defect; Berryman's form instead),
  Hertz-Mindlin and soft sand (Mavko et al. 2009 pp. 246, 258), Gassmann, Wood, and
  Batzle & Wang's (1992) water and brine velocities;
- Berryman, J. G. (1995), Mixture theories for rock properties, AGU Reference Shelf 3;
- closed-form limits: Gassmann's dry (K_f → 0) and stiff-fluid (K_f = K) limits, the
  bounds between Voigt and Reuss, Brie's Voigt (e = 1) and end-member limits;
- published values: pure water's sound speed at 20 °C and 0.1 MPa, 1482.3 m/s (e.g.
  Del Grosso & Mader 1972), and an ideal gas's adiabatic modulus (gamma P) at low
  pressure (Batzle & Wang's gamma_0 → 1.32 for methane).
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import (
    FlowState,
    PetroElasticError,
    SoftSandModel,
    StandardModel,
    petro_elastic_models,
)
from apex.algorithms.geophysics import rock_physics as rp

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "geophysics"
type Floats = NDArray[np.float64]


@pytest.fixture(scope="module")
def bruges() -> dict[str, Floats]:
    path = REFERENCE / "rock_physics_bruges.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; run tools/geophysics_reference/generate.py")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_hashin_shtrikman_matches_bruges_and_berryman(bruges: dict[str, Floats]) -> None:
    """Bulk bounds equal bruges'. bruges' shear bounds put the bulk moduli where the shear
    moduli belong (``bound(f, k, z)`` in ``bounds.hashin_shtrikman``; pet_parity.md, B1),
    so the shear bounds are checked against Berryman's (1995) form with the shear moduli,
    ``[Σ f_i/(G_i + z)]^-1 - z``, and bruges' defect is pinned with the bulk moduli."""
    b = bruges
    k_lo, k_hi, g_lo, g_hi = rp.hashin_shtrikman_bounds(
        b["in_k1"], b["in_g1"], b["in_k2"], b["in_g2"], b["in_f2"]
    )
    np.testing.assert_allclose(k_lo, b["out_hs_bulk_lower"], rtol=1e-12)
    np.testing.assert_allclose(k_hi, b["out_hs_bulk_upper"], rtol=1e-12)
    f = np.stack([1 - b["in_f2"], b["in_f2"]])
    k = np.stack([b["in_k1"], b["in_k2"]])
    g = np.stack([b["in_g1"], b["in_g2"]])

    def z(kk: Floats, gg: Floats) -> Floats:
        out: Floats = gg * (9 * kk + 8 * gg) / (kk + 2 * gg) / 6
        return out

    def berryman(moduli: Floats, zeta: Floats) -> Floats:
        out: Floats = 1 / np.sum(f / (moduli + zeta), axis=0) - zeta
        return out

    z_lo, z_hi = z(k.min(axis=0), g.min(axis=0)), z(k.max(axis=0), g.max(axis=0))
    np.testing.assert_allclose(g_lo, berryman(g, z_lo), rtol=1e-12)
    np.testing.assert_allclose(g_hi, berryman(g, z_hi), rtol=1e-12)
    np.testing.assert_allclose(b["out_hs_shear_lower"], berryman(k, z_lo), rtol=1e-12)
    np.testing.assert_allclose(b["out_hs_shear_upper"], berryman(k, z_hi), rtol=1e-12)


def test_hertz_mindlin_soft_sand_gassmann_wood_match_bruges(bruges: dict[str, Floats]) -> None:
    b = bruges
    hm = rp.hertz_mindlin(
        b["in_sigma_mpa"] / 1e3, b["in_k2"], b["in_g2"], b["in_coordination"],
        b["in_critical_porosity"],
    )  # fmt: skip
    np.testing.assert_allclose(hm[0], b["out_hm_bulk"], rtol=1e-12)
    np.testing.assert_allclose(hm[1], b["out_hm_shear"], rtol=1e-12)
    dry = rp.soft_sand_dry_moduli(
        b["in_porosity"] / b["in_critical_porosity"], *hm, b["in_k2"], b["in_g2"]
    )
    np.testing.assert_allclose(dry[0], b["out_soft_bulk"], rtol=1e-12)
    np.testing.assert_allclose(dry[1], b["out_soft_shear"], rtol=1e-12)
    sat = rp.gassmann(dry[0], b["in_k2"], b["in_fluid_bulk"], b["in_porosity"])
    np.testing.assert_allclose(sat, b["out_gassmann"], rtol=1e-12)
    wood = rp.wood([b["in_saturation"], 1 - b["in_saturation"]], [b["in_k1"], b["in_fluid_bulk"]])
    np.testing.assert_allclose(wood, b["out_wood"], rtol=1e-12)


def test_batzle_wang_water_and_brine_match_bruges(bruges: dict[str, Floats]) -> None:
    b = bruges
    for i in range(b["in_temperature_c"].size):
        t, p = float(b["in_temperature_c"][i]), b["in_pressure_pa"][i] / 1e6
        assert rp.water_velocity(t, p) == pytest.approx(b["out_v_water"][i], rel=1e-12)
        brine = rp.brine_velocity(t, p, float(b["in_salinity"][i]))
        assert brine == pytest.approx(b["out_v_brine"][i], rel=1e-12)


def test_water_velocity_at_20c_is_the_published_sound_speed() -> None:
    assert float(rp.water_velocity(20.0, 0.101325)) == pytest.approx(1482.3, abs=0.5)


def test_gas_modulus_tends_to_the_ideal_adiabatic_one() -> None:
    """At 0.01 MPa a gas is ideal: K = gamma P, gamma about 1.3 for methane (Batzle & Wang)."""
    k = float(rp.gas_bulk_modulus(20.0, 0.01, 0.5537))
    assert k / 0.01 == pytest.approx(1.32, rel=0.03)


def test_gassmann_limits() -> None:
    kd, k, phi = 8.0, 37.0, 0.25
    assert float(rp.gassmann(kd, k, 1e-9, phi)) == pytest.approx(kd, rel=1e-6)
    assert float(rp.gassmann(kd, k, k, phi)) == pytest.approx(k, rel=1e-12)


def test_hashin_shtrikman_between_reuss_and_voigt_and_at_the_end_members() -> None:
    f2 = np.linspace(0, 1, 11)
    k1, g1, k2, g2 = 15.0, 2.0, 37.0, 44.0
    k_lo, k_hi, g_lo, g_hi = rp.hashin_shtrikman_bounds(k1, g1, k2, g2, f2)
    reuss = 1 / ((1 - f2) / k1 + f2 / k2)
    voigt = (1 - f2) * k1 + f2 * k2
    assert np.all(reuss <= k_lo + 1e-12)
    assert np.all(k_lo <= k_hi + 1e-12)
    assert np.all(k_hi <= voigt + 1e-12)
    np.testing.assert_allclose([k_lo[0], k_hi[0], g_lo[0], g_hi[0]], [k1, k1, g1, g1])
    np.testing.assert_allclose([k_lo[-1], k_hi[-1], g_lo[-1], g_hi[-1]], [k2, k2, g2, g2])


def test_brie_limits() -> None:
    s = np.array([0.0, 0.3, 1.0])
    np.testing.assert_allclose(rp.brie(s, 2.5, 0.05, 1.0), s * 2.5 + (1 - s) * 0.05)
    out = rp.brie(s, 2.5, 0.05, 5.0)
    assert out[0] == pytest.approx(0.05)
    assert out[-1] == pytest.approx(2.5)


# ------------------------------------------------------------------- the models


def _state(**kw: object) -> FlowState:
    base: dict[str, object] = {
        "porosity": [0.2, 0.25],
        "pressure": [20e6, 25e6],
        "overburden_pressure": 45e6,
        "saturations": {"oil": [0.6, 0.3], "water": [0.4, 0.7]},
        "densities": {"oil": [750.0, 760.0], "water": [1030.0, 1030.0]},
    }
    base.update(kw)
    return FlowState(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["standard", "soft_sand"])
def test_registered_and_physical(name: str) -> None:
    """Both models give a sandstone's velocities (Vs < Vp/√2), and brine raises Vp."""
    model = petro_elastic_models.create(name)
    e = model.elastic(_state())
    assert np.all((e.vp > 1500) & (e.vp < 5500) & (e.vs > 500) & (e.vs < e.vp / np.sqrt(2)))
    wet = model.elastic(_state(saturations={"oil": [0.0, 0.0], "water": [1.0, 1.0]}))
    assert np.all(wet.vp > e.vp)
    assert np.all(wet.density > e.density)


def test_standard_fluid_fits_override_merges_over_the_reference() -> None:
    """A phase given in ``fluid_fits`` replaces only that phase's fit; with water alone,
    changing the oil fit leaves the result unchanged."""
    wet = _state(saturations={"oil": [0.0, 0.0], "water": [1.0, 1.0]})
    reference = StandardModel().elastic(wet)
    other_oil = StandardModel(fluid_fits={"oil": (700.0, 1.0, 800.0, 5.0)})
    np.testing.assert_array_equal(other_oil.elastic(wet).vp, reference.vp)
    assert np.all(other_oil.elastic(_state()).vp != StandardModel().elastic(_state()).vp)
    with pytest.raises(ValueError, match="four coefficients"):
        StandardModel(fluid_fits={"oil": (1.0, 2.0, 3.0)})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("model", "state", "match"),
    [
        (StandardModel(), _state(porosity=[0.2, -0.01]), "porosity = -0.01 at cell 1"),
        (SoftSandModel(porosity_ratio_cap=None), _state(porosity=[0.37, 0.2]),
         "porosity = 0.37 at cell 0"),
        (StandardModel(), _state(porosity=[0.45, 0.2]), "clay volume = -0.011 at cell 0"),
        (StandardModel(), _state(saturations={"oil": [0.6, 0.3], "water": [0.4, 0.6]}),
         "saturation sum = 0.9 at cell 1"),
        (StandardModel(), _state(saturations={"oil": [1.2, 0.3], "water": [-0.2, 0.7]}),
         "oil saturation = 1.2 at cell 0"),
        (SoftSandModel(), _state(densities={"oil": [1100.0, 760.0], "water": [1030.0] * 2}),
         "oil reference density = 1100 at cell 0"),
        (StandardModel(clay_volume_intercept=0.2, clay_volume_slope=1.0),
         _state(porosity=[0.1, 0.25]), "clay volume = -0.05 at cell 1"),
    ],
)  # fmt: skip
def test_cells_outside_the_model_are_refused(
    model: StandardModel | SoftSandModel, state: FlowState, match: str
) -> None:
    with pytest.raises(PetroElasticError, match=match):
        model.elastic(state)


def test_soft_sand_requires_densities_and_known_options() -> None:
    with pytest.raises(ValueError, match="needs the phases' densities"):
        SoftSandModel().elastic(_state(densities=None))
    with pytest.raises(ValueError, match="Poisson's ratio"):
        SoftSandModel(mineral_shear_modulus=0.0)  # nu = 0.5
    with pytest.raises(ValueError, match="gas must be"):
        SoftSandModel(gas="nitrogen")  # type: ignore[arg-type]


def test_effective_pressure_is_floored_continuously() -> None:
    """At or below ``min_effective_pressure`` (0.01 MPa), negative included, the floor
    applies; above it the rock stiffens with the effective pressure."""
    model = StandardModel()
    vp = [
        model.elastic(_state(pressure=[30e6, 30e6], overburden_pressure=ov)).vp[0]
        for ov in (29e6, 30e6, 30.005e6, 30.01e6, 31e6)
    ]
    assert vp[0] == vp[1] == vp[2] == pytest.approx(vp[3], rel=1e-12)
    assert vp[4] > vp[3]


def test_soft_sand_without_gas_is_wood_of_the_liquids() -> None:
    """No gas: Brie's weight of the liquid is 1, so the fluid is Wood's oil-water average
    (the reference fails without a gas phase)."""
    model = SoftSandModel()
    e = model.elastic(_state())
    assert np.isfinite(e.vp).all()
    pure_water = model.elastic(_state(saturations={"water": [1.0, 1.0]}))
    with_zero_gas = model.elastic(
        _state(
            saturations={"water": [1.0, 1.0], "gas": [0.0, 0.0]},
            densities={"water": [1030.0] * 2, "gas": [200.0] * 2},
        )
    )
    np.testing.assert_allclose(pure_water.vp, with_zero_gas.vp, rtol=1e-14)


def test_gas_lowers_vp_strongly_in_soft_sand() -> None:
    """A little gas (10 %) lowers Vp by more than 3 % under Brie's mixing."""
    model = SoftSandModel(gas="methane", temperature_c=60.0)
    dens = {"water": [1030.0] * 2, "gas": [250.0] * 2}
    wet = model.elastic(_state(saturations={"water": [1.0, 1.0], "gas": [0.0, 0.0]},
                               densities=dens))  # fmt: skip
    gas = model.elastic(_state(saturations={"water": [0.9, 0.9], "gas": [0.1, 0.1]},
                               densities=dens))  # fmt: skip
    assert np.all(gas.vp < 0.97 * wet.vp)


def test_soft_sand_fluid_path_by_hand() -> None:
    """``elastic`` = the rock given the fluid assembled by hand from the building blocks:
    brine alone, oil and brine (Wood), brine and methane (Brie, e = 5), at 60 °C."""
    model = SoftSandModel(gas="methane", temperature_c=60.0, oil_reference_density=860.0)
    phi = np.array([0.2, 0.25, 0.3])
    p = np.array([20.0, 22.0, 24.0])  # MPa
    ov = 45.0
    s = {"oil": np.array([0.0, 0.4, 0.0]), "water": np.array([1.0, 0.6, 0.8]),
         "gas": np.array([0.0, 0.0, 0.2])}  # fmt: skip
    rho = {"oil": np.full(3, 750.0), "water": np.full(3, 1040.0), "gas": np.full(3, 180.0)}
    e = model.elastic(FlowState(phi, p * 1e6, ov * 1e6, s, rho))

    k_w = 1040.0 * rp.brine_velocity(60.0, p, 0.035) ** 2 / 1e6
    k_o = 750.0 * rp.dead_oil_velocity(60.0, p, 860.0) ** 2 / 1e6
    k_g = rp.gas_bulk_modulus(60.0, p, 0.5537)
    k_f = np.array([
        k_w[0],
        1 / (0.4 / k_o[1] + 0.6 / k_w[1]),
        (k_w[2] - k_g[2]) * 0.8**5 + k_g[2],
    ])  # fmt: skip
    rho_f = s["oil"] * 750.0 + s["water"] * 1040.0 + s["gas"] * 180.0
    expected = model._rock(phi, ov - p, rho_f, k_f)
    np.testing.assert_allclose(e.vp, expected.vp, rtol=1e-13)
    np.testing.assert_allclose(e.vs, expected.vs, rtol=1e-13)
    np.testing.assert_allclose(e.density, expected.density, rtol=1e-13)
    assert e.clamped_cells == 0


def test_gas_correlation_outside_its_range_is_refused() -> None:
    """CO₂ at 37 °C is below the correlation's pseudo-reduced temperature (0.88 < 1.05):
    refused unless extrapolation is asked for; a non-physical modulus is refused always."""
    gas_state = _state(saturations={"water": [0.7, 1.0], "gas": [0.3, 0.0]},
                       densities={"water": [1030.0] * 2, "gas": [600.0, 0.0]})  # fmt: skip
    with pytest.raises(ValueError, match="extrapolate_gas_correlation"):
        SoftSandModel(gas_model="batzle_wang").elastic(gas_state)
    assert np.isfinite(SoftSandModel(extrapolate_gas_correlation=True).elastic(gas_state).vp).all()
    cold = SoftSandModel(temperature_c=2.0, extrapolate_gas_correlation=True)
    with pytest.raises(PetroElasticError, match="gas bulk modulus"):
        cold.elastic(
            _state(
                pressure=[30e6, 30e6],
                saturations={"water": [0.7, 1.0], "gas": [0.3, 0.0]},
                densities={"water": [1030.0] * 2, "gas": [800.0] * 2},
            )
        )
    deep = SoftSandModel(gas="methane", temperature_c=60.0)
    with pytest.raises(PetroElasticError, match="pseudo-reduced pressure"):
        deep.elastic(_state(pressure=[80e6, 30e6], overburden_pressure=120e6,
                            saturations={"water": [0.7, 1.0], "gas": [0.3, 0.0]},
                            densities={"water": [1030.0] * 2, "gas": [300.0] * 2}))  # fmt: skip


def test_absent_phases_densities_are_ignored_and_zero_porosity_refused() -> None:
    """A restart may hold 0 or garbage for a phase absent from a cell; φ = 0 is refused
    (Gassmann is 0/0 there)."""
    model = SoftSandModel()
    e = model.elastic(_state(saturations={"oil": [0.0, 0.3], "water": [1.0, 0.7]},
                             densities={"oil": [0.0, 760.0], "water": [1030.0] * 2}))  # fmt: skip
    assert np.isfinite(e.vp).all()
    with pytest.raises(PetroElasticError, match="porosity = 0 at cell 0"):
        model.elastic(_state(porosity=[0.0, 0.2]))


def test_clamped_cells_are_counted() -> None:
    e = StandardModel().elastic(_state(pressure=[50e6, 20e6], overburden_pressure=45e6))
    assert e.clamped_cells == 1


def test_above_critical_porosity_both_models_compute_as_the_reference() -> None:
    """``standard`` extrapolates past φc = 0.4 (to the clay limit 0.443); ``soft_sand``
    caps φ/φc at 0.9 and so holds up to φ < 1 (both as SimulatorWrap computes)."""
    above = _state(porosity=[0.38, 0.42])
    vp = StandardModel().elastic(above).vp
    assert np.isfinite(vp).all()
    assert vp[1] < vp[0]
    soft = SoftSandModel().elastic(_state(porosity=[0.3, 0.42]))
    assert np.isfinite(soft.vp).all()


def test_gas_correlation_is_checked_at_setup() -> None:
    """The simulator calls ``check_phases`` with the deck's phases before any member runs."""
    SoftSandModel().check_phases(["oil", "water"])
    SoftSandModel(gas="methane", temperature_c=60.0).check_phases(["water", "gas"])
    SoftSandModel(extrapolate_gas_correlation=True).check_phases(["water", "gas"])
    with pytest.raises(ValueError, match=r"pseudo-reduced temperature of 0\.876"):
        SoftSandModel(gas_model="batzle_wang").check_phases(["water", "gas"])


@pytest.mark.benchmark(group="geophysics")
@pytest.mark.parametrize("name", ["standard", "soft_sand"])
def test_petro_elastic_at_brugge_size(benchmark: pytest.FixtureRequest, name: str) -> None:
    """Brugge's 60,048 cells, oil and water."""
    rng = np.random.default_rng(1)
    n = 139 * 48 * 9
    state = FlowState(
        porosity=rng.uniform(0.1, 0.3, n), pressure=rng.uniform(20e6, 30e6, n),
        overburden_pressure=45e6,
        saturations={"oil": np.full(n, 0.5), "water": np.full(n, 0.5)},
        densities={"oil": np.full(n, 750.0), "water": np.full(n, 1030.0)},
    )  # fmt: skip
    out = benchmark(petro_elastic_models.create(name).elastic, state)  # type: ignore[operator]
    assert out.vp.shape == (n,)
