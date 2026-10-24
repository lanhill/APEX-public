"""Petro-elastic models against SimulatorWrap's ``standardrp`` and ``softsandrp``.

The references (``tests/data/reference/geophysics/pem_*.npz``) are the reference's own
``calc_props`` and building blocks at ``d29dc57``, on 200 random cells each (with four
negative effective pressures, which both clamp to 0.01 MPa). ``standard`` is compared
whole. ``soft_sand``'s fluids differ by design (pet_parity.md G3-G6), so its rock (dry
rock and Gassmann) is compared with the reference's own pore fluid injected, and its
fluid pieces one by one; the reference's defects are pinned where they show.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import ElasticProperties, FlowState, SoftSandModel, StandardModel
from apex.algorithms.geophysics import rock_physics as rp

pytestmark = pytest.mark.pet_parity

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "reference" / "geophysics"
#: The same closed forms in float64 (measured ≤ 4e-15).
RTOL = 1e-10
PHASES = {"OIL": "oil", "WAT": "water", "GAS": "gas"}
type Floats = NDArray[np.float64]


def _load(case: str) -> dict[str, Floats]:
    path = REFERENCE / f"{case}.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; run tools/geophysics_reference/generate.py {case}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _compare(e: ElasticProperties, ref: dict[str, Floats]) -> None:
    np.testing.assert_allclose(e.vp, ref["out_vp"], rtol=RTOL)
    np.testing.assert_allclose(e.vs, ref["out_vs"], rtol=RTOL)
    np.testing.assert_allclose(e.density, ref["out_density"], rtol=RTOL)
    np.testing.assert_allclose(e.bulk_modulus / 1e6, ref["out_bulk_modulus_mpa"], rtol=RTOL)
    np.testing.assert_allclose(e.shear_modulus / 1e6, ref["out_shear_modulus_mpa"], rtol=RTOL)


def test_standard_model_is_standardrp() -> None:
    ref = _load("pem_standard")
    state = FlowState(
        porosity=ref["in_porosity"],
        pressure=ref["in_pressure_mpa"] * 1e6,
        overburden_pressure=ref["in_overburden_mpa"] * 1e6,
        saturations={
            PHASES[str(p)]: s for p, s in zip(ref["in_phases"], ref["in_saturations"], strict=True)
        },
    )
    _compare(StandardModel().elastic(state), ref)


def test_soft_sand_rock_is_softsandrp_given_its_fluid() -> None:
    """Hertz-Mindlin (n = 9, φc = 0.36), soft sand with φ/φc capped at 0.9, Gassmann."""
    ref = _load("pem_soft_sand")
    e = SoftSandModel()._rock(
        ref["in_porosity"], ref["out_effective_mpa"], ref["out_fluid_density"],
        ref["out_fluid_bulk_mpa"],
    )  # fmt: skip
    _compare(e, ref)
    assert np.any(ref["in_porosity"] > 0.9 * 0.36)  # the cap is exercised


def test_soft_sand_building_blocks_are_softsandrp() -> None:
    ref = _load("pem_soft_sand")
    p, peff = ref["in_pressure_mpa"], ref["out_effective_mpa"]
    np.testing.assert_allclose(np.maximum(ref["in_overburden_mpa"] - p, 0.01), peff, rtol=0, atol=0)
    hm = rp.hertz_mindlin(peff, 37000.0, 44000.0, 9.0, 0.36)
    np.testing.assert_allclose(hm[0], ref["out_hm_bulk_mpa"], rtol=RTOL)
    np.testing.assert_allclose(hm[1], ref["out_hm_shear_mpa"], rtol=RTOL)
    ratio = np.minimum(ref["in_porosity"] / 0.36, 0.9)
    dry = rp.soft_sand_dry_moduli(ratio, *hm, 37000.0, 44000.0)
    np.testing.assert_allclose(dry[0], ref["out_dry_bulk_mpa"], rtol=RTOL)
    np.testing.assert_allclose(dry[1], ref["out_dry_shear_mpa"], rtol=RTOL)
    np.testing.assert_allclose(
        rp.gas_bulk_modulus(37.0, p, 1.5189), ref["out_co2_bulk_mpa"], rtol=RTOL
    )
    np.testing.assert_allclose(
        rp.gas_bulk_modulus(37.0, p, 0.5537), ref["out_methane_bulk_mpa"], rtol=RTOL
    )


def test_softsandrp_water_defects_are_pinned() -> None:
    """G3: the reference's water modulus uses the pore fluid's (mixture's) density.
    G4: its brine density adds Batzle & Wang's salinity term, in g/cm³, to kg/m³.
    The velocities themselves are Batzle & Wang's, as APEX's."""
    ref = _load("pem_soft_sand")
    p, rho_mix = ref["in_pressure_mpa"], ref["out_fluid_density"]
    v_water = rp.water_velocity(37.0, p)
    np.testing.assert_allclose(ref["out_water_bulk_mpa"], rho_mix * v_water**2 * 1e-6, rtol=RTOL)
    t, s = 37.0, 0.035
    x = 300 * p - 2400 * p * s + t * (80 + 3 * t - 3300 * s - 13 * p + 47 * p * s)
    rho_brine = rho_mix + s * (0.668 + 0.44 * s + 1e-6 * x)
    v_brine = rp.brine_velocity(t, p, s)
    np.testing.assert_allclose(ref["out_brine_bulk_mpa"], rho_brine * v_brine**2 * 1e-6, rtol=RTOL)


def test_softsandrp_oil_velocity_is_batzle_wang_and_its_modulus_is_in_pa() -> None:
    """G5: the reference's oil modulus is ``rho v²`` in Pa (the others are in MPa); its
    velocity, ``sqrt(K / rho)``, is Batzle & Wang's dead-oil velocity, as APEX's."""
    ref = _load("pem_soft_sand")
    # Where the mixture's density exceeds 1080 kg/m³ the formula is NaN in both.
    ok = ref["out_fluid_density"] < 1080.0
    assert ok.sum() > 100
    rho = ref["out_fluid_density"][ok]
    v = rp.dead_oil_velocity(37.0, ref["in_pressure_mpa"][ok], rho)
    np.testing.assert_allclose(ref["out_oil_bulk_pa"][ok], rho * v**2, rtol=RTOL)
