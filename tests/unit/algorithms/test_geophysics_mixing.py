"""How brine and CO₂ share the pores: soft_sand's mixing laws.

Independent references: the bounds themselves (Wood's uniform saturation below,
Gassmann-Hill's patchy saturation above; Mavko, Mukerji & Dvorkin 2009, The Rock Physics
Handbook, §6.17), Brie's law at e = 1 (the Voigt average), and a Utsira-like sand at
Sleipner's conditions (37 °C, 8.5 MPa, CO₂ at 700 kg/m3) for Brie's departures from the
bounds.
"""

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import FlowState, SoftSandModel
from apex.algorithms.geophysics import co2 as co2_module
from apex.algorithms.geophysics.petro_elastic import _cells

#: CO₂ saturations from full brine to full CO₂.
SG = np.linspace(0.0, 1.0, 101)


def _state(gas: str = "gas", other: str = "water") -> FlowState:
    n = SG.size
    return FlowState(
        porosity=np.full(n, 0.36),
        pressure=np.full(n, 8.5e6),
        overburden_pressure=np.full(n, 15.5e6),
        saturations={other: 1.0 - SG, gas: SG},
        densities={other: np.full(n, 1030.0), gas: np.full(n, 700.0)},
    )


def _model(**options: Any) -> SoftSandModel:  # noqa: ANN401
    """A Utsira-like soft sand (critical porosity 0.4)."""
    return SoftSandModel(temperature_c=37.0, critical_porosity=0.4, **options)


#: Batzle & Wang's CO₂ (extrapolated), so these need no CoolProp.
CORRELATION = {"gas_model": "batzle_wang", "extrapolate_gas_correlation": True}


def _vp(**options: Any) -> NDArray[np.float64]:  # noqa: ANN401
    out: NDArray[np.float64] = _model(**CORRELATION, **options).elastic(_state()).vp
    return out


def test_patchiness_lies_within_the_bounds_and_reaches_them() -> None:
    wood, patchy = _vp(mixing="wood"), _vp(mixing="patchy")
    assert np.all(wood <= patchy + 1e-9)
    for q in (0.0, 0.2, 0.5, 0.8, 1.0):
        blend = _vp(mixing="patchiness", patchiness=q)
        assert np.all(wood - 1e-9 <= blend)
        assert np.all(blend <= patchy + 1e-9)
    np.testing.assert_array_equal(_vp(mixing="patchiness", patchiness=0.0), wood)
    np.testing.assert_allclose(_vp(mixing="patchiness", patchiness=1.0), patchy, rtol=1e-14)
    # One fluid alone: every law is Gassmann with that fluid.
    for law in ("patchy", "patchiness", "brie"):
        np.testing.assert_allclose(_vp(mixing=law)[[0, -1]], wood[[0, -1]], rtol=1e-12)


def test_brie_at_exponent_one_is_the_voigt_average() -> None:
    model = _model(**CORRELATION, brie_exponent=1.0)
    c = _cells(_state(), 1.0, model.min_effective_pressure, True)
    _, k_fluid, _ = model.fluid(c)
    voigt = (1.0 - SG) * k_fluid[0] + SG * k_fluid[-1]
    np.testing.assert_allclose(k_fluid, voigt, rtol=1e-12)


def test_brie_leaves_both_bounds_in_a_utsira_like_sand() -> None:
    """Pinned, with CO₂'s equation of state at 700 kg/m3 and 37 °C
    (73 MPa): Brie at e = 5 falls below Wood from 52 % CO₂, by up to 23 % in the fluid's
    modulus and 9.6 m/s in Vp; at e = 2 it exceeds patchy below 24 % CO₂. The figures
    follow CoolProp's Span & Wagner (8.0.0, the co2 extra's pin)."""
    pytest.importorskip("CoolProp")
    assert k_fluid_at_full_co2() == pytest.approx(72.745, abs=0.01)  # MPa

    def k_fluid(**options: Any) -> NDArray[np.float64]:  # noqa: ANN401
        model = _model(**options)
        out: NDArray[np.float64] = model.fluid(
            _cells(_state(), 1.0, model.min_effective_pressure, True)
        )[1]
        return out

    wood_k, brie_k = k_fluid(mixing="wood"), k_fluid(brie_exponent=5.0)
    deficit = (wood_k - brie_k) / wood_k
    assert SG[deficit > 0].min() == pytest.approx(0.52)
    assert deficit.max() == pytest.approx(0.233, abs=0.001)
    wood_vp = _model(mixing="wood").elastic(_state()).vp
    brie_vp = _model(brie_exponent=5.0).elastic(_state()).vp
    assert (wood_vp - brie_vp).max() == pytest.approx(9.56, abs=0.01)
    patchy_vp = _model(mixing="patchy").elastic(_state()).vp
    above = SG[_model(brie_exponent=2.0).elastic(_state()).vp > patchy_vp]
    assert above.min() == pytest.approx(0.01)
    assert above.max() == pytest.approx(0.23)


def test_patchy_mixing_takes_two_fluid_phases() -> None:
    model = _model(**CORRELATION, mixing="patchy")
    model.check_phases(["water", "gas"])
    with pytest.raises(ValueError, match="two fluid phases"):
        model.check_phases(["oil", "water", "gas"])
    n = 3
    three = FlowState(
        porosity=np.full(n, 0.3),
        pressure=np.full(n, 8.5e6),
        overburden_pressure=np.full(n, 15.5e6),
        saturations={"oil": np.full(n, 0.2), "water": np.full(n, 0.5), "gas": np.full(n, 0.3)},
        densities={"oil": np.full(n, 800.0), "water": np.full(n, 1030.0),
                   "gas": np.full(n, 200.0)},
    )  # fmt: skip
    with pytest.raises(ValueError, match="two fluid phases"):
        model.elastic(three)
    _model(**CORRELATION, mixing="wood").elastic(three)  # Wood and Brie take three


@pytest.mark.parametrize("q", [-0.1, 1.5])
def test_patchiness_must_be_in_zero_to_one(q: float) -> None:
    with pytest.raises(ValueError, match="patchiness must be in"):
        _model(mixing="patchiness", patchiness=q)


def k_fluid_at_full_co2() -> float:
    """CO₂'s modulus (MPa) at 700 kg/m3 and 37 °C, through soft_sand's fluid."""
    model = _model(mixing="wood")
    return float(model.fluid(_cells(_state(), 1.0, model.min_effective_pressure, True))[1][-1])


def test_patchy_is_the_harmonic_mean_of_the_p_wave_moduli() -> None:
    """Gassmann-Hill by hand at an interior saturation, from the one-fluid rocks: the
    bounds alone would not tell M = K + 4/3 G from K (review of phase 2b, step 3)."""
    vp, vs, rho = (getattr(_model(**CORRELATION, mixing="patchy").elastic(_state()), name)
                   for name in ("vp", "vs", "density"))  # fmt: skip
    wood = _model(**CORRELATION, mixing="wood").elastic(_state())
    # The one-fluid rocks' P-wave moduli, from the ends (full brine, full CO₂), whose bulk
    # densities differ from the interior cell's: M = rho_b vp² of each.
    m_brine = wood.density[0] * wood.vp[0] ** 2
    m_co2 = wood.density[-1] * wood.vp[-1] ** 2
    for n in (10, 37, 80):
        s = SG[n]
        m = 1.0 / ((1.0 - s) / m_brine + s / m_co2)
        assert vp[n] == pytest.approx(np.sqrt(m / rho[n]), rel=1e-12)
        g = wood.density[0] * wood.vs[0] ** 2  # the dry shear modulus, the same everywhere
        assert vs[n] == pytest.approx(np.sqrt(g / rho[n]), rel=1e-12)
        assert rho[n] == pytest.approx((1 - s) * wood.density[0] + s * wood.density[-1])


def test_without_coolprop_only_a_gas_phase_needs_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs without CoolProp too (a review of phase 2b)."""

    def missing() -> None:
        raise ImportError("gas_model = 'span_wagner' needs CoolProp: install APEX's co2 extra")

    monkeypatch.setattr(co2_module, "_coolprop", missing)
    model = SoftSandModel()
    model.check_phases(["oil", "water"])  # no gas: nothing to build
    with pytest.raises(ValueError, match="co2 extra"):
        model.check_phases(["water", "gas"])
