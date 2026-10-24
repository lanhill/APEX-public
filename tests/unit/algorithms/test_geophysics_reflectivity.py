"""PP reflectivity models against PyLops, bruges and closed forms.

References (``tests/data/reference/geophysics/reflectivity.npz``, written by
``tools/geophysics_reference/generate.py`` with PyLops 2.8.0 and bruges 0.5.4 on 240
random sedimentary interfaces at 0-40 degrees):

- PyLops' ``approx_zoeppritz_pp`` (SimulatorWrap's ``pp_func`` in the 3dBox example) and
  ``zoeppritz_pp`` (the 4 x 4 scattering matrix); its ``akirichards`` and ``fatti``
  coefficients;
- bruges' ``zoeppritz_rpp``, ``zoeppritz`` (scattering matrix), ``akirichards``,
  ``fatti`` and ``shuey``.

Independent checks: Aki & Richards (1980), normal incidence ``(Z₂ - Z₁)/(Z₂ + Z₁)`` and
the reciprocity ``R₁₂ = -R₂₁`` there, and the approximations tending to the exact solution
as the second power of the contrasts.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import (
    AkiRichards,
    ApproxZoeppritzPP,
    EvanescentWaveError,
    Fatti,
    PPReflectivity,
    Shuey,
    Zoeppritz,
    reflectivity,
)

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "geophysics"

#: The kernels and the libraries compute the same closed forms in float64.
RTOL = 1e-12

MODELS: dict[str, PPReflectivity] = {
    "zoeppritz": Zoeppritz(),
    "approx_zoeppritz_pp": ApproxZoeppritzPP(),
    "aki_richards": AkiRichards(),
    "aki_richards_linear": AkiRichards(form="linear"),
    "fatti": Fatti(),
    "shuey": Shuey(),
}


@pytest.fixture(scope="module")
def ref() -> dict[str, NDArray[np.float64]]:
    path = REFERENCE / "reflectivity.npz"
    if not path.exists():
        pytest.fail(f"missing {path}; run tools/geophysics_reference/generate.py reflectivity")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


type Floats = NDArray[np.float64]


def _elastic(ref: dict[str, Floats]) -> tuple[Floats, Floats, Floats, Floats, Floats, Floats]:
    return (
        ref["in_vp1"], ref["in_vs1"], ref["in_rho1"], ref["in_vp2"], ref["in_vs2"], ref["in_rho2"]
    )  # fmt: skip


def _close(actual: NDArray[np.float64], expected: NDArray[np.float64]) -> None:
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * np.abs(expected).max())


def _contrast(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2 * (x2 - x1) / (x2 + x1)


@pytest.mark.parametrize(
    ("model", "library"),
    [
        ("zoeppritz", "pylops_approx_zoeppritz_pp"),
        ("zoeppritz", "pylops_zoeppritz_pp"),
        ("zoeppritz", "bruges_zoeppritz_rpp"),
        ("zoeppritz", "bruges_zoeppritz"),
        ("approx_zoeppritz_pp", "pylops_approx_zoeppritz_pp"),
        ("aki_richards", "bruges_akirichards"),
        ("fatti", "bruges_fatti"),
        ("shuey", "bruges_shuey"),
    ],
)
def test_matches_the_libraries(
    ref: dict[str, NDArray[np.float64]], model: str, library: str
) -> None:
    """Each model equals the library's function to rounding (references are (angle, n))."""
    out = MODELS[model].pp(*_elastic(ref), ref["in_angles"])
    _close(out.T, ref[f"out_{library}"])


def test_linear_forms_match_pylops_coefficients(ref: dict[str, NDArray[np.float64]]) -> None:
    """PyLops' Aki-Richards and Fatti coefficients times the contrasts ``2Δx/(x₁ + x₂)``."""
    vp1, vs1, rho1, vp2, vs2, rho2 = _elastic(ref)
    contrasts = {
        "akirichards": (_contrast(vp1, vp2), _contrast(vs1, vs2), _contrast(rho1, rho2)),
        "fatti": (
            _contrast(vp1 * rho1, vp2 * rho2),
            _contrast(vs1 * rho1, vs2 * rho2),
            _contrast(rho1, rho2),
        ),
    }
    for name, model in (("akirichards", MODELS["aki_richards_linear"]), ("fatti", Fatti())):
        expected = sum(ref[f"out_pylops_{name}_g{k + 1}"] * contrasts[name][k] for k in range(3))
        _close(model.pp(*_elastic(ref), ref["in_angles"]).T, np.asarray(expected))


def test_normal_incidence_and_reciprocity() -> None:
    """At 0°: exact and Fatti give ``(Z₂ - Z₁)/(Z₂ + Z₁)``; exact R₁₂ = -R₂₁ (Aki & Richards)."""
    rng = np.random.default_rng(1)
    vp1, vp2 = rng.uniform(1500, 4500, (2, 50))
    vs1, vs2 = 0.5 * vp1, 0.45 * vp2
    rho1, rho2 = rng.uniform(1800, 2700, (2, 50))
    z1, z2 = vp1 * rho1, vp2 * rho2
    expected = ((z2 - z1) / (z2 + z1))[:, np.newaxis]
    for model in (Zoeppritz(), Fatti()):
        np.testing.assert_allclose(
            model.pp(vp1, vs1, rho1, vp2, vs2, rho2, [0.0]), expected, rtol=1e-13, atol=1e-15
        )
    forward = Zoeppritz().pp(vp1, vs1, rho1, vp2, vs2, rho2, [0.0])
    backward = Zoeppritz().pp(vp2, vs2, rho2, vp1, vs1, rho1, [0.0])
    np.testing.assert_allclose(forward, -backward, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("model", ["aki_richards", "aki_richards_linear", "fatti", "shuey"])
@pytest.mark.parametrize(
    ("background", "direction"),
    [
        ((3000.0, 1350.0, 2300.0), (1.0, -0.7, 0.5)),  # Vs/Vp = 0.45
        ((2400.0, 1300.0, 2150.0), (-0.8, 1.0, 0.3)),  # Vp down, Vs up
    ],
)
def test_approximations_converge_quadratically(
    model: str, background: tuple[float, float, float], direction: tuple[float, float, float]
) -> None:
    """The linearised forms differ from the exact solution by O(contrast²), up to 30°.

    The error's slope against the contrast, fitted over three contrasts a decade apart,
    is about 2; a first-order error in a coefficient would make it about 1.
    """
    angles = np.array([5.0, 15.0, 30.0])
    vp, vs, rho = background
    contrasts = np.array([0.004, 0.0004, 0.00004])
    errors = []
    for eps in contrasts:
        vp2, vs2, rho2 = (x * (1 + eps * d) for x, d in zip(background, direction, strict=True))
        exact = Zoeppritz().pp(vp, vs, rho, vp2, vs2, rho2, angles)
        approx = MODELS[model].pp(vp, vs, rho, vp2, vs2, rho2, angles)
        errors.append(np.abs(approx - exact).max())
    slope = np.polyfit(np.log10(contrasts), np.log10(errors), 1)[0]
    assert 1.8 < slope < 2.2


@pytest.mark.parametrize("name", sorted(MODELS))
def test_evanescence_is_refused_just_past_the_critical_angle(name: str) -> None:
    """Up to the critical angle of the transmitted P wave the models answer; past it, they
    refuse, naming the interface, the angle and the wave."""
    model = MODELS[name]
    vp1 = np.array([[2000.0, 2000.0], [2000.0, 2000.0]])
    vp2 = np.array([[2100.0, 2100.0], [2100.0, 4000.0]])
    vs1, vs2 = 0.5 * vp1, 0.5 * vp2
    rho = np.full_like(vp1, 2200.0)
    critical = float(np.degrees(np.arcsin(2000.0 / 4000.0)))  # 30° at interface (1, 1)
    out = model.pp(vp1, vs1, rho, vp2, vs2, rho, [10.0, critical])
    assert out.shape == (2, 2, 2)
    assert np.isfinite(out).all()
    with pytest.raises(EvanescentWaveError) as info:
        model.pp(vp1, vs1, rho, vp2, vs2, rho, [10.0, critical + 1e-6])
    assert info.value.index == (1, 1)
    assert info.value.angle_deg == pytest.approx(critical + 1e-6)
    assert info.value.wave == "transmitted P"
    assert info.value.count == 1


def test_negative_bulk_modulus_is_refused() -> None:
    """``Vs >= (√3/2) Vp`` is a negative bulk modulus, named as such (not as evanescence)."""
    with pytest.raises(ValueError, match=r"lower medium .* negative bulk modulus"):
        Zoeppritz().pp(2000.0, 1000.0, 2000.0, 1500.0, 1400.0, 2000.0, [10.0])


def test_shapes_broadcast_and_angles_go_last() -> None:
    """``(n_trace, n_interface)`` elastic arrays and scalars broadcast; angles are last."""
    vp1 = np.full((4, 3), 2500.0)
    out = Zoeppritz().pp(vp1, 1200.0, 2200.0, 2700.0, 1300.0, 2300.0, [0.0, 10.0, 20.0])
    assert out.shape == (4, 3, 3)
    assert Zoeppritz().pp(2500.0, 1200.0, 2200.0, 2700.0, 1300.0, 2300.0, [5.0]).shape == (1,)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((2500.0, 0.0, 2200.0, 2700.0, 1300.0, 2300.0, [10.0]), "vs1 must be finite and positive"),
        ((2500.0, 1200.0, np.nan, 2700.0, 1300.0, 2300.0, [10.0]), "rho1 must be finite"),
        ((2500.0, 1200.0, 2200.0, 2700.0, 1300.0, 2300.0, [90.0]), r"\[0, 90\)"),
        ((2500.0, 1200.0, 2200.0, 2700.0, 1300.0, 2300.0, []), "non-empty 1-D"),
        ((np.ones(3), 1.0, 1.0, np.ones(2), 1.0, 1.0, [1.0]), "do not broadcast"),
    ],
)
def test_bad_inputs_are_refused(args: tuple[object, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Zoeppritz().pp(*args)  # type: ignore[arg-type]


def test_registry_names() -> None:
    assert set(reflectivity.names()) >= {
        "zoeppritz", "approx_zoeppritz_pp", "aki_richards", "fatti", "shuey"
    }  # fmt: skip
    assert isinstance(reflectivity.create("aki_richards", form="linear"), AkiRichards)
    with pytest.raises(ValueError, match="form"):
        AkiRichards(form="other")  # type: ignore[arg-type]


@pytest.mark.benchmark(group="geophysics")
def test_zoeppritz_at_brugge_size(benchmark: pytest.FixtureRequest) -> None:
    """Brugge's 139 x 48 columns with 9 layers (11 interfaces) at two angles."""
    rng = np.random.default_rng(3)
    shape = (139 * 48, 11)
    vp1, vp2 = rng.uniform(2500, 3500, (2, *shape))
    rho1, rho2 = rng.uniform(2100, 2500, (2, *shape))
    out = benchmark(  # type: ignore[operator]
        Zoeppritz().pp, vp1, 0.5 * vp1, rho1, vp2, 0.5 * vp2, rho2, [10.0, 20.0]
    )
    assert out.shape == (*shape, 2)
