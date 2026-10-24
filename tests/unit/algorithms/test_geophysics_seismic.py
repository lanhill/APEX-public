"""Trace synthesis and AVO attributes, against independent references.

- A single interface and a wedge: the trace is ``Σ R_k w(t - τ_k)`` with the exact
  reflectivities and two-way times; the wedge's peak amplitude is at a two-way
  thickness of about 1/(2.6 f) for a Ricker wavelet of peak frequency f (Chung, H. &
  Lawton, D. C. (1995). Amplitude responses of thin beds: sinusoidal approximation versus
  Ricker wavelet. Geophysics, 60(1), 223-230; cf. Widess 1973).
- ``nearest`` (the reference's staircase) converges to ``interfaces`` as its internal
  grid is refined.
- A finite-difference continuity check: under ``interfaces`` the trace's derivative in a
  layer's thickness and velocity converges (the staircase's in the thickness does not),
  also with a wavelet given at 1 ms.
- Zero thickness is no layer; inactive cells give no NaN; the layered overburden's
  two-way time; the attributes on traces with known values.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.geophysics import (
    Columns,
    ConstantOverburden,
    EvanescentWaveError,
    LayeredOverburden,
    RickerWavelet,
    SampledWavelet,
    SeismicModel,
    TimeAxis,
    Zoeppritz,
    avo_attributes,
    time_window,
)

type Floats = NDArray[np.float64]
SHALE = ConstantOverburden(2450.0, 849.0, 2300.0)
RICKER = RickerWavelet(45.0, 0.05)
AXIS = TimeAxis(1.90, 2.10, 0.001)


def _model(**kw: object) -> SeismicModel:
    base: dict[str, object] = {
        "reflectivity": Zoeppritz(), "wavelet": RICKER, "overburden": SHALE,
        "angles_deg": (0.0, 20.0), "time": AXIS,
    }  # fmt: skip
    base.update(kw)
    return SeismicModel(**base)  # type: ignore[arg-type]


def _column(dz: list[float], vp: list[float], top: float = 2400.0) -> Columns:
    v = np.array([vp])
    return Columns.of([top], [dz], v, 0.5 * v, np.full_like(v, 2250.0))


def test_one_layer_is_two_wavelets_at_the_exact_times() -> None:
    """A layer in shale: R at the top and base, at 2z/Vp_shale and + 2h/Vp."""
    col = _column([30.0], [3000.0])
    traces = _model().traces(col)
    t = AXIS.times
    tau1, tau2 = 2 * 2400.0 / 2450.0, 2 * 2400.0 / 2450.0 + 2 * 30.0 / 3000.0
    z = Zoeppritz()
    r1 = z.pp(2450.0, 849.0, 2300.0, 3000.0, 1500.0, 2250.0, [0.0, 20.0])
    r2 = z.pp(3000.0, 1500.0, 2250.0, 2450.0, 849.0, 2300.0, [0.0, 20.0])
    for a in range(2):
        expected = r1[a] * RICKER(t - tau1) + r2[a] * RICKER(t - tau2)
        np.testing.assert_allclose(traces[0, a], expected, rtol=1e-12, atol=1e-15)


def test_wedge_tunes_near_one_over_2_6_f() -> None:
    """The amplitude of a thinning layer peaks at a two-way thickness of about 1/(2.6 f)."""
    f = 30.0
    model = _model(wavelet=RickerWavelet(f, 0.2), angles_deg=(0.0,),
                   time=TimeAxis(1.85, 2.15, 0.0001))  # fmt: skip
    vp = 3000.0
    thickness = np.linspace(1.0, 60.0, 400)
    n = thickness.size
    cols = Columns.of(np.full(n, 2400.0), thickness[:, None], np.full((n, 1), vp),
                      np.full((n, 1), 0.5 * vp), np.full((n, 1), 2250.0))  # fmt: skip
    amp = np.abs(model.traces(cols)[:, 0, :]).max(axis=1)
    peak = 2 * thickness[np.argmax(amp)] / vp
    assert 1 / (2.75 * f) < peak < 1 / (2.45 * f)


def test_nearest_converges_to_interfaces() -> None:
    """The staircase's error falls as its internal grid is refined (about linearly)."""
    rng = np.random.default_rng(5)
    n, nz = 20, 4
    cols = Columns.of(
        rng.uniform(2380, 2420, n), rng.uniform(5, 25, (n, nz)), rng.uniform(2300, 3500, (n, nz)),
        rng.uniform(1100, 1700, (n, nz)), rng.uniform(2100, 2400, (n, nz)),
    )  # fmt: skip
    exact = _model().traces(cols)
    errors = [
        np.abs(_model(sampling="nearest", fine_dt=dt).traces(cols) - exact).max()
        for dt in (5e-4, 5e-5, 5e-6)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02 * np.abs(exact).max()


@pytest.mark.parametrize("parameter", ["thickness", "vp"])
def test_traces_are_smooth_in_the_model(parameter: str) -> None:
    """Central differences of the trace in a layer's thickness (or velocity) converge
    under ``interfaces``: halving the step changes the estimate by O(h²). The staircase's
    do not in the thickness (its traces jump as a time crosses a fine sample)."""

    def trace(x: float, model: SeismicModel) -> Floats:
        dz = [10.0, x if parameter == "thickness" else 12.0, 15.0]
        vp = [2800.0, x if parameter == "vp" else 3100.0, 2600.0]
        return model.traces(_column(dz, vp))

    def ratio(model: SeismicModel) -> float:
        x0, h = (12.0, 0.4) if parameter == "thickness" else (3100.0, 40.0)
        d = [(trace(x0 + s, model) - trace(x0 - s, model)) / (2 * s) for s in (h, h / 2, h / 4)]
        return float(np.abs(d[1] - d[2]).max() / np.abs(d[0] - d[1]).max())

    assert ratio(_model()) < 1 / 3  # 1/4 for a smooth trace
    if parameter == "thickness":
        assert ratio(_model(sampling="nearest")) > 1


def test_file_wavelet_is_accurate_and_smooth_at_real_sampling() -> None:
    """A 45 Hz Ricker given at 1 ms: the cubic spline is within 1e-4 of the peak, and
    the trace stays smooth in a thickness (the FD criterion above)."""
    samples, centre = RICKER.sampled(0.001)
    wavelet = SampledWavelet(samples, 0.001, centre)
    lags = np.linspace(-0.03, 0.03, 601)
    assert np.abs(wavelet(lags) - RICKER(lags)).max() < 1e-4
    model = _model(wavelet=wavelet)

    def trace(x: float) -> Floats:
        return model.traces(_column([10.0, x, 15.0], [2800.0, 3100.0, 2600.0]))

    d = [(trace(12.0 + s) - trace(12.0 - s)) / (2 * s) for s in (0.4, 0.2, 0.1)]
    assert np.abs(d[1] - d[2]).max() < np.abs(d[0] - d[1]).max() / 3
    assert 1 / 300 < wavelet.nyquist_interval() < 1 / 250


def test_zero_thickness_is_no_layer() -> None:
    """A zero-thickness cell changes nothing (its interfaces would not cancel:
    R₁₂ + R₂₃ ≠ R₁₃), and is never evaluated, even where it would be evanescent."""
    with_cell = _model().traces(_column([20.0, 0.0, 15.0], [2800.0, 4000.0, 3100.0]))
    without = _model().traces(_column([20.0, 15.0], [2800.0, 3100.0]))
    np.testing.assert_array_equal(with_cell, without)
    wide = _model(angles_deg=(0.0, 30.0))
    np.testing.assert_array_equal(
        wide.traces(_column([20.0, 0.0, 15.0], [2800.0, 6000.0, 3100.0])),
        wide.traces(_column([20.0, 15.0], [2800.0, 3100.0])),
    )


def test_inactive_cells_take_the_overburden_or_the_cell_above() -> None:
    v = np.array([[2800.0, np.nan, 3100.0]])
    dz = [[20.0, 10.0, 15.0]]
    active = [[True, False, True]]
    cols = Columns.of([2400.0], dz, v, 0.5 * v, np.full_like(v, 2250.0), active)
    for inactive, filled in (("overburden", (2450.0, 849.0, 2300.0)),
                             ("above", (2800.0, 1400.0, 2250.0))):  # fmt: skip
        out = _model(inactive=inactive).traces(cols)
        assert np.isfinite(out).all()
        same = Columns.of([2400.0], dz, [[2800.0, filled[0], 3100.0]],
                          [[1400.0, filled[1], 1550.0]], [[2250.0, filled[2], 2250.0]])  # fmt: skip
        np.testing.assert_allclose(out, _model().traces(same), rtol=1e-13, atol=1e-16)


def test_nan_in_an_active_cell_is_refused() -> None:
    with pytest.raises(ValueError, match=r"vp of the active cell at \(trace 0, layer 1\)"):
        _column([20.0, 10.0], [2800.0, np.nan])


def test_evanescence_names_the_trace_and_interface() -> None:
    """Interface 0 is the overburden's base; the error comes from the reflectivity."""
    model = _model(angles_deg=(0.0, 40.0))
    with pytest.raises(EvanescentWaveError) as info:
        model.traces(_column([20.0, 15.0], [2800.0, 5000.0]))
    assert info.value.index == (0, 1)


def test_layered_overburden_time_and_properties() -> None:
    ob = LayeredOverburden(thickness=(1000.0, 800.0), vp=(1800.0, 2200.0, 2600.0),
                           vs=(700.0, 900.0, 1100.0), rho=(2000.0, 2150.0, 2300.0))  # fmt: skip
    depth = np.array([500.0, 1500.0, 2400.0])
    expected = [1000 / 1800, 2000 / 1800 + 1000 / 2200,
                2000 / 1800 + 1600 / 2200 + 1200 / 2600]  # fmt: skip
    np.testing.assert_allclose(ob.two_way_time(depth), expected, rtol=1e-14)
    np.testing.assert_array_equal(ob.above(depth)[0], [1800.0, 2200.0, 2600.0])
    # At a boundary (1000 m): above is the upper layer, below the lower one.
    assert ob.above(np.array([1000.0]))[0][0] == 1800.0
    assert ob.below(np.array([1000.0]))[0][0] == 2200.0


def test_default_underburden_is_the_layer_below_the_base() -> None:
    """A reservoir whose base lies on an overburden boundary has the lower layer under it."""
    ob = LayeredOverburden(thickness=(2400.0, 30.0), vp=(2450.0, 2600.0, 3400.0),
                           vs=(849.0, 1100.0, 1700.0), rho=(2300.0, 2350.0, 2450.0))  # fmt: skip
    col = _column([30.0], [3000.0])  # top 2400 m, base 2430 m: the 3400 m/s layer's top
    traces = _model(overburden=ob, angles_deg=(0.0,)).traces(col)
    tau2 = 2 * 2400.0 / 2450.0 + 2 * 30.0 / 3000.0
    r2 = Zoeppritz().pp(3000.0, 1500.0, 2250.0, 3400.0, 1700.0, 2450.0, [0.0])[0]
    r1 = Zoeppritz().pp(2450.0, 849.0, 2300.0, 3000.0, 1500.0, 2250.0, [0.0])[0]
    t = AXIS.times
    expected = r1 * RICKER(t - 2 * 2400.0 / 2450.0) + r2 * RICKER(t - tau2)
    np.testing.assert_allclose(traces[0, 0], expected, rtol=1e-12, atol=1e-15)


def test_ricker_support_and_nyquist() -> None:
    """``length`` is the reference's positive half: 197 samples at 0.5 ms for 0.05 s."""
    samples, centre = RICKER.sampled(0.0005)
    assert (samples.size, centre) == (197, 98)
    assert RICKER.half_length == 0.05
    assert RICKER.nyquist_interval() == pytest.approx(1 / 288)


def test_configuration_errors_are_plain_value_errors() -> None:
    with pytest.raises(ValueError, match="angles"):
        _model(angles_deg=(95.0,))
    with pytest.raises(ValueError, match="sampling"):
        _model(sampling="linear")


# ------------------------------------------------------------------ attributes


def test_attributes_on_known_traces() -> None:
    t = np.linspace(0, 1, 1001)
    s = np.stack([np.sin(2 * np.pi * 5 * t), 2 * np.ones_like(t) - 3 * (t > 0.5)])
    traces = s[:, np.newaxis, :]  # (2 traces, 1 angle, n_time)
    window = time_window(t, 0.0, 1.0)
    rms = avo_attributes.create("rms")(traces, window)[:, 0]
    assert rms[0] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert rms[1] == pytest.approx(np.sqrt((501 * 4 + 500 * 1) / 1001), rel=1e-12)
    np.testing.assert_allclose(avo_attributes.create("min")(traces, window)[:, 0], [-1, -1],
                               atol=1e-5)  # fmt: skip
    np.testing.assert_allclose(avo_attributes.create("max")(traces, window)[:, 0], [1, 2],
                               atol=1e-5)  # fmt: skip
    assert avo_attributes.create("sum_negative")(traces, window)[1, 0] == pytest.approx(-500)
    assert avo_attributes.create("mean")(traces, window)[1, 0] == pytest.approx(
        (501 * 2 - 500) / 1001
    )


def test_per_trace_windows_and_the_trace_attribute() -> None:
    t = np.arange(10.0)
    traces = np.arange(20.0).reshape(2, 1, 10)
    window = time_window(t, [2.0, 5.0], [4.0, 9.0], n_trace=2)
    np.testing.assert_allclose(avo_attributes.create("mean")(traces, window)[:, 0], [3, 17])
    kept = avo_attributes.create("trace")(traces, window)
    assert kept[0, 0, 1] == 0
    assert kept[0, 0, 3] == 3
    assert kept[1, 0, 9] == 19
    with pytest.raises(ValueError, match="trace 1's window"):
        time_window(t, [2.0, 9.5], [4.0, 9.9])


@pytest.mark.benchmark(group="geophysics")
def test_synthesis_at_brugge_size(benchmark: pytest.FixtureRequest) -> None:
    """Brugge's 139 x 48 columns of 9 layers, two angles, 100 output samples."""
    rng = np.random.default_rng(0)
    n, nz = 139 * 48, 9
    cols = Columns.of(
        rng.uniform(1600, 1700, n), rng.uniform(2, 15, (n, nz)), rng.uniform(2500, 3500, (n, nz)),
        rng.uniform(1200, 1700, (n, nz)), rng.uniform(2100, 2400, (n, nz)),
    )  # fmt: skip
    model = _model(wavelet=RickerWavelet(30.0, 0.1), angles_deg=(10.0, 20.0),
                   time=TimeAxis(1.25, 1.45, 0.002))  # fmt: skip
    out = benchmark(model.traces, cols)  # type: ignore[operator]
    assert out.shape == (n, 2, 100)


def test_the_time_shift_of_a_layer_is_twice_its_change_of_vertical_slowness_times_h() -> None:
    """A layer of thickness h whose velocity changes from v1 to v2
    shifts the base by exactly 2h (1/v2 - 1/v1), positive when slower."""
    model = _model()
    h, v1, v2 = 30.0, 3000.0, 2600.0
    before = model.base_time(_column([h, 10.0], [v1, 3200.0]))
    after = model.base_time(_column([h, 10.0], [v2, 3200.0]))
    assert after - before == pytest.approx(2 * h * (1 / v2 - 1 / v1), rel=1e-12)
    assert (after - before)[0] > 0
    # The overburden's time to the top is the synthesis's: 2400 m at 2450 m/s.
    assert before[0] == pytest.approx(2 * 2400.0 / 2450.0 + 2 * h / v1 + 2 * 10.0 / 3200.0)


def test_the_time_shift_is_smooth_in_the_model() -> None:
    """Finite differences of the base time match its derivative, -2h/v² per layer."""
    model = _model()
    h, v = 25.0, 2800.0
    step = 1e-3
    up = model.base_time(_column([h], [v + step]))[0]
    down = model.base_time(_column([h], [v - step]))[0]
    assert (up - down) / (2 * step) == pytest.approx(-2 * h / v**2, rel=1e-6)


def test_the_time_shift_attribute_is_not_a_reduction_of_the_traces() -> None:
    attribute = avo_attributes.create("time_shift")
    assert attribute.is_map
    with pytest.raises(TypeError, match="computed from the columns"):
        attribute(np.zeros((1, 1, 3)), np.ones(3, dtype=bool))
