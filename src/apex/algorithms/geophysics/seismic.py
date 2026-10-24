"""Synthetic angle-stack traces from a column's elastic properties.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_avo``:
             ``_calc_avo_props_active_cells``, L1809-1938)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
References : Ricker, N. (1953). The form and laws of propagation of seismic wavelets.
             Geophysics, 18(1), 10-40.
             Widess, M. B. (1973). How thin is a thin bed? Geophysics, 38(6), 1176-1180
             (the tuning a wedge shows).
Behaviour  : each column (trace) is converted from depth to two-way time under an
             overburden; PP reflectivities are formed at the column's interfaces for each
             angle; the trace is the reflectivity convolved with a wavelet, at the
             output times. Two samplings (``sampling``):

             - ``"interfaces"`` (the default): the reflectivity R_k at the exact
               interface times τ_k, and the trace ``s(t) = Σ_k R_k w(t - τ_k)``
               evaluated at the output times. It is smooth in every velocity, density
               and positive thickness (a cell thinning to zero jumps to the merged
               interface, the physical limit), and needs no internal time grid.
             - ``"nearest"``: the reference's. Properties are sampled nearest-neighbour
               on a fine grid (``fine_dt``, 0.5 ms), reflectivities taken between
               consecutive samples, convolved with the sampled wavelet (``same``), and
               the result sampled nearest-neighbour at the output times. A staircase
               in every parameter: for parity only. Its memory is about a dozen
               (n_trace, n_fine) arrays at once (2.5 GB for Brugge's 6,672 traces over
               a 2 s window), unchunked.
Differences: - Zero-thickness cells take no time and form no interface (their neighbours'
               interface is formed directly). The reference's staircase never samples
               them either (their tied times), so the two agree, but for inactive ones.
             - Inactive cells take the overburden's properties (``inactive =
               "overburden"``, the default) or the nearest active cell's above
               (``"above"``); the reference's are NaN, and a NaN velocity turns every
               later time of the column into NaN.
             - Each angle is computed on its own. The reference's multi-angle loop
               returns the last angle's traces for every angle (``avo_data`` aliases the
               buffer it overwrites; pet_parity.md G7).

Shapes: ``top_depth`` ``(n_trace,)``; ``thickness``, ``vp``, ``vs``, ``rho``, ``active``
``(n_trace, n_z)`` from the top layer down; traces ``(n_trace, n_angle, n_time)``.
Depths and thicknesses in m, velocities in m/s, densities in kg/m³, times in s.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

from apex.algorithms.geophysics.errors import GeophysicsModelError
from apex.algorithms.geophysics.reflectivity import PPReflectivity

type Array = NDArray[np.float64]

#: Trace-chunk size: the interfaces' wavelet matrix is (chunk, n_interface, n_time), 128 MB
#: in float64; with the wavelet's temporaries the peak is about five times that.
_CHUNK_ELEMENTS = 2**24


@dataclass(frozen=True)
class TimeAxis:
    """Output times ``arange(t_min, t_max, dt)`` (s), as the reference forms them."""

    t_min: float
    t_max: float
    dt: float

    def __post_init__(self) -> None:
        """Refuse an empty or reversed axis."""
        if not (self.dt > 0 and self.t_max > self.t_min):
            raise ValueError(f"need dt > 0 and t_max > t_min, got {self}")

    @property
    def times(self) -> Array:
        """The sample times ``(n_time,)``."""
        out: Array = np.arange(self.t_min, self.t_max, self.dt)
        return out


class Wavelet(Protocol):
    """A zero-phase source wavelet, a function of the lag (s) from its centre."""

    #: Half the wavelet's support (s): beyond it the wavelet is 0 (or negligible).
    half_length: float

    def __call__(self, lag: Array) -> Array:
        """Return the wavelet at the lags (any shape)."""
        ...

    def sampled(self, dt: float) -> tuple[Array, int]:
        """Return the wavelet sampled at ``dt`` and the index of its centre."""
        ...


@dataclass(frozen=True)
class RickerWavelet:
    """``w(t) = (1 - 2π²f²t²) exp(-π²f²t²)`` (Ricker 1953), ``f`` the peak frequency.

    ``length`` is the reference's ``wave_len``: the positive half of the sampled
    wavelet's support (``arange(0, wave_len - dt, dt)``, mirrored).
    """

    frequency: float
    length: float

    def __post_init__(self) -> None:
        """Refuse a non-positive frequency or length."""
        if self.frequency <= 0 or self.length <= 0:
            raise ValueError(f"frequency and length must be positive, got {self}")

    @property
    def half_length(self) -> float:
        """Half the sampled support (s); the analytic wavelet is ~1e-21 there at 45 Hz."""
        return self.length

    def __call__(self, lag: Array) -> Array:
        """Return the wavelet at the lags, not truncated.

        A cut would make the trace jump whenever an interface moves a sample across it
        (by about 1e-4 of the peak at ±length/2 for the 3dBox wavelet); ``length``
        applies to :meth:`sampled`, the reference's form.
        """
        a = (np.pi * self.frequency * lag) ** 2
        out: Array = (1.0 - 2.0 * a) * np.exp(-a)
        return out

    def sampled(self, dt: float) -> tuple[Array, int]:
        """Return the reference's sampled wavelet (PyLops' ``ricker``) and its centre.

        As ``flow_avo`` builds it: the positive half ``arange(0, length - dt, dt)``,
        cropped to an odd count, mirrored.
        """
        half = np.arange(0.0, self.length - dt, dt)
        if half.size % 2 == 0:
            half = half[:-1]
        t = np.concatenate((-half[:0:-1], half))
        a = (np.pi * self.frequency * t) ** 2
        w: Array = (1.0 - 2.0 * a) * np.exp(-a)
        return w, int(np.argmax(np.abs(w)))

    def nyquist_interval(self) -> float:
        """Return the largest sampling interval that resolves the wavelet (s).

        The Ricker amplitude spectrum ``(f/f₀)² exp(1 - (f/f₀)²)`` (peak 1 at f₀) falls
        below 0.1 % beyond about 3.2 f₀, so ``1/(2 · 3.2 f₀)``.
        """
        return 1.0 / (6.4 * self.frequency)


@dataclass(frozen=True, eq=False)
class SampledWavelet:
    """A wavelet given as samples (e.g. from a file), a C² cubic spline in the lag.

    A linear interpolation would be off by 1.5 % of the peak for a 45 Hz Ricker at 1 ms
    and 5 % at 2 ms, and kinked at every sample (ensemble gradients see the kinks); the
    cubic is within 6e-5 and 1e-3. Outside its samples the wavelet is 0: a file's
    wavelet should taper to 0 at both ends.
    """

    samples: Array
    dt: float
    centre: int

    def __post_init__(self) -> None:
        """Refuse an empty or non-finite wavelet, or a centre outside it."""
        if self.samples.ndim != 1 or self.samples.size < 4 or self.dt <= 0:
            raise ValueError("a sampled wavelet needs a 1-D array of 4+ samples and dt > 0")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError("the wavelet's samples must be finite")
        if not 0 <= self.centre < self.samples.size:
            raise ValueError(f"centre {self.centre} is outside the {self.samples.size} samples")

    @property
    def _lags(self) -> Array:
        out: Array = (np.arange(self.samples.size, dtype=np.float64) - self.centre) * self.dt
        return out

    @property
    def half_length(self) -> float:
        """The larger side of the support (s)."""
        return max(self.centre, self.samples.size - 1 - self.centre) * self.dt

    def __call__(self, lag: Array) -> Array:
        """Return the wavelet at the lags, 0 beyond its samples."""
        t = self._lags
        inside = (lag >= t[0]) & (lag <= t[-1])
        out: Array = np.where(inside, self._spline(np.clip(lag, t[0], t[-1])), 0.0)
        return out

    @cached_property
    def _spline(self) -> "BSpline":
        """The interpolating cubic, built once.

        ``cached_property`` writes the instance's ``__dict__``, so it works although the
        class is frozen (it would not with ``slots=True``).
        """
        from scipy.interpolate import make_interp_spline

        return make_interp_spline(self._lags, self.samples, k=3)

    def sampled(self, dt: float) -> tuple[Array, int]:
        """Return the wavelet resampled at ``dt`` and its centre."""
        n_left = int(np.floor(self.centre * self.dt / dt))
        n_right = int(np.floor((self.samples.size - 1 - self.centre) * self.dt / dt))
        lags: Array = np.arange(-n_left, n_right + 1, dtype=np.float64) * dt
        return self(lags), n_left

    def nyquist_interval(self) -> float:
        """Return the largest sampling interval that resolves the wavelet (s).

        ``1/(2 f_max)``, ``f_max`` the highest frequency whose amplitude is at least
        0.1 % of the spectrum's peak (zero-padded eightfold).
        """
        spectrum = np.abs(np.fft.rfft(self.samples, n=8 * self.samples.size))
        freqs = np.fft.rfftfreq(8 * self.samples.size, self.dt)
        f_max = float(freqs[np.flatnonzero(spectrum >= 1e-3 * spectrum.max())[-1]])
        return 1.0 / (2.0 * f_max) if f_max > 0 else float("inf")


@dataclass(frozen=True)
class ConstantOverburden:
    """A uniform medium above (and below) the reservoir: the reference's shale."""

    vp: float
    vs: float
    rho: float

    def two_way_time(self, depth: Array) -> Array:
        """Return the two-way time (s) to ``depth`` (m)."""
        out: Array = 2.0 * depth / self.vp
        return out

    def above(self, depth: Array) -> tuple[Array, Array, Array]:
        """Return the elastic properties just above ``depth``."""
        one = np.ones_like(depth)
        return self.vp * one, self.vs * one, self.rho * one

    def below(self, depth: Array) -> tuple[Array, Array, Array]:
        """Return the elastic properties just below ``depth`` (the same medium)."""
        return self.above(depth)


@dataclass(frozen=True)
class LayeredOverburden:
    """Horizontal layers from the surface down (``thickness`` m; the last extends down)."""

    thickness: tuple[float, ...]
    vp: tuple[float, ...]
    vs: tuple[float, ...]
    rho: tuple[float, ...]

    def __post_init__(self) -> None:
        """Refuse inconsistent layer tables."""
        n = len(self.vp)
        if n == 0 or not (len(self.vs) == len(self.rho) == n and len(self.thickness) == n - 1):
            raise ValueError("give n velocities and densities and n - 1 thicknesses (the last "
                             "layer extends down)")  # fmt: skip
        if min(self.vp) <= 0 or min(self.vs) <= 0 or min(self.rho) <= 0:
            raise ValueError("layer velocities and densities must be positive")
        if self.thickness and min(self.thickness) <= 0:
            raise ValueError("layer thicknesses must be positive")

    def _layer(self, depth: Array, side: Literal["left", "right"] = "left") -> NDArray[np.intp]:
        bottoms = np.cumsum(self.thickness)
        out: NDArray[np.intp] = np.searchsorted(bottoms, depth, side=side)
        return out

    def two_way_time(self, depth: Array) -> Array:
        """Return the two-way time (s) to ``depth`` (m)."""
        tops = np.concatenate(([0.0], np.cumsum(self.thickness)))
        vp = np.asarray(self.vp)
        times = np.concatenate(([0.0], np.cumsum(2.0 * np.asarray(self.thickness) / vp[:-1])))
        k = self._layer(depth)
        out: Array = times[k] + 2.0 * (depth - tops[k]) / vp[k]
        return out

    def above(self, depth: Array) -> tuple[Array, Array, Array]:
        """Return the elastic properties just above ``depth`` (a boundary: the upper layer)."""
        k = self._layer(depth)
        return np.asarray(self.vp)[k], np.asarray(self.vs)[k], np.asarray(self.rho)[k]

    def below(self, depth: Array) -> tuple[Array, Array, Array]:
        """Return the elastic properties just below ``depth`` (a boundary: the lower layer)."""
        k = self._layer(depth, side="right")
        return np.asarray(self.vp)[k], np.asarray(self.vs)[k], np.asarray(self.rho)[k]


type Overburden = ConstantOverburden | LayeredOverburden


@dataclass(frozen=True)
class Columns:
    """The traces' columns, top layer first; each ``(n_trace, n_z)`` but ``top_depth``."""

    top_depth: Array
    thickness: Array
    vp: Array
    vs: Array
    rho: Array
    active: NDArray[np.bool_]

    @classmethod
    def of(
        cls,
        top_depth: ArrayLike,
        thickness: ArrayLike,
        vp: ArrayLike,
        vs: ArrayLike,
        rho: ArrayLike,
        active: ArrayLike | None = None,
    ) -> "Columns":
        """Validate and return the columns."""
        f = [np.asarray(x, dtype=np.float64) for x in (thickness, vp, vs, rho)]
        shape = f[0].shape
        if len(shape) != 2 or any(a.shape != shape for a in f):
            raise ValueError(f"thickness, vp, vs and rho must share one (n_trace, n_z) shape, "
                             f"got {[a.shape for a in f]}")  # fmt: skip
        top = np.asarray(top_depth, dtype=np.float64)
        if top.shape != shape[:1]:
            raise ValueError(f"top_depth must be (n_trace,) = {shape[:1]}, got {top.shape}")
        act = np.ones(shape, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        if act.shape != shape:
            raise ValueError(f"active must be {shape}, got {act.shape}")
        if not np.all(np.isfinite(top)):
            raise ValueError("top_depth must be finite")
        if not np.all(np.isfinite(f[0]) & (f[0] >= 0)):
            raise ValueError("thickness must be finite and non-negative")
        for name, a in zip(("vp", "vs", "rho"), f[1:], strict=True):
            bad = act & ~(np.isfinite(a) & (a > 0))
            if bad.any():
                trace, layer = (int(i) for i in np.argwhere(bad)[0])
                raise GeophysicsModelError(
                    f"{name} of the active cell at (trace {trace}, layer {layer}) is "
                    f"{a[trace, layer]!r}; it must be finite and positive"
                )
        return cls(top, f[0], f[1], f[2], f[3], act)


def _fill_inactive(
    c: Columns, overburden: Overburden, inactive: Literal["overburden", "above"]
) -> tuple[Array, Array, Array]:
    """Return vp, vs, rho with every inactive cell replaced (``(n_trace, n_z)``)."""
    ob = overburden.above(c.top_depth)
    out = []
    for values, fallback in zip((c.vp, c.vs, c.rho), ob, strict=True):
        filled = np.where(c.active, values, fallback[:, np.newaxis])
        if inactive == "above":
            # Index of the last active, thick cell at or above each cell (-1: none): a
            # zero-thickness cell is no layer, so it lends no properties.
            n_z = values.shape[1]
            last = np.where(c.active & (c.thickness > 0), np.arange(n_z), -1)
            last = np.maximum.accumulate(last, axis=1)
            rows = np.arange(values.shape[0])[:, np.newaxis]
            picked = values[rows, np.maximum(last, 0)]
            filled = np.where(last >= 0, picked, fallback[:, np.newaxis])
        out.append(filled)
    return out[0], out[1], out[2]


@dataclass(frozen=True)
class SeismicModel:
    """Angle-stack synthesis: columns in, traces ``(n_trace, n_angle, n_time)`` out.

    Parameters
    ----------
    reflectivity
        A model of the ``reflectivity`` registry.
    wavelet
        :class:`RickerWavelet` or :class:`SampledWavelet`.
    overburden
        Above the reservoir; also below it unless ``underburden`` is given.
    angles_deg
        Incidence angles.
    time
        The output times.
    sampling
        ``"interfaces"`` (default) or ``"nearest"`` (the reference's; parity only).
    inactive
        Properties of inactive cells: ``"overburden"`` (default) or ``"above"`` (the
        nearest active cell with thickness above; the overburden if none).
    underburden
        Below the reservoir; by default the overburden's medium at the reservoir's base
        depth (the layer there, for a layered overburden).
    fine_dt
        ``"nearest"``'s internal grid (s), the reference's 0.5 ms.
    """

    reflectivity: PPReflectivity
    wavelet: RickerWavelet | SampledWavelet
    overburden: Overburden
    angles_deg: tuple[float, ...]
    time: TimeAxis
    sampling: Literal["interfaces", "nearest"] = "interfaces"
    inactive: Literal["overburden", "above"] = "overburden"
    underburden: ConstantOverburden | None = None
    fine_dt: float = 0.0005

    def __post_init__(self) -> None:
        """Refuse settings that hold for no member (configuration errors)."""
        if self.sampling not in ("interfaces", "nearest"):
            raise ValueError(f"sampling must be 'interfaces' or 'nearest', got {self.sampling!r}")
        if self.inactive not in ("overburden", "above"):
            raise ValueError(f"inactive must be 'overburden' or 'above', got {self.inactive!r}")
        if not self.angles_deg or not all(0 <= a < 90 for a in self.angles_deg):
            raise ValueError(f"angles must be in [0, 90) degrees, got {self.angles_deg}")
        if self.fine_dt <= 0:
            raise ValueError("fine_dt must be positive")

    def base_time(self, columns: Columns) -> Array:
        """Return each trace's two-way time to its column's base ``(n_trace,)`` (s).

        At vertical incidence, through the overburden and the cells (inactive ones filled
        as the traces fill them): the times the synthesis places the base interface at.
        """
        vp, _, _ = _fill_inactive(columns, self.overburden, self.inactive)
        top_time = self.overburden.two_way_time(columns.top_depth)
        with np.errstate(divide="ignore", invalid="ignore"):
            layers = np.where(columns.thickness > 0, 2.0 * columns.thickness / vp, 0.0)
        out: Array = top_time + layers.sum(axis=1)
        return out

    def traces(self, columns: Columns) -> Array:
        """Return the traces ``(n_trace, n_angle, n_time)``.

        Raises
        ------
        EvanescentWaveError
            If an interface's transmitted wave is evanescent at an angle (its ``index``
            is ``(trace, interface)``, interface 0 the overburden's base).
        """
        vp, vs, rho = _fill_inactive(columns, self.overburden, self.inactive)
        under = self.underburden
        ob = self.overburden.above(columns.top_depth)
        base = columns.top_depth + columns.thickness.sum(axis=1)
        ub = (self.overburden if under is None else under).below(base)
        # Layers: the overburden, the cells, the underburden: (n_trace, n_z + 2).
        lvp = np.column_stack((ob[0], vp, ub[0]))
        lvs = np.column_stack((ob[1], vs, ub[1]))
        lrho = np.column_stack((ob[2], rho, ub[2]))
        top_time = self.overburden.two_way_time(columns.top_depth)
        if self.sampling == "nearest":
            return self._nearest(top_time, columns.thickness, lvp, lvs, lrho)
        return self._interfaces(top_time, columns.thickness, lvp, lvs, lrho)

    def _interfaces(
        self, top_time: Array, thickness: Array, lvp: Array, lvs: Array, lrho: Array
    ) -> Array:
        n_trace, n_layer = lvp.shape
        # τ_k, the top of layer k + 1 (interface k under layer k), k = 0 .. n_z.
        tau = top_time[:, np.newaxis] + np.concatenate(
            (np.zeros((n_trace, 1)), np.cumsum(2.0 * thickness / lvp[:, 1:-1], axis=1)), axis=1
        )
        # Zero-thickness cells form no interface: interface k counts only if the layer
        # below it is thick, and its upper medium is the last thick layer above.
        thick = np.column_stack((np.ones(n_trace, bool), thickness > 0, np.ones(n_trace, bool)))
        lower_ok = thick[:, 1:]
        index = np.where(thick, np.arange(n_layer), -1)
        upper = np.maximum.accumulate(index, axis=1)[:, :-1]
        rows = np.arange(n_trace)[:, np.newaxis]
        uvp, uvs, urho = lvp[rows, upper], lvs[rows, upper], lrho[rows, upper]
        # A masked interface's lower medium is set to its upper one, so that it is never
        # evaluated as a real contrast (it could be evanescent and fail the member).
        dvp = np.where(lower_ok, lvp[:, 1:], uvp)
        dvs = np.where(lower_ok, lvs[:, 1:], uvs)
        drho = np.where(lower_ok, lrho[:, 1:], urho)
        r = self.reflectivity.pp(uvp, uvs, urho, dvp, dvs, drho, self.angles_deg)
        r = np.where(lower_ok[..., np.newaxis], r, 0.0)  # (n_trace, n_int, n_angle)
        times = self.time.times
        n_int = tau.shape[1]
        out = np.empty((n_trace, len(self.angles_deg), times.size))
        chunk = max(1, _CHUNK_ELEMENTS // max(1, n_int * times.size))
        for start in range(0, n_trace, chunk):
            s = slice(start, start + chunk)
            w = self.wavelet(times[np.newaxis, np.newaxis, :] - tau[s, :, np.newaxis])
            out[s] = np.einsum("nka,nkt->nat", r[s], w)
        return out

    def _nearest(
        self, top_time: Array, thickness: Array, lvp: Array, lvs: Array, lrho: Array
    ) -> Array:
        n_trace = lvp.shape[0]
        cum_res = top_time[:, np.newaxis] + np.cumsum(2.0 * thickness / lvp[:, 1:-1], axis=1)
        # The reference puts the underburden's top at top + max over every column's
        # cumulative time (a scalar over the grid; flow_rock.py L1830). That counts the
        # top's time twice, so it lies below every base: the samples past a column's last
        # cell take the underburden whatever the maximum's columns (a subset's or the grid's).
        under = top_time + np.max(cum_res)
        cum = np.column_stack((top_time, cum_res, under))  # (n_trace, n_z + 2)
        fine = np.arange(self.time.t_min, self.time.t_max, self.fine_dt)
        # searchsorted(cum[n], t, side="left") for every trace: the count of cum < t.
        idx = np.zeros((n_trace, fine.size), dtype=np.intp)
        for k in range(cum.shape[1]):
            idx += cum[:, k : k + 1] < fine[np.newaxis, :]
        idx = np.minimum(idx, cum.shape[1] - 1)
        rows = np.arange(n_trace)[:, np.newaxis]
        svp, svs, srho = lvp[rows, idx], lvs[rows, idx], lrho[rows, idx]
        r = self.reflectivity.pp(
            svp[:, :-1],
            svs[:, :-1],
            srho[:, :-1],
            svp[:, 1:],
            svs[:, 1:],
            srho[:, 1:],
            self.angles_deg,
        )  # fmt: skip  (n_trace, n_fine - 1, n_angle)
        w, centre = self.wavelet.sampled(self.fine_dt)
        conv = _convolve_same(np.moveaxis(r, 2, 1), w, centre)  # (n_trace, n_angle, n_fine-1)
        # scipy's interp1d(kind="nearest", fill_value="extrapolate") from the
        # reflectivities' times (the upper samples') to the output times: ties go down.
        t_r = fine[:-1]
        bounds = (t_r[1:] + t_r[:-1]) / 2.0
        pick = np.searchsorted(bounds, self.time.times, side="left")
        out: Array = conv[:, :, pick]
        return out


def _convolve_same(x: Array, w: Array, centre: int) -> Array:
    """Return ``x * w`` along the last axis, aligned at ``centre``, cut to ``x``'s length.

    A linear (zero-padded) convolution: no wrap-around.
    """
    from scipy.signal import fftconvolve

    full = fftconvolve(x, w.reshape((1,) * (x.ndim - 1) + (-1,)), mode="full", axes=-1)
    out: Array = full[..., centre : centre + x.shape[-1]]
    return out
