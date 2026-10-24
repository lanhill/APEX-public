"""AVO attributes: a value per trace from its samples in a time window.

New in APEX (the reference's data are the whole traces, ``trace`` here). A map
attribute reduces each trace of one angle and vintage to one value over the samples its
window selects; maps are the default form of AVO data, about 1/n_time of the traces'
size.

Shapes: traces ``(n_trace, n_angle, n_time)``; a window ``(n_time,)`` or
``(n_trace, n_time)`` of booleans; a map ``(n_trace, n_angle)``.
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics.registries import avo_attributes

type Array = NDArray[np.float64]
type Mask = NDArray[np.bool_]


class AVOAttribute(Protocol):
    """An attribute of the ``avo_attributes`` registry.

    Optional: ``uses_times`` (``False`` when absent); an attribute with ``uses_times =
    True`` (the time shift) is taken from the columns' two-way times, not the traces.
    """

    #: ``True`` for map attributes (a value per trace), ``False`` for ``trace``.
    is_map: bool

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)`` (or, for ``trace``, the windowed traces)."""
        ...


def time_window(
    times: ArrayLike, start: ArrayLike, end: ArrayLike, n_trace: int | None = None
) -> Mask:
    """Return the samples with ``start <= t <= end``.

    ``start`` and ``end`` are scalars (one window, ``(n_time,)``) or ``(n_trace,)``
    (e.g. the reservoir's top and base times plus a margin: ``(n_trace, n_time)``).
    The window must be the same for every member (from the truth, the prior mean or
    the observations): one computed per member lets samples enter and leave it, and
    ``mean``, ``rms`` and ``sum_negative`` then jump between members. (``min`` and
    ``max`` are kinked where the extreme sample changes, whatever the window.)
    Raises ``ValueError`` if a trace's window holds no sample.
    """
    t = np.asarray(times, dtype=np.float64)
    a, b = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if a.ndim == 0 and b.ndim == 0:
        mask: Mask = (t >= a) & (t <= b)
        if not mask.any():
            raise ValueError(f"no output time lies in the window [{a:g}, {b:g}] s")
        return mask
    a, b = np.broadcast_arrays(np.atleast_1d(a), np.atleast_1d(b))
    if n_trace is not None and a.shape != (n_trace,):
        raise ValueError(f"window bounds must be scalars or (n_trace,) = ({n_trace},)")
    mask = (t[np.newaxis, :] >= a[:, np.newaxis]) & (t[np.newaxis, :] <= b[:, np.newaxis])
    empty = ~mask.any(axis=1)
    if empty.any():
        i = int(np.flatnonzero(empty)[0])
        raise ValueError(f"trace {i}'s window [{a[i]:g}, {b[i]:g}] s holds no output time")
    return mask


def _masked(traces: Array, window: Mask) -> tuple[Array, Mask]:
    if traces.ndim != 3:
        raise ValueError(f"traces must be (n_trace, n_angle, n_time), got {traces.shape}")
    w = np.broadcast_to(window if window.ndim == 2 else window[np.newaxis, :],
                        (traces.shape[0], traces.shape[2]))  # fmt: skip
    if w.shape[1] != traces.shape[2]:
        raise ValueError(f"the window has {w.shape[1]} samples, the traces {traces.shape[2]}")
    return traces, w[:, np.newaxis, :]


@avo_attributes.register("rms")
class RMS:
    """Root mean square amplitude ``sqrt(mean(s²))`` over the window (the default)."""

    is_map = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)``."""
        s, w = _masked(traces, window)
        out: Array = np.sqrt(np.sum(np.where(w, s * s, 0.0), axis=-1) / np.sum(w, axis=-1))
        return out


@avo_attributes.register("mean")
class Mean:
    """Mean amplitude over the window."""

    is_map = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)``."""
        s, w = _masked(traces, window)
        out: Array = np.sum(np.where(w, s, 0.0), axis=-1) / np.sum(w, axis=-1)
        return out


@avo_attributes.register("min")
class Minimum:
    """Most negative amplitude over the window (a trough)."""

    is_map = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)``."""
        s, w = _masked(traces, window)
        out: Array = np.min(np.where(w, s, np.inf), axis=-1)
        return out


@avo_attributes.register("max")
class Maximum:
    """Most positive amplitude over the window (a peak)."""

    is_map = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)``."""
        s, w = _masked(traces, window)
        out: Array = np.max(np.where(w, s, -np.inf), axis=-1)
        return out


@avo_attributes.register("sum_negative")
class SumNegative:
    """Sum of the negative amplitudes over the window (a soft-sand or gas indicator)."""

    is_map = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the map ``(n_trace, n_angle)``."""
        s, w = _masked(traces, window)
        out: Array = np.sum(np.where(w, np.minimum(s, 0.0), 0.0), axis=-1)
        return out


@avo_attributes.register("trace")
class Trace:
    """The whole trace: the samples in the window, zero outside it.

    With a window per trace the samples kept differ between traces, so the traces stay
    ``(n_trace, n_angle, n_time)`` with zeros outside; the simulator keeps the samples
    of the fixed mask.
    """

    is_map = False

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Return the windowed traces ``(n_trace, n_angle, n_time)``."""
        s, w = _masked(traces, window)
        out: Array = np.where(w, s, 0.0)
        return out


@avo_attributes.register("time_shift")
class TimeShift:
    """The two-way time to the reservoir's base, for the time shift.

    Not a reduction of the traces: the simulator takes each trace's two-way time to the
    base of its column (:meth:`~apex.algorithms.geophysics.seismic.SeismicModel.
    base_time`), and the data are its change from the baseline to the vintage (s, positive
    when the vintage is slower, as under a CO₂ plume): for a layer of thickness h whose
    velocity changes from v1 to v2, 2h (1/v2 - 1/v1). Vertical incidence, the same for
    every angle; fixed cell thicknesses (no dilation) and overburden (no geomechanical
    shifts): an observed shift's tuning and cross-correlation errors are model error,
    which its variance must cover. Under ``inactive = "above"`` an inactive cell takes the
    velocity of the active cell above it at each date, so its thickness shifts too, as in
    the synthesis; under ``"overburden"`` it cancels.
    """

    is_map = True
    uses_times = True

    def __call__(self, traces: Array, window: Mask) -> Array:
        """Refuse: the time shift is taken from the columns, not the traces.

        Raises
        ------
        TypeError
            Always.
        """
        raise TypeError("time_shift is computed from the columns' times, not the traces")
