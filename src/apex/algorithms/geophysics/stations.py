"""Survey stations over a reservoir, for gravity and displacement.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``mixIn_multi_data
             .measurement_locations``, L92-146)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
Behaviour  : a regular grid over the cells' horizontal extent widened by ``padding``,
             ``ceil(span / spacing)`` stations along each axis (so the spacing is about,
             not exactly, ``spacing``), x fastest; optionally finer (``spacing / 2``,
             rounded) within ``radius`` of a well, the coarse stations there dropped and
             the whole set then sorted by (x, y), as the reference's ``np.unique`` does.
             The stations' depth is a constant (water depth, positive down) or the nearest
             value of a seabed table.

Shapes: the cells' ``x``, ``y`` ``(n_cell,)``; the stations ``(n_station, 3)``, (x, y, z)
in m, z positive down.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Array = NDArray[np.float64]


def regular_stations(
    x: ArrayLike,
    y: ArrayLike,
    *,
    spacing: float,
    padding: float,
    depth: float | Array = 0.0,
    refine: tuple[float, float, float] | None = None,
    seabed: tuple[Array, Array, Array] | None = None,
) -> Array:
    """Return the reference's survey stations over the cells at ``x``, ``y``.

    Parameters
    ----------
    x, y
        ``(n_cell,)``: the cells' centres (m).
    spacing, padding
        The stations' approximate spacing and the margin around the cells (m).
    depth
        The stations' depth (m, positive down), when ``seabed`` is not given.
    refine
        ``(x, y, radius)``: stations at half the spacing within ``radius`` of the point.
    seabed
        ``(x, y, depth)`` of a seabed table: each station takes its nearest depth (absolute).

    Raises
    ------
    ValueError
        If the spacing is not positive, the padding negative, or ``depth`` neither one value
        nor one per station.
    """
    if not spacing > 0 or padding < 0:
        raise ValueError(
            f"spacing must be positive and padding not negative ({spacing}, {padding})"
        )
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x_min, x_max = float(xs.min()) - padding, float(xs.max()) + padding
    y_min, y_max = float(ys.min()) - padding, float(ys.max()) + padding
    nx = int(np.ceil((x_max - x_min) / spacing))
    ny = int(np.ceil((y_max - y_min) / spacing))
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    points = np.column_stack((gx.ravel(), gy.ravel()))
    if refine is not None:
        points = _refined(points, (x_min, x_max, y_min, y_max), spacing, refine)
    if seabed is not None:
        from scipy.interpolate import griddata  # the reference's nearest seabed depth

        z = griddata(
            (seabed[0], seabed[1]),
            np.abs(seabed[2]),
            (points[:, 0], points[:, 1]),
            method="nearest",
        )
    else:
        values = np.asarray(depth, dtype=np.float64)
        if values.ndim and values.shape != (points.shape[0],):
            raise ValueError(
                f"depth has {values.size} values for {points.shape[0]} stations; give one, or "
                "one per station"
            )
        z = np.broadcast_to(values, (points.shape[0],))
    out: Array = np.column_stack((points, z))
    return out


def _refined(
    coarse: Array, box: Sequence[float], spacing: float, refine: tuple[float, float, float]
) -> Array:
    """Return the reference's finer stations within a circle, with the coarse, sorted by (x, y)."""
    x_min, x_max, y_min, y_max = box
    xc, yc, radius = refine
    fine_spacing = round(spacing / 2)
    fx_min, fx_max = max(x_min, xc - radius), min(x_max, xc + radius)
    fy_min, fy_max = max(y_min, yc - radius), min(y_max, yc + radius)
    if not (fx_max > fx_min and fy_max > fy_min):
        return coarse  # the reference keeps the coarse grid, unsorted, then
    nfx = int(np.ceil((fx_max - fx_min) / fine_spacing)) + 1
    nfy = int(np.ceil((fy_max - fy_min) / fine_spacing)) + 1
    fx, fy = np.meshgrid(np.linspace(fx_min, fx_max, nfx), np.linspace(fy_min, fy_max, nfy))
    fine = np.column_stack((fx.ravel(), fy.ravel()))
    inside = (fine[:, 0] - xc) ** 2 + (fine[:, 1] - yc) ** 2 <= radius**2
    outside = (coarse[:, 0] - xc) ** 2 + (coarse[:, 1] - yc) ** 2 > radius**2
    both = np.vstack((coarse[outside], fine[inside]))
    # The reference rounds to collapse near-duplicates, then np.unique sorts by (x, y).
    tolerance = min(spacing, fine_spacing) * 1e-3
    digits = max(0, int(-np.floor(np.log10(tolerance))))
    out: Array = np.unique(np.round(both, digits), axis=0)
    return out
