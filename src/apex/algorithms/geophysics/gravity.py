"""Time-lapse gravity at stations from mass changes in the reservoir's cells.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_grav.calc_grav``, L2259-2310:
             point masses at the cells' centres)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
References : Nagy, D., Papp, G. & Benedek, J. (2000). The gravitational potential and its
             derivatives for the prism. Journal of Geodesy, 74(7-8), 552-560 (the exact
             right rectangular prism, ``prism_gravity``).
Behaviour  : the vertical attraction of each cell's mass change at each station, summed:
             ``point_mass_gravity`` (the reference's), or ``prism_gravity``, each cell a
             right rectangular prism of uniform density: exact up to the rounding of its
             cancelling corner terms, and its point mass beyond 50 diagonals, where those
             terms lose precision and the point mass is within 1e-4 of G V/r².
Differences: - The gravitational constant is CODATA 2018's, 6.67430e-11 m³ kg⁻¹ s⁻²; the
               reference takes 6.67e-11 (a 0.06 % difference), an option here.
             - SI throughout (m/s²); the reference returns µGal (1e-8 m/s²).

Coordinates are (x, y, z) in m with z positive down, so a mass below a station pulls it
down: ``g_z > 0``. Shapes: ``stations`` ``(n_station, 3)``; ``centres`` ``(n_cell, 3)``;
cell values ``(n_cell,)``; the result ``(n_station,)``.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Array = NDArray[np.float64]

#: CODATA 2018 (m³ kg⁻¹ s⁻²).
GRAVITATIONAL_CONSTANT = 6.67430e-11
#: The (station, cell) pairs processed at once: bounds the work arrays (about ten of this
#: many float64 values for a prism's corners, 8 MB each) whatever the grid's size.
_ELEMENTS = 1_000_000
#: Beyond this many times its diagonal a prism is taken as its point mass (relative error
#: below 1e-4, where the prism's own terms lose precision).
_FAR = 50.0


def point_mass_gravity(
    stations: ArrayLike,
    centres: ArrayLike,
    mass: ArrayLike,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> Array:
    """Return ``g_z`` (m/s²) at the stations of point masses at the cell centres.

    ``g_z = G Σ_c m_c (z_c - z_s) / r³``, ``r`` the station-to-centre distance.

    Raises
    ------
    ValueError
        If a station coincides with a centre carrying mass (the field is singular there),
        or the shapes do not match.
    """
    s, c, m = _inputs(stations, centres, mass)
    out = np.zeros(s.shape[0])
    for rows, cols in _blocks(s.shape[0], c.shape[0]):
        d = c[np.newaxis, cols, :] - s[rows, np.newaxis, :]  # (k, n_block, 3)
        r2 = np.einsum("knd,knd->kn", d, d)
        if np.any((r2 == 0) & (m[cols] != 0)):
            raise ValueError("a station coincides with a cell centre: its point mass is singular")
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.where(r2 > 0, d[..., 2] / (r2 * np.sqrt(r2)), 0.0)
        out[rows] += kernel @ m[cols]
    result: Array = gravitational_constant * out
    return result


def _blocks(n_station: int, n_cell: int) -> list[tuple[slice, slice]]:
    """Return (stations, cells) blocks of at most about :data:`_ELEMENTS` pairs.

    Cells are split only when one station's row exceeds the budget, so a grid of up to
    that many cells sums each station's cells in one product, as the reference does.
    """
    cells = min(n_cell, _ELEMENTS) or 1
    rows = max(1, _ELEMENTS // cells)
    return [
        (slice(i, i + rows), slice(j, j + cells))
        for i in range(0, n_station, rows)
        for j in range(0, n_cell, cells)
    ]


def prism_gravity(
    stations: ArrayLike,
    bounds: ArrayLike,
    density: ArrayLike,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT,
) -> Array:
    """Return ``g_z`` (m/s²) at the stations of uniform right rectangular prisms.

    ``bounds`` is ``(n_cell, 6)``: ``x1, x2, y1, y2, z1, z2`` (m, ``x1 < x2`` and so on),
    ``density`` the cells' density change (kg/m³). Nagy et al. (2000), eq. 7, in the form
    ``g_z = G rho Σ (-1)^(i+j+k) [x asinh(y/√(x²+z²)) + y asinh(x/√(y²+z²))
    - z atan(x y / (z r))]`` over the prism's corners relative to the station, ``i``, ``j``,
    ``k`` 0 at the lower bound and 1 at the upper. The asinh avoids the logarithms'
    cancellation (the ``ln √(x²+z²)`` it drops cancels in the sum).

    Raises
    ------
    ValueError
        If a prism is degenerate, or the shapes do not match.
    """
    s = np.atleast_2d(np.asarray(stations, dtype=np.float64))
    b = np.atleast_2d(np.asarray(bounds, dtype=np.float64))
    rho = np.asarray(density, dtype=np.float64).ravel()
    if s.shape[1] != 3 or b.shape[1] != 6 or rho.shape != (b.shape[0],):
        raise ValueError("stations must be (n, 3), bounds (n_cell, 6), density (n_cell,)")
    if np.any(b[:, 1] <= b[:, 0]) or np.any(b[:, 3] <= b[:, 2]) or np.any(b[:, 5] <= b[:, 4]):
        raise ValueError("every prism needs x1 < x2, y1 < y2 and z1 < z2")
    centre = np.column_stack(
        ((b[:, 0] + b[:, 1]) / 2, (b[:, 2] + b[:, 3]) / 2, (b[:, 4] + b[:, 5]) / 2)
    )
    extent = b[:, 1::2] - b[:, 0::2]
    size = np.linalg.norm(extent, axis=1)[np.newaxis, :]
    volume = np.prod(extent, axis=1)[np.newaxis, :]
    out = np.zeros(s.shape[0])
    for rows, cols in _blocks(s.shape[0], b.shape[0]):
        st = s[rows, np.newaxis, :]  # (k, 1, 3)
        bb = b[cols]
        total = np.zeros((st.shape[0], bb.shape[0]))
        for i, xi in enumerate((bb[:, 0], bb[:, 1])):
            for j, yj in enumerate((bb[:, 2], bb[:, 3])):
                for k, zk in enumerate((bb[:, 4], bb[:, 5])):
                    x = xi[np.newaxis, :] - st[..., 0]
                    y = yj[np.newaxis, :] - st[..., 1]
                    z = zk[np.newaxis, :] - st[..., 2]
                    sign = -1.0 if (i + j + k) % 2 else 1.0
                    total += sign * _prism_corner(x, y, z)
        # Far from a prism its eight corner terms cancel to rounding (relative 1e-4 at
        # 4000 sizes), while the point mass is exact to (size/distance)² there.
        d = centre[np.newaxis, cols, :] - st
        r2 = np.einsum("knd,knd->kn", d, d)
        far = r2 > (_FAR * size[:, cols]) ** 2
        point = np.where(far, volume[:, cols] * d[..., 2] / (np.where(far, r2, 1.0) ** 1.5), 0.0)
        out[rows] += np.where(far, point, total) @ rho[cols]
    result: Array = gravitational_constant * out
    return result


def _prism_corner(x: Array, y: Array, z: Array) -> Array:
    """Nagy et al.'s primitive of ``g_z`` at a corner, without its singular logarithms."""
    r = np.sqrt(x * x + y * y + z * z)
    xz = np.hypot(x, z)
    yz = np.hypot(y, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(xz > 0, x * np.arcsinh(y / np.where(xz > 0, xz, 1.0)), 0.0)
        b = np.where(yz > 0, y * np.arcsinh(x / np.where(yz > 0, yz, 1.0)), 0.0)
        c = np.where(
            (z != 0) & (r > 0), z * np.arctan(x * y / np.where(z * r != 0, z * r, 1.0)), 0.0
        )
    out: Array = a + b - c
    return out


def _inputs(
    stations: ArrayLike, centres: ArrayLike, values: ArrayLike
) -> tuple[Array, Array, Array]:
    s = np.atleast_2d(np.asarray(stations, dtype=np.float64))
    c = np.atleast_2d(np.asarray(centres, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64).ravel()
    if s.shape[1] != 3 or c.shape[1] != 3 or v.shape != (c.shape[0],):
        raise ValueError("stations and centres must be (n, 3), the cell values (n_cell,)")
    return s, c, v
