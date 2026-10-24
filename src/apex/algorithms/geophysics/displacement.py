"""Vertical displacement at stations from volume changes of the reservoir's cells.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_seafloor_disp``:
             ``_build_disp_kernel``, ``uHH_integrand``, ``compute_deformation_transfer``,
             L2514-2876, ``geertsma_mode = "integral"``)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
References : Geertsma, J. (1973). Land subsidence above compacting oil and gas reservoirs.
             Journal of Petroleum Technology, 25(6), 734-744 (the nucleus of strain in a
             homogeneous elastic half-space).
Behaviour  : each cell is a nucleus of strain at its centre (depth c) whose volume change
             ΔV displaces a station at depth z and horizontal distance rho vertically by

             u_z = ΔV / (4π) [ (c - z)/R₁³ + (3 - 4nu)(c + z)/R₂³
                               + 2z (2(c + z)² - rho²)/R₂⁵ ],

             R₁² = rho² + (c - z)², R₂² = rho² + (c + z)²: the reference's Hankel integral
             ``∫ λ (ε e^{λε(z-c)} + (3 - 4nu + 2zλ) e^{-λ(z+c)}) J₀(λrho) dλ · ΔV/(4π)``
             evaluated in closed form (∫ λ e^{-aλ} J₀ = a/(a²+rho²)^{3/2}, ∫ λ² e^{-aλ} J₀ =
             (2a²-rho²)/(a²+rho²)^{5/2}). At the free surface (z = 0) it is Geertsma's
             ``(1 - nu)/π ΔV c/R³``.
Differences: - Exact, where the reference integrates numerically (``quad``) up to
               λ_max = 15/max rho (km⁻¹, over the stations a cell reaches): with stations far
               from a cell that limit cuts the integral short (pet_parity.md G12).
             - Untruncated by default; the reference zeroes a station farther than 3 times
               the cell's depth below it, ``truncation_factor = 3``, which loses 1/√10 ≈
               32 % of a wide reservoir's subsidence.
             - Each station is on the free surface by default (``free_surface =
               "stations"``: the seabed, as water carries no shear); the reference puts the
               free surface at sea level and seabed stations inside the rock at the water
               depth (``"datum"``; pet_parity.md G13), which is 5-50 % off offshore.
             - SI (m); the reference returns cm.

Coordinates are (x, y, z) in m, z positive down; ΔV > 0 is a volume decrease (compaction)
and ``u_z > 0`` a downward displacement (subsidence), as the reference's. Shapes:
``stations`` ``(n_station, 3)``; ``centres`` ``(n_cell, 3)``; the matrix
``(n_station, n_cell)`` (m of displacement per m³ of volume change).
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics.registries import displacement_kernels

type Array = NDArray[np.float64]


class DisplacementKernel(Protocol):
    """A model of the vertical displacement a cell's volume change causes at a station."""

    def matrix(self, stations: ArrayLike, centres: ArrayLike) -> Array:
        """Return ``(n_station, n_cell)``: ``u_z`` (m) per m³ of each cell's volume decrease."""
        ...


#: The stations processed at once: bounds the (stations x cells) work arrays.
_CHUNK = 64


@displacement_kernels.register("geertsma")
@dataclass(frozen=True)
class Geertsma:
    """Geertsma's (1973) nucleus of strain in an elastic half-space.

    Parameters
    ----------
    poisson
        Poisson's ratio nu of the overburden and reservoir, in (-1, 0.5].
    truncation_factor
        Zero a station farther from a cell than this times the cell's depth below it, as
        the reference does with 3; ``None`` (the default) keeps every station.
    free_surface
        ``"stations"`` (the default): each station is on the half-space's free surface,
        the seabed offshore (water carries no shear) or the ground onshore, so a cell's
        depth counts from the station's (``c - z``; the free-surface form). ``"datum"``:
        the free surface is z = 0 and a station at depth z is inside the rock, the
        reference's (with seabed stations at the water depth: pet_parity.md G13).
    """

    poisson: float = 0.25
    truncation_factor: float | None = None
    free_surface: Literal["stations", "datum"] = "stations"

    def __post_init__(self) -> None:
        """Refuse a Poisson's ratio outside (-1, 0.5], a non-positive truncation, a mode."""
        if not -1.0 < self.poisson <= 0.5:
            raise ValueError(f"poisson must be in (-1, 0.5], got {self.poisson}")
        if self.truncation_factor is not None and not self.truncation_factor > 0:
            raise ValueError(f"truncation_factor must be positive, got {self.truncation_factor}")
        if self.free_surface not in ("stations", "datum"):
            raise ValueError(
                f"free_surface must be 'stations' or 'datum', got {self.free_surface!r}"
            )

    def matrix(self, stations: ArrayLike, centres: ArrayLike) -> Array:
        """Return ``(n_station, n_cell)``: ``u_z`` (m) per m³ of each cell's volume decrease.

        Raises
        ------
        ValueError
            If a station coincides with a cell centre (the nucleus is singular there); with
            ``free_surface = "stations"`` if a cell is not below a station; with
            ``"datum"`` if a station is above the free surface (z < 0); or the shapes do
            not match.
        """
        s = np.atleast_2d(np.asarray(stations, dtype=np.float64))
        c = np.atleast_2d(np.asarray(centres, dtype=np.float64))
        if s.shape[1] != 3 or c.shape[1] != 3:
            raise ValueError("stations and centres must be (n, 3)")
        if self.free_surface == "datum" and np.any(s[:, 2] < 0):
            raise ValueError("a station is above the free surface z = 0 (free_surface = 'datum')")
        out = np.empty((s.shape[0], c.shape[0]))
        for start in range(0, s.shape[0], _CHUNK):
            out[start : start + _CHUNK] = self._block(s[start : start + _CHUNK], c)
        return out

    def _block(self, s: Array, c: Array) -> Array:
        rho = np.hypot(
            s[:, np.newaxis, 0] - c[np.newaxis, :, 0], s[:, np.newaxis, 1] - c[np.newaxis, :, 1]
        )
        if self.free_surface == "stations":
            # Each station on the free surface: its own depth is the datum.
            depth = c[np.newaxis, :, 2] - s[:, np.newaxis, 2]
            z = np.zeros((s.shape[0], 1))
            if np.any(depth <= 0):
                raise ValueError(
                    "a cell is not below a station (free_surface = 'stations' puts each "
                    "station on the free surface)"
                )
        else:
            depth = np.broadcast_to(c[np.newaxis, :, 2], rho.shape)
            z = s[:, np.newaxis, 2]
            if np.any((rho == 0) & (z == depth)):
                raise ValueError("a station coincides with a cell centre: the nucleus is singular")
        r1 = np.sqrt(rho * rho + (depth - z) ** 2)
        a = depth + z
        r2 = np.sqrt(rho * rho + a * a)
        nu = self.poisson
        block: Array = (
            (depth - z) / r1**3
            + (3.0 - 4.0 * nu) * a / r2**3
            + 2.0 * z * (2.0 * a * a - rho * rho) / r2**5
        ) / (4.0 * np.pi)
        if self.truncation_factor is not None:
            block = np.where(rho > self.truncation_factor * np.abs(depth - z), 0.0, block)
        return block


#: Gauss-Legendre nodes for the basement's Hankel integral: at least this many, and enough
#: to resolve J0(λrho) with about 4 nodes per half-period over the integration range.
_MIN_NODES = 96
_MAX_NODES = 8192
#: The integrand decays at least as e^{-λ a}, a = 2k - c - z: integrate to λ = _DECAY / a.
_DECAY = 46.0
#: Cells times nodes per J₀ chunk: bounds the work arrays (about 30 MB of float64 each).
_CHUNK_ELEMENTS = 4_000_000


@displacement_kernels.register("van_opstal")
@dataclass(frozen=True)
class VanOpstal:
    """A nucleus of strain over a rigid basement (van Opstal 1974; Tempone et al. 2010).

    Geertsma's half-space solution (:class:`Geertsma`, Systems 1 + 2 of Tempone et al.)
    plus the basement's correction (System 3), which keeps the free surface traction-free
    and makes the displacement vanish at the basement, z = k:

    u₃ = -ΔV/(4π) ∫₀^∞ {B̂ [λz cosh λz - (1 - 2nu) sinh λz]
                        + Ĉ [2(1 - nu) cosh λz - λz sinh λz]} J₀(λrho) dλ,

    with B̂, Ĉ of Tempone et al. (2010), eqs. 28-32 (their L, M, Δ for ε = -1), evaluated
    in a form scaled by e^{-kλ} (algebraically the same) so that no cosh or sinh overflows,
    and integrated by Gauss-Legendre quadrature.

    Ported from SimulatorWrap
    -------------------------
    Module     : src/subsurface/multphaseflow/flow_rock.py (``makeL``, ``makeM``,
                 ``makeDelta``, ``makeB``, ``makeC``, ``compute_deformation_transfer``'s
                 ``System_3_vertical``, L2760-2876), which transcribes the paper's equations
                 but cannot run (pet_parity.md G16)
    Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
    References : van Opstal, G.H.C. (1974). The effect of base-rock rigidity on subsidence
                 due to reservoir compaction. Proc. 3rd Congr. ISRM, 2, 1102-1111.
                 Tempone, P., Fjær, E. & Landrø, M. (2010). Improved solution of
                 displacements due to a compacting reservoir over a rigid basement. Applied
                 Mathematical Modelling, 34(11), 3352-3362.

    Differences: the reference cannot run (G16); were it to, it would also zero stations
    beyond 3 depths (APEX keeps them), integrate with ``trapz`` on 500 points to
    λ_max = 30/max rho (too coarse; APEX's Gauss-Legendre nodes follow the decay and J₀'s
    oscillation, and refuse what they cannot resolve), put the free surface at sea level
    (G13; ``free_surface = "datum"`` here) and return cm (APEX: m).

    Parameters
    ----------
    basement_depth
        Depth of the rigid basement (m, positive down, from z = 0), below every cell.
    poisson, free_surface
        As :class:`Geertsma`: with ``"stations"`` each station is on the free surface and
        the basement's depth counts from the station too.
    """

    basement_depth: float
    poisson: float = 0.25
    free_surface: Literal["stations", "datum"] = "stations"

    def __post_init__(self) -> None:
        """Refuse a non-positive basement depth, and what :class:`Geertsma` refuses."""
        if not self.basement_depth > 0:
            raise ValueError(f"basement_depth must be positive, got {self.basement_depth}")
        Geertsma(self.poisson, None, self.free_surface)

    def matrix(self, stations: ArrayLike, centres: ArrayLike) -> Array:
        """Return ``(n_station, n_cell)``: ``u_z`` (m) per m³ of each cell's volume decrease.

        Raises
        ------
        ValueError
            If a cell is not above the basement, a station is below it, or as
            :meth:`Geertsma.matrix`.
        """
        s = np.atleast_2d(np.asarray(stations, dtype=np.float64))
        c = np.atleast_2d(np.asarray(centres, dtype=np.float64))
        if np.any(c[:, 2] >= self.basement_depth):
            raise ValueError(f"a cell is not above the rigid basement at {self.basement_depth:g} m")
        if np.any(s[:, 2] > self.basement_depth):
            raise ValueError(f"a station is below the rigid basement at {self.basement_depth:g} m")
        half_space = Geertsma(self.poisson, None, self.free_surface).matrix(s, c)
        out = np.empty_like(half_space)
        for n in range(s.shape[0]):  # one station at a time: (n_cell, n_node) work arrays
            out[n] = half_space[n] + self._correction(s[n], c)
        return out

    def _correction(self, station: Array, c: Array) -> Array:
        """Return System 3's ``u_z`` (m per m³) at one station from every cell.

        The integrand's factors other than J₀(λrho) depend only on a cell's depth, so they are
        formed once per depth. Cells are grouped by their slowest decay, 2k - c - z, within a
        factor of 2, so that one cell near the basement does not set every cell's range,
        and the J₀ factor is taken in chunks of cells, to bound memory.

        Raises
        ------
        ValueError
            If resolving J₀'s oscillation would take more than 8192 nodes (a station
            farther than about 140 times 2k - c - z from a cell).
        """
        from scipy.special import j0  # the Bessel function of the Hankel integral

        rho = np.hypot(c[:, 0] - station[0], c[:, 1] - station[1])
        if self.free_surface == "stations":
            depth = c[:, 2] - station[2]
            k = self.basement_depth - station[2]
            z = 0.0
        else:
            depth = c[:, 2]
            k = self.basement_depth
            z = float(station[2])
        decay = 2 * k - depth - z  # the slowest exponential, e^{-λ(2k - c - z)}
        groups = np.floor(np.log2(decay / decay.min())).astype(np.intp)
        out = np.empty(c.shape[0])
        for group in np.unique(groups):
            cells = np.flatnonzero(groups == group)
            upper = _DECAY / float(decay[cells].min())
            needed = _MIN_NODES + 4 * upper * float(rho[cells].max()) / np.pi
            if needed > _MAX_NODES:
                raise ValueError(
                    f"van_opstal: a station at {station.tolist()} is {rho[cells].max():.0f} m from "
                    f"a cell whose basement term decays over {decay[cells].min():.0f} m; its "
                    f"integral would need {needed:.0f} quadrature nodes (at most {_MAX_NODES})"
                )
            x, w = np.polynomial.legendre.leggauss(int(needed))
            lam = upper * (x + 1) / 2
            weights = upper * w / 2
            chunk = max(1, _CHUNK_ELEMENTS // lam.size)
            for start in range(0, cells.size, chunk):
                part = cells[start : start + chunk]
                # Once per depth within the chunk: few on a layered grid, at most the chunk.
                unique, which = np.unique(depth[part], return_inverse=True)
                factor = self._factor(unique, k, z, lam)  # (n_depth, n_node)
                bessel = j0(rho[part, np.newaxis] * lam[np.newaxis, :])
                integrand = factor[which] * bessel
                out[part] = -(integrand @ weights) / (4 * np.pi)
        return out

    def _factor(self, depth: Array, k: float, z: float, lam: Array) -> Array:
        """Return ``B̂ Z_B + Ĉ Z_C`` ``(n_depth, n_node)``, scaled so that nothing overflows.

        cosh, sinh of kλ times e^{-kλ}; Δ times e^{-2kλ}; L, M (A_g taken out, ε = -1)
        times e^{(z-k)λ}, folding in B, C's e^{-kλ} and Z_B, Z_C's e^{zλ}; Z_B, Z_C times
        e^{-zλ}: every factor is a decaying exponential or bounded.
        """
        nu = self.poisson
        lam_c = lam[np.newaxis, :]
        cc = depth[:, np.newaxis]
        kl = k * lam_c
        s2k = np.exp(-2 * kl)
        cosh_k, sinh_k = (1 + s2k) / 2, (1 - s2k) / 2
        delta = (4 * nu - 3) * cosh_k**2 - (kl**2 + (1 - 2 * nu) ** 2) * s2k
        e1 = np.exp(-lam_c * (2 * k + cc - z))
        e2 = np.exp(-lam_c * (2 * k - cc - z))
        big_l = (4 * nu - 3 + 2 * kl) * e1 - e2
        big_m = (4 * nu - 3 - 2 * kl) * e1 + e2
        p1 = 2 * (1 - nu) * cosh_k - kl * sinh_k
        p2 = (1 - 2 * nu) * sinh_k + kl * cosh_k
        p3 = (1 - 2 * nu) * sinh_k - kl * cosh_k
        p4 = 2 * (1 - nu) * cosh_k + kl * sinh_k
        zl = z * lam_c
        s2z = np.exp(-2 * zl)
        cosh_z, sinh_z = (1 + s2z) / 2, (1 - s2z) / 2
        z_b = zl * cosh_z - (1 - 2 * nu) * sinh_z
        z_c = 2 * (1 - nu) * cosh_z - zl * sinh_z
        out: Array = (
            lam_c / delta * ((big_l * p1 + big_m * p2) * z_b + (big_l * p3 + big_m * p4) * z_c)
        )
        return out
