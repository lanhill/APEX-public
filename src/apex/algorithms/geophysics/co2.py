"""CO₂'s adiabatic bulk modulus from Span & Wagner's equation of state.

New in APEX (the reference uses Batzle & Wang's natural-gas correlation, which overstates
CO₂'s modulus 2.6 to 6 times at storage conditions).

References : Span, R. & Wagner, W. (1996). A new equation of state for carbon dioxide
             covering the fluid region from the triple-point temperature to 1100 K at
             pressures up to 800 MPa. J. Phys. Chem. Ref. Data, 25(6), 1509-1596; through
             CoolProp (Bell et al. 2014, Ind. Eng. Chem. Res., 53(6), 2498-2508), MIT.
Behaviour  : K_S = rho c² at ``temperature_c`` (the deck's RTEMP, checked at setup) and the
             cell's simulated CO₂ density (explicit in temperature and density, so smooth
             across the critical region, and on the branch the simulator chose). A table
             in density, every ``step`` kg/m³ over ``density_range``, per branch below the
             critical temperature (up to each saturated density), interpolated by a C²
             cubic spline; wherever an interval's midpoint misses a direct call by more
             than 1e-4 (relative), the interval is halved, until none does. A
             density inside the two-phase dome (between the saturated vapour's and
             liquid's) has no speed of sound: it is snapped to the nearer saturated
             density, and counted.

Shapes: densities ``(n_cell,)`` (kg/m³), moduli ``(n_cell,)`` (Pa), pressures (Pa).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics.errors import GeophysicsModelError

type Array = NDArray[np.float64]

#: The table's largest accepted midpoint error, relative to direct calls.
TABLE_TOLERANCE = 1e-4
#: Pressures are rounded to this (Pa) before direct calls at the simulator's pressures:
#: a plume's few thousand distinct values, each one call.
_PRESSURE_ROUNDING = 1e3
#: The most halvings of an interval before the table is refused.
_MAX_REFINEMENTS = 12


def _coolprop() -> Any:  # noqa: ANN401 (CoolProp has no type stubs)
    try:
        import CoolProp.CoolProp as CP  # optional dependency (the co2 extra)
    except ImportError as error:
        raise ImportError(
            "gas_model = 'span_wagner' needs CoolProp: install APEX's co2 extra "
            "(uv sync --inexact --extra co2)"
        ) from error
    return CP


@dataclass
class SpanWagnerCO2:
    """CO₂'s adiabatic bulk modulus K_S(T, rho) at one temperature; see the module.

    Raises
    ------
    ImportError
        Without CoolProp (the ``co2`` extra).
    ValueError
        If the settings are invalid, or the table's midpoint error exceeds
        :data:`TABLE_TOLERANCE`.
    """

    temperature_c: float
    density_range: tuple[float, float] = (1.0, 1200.0)
    step: float = 0.5
    #: The saturated vapour's and liquid's densities below the critical temperature.
    saturated: tuple[float, float] | None = field(init=False, default=None)
    _branches: list[tuple[float, float, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build and check the table."""
        from scipy.interpolate import CubicSpline

        cp = _coolprop()
        low, high = self.density_range
        if not 0 < low < high or not self.step > 0:
            raise ValueError("eos_density_range must be (low, high) with 0 < low < high")
        t = self.temperature_c + 273.15
        if t < float(cp.PropsSI("Ttriple", "CO2")):
            raise ValueError(f"{self.temperature_c:g} °C is below CO₂'s triple point")
        spans = [(low, high)]
        if t < float(cp.PropsSI("Tcrit", "CO2")):
            vapour = float(cp.PropsSI("D", "T", t, "Q", 1, "CO2"))
            liquid = float(cp.PropsSI("D", "T", t, "Q", 0, "CO2"))
            self.saturated = (vapour, liquid)
            spans = [(low, min(high, vapour)), (max(low, liquid), high)]
        self._branches = []
        for a, b in spans:
            if not b > a:
                continue
            grid = np.arange(a, b, self.step)
            grid = np.append(grid[grid < b - 1e-9 * b], b)
            if grid.size < 4:  # a cubic spline needs a few nodes: densify a short branch
                grid = np.linspace(a, b, 4)
            values = self._direct(t, grid, self.saturated)
            for _ in range(_MAX_REFINEMENTS + 1):
                spline = CubicSpline(grid, values)
                mid = 0.5 * (grid[:-1] + grid[1:])
                exact = self._direct(t, mid, self.saturated)
                miss = np.abs(spline(mid) / exact - 1.0) > TABLE_TOLERANCE
                if not miss.any():
                    break
                # Halve the intervals that miss (CO₂'s modulus bends sharply next to a
                # saturated density just below the critical temperature).
                order = np.argsort(np.concatenate((grid, mid[miss])))
                grid = np.concatenate((grid, mid[miss]))[order]
                values = np.concatenate((values, exact[miss]))[order]
            else:
                raise ValueError(
                    f"CO₂'s modulus table at {self.temperature_c:g} °C misses direct calls by "
                    f"more than {TABLE_TOLERANCE:g} over {a:g}-{b:g} kg/m3 after "
                    f"{_MAX_REFINEMENTS} refinements"
                )
            self._branches.append((a, b, spline))

    @staticmethod
    def _direct(t: float, density: Array, saturated: tuple[float, float] | None = None) -> Array:
        """K_S = rho c² (Pa) by CoolProp at temperature ``t`` (K).

        A saturated density is taken at its saturation state (quality 1 or 0): from
        (T, rho) CoolProp's flash can round it into the two-phase dome near the critical
        point, where it has no speed of sound.
        """
        cp = _coolprop()
        state = cp.AbstractState("HEOS", "CO2")
        out = np.empty(np.shape(density))
        for n, d in enumerate(np.asarray(density, dtype=np.float64)):
            if saturated is not None and d in saturated:
                state.update(cp.QT_INPUTS, 1.0 if d == saturated[0] else 0.0, t)
            else:
                state.update(cp.DmassT_INPUTS, float(d), t)
            c = float(state.speed_sound())
            out[n] = d * c * c
        return out

    def modulus(self, density: ArrayLike) -> tuple[Array, NDArray[np.bool_]]:
        """Return K_S (Pa) at each density, and which densities were snapped.

        A density inside the two-phase dome is taken at the nearer saturated density.

        Raises
        ------
        GeophysicsModelError
            If a density is outside ``density_range`` (or not finite).
        """
        rho = np.asarray(density, dtype=np.float64)
        low, high = self.density_range
        bad = ~(np.isfinite(rho) & (rho >= low) & (rho <= high))
        if bad.any():
            cell = int(np.argmax(bad))
            raise GeophysicsModelError(
                f"CO₂ density {rho[cell]:g} kg/m3 (cell {cell}, {int(bad.sum())} cell(s)) is "
                f"outside eos_density_range {low:g}-{high:g}"
            )
        snapped = np.zeros(rho.shape, dtype=bool)
        if self.saturated is not None:
            vapour, liquid = self.saturated
            snapped = (rho > vapour) & (rho < liquid)
            nearer = np.where(rho - vapour < liquid - rho, vapour, liquid)
            rho = np.where(snapped, nearer, rho)
        out = np.empty(rho.shape)
        placed = np.zeros(rho.shape, dtype=bool)
        for a, b, spline in self._branches:
            inside = ~placed & (rho >= a) & (rho <= b)
            out[inside] = spline(rho[inside])
            placed |= inside
        # A short branch outside the range leaves no cell unplaced: snapping lands on the
        # saturated densities, which end a branch.
        if not placed.all():
            cell = int(np.argmax(~placed))
            raise GeophysicsModelError(f"CO₂ density {rho[cell]:g} kg/m3 is on no branch")
        return out, snapped

    def modulus_at_pressure(self, pressure: ArrayLike) -> Array:
        """Return K_S (Pa) at each pressure (Pa) and the temperature, by direct calls.

        For the consistency check against the modulus at the simulator's density: the
        pressure given is the one the restart reports (the reference phase's, without the
        capillary pressure), rounded to 1 kPa, each distinct one called. NaN where
        CoolProp has no state there (e.g. above the melting line), which the check counts.
        """
        cp = _coolprop()
        state = cp.AbstractState("HEOS", "CO2")
        t = self.temperature_c + 273.15
        p = np.round(np.asarray(pressure, dtype=np.float64) / _PRESSURE_ROUNDING)
        values, inverse = np.unique(p, return_inverse=True)
        k = np.full(values.size, np.nan)
        for n, v in enumerate(values * _PRESSURE_ROUNDING):
            try:
                state.update(cp.PT_INPUTS, float(v), t)
                k[n] = float(state.rhomass()) * float(state.speed_sound()) ** 2
            except ValueError:
                continue  # no state: counted as inconsistent
        out: Array = k[inverse].reshape(np.shape(pressure))
        return out


#: Half-width (K) of the band around CO₂'s critical temperature where the modulus has its
#: cusp: cells there are computed by direct calls, never interpolated.
CRITICAL_BAND_K = 0.05
#: Rounding of a band cell's temperature (K) and density (kg/m³) before its direct call.
_BAND_T_ROUNDING, _BAND_RHO_ROUNDING = 1e-3, 1e-2
#: Scaled-density samples at which a temperature interval is checked.
_S_SAMPLES = 0.5 * (1.0 - np.cos(np.pi * (np.arange(160) + 0.5) / 160))
#: Where in each temperature interval it is checked: a four-point cubic's error peaks off
#: the midpoint in the end intervals, so the quarter points too.
_T_CHECKS = (0.25, 0.5, 0.75)
#: The interpolation in temperature is held to half the tolerance: the nodes' own tables
#: may miss by up to TABLE_TOLERANCE, which the cubic's weights carry over (up to about
#: 1.6 in an end interval), so the two together stay near it.
_T_TOLERANCE = TABLE_TOLERANCE / 2


@dataclass
class SpanWagnerCO2Field:
    """CO₂'s adiabatic bulk modulus K_S(T, rho) over a temperature range.

    A cell's temperature varies (a geothermal gradient, a thermal run), so the modulus is
    tabulated in temperature as well as density:

    - **Nodes in temperature**, every ``temperature_step`` K at first, on each side of the
      critical band; at each node the one-dimensional table of :class:`SpanWagnerCO2`
      (density, per branch below the critical temperature, checked to 1e-4).
    - **A scaled density** ``s`` per branch, so that a branch is a rectangle in (T, s)
      although its saturated density moves with T: the vapour branch
      ``s = (rho - low) / (rho_v(T) - low)``, the liquid ``s = (rho - rho_l(T)) /
      (high - rho_l(T))``, above the critical temperature ``s = (rho - low) / (high - low)``.
    - **In temperature**, a cubic through the four nearest nodes of the cell's side of the
      critical band, at the cell's ``s`` mapped back to each node's density.
    - **Refinement:** every interval is checked at its quarter points and midpoint
      (:data:`_T_CHECKS`), at :data:`_S_SAMPLES` on each branch, against direct calls; an
      interval that misses by more than :data:`_T_TOLERANCE` (relative) is halved, until
      none does. The checks are at samples: about 1e-4 is what is measured between them
      (the nodes' density tables may themselves miss by up to 1e-4).
    - **The critical band** (``|T - T_c| < CRITICAL_BAND_K``), where K has a cusp (6.7 to
      8.5 MPa between 30.99 and 31.1 °C at the critical density), is computed by direct
      calls at the cells' (T, rho) rounded to 1 mK and 0.01 kg/m³.

    A density inside the two-phase dome at the cell's temperature is snapped to the nearer
    saturated density and counted, as in :class:`SpanWagnerCO2`.

    Shapes: temperatures (°C) and densities (kg/m³) ``(n_cell,)``; moduli (Pa).

    Raises
    ------
    ImportError
        Without CoolProp (the ``co2`` extra).
    ValueError
        If the settings are invalid, or the table cannot meet :data:`TABLE_TOLERANCE`.
    """

    temperature_range: tuple[float, float]
    density_range: tuple[float, float] = (1.0, 1200.0)
    temperature_step: float = 1.0
    #: The nodes (K) of each side of the critical band, and their one-dimensional tables.
    _groups: list[tuple[Array, list[SpanWagnerCO2]]] = field(init=False, repr=False)
    _t_crit: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build and check the table."""
        cp = _coolprop()
        low_c, high_c = self.temperature_range
        if not low_c <= high_c or not self.temperature_step > 0:
            raise ValueError("temperature_range must be (low, high) with low <= high")
        low, high = low_c + 273.15, high_c + 273.15
        if low < float(cp.PropsSI("Ttriple", "CO2")):
            raise ValueError(f"{low_c:g} °C is below CO₂'s triple point")
        self._t_crit = float(cp.PropsSI("Tcrit", "CO2"))
        band_low, band_high = self._t_crit - CRITICAL_BAND_K, self._t_crit + CRITICAL_BAND_K
        self._groups = []
        for a, b in ((low, min(high, band_low)), (max(low, band_high), high)):
            if b < a:
                continue  # the range has no temperature on this side of the band
            # At least four nodes, so that every cell of this side has a cubic stencil.
            count = max(4, int(np.ceil((b - a) / self.temperature_step)) + 1)
            nodes = np.linspace(a, b, count) if b > a else a + np.linspace(0.0, 1e-3, 4)
            tables = [self._table(t) for t in nodes]
            self._check_branches(tables)
            for _ in range(_MAX_REFINEMENTS + 1):
                mids = 0.5 * (nodes[:-1] + nodes[1:])
                missing = [
                    i
                    for i in range(nodes.size - 1)
                    if any(
                        self._misses(nodes[i] + f * (nodes[i + 1] - nodes[i]), nodes, tables)
                        for f in _T_CHECKS
                    )
                ]
                if not missing:
                    break
                for i in reversed(missing):
                    nodes = np.insert(nodes, i + 1, mids[i])
                    tables.insert(i + 1, self._table(mids[i]))
                self._check_branches(tables)
            else:
                raise ValueError(
                    f"CO₂'s modulus table over {low_c:g}-{high_c:g} °C misses direct calls by "
                    f"more than {TABLE_TOLERANCE:g} after {_MAX_REFINEMENTS} refinements in "
                    "temperature"
                )
            self._groups.append((nodes, tables))

    def _check_branches(self, tables: list[SpanWagnerCO2]) -> None:
        """Refuse a density range that leaves a branch out at a node below T_c."""
        low, high = self.density_range
        for table in tables:
            if table.saturated is not None and len(table._branches) != 2:
                vapour, liquid = table.saturated
                raise ValueError(
                    f"eos_density_range {low:g}-{high:g} kg/m3 must reach both of CO₂'s branches "
                    f"over temperature_range (at {table.temperature_c:.3f} °C the saturated "
                    f"densities are {vapour:.1f} and {liquid:.1f})"
                )

    def _table(self, t: float) -> SpanWagnerCO2:
        return SpanWagnerCO2(t - 273.15, self.density_range)

    def _saturated(self, t: Array) -> tuple[Array, Array]:
        """Return the saturated vapour's and liquid's densities at each temperature below T_c."""
        cp = _coolprop()
        state = cp.AbstractState("HEOS", "CO2")
        values, inverse = np.unique(t, return_inverse=True)
        vapour, liquid = np.empty(values.size), np.empty(values.size)
        for n, v in enumerate(values):
            state.update(cp.QT_INPUTS, 1.0, float(v))
            vapour[n] = float(state.rhomass())
            state.update(cp.QT_INPUTS, 0.0, float(v))
            liquid[n] = float(state.rhomass())
        return vapour[inverse], liquid[inverse]

    def _scaled(
        self,
        t: Array,
        rho: Array,
        subcritical: bool,
        saturated: tuple[Array, Array] | None = None,
    ) -> tuple[Array, NDArray[np.bool_]]:
        """Return each cell's ``s`` and whether it is on the liquid branch."""
        low, high = self.density_range
        if not subcritical:
            return (rho - low) / (high - low), np.zeros(rho.shape, dtype=bool)
        vapour, liquid = self._saturated(t) if saturated is None else saturated
        on_liquid = rho >= liquid
        s = np.where(
            on_liquid,
            (rho - liquid) / np.maximum(high - liquid, 1e-12),
            (rho - low) / np.maximum(vapour - low, 1e-12),
        )
        return s, on_liquid

    def _at_node(self, table: SpanWagnerCO2, s: Array, on_liquid: NDArray[np.bool_]) -> Array:
        """Return K at a node's temperature for scaled densities ``s`` on the given branches."""
        low, high = self.density_range
        out = np.empty(s.shape)
        if table.saturated is None:
            [(_, _, spline)] = table._branches
            out[:] = spline(low + s * (high - low))
            return out
        vapour, liquid = table.saturated
        by_branch = {a: spline for a, _, spline in table._branches}
        # A branch shorter than the density range is the vapour's (from low) or the
        # liquid's (from its saturated density): both exist below T_c with these ranges.
        vap, liq = by_branch[low], by_branch[max(low, liquid)]
        out[~on_liquid] = vap(low + s[~on_liquid] * (vapour - low))
        out[on_liquid] = liq(liquid + s[on_liquid] * (high - liquid))
        return out

    @staticmethod
    def _lagrange(t: Array, nodes: Array, index: NDArray[np.intp]) -> Array:
        """Return the cubic Lagrange weights ``(n_cell, 4)`` of the nodes ``index .. index + 3``."""
        x = nodes[index[:, None] + np.arange(4)]
        w = np.ones(x.shape)
        for j in range(4):
            for m in range(4):
                if m != j:
                    w[:, j] *= (t - x[:, m]) / (x[:, j] - x[:, m])
        return w

    def _interpolate(
        self,
        t: Array,
        rho: Array,
        nodes: Array,
        tables: list[SpanWagnerCO2],
        saturated: tuple[Array, Array] | None = None,
    ) -> Array:
        """Return K (Pa) interpolated at ``t`` (K, one side of the band) and ``rho``."""
        s, on_liquid = self._scaled(t, rho, nodes[0] < self._t_crit, saturated)
        # The four nearest nodes: the interval's left node minus one, kept inside the side.
        index = np.clip(np.searchsorted(nodes, t, side="right") - 2, 0, nodes.size - 4)
        weights = self._lagrange(t, nodes, index)
        out = np.zeros(t.shape)
        for j in range(4):
            for node in np.unique(index + j):
                at = index + j == node
                out[at] += weights[at, j] * self._at_node(tables[node], s[at], on_liquid[at])
        return out

    def _misses(self, t: float, nodes: Array, tables: list[SpanWagnerCO2]) -> bool:
        """Whether the interpolation at temperature ``t`` (K) misses direct calls."""
        low, high = self.density_range
        if t < self._t_crit:
            vapour, liquid = self._saturated(np.array([t]))
            rho = np.concatenate((
                low + _S_SAMPLES * (vapour[0] - low),
                liquid[0] + _S_SAMPLES * (high - liquid[0]),
            ))  # fmt: skip
            saturated: tuple[float, float] | None = (float(vapour[0]), float(liquid[0]))
        else:
            rho = low + _S_SAMPLES * (high - low)
            saturated = None
        temperature = np.full(rho.shape, t)
        approx = self._interpolate(temperature, rho, nodes, tables)
        exact = SpanWagnerCO2._direct(t, rho, saturated)
        return bool(np.any(np.abs(approx / exact - 1.0) > _T_TOLERANCE))

    def modulus(
        self, temperature_c: ArrayLike, density: ArrayLike
    ) -> tuple[Array, NDArray[np.bool_], NDArray[np.bool_]]:
        """Return K_S (Pa) per cell, which densities were snapped, and which were in the band.

        Raises
        ------
        GeophysicsModelError
            If a temperature is outside ``temperature_range`` or a density outside
            ``density_range`` (or either is not finite).
        """
        t = np.asarray(temperature_c, dtype=np.float64) + 273.15
        rho = np.asarray(density, dtype=np.float64).copy()
        low_c, high_c = self.temperature_range
        bad_t = ~(np.isfinite(t) & (t >= low_c + 273.15) & (t <= high_c + 273.15))
        if bad_t.any():
            cell = int(np.argmax(bad_t))
            raise GeophysicsModelError(
                f"temperature {t[cell] - 273.15:g} °C (cell {cell}, {int(bad_t.sum())} cell(s)) "
                f"is outside temperature_range {low_c:g}-{high_c:g}"
            )
        low, high = self.density_range
        bad = ~(np.isfinite(rho) & (rho >= low) & (rho <= high))
        if bad.any():
            cell = int(np.argmax(bad))
            raise GeophysicsModelError(
                f"CO₂ density {rho[cell]:g} kg/m3 (cell {cell}, {int(bad.sum())} cell(s)) is "
                f"outside eos_density_range {low:g}-{high:g}"
            )
        snapped = np.zeros(rho.shape, dtype=bool)
        below = t < self._t_crit
        vapour_all, liquid_all = np.full(t.shape, np.nan), np.full(t.shape, np.nan)
        if below.any():
            vapour, liquid = self._saturated(t[below])
            vapour_all[below], liquid_all[below] = vapour, liquid
            r = rho[below]
            dome = (r > vapour) & (r < liquid)
            r = np.where(dome, np.where(r - vapour < liquid - r, vapour, liquid), r)
            rho[below], snapped[below] = r, dome
        out = np.empty(rho.shape)
        placed = np.zeros(rho.shape, dtype=bool)
        for nodes, tables in self._groups:
            side = (t >= nodes[0]) & (t <= nodes[-1])
            if side.any():
                saturated = (vapour_all[side], liquid_all[side])
                out[side] = self._interpolate(t[side], rho[side], nodes, tables, saturated)
                placed |= side
        # The band is what no side holds (so no cell falls between them by rounding).
        band = ~placed
        if band.any():
            out[band] = self._band(t[band], rho[band])
        return out, snapped, band

    def _band(self, t: Array, rho: Array) -> Array:
        """Direct calls at the rounded (T, rho) of the band's cells, each distinct pair once.

        Below T_c a rounded density inside the dome, or within the rounding of a saturated
        density, is taken at that saturation state (quality 1 or 0): a (T, rho) flash there
        may land in the two-phase region, which has no speed of sound.
        """
        cp = _coolprop()
        state = cp.AbstractState("HEOS", "CO2")
        pairs = np.stack(
            (np.round(t / _BAND_T_ROUNDING), np.round(rho / _BAND_RHO_ROUNDING)), axis=1
        )
        values, inverse = np.unique(pairs, axis=0, return_inverse=True)
        k = np.empty(values.shape[0])
        for n, (tt, rr) in enumerate(values):
            temperature, density = float(tt) * _BAND_T_ROUNDING, float(rr) * _BAND_RHO_ROUNDING
            if temperature < self._t_crit:
                state.update(cp.QT_INPUTS, 1.0, temperature)
                vapour = float(state.rhomass())
                state.update(cp.QT_INPUTS, 0.0, temperature)
                liquid = float(state.rhomass())
                margin = _BAND_RHO_ROUNDING
                if vapour - margin <= density <= liquid + margin:
                    quality = 1.0 if density - vapour < liquid - density else 0.0
                    state.update(cp.QT_INPUTS, quality, temperature)
                    k[n] = float(state.rhomass()) * float(state.speed_sound()) ** 2
                    continue
            state.update(cp.DmassT_INPUTS, density, temperature)
            k[n] = density * float(state.speed_sound()) ** 2
        out: Array = k[inverse.ravel()]
        return out

    def modulus_at_pressure(self, pressure: ArrayLike, temperature_c: ArrayLike) -> Array:
        """K_S (Pa) at each cell's pressure (Pa) and temperature (°C), by direct calls.

        For the consistency count, as :meth:`SpanWagnerCO2.modulus_at_pressure`: pressures
        rounded to 1 kPa and temperatures to 1 mK, each distinct pair called once; NaN where
        CoolProp has no state.
        """
        cp = _coolprop()
        state = cp.AbstractState("HEOS", "CO2")
        p = np.round(np.asarray(pressure, dtype=np.float64) / _PRESSURE_ROUNDING)
        t = np.round((np.asarray(temperature_c, dtype=np.float64) + 273.15) / _BAND_T_ROUNDING)
        pairs = np.stack((np.broadcast_to(p, np.shape(t)), t), axis=1) if t.ndim else None
        if pairs is None:
            raise ValueError("temperature_c must be per cell")
        values, inverse = np.unique(pairs, axis=0, return_inverse=True)
        k = np.full(values.shape[0], np.nan)
        for n, (pp, tt) in enumerate(values):
            try:
                state.update(
                    cp.PT_INPUTS, float(pp) * _PRESSURE_ROUNDING, float(tt) * _BAND_T_ROUNDING
                )
                k[n] = float(state.rhomass()) * float(state.speed_sound()) ** 2
            except ValueError:
                continue  # no state: counted as inconsistent
        out: Array = k[inverse.ravel()].reshape(np.shape(t))
        return out
