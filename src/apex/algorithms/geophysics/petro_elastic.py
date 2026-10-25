"""Petro-elastic models: Vp, Vs and density of each cell from its flow state.

Ported from SimulatorWrap
-------------------------
Modules    : src/subsurface/rockphysics/standardrp.py (``elasticproperties``) and
             src/subsurface/rockphysics/softsandrp.py (``elasticproperties``), as
             ``flow_rock.calc_pem`` (src/subsurface/multphaseflow/flow_rock.py L364) calls
             them
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)
References : see :mod:`apex.algorithms.geophysics.rock_physics`; dissolved CO₂: Lebedev, M.,
             Bilenko, O., Mikhaltsevitch, V., Pervukhina, M. & Gurevich, B. (2014).
             Laboratory measurements of ultrasonic velocities in CO₂ saturated brines.
             Energy Procedia 63, 4273-4280; ``standard`` follows
             Fahimuddin, A. (2010), PhD thesis, University of Bergen, Report 1.
Behaviour  : ``standard`` is the reference's model, its constants options with the
             reference's values as defaults. ``soft_sand`` is the reference's dry rock
             and mixing with Batzle & Wang's fluids taken from the simulator's phase
             densities, which it requires (so CoolProp is never needed).
Differences: - Cells outside a model's validity are refused (:class:`PetroElasticError`),
               naming the cell (an index into the cells given): a non-positive porosity
               (``soft_sand`` without its cap: porosity >= φc), a negative clay volume,
               saturations outside [0, 1] or not summing to 1, a non-positive dry or
               saturated modulus. The reference computes on (its ``abs(K)`` hides a
               negative modulus). Above φc both models compute as the reference does:
               ``standard`` extrapolates its interpolation (φ/φc > 1), ``soft_sand``
               caps φ/φc at ``porosity_ratio_cap``.
             - The effective pressure is ``max(P_ov - P, min_effective_pressure)``, with
               Biot's coefficient 1 as in the reference; the reference replaces only
               negative values by 0.01 MPa, and divides by zero at exactly 0.
             - ``soft_sand``'s fluids differ from the reference's (pet_parity.md, G3-G6):
               each phase's modulus uses its own density, not the mixture's; the brine's
               salinity term of Batzle & Wang's density is not added (the simulator's
               water density already holds it); oil's modulus is in MPa (the reference's
               is in Pa); the liquid of Brie's mixture is the Wood average of oil and
               water (the reference keeps the last liquid phase listed, and fails
               without gas).
             - ``soft_sand`` refuses Batzle & Wang's correlations outside their ranges
               (the gas's, unless ``extrapolate_gas_correlation``; see the class).
             - The oil's reference density is ``oil_reference_density``; without it,
               the in-situ density stands in (an approximation, documented).

Shapes: every array is ``(n_cell,)`` (or broadcasts to it); pressures are in Pa,
densities in kg/m³, velocities in m/s, moduli in Pa. The published fits are in MPa, so
the models convert at their boundary.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics import rock_physics as rp
from apex.algorithms.geophysics.co2 import SpanWagnerCO2, SpanWagnerCO2Field
from apex.algorithms.geophysics.errors import GeophysicsModelError
from apex.algorithms.geophysics.registries import petro_elastic_models

type Array = NDArray[np.float64]

type Phase = Literal["oil", "water", "gas"]
PHASES: tuple[Phase, ...] = ("oil", "water", "gas")
_PHASE: dict[str, Phase] = {p: p for p in PHASES}
_MPA = 1e6
#: Saturations come from single-precision restart files.
_SATURATION_TOLERANCE = 1e-5
#: Densities used where a phase is absent (its saturation is 0, so they weigh nothing);
#: they only keep the formulas finite.
_PLACEHOLDER_DENSITY = {"oil": 800.0, "water": 1000.0, "gas": 200.0}


class PetroElasticError(GeophysicsModelError):
    """A cell is outside the petro-elastic model's validity.

    Attributes
    ----------
    cell
        The first such cell's index.
    quantity, value
        What is out of range and its value there.
    count
        How many cells are.
    """

    def __init__(self, quantity: str, cell: int, value: float, count: int, rule: str) -> None:
        self.quantity = quantity
        self.cell = cell
        self.value = value
        self.count = count
        super().__init__(f"{quantity} = {value:g} at cell {cell} ({count} cell(s) in all): {rule}")


def _refuse(bad: NDArray[np.bool_], quantity: str, values: Array, rule: str) -> None:
    if np.any(bad):
        cell = int(np.flatnonzero(bad)[0])
        raise PetroElasticError(quantity, cell, float(values.flat[cell]), int(bad.sum()), rule)


@dataclass(frozen=True)
class FlowState:
    """The flow simulator's state of each cell at one time.

    Attributes
    ----------
    porosity
        ``(n_cell,)``, fraction.
    pressure
        ``(n_cell,)``, pore pressure (Pa).
    overburden_pressure
        ``(n_cell,)`` or scalar, total vertical stress (Pa).
    saturations
        Phase name (``"oil"``, ``"water"``, ``"gas"``) to ``(n_cell,)``; a phase left
        out has saturation 0.
    densities
        Phase name to ``(n_cell,)`` reservoir density (kg/m³), from the simulator's
        restart (``OIL_DEN``, ``WAT_DEN``, ``GAS_DEN``), or ``None``.
    temperature
        ``(n_cell,)`` temperature (°C) of each cell, for a model that takes the deck's
        (``soft_sand``'s ``temperature = "deck"``), or ``None``.
    """

    porosity: ArrayLike
    pressure: ArrayLike
    overburden_pressure: ArrayLike
    saturations: Mapping[str, ArrayLike]
    densities: Mapping[str, ArrayLike] | None = None
    temperature: ArrayLike | None = None


@dataclass(frozen=True)
class ElasticProperties:
    """Saturated elastic properties per cell, each ``(n_cell,)``.

    Attributes
    ----------
    vp, vs
        P and S velocities (m/s).
    density
        Bulk density (kg/m³).
    bulk_modulus, shear_modulus
        Saturated moduli (Pa).
    clamped_cells
        How many cells' effective pressure was raised to ``min_effective_pressure``
        (pore pressure at or above the overburden), for the simulator to report.
    counts
        Other counts for the simulator to report, by name (e.g. CO₂ densities snapped out
        of the two-phase dome).
    """

    vp: Array
    vs: Array
    density: Array
    bulk_modulus: Array
    shear_modulus: Array
    clamped_cells: int = 0
    counts: Mapping[str, int] = field(default_factory=dict)


class PetroElasticModel(Protocol):
    """A petro-elastic model (registry ``petro_elastic_models``).

    Optional capabilities, read when present (``soft_sand`` has them all):

    - ``check_phases(phases)``: refuse the deck's phases at setup (``ValueError``);
    - ``check_temperature(deck_c, varies, phases) -> list[str]``: refuse a temperature the
      deck contradicts, return what cannot be checked;
    - ``gas``: the gas phase's fluid (``"co2"`` brings CO2STORE's checks);
    - ``uncertain_options``: ``{option: transforms}``, the options a state variable may
      set per member, each with the transforms that keep it valid.
    """

    #: Whether :attr:`FlowState.densities` must be given.
    requires_densities: bool

    def elastic(self, state: FlowState) -> ElasticProperties:
        """Return each cell's elastic properties.

        Raises
        ------
        PetroElasticError
            If a cell is outside the model's validity.
        """
        ...


@dataclass(frozen=True)
class _Cells:
    porosity: Array
    pressure_mpa: Array
    effective_mpa: Array
    saturations: dict[Phase, Array]
    densities: dict[Phase, Array] | None
    clamped: int
    temperature: Array | None = None


def _cells(
    state: FlowState, max_porosity: float | None, min_effective_pressure: float, needs_density: bool
) -> _Cells:
    phi = np.atleast_1d(np.asarray(state.porosity, dtype=np.float64))
    if phi.ndim != 1:
        raise ValueError(f"porosity must be (n_cell,), got shape {phi.shape}")
    n = phi.size

    def cellwise(name: str, x: ArrayLike) -> Array:
        a = np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()
        _refuse(~np.isfinite(a), name, a, "must be finite")
        return a

    phi = cellwise("porosity", phi)
    if max_porosity is None:
        _refuse(phi <= 0, "porosity", phi, "must be positive (Gassmann is 0/0 at zero porosity)")
    else:
        _refuse(
            (phi <= 0) | (phi >= max_porosity), "porosity", phi,
            f"the model holds for 0 < porosity < {max_porosity:g} (Gassmann is 0/0 at zero "
            "porosity)",
        )  # fmt: skip
    unknown = sorted(set(state.saturations) - set(PHASES))
    if unknown or not state.saturations:
        raise ValueError(f"saturations must name some of {PHASES}; got {sorted(state.saturations)}")
    # In the caller's order, so that the mixtures sum as the reference's do.
    sats = {_PHASE[k]: cellwise(f"{k} saturation", v) for k, v in state.saturations.items()}
    for k, s in sats.items():
        _refuse(
            (s < -_SATURATION_TOLERANCE) | (s > 1 + _SATURATION_TOLERANCE), f"{k} saturation", s,
            "must be in [0, 1]",
        )  # fmt: skip
    total = sum(sats.values(), np.zeros(n))
    _refuse(
        np.abs(total - 1.0) > _SATURATION_TOLERANCE, "saturation sum", total,
        "the phases' saturations must sum to 1",
    )  # fmt: skip
    sats = {k: np.clip(s, 0.0, 1.0) for k, s in sats.items()}
    pressure = cellwise("pressure", state.pressure) / _MPA
    overburden = cellwise("overburden pressure", state.overburden_pressure) / _MPA
    floor = min_effective_pressure / _MPA
    effective = np.maximum(overburden - pressure, floor)
    clamped = int(np.sum(overburden - pressure < floor))
    densities = None
    if needs_density:
        if state.densities is None:
            raise ValueError(
                "this petro-elastic model needs the phases' densities (OIL_DEN, WAT_DEN, "
                "GAS_DEN from the restart files); none were given"
            )
        missing = sorted(set(sats) - set(state.densities))
        if missing:
            raise ValueError(f"no density for the phase(s) {missing}")
        densities = {}
        for k, s in sats.items():
            d = np.broadcast_to(np.asarray(state.densities[k], dtype=np.float64), (n,))
            present = s > 0
            # A restart may hold anything for a phase absent from a cell; only the cells
            # where the phase is present are checked and used.
            _refuse(
                present & ~(np.isfinite(d) & (d > 0)), f"{k} density", d,
                "must be finite and positive where the phase is present",
            )  # fmt: skip
            densities[k] = np.where(present, d, _PLACEHOLDER_DENSITY[k])
    temperature = None if state.temperature is None else cellwise("temperature", state.temperature)
    return _Cells(phi, pressure, effective, sats, densities, clamped, temperature)


def _saturated(
    phi: Array,
    dry_bulk: Array,
    dry_shear: Array,
    mineral_bulk: ArrayLike,
    mineral_density: ArrayLike,
    fluid_bulk: Array,
    fluid_density: Array,
) -> ElasticProperties:
    """Gassmann and the velocities; moduli in MPa in, Pa out."""
    bulk = rp.gassmann(dry_bulk, mineral_bulk, fluid_bulk, phi)
    return _saturated_moduli(phi, dry_bulk, dry_shear, bulk, mineral_density, fluid_density)


def _saturated_moduli(
    phi: Array,
    dry_bulk: Array,
    dry_shear: Array,
    bulk: Array,
    mineral_density: ArrayLike,
    fluid_density: Array,
) -> ElasticProperties:
    """Return the velocities of a saturated rock of bulk modulus ``bulk``; MPa in, Pa out."""
    for name, modulus in (("dry bulk modulus (MPa)", dry_bulk), ("dry shear modulus (MPa)",
                                                                   dry_shear)):  # fmt: skip
        _refuse(
            ~(modulus > 0), name, modulus,
            "must be positive; the cell's porosity is beyond the model's extrapolation",
        )  # fmt: skip
    _refuse(
        ~(bulk > 0), "saturated bulk modulus (MPa)", bulk,
        "must be positive; the cell's porosity or pressure is outside the model's range",
    )  # fmt: skip
    density = phi * fluid_density + (1.0 - phi) * np.asarray(mineral_density, dtype=np.float64)
    # Moduli in MPa over kg/m³: sqrt(MPa / (kg/m³)) is km/s.
    vp = 1000.0 * np.sqrt((bulk + 4.0 * dry_shear / 3.0) / density)
    vs = 1000.0 * np.sqrt(dry_shear / density)
    return ElasticProperties(
        vp=vp, vs=vs, density=density, bulk_modulus=bulk * _MPA, shear_modulus=dry_shear * _MPA
    )


#: The reference's fluid fits at 100 °C (``standardrp._phaseprops``), read off Batzle &
#: Wang's (1992) plots: density ``a + b P`` (kg/m³, P in MPa) and bulk modulus
#: ``c + d P`` (MPa) as ``(a, b, c, d)``.
STANDARD_FLUID_FITS: dict[Phase, tuple[float, float, float, float]] = {
    "oil": (829.9, 0.8, 995.79, 10.42),
    "water": (1067.3, 0.3, 2807.6, 9.0),
    "gas": (13.4, 4.7, 0.0, 2.75),
}


@petro_elastic_models.register("standard")
@dataclass
class StandardModel:
    """SimulatorWrap's ``standardrp``: fluid fits, mixed minerals, power-law dry rock.

    Then Gassmann.

    - Fluids: each phase's density and bulk modulus linear in pressure
      (:data:`STANDARD_FLUID_FITS`), mixed by saturation (density) and Wood (modulus).
      Simulator densities are not used.
    - Mineral: clay volume ``v = clay_volume_intercept - clay_volume_slope φ``; density
      linear in it; moduli the means of the clay-quartz Hashin-Shtrikman bounds.
    - Dry rock: pack moduli ``M_max (P_eff / P_ref)^κ`` at ``critical_porosity``, then
      the soft-sand interpolation to the mineral (``q = φ/φc``).

    Parameters are in SI (Pa, kg/m³); the defaults are the reference's.

    Parameters
    ----------
    critical_porosity
        The porosity at which the dry rock takes the pack's moduli, in ``(0, 1)`` (default
        0.4).
    clay_volume_intercept, clay_volume_slope
        The clay volume fraction is ``clay_volume_intercept - clay_volume_slope * porosity``
        (defaults 0.7 and 1.58); a cell where it leaves ``[0, 1]`` is refused.
    clay_bulk_modulus, clay_shear_modulus
        The clay's moduli in Pa (defaults 14.9 GPa and 1.95 GPa).
    clay_density
        The clay's density in kg/m³ (default 2540).
    quartz_bulk_modulus, quartz_shear_modulus
        The quartz's moduli in Pa (defaults 37 GPa and 44 GPa).
    quartz_density
        The quartz's density in kg/m³ (default 2650).
    pack_bulk_modulus, pack_shear_modulus
        The grain pack's moduli in Pa at ``pack_reference_pressure`` (defaults 3.31 GPa and
        2.84 GPa).
    pack_reference_pressure
        The effective pressure in Pa at which the pack has those moduli (default 8.8 MPa).
    pack_exponent
        The power of the pack moduli's growth with effective pressure, ``(P_eff /
        P_ref)^exponent`` (default 0.233).
    min_effective_pressure
        The lowest effective pressure in Pa (overburden minus pore pressure); cells below
        it are raised to it and counted (positive, default 0.01 MPa).
    fluid_fits
        Phase (``oil``, ``water``, ``gas``) to ``[a, b, k0, k1]``: density ``a + b P`` in
        kg/m³ and bulk modulus ``k0 + k1 P`` in MPa, ``P`` the pore pressure in MPa. A
        phase not given keeps the reference's fit.
    """

    critical_porosity: float = 0.4
    clay_volume_intercept: float = 0.7
    clay_volume_slope: float = 1.58
    clay_bulk_modulus: float = 14.9e9
    clay_shear_modulus: float = 1.95e9
    clay_density: float = 2540.0
    quartz_bulk_modulus: float = 37.0e9
    quartz_shear_modulus: float = 44.0e9
    quartz_density: float = 2650.0
    pack_bulk_modulus: float = 3.31e9
    pack_shear_modulus: float = 2.84e9
    pack_reference_pressure: float = 8.8e6
    pack_exponent: float = 0.233
    min_effective_pressure: float = 0.01e6
    #: Phases' fits overriding :data:`STANDARD_FLUID_FITS`, merged over them (a phase not
    #: given keeps the reference's fit). ``None`` (not a factory) keeps the option's schema
    #: serializable.
    fluid_fits: Mapping[Phase, tuple[float, float, float, float]] | None = None
    _fits: dict[Phase, tuple[float, float, float, float]] = field(
        init=False, repr=False, compare=False
    )

    requires_densities = False

    def __post_init__(self) -> None:
        """Refuse parameters outside the model's validity."""
        if not 0 < self.critical_porosity < 1:
            raise ValueError(f"critical_porosity must be in (0, 1), got {self.critical_porosity}")
        if self.min_effective_pressure <= 0:
            raise ValueError("min_effective_pressure must be positive")
        self._fits = {**STANDARD_FLUID_FITS, **(self.fluid_fits or {})}
        for phase, fit in self._fits.items():
            if len(fit) != 4:
                raise ValueError(
                    f"fluid_fits[{phase!r}] needs four coefficients (density a + b P, modulus "
                    f"k0 + k1 P), got {len(fit)}"
                )

    def elastic(self, state: FlowState) -> ElasticProperties:
        """Return each cell's elastic properties; see :class:`PetroElasticModel`."""
        # Above φc the reference extrapolates the interpolation (q = φ/φc > 1) up to where
        # the clay volume or a dry modulus turns negative; APEX does the same and refuses
        # there.
        c = _cells(state, None, self.min_effective_pressure, False)
        phi, p = c.porosity, c.pressure_mpa
        densities, moduli, sats = [], [], []
        for phase, s in c.saturations.items():
            a, b, k0, k1 = self._fits[phase]
            densities.append(a + b * p)
            moduli.append(k0 + k1 * p)
            sats.append(s)
        fluid_density = sum((s * d for s, d in zip(sats, densities, strict=True)), np.zeros(1))
        fluid_bulk = rp.wood(sats, moduli)
        clay = self.clay_volume_intercept - self.clay_volume_slope * phi
        _refuse(
            (clay < 0) | (clay > 1), "clay volume", clay,
            "clay_volume_intercept - clay_volume_slope * porosity must be in [0, 1]",
        )  # fmt: skip
        mineral_density = self.quartz_density + clay * (self.clay_density - self.quartz_density)
        k_lo, k_hi, g_lo, g_hi = rp.hashin_shtrikman_bounds(
            self.clay_bulk_modulus / _MPA, self.clay_shear_modulus / _MPA,
            self.quartz_bulk_modulus / _MPA, self.quartz_shear_modulus / _MPA, 1.0 - clay,
        )  # fmt: skip
        mineral_bulk, mineral_shear = (k_lo + k_hi) / 2, (g_lo + g_hi) / 2
        pack_bulk, pack_shear = rp.hertz_mindlin_power_law(
            c.effective_mpa, self.pack_bulk_modulus / _MPA, self.pack_shear_modulus / _MPA,
            self.pack_reference_pressure / _MPA, self.pack_exponent,
        )  # fmt: skip
        dry_bulk, dry_shear = rp.soft_sand_dry_moduli(
            phi / self.critical_porosity, pack_bulk, pack_shear, mineral_bulk, mineral_shear
        )
        e = _saturated(
            phi, dry_bulk, dry_shear, mineral_bulk, mineral_density, fluid_bulk, fluid_density
        )
        return replace(e, clamped_cells=c.clamped)


#: Specific gravities (air = 1) of the gases ``soft_sand`` knows, as the reference takes
#: them.
GAS_GRAVITY = {"co2": 1.5189, "methane": 0.5537}
#: The ranges of Batzle & Wang's gas correlation (a fit to Standing & Katz's natural-gas
#: chart): pseudo-reduced temperature and pressure.
GAS_T_PR = (1.05, 3.0)
GAS_P_PR_MAX = 15.0
#: The relative difference between CO₂'s modulus at the simulator's density and at its
#: pressure above which a cell is counted as inconsistent.
CO2_CONSISTENCY = 0.05
#: Batzle & Wang's water and brine velocity fits hold to about these.
WATER_TEMPERATURE_C = (0.0, 100.0)
#: Lebedev et al. (2014)'s measurements of CO₂-saturated brine, the fit's range.
LEBEDEV_RANGE = {
    "pressure_mpa": (2.0, 38.0),
    "temperature_c": (30.0, 70.0),
    "salinity_ppm": (0.0, 100_000.0),
}
WATER_PRESSURE_MPA = 100.0


@petro_elastic_models.register("soft_sand")
@dataclass
class SoftSandModel:
    """SimulatorWrap's ``softsandrp``: soft sand, Batzle & Wang fluids, Brie mixing.

    Then Gassmann.

    - Fluids (Batzle & Wang 1992, at ``temperature_c``): brine velocity with
      ``salinity`` (a weight fraction), dead-oil velocity, each phase's modulus
      ``rho v²`` from the simulator's density of that phase; the gas's adiabatic modulus
      (``gas``: CO₂, the reference's, or methane). Oil and water are averaged by Wood
      into the liquid, and liquid and gas mixed by Brie with ``brie_exponent`` (``mixing``
      ``"brie"``, the reference's). ``"wood"`` mixes every phase by Wood (uniform
      saturation, the lower bound); ``"patchy"`` saturates the rock with each fluid
      alone and averages the P-wave moduli harmonically by saturation (Gassmann-Hill, the
      upper bound at seismic frequencies); ``"patchiness"`` takes the P-wave modulus
      ``(1 - q) M_wood + q M_patchy`` with ``q`` = ``patchiness`` in [0, 1], within both
      bounds for every q. Brie is bounded by neither: for e > 1 it falls below Wood as
      the gas saturation approaches 1, and for small e it exceeds patchy at low gas
      saturation. ``"patchy"`` and ``"patchiness"`` take two fluid phases.
    - ``patchiness``, ``coordination_number`` and the mineral's moduli and density may be
      set per member by a state variable (``{ state = "<variable>", value = <fallback> }``
      in a configuration; :attr:`uncertain_options`), through the logistic (q) and the
      exponential (the rest).
    - CO₂'s modulus (``gas_model``): ``"span_wagner"``, K_S = rho c² of Span & Wagner's
      equation of state at the cell's simulated CO₂ density
      (:class:`~apex.algorithms.geophysics.co2.SpanWagnerCO2`, the ``co2`` extra; a
      table over ``eos_density_range``), the default for CO₂; or ``"batzle_wang"``, the
      reference's correlation, for methane and for parity. A configuration that sets
      ``extrapolate_gas_correlation`` without ``gas_model`` keeps Batzle & Wang (that
      option means something only for the correlation). Densities in the two-phase dome
      are snapped to the nearer saturated density, and the cells whose modulus at the
      simulator's pressure differs by more than 5 % are counted (both reported).
    - Dry rock: Hertz-Mindlin at ``critical_porosity`` with ``coordination_number``,
      then the soft-sand interpolation to the mineral, with ``φ/φc`` capped at
      ``porosity_ratio_cap`` as the reference does (``None``: no cap).
    - The temperature: ``temperature_c``, one for every cell, by
      default; ``temperature = "deck"`` takes each cell's from the deck (the simulator
      gives :attr:`FlowState.temperature`: a depth table's at the cell's depth, or a
      thermal run's restart ``TEMP``). Each cell's must be in Batzle & Wang's 0-100 °C.
      CO₂'s modulus is then :class:`~apex.algorithms.geophysics.co2.SpanWagnerCO2Field`
      over ``temperature_range`` (°C; the composition takes a depth table's range, a
      thermal run must give it), and a cell outside it fails its member.
    - Dissolved CO₂: ``dissolved_co2 = "lebedev2014"`` corrects the
      brine's velocity in each cell with free CO₂ (gas saturation above
      ``dissolved_co2_min_sg``), Lebedev et al. (2014)::

          V = V_BW + 20.42 - 0.08377 T - 0.18 P - 0.0004173 S   (m/s; °C, MPa, ppm)

      for brine at equilibrium with free CO₂ (full saturation only: brine without free CO₂
      in its cell is not corrected). Outside the paper's fitted 2-38 MPa and 30-70 °C a
      cell is not corrected and is counted; ``salinity`` must be within its 0-100,000 ppm.
      The brine's modulus is the simulator's density times the corrected velocity squared.

    Parameters are in SI (Pa, kg/m³, °C); the defaults are the reference's.

    Limits, refused rather than extrapolated:

    - Batzle & Wang's gas modulus fits natural gases for pseudo-reduced temperatures
      1.05-3 and pressures up to 15. CO₂ at the reference's 37 °C is at 0.88, outside
      it: its modulus is then biased (about 2.6 times Span & Wagner's at 20 MPa) and turns
      negative or infinite below about 10 °C. A cell with gas is refused there unless
      ``extrapolate_gas_correlation`` is set, as the reference would need; a non-positive
      or infinite gas modulus is refused always.
    - The dead-oil velocity takes the oil's density at standard conditions,
      ``oil_reference_density``; without it the simulator's in-situ (live) oil density
      stands in, an approximation (about 10 % in velocity for a live oil of 0.75 in situ
      and 0.85 at the surface).
    - Water and brine: 0-100 °C and up to 100 MPa.

    Parameters
    ----------
    critical_porosity
        The porosity at which the dry rock takes the Hertz-Mindlin pack's moduli, in
        ``(0, 1)`` (default 0.36).
    coordination_number
        The average number of contacts per grain in the pack (positive, default 9); may
        be set per member by a state variable.
    mineral_bulk_modulus, mineral_shear_modulus
        The mineral's moduli in Pa (defaults 37 GPa and 44 GPa, quartz); their Poisson's
        ratio must be in ``(-1, 0.5)``. May be set per member by a state variable.
    mineral_density
        The mineral's density in kg/m³ (default 2650); may be set per member by a state
        variable.
    temperature_c
        The reservoir temperature in °C for every cell, within 0-100 (default 37); not
        used for the cells with ``temperature = "deck"``.
    salinity
        The brine's salt content as a weight fraction, in ``[0, 0.35)`` (default 0.035);
        at most 0.1 with ``dissolved_co2 = "lebedev2014"``.
    gas
        The gas phase: ``"co2"`` (default) or ``"methane"``.
    brie_exponent
        The exponent of Brie's mixing of liquid and gas (positive, default 5); larger
        values approach uniform (Wood) mixing.
    porosity_ratio_cap
        The cap on ``porosity / critical_porosity`` in the dry-rock interpolation, in
        ``(0, 1]`` (default 0.9); unset for no cap.
    min_effective_pressure
        The lowest effective pressure in Pa (overburden minus pore pressure); cells below
        it are raised to it and counted (positive, default 0.01 MPa).
    oil_reference_density
        The oil's density at standard conditions in kg/m³, below 1080, for the dead-oil
        velocity; unset uses the simulator's in-situ oil density, an approximation.
    extrapolate_gas_correlation
        Allow Batzle & Wang's gas modulus outside its fitted range (default false: cells
        with gas are refused there). Only with ``gas_model = "batzle_wang"``.
    gas_model
        The gas modulus: ``"span_wagner"``, CO₂'s equation of state at the simulated
        density (needs the ``co2`` extra), or ``"batzle_wang"``, the reference's
        correlation. Unset: Span & Wagner for CO₂, unless ``extrapolate_gas_correlation``
        is set; Batzle & Wang for methane.
    mixing
        How the fluids are mixed: ``"brie"`` (default), ``"wood"`` (uniform saturation,
        the lower bound), ``"patchy"`` (the upper bound) or ``"patchiness"`` (between
        the two by ``patchiness``). The last two take two fluid phases.
    patchiness
        With ``mixing = "patchiness"``, the weight of the patchy bound against Wood's, in
        ``[0, 1]`` (default 0.5); may be set per member by a state variable.
    eos_density_range
        The CO₂ densities in kg/m³ spanned by the equation of state's table, ``[low,
        high]`` with ``0 < low < high`` (default ``[1, 1200]``).
    temperature
        ``"deck"`` takes each cell's temperature from the deck (a temperature-depth table,
        or a thermal run's restart); unset (default) uses ``temperature_c`` everywhere.
    temperature_range
        With ``temperature = "deck"``, the temperatures in °C ``[low, high]``, within
        0-100, over which CO₂'s table is built; taken from the deck's depth table when
        unset, but a thermal run must give it.
    dissolved_co2
        ``"lebedev2014"`` corrects the brine's velocity in cells with free CO₂ for the CO₂
        dissolved in it (full saturation only; CO₂ only); ``"none"`` (default) does not.
    dissolved_co2_min_sg
        The gas saturation above which a cell's brine is corrected for dissolved CO₂, in
        ``[0, 1)`` (default 1e-6).
    """

    critical_porosity: float = 0.36
    coordination_number: float = 9.0
    mineral_bulk_modulus: float = 37.0e9
    mineral_shear_modulus: float = 44.0e9
    mineral_density: float = 2650.0
    temperature_c: float = 37.0
    salinity: float = 0.035
    gas: Literal["co2", "methane"] = "co2"
    brie_exponent: float = 5.0
    porosity_ratio_cap: float | None = 0.9
    min_effective_pressure: float = 0.01e6
    oil_reference_density: float | None = None
    extrapolate_gas_correlation: bool = False
    gas_model: Literal["span_wagner", "batzle_wang"] | None = None
    mixing: Literal["brie", "wood", "patchy", "patchiness"] = "brie"
    patchiness: float = 0.5
    eos_density_range: tuple[float, float] = (1.0, 1200.0)
    temperature: Literal["deck"] | None = None
    temperature_range: tuple[float, float] | None = None
    dissolved_co2: Literal["none", "lebedev2014"] = "none"
    dissolved_co2_min_sg: float = 1e-6
    _co2: SpanWagnerCO2 | None = field(default=None, init=False, repr=False, compare=False)
    _co2_field: SpanWagnerCO2Field | None = field(
        default=None, init=False, repr=False, compare=False
    )

    requires_densities = True
    #: The options a state variable may set per member, each with the
    #: transforms that keep it in its range (the first is the default): q in (0, 1) by
    #: the logistic, the rest positive by the exponential (any positive mineral moduli give
    #: a Poisson ratio in (-1, 0.5); only a bulk modulus far below the shear modulus can put
    #: the dry frame above the mineral, which Gassmann refuses). Only options whose setup
    #: checks every transformed value passes (the minerals' Poisson ratio does, for any
    #: positive moduli) and that CO₂'s table does not depend on: not the temperature, nor
    #: the critical porosity (below a cell's porosity the dry frame extrapolates or is
    #: capped).
    uncertain_options: ClassVar[Mapping[str, tuple[str, ...]]] = MappingProxyType(
        {
            "patchiness": ("logit",),
            "coordination_number": ("log",),
            "mineral_bulk_modulus": ("log",),
            "mineral_shear_modulus": ("log",),
            "mineral_density": ("log",),
        }
    )

    def __post_init__(self) -> None:
        """Refuse parameters outside the model's validity."""
        if not 0 < self.critical_porosity < 1:
            raise ValueError(f"critical_porosity must be in (0, 1), got {self.critical_porosity}")
        nu = float(rp.poisson_ratio(self.mineral_bulk_modulus, self.mineral_shear_modulus))
        if not -1 < nu < 0.5:
            raise ValueError(
                f"the mineral's Poisson's ratio is {nu:g} (from its bulk and shear moduli); "
                "Hertz-Mindlin needs it in (-1, 0.5)"
            )
        if self.gas not in GAS_GRAVITY:
            raise ValueError(f"gas must be one of {sorted(GAS_GRAVITY)}, got {self.gas!r}")
        if self.porosity_ratio_cap is not None and not 0 < self.porosity_ratio_cap <= 1:
            raise ValueError("porosity_ratio_cap must be in (0, 1], or None")
        if not 0 <= self.salinity < 0.35:
            raise ValueError(f"salinity is a weight fraction in [0, 0.35), got {self.salinity}")
        if self.min_effective_pressure <= 0:
            raise ValueError("min_effective_pressure must be positive")
        if self.brie_exponent <= 0:
            raise ValueError(f"brie_exponent must be positive, got {self.brie_exponent}")
        low, high = WATER_TEMPERATURE_C
        if not low <= self.temperature_c <= high:
            raise ValueError(
                f"temperature_c = {self.temperature_c:g}: Batzle & Wang's water fits hold for "
                f"{low:g}-{high:g} °C"
            )
        # Literals are not checked by the dataclass: a misspelt value would be ignored.
        for name, value, allowed in (
            ("temperature", self.temperature, (None, "deck")),
            ("dissolved_co2", self.dissolved_co2, ("none", "lebedev2014")),
            ("mixing", self.mixing, ("brie", "wood", "patchy", "patchiness")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")
        if self.temperature_range is not None and self.temperature != "deck":
            raise ValueError("temperature_range is for temperature = 'deck'")
        if self.temperature_range is not None:
            if len(self.temperature_range) != 2:
                raise ValueError("temperature_range must be [low, high] (°C)")
            t_low, t_high = (float(v) for v in self.temperature_range)
            self.temperature_range = (t_low, t_high)
            if not low <= t_low <= t_high <= high:
                raise ValueError(
                    f"temperature_range = {list(self.temperature_range)}: it must be (low, high) "
                    f"within Batzle & Wang's water fits, {low:g}-{high:g} °C"
                )
        if self.dissolved_co2 == "lebedev2014":
            if self.gas != "co2":
                raise ValueError("dissolved_co2 = 'lebedev2014' is for CO₂ (gas = 'co2')")
            ppm_high = LEBEDEV_RANGE["salinity_ppm"][1]
            if not 0.0 <= self.salinity * 1e6 <= ppm_high:
                raise ValueError(
                    f"salinity = {self.salinity:g} ({self.salinity * 1e6:g} ppm) is outside "
                    f"Lebedev et al.'s fitted 0-{ppm_high:g} ppm"
                )
            if not 0.0 <= self.dissolved_co2_min_sg < 1.0:
                raise ValueError("dissolved_co2_min_sg must be in [0, 1)")
        if self.oil_reference_density is not None and not (0 < self.oil_reference_density < 1080.0):
            raise ValueError("oil_reference_density must be in (0, 1080) kg/m3")
        if self.coordination_number <= 0:
            raise ValueError(
                f"coordination_number must be positive, got {self.coordination_number}"
            )
        if not 0.0 <= self.patchiness <= 1.0:
            raise ValueError(f"patchiness must be in [0, 1], got {self.patchiness}")
        if self.gas_model is None:
            # The equation of state for CO₂; extrapolate_gas_correlation alone means the
            # correlation, as before revision 1, so such configurations do not change.
            eos = self.gas == "co2" and not self.extrapolate_gas_correlation
            self.gas_model = "span_wagner" if eos else "batzle_wang"
        if self.gas_model == "span_wagner":
            if self.gas != "co2":
                raise ValueError("gas_model = 'span_wagner' is CO₂'s equation of state")
            if self.extrapolate_gas_correlation:
                raise ValueError("extrapolate_gas_correlation is for gas_model = 'batzle_wang'")
            low, high = self.eos_density_range
            if not 0 < low < high:
                raise ValueError("eos_density_range must be (low, high), 0 < low < high")

    def _eos(self) -> SpanWagnerCO2:
        """Return CO₂'s modulus table, built once (the deck's phases decide whether at all).

        Raises
        ------
        ValueError
            Without CoolProp (the ``co2`` extra), naming it; or if the table fails its
            check.
        """
        if self._co2 is None:
            try:
                low, high = self.eos_density_range
                self._co2 = SpanWagnerCO2(self.temperature_c, (float(low), float(high)))
            except ImportError as error:
                raise ValueError(str(error)) from error
        return self._co2

    def _eos_field(self) -> SpanWagnerCO2Field:
        """Return CO₂'s modulus table over the cells' temperatures, built once.

        Raises
        ------
        ValueError
            Without ``temperature_range`` or CoolProp, naming them; or if the table fails
            its check.
        """
        if self._co2_field is None:
            if self.temperature_range is None:
                raise ValueError(
                    "temperature = 'deck' with CO₂'s equation of state needs temperature_range "
                    "(°C), over which its table is built (a thermal run must give it)"
                )
            try:
                low, high = self.eos_density_range
                self._co2_field = SpanWagnerCO2Field(
                    (float(self.temperature_range[0]), float(self.temperature_range[1])),
                    (float(low), float(high)),
                )
            except ImportError as error:
                raise ValueError(str(error)) from error
        return self._co2_field

    def prepare(self) -> None:
        """Build CO₂'s table over the cells' temperatures now, at setup (``"deck"`` only).

        Raises
        ------
        ValueError
            As :meth:`_eos_field` (no ``temperature_range``, no CoolProp, a failed check).
        """
        if self.temperature == "deck" and self.gas_model == "span_wagner":
            self._eos_field()

    def check_phases(self, phases: Collection[str]) -> None:
        """Refuse, before any member runs, a gas outside Batzle & Wang's correlation.

        The simulator calls it with the deck's phases. The pseudo-reduced temperature
        depends only on ``gas`` and ``temperature_c``, so this is a configuration error:
        with the defaults (CO₂ at 37 °C) every run with gas is refused until
        ``extrapolate_gas_correlation = true`` is set, as the reference's behaviour needs.
        """
        if self.mixing in ("patchy", "patchiness") and {"oil", "water", "gas"} <= set(phases):
            raise ValueError(
                f"mixing = {self.mixing!r} takes two fluid phases (brine and CO₂); the deck has "
                "oil, water and gas: use 'wood' or 'brie'"
            )
        if "gas" not in phases or self.extrapolate_gas_correlation:
            return
        if self.gas_model == "span_wagner":
            # Built at setup, before any member runs.
            if self.temperature == "deck":
                self._eos_field()  # refuses a missing temperature_range
            else:
                self._eos()
            return
        if self.temperature == "deck":
            return  # each cell's temperature is checked against the correlation as it runs
        t_pr, _ = self._gas_pseudo_reduced(np.zeros(1))
        if not GAS_T_PR[0] <= t_pr <= GAS_T_PR[1]:
            raise ValueError(
                f"{self.gas} at {self.temperature_c:g} °C has a pseudo-reduced temperature of "
                f"{t_pr:.3f}, outside Batzle & Wang's gas correlation ({GAS_T_PR[0]}-"
                f"{GAS_T_PR[1]}); set extrapolate_gas_correlation = true to use it anyway (the "
                "reference does)"
            )

    def check_temperature(
        self, deck_c: float | None, varies: bool, phases: Collection[str] | None
    ) -> list[str]:
        """Refuse a CO₂ temperature the deck contradicts; return what cannot be checked.

        Near CO₂'s critical point half a degree changes its density by up to 35 %, so with
        the equation of state ``temperature_c`` must be the deck's reservoir temperature
        (``deck_c``) within 0.1 °C. A temperature varying with depth or in time is refused
        (the table is at one temperature), or only noted when ``phases`` is unknown (a
        templated RUNSPEC).

        Raises
        ------
        ValueError
            If the deck varies the temperature, or its ``RTEMP`` differs.
        """
        if self.gas_model != "span_wagner" or (phases is not None and "gas" not in phases):
            return []
        if varies and self.temperature == "deck":
            return []  # each cell's own temperature
        if varies and phases is None:
            # A templated RUNSPEC: the deck may have no gas, and then its temperature
            # does not matter here.
            return [
                "soft_sand: the deck varies the temperature (RTEMPVD, TEMPVD or a thermal "
                "run), which CO₂'s equation of state cannot follow; its templated RUNSPEC "
                "cannot tell whether it has a gas phase, so this is not refused"
            ]
        if varies:
            raise ValueError(
                "the deck varies the temperature (RTEMPVD, TEMPVD or a thermal run); set "
                "temperature = 'deck' for each cell's own (temperature_c is one for all)"
            )
        if deck_c is None:
            return [
                f"soft_sand: the deck's text states no readable reservoir temperature (RTEMP "
                "or RTEMPVD defaulted or templated, a templated RUNSPEC, or none outside "
                "CO2STORE), so CO₂'s modulus takes "
                f"temperature_c = {self.temperature_c:g} °C unchecked; near CO₂'s critical "
                "point half a degree matters"
            ]
        if abs(deck_c - self.temperature_c) > 0.1:
            raise ValueError(
                f"temperature_c = {self.temperature_c:g} °C, but the deck's reservoir "
                f"temperature is {deck_c:.4g} °C (RTEMP, RTEMPVD, or in CO2STORE without "
                "either OPM Flow's default of 60 °F): CO₂'s modulus must be at the "
                f"simulator's temperature; set temperature_c = {deck_c:.4g}"
            )
        return []

    def _gas_pseudo_reduced(
        self, pressure_mpa: Array, temperature_c: float | Array | None = None
    ) -> tuple[float | Array, Array]:
        gravity = GAS_GRAVITY[self.gas]
        t = self.temperature_c if temperature_c is None else temperature_c
        t_pr = (t + 273.15) / (94.72 + 170.75 * gravity)
        return t_pr, pressure_mpa / (4.892 - 0.4048 * gravity)

    def fluid(self, c: _Cells) -> tuple[Array, Array, dict[str, int]]:
        """Return the pore fluid's density (kg/m³) and bulk modulus (MPa) per cell.

        With ``mixing`` ``"brie"`` (oil and water by Wood, then Brie with the gas) or
        ``"wood"`` (every phase by Wood); and the counts to report (CO₂'s snapped and
        inconsistent cells).
        """
        s, rho, k, counts = self._phase_moduli(c)
        density = sum((s[p] * rho[p] for p in c.saturations), np.zeros(c.porosity.size))
        if self.mixing == "wood":
            return density, rp.wood([s[p] for p in PHASES], [k[p] for p in PHASES]), counts
        liquid = s["oil"] + s["water"]
        # Wood's average of the liquids, by their shares of the liquid; any value serves
        # where there is no liquid (Brie's weight of it is then 0).
        share = np.divide(s["water"], liquid, out=np.ones(liquid.size), where=liquid > 0)
        k_liquid = rp.wood([share, 1.0 - share], [k["water"], k["oil"]])
        bulk = rp.brie(liquid, k_liquid, k["gas"], self.brie_exponent)
        return density, bulk, counts

    def _phase_moduli(
        self, c: _Cells
    ) -> tuple[dict[Phase, Array], dict[Phase, Array], dict[Phase, Array], dict[str, int]]:
        """Return each phase's saturation, density (kg/m³) and bulk modulus (MPa).

        Every phase of :data:`PHASES`; an absent phase has saturation 0, a placeholder
        density and a modulus of 1 (it weighs nothing). And the counts to report.
        """
        assert c.densities is not None
        p, n = c.pressure_mpa, c.porosity.size
        t: float | Array = self.temperature_c
        if self.temperature == "deck" and c.temperature is None:
            raise ValueError(
                "temperature = 'deck': the simulator gave no temperature per cell (the deck's "
                "depth table or thermal TEMP)"
            )
        if self.temperature == "deck" and c.temperature is not None:
            t = c.temperature
            low, high = WATER_TEMPERATURE_C
            _refuse(
                (t < low) | (t > high), "temperature (°C)", t,
                f"Batzle & Wang's water fits hold for {low:g}-{high:g} °C",
            )  # fmt: skip
            if self.temperature_range is not None:
                t_low, t_high = self.temperature_range
                _refuse(
                    (t < t_low) | (t > t_high), "temperature (°C)", t,
                    f"outside temperature_range {t_low:g}-{t_high:g} °C",
                )  # fmt: skip
        zero = np.zeros(n)
        s = {k: c.saturations.get(k, zero) for k in PHASES}
        rho = {k: c.densities.get(k, np.full(n, _PLACEHOLDER_DENSITY[k])) for k in PHASES}
        has_water, has_oil, has_gas = s["water"] > 0, s["oil"] > 0, s["gas"] > 0
        _refuse(
            (has_water | has_oil) & (p > WATER_PRESSURE_MPA), "pressure (MPa)", p,
            "Batzle & Wang's liquid fits hold to 100 MPa",
        )  # fmt: skip
        v_brine = rp.brine_velocity(t, p, self.salinity)
        dissolved: dict[str, int] = {}
        if self.dissolved_co2 == "lebedev2014":
            v_brine, dissolved = self._dissolved_co2(v_brine, t, p, s["gas"])
        k_water = rho["water"] * v_brine**2 / _MPA
        rho0 = rho["oil"] if self.oil_reference_density is None else self.oil_reference_density
        rho0 = np.broadcast_to(rho0, (n,))
        _refuse(
            has_oil & (rho0 >= 1080.0), "oil reference density", rho0,
            "Batzle & Wang's dead-oil velocity holds below 1080 kg/m3",
        )  # fmt: skip
        rho0 = np.where(has_oil, rho0, _PLACEHOLDER_DENSITY["oil"])
        k_oil = rho["oil"] * rp.dead_oil_velocity(t, p, rho0) ** 2 / _MPA
        k_gas = np.ones(n)  # where there is no gas, Brie's weight of the liquid is 1
        counts: dict[str, int] = {}
        if has_gas.any() and self.gas_model == "span_wagner" and not np.isscalar(t):
            assert isinstance(t, np.ndarray)
            co2 = self._eos_field()
            k_density, snapped, band = co2.modulus(t[has_gas], rho["gas"][has_gas])
            k_pressure = co2.modulus_at_pressure(p[has_gas] * _MPA, t[has_gas])
            agree = np.abs(k_density - k_pressure) <= CO2_CONSISTENCY * k_density
            counts = {
                "co2_cells": int(has_gas.sum()),
                "co2_two_phase_snapped": int(snapped.sum()),
                "co2_pressure_inconsistent": int((~agree).sum()),
                "co2_critical_band": int(band.sum()),
            }
            k_gas = np.ones(n)
            k_gas[has_gas] = k_density / _MPA
        elif has_gas.any() and self.gas_model == "span_wagner":
            eos = self._eos()
            # Only the cells with CO₂: an absent phase's density may be anything.
            k_density, snapped = eos.modulus(rho["gas"][has_gas])
            # The reported pressure (the reference phase's): what the 5 % count measures.
            k_pressure = eos.modulus_at_pressure(p[has_gas] * _MPA)
            # Not "differ by more": a pressure CoolProp has no state at (NaN) counts too.
            agree = np.abs(k_density - k_pressure) <= CO2_CONSISTENCY * k_density
            counts = {
                "co2_cells": int(has_gas.sum()),
                "co2_two_phase_snapped": int(snapped.sum()),
                "co2_pressure_inconsistent": int((~agree).sum()),
            }
            k_gas = np.ones(n)
            k_gas[has_gas] = k_density / _MPA
        elif has_gas.any():
            self.check_phases(c.saturations)  # the simulator checks it at setup
            t_pr, p_pr = self._gas_pseudo_reduced(p, t)
            if not self.extrapolate_gas_correlation and not np.isscalar(t_pr):
                t_pr_cells = np.broadcast_to(np.asarray(t_pr, dtype=np.float64), (n,))
                _refuse(
                    has_gas & ((t_pr_cells < GAS_T_PR[0]) | (t_pr_cells > GAS_T_PR[1])),
                    "gas pseudo-reduced temperature", t_pr_cells,
                    f"Batzle & Wang's gas correlation holds for {GAS_T_PR[0]}-{GAS_T_PR[1]}; set "
                    "extrapolate_gas_correlation = true to use it anyway",
                )  # fmt: skip
            if not self.extrapolate_gas_correlation:
                _refuse(
                    has_gas & (p_pr > GAS_P_PR_MAX), "gas pseudo-reduced pressure", p_pr,
                    f"Batzle & Wang's gas correlation holds to {GAS_P_PR_MAX:g}",
                )  # fmt: skip
            k = rp.gas_bulk_modulus(t, p, GAS_GRAVITY[self.gas])
            _refuse(
                has_gas & ~(np.isfinite(k) & (k > 0)), "gas bulk modulus (MPa)", k,
                "Batzle & Wang's gas correlation gives no physical modulus here",
            )  # fmt: skip
            k_gas = np.where(has_gas, k, 1.0)
        moduli: dict[Phase, Array] = {"water": k_water, "oil": k_oil, "gas": k_gas}
        return s, rho, moduli, {**counts, **dissolved}

    def _dissolved_co2(
        self, v_brine: Array, t: float | Array, p: Array, s_gas: Array
    ) -> tuple[Array, dict[str, int]]:
        """Return the brine velocity (m/s) with Lebedev et al.'s (2014) correction, and counts.

        Only cells with free CO₂ (full saturation) inside the paper's fitted pressures and
        temperatures are corrected; the others with free CO₂ are counted, never
        extrapolated.
        """
        t_cells = np.broadcast_to(np.asarray(t, dtype=np.float64), p.shape)
        (p_low, p_high), (t_low, t_high) = (
            LEBEDEV_RANGE["pressure_mpa"],
            LEBEDEV_RANGE["temperature_c"],
        )
        free = s_gas > self.dissolved_co2_min_sg
        fitted = (p >= p_low) & (p <= p_high) & (t_cells >= t_low) & (t_cells <= t_high)
        corrected = free & fitted
        ppm = self.salinity * 1e6
        shift = 20.42 - 0.08377 * t_cells - 0.18 * p - 0.0004173 * ppm
        out: Array = np.where(corrected, v_brine + shift, v_brine)
        return out, {
            "co2_dissolved_corrected": int(corrected.sum()),
            "co2_dissolved_out_of_range": int((free & ~fitted).sum()),
        }

    def elastic(self, state: FlowState) -> ElasticProperties:
        """Return each cell's elastic properties; see :class:`PetroElasticModel`."""
        # With the cap, φ/φc never exceeds it and Gassmann uses the real porosity, so the
        # model holds up to φ < 1, as the reference computes; without it, φ < φc.
        limit = 1.0 if self.porosity_ratio_cap is not None else self.critical_porosity
        c = _cells(state, limit, self.min_effective_pressure, True)
        if self.mixing in ("patchy", "patchiness"):
            e, counts = self._patchy(c)
        else:
            fluid_density, fluid_bulk, counts = self.fluid(c)
            e = self._rock(c.porosity, c.effective_mpa, fluid_density, fluid_bulk)
        return replace(e, clamped_cells=c.clamped, counts=counts)

    def _patchy(self, c: _Cells) -> tuple[ElasticProperties, dict[str, int]]:
        """Return Gassmann-Hill's patchy rock, or its blend with Wood's (``patchiness``).

        Each fluid saturates the rock alone (Gassmann), and the P-wave moduli
        M = K + 4/3 G are averaged harmonically by saturation (Mavko et al. 2009);
        ``patchiness`` takes M = (1 - q) M_wood + q M_patchy, within both bounds for
        every q. The shear modulus is the dry rock's either way.
        """
        # A guard: check_phases refuses three phases at setup, from the deck's RUNSPEC.
        if sum(bool(np.any(sat > 0)) for sat in c.saturations.values()) > 2:
            raise ValueError(
                f"mixing = {self.mixing!r} takes two fluid phases (brine and CO₂); with oil, "
                "brine and gas use 'wood' or 'brie'"
            )
        s, rho, k, counts = self._phase_moduli(c)
        phi = c.porosity
        k_min = self.mineral_bulk_modulus / _MPA
        dry_bulk, dry_shear = self._dry(phi, c.effective_mpa)
        shear = 4.0 * dry_shear / 3.0
        p_moduli = [rp.gassmann(dry_bulk, k_min, k[p], phi) + shear for p in PHASES]
        patchy = rp.wood([s[p] for p in PHASES], p_moduli)  # the harmonic mean of M
        q = 1.0 if self.mixing == "patchy" else self.patchiness
        wood_fluid = rp.wood([s[p] for p in PHASES], [k[p] for p in PHASES])
        wood = rp.gassmann(dry_bulk, k_min, wood_fluid, phi) + shear
        bulk = (1.0 - q) * wood + q * patchy - shear
        density = sum((s[p] * rho[p] for p in c.saturations), np.zeros(phi.size))
        e = _saturated_moduli(phi, dry_bulk, dry_shear, bulk, self.mineral_density, density)
        return e, counts

    def _dry(self, phi: Array, effective_mpa: Array) -> tuple[Array, Array]:
        """Return the dry rock's bulk and shear moduli (MPa): Hertz-Mindlin, soft sand."""
        k, g = self.mineral_bulk_modulus / _MPA, self.mineral_shear_modulus / _MPA
        hm_bulk, hm_shear = rp.hertz_mindlin(
            effective_mpa, k, g, self.coordination_number, self.critical_porosity
        )
        ratio = phi / self.critical_porosity
        if self.porosity_ratio_cap is not None:
            ratio = np.minimum(ratio, self.porosity_ratio_cap)
        dry_bulk, dry_shear = rp.soft_sand_dry_moduli(ratio, hm_bulk, hm_shear, k, g)
        return dry_bulk, dry_shear

    def _rock(
        self, phi: Array, effective_mpa: Array, fluid_density: Array, fluid_bulk: Array
    ) -> ElasticProperties:
        """Return the properties for a given pore fluid: dry rock, then Gassmann.

        ``phi``, ``effective_mpa`` (already floored), ``fluid_density`` (kg/m³) and
        ``fluid_bulk`` (MPa) are ``(n_cell,)``. The parity tests call it with the
        reference's own fluid.
        """
        dry_bulk, dry_shear = self._dry(phi, effective_mpa)
        return _saturated(
            phi, dry_bulk, dry_shear, self.mineral_bulk_modulus / _MPA, self.mineral_density,
            fluid_bulk, fluid_density,
        )  # fmt: skip
