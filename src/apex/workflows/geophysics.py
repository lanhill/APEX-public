"""The geophysical forward model of a deck run: flow state to field data.

:class:`GeophysicsModel` is the :class:`~apex.core.simulation.CaseFieldModel` that
:class:`~apex.simulation.geophysics.GeophysicsSimulator` asks for its field keys. For each
vintage it reads the run's state (:class:`~apex.simulation.geophysics.CaseState`); for AVO
it runs the petro-elastic model over the active cells, builds each trace's column from the
grid, and synthesizes and reduces the traces with the kernels of
:mod:`apex.algorithms.geophysics`; for gravity and displacement it takes the active cells'
change of fluid mass or volume since the baseline and its response at the stations.

Keys: an AVO datum ``[[geophysics.data]]`` with ``datatype = "avo"`` gives a field
``DataKey(vintage, "avo_<angle>", i)`` per angle and vintage (:func:`avo_datatype`).
``i`` runs over the traces, the grid's columns with an active cell kept by ``step_x`` and
``step_y``, in Eclipse (F) order, i fastest; with ``form = "traces"`` it is
``trace * n_samples + sample`` over the samples of ``time_window``. The traces are those
of the configured grid (``grid_file``), fixed for every member: a member whose own grid
has other active columns is refused, naming them.

A gravity or displacement datum gives ``DataKey(vintage, datatype, i)``, ``i`` over its
stations: Δg (µGal) or the vertical displacement (cm), both positive down, from the
baseline to the vintage (:class:`GravitySurvey`, :class:`DisplacementSurvey`). The cells
are ``grid_file``'s active cells, their centres and volumes fixed for every member: a
member with other active cells is refused.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_avo.get_avo_result``,
             L1069-1520: the PEM at each vintage, ``calc_velocities``, the traces, the
             mask of active columns, ``step_x``/``step_y`` and the region of interest,
             and the fixed baseline's differences)
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0)

Differences from the reference (SimulatorWrap ``flow_avo``):

- Restarts are read unified (by SEQNUM block) or separate, and a missing date is refused
  with its cause; the reference reads separate ``.X`` files only, and asserts.
- Pressures and densities are converted from the deck's units (INIT's unit system), not by
  a ``press_conv`` factor.
- The porosity is INIT's PORO; the reference's ``compaction`` (PORV_RC) is not ported yet.
- Column geometry (each cell's top and base) comes from the run's EGRID, not from DZ and
  TOPS files.
- NTG is not read: neither ported petro-elastic model uses it (the reference passes it on
  unused).
- ``step_x`` thins the grid's I axis and ``step_y`` its J axis; the reference's
  ``step_x`` thins its first array axis, which is J (``apex pet-import`` swaps them).


Differences from the reference (SimulatorWrap ``flow_grav``, ``flow_seafloor_disp``,
d29dc57):

- A cell's centre is the mean of its eight corners; the reference takes the mean depth
  and interpolates x and y along the cell's mean pillar (the same for vertical pillars).
- Every active cell counts: the reference's ``filter_rporv`` (dropping the cells above
  RPORV's 90th percentile, an aquifer heuristic) is not ported.
- The displacement's volume change is ``c_m V ΔP`` with ``V`` the bulk volume by default
  (Geertsma's), or the baseline's pore volume (``volume = "pore"``, the reference's).
"""

import copy
import dataclasses
import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geophysics import (
    AVOAttribute,
    Columns,
    ConstantOverburden,
    DisplacementKernel,
    FlowState,
    GeophysicsModelError,
    LayeredOverburden,
    PetroElasticModel,
    PPReflectivity,
    RickerWavelet,
    SampledWavelet,
    SeismicModel,
    TimeAxis,
    point_mass_gravity,
    prism_gravity,
    regular_stations,
    time_window,
)
from apex.core.errors import ConfigError, SimulationError
from apex.domain.config import (
    AVOOptions,
    ConstantOverburdenSection,
    DisplacementOptions,
    GeophysicsDataSection,
    GeophysicsSection,
    GravityOptions,
    OverburdenPressureFile,
    RegionAroundWellsSection,
    RegionFileSection,
    RickerWaveletSection,
    StationFileSection,
    avo_datatype,
)
from apex.domain.observations import DataKey
from apex.simulation.geophysics import CaseState, Grid, column_centres, grid_unit_stated

__all__ = [
    "PARAMETER_TRANSFORMS",
    "AVOSurvey",
    "CellPressure",
    "DisplacementSurvey",
    "GeophysicsModel",
    "GravitySurvey",
    "LithostaticPressure",
    "PemParameter",
    "build_geophysics_model",
]

logger = logging.getLogger(__name__)

type Array = NDArray[np.float64]

#: µGal and cm in SI: the gravity and displacement data's units, the reference's.
_MICROGAL = 1e-8
_CENTIMETRE = 1e-2
#: The most station-cell pairs a displacement datum may have: its matrix is 40 MB of
#: float64, copied with the simulator to every member's task (a per-worker cache would
#: lift this). Gravity keeps no matrix (its kernels work in bounded blocks): its time per
#: member is stated above _GRAVITY_NOTE.
_MAX_PAIRS = 5_000_000
_GRAVITY_NOTE = 50_000_000
#: Above this many MB, the plan states the displacement matrices' size.
_MATRIX_NOTE_MB = 8.0

#: Pressure units of an overburden file, in Pa.
_PRESSURE_UNITS = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "psi": 6894.757293168361}


@dataclass(frozen=True)
class LithostaticPressure:
    """``surface + gradient * depth`` (Pa) at each active cell's centre depth (m)."""

    gradient: float
    surface: float

    def values(self, state: CaseState) -> Array:
        """Return the overburden pressure of each active cell (Pa)."""
        out: Array = self.surface + self.gradient * state.depth
        return out


@dataclass(frozen=True, eq=False)
class CellPressure:
    """Given overburden pressures (Pa) of every grid cell, ``(n_cell,)`` in F order.

    NaN where the file gave none (cells inactive in ``grid_file``, for a file of its
    active cells).
    """

    pascal: Array

    @classmethod
    def of(cls, values: Array, reference: Grid) -> "CellPressure":
        """Lay a file's values on the grid: one per cell, or one per ``reference``'s active.

        Raises
        ------
        ConfigError
            If the file's size is neither the grid's cell count nor its active count.
        """
        actnum = reference.actnum.ravel(order="F")
        if values.size == actnum.size:
            return cls(values)
        if values.size == int(actnum.sum()):
            full = np.full(actnum.size, np.nan)
            full[actnum] = values
            return cls(full)
        raise ConfigError(
            f"[geophysics] overburden_pressure: the file has {values.size} values; grid_file "
            f"has {actnum.size} cells, {int(actnum.sum())} active"
        )

    def values(self, state: CaseState) -> Array:
        """Return the overburden pressure of each of the case's active cells (Pa).

        Raises
        ------
        SimulationError
            If the case has active cells the file gives no value for (inactive in
            ``grid_file``).
        """
        out: Array = self.pascal[state.grid.actnum.ravel(order="F")]
        missing = int(np.isnan(out).sum())
        if missing:
            raise SimulationError(
                f"{state.case.name}: {missing} active cell(s) are inactive in grid_file, so "
                "the overburden file has no pressure for them"
            )
        return out


type OverburdenPressure = LithostaticPressure | CellPressure


@dataclass(frozen=True)
class AVOSurvey:
    """One ``[[geophysics.data]]`` AVO datum: its traces, synthesis and reduction.

    Attributes
    ----------
    datatype
        The datatype prefix; the fields are ``<datatype>_<angle>``.
    vintages
        The survey dates.
    model
        The synthesis (angles, wavelet, overburden, times).
    attribute
        A map attribute, or ``trace``.
    window
        ``(n_time,)``: the samples the attribute reduces, or a trace keeps.
    columns
        ``(n_trace, 2)``: each trace's column ``(i, j)``, 0-based, in F order.
    baseline
        For ``four_d = "differences"`` or a time shift: the baseline survey, whose values
        every vintage's are taken from.
    rolling
        ``baseline_mode = "rolling"``: each vintage's values are taken from the previous
        vintage's instead (the first vintage's from the baseline's), as SimulatorWrap's.
    """

    datatype: str
    vintages: tuple[dt.date, ...]
    model: SeismicModel
    attribute: AVOAttribute
    window: NDArray[np.bool_]
    columns: NDArray[np.intp]
    baseline: dt.date | None = None
    rolling: bool = False
    #: The kept columns' spacing (step_x, step_y): the traces sit on a lattice of every
    #: step_x-th and step_y-th column (:meth:`GeophysicsModel.wavelet_layout`).
    steps: tuple[int, int] = (1, 1)

    @property
    def n_trace(self) -> int:
        """The number of traces."""
        return int(self.columns.shape[0])

    @property
    def field_size(self) -> int:
        """The number of values of one field (an angle at a vintage)."""
        return self.n_trace * (1 if self.attribute.is_map else int(self.window.sum()))

    @property
    def extent(self) -> str:
        """What a field's values run over, for messages."""
        return f"{self.n_trace} traces"

    def fields(self) -> dict[str, int]:
        """Return each field datatype and its angle's index."""
        return {avo_datatype(self.datatype, a): n for n, a in enumerate(self.model.angles_deg)}

    def values(self, grid: Grid, vp: Array, vs: Array, rho: Array) -> Array:
        """Return the fields ``(n_angle, field_size)`` from the grid's elastic properties.

        ``vp``, ``vs`` and ``rho`` are ``(n_x, n_y, n_z)``; inactive cells' values are
        not used.
        """
        i, j = self.columns[:, 0], self.columns[:, 1]
        active = grid.actnum[i, j, :]
        # Inactive cells of corner-point grids may be collapsed or crossed (base above
        # top): they take no time then. An active one is checked by Columns.of.
        thickness = grid.bottom[i, j, :] - grid.top[i, j, :]
        thickness = np.where(active, thickness, np.maximum(thickness, 0.0))
        columns = Columns.of(
            top_depth=grid.top[i, j, 0],
            thickness=thickness,
            vp=vp[i, j, :],
            vs=vs[i, j, :],
            rho=rho[i, j, :],
            active=active,
        )
        if getattr(self.attribute, "uses_times", False):
            # The time to the base, the same at every angle (vertical incidence); its
            # change from the baseline is the time shift.
            base = self.model.base_time(columns)
            return np.ascontiguousarray(np.broadcast_to(base, (len(self.fields()), base.size)))
        traces = self.model.traces(columns)  # (n_trace, n_angle, n_time)
        if self.attribute.is_map:
            out: Array = self.attribute(traces, self.window).T  # (n_angle, n_trace)
            return np.ascontiguousarray(out)
        kept = traces[:, :, self.window]  # (n_trace, n_angle, n_kept)
        return np.ascontiguousarray(np.moveaxis(kept, 1, 0).reshape(len(self.fields()), -1))


class CaseQuantities:
    """One case's cell quantities at its dates (SI, over the active cells), each read once."""

    def __init__(self, state: CaseState, n_active: int) -> None:
        self.state = state
        self.n_active = n_active
        self._mass: dict[dt.date, Array] = {}
        self._pore: dict[dt.date, Array] = {}
        self._pressure: dict[dt.date, Array] = {}

    def fluid_mass(self, date: dt.date) -> Array:
        """Return each active cell's fluid mass (kg): Σ_phases density * saturation * RPORV."""
        if date not in self._mass:
            pressure, saturations, densities = self.state.flow_state(date, densities=True)
            pore = self.pore_volume(date)
            if date not in self._pressure:
                self._check(pressure, "PRESSURE", date)
                self._pressure[date] = pressure
            self._mass[date] = pore * sum(
                (densities[p] * saturations[p] for p in saturations), np.zeros_like(pore)
            )
            self._check(self._mass[date], "the restart's arrays", date)
        return self._mass[date]

    def _check(self, values: Array, what: str, date: dt.date) -> None:
        """Refuse cell vectors that are not one per active cell of grid_file."""
        if values.shape != (self.n_active,):
            raise SimulationError(
                f"{self.state.case.name} at {date.isoformat()}: {what} have {values.size} "
                f"values, grid_file {self.n_active} active cells (dual porosity and local "
                "grids are not supported for gravity and displacement)"
            )

    def pore_volume(self, date: dt.date) -> Array:
        """Return each active cell's pore volume at reservoir conditions (m³)."""
        if date not in self._pore:
            self._pore[date] = self.state.pore_volume(date)
            self._check(self._pore[date], "RPORV", date)
        return self._pore[date]

    def pressure(self, date: dt.date) -> Array:
        """Return each active cell's pressure (Pa)."""
        if date not in self._pressure:
            self._pressure[date] = self.state.flow_state(date, densities=False)[0]
            self._check(self._pressure[date], "PRESSURE", date)
        return self._pressure[date]


@dataclass(frozen=True)
class _StationSurvey:
    """A datum at stations: one field per vintage, the change since the baseline."""

    datatype: str
    vintages: tuple[dt.date, ...]
    baseline: dt.date
    stations: Array  # (n_station, 3)

    @property
    def field_size(self) -> int:
        """The number of values of one field (the stations)."""
        return int(self.stations.shape[0])

    @property
    def extent(self) -> str:
        """What a field's values run over, for messages."""
        return f"{self.field_size} stations"

    def fields(self) -> dict[str, int]:
        """Return the one field datatype."""
        return {self.datatype: 0}


@dataclass(frozen=True)
class GravitySurvey(_StationSurvey):
    """A ``kind = "gravity"`` datum: Δg (µGal, positive down) at the stations.

    Attributes
    ----------
    centres
        ``(n_active, 3)``: the active cells' centres (m), where ``model = "point_mass"``
        puts their masses.
    bounds
        ``(n_active, 6)``: the active cells' prisms, ``x1, x2, y1, y2, z1, z2`` (m), for
        ``model = "prism"`` (else ``None``): their corners' horizontal extent, top and base.
    gravitational_constant
        m³ kg⁻¹ s⁻².
    """

    centres: Array
    bounds: Array | None
    gravitational_constant: float
    #: The closest station-centre distance over that cell's diagonal, for point masses.
    closest: float = float("inf")

    def values(self, quantities: CaseQuantities, date: dt.date) -> Array:
        """Return Δg ``(n_station,)`` (µGal) from the baseline to ``date``."""
        dm = quantities.fluid_mass(date) - quantities.fluid_mass(self.baseline)
        try:
            if self.bounds is None:
                g = point_mass_gravity(self.stations, self.centres, dm, self.gravitational_constant)
            else:
                # Each cell's mass change spread uniformly over its prism (its mass kept).
                b = self.bounds
                volume = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) * (b[:, 5] - b[:, 4])
                g = prism_gravity(self.stations, b, dm / volume, self.gravitational_constant)
        except ValueError as error:  # the stations and cells are every member's
            raise ConfigError(f"[geophysics] {self.datatype}: {error}") from error
        out: Array = g / _MICROGAL
        return out


@dataclass(frozen=True)
class DisplacementSurvey(_StationSurvey):
    """A ``kind = "seafloor_displacement"`` datum: u_z (cm, positive down) at the stations.

    Attributes
    ----------
    matrix
        ``(n_station, n_active)``: u_z (m) per m³ of each active cell's volume decrease,
        the kernel's.
    volume_change
        ``"pressure"`` (``c_m V ΔP``) or ``"pore_volume"`` (RPORV's decrease).
    compressibility
        c_m (1/Pa), for ``"pressure"``.
    bulk
        ``(n_active,)``: the cells' bulk volumes (m³), for ``"pressure"`` over the bulk
        volume; ``None`` for the baseline's pore volume.
    """

    matrix: Array
    volume_change: str
    compressibility: float | None
    bulk: Array | None

    def values(self, quantities: CaseQuantities, date: dt.date) -> Array:
        """Return u_z ``(n_station,)`` (cm) from the baseline to ``date``."""
        if self.volume_change == "pore_volume":
            decrease = quantities.pore_volume(self.baseline) - quantities.pore_volume(date)
        else:
            assert self.compressibility is not None
            volume = quantities.pore_volume(self.baseline) if self.bulk is None else self.bulk
            drop = quantities.pressure(self.baseline) - quantities.pressure(date)
            decrease = self.compressibility * volume * drop
        out: Array = self.matrix @ decrease / _CENTIMETRE
        return out


type Survey = AVOSurvey | GravitySurvey | DisplacementSurvey


#: How an uncertain petro-elastic parameter's state value becomes the option's value.
PARAMETER_TRANSFORMS = ("none", "logit", "log")


@dataclass(frozen=True)
class PemParameter:
    """A petro-elastic option that a state variable sets.

    Attributes
    ----------
    option
        The option of the petro-elastic model, e.g. ``patchiness``.
    state
        The state variable (one value) the methods update with the reservoir model.
    transform
        ``"logit"`` (the option is ``1 / (1 + e^-x)``, in (0, 1)), ``"log"`` (``e^x``) or
        ``"none"``.
    value
        The option's value for a member without the variable (the truth, which has no
        true value of it); ``None``: such a member is refused.
    """

    option: str
    state: str
    transform: str = "none"
    value: float | None = None

    def apply(self, x: float) -> float:
        """Return the option's value for the state value ``x``."""
        from scipy.special import expit

        if self.transform == "logit":
            return float(expit(x))  # no overflow for large |x|
        if self.transform == "log":
            with np.errstate(over="ignore"):
                return float(np.exp(x))
        return float(x)


#: The petro-elastic counts that warn when any cell has them.
_NOTED_COUNTS = (
    "co2_two_phase_snapped",
    "co2_pressure_inconsistent",
    "co2_dissolved_out_of_range",
)


@dataclass(frozen=True)
class CellTemperature:
    """Each cell's temperature (°C) from the deck, for ``temperature = "deck"``.

    Parameters
    ----------
    tables
        A depth table per equilibration region (``EQLNUM`` 1, 2, ...), each
        ``((depth_m, temperature_c), ...)`` increasing in depth, from the deck's
        ``RTEMPVD``/``TEMPVD``: a cell takes its region's table at its INIT ``DEPTH``,
        linearly, the end values beyond the table's depths. ``None`` for a thermal run.
    thermal
        A thermal run: each cell's ``TEMP`` from the restart at the vintage.
    """

    tables: tuple[tuple[tuple[float, float], ...], ...] | None = None
    thermal: bool = False

    def __post_init__(self) -> None:
        """Refuse a source that is neither, or both."""
        if (self.tables is None) == (not self.thermal):
            raise ValueError("CellTemperature takes depth tables or a thermal run, one of them")

    def cells(self, state: CaseState, date: dt.date) -> Array:
        """Return each active cell's temperature (°C) in ``state`` at ``date``.

        Raises
        ------
        ConfigError
            If a thermal run's restart lacks ``TEMP``, or INIT's ``EQLNUM`` names a region
            without a table: every member would fail so.
        """
        if self.thermal:
            arrays = state.restart(date, ["TEMP"])
            if "TEMP" not in arrays:
                raise ConfigError(
                    f"{state.case.name} at {date.isoformat()}: the thermal run's restart has no "
                    "TEMP (OPM writes it only when energy is on: THERMAL or TEMP in RUNSPEC)"
                )
            temp = arrays["TEMP"]
            out: Array = (temp - 32.0) / 1.8 if state.units.name == "FIELD" else temp
            return out
        assert self.tables is not None
        depth = state.depth
        # OPM writes EQLNUM to INIT, all 1 where the deck leaves it out (measured on OPM
        # Flow 2026.04); without it every cell is in region 1, as a defaulted EQLNUM is.
        if len(self.tables) > 1 and "EQLNUM" in state.init:
            region = state.static("EQLNUM").astype(np.intp)
        else:
            region = np.ones(depth.size, dtype=np.intp)
        bad = (region < 1) | (region > len(self.tables))
        if bad.any():
            raise ConfigError(
                f"{state.case.name}: EQLNUM {int(region[bad][0])} has no temperature table "
                f"(the deck gives {len(self.tables)})"
            )
        cells = np.empty(depth.size)
        for r, table in enumerate(self.tables, start=1):
            at = region == r
            d, t = (np.array(v, dtype=np.float64) for v in zip(*table, strict=True))
            cells[at] = np.interp(depth[at], d, t)
        return cells


@dataclass(eq=False)
class GeophysicsModel:
    """The forward model of ``[geophysics]`` (a :class:`CaseFieldModel`).

    Parameters
    ----------
    pem
        A model of the ``petro_elastic_models`` registry, for AVO data (else ``None``).
    overburden
        The overburden pressure, for AVO data (else ``None``).
    surveys
        The data: AVO over the traces of ``reference``, gravity and displacement over its
        active cells.
    reference
        The configured grid (``grid_file``): its shape and active columns (all its active
        cells, with gravity or displacement data) are every member's.
    brine_as_oil
        The deck is OPM Flow's CO2STORE in its oil-gas form, whose "oil" is brine: the
        oil phase's saturation and density are the water's for the petro-elastic model.
        ``None`` when the deck's text cannot tell (a templated RUNSPEC): an oil-gas run
        with a CO₂ petro-elastic model is then refused.
    """

    pem: PetroElasticModel | None
    overburden: OverburdenPressure | None
    surveys: tuple[Survey, ...]
    reference: Grid
    brine_as_oil: bool | None = False
    parameters: tuple[PemParameter, ...] = ()
    temperature: CellTemperature | None = None
    _fields: dict[str, tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index the field datatypes."""
        self._fields = {
            name: (s, angle)
            for s, survey in enumerate(self.surveys)
            for name, angle in survey.fields().items()
        }

    @property
    def state_variables(self) -> tuple[str, ...]:
        """The state variables the model reads (the deck never sees them)."""
        return tuple(p.state for p in self.parameters)

    def for_state(self, state: Mapping[str, ArrayLike]) -> "GeophysicsModel":
        """Return the model with the petro-elastic options a member's state sets.

        Raises
        ------
        SimulationError
            If the member has no value of a parameter without a fallback ``value``, or
            more than one value.
        """
        if not self.parameters or self.pem is None:
            return self
        values: dict[str, float] = {}
        for p in self.parameters:
            if p.state in state:
                x = np.asarray(state[p.state], dtype=np.float64).ravel()
                if x.size != 1:
                    raise SimulationError(f"state variable {p.state!r} has {x.size} values, not 1")
                values[p.option] = p.apply(float(x[0]))
                if not np.isfinite(values[p.option]):
                    raise SimulationError(
                        f"state variable {p.state!r} = {x[0]:g} gives pem {p.option} = "
                        f"{values[p.option]:g}"
                    )
            elif p.value is not None:
                values[p.option] = p.value
            else:
                raise SimulationError(
                    f"the member has no state variable {p.state!r} (pem {p.option}) and no "
                    "fallback value is configured"
                )
        try:
            params = getattr(self.pem, "__dataclass_params__", None)
            if params is not None and params.frozen:
                pem = dataclasses.replace(self.pem, **values)  # type: ignore[type-var]
            else:
                # A shallow copy keeps what the model built at setup (e.g. CO₂'s modulus
                # table, which the uncertain options do not affect), then its checks run.
                pem = copy.copy(self.pem)
                for name, value in values.items():
                    setattr(pem, name, value)
                check = getattr(pem, "__post_init__", None)
                if callable(check):
                    check()
        except ValueError as error:
            raise SimulationError(f"pem with {values}: {error}") from error
        return replace(self, pem=pem)

    def positions(
        self, datatype: str
    ) -> tuple[Literal["columns", "stations"], NDArray[np.float64]] | None:
        """Return where a field's values sit, for maps of them.

        ``("columns", (n, 2))``: an AVO map's traced columns ``(i, j)``, 0-based;
        ``("stations", (n, 2))``: the stations' ``(x, y)`` (m); ``None`` for whole traces
        (several values per column) and for a field the model does not have.
        """
        if datatype not in self._fields:
            return None
        survey = self.surveys[self._fields[datatype][0]]
        if isinstance(survey, AVOSurvey):
            if not survey.attribute.is_map:
                return None
            return "columns", survey.columns.astype(np.float64)
        return "stations", np.asarray(survey.stations[:, :2], dtype=np.float64)

    def column_steps(self, datatype: str) -> tuple[int, int]:
        """Return an AVO field's column spacing ``(step_x, step_y)`` ((1, 1) otherwise)."""
        survey = self.surveys[self._fields[datatype][0]] if datatype in self._fields else None
        return survey.steps if isinstance(survey, AVOSurvey) else (1, 1)

    def wavelet_layout(self, datatype: str) -> tuple[tuple[int, ...], NDArray[np.bool_]]:
        """Return the grid and mask a wavelet transform of a field takes (order ``"F"``).

        A map is a 2-D field over the lattice of kept columns (every ``step_x``-th and
        ``step_y``-th), cropped to their bounding box; traces are 3-D, ``(n_sample,
        lattice_x, lattice_y)``, one transform over time and space (as the user's
        ``wavelet3D_sparse_representation``, MATLAB's ``wavedec3`` on each angle's cube).
        The masked grid's F order is the field's own: the kept columns in Eclipse order,
        and for traces each column's samples in turn. The mask keeps the traced columns;
        the transform fills the rest.

        Raises
        ------
        ConfigError
            If ``datatype`` is not an AVO field of the model (station data have no grid).
        """
        if datatype not in self._fields:
            raise ConfigError(f"{datatype} is not a field of [geophysics]")
        survey = self.surveys[self._fields[datatype][0]]
        if not isinstance(survey, AVOSurvey):
            raise ConfigError(
                f"{datatype}: gravity and displacement data sit at stations, not on a grid a "
                "wavelet transform can take"
            )
        steps = np.asarray(survey.steps)
        lattice = survey.columns // steps  # monotone in each axis, so the F order holds
        low = lattice.min(axis=0)
        index = lattice - low
        shape = tuple(int(n) for n in index.max(axis=0) + 1)
        mask = np.zeros(shape, dtype=bool)
        mask[index[:, 0], index[:, 1]] = True
        if survey.attribute.is_map:
            return shape, mask
        n_sample = int(survey.window.sum())
        return (n_sample, *shape), np.broadcast_to(mask, (n_sample, *shape)).copy()

    def field_sizes(self) -> dict[str, int]:
        """Return each field datatype's number of values (at one vintage)."""
        return {name: self.surveys[s].field_size for name, (s, _) in self._fields.items()}

    def check_observations(self, keys: Sequence[DataKey]) -> None:
        """Refuse, before any member runs, observed field data the model cannot forecast.

        Raises
        ------
        ConfigError
            If a field datum is not one of the configured fields, is at a time that is
            not one of its vintages, or has another number of values than the field.
        """
        indices: dict[tuple[str, str], list[int]] = {}
        for key in keys:
            if key.index is not None:
                indices.setdefault((key.datatype, key.time), []).append(key.index)
        problems: list[str] = []
        for (datatype, time), observed in indices.items():
            if datatype not in self._fields:
                problems.append(
                    f"{datatype}@{time} is not a field of [geophysics] "
                    f"({', '.join(sorted(self._fields))})"
                )
                continue
            survey = self.surveys[self._fields[datatype][0]]
            try:
                _vintage(DataKey(time, datatype, 0), survey)
            except ConfigError as error:
                problems.append(str(error))
                continue
            if sorted(observed) != list(range(survey.field_size)):
                problems.append(
                    f"{datatype}@{time} has {len(observed)} values; the field has "
                    f"{survey.field_size} ({survey.extent}), indices 0 to "
                    f"{survey.field_size - 1}"
                )
        if problems:
            raise ConfigError("[geophysics]: " + "; ".join(problems[:5]))

    def field_vintages(self) -> dict[str, tuple[str, ...]]:
        """Return each field datatype's vintages (ISO dates)."""
        return {
            name: tuple(v.isoformat() for v in self.surveys[s].vintages)
            for name, (s, _) in self._fields.items()
        }

    def warnings(self) -> list[str]:
        """Return what may degrade the data without making them wrong (for the run plan)."""
        out = []
        matrices = [s for s in self.surveys if isinstance(s, DisplacementSurvey)]
        mb = sum(s.matrix.nbytes for s in matrices) / 1e6
        if mb > _MATRIX_NOTE_MB:
            out.append(
                f"[geophysics] the displacement kernels take {mb:.0f} MB, copied to every "
                "member's task; fewer stations make them smaller"
            )
        far = self._far_stations()
        if far:
            out.append(
                f"[geophysics] {', '.join(far)}: no station lies over the cells or within "
                "their extent of them; stations are in the grid's own frame (its COORD, "
                "before MAPAXES), in m"
            )
        for survey in self.surveys:
            if isinstance(survey, GravitySurvey):
                pairs = survey.field_size * survey.centres.shape[0]
                if pairs > _GRAVITY_NOTE:
                    out.append(
                        f"[geophysics] {survey.datatype}: {pairs:.3g} station-cell pairs, "
                        "computed for every member at every vintage; fewer stations are "
                        "faster"
                    )
                if survey.closest < 1.0:
                    out.append(
                        f"[geophysics] {survey.datatype}: a station is {survey.closest:.2f} of "
                        "a cell's diagonal from its centre, where a point mass is far from "
                        "the cell's attraction; model = 'prism' is exact there"
                    )
            if isinstance(survey, GravitySurvey) and survey.bounds is not None:
                b = survey.bounds
                boxes = float(
                    np.sum((b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2]) * (b[:, 5] - b[:, 4]))
                )
                cells = float(np.sum(self.reference.active(self.reference.volume)))
                if boxes > 1.01 * cells:
                    out.append(
                        f"[geophysics] {survey.datatype}: the cells' prisms (their corners' "
                        f"horizontal extent) take {boxes / cells:.2f} times the cells' volume, "
                        "as on a grid skewed or rotated against x and y; near the cells, "
                        "point masses may be closer"
                    )
            if not isinstance(survey, AVOSurvey) or getattr(survey.attribute, "uses_times", False):
                continue  # a time shift has no samples to alias
            model = survey.model
            nyquist = model.wavelet.nyquist_interval()
            if model.sampling == "interfaces" and model.time.dt > nyquist:
                out.append(
                    f"[geophysics] {survey.datatype}: times.step = {model.time.dt:g} s is above "
                    f"the wavelet's Nyquist interval ({nyquist:.3g} s), so the traces are "
                    "aliased; use a smaller step"
                )
        return out

    def _far_stations(self) -> list[str]:
        """Return the station data none of whose stations is near the cells (a frame slip)."""
        surveys = [s for s in self.surveys if not isinstance(s, AVOSurvey)]
        if not surveys:
            return []
        centres = self.reference.active(self.reference.centres)
        low, high = centres[:, :2].min(axis=0), centres[:, :2].max(axis=0)
        reach = max(float(np.max(high - low)), 1.0)
        out = []
        for survey in surveys:
            xy = survey.stations[:, :2]
            gap = np.maximum(np.maximum(low - xy, xy - high), 0.0)  # (n_station, 2)
            if float(np.min(np.hypot(gap[:, 0], gap[:, 1]))) > reach:
                out.append(survey.datatype)
        return out

    def restart_needs(self) -> dict[str, str]:
        """Return the restart arrays (RPTRST mnemonics) the data need, and what needs them."""
        needs: dict[str, list[str]] = {}
        if self.pem is not None and self.pem.requires_densities:
            needs.setdefault("DEN", []).append("pem")
        for survey in self.surveys:
            if isinstance(survey, GravitySurvey):
                needs.setdefault("DEN", []).append(survey.datatype)
                needs.setdefault("RPORV", []).append(survey.datatype)
            elif isinstance(survey, DisplacementSurvey) and (
                survey.volume_change == "pore_volume" or survey.bulk is None
            ):
                needs.setdefault("RPORV", []).append(survey.datatype)
        return {mnemonic: ", ".join(who) for mnemonic, who in needs.items()}

    def owns(self, key: DataKey) -> bool:
        """Return whether ``key`` is a component of one of the fields."""
        return key.index is not None and key.datatype in self._fields

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> Array:
        """Return the field values at ``keys`` from the case's files; see the module.

        Raises
        ------
        SimulationError
            If the member's grid has other active columns (or, with gravity or
            displacement data, cells) than the configured grid's, or its state is outside
            a kernel's validity (a petro-elastic model's range, an evanescent wave).
        ConfigError
            If a key is not one of the configured fields, or the case lacks a restart or
            an array at a vintage.
        """
        state = CaseState(case)
        if (
            self.brine_as_oil is None
            and set(state.phases) == {"oil", "gas"}
            and getattr(self.pem, "gas", None) == "co2"
        ):
            raise ConfigError(
                f"{case.name} is an oil-gas run with a CO₂ petro-elastic model, and the deck's "
                "templated RUNSPEC does not say whether it is CO2STORE (whose oil is brine); "
                "write its phase keywords and CO2STORE as plain text"
            )
        self._check_phases(self._phases(state.phases))
        grid = state.grid
        self._check_grid(grid, case)
        wanted: dict[tuple[int, dt.date], list[int]] = {}
        for n, key in enumerate(keys):
            s, _ = self._fields[key.datatype]
            date = _vintage(key, self.surveys[s])
            wanted.setdefault((s, date), []).append(n)
        out = np.empty(len(keys))
        elastic: dict[dt.date, tuple[Array, Array, Array]] = {}
        computed: dict[tuple[int, dt.date], Array] = {}
        quantities = CaseQuantities(state, int(self.reference.actnum.sum()))

        def values(s: int, date: dt.date) -> Array:
            survey = self.surveys[s]
            if not isinstance(survey, AVOSurvey):
                return survey.values(quantities, date)[np.newaxis, :]
            if (s, date) not in computed:
                if date not in elastic:
                    elastic[date] = self._elastic(state, grid, date)
                try:
                    computed[s, date] = survey.values(grid, *elastic[date])
                except ValueError as error:
                    # GeophysicsModelError (an evanescent wave, a non-physical velocity)
                    # or the member's own geometry (an active cell with a negative
                    # thickness).
                    raise SimulationError(f"{case.name} at {date.isoformat()}: {error}") from error
            return computed[s, date]

        for (s, date), rows in wanted.items():
            survey = self.surveys[s]
            fields = values(s, date)
            if isinstance(survey, AVOSurvey) and survey.baseline is not None:
                # four_d = "differences" (or a time shift), from the fixed baseline, or with
                # a rolling one from the vintage before (the config orders them).
                k = survey.vintages.index(date)
                reference = survey.vintages[k - 1] if survey.rolling and k else survey.baseline
                fields = fields - values(s, reference)
            for n in rows:
                key = keys[n]
                angle = self._fields[key.datatype][1]
                index = key.index
                assert index is not None
                if not 0 <= index < survey.field_size:
                    raise ConfigError(
                        f"{key}: the field {key.datatype} has {survey.field_size} values "
                        f"({survey.extent}); the observations have more"
                    )
                out[n] = fields[angle, index]
        return out

    def check_phases(self, phases: Sequence[str]) -> None:
        """Refuse phases the petro-elastic model cannot take (a configuration error).

        Raises
        ------
        ConfigError
            If the model refuses them.
        """
        self._check_phases(self._phases(phases))

    def petro_elastic_phases(self, phases: Sequence[str]) -> list[str]:
        """Return the phases as the petro-elastic model sees them (CO2STORE's oil is brine)."""
        return self._phases(phases)

    def _phases(self, phases: Sequence[str]) -> list[str]:
        if not self.brine_as_oil:
            return list(phases)
        return ["water" if p == "oil" else p for p in phases]

    def check_temperature(
        self, deck_c: float | None, varies: bool, phases: Sequence[str] | None
    ) -> list[str]:
        """Refuse a petro-elastic temperature the deck contradicts; return the warnings.

        Raises
        ------
        ConfigError
            If the model refuses the deck's temperature.
        """
        check = getattr(self.pem, "check_temperature", None)
        if not callable(check):
            return []
        try:
            out: list[str] = check(deck_c, varies, phases)
        except ValueError as error:
            raise ConfigError(f"[geophysics] pem: {error}") from error
        return [f"[geophysics] {w}" for w in out]

    def _check_phases(self, phases: Sequence[str]) -> None:
        if self.pem is None:
            return
        check = getattr(self.pem, "check_phases", None)
        if callable(check):
            try:
                check(phases)
            except ValueError as error:
                raise ConfigError(f"[geophysics] pem: {error}") from error

    def _check_grid(self, grid: Grid, case: Path) -> None:
        if grid.shape != self.reference.shape:
            raise ConfigError(
                f"{case.name}'s grid is {grid.shape}, grid_file's {self.reference.shape}"
            )
        differ = grid.active_columns != self.reference.active_columns
        if differ.any() and any(isinstance(s, AVOSurvey) for s in self.surveys):
            ij = [f"({i + 1}, {j + 1})" for i, j in np.argwhere(differ)[:5]]
            raise SimulationError(
                f"{case.name}: {int(differ.sum())} column(s) are active in one of this "
                f"member's grid and grid_file but not the other, e.g. (I, J) = {', '.join(ij)}; "
                "the traces must be the same for every member (an updated porosity under "
                "MINPV can deactivate cells)"
            )
        if any(not isinstance(s, AVOSurvey) for s in self.surveys):
            cells = grid.actnum != self.reference.actnum
            if cells.any():
                ijk = [f"({i + 1}, {j + 1}, {k + 1})" for i, j, k in np.argwhere(cells)[:5]]
                raise SimulationError(
                    f"{case.name}: {int(cells.sum())} cell(s) are active in one of this "
                    f"member's grid and grid_file but not the other, e.g. (I, J, K) = "
                    f"{', '.join(ijk)}; gravity and displacement take grid_file's active "
                    "cells (an updated porosity under MINPV can deactivate cells)"
                )

    def _elastic(self, state: CaseState, grid: Grid, date: dt.date) -> tuple[Array, Array, Array]:
        """Return Vp, Vs and density on the grid ``(n_x, n_y, n_z)`` at ``date``."""
        # The configuration gives AVO data both.
        assert self.pem is not None
        assert self.overburden is not None
        pressure, saturations, densities = state.flow_state(date, self.pem.requires_densities)
        if self.brine_as_oil:
            # CO2STORE's oil-gas form: its "oil" is brine (no dead-oil fit for it).
            saturations = {("water" if p == "oil" else p): s for p, s in saturations.items()}
            densities = {("water" if p == "oil" else p): d for p, d in densities.items()}
        flow = FlowState(
            porosity=state.static("PORO"),
            pressure=pressure,
            overburden_pressure=self.overburden.values(state),
            saturations=saturations,
            densities=densities or None,
            temperature=None if self.temperature is None else self.temperature.cells(state, date),
        )
        try:
            props = self.pem.elastic(flow)
        except GeophysicsModelError as error:
            raise SimulationError(f"{state.case.name} at {date.isoformat()}: {error}") from error
        except ValueError as error:
            # The model's own settings or inputs every member shares (e.g. no densities); a
            # member's array of the wrong size would also land here, which the deck
            # simulators' consistent INIT and restarts do not produce.
            raise ConfigError(f"[geophysics] pem at {date.isoformat()}: {error}") from error
        if props.clamped_cells:
            logger.debug(
                "geophysics_effective_pressure_clamped",
                extra={"case": state.case.name, "date": date.isoformat(),
                       "cells": props.clamped_cells},
            )  # fmt: skip
        if props.counts:
            # A warning when cells are snapped, pressure-inconsistent (revision 1: the PVT
            # tables' resolution or the temperature) or outside Lebedev's fit (revision 2,
            # uncorrected); corrected cells and the critical band's are normal operation.
            noted = any(props.counts.get(k, 0) for k in _NOTED_COUNTS)
            (logger.warning if noted else logger.debug)(
                "geophysics_co2_counts",
                extra={"case": state.case.name, "date": date.isoformat(), **props.counts},
            )
        active = grid.actnum.ravel(order="F")
        grids = []
        for values in (props.vp, props.vs, props.density):
            g = np.ones(active.size)  # inactive cells: never read (Columns' active mask)
            g[active] = values
            grids.append(g.reshape(grid.shape, order="F"))
        return grids[0], grids[1], grids[2]


def _vintage(key: DataKey, survey: Survey) -> dt.date:
    try:
        date = dt.date.fromisoformat(key.time)
    except ValueError as error:
        raise ConfigError(f"{key}: the time is not an ISO date") from error
    if date not in survey.vintages:
        raise ConfigError(
            f"{key}: {key.time} is not a vintage of {survey.datatype} "
            f"({', '.join(v.isoformat() for v in survey.vintages)})"
        )
    return date


def _load_array(path: Path, key: str | None) -> Array:
    """Return the array of an ``.npy``, or of an ``.npz`` (``key``, default its first)."""
    try:
        loaded = np.load(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        return np.asarray(loaded, dtype=np.float64).ravel()
    with loaded:
        name = key or loaded.files[0]
        if name not in loaded.files:
            raise ConfigError(f"{path.name} has no array {name!r} ({loaded.files})")
        return np.asarray(loaded[name], dtype=np.float64).ravel()


def _overburden(section: GeophysicsSection, reference: Grid) -> OverburdenPressure | None:
    p = section.overburden_pressure
    if p is None:
        return None
    if isinstance(p, OverburdenPressureFile):
        return CellPressure.of(_load_array(p.file, p.key) * _PRESSURE_UNITS[p.unit], reference)
    return LithostaticPressure(p.gradient, p.surface)


def _medium(section: ConstantOverburdenSection) -> ConstantOverburden:
    return ConstantOverburden(section.vp, section.vs, section.density)


def _survey(
    datatype: str,
    vintages: Sequence[str],
    options: AVOOptions,
    reflectivity: PPReflectivity,
    attribute: AVOAttribute,
    reference: Grid,
    region: NDArray[np.bool_],
) -> AVOSurvey:
    w = options.wavelet
    wavelet: RickerWavelet | SampledWavelet
    if isinstance(w, RickerWaveletSection):
        wavelet = RickerWavelet(w.frequency, w.length)
    else:
        samples = _load_array(w.file, w.key)
        centre = int(np.argmax(np.abs(samples))) if w.centre is None else w.centre
        wavelet = SampledWavelet(samples, w.dt, centre)
    o = options.overburden
    overburden = (
        _medium(o)
        if isinstance(o, ConstantOverburdenSection)
        else LayeredOverburden(tuple(o.thickness), tuple(o.vp), tuple(o.vs), tuple(o.density))
    )
    time = TimeAxis(options.times.start, options.times.end, options.times.step)
    model = SeismicModel(
        reflectivity=reflectivity,
        wavelet=wavelet,
        overburden=overburden,
        angles_deg=tuple(options.angles),
        time=time,
        sampling=options.sampling,
        inactive=options.inactive,
        underburden=None if options.underburden is None else _medium(options.underburden),
    )
    times = time.times
    window = (
        np.ones(times.size, dtype=bool)
        if options.time_window is None
        else time_window(times, *options.time_window)
    )
    nx, ny, _ = reference.shape
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    kept = reference.active_columns & region & (i % options.step_x == 0) & (j % options.step_y == 0)
    flat = np.flatnonzero(kept.ravel(order="F"))
    columns = np.column_stack(np.unravel_index(flat, (nx, ny), order="F")).astype(np.intp)
    if columns.shape[0] == 0:
        raise ConfigError(f"[geophysics] {datatype}: no active column is kept")
    return AVOSurvey(
        datatype=datatype,
        vintages=tuple(dt.date.fromisoformat(v) for v in vintages),
        model=model,
        attribute=attribute,
        window=window,
        columns=columns,
        baseline=None if options.baseline is None else dt.date.fromisoformat(options.baseline),
        rolling=options.baseline_mode == "rolling",
        steps=(options.step_x, options.step_y),
    )


def region_mask(
    region: RegionAroundWellsSection | RegionFileSection | None,
    reference: Grid,
    grid_file: Path,
    completions: Mapping[str, Sequence[tuple[int, int]]] | None,
    datatype: str,
) -> NDArray[np.bool_]:
    """Return ``(n_x, n_y)``: the columns a datum's region of interest keeps.

    Around wells, the completions' column centres (:func:`column_centres`) span
    ``[x_min, x_max] x [y_min, y_max]`` (or ``x_range``/``y_range``); a ``rectangle``
    widens it by ``margin`` on each side, and an ``ellipse`` is centred on it with
    semi-axes ``max(half the extent + margin, 1 m)``: the reference's formulas. (Its seeds
    differ: it reads completions from the rendered deck only, and with fewer than two it
    surrounds every active column; see :class:`~apex.domain.config.RegionAroundWellsSection`.)

    Raises
    ------
    ConfigError
        If the file cannot be read or has another shape than the grid's columns, or the
        wells are unknown, have no completion, or lie outside the grid.
    """
    nx, ny, _ = reference.shape
    where = f"[geophysics] {datatype} region"
    if region is None:
        return np.ones((nx, ny), dtype=bool)
    if isinstance(region, RegionFileSection):
        values = _load_grid_array(region.file, region.key)
        if values.shape != (nx, ny) or not np.all(np.isfinite(values)):
            raise ConfigError(
                f"{where}: {region.file.name} must hold finite values of shape (n_x, n_y) = "
                f"{(nx, ny)}, got {values.shape}"
            )
        out: NDArray[np.bool_] = values != 0
        return out
    if completions is None:
        raise ConfigError(f"{where}: a region around wells needs the deck's completions")
    names = region.wells if region.wells is not None else sorted(completions)
    unknown = sorted(set(names) - set(completions))
    if unknown:
        raise ConfigError(f"{where}: the deck has no completions of the wells {unknown}")
    cells = np.array([c for name in names for c in completions[name]], dtype=np.intp) - 1
    if cells.size == 0:
        raise ConfigError(f"{where}: the deck has no well completions")
    if np.any(cells < 0) or np.any(cells[:, 0] >= nx) or np.any(cells[:, 1] >= ny):
        raise ConfigError(f"{where}: a completion lies outside the grid {(nx, ny)}")
    try:
        centres = column_centres(grid_file)
    except (OSError, ValueError) as error:
        raise ConfigError(f"[geophysics] grid_file: cannot read it: {error}") from error
    x, y = centres[..., 0], centres[..., 1]
    seeds = centres[cells[:, 0], cells[:, 1]]
    x_min, x_max = region.x_range or (float(seeds[:, 0].min()), float(seeds[:, 0].max()))
    y_min, y_max = region.y_range or (float(seeds[:, 1].min()), float(seeds[:, 1].max()))
    m = region.margin
    if region.shape == "rectangle":
        rect: NDArray[np.bool_] = (
            (x >= x_min - m) & (x <= x_max + m) & (y >= y_min - m) & (y <= y_max + m)
        )
        return rect
    # The reference's ellipse: semi-axes of at least 1 m, so one completion keeps a disc.
    rx = max(1.0, 0.5 * (x_max - x_min) + m)
    ry = max(1.0, 0.5 * (y_max - y_min) + m)
    dx = (x - 0.5 * (x_min + x_max)) / rx
    dy = (y - 0.5 * (y_min + y_max)) / ry
    ellipse: NDArray[np.bool_] = dx * dx + dy * dy <= 1.0
    return ellipse


def _load_grid_array(path: Path, key: str | None, single: bool = False) -> Array:
    """Return the array of an ``.npy``, or of an ``.npz`` (``key``, default its first), as is.

    With ``single``, an ``.npz`` of several arrays needs ``key``.
    """
    try:
        loaded = np.load(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        return np.asarray(loaded, dtype=np.float64)
    with loaded:
        if key is None and single and len(loaded.files) > 1:
            raise ConfigError(
                f"{path.name} holds several arrays ({loaded.files}); name one (its key)"
            )
        name = key or loaded.files[0]
        if name not in loaded.files:
            raise ConfigError(f"{path.name} has no array {name!r} ({loaded.files})")
        return np.asarray(loaded[name], dtype=np.float64)


def _stations(datum: GeophysicsDataSection, centres: Array) -> Array:
    """Return a datum's stations ``(n_station, 3)``: over the active cells, or a file's."""
    options = datum.options
    assert isinstance(options, GravityOptions | DisplacementOptions)
    where = f"[geophysics] {datum.datatype} stations"
    s = options.stations
    if isinstance(s, StationFileSection):
        stations = _load_grid_array(s.file, s.key)
        if stations.ndim != 2 or stations.shape[1] != 3 or stations.shape[0] == 0:
            raise ConfigError(
                f"{where}: {s.file.name} must hold an (n, 3) array, got {stations.shape}"
            )
    else:
        seabed = None
        if s.seabed_file is not None:
            table = _load_grid_array(s.seabed_file, s.seabed_key, single=True)
            if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] == 0:
                raise ConfigError(
                    f"{where}: {s.seabed_file.name} must hold an (n, 3) table of x, y, depth, "
                    f"got {table.shape}"
                )
            seabed = (table[:, 0], table[:, 1], table[:, 2])
        stations = regular_stations(
            centres[:, 0],
            centres[:, 1],
            spacing=s.spacing,
            padding=s.padding,
            depth=s.depth,
            refine=s.refine,
            seabed=seabed,
        )
    if stations.shape[0] == 0:
        raise ConfigError(
            f"{where}: none: the active cells span nothing along x or y (a cross-section); "
            "give a padding"
        )
    if not np.all(np.isfinite(stations)):
        raise ConfigError(f"{where}: not every coordinate is finite")
    return np.ascontiguousarray(stations, dtype=np.float64)


#: The (station, cell) pairs :func:`_closest` takes at once (its work arrays' bound).
_CLOSEST_ELEMENTS = 1_000_000


def _closest(stations: Array, centres: Array, diagonal: Array) -> float:
    """Return the least station-centre distance over that cell's diagonal.

    0 when a station is at a centre, whatever the cell's size (a degenerate cell's
    diagonal of 0 would otherwise hide it). Taken in blocks of about
    :data:`_CLOSEST_ELEMENTS` pairs.
    """
    out = float("inf")
    n = centres.shape[0]
    cells = min(n, _CLOSEST_ELEMENTS) or 1
    rows = max(1, _CLOSEST_ELEMENTS // cells)
    for i in range(0, stations.shape[0], rows):
        for j in range(0, n, cells):
            d = centres[np.newaxis, j : j + cells, :] - stations[i : i + rows, np.newaxis, :]
            r = np.sqrt(np.einsum("knd,knd->kn", d, d))
            if np.any(r == 0):
                return 0.0
            size = diagonal[np.newaxis, j : j + cells]
            ratio = np.where(size > 0, r / np.where(size > 0, size, 1.0), np.inf)
            out = min(out, float(ratio.min()))
    return out


def _station_survey(
    datum: GeophysicsDataSection, reference: Grid, kernel: DisplacementKernel | None
) -> GravitySurvey | DisplacementSurvey:
    centres = reference.active(reference.centres)
    stations = _stations(datum, centres)
    where = f"[geophysics] {datum.datatype}"
    vintages = tuple(dt.date.fromisoformat(v) for v in datum.vintages)
    baseline = dt.date.fromisoformat(datum.options.baseline or "")
    if isinstance(datum.options, GravityOptions):
        bounds = None
        if datum.options.model == "prism":
            boxes = reference.active(reference.boxes)
            top, bottom = reference.active(reference.top), reference.active(reference.bottom)
            bounds = np.column_stack((boxes, top, bottom))
            flat = (bounds[:, 1::2] <= bounds[:, 0::2]).any(axis=1)
            if flat.any():
                raise ConfigError(
                    f"{where}: {int(flat.sum())} active cell(s) have no extent in x, y or "
                    "depth, so no prism; use model = 'point_mass'"
                )
        closest = float("inf")
        if bounds is None:
            boxes = reference.active(reference.boxes)
            depth = reference.active(reference.bottom) - reference.active(reference.top)
            diagonal = np.sqrt(
                (boxes[:, 1] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 2]) ** 2 + depth**2
            )
            closest = _closest(stations, centres, diagonal)
            if closest == 0:
                raise ConfigError(
                    f"{where}: a station is at a cell's centre, where its point mass is "
                    "singular; move the station, or use model = 'prism'"
                )
        return GravitySurvey(
            datum.datatype, vintages, baseline, stations, centres, bounds,
            datum.options.gravitational_constant, closest,
        )  # fmt: skip
    options = datum.displacement()
    if kernel is None:
        raise ValueError(f"{where}: a displacement datum needs its kernel")
    pairs = stations.shape[0] * centres.shape[0]
    if pairs > _MAX_PAIRS:
        raise ConfigError(
            f"{where}: {stations.shape[0]} stations x {centres.shape[0]} active cells is "
            f"above {_MAX_PAIRS} station-cell pairs, whose kernel matrix every member's task "
            "carries; use fewer stations (a larger spacing)"
        )
    try:
        matrix = np.ascontiguousarray(kernel.matrix(stations, centres), dtype=np.float64)
    except ValueError as error:
        raise ConfigError(f"{where}: {error}") from error
    by_pressure = options.volume_change == "pressure"
    return DisplacementSurvey(
        datum.datatype, vintages, baseline, stations, matrix, options.volume_change,
        options.compressibility,
        reference.active(reference.volume) if by_pressure and options.volume == "bulk" else None,
    )  # fmt: skip


def build_geophysics_model(
    section: GeophysicsSection,
    pem: PetroElasticModel | None,
    reflectivities: Sequence[PPReflectivity | None],
    attributes: Sequence[AVOAttribute | None],
    completions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    kernels: Sequence[DisplacementKernel | None] | None = None,
    brine_as_oil: bool | None = False,
    parameters: Sequence[PemParameter] = (),
    temperature: CellTemperature | None = None,
) -> GeophysicsModel:
    """Build the forward model of ``[geophysics]`` from its plugins, one per datum.

    ``reflectivities``, ``attributes`` and ``kernels`` are the plugins of each
    ``[[geophysics.data]]``, in order (``None`` where a kind has none), created by the
    composition root from the registries. ``completions`` are the deck's wells' completed
    columns, 1-based ``(I, J)``, for a region around them.

    Raises
    ------
    ConfigError
        If ``grid_file`` or a data file cannot be read, a datum keeps no trace, or a
        kernel refuses the stations and cells.
    ValueError
        If a kernel refuses its settings (e.g. a layered overburden's tables), or a datum
        lacks its plugins.
    """
    try:
        reference = Grid.read(section.grid_file)
    except (OSError, ValueError) as error:
        raise ConfigError(f"[geophysics] grid_file: cannot read it: {error}") from error
    thin = reference.actnum & (reference.bottom < reference.top)
    if thin.any():
        i, j, k = (int(v) + 1 for v in np.argwhere(thin)[0])
        raise ConfigError(
            f"[geophysics] grid_file: {int(thin.sum())} active cell(s) have their base above "
            f"their top, e.g. (I, J, K) = ({i}, {j}, {k})"
        )
    if kernels is None:
        kernels = [None] * len(section.data)
    if any(d.kind != "avo" for d in section.data) and not grid_unit_stated(section.grid_file):
        # Without it, a FIELD grid's feet would be taken as metres: every station distance
        # and bulk volume wrong, and no member's check would notice.
        raise ConfigError(
            "[geophysics] grid_file states no length unit (GRIDUNIT), which gravity and "
            "displacement need; use an EGRID the simulator wrote"
        )
    surveys: list[Survey] = []
    for d, r, a, kernel in zip(section.data, reflectivities, attributes, kernels, strict=True):
        if d.kind != "avo":
            surveys.append(_station_survey(d, reference, kernel))
            continue
        if r is None or a is None:
            raise ValueError(f"[geophysics] {d.datatype}: an AVO datum needs its plugins")
        options = d.avo()
        region = region_mask(options.region, reference, section.grid_file, completions, d.datatype)
        surveys.append(_survey(d.datatype, d.vintages, options, r, a, reference, region))
    if pem is None and any(isinstance(s, AVOSurvey) for s in surveys):
        raise ValueError("[geophysics]: AVO data need a petro-elastic model")
    return GeophysicsModel(
        pem, _overburden(section, reference), tuple(surveys), reference, brine_as_oil,
        tuple(parameters), temperature,
    )  # fmt: skip
