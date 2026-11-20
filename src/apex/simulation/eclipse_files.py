"""The Eclipse-format case reader: EGRID, INIT and restarts.

:class:`EclipseFiles` is the :class:`~apex.core.case_state.CornerGeometryReader` of the deck
simulators (``opm_flow``, ``eclipse``, and others writing Eclipse-format files through
their packages' entry points). It is what :class:`~apex.simulation.geophysics.CaseState`
and :class:`~apex.simulation.model_error.CaseFiles` read through, so their results are
those of the reading they replaced, bit for bit.

Arrays are as written, in the deck's units, and :meth:`EclipseFiles.unit`
states each known one's unit from INTEHEAD's unit system; any other array passes through
with no unit stated. States are the restarts, each a
:class:`~apex.domain.case_state.StateTime` whose index is its report step (``SEQNUM`` in a
unified file, the suffix of a ``.Xnnnn`` one) and whose date is its INTEHEAD's; two
restarts on one day are two states. Formatted files (``.FUNRST``, ``.Fnnnn``), dual
porosity and local grids are not read.

Shapes: grid arrays are ``(n_x, n_y, n_z)`` in Eclipse (F) order, i fastest; cell arrays
over the active cells ``(n_active,)``, in the simulator's active order.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from apex.core.case_state import MissingState, check_region
from apex.core.errors import SimulationError
from apex.core.plugins import case_state_readers
from apex.domain.case_state import CaseMesh, CornerGrid, StateTime, UnitSystem

if TYPE_CHECKING:
    from resdata.resfile import ResdataFile, ResdataFileView

__all__ = [
    "UNIT_SYSTEMS",
    "EclipseFiles",
    "Grid",
    "column_centres",
    "grid_unit_stated",
]

#: INTEHEAD items (0-based): the unit system, the phases, the date (day, month, year).
_UNITS_ITEM, _PHASES_ITEM, _DATE_ITEMS = 2, 14, (64, 65, 66)

#: INTEHEAD's unit-system codes (Eclipse file format): pressure bar, psia or atm.
UNIT_SYSTEMS = {
    1: UnitSystem("METRIC", 1.0e5, 1.0, 1.0, 1.0, "C"),
    2: UnitSystem("FIELD", 6894.757293168361, 16.018463373960138, 0.3048, 0.158987294928, "F"),
    3: UnitSystem("LAB", 101325.0, 1000.0, 0.01, 1.0e-6, "C"),
    4: UnitSystem("PVT-M", 101325.0, 1.0, 1.0, 1.0, "C"),
}

#: Each unit system's units by quantity, as keys of ``apex.domain.case_state.SI_UNITS``
#: (the same factors as :data:`UNIT_SYSTEMS`).
_UNITS = {
    "METRIC": {"pressure": "bar", "density": "kg/m3", "length": "m", "volume": "m3"},
    "FIELD": {"pressure": "psia", "density": "lb/ft3", "length": "ft", "volume": "bbl"},
    "LAB": {"pressure": "atm", "density": "g/cm3", "length": "cm", "volume": "cm3"},
    "PVT-M": {"pressure": "atm", "density": "kg/m3", "length": "m", "volume": "m3"},
}

#: The arrays whose quantity the reader knows; the rest pass through with no unit.
_QUANTITIES = {
    **dict.fromkeys(("PRESSURE", "PCOW", "PCOG", "PCGW"), "pressure"),
    **dict.fromkeys(("OIL_DEN", "WAT_DEN", "GAS_DEN", "DENO", "DENW", "DENG"), "density"),
    **dict.fromkeys(("DEPTH", "DX", "DY", "DZ", "TOPS"), "length"),
    **dict.fromkeys(("RPORV", "PORV"), "volume"),
    **dict.fromkeys(("SWAT", "SGAS", "SOIL", "PORO", "NTG"), "fraction"),
    **dict.fromkeys(("PERMX", "PERMY", "PERMZ"), "permeability"),
    "TEMP": "temperature",
}

#: INTEHEAD's phase bits.
_PHASE_BITS = {"oil": 1, "water": 2, "gas": 4}

#: EGRID ``GRIDUNIT`` lengths in m.
_GRID_UNITS = {"METRES": 1.0, "FEET": 0.3048, "CM": 0.01}


@dataclass(frozen=True)
class Grid(CornerGrid):
    """A corner-point grid's geometry in SI units, read from an EGRID (see the base)."""

    @classmethod
    def read(cls, path: Path, length: float = 1.0) -> "Grid":
        """Read an EGRID file in m, in the grid's own frame (before MAPAXES).

        Lengths are converted by the file's ``GRIDUNIT``, or by ``length`` (m per the
        file's unit) when it has none. Coordinates are the grid's COORD and ZCORN, not
        mapped by MAPAXES: the frame of SimulatorWrap's stations and of the AVO regions
        (:func:`column_centres`), so stations, seabed tables and refinement points are
        given in it.

        Raises
        ------
        OSError, ValueError
            If the file cannot be read as a grid, or its length unit is unknown.
        """
        from resdata.grid import Grid as ResGrid  # optional dependency (the reservoir extra)

        if not path.is_file():
            raise OSError(f"no grid file {path}")
        grid = ResGrid(str(path), apply_mapaxes=False)
        length = _grid_unit(path, length)
        shape = (int(grid.nx), int(grid.ny), int(grid.nz))
        index = grid.export_index(active_only=False)  # every cell, global (F) order
        corners = np.asarray(grid.export_corners(index), dtype=np.float64) * length  # (n, 24)
        x, y, z = corners[:, 0::3], corners[:, 1::3], corners[:, 2::3]
        top = z[:, :4].mean(axis=1)  # the eight corners: the first four are the top
        bottom = z[:, 4:].mean(axis=1)
        centres = np.column_stack((x.mean(axis=1), y.mean(axis=1), z.mean(axis=1)))
        boxes = np.column_stack((x.min(axis=1), x.max(axis=1), y.min(axis=1), y.max(axis=1)))
        volume = np.asarray(grid.export_volume(index), dtype=np.float64) * length**3
        actnum = np.asarray(index["active"].to_numpy() >= 0)

        def cells(values: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = values.reshape((*shape, *values.shape[1:]), order="F")
            return out

        return cls(
            shape,
            actnum.reshape(shape, order="F"),
            cells(top),
            cells(bottom),
            cells(centres),
            cells(boxes),
            cells(np.abs(volume)),
        )


def column_centres(path: Path) -> NDArray[np.float64]:
    """Return ``(n_x, n_y, 2)``: each column's x and y (m), from an EGRID's pillars.

    A column's centre is the mean of the tops of its four pillars (``COORD``), as the
    reference places its region of interest; the lengths are converted with ``GRIDUNIT``.

    Raises
    ------
    OSError, ValueError
        If the file cannot be read as a grid, or its length unit is unknown.
    """
    from resdata.grid import Grid as ResGrid  # optional dependency (the reservoir extra)

    if not path.is_file():
        raise OSError(f"no grid file {path}")
    grid = ResGrid(str(path))
    nx, ny = int(grid.nx), int(grid.ny)
    # numpy_view does not own its buffer: the keyword is held until the copy is made (a
    # temporary would be freed as soon as numpy_view returns).
    keyword = grid.export_coord()
    coord = np.array(keyword.numpy_view(), dtype=np.float64, copy=True)
    del keyword
    # COORD: one pillar (x, y, z top, x, y, z bottom) per corner node, I fastest.
    tops = coord.reshape(ny + 1, nx + 1, 6)[:, :, :2].transpose(1, 0, 2)  # (nx+1, ny+1, 2)
    centres = 0.25 * (tops[:-1, :-1] + tops[1:, :-1] + tops[:-1, 1:] + tops[1:, 1:])
    out: NDArray[np.float64] = centres * _grid_unit(path, 1.0)
    return out


def grid_unit_stated(path: Path) -> bool:
    """Return whether an EGRID states its length unit (``GRIDUNIT``)."""
    from resdata.resfile import ResdataFile  # optional dependency (the reservoir extra)

    egrid = ResdataFile(str(path))
    return "GRIDUNIT" in egrid and bool(str(egrid["GRIDUNIT"][0][0]).strip())


def _grid_unit(path: Path, default: float) -> float:
    """Return the m per length unit of an EGRID: its ``GRIDUNIT``, else ``default``.

    Raises
    ------
    ValueError
        If the unit is unknown.
    """
    from resdata.resfile import ResdataFile  # optional dependency (the reservoir extra)

    egrid = ResdataFile(str(path))
    unit = str(egrid["GRIDUNIT"][0][0]).strip().upper() if "GRIDUNIT" in egrid else ""
    if not unit:  # not stated (resdata writes a blank one for PVT-M)
        return default
    if unit not in _GRID_UNITS:
        raise ValueError(f"{path.name}: unknown GRIDUNIT {unit!r}")
    return _GRID_UNITS[unit]


def _date(intehead: NDArray[np.float64]) -> dt.date:
    day, month, year = (int(intehead[i]) for i in _DATE_ITEMS)
    try:
        return dt.date(year, month, day)
    except ValueError as error:
        raise SimulationError(f"a restart's INTEHEAD holds no date: {error}") from error


def _start(intehead: NDArray[np.float64]) -> dt.date | None:
    """Return the run's start, INIT's INTEHEAD date, or ``None`` if it holds none."""
    try:
        return _date(intehead)
    except SimulationError:
        return None


@dataclass(frozen=True)
class _Restart:
    """Where one report step's restart is: its file and, in a unified file, its block."""

    step: int
    path: Path
    block: int | None


class EclipseFiles:
    """Reads a deck run's EGRID, INIT and restarts, in the deck's units, restarts by step.

    Parameters
    ----------
    case
        The run's path without a suffix (``member/MODEL``).

    Raises
    ------
    SimulationError
        If the INIT cannot be read (the run did not finish as it should), or states an
        unknown unit system.
    """

    #: What it gives beyond the state: the grid's corner geometry, with the
    #: deck's unit system and phases, and always a structured mesh.
    capabilities: ClassVar[frozenset[str]] = frozenset({"corner_geometry", "structured_mesh"})

    def __init__(self, case: Path) -> None:
        from resdata.resfile import ResdataFile  # optional dependency (the reservoir extra)

        self._case = Path(case)
        self._file = ResdataFile
        self.init = self._open(Path(f"{case}.INIT"))
        header = self.array(self.init, "INTEHEAD")
        code = int(header[_UNITS_ITEM])
        if code not in UNIT_SYSTEMS:
            raise SimulationError(f"{self._case.name}.INIT: unknown unit system code {code}")
        self._units = UNIT_SYSTEMS[code]
        bits = int(header[_PHASES_ITEM])
        self._phases = tuple(p for p, bit in _PHASE_BITS.items() if bits & bit)
        self._start_date = _start(header)

    # -- files -------------------------------------------------------------------------

    def _open(self, path: Path) -> "ResdataFile":
        """Open a result file; a run's unreadable file is that member's failure."""
        try:
            return self._file(str(path))
        except (OSError, ValueError) as error:
            raise SimulationError(f"cannot read {path.name}: {error}") from error

    @staticmethod
    def array(view: "ResdataFile | ResdataFileView", name: str) -> NDArray[np.float64]:
        """Return a copy of ``name``'s first occurrence in ``view``.

        Copied while ``view``, which owns the buffer, is alive: ``numpy_view()`` keeps no
        reference to it, so a view kept past its file reads freed memory.
        """
        if name not in view:
            raise SimulationError(f"a result file lacks {name} (truncated or corrupt?)")
        return np.array(view[name][0].numpy_view(), dtype=np.float64, copy=True)

    @cached_property
    def _unified(self) -> "ResdataFile | None":
        path = Path(f"{self._case}.UNRST")
        return self._open(path) if path.exists() else None

    def _time(self, restart: _Restart) -> StateTime:
        """Return a restart's time: its date (INTEHEAD) and elapsed time.

        The elapsed time is DOUBHEAD's first item (days since the start, exact for
        sub-daily steps) where the restart has it, else the whole days since INIT's start
        date, else unknown.
        """
        view = self._view(restart)
        date = _date(self.array(view, "INTEHEAD"))
        seconds: float | None = None
        if "DOUBHEAD" in view:
            head = self.array(view, "DOUBHEAD")
            # A truncated DOUBHEAD (no item 0) states no elapsed time: the dates' whole days.
            if head.size:
                seconds = float(head[0]) * 86400.0
        if seconds is None and self._start_date is not None:
            seconds = float((date - self._start_date).days) * 86400.0
        return StateTime(restart.step, seconds, date)

    @cached_property
    def _steps(self) -> dict[int, _Restart]:
        """Each report step's restart, in the files' order; no date is read.

        In a unified file a block's report step is its SEQNUM value (not its position:
        restarts at steps 0, 3 and 6 are blocks 0, 1 and 2); a separate file's is its
        suffix. Local grids add INTEHEADs of their own, so blocks are counted by SEQNUM.

        A SEQNUM repeated in a unified file keeps its **first** block, as resdata's own
        lookup by report step does (``restart_view(report_step=...)``: checked on resdata
        6.3.5, a file of SEQNUMs 0, 1, 1 gives the second block, not the third, for step 1).
        A unified file with no SEQNUM has no steps: :meth:`times` is empty (as geophysics'
        reading found no restart before), and a lookup by step fails naming the file
        (:meth:`_restart`).
        """
        unified = self._unified
        if unified is not None:
            path = Path(f"{self._case}.UNRST")
            if "SEQNUM" not in unified:
                return {}
            steps = [int(np.array(kw.numpy_view(), copy=True)[0]) for kw in unified["SEQNUM"]]
            found: dict[int, _Restart] = {}
            for n, step in enumerate(steps):
                found.setdefault(step, _Restart(step, path, n))
            return found
        pattern = f"{self._case.name}.X[0-9][0-9][0-9][0-9]"
        return {
            int(path.suffix[2:]): _Restart(int(path.suffix[2:]), path, None)
            for path in sorted(self._case.parent.glob(pattern))
        }

    @cached_property
    def _times(self) -> tuple[StateTime, ...]:
        """Every restart's time, in the files' order (each restart's dates read once)."""
        return tuple(self._time(r) for r in self._steps.values())

    def _restart(self, time: StateTime) -> _Restart:
        """Return the restart of ``time``'s report step (its index; no date is read)."""
        restart = self._steps.get(time.index)
        unified = self._unified
        if restart is None and unified is not None and "SEQNUM" not in unified:
            # Not MissingState: no step can be found in this file, whichever is asked for.
            raise SimulationError(
                f"{self._case.name}.UNRST has no SEQNUM keyword: its restarts cannot be found "
                "by report step (truncated or corrupt?)"
            )
        if restart is None:
            written = ", ".join(str(s) for s in self._steps) or "none"
            raise MissingState(
                f"{self._case.name} wrote no restart at {time.describe()} (report steps "
                f"written: {written})"
            )
        return restart

    def _view(self, restart: _Restart) -> "ResdataFile | ResdataFileView":
        if restart.block is None:
            return self._open(restart.path)
        assert self._unified is not None
        return self._unified.restart_view(seqnum_index=restart.block)

    def at_step(self, step: int) -> StateTime:
        """Return the state of report step ``step``, found without reading any date.

        Its ``seconds`` and ``date`` are ``None``: :meth:`state` finds a state by its index.
        For a consumer that reads by report step (model error), so that a restart it never
        asks for (a bad date, say) cannot fail it.

        Raises
        ------
        MissingState
            If the run wrote no restart at that step.
        """
        time = StateTime(step, None, None)
        self._restart(time)
        return time

    # -- CaseStateReader ---------------------------------------------------------------

    @property
    def case(self) -> Path:
        """The case's path without a suffix."""
        return self._case

    def regions(self) -> tuple[str, ...]:
        """Return no names: an Eclipse-format case is one region."""
        return ()

    @cached_property
    def _grid(self) -> Grid:
        try:
            # The deck's length unit when the EGRID states none.
            return Grid.read(Path(f"{self._case}.EGRID"), self._units.length)
        except (OSError, ValueError) as error:
            raise SimulationError(f"cannot read {self._case.name}.EGRID: {error}") from error

    def corner_geometry(self, region: str | None = None) -> Grid:
        """Return the case's grid (EGRID) in SI units."""
        check_region(self._case, (), region)
        return self._grid

    def mesh(self, region: str | None = None) -> CaseMesh:
        """Return the active cells: their centres and volumes (SI), and the grid's shape."""
        grid = self.corner_geometry(region)
        return CaseMesh(
            cells=int(grid.actnum.sum()),
            centres=grid.active(grid.centres),
            volumes=grid.active(grid.volume).reshape(-1),
            dims=grid.shape,
            active=grid.actnum.ravel(order="F"),
        )

    def static_names(self, region: str | None = None) -> frozenset[str]:
        """Return the INIT's array names."""
        check_region(self._case, (), region)
        return frozenset(str(k).strip() for k in self.init.keys())  # noqa: SIM118 (a ResdataFile, not a dict: iterating it gives keywords)

    def static(self, name: str, region: str | None = None) -> NDArray[np.float64]:
        """Return an INIT array over the active cells, as written (deck units)."""
        check_region(self._case, (), region)
        if name not in self.init:
            raise SimulationError(f"{self._case.name}.INIT has no {name}")
        return self.array(self.init, name)

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        """Return the restarts' times, in the files' order."""
        check_region(self._case, (), region)
        return self._times

    def names(self, time: StateTime, region: str | None = None) -> frozenset[str]:
        """Return the names of the arrays in the restart at ``time``."""
        check_region(self._case, (), region)
        view = self._view(self._restart(time))
        # A separate file is a ResdataFile (keys), a unified file's block a view (unique_kw).
        found = view.keys() if hasattr(view, "keys") else view.unique_kw()
        return frozenset(str(k).strip() for k in found)

    def state(
        self,
        time: StateTime,
        names: Sequence[str],
        region: str | None = None,
        *,
        skip_missing: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        """Return the restart arrays ``names`` at ``time`` as written (deck units).

        With ``skip_missing``, those of ``names`` the restart holds, in their order.

        The restart is found by ``time.index`` (its report step), so a time from
        :meth:`at_step` serves as well as one from :meth:`times`.

        Raises
        ------
        MissingState
            If the run wrote no restart at ``time``.
        SimulationError
            If a name is not in it (unless ``skip_missing``), or the file cannot be read.
        """
        check_region(self._case, (), region)
        restart = self._restart(time)
        view = self._view(restart)
        absent = [n for n in names if n not in view]
        if skip_missing:
            names = [n for n in names if n not in absent]
        elif absent:
            raise SimulationError(
                f"{self._case.name} at {time.describe()} has no {', '.join(absent)}"
            )
        try:
            return {n: self.array(view, n) for n in names}
        except (OSError, ValueError) as error:
            raise SimulationError(
                f"cannot read {restart.path.name} at {time.describe()}: {error}"
            ) from error

    def unit(self, name: str) -> str | None:
        """Return the unit of an array the reader knows (deck units), else ``None``."""
        quantity = _QUANTITIES.get(name.upper())
        if quantity is None:
            return None
        if quantity == "fraction":
            return "1"
        if quantity == "permeability":
            return "mD"
        if quantity == "temperature":
            return self._units.temperature
        return _UNITS[self._units.name][quantity]

    @cached_property
    def run_end(self) -> dt.date | None:
        """The date the run's summary ends at, or ``None`` if it has no readable summary."""
        from resdata.summary import Summary

        try:
            end = Summary(str(self._case)).end_date
        except (OSError, ValueError):
            return None
        return end.date() if isinstance(end, dt.datetime) else end

    def reached(self) -> StateTime | None:
        """Return the summary's end (index -1: not an output), or ``None`` without one."""
        end = self.run_end
        if end is None:
            return None
        start = self._start_date
        seconds = None if start is None else float((end - start).days) * 86400.0
        return StateTime(-1, seconds, end)

    # -- CornerGeometryReader ----------------------------------------------------------

    @property
    def unit_system(self) -> UnitSystem:
        """The deck's unit system (INTEHEAD)."""
        return self._units

    @property
    def phases(self) -> tuple[str, ...]:
        """The deck's phases (INTEHEAD), in the order oil, water, gas."""
        return self._phases


# The deck simulators' reader: readers are keyed by the simulator's name.
case_state_readers.add("opm_flow", EclipseFiles)
case_state_readers.add("eclipse", EclipseFiles)
