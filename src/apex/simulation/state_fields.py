"""State fields of a simulated case, as field keys: field surrogates' snapshots.

A surrogate build with ``[case] fields = ["PRESSURE", "SWAT"]`` and ``field_times`` wraps the
simulator in a :class:`~apex.simulation.geophysics.GeophysicsSimulator` with
:class:`StateFieldModel`: the summary keys go to the simulator, and the field keys
``DataKey(time, name, cell)`` are read here from the run's states through the simulator's
case-state reader (which the services choose), by its ``mesh()``, ``times()`` and
``state()`` alone: no corner geometry is needed. The user approved
this core change on 2026-11-02, for the field surrogates.

- **A structured grid** (``CaseMesh.dims``; the deck simulators' Eclipse-format files): each
  field is given on the whole grid ``(nx * ny * nz,)``, in Eclipse (F) order, i fastest;
  inactive cells are 0.
- **An unstructured mesh** (``dims`` ``None``): each field is given over the mesh's cells in
  the reader's own order, with no padding (decision 5: never mapped onto a grid).

Values are as the simulator wrote them, in its units (a deck's pressure in bar
or psia, saturations as fractions). A field time is a date: the state written on it, the
later of two on one day (as geophysics takes them, the summary's report step's last); or a
number of seconds since the run's start (for a run without a calendar):
the state whose ``StateTime.seconds`` equals it to the millisecond, the later of two. A
build's times are all dates or all seconds. ``region`` names the region of a multi-region
case the states are read from (``[case] field_region``); ``None`` is a single region's.

``connectivity_file``: for a surrogate that takes the runs' cells and
faces (``uses_connectivity``), the first run read whose reader gives its ``connectivity``
writes its mesh and cell graph there (``apex.domain.cell_graph.cell_graph_arrays``, a
``.npz`` written whole or not at all); later runs leave it as it is (one mesh for every
run: the snapshots' cell counts are checked per run). A reader that cannot give its faces
leaves no file, and the member's fields stand (logged). A build without fields records the
same file through :class:`CellGraphRecorder`, which wraps the simulator for that alone.
"""

import datetime as dt
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.case_state import CaseStateReader, CaseStateReaderFactory, ConnectivityReader
from apex.core.errors import ConfigError, SimulationError
from apex.core.simulation import Simulator
from apex.domain.case_state import (
    StateAt,
    StateTime,
    describe_state_time,
    parse_state_times,
    same_instant,
)
from apex.domain.cell_graph import cell_graph_arrays
from apex.domain.observations import DataKey
from apex.simulation.geophysics import GeophysicsSimulator

__all__ = [
    "CellGraphRecorder",
    "StateFieldModel",
    "field_keys",
    "record_cell_graph",
    "split_fields",
]

logger = logging.getLogger(__name__)


def field_keys(names: Sequence[str], times: Sequence[str], cells: int) -> tuple[DataKey, ...]:
    """Return the field keys of ``names`` at ``times``, each over ``cells``, field-major.

    Order: field, then time, then cell: the order :func:`split_fields` reads back.
    """
    return tuple(
        DataKey(time, name, cell) for name in names for time in times for cell in range(cells)
    )


def split_fields(
    values: NDArray[np.float64], names: Sequence[str], times: Sequence[str], cells: int
) -> dict[str, NDArray[np.float64]]:
    """Return ``{name: (cells, len(times), n)}`` from values ``(n_field_keys, n)`` in key order."""
    array = np.asarray(values, dtype=np.float64)
    n = array.shape[-1] if array.ndim == 2 else 1
    shaped = array.reshape(len(names), len(times), cells, n)
    return {
        name: np.ascontiguousarray(shaped[i].transpose(1, 0, 2)) for i, name in enumerate(names)
    }


class StateFieldModel:
    """States' arrays at given dates as field keys (module docstring).

    Parameters
    ----------
    names
        The state arrays (``PRESSURE``, ``SWAT``, ``SGAS``, ...).
    times
        The times the states are read at: all dates (ISO, ``YYYY-MM-DD``) or all seconds
        since the run's start (numbers as text).
    cells
        A structured grid's cells ``nx * ny * nz``, or an unstructured mesh's cells.
    reader
        The simulator's case-state reader, chosen by the services; ``None``
        reads the Eclipse-format files.
    structured
        Whether the build's surrogate needs a structured grid (the services' choice, from
        the plugin): an unstructured mesh is then refused, naming the reason.
    region
        The case's region the states are read from (``None``: its only one).
    connectivity_file
        Where the first run read writes its mesh and cell graph (module docstring), when its
        reader gives them; ``None``: not written.

    Raises
    ------
    ConfigError
        If a time is neither a date nor seconds, the times mix them, or no field is named.
    """

    def __init__(
        self,
        names: Sequence[str],
        times: Sequence[str],
        cells: int,
        reader: CaseStateReaderFactory | None = None,
        *,
        structured: bool = False,
        region: str | None = None,
        connectivity_file: Path | None = None,
    ) -> None:
        if not names:
            raise ConfigError("state fields: name at least one restart array")
        self.names = tuple(str(n).upper() for n in names)
        self.times = tuple(str(t) for t in times)
        self.cells = int(cells)
        self.reader = reader
        self.structured = structured
        self.region = region
        self.connectivity_file = None if connectivity_file is None else Path(connectivity_file)
        try:
            parsed = parse_state_times(self.times)
        except ValueError as error:
            raise ConfigError(
                f"state fields: field_times must be ISO dates or seconds ({error})"
            ) from None
        self._times: dict[str, StateAt] = dict(zip(self.times, parsed, strict=True))

    def owns(self, key: DataKey) -> bool:
        """Return whether ``key`` is one of the fields' keys."""
        return key.datatype in self.names and key.index is not None and key.time in self._times

    def _open(self, case: Path) -> CaseStateReader:
        if self.reader is not None:
            return self.reader(case)
        from apex.simulation.eclipse_files import EclipseFiles

        return EclipseFiles(case)

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the fields' values at ``keys`` from the case's states, ``(len(keys),)``.

        Raises
        ------
        ConfigError
            If the mesh is not the one the keys assume (or not a structured grid when the
            surrogate needs one), or a state lacks an array (every member's would: the
            simulator's output must write it).
        SimulationError
            If the run ended before a date, or its files are broken (this member).
        """
        reader = self._open(case)
        region = self.region
        mesh = reader.mesh(region) if region is not None else reader.mesh()
        if mesh.dims is None:
            if self.structured:
                raise ConfigError(
                    f"state fields: {reader.case.name}'s mesh is unstructured, and the build's "
                    "surrogate needs a structured grid (decision 5: never mapped onto "
                    "one)"
                )
            size, active = mesh.cells, None
            if size != self.cells:
                raise ConfigError(
                    f"state fields: the run's mesh has {size} cells, the build {self.cells}"
                )
        else:
            size = int(np.prod(mesh.dims))
            active = np.ones(size, dtype=bool) if mesh.active is None else mesh.active
            if size != self.cells:
                raise ConfigError(
                    f"state fields: the run's grid has {size} cells, the build {self.cells}"
                )
        n_cells = mesh.cells if active is None else int(active.sum())
        self._record_connectivity(reader)
        written = reader.times(region) if region is not None else reader.times()
        full: dict[tuple[str, str], NDArray[np.float64]] = {}
        for time in sorted({k.time for k in keys}):
            at = _state_at(reader, written, self._times[time])
            arrays = (
                reader.state(at, self.names, region, skip_missing=True)
                if region is not None
                else reader.state(at, self.names, skip_missing=True)
            )
            for name in {k.datatype for k in keys if k.time == time}:
                if name not in arrays:
                    raise ConfigError(
                        f"state fields: the state at {time} has no {name}: the simulator must "
                        "write it (for a deck, its RPTRST: e.g. RPTRST BASIC=2 with the "
                        "field's mnemonic)"
                    )
                found = np.asarray(arrays[name], dtype=np.float64).reshape(-1)
                if found.size != n_cells:
                    raise ConfigError(
                        f"state fields: {name} at {time} has {found.size} values, the mesh "
                        f"{n_cells} active cells (dual porosity, local grids or another "
                        "grid's states are not read)"
                    )
                if active is None:
                    values = found
                else:
                    values = np.zeros(self.cells)
                    values[active] = found
                full[(time, name)] = values
        return np.array([full[(k.time, k.datatype)][int(k.index or 0)] for k in keys])

    def _record_connectivity(self, reader: CaseStateReader) -> None:
        """Write the run's mesh and cell graph to ``connectivity_file`` once (module docstring)."""
        if self.connectivity_file is not None:
            record_cell_graph(reader, self.connectivity_file, self.region)


def record_cell_graph(reader: CaseStateReader, target: Path, region: str | None = None) -> None:
    """Write ``reader``'s mesh and cell graph to ``target`` unless it exists (module docstring).

    Nothing is written when the reader does not declare ``connectivity``, or when it cannot
    give its faces (logged: the member stands, and a surrogate that needs the faces refuses
    with its own reason).
    """
    if target.exists() or not isinstance(reader, ConnectivityReader):
        return
    if "connectivity" not in frozenset(getattr(type(reader), "capabilities", ())):
        return  # the method alone is no declared capability
    try:
        mesh = reader.mesh(region) if region is not None else reader.mesh()
        graph = reader.connectivity(region) if region is not None else reader.connectivity()
        arrays = cell_graph_arrays(mesh, graph)
    except (ConfigError, SimulationError, ValueError, OSError, LookupError) as error:
        logger.warning(
            "state fields: %s gives no cell graph (%s); none is recorded",
            reader.case.name,
            error,
        )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed: members read in parallel never see half a file.
    handle, name = tempfile.mkstemp(prefix=".connectivity-", suffix=".npz", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, allow_pickle=False, **arrays)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


class _NoFields:
    """A case-field model that owns no key: :class:`CellGraphRecorder` reads no field."""

    def owns(self, key: DataKey) -> bool:
        """Return ``False``: every key is the wrapped simulator's."""
        return False

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Never called (no key is owned)."""
        raise AssertionError("a cell-graph recorder owns no keys")


class CellGraphRecorder(GeophysicsSimulator):
    """A case-file simulator whose first run read also records its cell graph.

    For a surrogate build without ``[case] fields`` whose plugin takes the runs' cells and
    faces (``uses_connectivity``): every key is the wrapped simulator's, and after a member's
    reading, while its files are still there, the case-state ``reader`` is opened on the
    case and its mesh and cell graph written to ``target`` (:func:`record_cell_graph`; once,
    the first run read). Everything about where and how the run happens is the wrapped
    simulator's, as for :class:`StateFieldModel`'s stack.
    """

    def __init__(self, inner: Simulator, reader: CaseStateReaderFactory, target: Path) -> None:
        super().__init__(inner, _NoFields())
        self.reader = reader
        self.target = Path(target)
        # What executors read of the simulator stays the wrapped one's (its work directories,
        # its members' time limit), so a field-less build runs as it did unwrapped.
        self.needs_work_dir = bool(getattr(inner, "needs_work_dir", False))
        if hasattr(inner, "timeout_s"):
            self.timeout_s = getattr(inner, "timeout_s")  # noqa: B009 (the protocol has none)

    @property
    def stops_at_report_dates(self) -> bool:
        """The wrapped simulator's: a member's stop date reaches it unchanged."""
        return bool(getattr(self.inner, "stops_at_report_dates", False))

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the wrapped simulator's values, recording the cell graph the first time.

        The member never fails for the graph: a case the reader cannot open is logged and
        records none (the next member's may).
        """
        values = super().read(case, keys)
        if not self.target.exists():
            try:
                reader = self.reader(case)
            except (ConfigError, SimulationError, ValueError, OSError) as error:
                logger.warning(
                    "state fields: %s cannot be read for its cell graph (%s); none is recorded",
                    case.name,
                    error,
                )
                return values
            record_cell_graph(reader, self.target)
        return values


def _by_date(written: Sequence[StateTime]) -> dict[dt.date, StateTime]:
    """Each date's state: the later of two on one day, as geophysics' ``CaseState``.

    The summary's values are its report steps' last (``deck.report_values``), so the later
    state of a day is the one that matches them.
    """
    found: dict[dt.date, StateTime] = {}
    for time in written:
        if time.date is not None:
            found[time.date] = time
    return found


def _state_at(reader: CaseStateReader, written: Sequence[StateTime], time: StateAt) -> StateTime:
    """Return the state at a date or at seconds, or tell whose error its absence is."""
    if isinstance(time, dt.date):
        return _state_on(reader, _by_date(written), time)
    found = [t for t in written if t.seconds is not None and same_instant(t.seconds, time)]
    if found:
        return found[-1]  # the later of two at one millisecond, as of two on one day
    shown = ", ".join(f"{t.seconds:.15g}" for t in written if t.seconds is not None) or "none"
    message = f"{reader.case.name} has no state at {describe_state_time(time)} (written: {shown})"
    reached = reader.reached()
    end = None if reached is None else reached.seconds
    if end is None:
        raise SimulationError(
            f"{message}, and nothing tells how far it ran (if every member fails so, check "
            "that the simulator writes a state at every field time)"
        )
    if end < time and not same_instant(end, time):
        raise SimulationError(f"{message}; the run ended at {end:.15g} s")
    raise ConfigError(f"{message}; the simulator must write a state at every field time")


def _state_on(
    reader: CaseStateReader, states: dict[dt.date, StateTime], date: dt.date
) -> StateTime:
    """Return the state at ``date``, or tell whose error its absence is.

    Raises
    ------
    ConfigError
        If the run reached ``date`` but wrote no state there: its input does not ask for one,
        so every member would fail the same way.
    SimulationError
        If the run ended before ``date`` (this member), or nothing tells how far it ran.
    """
    if date in states:
        return states[date]
    written = ", ".join(d.isoformat() for d in sorted(states)) or "none"
    message = f"{reader.case.name} has no state at {date.isoformat()} (written: {written})"
    reached = reader.reached()
    end = None if reached is None else reached.date
    if end is None:
        raise SimulationError(
            f"{message}, and nothing tells how far it ran (if every member fails so, check "
            "that the simulator writes a state at every field time)"
        )
    if end < date:
        raise SimulationError(f"{message}; the run ended on {end.isoformat()}")
    raise ConfigError(
        f"{message}; the simulator must write a state at every field time (for a deck, "
        "RPTRST BASIC=2 with a report step at each date)"
    )
