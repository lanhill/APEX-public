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
later of two on one day (as geophysics takes them, the summary's report step's last).
"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.case_state import CaseStateReader, CaseStateReaderFactory
from apex.core.errors import ConfigError, SimulationError
from apex.domain.case_state import StateTime
from apex.domain.observations import DataKey

__all__ = ["StateFieldModel", "field_keys", "split_fields"]


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
        The dates (ISO, ``YYYY-MM-DD``) the states are read at.
    cells
        A structured grid's cells ``nx * ny * nz``, or an unstructured mesh's cells.
    reader
        The simulator's case-state reader, chosen by the services; ``None``
        reads the Eclipse-format files.
    structured
        Whether the build's surrogate needs a structured grid (the services' choice, from
        the plugin): an unstructured mesh is then refused, naming the reason.

    Raises
    ------
    ConfigError
        If a time is not a date, or no field is named.
    """

    def __init__(
        self,
        names: Sequence[str],
        times: Sequence[str],
        cells: int,
        reader: CaseStateReaderFactory | None = None,
        *,
        structured: bool = False,
    ) -> None:
        if not names:
            raise ConfigError("state fields: name at least one restart array")
        self.names = tuple(str(n).upper() for n in names)
        self.times = tuple(str(t) for t in times)
        self.cells = int(cells)
        self.reader = reader
        self.structured = structured
        try:
            self._dates = {t: dt.date.fromisoformat(t) for t in self.times}
        except ValueError as error:
            raise ConfigError(f"state fields: field_times must be ISO dates ({error})") from None

    def owns(self, key: DataKey) -> bool:
        """Return whether ``key`` is one of the fields' keys."""
        return key.datatype in self.names and key.index is not None and key.time in self._dates

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
        mesh = reader.mesh()
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
        states = _by_date(reader)
        full: dict[tuple[str, str], NDArray[np.float64]] = {}
        for time in sorted({k.time for k in keys}):
            arrays = reader.state(
                _state_at(reader, states, self._dates[time]), self.names, skip_missing=True
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


def _by_date(reader: CaseStateReader) -> dict[dt.date, StateTime]:
    """Each date's state: the later of two on one day, as geophysics' ``CaseState``.

    The summary's values are its report steps' last (``deck.report_values``), so the later
    state of a day is the one that matches them.
    """
    found: dict[dt.date, StateTime] = {}
    for time in reader.times():
        if time.date is not None:
            found[time.date] = time
    return found


def _state_at(
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
