"""State fields of a deck run, as field keys: the snapshots of field surrogates.

A surrogate build with ``[case] fields = ["PRESSURE", "SWAT"]`` and ``field_times`` wraps the
deck simulator (``opm_flow``, ``eclipse``) in a
:class:`~apex.simulation.geophysics.GeophysicsSimulator` with :class:`StateFieldModel`: the
summary keys go to the deck simulator, and the field keys ``DataKey(time, name, cell)`` are
read here from the run's restart files (:class:`~apex.simulation.geophysics.CaseState`,
restarts found by their own date). The user approved this core change on 2026-11-02, for
``unet_fields``.

Each field is given on the whole grid ``(nx * ny * nz,)``, in Eclipse (F) order, i fastest;
inactive cells are 0. Values are in the deck's units (pressure in bar or psia, saturations as
fractions).
"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.simulation.geophysics import CaseState

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
    """Restart arrays at given dates as field keys (module docstring).

    Parameters
    ----------
    names
        The restart arrays (``PRESSURE``, ``SWAT``, ``SGAS``, ...).
    times
        The dates (ISO, ``YYYY-MM-DD``) the restarts are read at.
    cells
        The grid's cells ``nx * ny * nz``.

    Raises
    ------
    ConfigError
        If a time is not a date, or no field is named.
    """

    def __init__(self, names: Sequence[str], times: Sequence[str], cells: int) -> None:
        if not names:
            raise ConfigError("state fields: name at least one restart array")
        self.names = tuple(str(n).upper() for n in names)
        self.times = tuple(str(t) for t in times)
        self.cells = int(cells)
        try:
            self._dates = {t: dt.date.fromisoformat(t) for t in self.times}
        except ValueError as error:
            raise ConfigError(f"state fields: field_times must be ISO dates ({error})") from None

    def owns(self, key: DataKey) -> bool:
        """Return whether ``key`` is one of the fields' keys."""
        return key.datatype in self.names and key.index is not None and key.time in self._dates

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the fields' values at ``keys`` from the case's restarts, ``(len(keys),)``.

        Raises
        ------
        ConfigError
            If the grid is not the one the keys assume, or a restart lacks an array (every
            member's would: the deck's RPTRST must write it).
        SimulationError
            If the run ended before a date, or its files are broken (this member).
        """
        state = CaseState(case)
        grid = state.grid
        active = grid.actnum.ravel(order="F")
        if active.size != self.cells:
            raise ConfigError(
                f"state fields: the run's grid has {active.size} cells, the build {self.cells}"
            )
        full: dict[tuple[str, str], NDArray[np.float64]] = {}
        for time in sorted({k.time for k in keys}):
            arrays = state.restart(self._dates[time], self.names)
            for name in {k.datatype for k in keys if k.time == time}:
                if name not in arrays:
                    raise ConfigError(
                        f"state fields: the restart at {time} has no {name}: the deck's RPTRST "
                        "must write it (e.g. RPTRST BASIC=2 with the field's mnemonic)"
                    )
                found = np.asarray(arrays[name], dtype=np.float64).reshape(-1)
                if found.size != int(active.sum()):
                    raise ConfigError(
                        f"state fields: {name} at {time} has {found.size} values, the grid "
                        f"{int(active.sum())} active cells (dual porosity, local grids or "
                        "another grid's restart are not read)"
                    )
                values = np.zeros(self.cells)
                values[active] = found
                full[(time, name)] = values
        return np.array([full[(k.time, k.datatype)][int(k.index or 0)] for k in keys])
