"""A simulated case's state as data: times, meshes, grids and units.

What a :class:`~apex.core.case_state.CaseStateReader` gives, with no I/O: the times a run
wrote a state at (:class:`StateTime`), its cells (:class:`CaseMesh`), a corner-point grid's
geometry (:class:`CornerGrid`, for the consumers that need one), a deck's unit system
(:class:`UnitSystem`), and the one conversion of a unit to SI (:func:`to_si`).

Arrays are in the units the simulator wrote (each simulator's own); a reader
states each array's unit, and :func:`to_si` converts one for a consumer that needs SI.

Shapes: grid arrays are ``(n_x, n_y, n_z)`` in Eclipse (F) order, i fastest; cell arrays
``(n_cells,)``, in the simulator's own (active) order.
"""

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SI_UNITS",
    "CaseMesh",
    "CornerGrid",
    "StateTime",
    "UnitSystem",
    "to_si",
]


@dataclass(frozen=True)
class StateTime:
    """One state a run wrote.

    Attributes
    ----------
    index
        The simulator's own number of the output: the report step for Eclipse-format runs.
        Two states on one day are two entries with their own indices.
    seconds
        The time since the run's start (s), or ``None`` when the run's files do not state
        its start.
    date
        The state's date when the run has a calendar, else ``None``.
    """

    index: int
    seconds: float | None
    date: dt.date | None

    def describe(self) -> str:
        """Return the time for messages: its date, else its seconds, with its index."""
        if self.date is not None:
            return f"{self.date.isoformat()} (output {self.index})"
        if self.seconds is not None:
            return f"{self.seconds:g} s (output {self.index})"
        return f"output {self.index}"


@dataclass(frozen=True)
class CaseMesh:
    """A case's cells: what every reader gives.

    Attributes
    ----------
    cells
        The number of (active) cells the state arrays hold.
    centres
        ``(cells, 3)``: each cell's x, y and depth or z (m).
    volumes
        ``(cells,)``: each cell's bulk volume (m³).
    dims
        ``(n_x, n_y, n_z)`` of a structured grid, ``None`` for an unstructured mesh.
    active
        ``(n_x * n_y * n_z,)`` in Eclipse (F) order: which of a structured grid's cells the
        arrays hold; ``None`` for an unstructured mesh (its arrays hold every cell, in the
        simulator's own order).
    """

    cells: int
    centres: NDArray[np.float64]
    volumes: NDArray[np.float64]
    dims: tuple[int, int, int] | None = None
    active: NDArray[np.bool_] | None = None

    @property
    def structured(self) -> bool:
        """Whether the mesh is a structured grid (``dims`` given)."""
        return self.dims is not None


@dataclass(frozen=True)
class CornerGrid:
    """A corner-point grid's geometry in SI units (the capability ``corner_geometry``).

    Attributes
    ----------
    shape
        ``(n_x, n_y, n_z)``.
    actnum
        ``(n_x, n_y, n_z)``: whether each cell is active.
    top, bottom
        ``(n_x, n_y, n_z)``: each cell's top and base depth (m, positive down), the means
        of its four upper and four lower corners; defined for inactive cells too.
    centres
        ``(n_x, n_y, n_z, 3)``: each cell's x, y and depth (m), the means of its eight
        corners.
    boxes
        ``(n_x, n_y, n_z, 4)``: the horizontal extent of each cell's corners, ``x_min,
        x_max, y_min, y_max`` (m).
    volume
        ``(n_x, n_y, n_z)``: each cell's bulk volume (m³).
    """

    shape: tuple[int, int, int]
    actnum: NDArray[np.bool_]
    top: NDArray[np.float64]
    bottom: NDArray[np.float64]
    centres: NDArray[np.float64]
    boxes: NDArray[np.float64]
    volume: NDArray[np.float64]

    @property
    def active_columns(self) -> NDArray[np.bool_]:
        """``(n_x, n_y)``: whether each column has an active cell."""
        out: NDArray[np.bool_] = self.actnum.any(axis=2)
        return out

    def active(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return a grid array's values ``(n_x, n_y, n_z, ...)`` over the active cells.

        In the simulators' active order, F (i fastest): ``(n_active, ...)``.
        """
        flat = values.reshape(int(np.prod(self.shape)), -1, order="F")
        out: NDArray[np.float64] = flat[self.actnum.ravel(order="F")]
        return out.reshape((-1, *values.shape[3:]))


@dataclass(frozen=True)
class UnitSystem:
    """A deck's unit system, as factors to SI (Pa, kg/m³, m, and m³ of reservoir volume).

    ``temperature`` is the deck's temperature unit (``"C"`` or ``"F"``), which is not a
    factor (an offset too): convert it with :func:`to_si`.
    """

    name: str
    pressure: float
    density: float
    length: float
    volume: float
    temperature: str = "C"


#: Each unit's conversion to SI, ``(factor, offset)``: ``SI = value * factor + offset``.
#: The factors of the deck unit systems are those the readers have always used (bar
#: 1e5 Pa, psia 6894.757293168361 Pa, ...), so a conversion is bit-identical to theirs.
SI_UNITS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        # Dimensionless: fractions (saturations, porosity, net-to-gross).
        "1": (1.0, 0.0),
        # Pressure.
        "Pa": (1.0, 0.0),
        "bar": (1.0e5, 0.0),
        "psia": (6894.757293168361, 0.0),
        "atm": (101325.0, 0.0),
        # Density.
        "kg/m3": (1.0, 0.0),
        "lb/ft3": (16.018463373960138, 0.0),
        "g/cm3": (1000.0, 0.0),
        # Length.
        "m": (1.0, 0.0),
        "ft": (0.3048, 0.0),
        "cm": (0.01, 0.0),
        # Volume (at reservoir conditions, as Eclipse's RPORV).
        "m3": (1.0, 0.0),
        "bbl": (0.158987294928, 0.0),
        "cm3": (1.0e-6, 0.0),
        # Temperature (K).
        "K": (1.0, 0.0),
        "C": (1.0, 273.15),
        "F": (1.0 / 1.8, 459.67 / 1.8),
        # Permeability (m²).
        "m2": (1.0, 0.0),
        "mD": (9.869233e-16, 0.0),
    }
)


def to_si(values: NDArray[np.float64], unit: str | None) -> NDArray[np.float64]:
    """Return ``values`` in SI from ``unit`` (a key of :data:`SI_UNITS`).

    A pure factor is applied alone, never with a zero offset added, so that a conversion is
    bit-identical to ``values * factor`` (adding 0.0 would turn -0.0 into 0.0).

    Raises
    ------
    ValueError
        If the unit is not known (``None``: the reader cannot state it), naming the known.
    """
    if unit is None or unit not in SI_UNITS:
        raise ValueError(
            f"no conversion to SI from {unit!r} (known: {', '.join(SI_UNITS)}); the array is "
            "as its simulator wrote it"
        )
    factor, offset = SI_UNITS[unit]
    out: NDArray[np.float64] = values * factor
    if offset:
        out = out + offset
    return out
