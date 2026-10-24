"""Settings for every test; the shared fixtures come from :mod:`apex.testing`,
which the repository's own conftest.py loads."""

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

# --- deck-run outputs for the geophysics tests ------------------------------
# Tiny EGRID, INIT and restart files written with resdata: what a run of a box grid would
# leave, with the given porosity, pressure, saturations and densities. No simulator runs.

#: INTEHEAD's length in the files Flow writes (only items 2, 14 and 64-66 are read).
_INTEHEAD = 411


def _kw(name: str, values: ArrayLike, kind: str) -> object:
    from resdata import ResDataType
    from resdata.resfile import ResdataKW

    data = np.asarray(values).ravel()
    types = {"int": ResDataType.RD_INT, "float": ResDataType.RD_FLOAT}
    kw = ResdataKW(name, data.size, types[kind])
    for i, v in enumerate(data):
        kw[i] = v.item()
    return kw


def _write(path: Path, keywords: Sequence[object]) -> None:
    from resdata.resfile import FortIO, openFortIO

    with openFortIO(str(path), FortIO.WRITE_MODE) as f:
        for kw in keywords:
            kw.fwrite(f)  # type: ignore[attr-defined]


def _header(units: int = 1, phases: int = 3, date: dt.date | None = None) -> object:
    values = np.zeros(_INTEHEAD, dtype=np.int32)
    values[2], values[14] = units, phases
    if date is not None:
        values[64:67] = (date.day, date.month, date.year)
    return _kw("INTEHEAD", values, "int")


def _write_grid(
    path: Path,
    shape: tuple[int, int, int],
    cell: tuple[float, float, float] = (100.0, 100.0, 5.0),
    top: float = 2000.0,
    actnum: ArrayLike | None = None,
    thickness: ArrayLike | None = None,
    units: int = 1,
    mapaxes: tuple[float, float, float, float, float, float] | None = None,
) -> Path:
    """Write a box grid's EGRID and return its path.

    ``mapaxes`` (Eclipse's MAPAXES: a point on the y axis, the origin, a point on the x
    axis, in map coordinates) maps the grid's frame to the map's.

    ``actnum`` and ``thickness`` (per layer, default ``cell[2]``) in F order; the top
    layer starts at depth ``top``. ``units`` is INTEHEAD's unit system, whose length the
    lengths are in (its GRIDUNIT: METRES, FEET or CM).
    """
    from resdata import UnitSystem
    from resdata.grid import Grid

    nx, ny, nz = shape
    dz = np.full(nz, cell[2]) if thickness is None else np.asarray(thickness, dtype=float)
    bounds = top + np.concatenate(([0.0], np.cumsum(dz)))
    x, y = np.meshgrid(np.arange(nx + 1) * cell[0], np.arange(ny + 1) * cell[1], indexing="xy")
    coord = np.column_stack(
        (x.ravel(), y.ravel(), np.full(x.size, bounds[0]),
         x.ravel(), y.ravel(), np.full(x.size, bounds[-1]))
    )  # fmt: skip
    # ZCORN: per layer its top then its base, each 2nx x 2ny corners.
    zcorn = np.concatenate([np.full(4 * nx * ny, z) for k in range(nz) for z in bounds[k : k + 2]])
    act = np.ones(nx * ny * nz, dtype=int) if actnum is None else np.asarray(actnum, dtype=int)
    # resdata holds a grid in metres and converts it to output_unit's length on saving.
    metres = {1: 1.0, 2: 0.3048, 3: 0.01}[units]  # PVT-M: resdata writes no GRIDUNIT
    coord, zcorn = coord * metres, zcorn * metres
    grid = Grid.create(
        shape, _kw("ZCORN", zcorn, "float"), _kw("COORD", coord, "float"),
        _kw("ACTNUM", act, "int"), None if mapaxes is None else _kw("MAPAXES", mapaxes, "float"),
    )  # fmt: skip
    grid.save_EGRID(str(path), output_unit=UnitSystem(units))
    return path


def _write_case(
    case: Path,
    shape: tuple[int, int, int],
    restarts: Mapping[dt.date, Mapping[str, ArrayLike]],
    *,
    porosity: ArrayLike = 0.25,
    actnum: ArrayLike | None = None,
    units: int = 1,
    phases: int = 3,
    unified: bool = False,
    cell: tuple[float, float, float] = (100.0, 100.0, 5.0),
    top: float = 2000.0,
    summary_end: dt.date | None = None,
) -> Path:
    """Write ``case``'s EGRID, INIT and restarts (arrays over the active cells).

    With ``summary_end``, also a summary from the first restart's date to it (FOPT only).

    ``restarts`` maps each date to its arrays (``PRESSURE``, ``SWAT``, ``OIL_DEN``, ...),
    written in order as ``.X0000``, ``.X0001``, ... or one ``.UNRST``. Depths are in the
    unit system's length (``cell`` and ``top`` as given).
    """
    case.parent.mkdir(parents=True, exist_ok=True)
    _write_grid(case.with_suffix(".EGRID"), shape, cell, top, actnum, units=units)
    act = (
        np.ones(int(np.prod(shape)), dtype=bool)
        if actnum is None
        else np.asarray(actnum, dtype=bool).ravel()
    )
    n_active = int(act.sum())
    k = np.arange(int(np.prod(shape))) // (shape[0] * shape[1])
    depth = (top + (k + 0.5) * cell[2])[act]
    poro = np.broadcast_to(np.asarray(porosity, dtype=float), (n_active,))
    _write(
        case.with_suffix(".INIT"),
        [_header(units, phases), _kw("PORO", poro, "float"), _kw("DEPTH", depth, "float")],
    )
    blocks = []
    for step, (date, arrays) in enumerate(restarts.items()):
        block = [_header(units, phases, date)]
        block += [
            _kw(name, np.broadcast_to(np.asarray(v, dtype=float), (n_active,)), "float")
            for name, v in arrays.items()
        ]
        if unified:
            blocks += [_kw("SEQNUM", [step], "int"), *block]
        else:
            _write(case.parent / f"{case.name}.X{step:04d}", block)
    if unified:
        _write(case.parent / f"{case.name}.UNRST", blocks)
    if summary_end is not None:
        from resdata.summary import Summary

        start = min(restarts)
        nx, ny, nz = shape
        writer = Summary.writer(str(case), dt.datetime(start.year, start.month, start.day),
                                nx, ny, nz)  # fmt: skip
        writer.add_variable("FOPT")
        writer.add_t_step(1, float((summary_end - start).days))["FOPT"] = 0.0
        writer.fwrite()
    return case


@pytest.fixture
def write_case() -> Callable[..., Path]:
    """Return the writer of a deck run's EGRID, INIT and restarts (see ``_write_case``)."""
    pytest.importorskip("resdata")
    return _write_case


@pytest.fixture
def write_grid() -> Callable[..., Path]:
    """Return the writer of a box grid's EGRID (see ``_write_grid``)."""
    pytest.importorskip("resdata")
    return _write_grid
