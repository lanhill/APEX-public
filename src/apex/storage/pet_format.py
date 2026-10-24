"""Import PET-format inputs into APEX's native formats.

Ported from PET
---------------
PET module : src/misc/read_input_csv.py (DataReader.get_data, _extract_cell_variance)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : n/a (file format)
Differences: only ``['abs', v]`` (variance v) and ``['rel', p]`` (variance
             ``(0.01 p d)^2``) cells are supported; ``emp`` and ``cd`` raise. Data cells
             naming an ``.npz`` file are field data (PET loads its first array, squeezed),
             resolved against the configuration's directory (PET: its working directory).
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey, ObservationSet
from apex.storage.inputs import OBSERVATION_COLUMNS


@dataclass(frozen=True)
class PetScalar:
    """One scalar datum of a PET data file, with its variance."""

    time: str
    datatype: str
    value: float
    variance: float


@dataclass(frozen=True)
class PetField:
    """One field datum of a PET data file (an ``.npz`` cell) and its variance cell.

    ``variance_cell`` is the raw cell: PET never reads it for a compressed field, so it is
    parsed (:func:`field_variance`) only when needed.
    """

    time: str
    datatype: str
    source: Path
    values: NDArray[np.float64]
    variance_cell: str


def read_pet_data(data_csv: Path, var_csv: Path, base: Path) -> list[PetScalar | PetField]:
    """Read PET's wide ``data.csv`` and ``var.csv`` cell by cell, in PET's order.

    PET walks the table row (report label) by row, then column by column; it numbers
    compressed vintages in this order, so the order is kept. Empty, ``None`` and ``nan``
    cells are skipped. Cells ending in ``.npz`` are fields, resolved against ``base``.

    Raises
    ------
    ConfigError
        If the files disagree in shape or labels, a cell cannot be parsed, a variance
        cell is unsupported, or a field file is missing or not 1-D.
    """
    data_header, data_rows = _read_wide(data_csv)
    var_header, var_rows = _read_wide(var_csv)
    if data_header != var_header or [r[0] for r in data_rows] != [r[0] for r in var_rows]:
        raise ConfigError(f"{data_csv} and {var_csv} must have the same columns and index")
    entries: list[PetScalar | PetField] = []
    for data_row, var_row in zip(data_rows, var_rows, strict=True):
        time = data_row[0]
        for datatype, data_cell, var_cell in zip(
            data_header[1:], data_row[1:], var_row[1:], strict=True
        ):
            cell = data_cell.strip()
            if cell in ("", "None", "nan"):
                continue
            where = f"{var_csv}: {datatype}@{time}"
            if cell.endswith(".npz"):
                source = (base / cell).resolve()
                entries.append(PetField(time, datatype, source, _field(source), var_cell))
                continue
            try:
                value = float(cell)
            except ValueError as error:
                raise ConfigError(
                    f"{data_csv}: {datatype}@{time}: cannot parse {cell!r}"
                ) from error
            entries.append(PetScalar(time, datatype, value, _variance(var_cell, value, where)))
    return entries


def read_pet_observations(data_csv: Path, var_csv: Path) -> ObservationSet:
    """Read PET's wide ``data.csv`` and ``var.csv`` holding scalar data only.

    Empty cells and ``None`` data are skipped. The first column's values become
    :attr:`DataKey.time`, the other column names :attr:`DataKey.datatype`.

    Raises
    ------
    ConfigError
        If the files disagree in shape or labels, a variance cell is unsupported, or a
        cell is a field (use :func:`read_pet_data`).
    """
    entries = read_pet_data(data_csv, var_csv, data_csv.parent)
    fields = [f"{e.datatype}@{e.time}" for e in entries if isinstance(e, PetField)]
    if fields:
        raise ConfigError(f"{data_csv}: field data ({', '.join(fields)}) need read_pet_data")
    try:
        return scalar_observations([e for e in entries if isinstance(e, PetScalar)])
    except ValueError as error:
        raise ConfigError(f"{data_csv}: {error}") from error


def scalar_observations(scalars: list[PetScalar]) -> ObservationSet:
    """Build an :class:`ObservationSet` from scalar PET data, in the given order."""
    return ObservationSet(
        [DataKey(e.time, e.datatype) for e in scalars],
        [e.value for e in scalars],
        [e.variance for e in scalars],
    )


def write_observations(observations: ObservationSet, path: Path) -> None:
    """Write observations as APEX's long-format CSV (``time,datatype,value,variance``).

    Raises
    ------
    ValueError
        If the observations contain field data, which the CSV cannot hold.
    """
    fields = sorted({f"{k.datatype}@{k.time}" for k in observations.keys if k.index is not None})
    if fields:
        raise ValueError(f"field data cannot be written as CSV rows: {', '.join(fields)}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(OBSERVATION_COLUMNS)
        for key, value, variance in zip(
            observations.keys, observations.values, observations.variances, strict=True
        ):
            writer.writerow([key.time, key.datatype, repr(float(value)), repr(float(variance))])


def _read_wide(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise ConfigError(f"PET input not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise ConfigError(f"{path}: needs a header and at least one row")
    return rows[0], rows[1:]


def _variance(cell: str, value: float, where: str) -> float:
    kind, amount = _variance_cell(cell, where)
    return amount if kind == "abs" else (0.01 * amount * value) ** 2


def field_variance(entry: PetField, var_csv: Path) -> tuple[str, float]:
    """Parse a field's variance cell: ``("abs" | "rel", amount)``.

    Raises
    ------
    ConfigError
        If the cell cannot be parsed or is not ``abs``/``rel``.
    """
    return _variance_cell(entry.variance_cell, f"{var_csv}: {entry.datatype}@{entry.time}")


def _variance_cell(cell: str, where: str) -> tuple[str, float]:
    try:
        kind, amount = ast.literal_eval(cell)
        amount = float(amount)
    except (ValueError, SyntaxError, TypeError) as error:
        raise ConfigError(f"{where}: cannot parse variance cell {cell!r}") from error
    kind = str(kind).lower()
    if kind not in ("abs", "rel"):
        raise ConfigError(f"{where}: unsupported variance type {kind!r} (only 'abs' and 'rel')")
    return kind, amount


def _field(path: Path) -> NDArray[np.float64]:
    """Load a PET field cell: the first array of an ``.npz``, squeezed, 1-D."""
    if not path.is_file():
        raise ConfigError(f"PET field data file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        values = np.squeeze(np.asarray(data[data.files[0]], dtype=np.float64))
    if values.ndim != 1:
        raise ConfigError(f"{path}: a field must be a 1-D array, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        # PET would skip such a field when numbering compressed vintages; refuse it here.
        raise ConfigError(f"{path}: field data must be finite")
    return values
