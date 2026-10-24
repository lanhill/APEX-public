"""Readers for run inputs: observations and prior ensembles."""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.config import ObservationsSection
from apex.domain.observations import DataKey, ObservationSet
from apex.domain.state import StateLayout

OBSERVATION_COLUMNS = ("time", "datatype", "value", "variance")


def read_observations(path: Path) -> ObservationSet:
    """Read a long-format observation CSV with columns ``time,datatype,value,variance``.

    Raises
    ------
    ConfigError
        If the file is missing, has other columns, or holds invalid values.
    """
    if not path.is_file():
        raise ConfigError(f"observation file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        # The errors name what is wrong, never the file's text: it may reach a client that
        # named a file that is not an observation file (release tester N3, 2026-10-20).
        if tuple(reader.fieldnames or ()) != OBSERVATION_COLUMNS:
            raise ConfigError(
                f"{path}: expected the header line {','.join(OBSERVATION_COLUMNS)}, and the "
                f"first line differs (it has {len(reader.fieldnames or ())} columns)"
            )
        rows = list(reader)
    for number, row in enumerate(rows, start=2):
        for column in ("value", "variance"):
            try:
                float(row[column] or "")
            except (TypeError, ValueError):
                raise ConfigError(f"{path}, line {number}: {column} is not a number") from None
    keys = [DataKey(row["time"], row["datatype"]) for row in rows]
    values = [float(row["value"]) for row in rows]
    variances = [float(row["variance"]) for row in rows]
    try:
        return ObservationSet(keys, values, variances)
    except ValueError as error:
        raise ConfigError(f"{path}: {error}") from error


def read_observation_section(section: ObservationsSection) -> ObservationSet:
    """Read the scalar CSV and the field files of ``[observations]``, in that order.

    Raises
    ------
    ConfigError
        If a file is missing or invalid, a field is not a 1-D finite array, or data repeat.
    """
    keys: list[DataKey] = []
    values: list[float] = []
    variances: list[float] = []
    if section.file is not None:
        scalar = read_observations(section.file)
        keys.extend(scalar.keys)
        values.extend(scalar.values.tolist())
        variances.extend(scalar.variances.tolist())
    for entry in section.fields:
        field = read_field(entry.file, entry.key)
        keys.extend(DataKey(entry.time, entry.datatype, i) for i in range(field.size))
        values.extend(field.tolist())
        variances.extend([entry.variance] * field.size)
    try:
        return ObservationSet(keys, values, variances)
    except ValueError as error:
        raise ConfigError(f"[observations]: {error}") from error


def read_field(path: Path, key: str | None = None) -> NDArray[np.float64]:
    """Read a field datum: a 1-D array from ``.npy`` or ``.npz`` (``key``, or the first).

    Singleton axes are squeezed, as PET does for its ``.npz`` data cells.

    Raises
    ------
    ConfigError
        If the file or key is missing or the array is not 1-D.
    """
    if not path.is_file():
        raise ConfigError(f"field observation file not found: {path}")
    if path.suffix == ".npy":
        array = np.load(path)
    else:
        with np.load(path) as data:
            name = key or data.files[0]
            if name not in data:
                # The count, not the names: a file's own keys are its contents.
                raise ConfigError(f"{path} has no array {name!r} (it has {len(data.files)})")
            array = data[name]
    field = np.squeeze(np.asarray(array, dtype=np.float64))
    if field.ndim != 1:
        raise ConfigError(f"{path}: a field must be a 1-D array, got shape {field.shape}")
    return field


def read_prior_ensemble(path: Path, layout: StateLayout) -> NDArray[np.float64]:
    """Read a prior ensemble ``.npz`` with one ``(size, n_ensemble)`` array per variable.

    Returns the stacked ``(n_state, n_ensemble)`` matrix.

    Raises
    ------
    ConfigError
        If the file is missing, a variable is absent or mis-shaped, or sizes disagree.
    """
    if not path.is_file():
        raise ConfigError(f"prior ensemble file not found: {path}")
    with np.load(path) as data:
        blocks = {
            name: np.asarray(data[name], dtype=np.float64) for name in layout.names if name in data
        }
    try:
        n_ensemble = {block.shape[1] for block in blocks.values() if block.ndim == 2}
        if any(block.ndim != 2 for block in blocks.values()) or len(n_ensemble) > 1:
            raise ValueError(
                "every variable must be a (size, n_ensemble) array of equal n_ensemble"
            )
        return layout.stack(blocks)
    except ValueError as error:
        raise ConfigError(f"{path}: {error}") from error


def read_truth(
    path: Path, layout: StateLayout, optional: frozenset[str] = frozenset()
) -> NDArray[np.float64]:
    """Read a synthetic case's true model: an ``.npz`` with one ``(size,)`` array per variable.

    Returns the stacked ``(n_state,)`` vector. Singleton axes are squeezed. Variables in
    ``optional`` (a model-error model's parameters, which have no true value) may
    be absent and are then zero; metrics leave them out.

    Raises
    ------
    ConfigError
        If the file is missing, a variable is absent or its size is wrong, or values are
        not finite.
    """
    if not path.is_file():
        raise ConfigError(f"true-model file not found: {path}")
    sizes = {v.name: v.size for v in layout.variables}
    with np.load(path) as data:
        missing = [name for name in layout.names if name not in data and name not in optional]
        if missing:
            raise ConfigError(
                f"{path}: no true values of {', '.join(missing)} (has {', '.join(data.files)})"
            )
        blocks = {
            name: np.squeeze(np.asarray(data[name], dtype=np.float64)).reshape(-1)
            if name in data
            else np.zeros(sizes[name])
            for name in layout.names
        }
    for variable in layout.variables:
        block = blocks[variable.name]
        if block.shape != (variable.size,):
            raise ConfigError(
                f"{path}: {variable.name!r} has {block.size} true values, the state {variable.size}"
            )
        if not np.all(np.isfinite(block)):
            raise ConfigError(f"{path}: true values of {variable.name!r} must be finite")
    return np.concatenate([blocks[name] for name in layout.names])
