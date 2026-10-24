"""Observed data and its error model."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True, order=True)
class DataKey:
    """Identifies one datum: a report time and a data type, e.g. ``("2024-03-11", "WOPR:PRO1")``.

    ``time`` is a label (an ISO date, or a step index as text); simulators map it to their
    own report points. ``index`` is ``None`` for a scalar datum and the component number
    for one element of a field datum (e.g. a seismic vintage over grid cells), so a field
    observed at one time is the run of keys ``(time, datatype, 0..n-1)``.
    """

    time: str
    datatype: str
    index: int | None = None

    def __str__(self) -> str:
        """Return ``datatype@time``, with ``[index]`` for a field component."""
        suffix = "" if self.index is None else f"[{self.index}]"
        return f"{self.datatype}@{self.time}{suffix}"


class ObservationSet:
    """Observed values with uncorrelated error variances, in a fixed key order.

    Field data are stored flat, one key per component (see :class:`DataKey`).

    Parameters
    ----------
    keys
        One key per datum; must be unique.
    values
        Observed values, shape ``(n_data,)``.
    variances
        Error variances, shape ``(n_data,)``, finite and positive.

    Raises
    ------
    ValueError
        If the lengths differ, keys repeat, a time and type is both scalar and field, or
        values/variances are invalid.
    """

    __slots__ = ("_index", "keys", "values", "variances")

    def __init__(self, keys: Sequence[DataKey], values: ArrayLike, variances: ArrayLike) -> None:
        self.keys: tuple[DataKey, ...] = tuple(keys)
        self.values: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
        self.variances: NDArray[np.float64] = np.asarray(variances, dtype=np.float64)
        n = len(self.keys)
        if n == 0:
            raise ValueError("an observation set needs at least one datum")
        if self.values.shape != (n,) or self.variances.shape != (n,):
            raise ValueError(
                f"expected {n} values and variances, got shapes "
                f"{self.values.shape} and {self.variances.shape}"
            )
        if len(set(self.keys)) != n:
            raise ValueError("observation keys must be unique")
        scalar = {(k.time, k.datatype) for k in self.keys if k.index is None}
        mixed = sorted(
            {
                f"{k.datatype}@{k.time}"
                for k in self.keys
                if k.index is not None and (k.time, k.datatype) in scalar
            }
        )
        # The names are a field entry's datatype and time, from the configuration: a key in
        # both sets, never one known from a file alone.
        if mixed:
            raise ValueError(f"data observed both as a scalar and as a field: {', '.join(mixed)}")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("observed values must be finite")
        if not np.all(np.isfinite(self.variances)) or np.any(self.variances <= 0):
            raise ValueError("observation variances must be finite and positive")
        self._index = {key: i for i, key in enumerate(self.keys)}

    @property
    def n_data(self) -> int:
        """Number of observed data."""
        return len(self.keys)

    def subset(self, rows: Sequence[int] | NDArray[np.intp]) -> "ObservationSet":
        """Return the data at ``rows`` (positions in :attr:`keys`), in that order."""
        rows = np.asarray(rows, dtype=np.intp)
        return ObservationSet([self.keys[i] for i in rows], self.values[rows], self.variances[rows])

    def index_of(self, key: DataKey) -> int:
        """Position of ``key`` in :attr:`keys`.

        Raises
        ------
        KeyError
            If ``key`` is not observed.
        """
        return self._index[key]
