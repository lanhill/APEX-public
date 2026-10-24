"""Split observed data into history-matching, cross-validation and test roles.

Production data are named ``KEYWORD:WELL`` (Eclipse summary style, e.g. ``WOPR:PRO1``).
A datatype without ``:`` (e.g. ``FOPR``, or a field datatype such as ``SEIS``) is its own
well and keyword. Field data (keys with a component index, e.g. seismic vintages) are
always history-matching data: their vintages are too infrequent in time for correlation
to guide a split, so they cannot be selected for cross validation or testing.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex.domain.config import DataSelection
from apex.domain.observations import DataKey, ObservationSet


def well_of(datatype: str) -> str:
    """Return the well of ``KEYWORD:WELL`` (the datatype itself when it has no well)."""
    return datatype.split(":", 1)[1] if ":" in datatype else datatype


def keyword_of(datatype: str) -> str:
    """Return the keyword of ``KEYWORD:WELL`` (the datatype itself when it has no well)."""
    return datatype.split(":", 1)[0]


@dataclass(frozen=True)
class DataSplit:
    """The keys of each role, in the observations' order.

    Attributes
    ----------
    history_matching
        Assimilated data (always including every field datum).
    cv
        Cross-validation data: forecast, scored, never assimilated.
    test
        Test data: forecast and scored only.
    unused
        Scalar data in no role (only when ``history_matching`` is given explicitly).
    """

    history_matching: tuple[DataKey, ...]
    cv: tuple[DataKey, ...] = ()
    test: tuple[DataKey, ...] = ()
    unused: tuple[DataKey, ...] = ()

    @property
    def has_cv(self) -> bool:
        """Whether cross validation is configured."""
        return bool(self.cv)

    def describe(self) -> dict[str, object]:
        """Return counts, keys and the history-matching to CV ratio (``data_split.json``)."""
        roles = {
            "history_matching": self.history_matching,
            "cv": self.cv,
            "test": self.test,
            "unused": self.unused,
        }
        return {
            "counts": {role: len(keys) for role, keys in roles.items()},
            "ratio_history_matching_to_cv": (
                len(self.history_matching) / len(self.cv) if self.cv else None
            ),
            "datatypes": {role: sorted({k.datatype for k in keys}) for role, keys in roles.items()},
            # Field data are summarised by datatype and time, not listed per component.
            "keys": {
                role: sorted({f"{k.datatype}@{k.time}" for k in keys})
                for role, keys in roles.items()
            },
        }


def resolve(
    observations: ObservationSet,
    *,
    cv: DataSelection | None = None,
    test: DataSelection | None = None,
    history_matching: DataSelection | None = None,
) -> DataSplit:
    """Assign every datum of ``observations`` to a role.

    Parameters
    ----------
    observations
        The raw observations (before data transforms).
    cv, test
        Scalar data held out of the assimilation.
    history_matching
        The assimilated scalar data; default every scalar datum not in ``cv`` or ``test``.
        Field data are always included.

    Raises
    ------
    ValueError
        If a selector matches nothing or field data, or the roles overlap.
    """
    selected: dict[str, set[DataKey]] = {}
    for role, selection in (("cv", cv), ("test", test), ("history_matching", history_matching)):
        if selection is not None:
            selected[role] = _select(observations.keys, selection, role)
    for first, second in (("cv", "test"), ("cv", "history_matching"), ("test", "history_matching")):
        both = selected.get(first, set()) & selected.get(second, set())
        if both:
            raise ValueError(
                f"[validation] {first} and {second} overlap: {_listing(both, observations)}"
            )
    held_out = selected.get("cv", set()) | selected.get("test", set())
    fields = {k for k in observations.keys if k.index is not None}
    if history_matching is None:
        assimilated = set(observations.keys) - held_out
    else:
        assimilated = selected["history_matching"] | fields
    unused = set(observations.keys) - held_out - assimilated
    return DataSplit(
        history_matching=_ordered(assimilated, observations),
        cv=_ordered(selected.get("cv", set()), observations),
        test=_ordered(selected.get("test", set()), observations),
        unused=_ordered(unused, observations),
    )


def _select(keys: Sequence[DataKey], selection: DataSelection, role: str) -> set[DataKey]:
    times = None if selection.times is None else set(selection.times)
    in_time = [k for k in keys if times is None or k.time in times]
    if times is not None:
        missing = sorted(times - {k.time for k in keys})
        if missing:
            raise ValueError(f"[validation] {role}: no data at times {', '.join(missing)}")
    chosen: set[DataKey] = set()
    criteria: tuple[tuple[str, list[str], Callable[[str], str]], ...] = (
        ("wells", selection.wells, well_of),
        ("datatypes", selection.datatypes, str),
        ("keywords", selection.keywords, keyword_of),
    )
    for name, values, part in criteria:
        for value in values:
            found = {k for k in in_time if part(k.datatype) == value}
            if not found:
                raise ValueError(f"[validation] {role}: {name} entry {value!r} matches no data")
            fields = sorted({k.datatype for k in found if k.index is not None})
            if fields and role != "history_matching":
                raise ValueError(
                    f"[validation] {role}: {', '.join(fields)} is field data; field data are "
                    "used for history matching only"
                )
            chosen |= found
    return chosen


def _ordered(keys: Iterable[DataKey], observations: ObservationSet) -> tuple[DataKey, ...]:
    return tuple(sorted(keys, key=observations.index_of))


def _listing(keys: set[DataKey], observations: ObservationSet, limit: int = 5) -> str:
    names = [str(k) for k in _ordered(keys, observations)]
    more = f" (and {len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


@dataclass(frozen=True)
class DataGroup:
    """One production-data group of the split explorer: a well (unit) and a keyword.

    ``unit`` is the well of ``KEYWORD:WELL``, or the datatype itself when it has none.
    """

    unit: str
    keyword: str
    datatype: str
    n_data: int


@dataclass(frozen=True)
class SplitCandidate:
    """A candidate cross-validation set of the explorer: the ``cv_units`` lowest-scoring wells.

    ``ratio`` is history-matching data per CV datum, as in the paper's tables.
    """

    cv_units: tuple[str, ...]
    n_history_matching: int
    n_cv: int

    @property
    def ratio(self) -> float:
        """History-matching data per CV datum."""
        return self.n_history_matching / self.n_cv

    def snippet(self) -> str:
        """Return a ``[validation]`` section selecting this candidate, to paste in a config."""
        wells = ", ".join(f'"{unit}"' for unit in self.cv_units)
        return f"[validation]\ncv = {{ wells = [{wells}] }}\n"


@dataclass(frozen=True)
class SplitExploration:
    """What the data-split explorer found.

    Attributes
    ----------
    groups
        Production-data groups, in the observations' order.
    correlation
        Lag-0 Pearson correlation of the groups' observed time series, ``(n_g, n_g)``,
        over the report times each pair shares; ``0`` where undefined (see ``notes``).
    scores
        Per unit (well), the paper's Step-3 double-averaged absolute correlation with the
        other units' groups, lowest (best CV candidate) first.
    redundancy
        Per unit, the mean absolute correlation across prior-ensemble members between its
        predictions and the other units' data, field data included (``None`` without a
        prior forecast), in the order of ``scores``.
    candidates
        Candidate CV sets for ``1, 2, ...`` wells.
    field_datatypes
        Field data, which always stay in history matching.
    notes
        Undefined correlations counted as zero, and similar remarks.
    """

    groups: tuple[DataGroup, ...]
    correlation: NDArray[np.float64]
    scores: dict[str, float]
    redundancy: dict[str, float] | None
    candidates: tuple[SplitCandidate, ...]
    field_datatypes: tuple[str, ...]
    notes: tuple[str, ...]
