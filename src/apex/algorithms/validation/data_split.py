"""Computations of the data-split explorer.

Reference: X. Luo and P. Ranazzi (2024), *Better stopping through cross validation in an
iterative ensemble smoother: A perspective from supervised machine learning*. Geoenergy
Science and Engineering, 243, 213298. doi:10.1016/j.geoen.2024.213298. Sections as in the
manuscript's revision R1: §2.3, Steps 1-4. No PET equivalent.

1. Group the production data by well and keyword (``WOPR:PRO1`` is well ``PRO1``,
   keyword ``WOPR``).
2. Correlate the groups' observed time series. Here this is the lag-0 Pearson
   coefficient ``r_ij`` over the report times the two groups share.
3. Score each well ``A`` by its double-averaged absolute correlation with the groups of
   the other wells, ``s_A = mean_{i in Idx(A)} [ mean_{j notin Idx(A)} |r_ij| ]``.
4. Candidate CV sets are the wells with the lowest scores.

A correlation that is undefined, either because a series is constant or because two
groups share fewer than two times, counts as ``|r| = 0`` and is reported in the notes.
Field data (e.g. seismic) never enter the split: they always stay in history matching.

The prior-ensemble redundancy is an addition to the paper. It correlates, across the
members of a prior forecast, each well's predictions with the other wells' data and with
the field data, averaged per group as in Step 3. Data that the remaining data already
constrain make weak CV data, even when their observed series are uncorrelated.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from apex.domain.data_split import (
    DataGroup,
    SplitCandidate,
    SplitExploration,
    keyword_of,
    well_of,
)
from apex.domain.observations import DataKey, ObservationSet


def production_groups(observations: ObservationSet) -> list[DataGroup]:
    """Step 1: one group per scalar datatype (well, keyword), in the observations' order."""
    counts: dict[str, int] = {}
    for key in observations.keys:
        if key.index is None:
            counts[key.datatype] = counts.get(key.datatype, 0) + 1
    return [DataGroup(well_of(d), keyword_of(d), d, n) for d, n in counts.items()]


def observed_correlation(
    observations: ObservationSet, groups: Sequence[DataGroup]
) -> tuple[NDArray[np.float64], list[str]]:
    """Step 2: lag-0 Pearson correlation of the groups' observed series, ``(n_g, n_g)``.

    Returns the matrix (unit diagonal) and notes on the pairs counted as ``r = 0``.
    """
    series: list[dict[str, float]] = []
    for group in groups:
        series.append(
            {
                k.time: float(observations.values[i])
                for i, k in enumerate(observations.keys)
                if k.index is None and k.datatype == group.datatype
            }
        )
    n = len(groups)
    correlation = np.eye(n)
    notes: list[str] = []
    constant = {
        g.datatype for g, s in zip(groups, series, strict=True) if np.ptp(list(s.values())) == 0
    }
    for name in sorted(constant):
        notes.append(f"{name}: constant observed series; its correlations count as 0")
    for i in range(n):
        for j in range(i + 1, n):
            shared = sorted(series[i].keys() & series[j].keys())
            a = np.array([series[i][t] for t in shared])
            b = np.array([series[j][t] for t in shared])
            if len(shared) < 2:
                notes.append(
                    f"{groups[i].datatype} and {groups[j].datatype}: {len(shared)} shared "
                    "report time(s); correlation counts as 0"
                )
                r = 0.0
            elif np.ptp(a) == 0 or np.ptp(b) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(a, b)[0, 1])
            correlation[i, j] = correlation[j, i] = r
    return correlation, notes


def unit_scores(correlation: NDArray[np.float64], groups: Sequence[DataGroup]) -> dict[str, float]:
    """Step 3: each unit's double-averaged ``|r|`` with other units' groups, lowest first.

    Raises
    ------
    ValueError
        If there are fewer than two units (no split is possible).
    """
    units = list(dict.fromkeys(g.unit for g in groups))
    if len(units) < 2:
        raise ValueError(f"cross validation needs at least two wells, found {units}")
    magnitude = np.abs(correlation)
    scores = {}
    for unit in units:
        own = [i for i, g in enumerate(groups) if g.unit == unit]
        others = [j for j, g in enumerate(groups) if g.unit != unit]
        scores[unit] = float(np.mean([magnitude[i, others].mean() for i in own]))
    # Lowest first; ties by name, so the ranking is reproducible.
    return dict(sorted(scores.items(), key=lambda item: (item[1], item[0])))


def ensemble_redundancy(
    keys: Sequence[DataKey], predicted: NDArray[np.float64], units: Sequence[str]
) -> dict[str, float]:
    """Mean ``|corr|`` across members between each unit's data and all other data.

    Parameters
    ----------
    keys
        Keys of the rows of ``predicted`` (analysis data space; field data, possibly
        compressed, have a component index).
    predicted
        A prior forecast, ``(n_data, n_members)``.
    units
        Units to score, e.g. in the order of :func:`unit_scores`.

    Returns
    -------
    dict
        Per unit, the mean over its groups of the mean over every other group of the mean
        ``|corr|`` of their row pairs (Step 3's averaging, with field vintages as groups).
        Rows without spread across members count as ``0``.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape[0] != len(keys) or predicted.ndim != 2 or predicted.shape[1] < 2:
        raise ValueError(f"predicted must be ({len(keys)}, n >= 2), got {predicted.shape}")
    anomalies = predicted - predicted.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(anomalies, axis=-1)
    standardized = np.divide(
        anomalies, norms[:, None], out=np.zeros_like(anomalies), where=norms[:, None] > 0
    )
    # A group is a scalar datatype, or one vintage (datatype, time) of a field.
    labels = [k.datatype if k.index is None else f"{k.datatype}@{k.time}" for k in keys]
    rows: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        rows.setdefault(label, []).append(i)
    unit_of = {
        label: well_of(label) if keys[r[0]].index is None else None for label, r in rows.items()
    }
    result = {}
    for unit in units:
        own = [label for label, u in unit_of.items() if u == unit]
        if not own:
            raise ValueError(f"the prior forecast has no data of {unit!r}")
        others = [label for label, u in unit_of.items() if u != unit]
        per_group = []
        for label in own:
            block = np.abs(standardized[rows[label]] @ standardized.T)
            per_group.append(np.mean([block[:, rows[o]].mean() for o in others]))
        result[unit] = float(np.mean(per_group))
    return result


def explore(
    observations: ObservationSet,
    *,
    prior_keys: Sequence[DataKey] | None = None,
    prior_predicted: NDArray[np.float64] | None = None,
    max_cv: int | None = None,
) -> SplitExploration:
    """Run the explorer on raw observations, optionally with a prior forecast.

    Parameters
    ----------
    observations
        The raw observations of a case.
    prior_keys, prior_predicted
        A prior forecast in the analysis data space, for the redundancy score.
    max_cv
        Largest candidate CV set, in wells; default all wells but one.

    Raises
    ------
    ValueError
        With fewer than two wells, or an invalid ``max_cv`` or prior forecast.
    """
    groups = production_groups(observations)
    correlation, notes = observed_correlation(observations, groups)
    scores = unit_scores(correlation, groups)
    units = list(scores)
    limit = len(units) - 1 if max_cv is None else max_cv
    if not 1 <= limit <= len(units) - 1:
        raise ValueError(f"max_cv must be in [1, {len(units) - 1}], got {max_cv}")
    redundancy = None
    if prior_predicted is not None:
        if prior_keys is None:
            raise ValueError("a prior forecast needs its keys")
        redundancy = ensemble_redundancy(prior_keys, prior_predicted, units)
    per_unit = {u: sum(g.n_data for g in groups if g.unit == u) for u in units}
    candidates = []
    for n in range(1, limit + 1):
        chosen = tuple(units[:n])
        n_cv = sum(per_unit[u] for u in chosen)
        candidates.append(SplitCandidate(chosen, observations.n_data - n_cv, n_cv))
    fields = tuple(dict.fromkeys(k.datatype for k in observations.keys if k.index is not None))
    return SplitExploration(
        groups=tuple(groups),
        correlation=correlation,
        scores=scores,
        redundancy=redundancy,
        candidates=tuple(candidates),
        field_datatypes=fields,
        notes=tuple(notes),
    )
