"""Statistics over model realizations as objectives.

A robust optimization evaluates every member on every realization; each objective's values
``(n_r,)`` become the one number a multi-objective optimizer ranks by a statistic:

- ``mean``: the mean (the default);
- ``std``: the sample standard deviation (``ddof = 1``; needs at least two realizations);
- ``worst``: the largest value (every objective is minimised: for -NPV, the realization with
  the lowest NPV);
- ``p<q>`` (``q`` from 1 to 99), e.g. ``p10``, ``p50``, ``p90``: the ``q``-th percentile of
  the values, NumPy's linear interpolation. Of the objective as minimised: for -NPV, ``p90``
  is the 10th percentile of NPV, a pessimistic value.

The kernel is pure: arrays in and out, realizations on axis 1 and members last.
"""

import re
from collections.abc import Sequence

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray

#: The statistics' names (``p<q>``: a percentile, ``q`` from 1 to 99).
STATISTIC_PATTERN = r"^(mean|std|worst|p([1-9][0-9]?))$"
#: The default: the mean over the realizations.
MEAN = "mean"


def check_statistic(name: str) -> str:
    """Return ``name`` if it is a statistic, else raise ``ValueError`` naming the choices."""
    if re.fullmatch(STATISTIC_PATTERN, name) is None:
        raise ValueError(
            f"unknown statistic {name!r}: mean, std, worst, or a percentile p1 ... p99 "
            "(e.g. p10, p50, p90)"
        )
    return name


def needs_realizations(statistics: Sequence[str]) -> int:
    """Return the fewest realizations the statistics need (``std``: 2; the others: 1)."""
    return 2 if "std" in statistics else 1


def reduce_realizations(values: FloatArray, statistics: Sequence[str]) -> FloatArray:
    """Return each objective's statistic over the realizations.

    Parameters
    ----------
    values
        Objective values ``(n_obj, n_r, n)``: objectives, realizations, members.
    statistics
        One statistic per objective (see the module docstring).

    Returns
    -------
    FloatArray
        ``(n_obj, n)``.

    Raises
    ------
    ValueError
        If the statistics do not match the objectives, one is unknown, or ``std`` has fewer
        than two realizations.
    """
    n_obj, n_r, n = values.shape
    if len(statistics) != n_obj:
        raise ValueError(f"{len(statistics)} statistics for {n_obj} objectives")
    out = np.empty((n_obj, n))
    for i, name in enumerate(statistics):
        check_statistic(name)
        row = values[i]  # (n_r, n)
        if name == MEAN:
            out[i] = np.mean(row, axis=0)
        elif name == "std":
            if n_r < 2:
                raise ValueError("the statistic std needs at least two realizations")
            out[i] = np.std(row, axis=0, ddof=1)
        elif name == "worst":
            out[i] = np.max(row, axis=0)
        else:
            out[i] = np.percentile(row, float(name[1:]), axis=0)
    return np.ascontiguousarray(out)
