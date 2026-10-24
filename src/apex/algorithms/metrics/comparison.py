"""Statistics that compare variants of a study over replicates.

Each variant's metric is summarised over its replicates; two variants are compared by
their **paired** differences, replicate by replicate (they share the replicate seeds):

- mean difference ``mean_r (a_r - b_r)`` and its standard error ``std_r(a_r - b_r) / sqrt(n)``
  (``ddof = 1``);
- the Wilcoxon signed-rank test of the differences (two-sided; ``scipy.stats.wilcoxon``,
  exact for small ``n``). With ``n`` pairs the smallest attainable two-sided p-value is
  ``2 / 2^n``, so fewer than 6 pairs can never reach p < 0.05 (``2 / 32 = 0.0625``).
  Zero differences are dropped (``zero_method="wilcox"``, Wilcoxon's own rule), so the
  test's effective ``n`` is the number of non-zero differences (``Paired.n_nonzero``), and
  the bound applies to it. Differences at rounding level are not zero and count.

With one replicate there is no spread and no test: the values and the difference only.

Reference: Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics
Bulletin, 1(6), 80-83.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Summary:
    """A variant's values over replicates: ``n``, mean, std (``None`` for n = 1), range."""

    n: int
    mean: float
    std: float | None
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class Paired:
    """Paired differences ``a - b``: count, mean, standard error, Wilcoxon p-value."""

    n: int
    mean_difference: float
    standard_error: float | None
    p_value: float | None
    n_nonzero: int = 0


def summarise(values: Sequence[float]) -> Summary:
    """Summarise one variant's values.

    Raises
    ------
    ValueError
        If there are no values or a value is not finite.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"need finite values, got {list(values)}")
    std = float(array.std(ddof=1)) if array.size > 1 else None
    return Summary(
        int(array.size), float(array.mean()), std, float(array.min()), float(array.max())
    )


def smallest_p_value(n: int) -> float:
    """Return the smallest two-sided Wilcoxon signed-rank p-value of ``n`` pairs: ``2 / 2^n``."""
    return min(1.0, 2.0 / 2.0**n)


def paired(a: Sequence[float], b: Sequence[float]) -> Paired:
    """Compare paired values ``a`` and ``b`` (the same replicates, in the same order).

    Raises
    ------
    ValueError
        If the lengths differ or are zero.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError(f"need paired values of one length, got {x.size} and {y.size}")
    difference = x - y
    n = int(difference.size)
    nonzero = int(np.count_nonzero(difference))
    if n == 1:
        return Paired(1, float(difference[0]), None, None, nonzero)
    standard_error = float(difference.std(ddof=1) / np.sqrt(n))
    if nonzero == 0:
        return Paired(n, 0.0, standard_error, 1.0, 0)
    from scipy.stats import wilcoxon

    test = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    return Paired(n, float(difference.mean()), standard_error, float(test.pvalue), nonzero)
