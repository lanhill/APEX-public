"""Comparison statistics of studies, against direct computations."""

import numpy as np
import pytest
from scipy.stats import wilcoxon

from apex.algorithms.metrics.comparison import paired, smallest_p_value, summarise


def test_summary_of_several_replicates_and_of_one() -> None:
    values = [1.0, 2.0, 4.0]
    summary = summarise(values)

    assert (summary.n, summary.minimum, summary.maximum) == (3, 1.0, 4.0)
    assert summary.mean == pytest.approx(7 / 3)
    assert summary.std == pytest.approx(np.std(values, ddof=1))
    assert summarise([2.5]).std is None  # one seed: no spread
    with pytest.raises(ValueError, match="finite"):
        summarise([1.0, np.nan])


def test_paired_differences_match_direct_computation() -> None:
    rng = np.random.default_rng(4)
    a = rng.standard_normal(8)
    b = a - 0.5 + 0.1 * rng.standard_normal(8)

    result = paired(a.tolist(), b.tolist())

    difference = a - b
    assert result.n == 8
    assert result.mean_difference == pytest.approx(difference.mean())
    assert result.standard_error == pytest.approx(difference.std(ddof=1) / np.sqrt(8))
    assert result.p_value == pytest.approx(wilcoxon(difference).pvalue)
    # All eight differences positive: the exact two-sided p-value is 2 / 2^8.
    assert result.p_value == pytest.approx(smallest_p_value(8))


def test_one_replicate_gives_the_difference_only_and_five_cannot_be_significant() -> None:
    single = paired([3.0], [1.0])
    assert (single.n, single.mean_difference, single.standard_error, single.p_value) == (
        1,
        2.0,
        None,
        None,
    )
    # Five pairs all favouring one variant: the best a two-sided test can say is 1/16.
    five = paired([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert five.p_value == pytest.approx(0.0625)
    assert smallest_p_value(5) > 0.05 > smallest_p_value(6)
    with pytest.raises(ValueError, match="one length"):
        paired([1.0, 2.0], [1.0])


def test_zero_differences_are_dropped_from_the_test() -> None:
    # Two identical pairs and five positive ones: the test uses the five (Wilcoxon's rule).
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    b = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    result = paired(a, b)

    assert (result.n, result.n_nonzero) == (7, 5)
    assert result.p_value == pytest.approx(wilcoxon(np.subtract(a, b), zero_method="wilcox").pvalue)
    assert result.p_value is not None
    assert result.p_value >= smallest_p_value(5)
