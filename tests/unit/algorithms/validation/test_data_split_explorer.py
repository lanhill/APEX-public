"""The data-split explorer: Luo & Ranazzi's Steps 1-4, checked by hand."""

import numpy as np
import pytest

from apex.algorithms.validation.data_split import (
    ensemble_redundancy,
    explore,
    observed_correlation,
    production_groups,
    unit_scores,
)
from apex.domain.data_split import DataGroup
from apex.domain.observations import DataKey, ObservationSet

TIMES = ["t0", "t1", "t2", "t3"]


def _observations(series: dict[str, list[float]], field: int = 0) -> ObservationSet:
    keys, values = [], []
    for datatype, observed in series.items():
        for time, value in zip(TIMES, observed, strict=False):
            keys.append(DataKey(time, datatype))
            values.append(value)
    keys += [DataKey("t3", "SEIS", i) for i in range(field)]
    values += [0.0] * field
    return ObservationSet(keys, values, [1.0] * len(keys))


def test_groups_are_well_and_keyword_and_skip_field_data() -> None:
    observations = _observations({"WOPR:A": [1, 2, 3, 4], "FOPR": [1, 1, 2, 2]}, field=3)
    groups = production_groups(observations)
    assert groups == [DataGroup("A", "WOPR", "WOPR:A", 4), DataGroup("FOPR", "FOPR", "FOPR", 4)]


def test_step_3_scores_match_the_formula_on_a_given_matrix() -> None:
    # Wells A (two groups) and B, C (one each); r chosen by hand.
    groups = [
        DataGroup("A", "WOPR", "WOPR:A", 4),
        DataGroup("A", "WWPR", "WWPR:A", 4),
        DataGroup("B", "WOPR", "WOPR:B", 4),
        DataGroup("C", "WOPR", "WOPR:C", 4),
    ]
    r = np.array(
        [
            [1.0, 0.9, -0.4, 0.2],
            [0.9, 1.0, 0.6, -0.8],
            [-0.4, 0.6, 1.0, 0.1],
            [0.2, -0.8, 0.1, 1.0],
        ]
    )
    scores = unit_scores(r, groups)
    # s_A = mean(mean(|-0.4|, |0.2|), mean(|0.6|, |-0.8|)) = mean(0.3, 0.7) = 0.5
    # s_B = mean(|-0.4|, |0.6|, |0.1|) = 1.1 / 3; s_C = mean(0.2, 0.8, 0.1) = 1.1 / 3
    assert scores == pytest.approx({"B": 1.1 / 3, "C": 1.1 / 3, "A": 0.5})
    assert list(scores) == ["B", "C", "A"]  # lowest first, ties by name


def test_pearson_at_lag_0_with_undefined_correlations_counted_as_zero() -> None:
    observations = _observations(
        {
            "WOPR:A": [1.0, 2.0, 3.0, 4.0],
            "WOPR:B": [2.0, 4.0, 6.0, 9.0],
            "WWPR:B": [5.0, 5.0, 5.0, 5.0],  # constant
            "WOPR:C": [7.0],  # one report time only
        }
    )
    groups = production_groups(observations)
    r, notes = observed_correlation(observations, groups)
    expected = np.corrcoef([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 9.0])[0, 1]
    assert r[0, 1] == pytest.approx(expected)
    assert r[0, 2] == r[1, 2] == 0.0
    assert r[0, 3] == r[3, 1] == 0.0
    np.testing.assert_array_equal(np.diag(r), 1.0)
    np.testing.assert_array_equal(r, r.T)
    assert any("WWPR:B: constant" in note for note in notes)
    assert any("WOPR:A and WOPR:C: 1 shared" in note for note in notes)


def test_candidates_take_the_lowest_scoring_wells() -> None:
    observations = _observations(
        {
            "WOPR:A": [1.0, 2.0, 3.0, 4.0],
            "WOPR:B": [1.1, 2.1, 2.9, 4.2],
            "WOPR:C": [4.0, 1.0, 3.0, 2.0],
        },
        field=5,
    )
    report = explore(observations)
    ranked = list(report.scores)
    assert ranked[0] == "C"
    assert [c.cv_units for c in report.candidates] == [("C",), ("C", ranked[1])]
    first = report.candidates[0]
    assert (first.n_cv, first.n_history_matching) == (4, 8 + 5)
    assert first.snippet() == '[validation]\ncv = { wells = ["C"] }\n'
    assert report.field_datatypes == ("SEIS",)
    with pytest.raises(ValueError, match="max_cv"):
        explore(observations, max_cv=3)


def test_one_well_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="at least two wells"):
        explore(_observations({"WOPR:A": [1, 2, 3, 4], "WWPR:A": [0, 1, 0, 1]}))


def test_ensemble_redundancy_follows_the_prior_members() -> None:
    rng = np.random.default_rng(3)
    driver = rng.standard_normal(50)
    keys = [DataKey("t0", "WOPR:A"), DataKey("t0", "WOPR:B"), DataKey("t0", "WOPR:C")]
    keys += [DataKey("t1", "SEIS", i) for i in range(2)]
    predicted = np.vstack(
        [
            driver,  # A: the same parameters as the seismic
            rng.standard_normal(50),  # B: independent of everything
            np.full(50, 3.0),  # C: no spread across members
            driver + 0.01 * rng.standard_normal(50),
            -driver,
        ]
    )
    redundancy = ensemble_redundancy(keys, predicted, ["A", "B", "C"])
    assert redundancy["C"] == 0.0
    assert redundancy["A"] > 2 * redundancy["B"]
    # A: mean over the other groups (B, C, SEIS@t1) of the mean |corr| of their rows.
    corr = np.corrcoef(predicted[[0, 1, 3, 4]])
    expected = np.mean([abs(corr[0, 1]), 0.0, np.mean(np.abs(corr[0, 2:]))])
    assert redundancy["A"] == pytest.approx(expected)
