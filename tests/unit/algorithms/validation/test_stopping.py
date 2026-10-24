"""CV stopping rules: the minimum, ties, patience and checkpoints."""

import math

import pytest

from apex.algorithms.validation import PatienceStopping, SelectStopping
from apex.core.plugins import stopping_rules

#: A V-shaped CV curve with a tie at its minimum (iterations 2 and 4).
CURVE = [9.0, 5.0, 3.0, 4.0, 3.0, 6.0, 7.0, 8.0]


def _feed(rule: PatienceStopping | SelectStopping, curve: list[float]) -> int | None:
    for iteration, value in enumerate(curve):
        if rule.observe(iteration, value):
            return iteration
    return None


def test_patience_stops_after_that_many_iterates_without_a_new_minimum() -> None:
    rule = PatienceStopping(patience=2)
    # The tie at 4 is not a new minimum: 3 and 4 are the two iterates past iteration 2.
    assert _feed(rule, CURVE) == 4
    assert rule.selected == 2


def test_patience_three_waits_one_more() -> None:
    rule = PatienceStopping(patience=3)
    assert _feed(rule, CURVE) == 5
    assert rule.selected == 2


def test_select_never_stops_and_keeps_the_earliest_minimum() -> None:
    rule = SelectStopping()
    assert _feed(rule, CURVE) is None
    assert rule.selected == 2


def test_a_monotone_curve_selects_its_last_iterate() -> None:
    rule = PatienceStopping(patience=2)
    assert _feed(rule, [5.0, 4.0, 3.0, 2.0]) is None
    assert rule.selected == 3


def test_state_round_trip_continues_identically() -> None:
    whole = PatienceStopping(patience=3)
    first = PatienceStopping(patience=3)
    for iteration, value in enumerate(CURVE[:4]):
        whole.observe(iteration, value)
        first.observe(iteration, value)
    resumed = PatienceStopping(patience=3)
    resumed.restore(first.state())
    for iteration, value in enumerate(CURVE[4:], start=4):
        assert resumed.observe(iteration, value) == whole.observe(iteration, value)
    assert resumed.selected == whole.selected
    assert resumed.state() == whole.state()


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_values_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="CV data mismatch of iterate 1"):
        PatienceStopping().observe(1, value)


@pytest.mark.parametrize("patience", [0, -1, 1.5, True])
def test_invalid_patience_is_refused(patience: object) -> None:
    with pytest.raises(ValueError, match="patience"):
        PatienceStopping(patience=patience)  # type: ignore[arg-type]


def test_rules_are_registered() -> None:
    assert {"patience", "select"} <= set(stopping_rules.names())
