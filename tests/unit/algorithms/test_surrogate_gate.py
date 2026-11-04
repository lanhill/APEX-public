"""The accuracy gate (validation 1) on constructed errors with known answers.

Every expected error is computed by hand from the definitions in the gate's docstring: the
RMS of (surrogate - simulator) / sigma per datatype, the largest relative error against the
reference's change (pressure) or magnitude (others), and Spearman's rho from its closed form
1 - 6 sum d^2 / (n (n^2 - 1)) for untied ranks.
"""

import numpy as np
import pytest

from apex.algorithms.surrogates.gate import gate, quantity_kind
from apex.domain.observations import DataKey
from apex.domain.surrogate import DatatypeGate, GateResult, GateTolerances


def _keys(datatype: str, n: int) -> list[DataKey]:
    return [DataKey(str(t), datatype) for t in range(n)]


def test_sigma_ratio_is_the_rms_of_the_normalized_errors_per_datatype() -> None:
    keys = _keys("WOPR:P1", 2) + _keys("WBHP:P1", 2)
    simulator = np.arange(12.0).reshape(4, 3)
    # sigma = 2 everywhere: WOPR's errors +-0.8 give 0.4 (passes 0.5), WBHP's +-1.2 give 0.6.
    error = np.array([[0.8, -0.8, 0.8], [-0.8, 0.8, -0.8], [1.2, -1.2, 1.2], [1.2, 1.2, -1.2]])

    result = gate(simulator + error, simulator, keys, [5, 6, 7], variances=np.full(4, 4.0))

    by = {d.datatype: d for d in result.datatypes}
    assert by["WOPR:P1"].error == pytest.approx(0.4)
    assert by["WOPR:P1"].passed
    assert by["WBHP:P1"].error == pytest.approx(0.6)
    assert not by["WBHP:P1"].passed
    assert by["WOPR:P1"].values == 6
    assert by["WOPR:P1"].measure == "sigma_ratio"
    assert not result.passed
    assert result.held_out == [5, 6, 7]
    assert result.failures() == ["WBHP:P1: sigma_ratio error 0.6 > 0.5"]
    # A tighter gate, as a user of the surrogate may ask, fails WOPR too.
    tight = GateTolerances().tightened(sigma_ratio=0.3)
    assert not gate(simulator + error, simulator, keys, [5, 6, 7], variances=np.full(4, 4.0),
                    tolerances=tight).datatypes[0].passed  # fmt: skip


def test_relative_errors_judge_pressure_by_its_change_and_rates_by_their_magnitude() -> None:
    # One run; keys out of time order on purpose (the gate orders each series by time).
    keys = [
        DataKey("2", "WBHP:P1"), DataKey("0", "WBHP:P1"), DataKey("1", "WBHP:P1"),
        DataKey("0", "WOPR:P1"), DataKey("1", "WOPR:P1"), DataKey("2", "WOPR:P1"),
    ]  # fmt: skip
    simulator = np.array([[130.0], [100.0], [110.0], [10.0], [20.0], [40.0]])
    # Pressure: change 30, largest error 1.2 -> 0.04 <= 0.05 (by magnitude it would be 0.009).
    # Rate: magnitude 40, largest error 2.4 -> 0.06 > 0.05.
    error = np.array([[1.2], [0.0], [-0.6], [0.0], [2.4], [-1.0]])

    result = gate(simulator + error, simulator, keys, [0])

    by = {d.datatype: d for d in result.datatypes}
    assert by["WBHP:P1"].kind == "pressure"
    assert by["WBHP:P1"].error == pytest.approx(0.04)
    assert by["WBHP:P1"].passed
    assert by["WOPR:P1"].kind == "rate"
    assert by["WOPR:P1"].error == pytest.approx(0.06)
    assert not by["WOPR:P1"].passed
    assert not result.passed
    # Pressure's error 1.8 is 0.06 of its change: it fails too.
    error[0, 0] = 1.8
    assert not gate(simulator + error, simulator, keys, [0]).datatypes[0].passed


def test_a_constant_reference_is_judged_by_its_magnitude_and_zero_by_exact_agreement() -> None:
    keys = _keys("FPR", 2) + _keys("FOPT", 2)
    simulator = np.array([[200.0], [200.0], [0.0], [0.0]])
    surrogate = np.array([[201.0], [199.0], [0.0], [0.0]])

    result = gate(surrogate, simulator, keys, [0])

    assert [d.error for d in result.datatypes] == [pytest.approx(0.005), 0.0]
    assert result.passed
    surrogate[3, 0] = 1e-9
    failed = gate(surrogate, simulator, keys, [0]).datatypes[1]
    assert failed.error is None
    assert not failed.passed


def test_a_shut_well_is_judged_against_its_neighbours_scale() -> None:
    """SCALE_FLOOR: a shut well's zero rates against the other wells' WOPR."""
    keys = _keys("WOPR:P1", 2) + _keys("WOPR:P2", 2)
    simulator = np.array([[100.0], [200.0], [0.0], [0.0]])
    surrogate = simulator + np.array([[0.0], [0.0], [1e-13], [-1e-13]])

    result = gate(surrogate, simulator, keys, [0])

    # 1e-13 / (1e-3 * 200): round-off, not an infinite error.
    assert result.passed
    assert result.datatypes[1].error == pytest.approx(5e-13)
    # A real error on the shut well is still seen: 0.5 / 0.2 = 2.5.
    surrogate[3, 0] = 0.5
    assert gate(surrogate, simulator, keys, [0]).datatypes[1].error == pytest.approx(2.5)


def test_a_pressure_is_judged_by_its_change_and_by_magnitude_only_if_none_changes() -> None:
    """Review case: WBHP held at 250 bar with a 10 bar error passed on its magnitude, and a
    1e-9 bar change made it fail; now the fallback is only for a vector that never changes,
    and it is reported."""
    keys = _keys("WBHP:P1", 3)
    held = np.full((3, 1), 250.0)

    flat = gate(held + 10.0, held, keys, [0]).datatypes[0]
    assert flat.passed
    assert flat.scale == "magnitude"
    assert flat.error == pytest.approx(0.04)
    assert any("judged on its magnitude" in n for n in flat.notes)

    moving = held.copy()
    moving[2, 0] += 1e-9
    changed = gate(moving + 10.0, moving, keys, [0]).datatypes[0]
    assert changed.scale == "change"
    assert not changed.passed
    assert changed.notes == []

    # A constant well among changing ones is judged against the vector's change: 0.01 bar
    # against the floor 1e-3 * 20.
    keys2 = _keys("WBHP:P1", 2) + _keys("WBHP:P2", 2)
    simulator = np.array([[200.0], [220.0], [250.0], [250.0]])
    surrogate = simulator + np.array([[0.0], [0.0], [0.01], [0.0]])
    by = {d.datatype: d for d in gate(surrogate, simulator, keys2, [0]).datatypes}
    assert by["WBHP:P2"].scale == "change"
    assert by["WBHP:P2"].error == pytest.approx(0.5)
    assert not by["WBHP:P2"].passed


def test_a_pressure_series_starting_after_the_first_time_gated_is_noted() -> None:
    keys = [DataKey("1", "WBHP:P1"), DataKey("2", "WBHP:P1"), *_keys("WOPR:P1", 3)]
    simulator = np.array([[210.0], [220.0], [1.0], [2.0], [3.0]])

    result = gate(simulator, simulator, keys, [0])

    by = {d.datatype: d for d in result.datatypes}
    assert result.passed
    assert by["WBHP:P1"].notes
    assert "measured from 1" in by["WBHP:P1"].notes[0]
    assert by["WOPR:P1"].notes == []


def test_the_largest_sigma_ratio_is_recorded_beside_the_rms() -> None:
    keys = _keys("WOPR:P1", 1000)
    simulator = np.zeros((1000, 1))
    surrogate = simulator.copy()
    surrogate[123, 0] = 10.0  # one key at 10 sigma

    [entry] = gate(surrogate, simulator, keys, [0], variances=np.ones(1000)).datatypes

    assert entry.error == pytest.approx(np.sqrt(100 / 1000))
    assert entry.passed
    assert entry.max_sigma_ratio == pytest.approx(10.0)


def test_a_prediction_that_is_not_finite_fails_its_datatype() -> None:
    keys = _keys("WOPR:P1", 2)
    simulator = np.ones((2, 2))
    surrogate = simulator.copy()
    surrogate[1, 0] = np.nan

    for variances in (np.ones(2), None):
        result = gate(surrogate, simulator, keys, [0, 1], variances=variances)
        assert result.datatypes[0].error is None
        assert not result.passed
        assert "not finite" in result.failures()[0]


def test_the_rank_correlation_of_objectives_is_spearmans_rho() -> None:
    simulator = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    empty = np.zeros((0, 5))

    # A monotone distortion keeps the ranks: rho = 1.
    same = gate(empty, empty, [], range(5), objectives=(simulator**3 - 7, simulator))
    assert same.rank is not None
    assert same.rank.correlation == pytest.approx(1.0)
    assert same.passed
    # One swapped pair: rho = 1 - 6 * 2 / (5 * 24) = 0.9 < 0.95.
    swapped = gate(empty, empty, [], range(5), objectives=(np.array([1, 2, 3, 5, 4.0]), simulator))
    assert swapped.rank is not None
    assert swapped.rank.correlation == pytest.approx(0.9)
    assert not swapped.passed
    # A constant surrogate has no correlation, and fails.
    flat = gate(empty, empty, [], range(5), objectives=(np.ones(5), simulator))
    assert flat.rank is not None
    assert flat.rank.correlation is None
    assert not flat.passed


def _field(fields: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[DatatypeGate]:
    return gate(np.zeros((0, 1)), np.zeros((0, 1)), [], [0], fields=fields).datatypes


def test_fields_are_gated_by_a_pooled_relative_rms() -> None:
    """RMS error over cells, times and runs, against a
    pressure's RMS change from the first time (by hand: sqrt(0.64 / 6) / sqrt(1000 / 6))."""
    # Two cells, three times, one run: pressure rising by 10 per step in each cell.
    reference = np.array([[[100.0], [110.0], [120.0]], [[200.0], [210.0], [220.0]]])
    surrogate = reference.copy()
    surrogate[1, 2, 0] += 0.8
    [entry] = _field({"PRESSURE": (surrogate, reference)})
    assert entry.measure == "field_rms"
    assert entry.kind == "pressure"
    assert entry.scale == "change"
    assert entry.error == pytest.approx(np.sqrt(0.64 / 1000.0))
    assert entry.passed


def test_a_cell_that_barely_changes_does_not_fail_a_field() -> None:
    """Review case: a cell changing by 1e-9 with a 1e-6 error failed the cell-wise gate."""
    reference = np.array([[[100.0], [100.0 + 1e-9]], [[200.0], [220.0]]])
    noisy = reference.copy()
    noisy[0, 1, 0] += 1e-6
    [entry] = _field({"PRESSURE": (noisy, reference)})
    # sqrt(1e-12 / 4) / sqrt((1e-18 + 400) / 4)
    assert entry.error == pytest.approx(1e-6 / 20.0, rel=1e-6)
    assert entry.passed


def test_saturations_by_their_range_and_inactive_cells_left_out() -> None:
    reference = np.array([[[0.2], [0.5]], [[0.2], [0.8]], [[0.0], [0.0]]])  # cell 2 inactive
    surrogate = reference.copy()
    surrogate[1, 1, 0] += 0.06
    [entry] = _field({"SWAT": (surrogate, reference)})
    assert entry.kind == "other"
    assert entry.scale == "range"
    assert entry.values == 4  # the inactive cell is left out
    assert entry.error == pytest.approx(np.sqrt(0.06**2 / 4) / 0.6)
    flat = np.full((2, 2, 1), 0.3)
    [constant] = _field({"SWAT": (flat + 0.003, flat)})
    assert constant.scale == "magnitude"
    assert constant.error == pytest.approx(0.01)


def test_nothing_gated_does_not_pass_and_failures_are_recorded() -> None:
    result = gate(np.zeros((0, 2)), np.zeros((0, 2)), [], [1, 2], failed=[3])
    assert not result.passed
    assert result.failed == [3]
    assert result.failures() == ["nothing was gated"]
    # It survives JSON (a manifest's gate).
    assert GateResult.model_validate_json(result.model_dump_json()) == result


def test_tolerances_may_be_tightened_but_never_loosened() -> None:
    defaults = GateTolerances()
    assert (defaults.sigma_ratio, defaults.min_rank_correlation) == (0.5, 0.95)
    assert defaults.relative["mass"] == 0.01

    tight = defaults.tightened(sigma_ratio=0.3, min_rank_correlation=0.99, relative={"rate": 0.02})
    assert (tight.sigma_ratio, tight.min_rank_correlation, tight.relative["rate"]) == (
        0.3, 0.99, 0.02,
    )  # fmt: skip
    assert tight.relative["pressure"] == 0.05
    for looser in ({"sigma_ratio": 0.6}, {"min_rank_correlation": 0.9},
                   {"relative": {"rate": 0.1}}):  # fmt: skip
        with pytest.raises(ValueError, match="only be tightened"):
            defaults.tightened(**looser)
    # Kinds not given keep their defaults.
    assert GateTolerances(relative={"rate": 0.2}).relative["mass"] == 0.01


@pytest.mark.parametrize(
    ("datatype", "kind"),
    [
        ("WBHP:INJ", "pressure"), ("FPR", "pressure"), ("BPR:1,1,1", "pressure"),
        ("WTHP:P1", "pressure"), ("WBP4:P1", "pressure"), ("WBP:P1", "pressure"),
        ("WGOR:P1", "other"), ("WWGR:P1", "other"), ("WOGR:P1", "other"), ("WGLR:P1", "other"),
        ("FGOR", "other"), ("WWCT:P1", "other"), ("WOIR:I1", "rate"), ("FLPR", "rate"),
        ("WOPR:P1", "rate"), ("WGIR:CO2INJ", "rate"), ("FOPT", "mass"), ("FGIP", "mass"),
        ("PLUME_AREA", "plume"), ("FWCT", "other"),
    ],
)  # fmt: skip
def test_the_kind_of_a_datatype_is_read_from_its_mnemonic(datatype: str, kind: str) -> None:
    assert quantity_kind(datatype) == kind


def test_shapes_and_references_are_checked() -> None:
    keys = _keys("WOPR:P1", 2)
    with pytest.raises(ValueError, match="must both be"):
        gate(np.ones((2, 3)), np.ones((2, 2)), keys, [0, 1])
    with pytest.raises(ValueError, match="finite"):
        gate(np.ones((2, 1)), np.full((2, 1), np.inf), keys, [0])
    with pytest.raises(ValueError, match="variances"):
        gate(np.ones((2, 1)), np.ones((2, 1)), keys, [0], variances=np.array([1.0, 0.0]))


def test_several_objectives_are_gated_on_their_lowest_rank_correlation() -> None:
    """Decision 2: each objective must keep its order; one anti-ranked or
    constant objective fails the gate, whatever the others."""
    simulator = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 3.0, 4.0, 1.0, 2.0]])
    empty = np.zeros((0, 5))
    both = gate(empty, empty, [], range(5), objectives=(2.0 * simulator + 1.0, simulator))
    assert both.rank is not None
    assert both.rank.correlation == pytest.approx(1.0)
    assert both.passed
    reversed_second = np.vstack([simulator[0], -simulator[1]])
    anti = gate(empty, empty, [], range(5), objectives=(reversed_second, simulator))
    assert anti.rank is not None
    assert anti.rank.correlation == pytest.approx(-1.0)
    assert not anti.passed
    flat_second = np.vstack([simulator[0], np.ones(5)])
    flat = gate(empty, empty, [], range(5), objectives=(flat_second, simulator))
    assert flat.rank is not None
    assert flat.rank.correlation is None
    assert not flat.passed
    with pytest.raises(ValueError, match="objectives must be"):
        gate(empty, empty, [], range(5), objectives=(simulator, simulator[:, :4]))
