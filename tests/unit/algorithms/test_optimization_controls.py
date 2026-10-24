"""Control parameterizations (2026-10-04): schedules and scaling."""

import numpy as np
import pytest

from apex.algorithms.optimization import ControlScaling, ControlSchedule
from apex.core.plugins import parameterizations


def test_a_held_schedule_keeps_each_value_until_the_next_knot() -> None:
    schedule = ControlSchedule("rate", knots=[0, 2, 5], n_steps=7, output="rates")
    out = schedule.properties({"rate": np.array([10.0, 20.0, 30.0])})
    np.testing.assert_array_equal(out["rates"], [10, 10, 20, 20, 20, 30, 30])
    assert schedule.variables == ("rate",)
    assert schedule.outputs == ("rates",)


def test_a_linear_schedule_interpolates_and_holds_the_last_value() -> None:
    schedule = ControlSchedule("rate", knots=[0, 2, 4], n_steps=6, interpolation="linear")
    out = schedule.properties({"rate": np.array([0.0, 10.0, 30.0])})
    np.testing.assert_allclose(out["rate"], [0, 5, 10, 20, 30, 30])


def test_scaling_maps_the_unit_interval_onto_the_bounds() -> None:
    scaling = ControlScaling("u", lower=[10.0, 0.0], upper=[400.0, 1.0], output="inj")
    out = scaling.properties({"u": np.array([0.5, 0.25])})
    np.testing.assert_allclose(out["inj"], [205.0, 0.25])
    scalar = ControlScaling("u", lower=10.0, upper=20.0)
    np.testing.assert_allclose(scalar.properties({"u": np.array([0.0, 1.0])})["u"], [10.0, 20.0])
    assert isinstance(parameterizations.create("control_scaling", variable="u", lower=0, upper=1),
                      ControlScaling)  # fmt: skip


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"knots": [1, 2], "n_steps": 4}, "start at step 0"),
        ({"knots": [0, 2, 2], "n_steps": 4}, "increase"),
        ({"knots": [0, 4], "n_steps": 4}, "below n_steps"),
        ({"knots": [0], "n_steps": 4, "interpolation": "cubic"}, "hold"),
        ({"knots": [0, 2.5], "n_steps": 4}, "whole report steps"),
    ],
)
def test_bad_schedules_are_refused(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ControlSchedule("rate", **kwargs)  # type: ignore[arg-type]


def test_bad_scaling_and_wrong_blocks_are_refused() -> None:
    with pytest.raises(ValueError, match="exceed"):
        ControlScaling("u", lower=1.0, upper=1.0)
    with pytest.raises(ValueError, match="one length"):
        ControlScaling("u", lower=[0.0, 0.0], upper=[1.0])
    with pytest.raises(ValueError, match="has 3 values"):
        ControlScaling("u", lower=[0.0, 0.0], upper=[1.0, 1.0]).properties({"u": np.ones(3)})
    with pytest.raises(ValueError, match="shape"):
        ControlSchedule("rate", knots=[0, 1], n_steps=3).properties({"rate": np.ones(3)})
    with pytest.raises(ValueError, match="missing"):
        ControlSchedule("rate", knots=[0], n_steps=3).properties({})
