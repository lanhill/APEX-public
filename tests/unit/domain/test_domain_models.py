import math

import numpy as np
import pytest
from pydantic import ValidationError

from apex.domain import DataKey, HistoryMatchConfig, ObservationSet, StateLayout, StateVariable


def test_observation_set_indexes_keys() -> None:
    keys = [DataKey("t0", "WOPR:P1"), DataKey("t1", "WOPR:P1")]

    observations = ObservationSet(keys, [1.0, 2.0], [0.1, 0.2])

    assert observations.n_data == 2
    assert observations.index_of(DataKey("t1", "WOPR:P1")) == 1
    assert str(keys[0]) == "WOPR:P1@t0"


@pytest.mark.parametrize(
    ("keys", "values", "variances", "message"),
    [
        ([], [], [], "at least one"),
        ([DataKey("t", "a")], [1.0, 2.0], [1.0], "expected 1"),
        ([DataKey("t", "a"), DataKey("t", "a")], [1.0, 2.0], [1.0, 1.0], "unique"),
        ([DataKey("t", "a")], [np.nan], [1.0], "finite"),
        ([DataKey("t", "a")], [1.0], [0.0], "positive"),
    ],
)
def test_observation_set_rejects_invalid_input(
    keys: list[DataKey], values: list[float], variances: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ObservationSet(keys, values, variances)


def test_state_layout_splits_stacks_and_bounds() -> None:
    layout = StateLayout([StateVariable("a", 2, lower=0.0), StateVariable("b", 3, upper=5.0)])
    matrix = np.arange(15, dtype=np.float64).reshape(5, 3)

    blocks = layout.split(matrix)
    lower, upper = layout.bounds()

    assert layout.n_state == 5
    np.testing.assert_array_equal(blocks["b"], matrix[2:])
    np.testing.assert_array_equal(layout.stack(blocks), matrix)
    np.testing.assert_array_equal(lower, [0, 0, -math.inf, -math.inf, -math.inf])
    np.testing.assert_array_equal(upper, [math.inf, math.inf, 5, 5, 5])


def test_state_layout_rejects_duplicates_and_bad_blocks() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateLayout([StateVariable("a", 1), StateVariable("a", 2)])
    layout = StateLayout([StateVariable("a", 2)])
    with pytest.raises(ValueError, match="needs 2 rows"):
        layout.stack({"a": np.zeros((3, 4))})
    with pytest.raises(ValueError, match="lower bound above"):
        StateVariable("c", 1, lower=1.0, upper=0.0)


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "run": {"name": "r", "seed": 1},
        "state": {"variables": [{"name": "x", "size": 2}]},
        "prior": {"ensemble_size": 10},
        "observations": {"file": "obs.csv"},
        "simulator": {"name": "linear"},
        "method": {"name": "esmda"},
        "case": {"truth_file": "truth.npz"},
    }
    config.update(overrides)
    return config


def test_config_applies_defaults() -> None:
    config = HistoryMatchConfig.model_validate(_config())

    assert config.execution.name == "serial"
    assert config.failures.min_success_fraction == 0.9
    assert config.method.max_iterations == 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prior": {"ensemble_size": 1}}, "ensemble_size"),
        ({"prior": {"generator": {"options": {}}}}, "name"),
        ({"run": {"name": "r", "seed": 1, "typo": 2}}, "typo"),
        ({"failures": {"min_success_fraction": 0.0}}, "min_success_fraction"),
        ({"case": {}}, "synthetic case study needs"),  # synthetic is the default
        ({"case": {"kind": "real", "truth_file": "t.npz"}}, "real case study has no truth"),
        ({"case": {"kind": "other"}}, "kind"),
    ],
)
def test_config_rejects_invalid_sections(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        HistoryMatchConfig.model_validate(_config(**overrides))


def test_a_real_case_needs_no_truth() -> None:
    config = HistoryMatchConfig.model_validate(_config(case={"kind": "real"}))
    assert config.case.kind == "real"
    assert config.metrics is None  # defaults are chosen by the composition root


def test_a_state_grid_must_match_its_size() -> None:
    from apex.domain.state import StateVariable

    assert StateVariable("permx", 200, grid=(10, 10, 2)).grid == (10, 10, 2)
    with pytest.raises(ValueError, match="product is its size"):
        StateVariable("permx", 200, grid=(10, 10, 3))
