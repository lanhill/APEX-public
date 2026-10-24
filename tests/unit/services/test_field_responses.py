"""Field data a simulator computes itself: checked before any run."""

from collections.abc import Mapping

import pytest

from apex.core.errors import ConfigError
from apex.core.simulation import FieldResponses
from apex.domain.observations import DataKey
from apex.services.composition import check_field_responses


class _Coupled:
    def field_datatypes(self) -> Mapping[str, int]:
        return {"vertical_displacement": 3}


def _field(datatype: str, time: str, size: int) -> list[DataKey]:
    return [DataKey(time, datatype, i) for i in range(size)]


def test_a_simulator_declares_its_field_data() -> None:
    assert isinstance(_Coupled(), FieldResponses)
    assert not isinstance(object(), FieldResponses)


def test_declared_fields_are_checked_and_the_others_returned() -> None:
    keys = [
        DataKey("2020-01-31", "WOPR:P"),
        *_field("vertical_displacement", "2020-01-31", 3),
        *_field("avo_10", "2020-01-31", 2),
    ]
    others = check_field_responses(keys, _Coupled().field_datatypes())

    assert others == _field("avo_10", "2020-01-31", 2)  # [geophysics]'s, as before


@pytest.mark.parametrize("indices", [[0, 1], [0, 1, 2, 3], [0, 1, 1]])
def test_a_declared_field_observed_otherwise_than_whole_is_refused(indices: list[int]) -> None:
    keys = [DataKey("2020-01-31", "vertical_displacement", i) for i in indices]
    with pytest.raises(
        ConfigError,
        match=r"vertical_displacement@2020-01-31 is observed at .* indices 0 to 2, each once",
    ):
        check_field_responses(keys, {"vertical_displacement": 3})


def test_without_declared_fields_every_field_key_is_left_as_before() -> None:
    keys = _field("vertical_displacement", "2020-01-31", 2)
    assert check_field_responses(keys, {}) == keys
