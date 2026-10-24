"""Errors shared across layers (``apex.core.errors``)."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from apex.core.errors import validation_problems


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int


def test_validation_problems_name_the_keys_but_never_the_values() -> None:
    """Release tester N3 (2026-10-20): ``str(ValidationError)`` repeats the values given."""
    with pytest.raises(ValidationError) as caught:
        _Strict.model_validate({"count": "sk-FAKE-1", "API_KEY": "sk-FAKE-2"})
    assert "sk-FAKE" in str(caught.value)  # what pydantic itself would show
    text = validation_problems(caught.value)
    assert "sk-FAKE" not in text
    assert "input_value" not in text
    assert "count: Input should be a valid integer" in text
    assert "API_KEY: Extra inputs are not permitted" in text


def test_validation_problems_are_bounded() -> None:
    class Many(BaseModel):
        model_config = ConfigDict(extra="forbid")

    with pytest.raises(ValidationError) as caught:
        Many.model_validate({f"k{i}": i for i in range(30)})
    text = validation_problems(caught.value, limit=5)
    assert text.count("Extra inputs") == 5
    assert text.endswith("... and 25 more")
