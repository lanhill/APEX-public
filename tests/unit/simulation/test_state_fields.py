"""State fields as field keys: the restart reader's placement and refusals.

The run's files are stood in for by a fake ``CaseState`` (a grid's ACTNUM and restart arrays
over its active cells, in the simulator's active order); the real reading is ``CaseState``'s,
tested with the geophysics (``test_geophysics*.py``).
"""

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.domain.observations import DataKey
from apex.simulation import state_fields
from apex.simulation.state_fields import StateFieldModel, field_keys, split_fields

#: A 3 x 2 x 1 grid whose cell (i=1, j=0) is inactive: global F order 0..5, active 0,2,3,4,5.
ACTNUM = np.ones((3, 2, 1), dtype=bool)
ACTNUM[1, 0, 0] = False


class FakeCase:
    def __init__(self, case: Path) -> None:
        self.grid = SimpleNamespace(actnum=ACTNUM)

    def restart(self, date: dt.date, names: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        base = 100.0 * date.month
        return {"PRESSURE": base + np.arange(5.0)} if "PRESSURE" in names else {}


@pytest.fixture(autouse=True)
def _fake_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_fields, "CaseState", FakeCase)


def test_fields_are_placed_on_the_whole_grid_in_eclipse_order() -> None:
    model = StateFieldModel(["pressure"], ["2025-01-01", "2025-02-01"], cells=6)
    keys = field_keys(model.names, model.times, 6)
    assert all(model.owns(k) for k in keys)
    assert not model.owns(DataKey("2025-01-01", "WOPR:P1"))
    values = model.forecast(Path("CASE"), keys)
    january = values[:6]
    # Active cells in F order take the restart's values in order; the inactive one is 0.
    np.testing.assert_array_equal(january, [100.0, 0.0, 101.0, 102.0, 103.0, 104.0])
    np.testing.assert_array_equal(values[6:], [200.0, 0.0, 201.0, 202.0, 203.0, 204.0])
    split = split_fields(values[:, None], model.names, model.times, 6)
    assert split["PRESSURE"].shape == (6, 2, 1)
    np.testing.assert_array_equal(split["PRESSURE"][:, 1, 0], values[6:])


def test_a_missing_array_and_another_grid_are_the_decks_errors() -> None:
    swat = StateFieldModel(["SWAT"], ["2025-01-01"], cells=6)
    with pytest.raises(ConfigError, match="no SWAT"):
        swat.forecast(Path("CASE"), field_keys(swat.names, swat.times, 6))
    wrong = StateFieldModel(["PRESSURE"], ["2025-01-01"], cells=8)
    with pytest.raises(ConfigError, match="6 cells, the build 8"):
        wrong.forecast(Path("CASE"), field_keys(wrong.names, wrong.times, 8))
    with pytest.raises(ConfigError, match="ISO dates"):
        StateFieldModel(["PRESSURE"], ["step 3"], cells=6)
