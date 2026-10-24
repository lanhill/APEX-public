"""APEX's iterates as ERT responses (write-back), and the storage they go to."""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from apex.domain.observations import DataKey
from apex.integrations.ert.writeback import member_responses


def test_each_member_gets_its_forecast_at_the_summary_keys() -> None:
    keys = [DataKey("2020-01-31", "WOPR:P1"), DataKey("2020-02-29", "FOPR"),
            DataKey("2020-01-31", "SEIS", 0)]  # fmt: skip
    predicted = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    rows, skipped = member_responses(keys, predicted)

    assert skipped == {"field data"}
    assert rows["summary"] == [
        {
            "response_key": ["WOPR:P1", "FOPR"],
            "time": [dt.datetime(2020, 1, 31), dt.datetime(2020, 2, 29)],
            "values": [1.0, 3.0],
        },
        {
            "response_key": ["WOPR:P1", "FOPR"],
            "time": [dt.datetime(2020, 1, 31), dt.datetime(2020, 2, 29)],
            "values": [2.0, 4.0],
        },
    ]  # fmt: skip  (field data are not summary responses)


def test_rows_must_match_the_keys() -> None:
    with pytest.raises(ValueError, match="2 forecast rows for 1 keys"):
        member_responses([DataKey("2020-01-31", "FOPR")], np.zeros((2, 3)))


def test_general_observations_are_written_as_gen_data_and_other_types_named() -> None:
    keys = [DataKey("0, 3", "RESP"), DataKey("2020-01-31", "FOPR"), DataKey("1, 2, 3", "RFT1")]
    types = {"RESP": "gen_data", "FOPR": "summary", "RFT1": "rft"}

    rows, skipped = member_responses(keys, np.array([[1.0], [2.0], [3.0]]), types)

    assert rows["gen_data"] == [
        {"response_key": ["RESP"], "report_step": [0], "index": [3], "values": [1.0]}
    ]
    assert rows["summary"][0]["values"] == [2.0]
    assert skipped == {"rft"}


def test_an_older_ert_storage_is_migrated_before_apex_writes_to_it(tmp_path: Path) -> None:
    """ERT would write the new format under the old index; APEX migrates first, as ERT does."""
    pytest.importorskip("ert")
    from ert.storage import open_storage
    from ert.storage.local_storage import _LOCAL_STORAGE_VERSION

    from apex.integrations.ert.case import open_for_writing

    with open_storage(tmp_path, mode="w") as storage:
        storage.create_ensemble(storage.create_experiment(name="old"), ensemble_size=2)
    index = tmp_path / "index.json"
    older = {**json.loads(index.read_text()), "version": _LOCAL_STORAGE_VERSION - 1}
    index.write_text(json.dumps(older))

    with open_for_writing(tmp_path) as storage:
        assert [e.name for e in storage.experiments] == ["old"]

    assert json.loads(index.read_text())["version"] == _LOCAL_STORAGE_VERSION


def test_another_ert_major_version_is_refused_with_the_supported_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import metadata

    from apex.core.errors import ConfigError
    from apex.integrations.ert.config import require_supported_ert

    monkeypatch.setattr(metadata, "version", lambda name: "25.0.4")
    with pytest.raises(ConfigError, match=r"needs ERT 26\.x, but ERT 25\.0\.4 is installed"):
        require_supported_ert()
    for supported in ("26.0.1", "26.3.0.dev2+g1a2b3c"):
        monkeypatch.setattr(metadata, "version", lambda name, v=supported: v)
        require_supported_ert()
