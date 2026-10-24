"""APEX's keys of ERT's observation rows (every observation type)."""

from apex.domain.observations import DataKey
from apex.integrations.ert.observations import apex_key


def test_summary_rows_keep_apexs_date_keys_and_others_erts_index() -> None:
    assert apex_key("summary", "WOPR:P1", "2020-01-31 00:00:00.000") == DataKey(
        "2020-01-31", "WOPR:P1"
    )
    # A time of day is kept: APEX's summary reading would refuse it as a report date.
    assert apex_key("summary", "FOPR", "2020-01-31 12:00:00.000").time == "2020-01-31 12:00:00.000"
    assert apex_key("gen_data", "RESP", "0, 42") == DataKey("0, 42", "RESP")
    assert apex_key("rft", "OP1:2020-01-31:PRESSURE", "10.0, 20.0, 1500.0").index is None
