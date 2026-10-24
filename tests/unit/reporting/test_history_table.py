"""The history table names each method's per-trial parameter (the release tester, 2026-10-19).

GN-EnRML's trials record a step length gamma, not a damping; ``apex run``/``resume``/``report``
headed its column "damping".
"""

from apex.domain.results import IterationRecord
from apex.reporting.summary import history_table
from apex.services.plugins import control_name


def test_gn_enrml_column_is_its_step_length() -> None:
    assert control_name("gn_enrml", {}) == "step length"
    assert control_name("gies", {}) == "damping"
    assert control_name("lm_enrml", {}) == "damping"
    assert control_name("es", {}) == "damping"  # no trials: the column stays empty
    assert control_name("chop", {"inner": "gies"}) == "damping"  # the wrapped method's


def test_history_table_heads_the_column_with_the_control() -> None:
    history = [
        IterationRecord(0, 10.0, 1.0, 5),
        IterationRecord(1, 8.0, 1.0, 5, accepted=True, damping=0.5),
    ]
    table = history_table(history, control_name("gn_enrml", {}))
    header = table.splitlines()[0]
    assert header.rstrip().endswith("step length")
    assert "damping" not in header
    assert "0.5" in table.splitlines()[3]
    assert history_table(history).splitlines()[0].rstrip().endswith("damping")
