"""A tiny exportable case (OPM Flow deck template, grid fields) for the ERT tests."""

from collections.abc import Callable
from pathlib import Path

import pytest

DECK = """-- a tiny deck
PORO
% for v in poro:
${"%.6e" % v}
% endfor
/
PERMX
% for v in permx:
${"%.6e" % math.exp(v)}
% endfor
/
-- member ${member_id}, level ${level}, ${label}
<%! import math %>
"""


def make_case(
    tmp_path: Path, validation: str = "", simulator: str = "opm_flow", method: str = ""
) -> Path:
    (tmp_path / "deck.mako").write_text(DECK)
    rows = ["time,datatype,value,variance"]
    for t, time in enumerate(["2020-01-31", "2020-02-29"]):
        for k, key in enumerate(["WOPR:P1", "WOPR:P2", "WBHP:I1", "FOPR"]):
            rows.append(f"{time},{key},{100.0 + 10 * k + t},{4.0 + k}")
    (tmp_path / "observations.csv").write_text("\n".join(rows) + "\n")
    method = method or 'name = "esmda"\noptions = { inflation = [2, 2] }'
    config = tmp_path / "case.toml"
    config.write_text(f"""
[run]
name = "tiny-case"
seed = 11

[[state.variables]]
name = "poro"
size = 12
grid = [4, 3, 1]
lower = 0.05
upper = 0.4

[[state.variables]]
name = "permx"
size = 12
grid = [4, 3, 1]

[prior]
ensemble_size = 5
generator = {{ name = "gaussian", options = {{ mean = 0.2, std = 0.05 }} }}

[prior.variables.permx]
name = "gaussian"
options = {{ mean = 5.0, std = 1.0 }}

[observations]
file = "observations.csv"

[case]
kind = "real"

[simulator]
name = "{simulator}"
options = {{ deck_template_file = "deck.mako", template_values = {{ label = "tiny" }} }}

[execution]
name = "process"
options = {{ workers = 3 }}

[method]
{method or 'name = "esmda"' + chr(10) + "options = {{ inflation = [2, 2] }}"}
strategy = {{ name = "approx" }}
{validation}
""")
    return config


CaseFactory = Callable[..., Path]


@pytest.fixture
def tiny_case(tmp_path: Path) -> CaseFactory:
    """Return a factory writing the tiny case in ``tmp_path`` (see :func:`make_case`)."""

    def make(**options: str) -> Path:
        return make_case(tmp_path, **options)

    return make
