"""The case-converter on a real deck (validation 1): 3dBox imported back, and run.

The example's template rendered with its truth is a plain OPM Flow deck, the kind a user
brings. Its history is the example's own observed rates (``true_rates.csv``). The converted
case must take the example's state (ln PERMX and PORO over the 60 x 60 x 5 grid) and its
observed vectors, validate, and run one GIES iteration on OPM Flow with no failed member.
"""

import csv
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from apex.agents.converter import CaseConverter
from apex.core.simulation import MemberInput
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import CaseConverterSection
from apex.simulation.deck import render_deck

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "3dbox_production"


def test_3dbox_imported_back_runs_on_opm_flow(tmp_path: Path) -> None:
    truth = np.load(EXAMPLE / "truth.npz")
    state = {"permx": truth["permx"], "poro": truth["poro"]}
    deck = tmp_path / "user" / "BOX.DATA"
    deck.parent.mkdir()
    deck.write_text(render_deck(EXAMPLE / "deck.mako", MemberInput(0, state),
                                {"permx_cap": 8.0}, case_dir=EXAMPLE))  # fmt: skip
    service = ApexService(tmp_path / "service")
    settings = CaseConverterSection(ensemble_size=10, iterations=1)

    outcome = CaseConverter(service, None, settings=settings).convert(
        deck, tmp_path / "case", history=EXAMPLE / "true_rates.csv", name="box"
    )

    assert outcome.valid, outcome.validation.errors
    assert [(v.property, v.transform) for v in outcome.choices.variables] == [
        ("PERMX", "log"),
        ("PORO", "none"),
    ]
    with (EXAMPLE / "observations.csv").open() as handle:
        expected = {row["datatype"] for row in csv.DictReader(handle)}
    observed = {o.vector for o in outcome.choices.observations}
    # The rules leave out vectors that are 0 throughout (no water before breakthrough).
    assert observed <= expected
    assert {v for v in expected if v.startswith("WOPR")} <= observed
    config = tomllib.loads((tmp_path / "case" / "config.toml").read_text())
    assert [v["grid"] for v in config["state"]["variables"]] == [[60, 60, 5]] * 2

    result = composition.start(tmp_path / "case" / "config.toml", max_steps=1)

    assert result is not None
    run = tmp_path / "case" / "runs" / "box"
    assert (run / "checkpoints" / "iter_0.npz").is_file()
    assert not list((run / "simulations").rglob("APEX_FAILED"))
