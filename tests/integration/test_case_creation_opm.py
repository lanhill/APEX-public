"""A created synthetic case runs on OPM Flow (validation 2).

The rules' five-spot: its deck written, its truth drawn and run once, its noisy
observations made, the case validated and one GIES iteration run with no failed member.
"""

import shutil
from pathlib import Path

import pytest

from apex.agents.converter import CaseConverter
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import CaseConverterSection

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]


def test_a_created_five_spot_runs_on_opm_flow(tmp_path: Path) -> None:
    service = ApexService(tmp_path / "service")
    settings = CaseConverterSection(ensemble_size=10, iterations=1)

    created = CaseConverter(service, None, settings=settings).create(
        tmp_path / "case", name="box", simulate=True
    )

    assert created.valid, created.validation.errors
    observations = (tmp_path / "case" / "observations.csv").read_text().splitlines()
    assert any(row.split(",")[1] == "WOPR:PROD1" for row in observations[1:])
    result = composition.start(tmp_path / "case" / "config.toml", max_steps=1)
    assert result is not None
    run = tmp_path / "case" / "runs" / "box"
    assert (run / "checkpoints" / "iter_0.npz").is_file()
    assert not list((run / "simulations").rglob("APEX_FAILED"))
