"""The prediction-with-controls example on OPM Flow (``examples/prediction_controls``).

Release tester H35 (b): no shipped example let a user forecast a deck under a control
scenario, so whether ``apex summary`` and ``provenance.json`` record the scenario could not
be checked as a user would. The example's history match runs, then its two scenarios: both
reproduce the run over the history (the prediction's history check), differ after it in
the direction the bottom-hole pressures say, and are recorded.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from apex.domain.prediction import PredictionResult
from apex.services import composition, prediction
from apex.services.summaries import PredictionSummarizer

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("flow") is None, reason="OPM Flow is not installed"),
]

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "prediction_controls"


def test_the_examples_scenarios_reach_the_forecast_and_are_recorded(tmp_path: Path) -> None:
    case = tmp_path / "prediction_controls"
    shutil.copytree(EXAMPLE, case)
    config = case / "config_esmda.toml"
    # Two steps, not four: the scenarios' effect does not need a good match.
    text = config.read_text(encoding="utf-8").replace("[4, 4, 4, 4]", "[2, 2]")
    config.write_text(text, encoding="utf-8")
    composition.start(config)

    hold = prediction.predict(case / "predict_hold.toml")
    bhp = prediction.predict(case / "predict_bhp.toml")
    folder = case / "runs" / "control_scenario_esmda" / "predictions"

    for result, name, values in ((hold, "hold", [90.0, 90.0]), (bhp, "bhp", [60.0, 120.0])):
        recorded = json.loads((folder / name / "provenance.json").read_text())
        assert recorded["controls"] == {"bhp": values}
        notes = PredictionSummarizer().summarize(folder / name).notes
        assert any("control scenario" in n and f"bhp = {values}" in n for n in notes), notes
        assert result.truth is not None

    first = hold.first_prediction_time
    assert first > 0
    forecast_hold, forecast_bhp = hold.forecasts["posterior"], bhp.forecasts["posterior"]
    assert forecast_hold.times == forecast_bhp.times
    assert forecast_hold.vectors == forecast_bhp.vectors
    for forecast in (forecast_hold, forecast_bhp):  # every member forecast, the same ones
        assert forecast.failed_member_ids.size == 0
    np.testing.assert_array_equal(forecast_hold.member_ids, forecast_bhp.member_ids)
    # The history is the run's under either scenario; the forecasts differ only after it.
    np.testing.assert_allclose(forecast_hold.values[:, :first], forecast_bhp.values[:, :first])
    vectors = list(forecast_hold.vectors)

    def produced(result: PredictionResult, vector: str) -> float:
        series = result.forecasts["posterior"].values[vectors.index(vector), first:]
        return float(series.mean())

    # PRO1 drawn down to 60 bar produces more, PRO2 held back at 120 bar less.
    assert produced(bhp, "WOPR:PRO1") > produced(hold, "WOPR:PRO1")
    assert produced(bhp, "WOPR:PRO2") < produced(hold, "WOPR:PRO2")
