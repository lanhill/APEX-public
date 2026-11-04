"""A plain prediction's configuration dumps as before ``[surrogate]`` existed.

Its ``config.json`` and fingerprint are the dump: a changed dump would refuse to resume
every prediction folder made before (``PredictionDirectory.open``). The literal is the dump
at the commit before the section (1fb0b331).
"""

import pytest

from apex.domain.prediction import PredictionConfig

BEFORE = (
    '{"prediction":{"name":"p","run_dir":"/r","ensembles":["posterior","prior"],'
    '"include_mean_model":false,"check_history":true,"percentile_convention":"percentile",'
    '"scores":["coverage_p10_p90","crps","mean_error","spread"],"output_dir":null,'
    '"outputs":{"vectors":["WOPR:*"],"times":"report"}},"simulator":{"name":null,'
    '"options":{}},"execution":null}'
)


def test_a_plain_predictions_dump_is_unchanged() -> None:
    config = PredictionConfig.model_validate(
        {"prediction": {"name": "p", "run_dir": "/r", "outputs": {"vectors": ["WOPR:*"]}}}
    )
    assert config.model_dump_json() == BEFORE
    assert '"surrogate"' not in config.model_dump_json(indent=2)
    assert "surrogate" not in config.model_dump()
    assert PredictionConfig.model_validate_json(BEFORE) == config


def test_a_surrogate_section_is_dumped_and_read_back() -> None:
    config = PredictionConfig.model_validate(
        {
            "prediction": {"name": "p", "run_dir": "/r", "outputs": {"vectors": ["WOPR:*"]}},
            "surrogate": {"model_file": "/m/surrogate.json"},
        }
    )
    dumped = config.model_dump_json()
    assert '"surrogate":{"model_file":"/m/surrogate.json"' in dumped
    assert PredictionConfig.model_validate_json(dumped) == config


def test_a_plain_predictions_dump_has_no_controls() -> None:
    config = PredictionConfig.model_validate_json(BEFORE)
    assert "controls" not in config.model_dump()
    assert config.model_dump_json() == BEFORE


def test_a_control_scenario_is_dumped_and_read_back() -> None:
    config = PredictionConfig.model_validate(
        {
            "prediction": {"name": "p", "run_dir": "/r", "outputs": {"vectors": ["WOPR:*"]}},
            "controls": {"variables": [{"name": "bhp", "values": [130.0, 140.0]}]},
        }
    )
    dumped = config.model_dump_json()
    assert '"controls":{"variables":[{"name":"bhp","values":[130.0,140.0]' in dumped
    assert PredictionConfig.model_validate_json(dumped) == config


def test_the_serialization_schema_keeps_its_shape() -> None:
    # The re-review's note: a wrap serializer annotated Any left this schema empty.
    schema = PredictionConfig.model_json_schema(mode="serialization")
    assert schema.get("type") == "object"


@pytest.mark.parametrize(
    ("variable", "message"),
    [
        ({"name": "bhp", "values": [500.0], "upper": 300.0}, "outside their bounds"),
        ({"name": "bhp", "values": []}, "no value"),
        ({"name": "bhp", "values": [1.0], "lower": 5.0, "upper": 2.0}, "lower must be below"),
    ],
)
def test_invalid_controls_are_refused(variable: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PredictionConfig.model_validate(
            {
                "prediction": {"name": "p", "run_dir": "/r", "outputs": {"vectors": ["y"]}},
                "controls": {"variables": [variable]},
            }
        )


def test_repeated_controls_are_refused() -> None:
    with pytest.raises(ValueError, match="controls repeat"):
        PredictionConfig.model_validate(
            {
                "prediction": {"name": "p", "run_dir": "/r", "outputs": {"vectors": ["y"]}},
                "controls": {
                    "variables": [{"name": "u", "values": 1.0}, {"name": "u", "values": 2.0}]
                },
            }
        )
