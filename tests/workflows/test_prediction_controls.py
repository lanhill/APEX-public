"""A prediction's control scenario (``[controls]``; decision 6).

The prediction "deck" is the case's linear model with four more report times and two more
columns, the controls ``u``: ``y = G_pred @ [x; u]``, the history rows' control columns 0,
so a scenario changes only the forecast and every value is known exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition, prediction
from apex.testing import CaseFactory

N_HISTORY, N_FUTURE = 8, 4


def _prediction(
    case: object, controls: str, *, name: str = "scenario", n_controls: int = 2, body: str = ""
) -> tuple[Path, np.ndarray]:
    """Write the controlled model and a prediction with ``controls``; return it and ``G_pred``."""
    directory = Path(case.directory)  # type: ignore[attr-defined]
    forward = np.asarray(case.forward)  # type: ignore[attr-defined]
    rng = np.random.default_rng(11)
    history = np.hstack([forward, np.zeros((forward.shape[0], n_controls))])
    future = rng.standard_normal((N_FUTURE, forward.shape[1] + n_controls))
    g_pred = np.vstack([history, future])
    np.savez(directory / f"forward_{name}.npz", matrix=g_pred)
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(N_HISTORY + N_FUTURE))
    path = directory / f"{name}.toml"
    path.write_text(
        f"""
[prediction]
name = "{name}"
run_dir = "runs/case"
ensembles = ["posterior", "truth"]
{body}

[prediction.outputs]
vectors = ["y"]

[simulator]
options = {{ forward_file = "forward_{name}.npz", outputs = [{outputs}], variables = ["x", "u"] }}

{controls}
"""
    )
    return path, g_pred


@pytest.fixture
def finished(linear_case: CaseFactory) -> object:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    return case


def _posterior(case: object) -> np.ndarray:
    results = composition.load_results(Path(case.directory) / "runs" / "case")  # type: ignore[attr-defined]
    return np.asarray(results.iterate(results.posterior_iteration).state["x"])


def test_every_member_and_the_truth_are_given_the_scenario(finished: object) -> None:
    path, g_pred = _prediction(
        finished, '[[controls.variables]]\nname = "u"\nvalues = [1.5, -0.5]\n'
    )

    result = prediction.predict(path)

    u = np.array([1.5, -0.5])
    posterior = _posterior(finished)
    stacked = np.vstack([posterior, np.tile(u[:, None], (1, posterior.shape[1]))])
    np.testing.assert_allclose(
        result.forecasts["posterior"].values[0], g_pred @ stacked, rtol=1e-13
    )
    truth = np.load(Path(finished.directory) / "truth.npz")["x"]  # type: ignore[attr-defined]
    assert result.truth is not None
    np.testing.assert_allclose(result.truth[0], g_pred @ np.concatenate([truth, u]), rtol=1e-13)


def test_controls_are_mapped_by_their_own_parameterizations(finished: object) -> None:
    # Two values held over three report steps by control_schedule: knots 0 and 2 give
    # u = [3, 3, 7] (an identity map would pass a wrong spread unnoticed: the review).
    path, g_pred = _prediction(
        finished,
        '[[controls.variables]]\nname = "rate"\nvalues = [3.0, 7.0]\n\n'
        '[[controls.parameterizations]]\nname = "control_schedule"\n'
        'options = { variable = "rate", output = "u", knots = [0, 2], n_steps = 3 }\n',
        n_controls=3,
    )

    result = prediction.predict(path)

    posterior = _posterior(finished)
    u = np.array([3.0, 3.0, 7.0])
    stacked = np.vstack([posterior, np.tile(u[:, None], (1, posterior.shape[1]))])
    np.testing.assert_allclose(
        result.forecasts["posterior"].values[0], g_pred @ stacked, rtol=1e-13
    )


def test_the_mean_model_is_given_the_scenario_too(finished: object) -> None:
    path, g_pred = _prediction(
        finished,
        '[[controls.variables]]\nname = "u"\nvalues = [0.5, 2.0]\n',
        body="include_mean_model = true",
    )

    result = prediction.predict(path)

    mean = result.forecasts["posterior"].mean_model
    assert mean is not None
    x = _posterior(finished).mean(axis=1)
    np.testing.assert_allclose(mean[0], g_pred @ np.concatenate([x, [0.5, 2.0]]), rtol=1e-13)


def test_a_control_named_as_a_deck_template_name_is_refused(finished: object) -> None:
    path, _ = _prediction(finished, '[[controls.variables]]\nname = "case_dir"\nvalues = [1.0]\n')

    with pytest.raises(ConfigError, match=r"names \['case_dir'\], which the deck template"):
        prediction.predict(path)


def test_the_controls_parameterizations_are_resolved_and_confined(tmp_path: Path) -> None:
    from apex.domain.prediction import PredictionConfig
    from apex.services.policy import _plugin_sections

    config = PredictionConfig.model_validate(
        {
            "prediction": {"name": "p", "run_dir": "run", "outputs": {"vectors": ["y"]}},
            "controls": {
                "variables": [{"name": "u", "values": [1.0]}],
                "parameterizations": [
                    {"name": "pgs_truncation", "options": {"rule_file": "r.toml"}}
                ],
            },
        }
    )
    resolved = prediction.resolve_prediction_config(config, tmp_path)
    assert resolved.controls is not None
    rule = resolved.controls.parameterizations[0].options["rule_file"]
    assert rule == str((tmp_path / "r.toml").resolve())
    # A [policy] and the tools' confinement see what the controls' parameterizations name.
    sections = _plugin_sections(resolved)
    assert ("parameterizations", resolved.controls.parameterizations[0]) in sections


def test_another_scenario_is_another_prediction(finished: object) -> None:
    path, _ = _prediction(finished, '[[controls.variables]]\nname = "u"\nvalues = [1.0, 1.0]\n')
    prediction.predict(path)
    path.write_text(path.read_text().replace("values = [1.0, 1.0]", "values = [2.0, 1.0]"))

    with pytest.raises(ConfigError, match="another configuration"):
        prediction.predict(path)


def test_a_control_named_as_a_state_variable_is_refused(finished: object) -> None:
    path, _ = _prediction(finished, '[[controls.variables]]\nname = "x"\nvalues = [1.0]\n')

    with pytest.raises(ConfigError, match=r"names \['x'\]"):
        prediction.predict(path)


def test_an_unknown_control_parameterization_is_named(finished: object) -> None:
    path, _ = _prediction(
        finished,
        '[[controls.variables]]\nname = "rate"\nvalues = [1.0, 2.0]\n\n'
        '[[controls.parameterizations]]\nname = "no_such_map"\n',
    )

    with pytest.raises(ConfigError, match=r"\[\[controls.parameterizations\]\] no_such_map"):
        prediction.predict(path)
