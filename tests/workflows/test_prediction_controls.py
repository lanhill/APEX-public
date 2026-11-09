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


def test_a_scenario_the_simulator_does_not_read_is_refused(finished: object) -> None:
    """Release tester H35: the linear model read only ``x``, so a scenario ``u`` was accepted
    and forecast as if absent; a simulator that declares what it reads refuses it."""
    path, _ = _prediction(finished, '[[controls.variables]]\nname = "u"\nvalues = [1.5, -0.5]\n')
    path.write_text(path.read_text().replace('variables = ["x", "u"]', 'variables = ["x"]'))

    with pytest.raises(ConfigError, match=r"gives \['u'\], which the linear simulator does not"):
        prediction.predict(path)
    # Refused before anything is written: no folder (nor its lock) is left behind.
    folder = Path(finished.directory) / "runs" / "case" / "predictions" / "scenario"  # type: ignore[attr-defined]
    assert not folder.exists()


def test_a_deck_templates_names_are_the_values_it_reads(tmp_path: Path) -> None:
    from apex.simulation.deck import template_names

    template = tmp_path / "case.mako"
    template.write_text(
        'WCONINJE\n "INJ" WATER OPEN BHP 2* ${bhp[0]} /\n/\n'
        "% for value in poro:\n${value}\n% endfor\n"
        'INCLUDE\n "${case_dir}/grid.inc" /\n'
    )
    assert {"bhp", "poro", "case_dir"} <= template_names(template)  # type: ignore[operator]
    assert "rate" not in template_names(template)  # type: ignore[operator]
    template.write_text('<%include file="other.mako"/>\n')
    assert template_names(template) is None  # what another template reads is not seen


def test_the_scenario_is_in_the_provenance_and_the_summary(finished: object) -> None:
    """Release tester H35: the scenario was only in the prediction's config.json."""
    import json

    from apex.services.summaries import PredictionSummarizer

    path, _ = _prediction(finished, '[[controls.variables]]\nname = "u"\nvalues = [1.5, -0.5]\n')
    prediction.predict(path)
    folder = Path(finished.directory) / "runs" / "case" / "predictions" / "scenario"  # type: ignore[attr-defined]

    recorded = json.loads((folder / "provenance.json").read_text())
    assert recorded["controls"] == {"u": [1.5, -0.5]}
    summary = PredictionSummarizer().summarize(folder)
    assert (
        "control scenario (as given, before its parameterizations): u = [1.5, -0.5]"
        in summary.notes
    )


def test_every_levels_template_counts_and_an_unknown_one_makes_all_unknown(tmp_path: Path) -> None:
    """The review: a multilevel deck's coarse level may render a template of its own."""
    from apex.simulation.deck import FidelityLevels, templates_names

    fine, coarse = tmp_path / "fine.mako", tmp_path / "coarse.mako"
    fine.write_text("${poro}\n")
    coarse.write_text("${poro} ${bhp[0]}\n")
    levels = FidelityLevels([{"first": 0, "size": 2, "deck_template_file": str(coarse)}], fine)
    assert levels.templates() == [fine, coarse.resolve()]
    assert {"poro", "bhp"} <= templates_names(levels.templates())  # type: ignore[operator]
    coarse.write_text("${context['bhp']}\n")
    assert templates_names(levels.templates()) is None


def test_a_simulator_that_cannot_tell_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    """The review: no capability, or None, is a warning, never a refusal."""
    for simulator in (object(), type("Unknown", (), {"state_names_read": lambda self: None})()):
        caplog.clear()
        prediction._check_controls_read(simulator, "custom", {"bhp": np.ones(2)})
        assert [r for r in caplog.records if getattr(r, "event", None) == "controls_unchecked"]


def test_validation_refuses_a_scenario_the_start_would(finished: object) -> None:
    """Release tester (H35 re-check): `apex validate` accepted a scenario that `apex predict`
    then refused at its start; the scenario's checks now run before submission too."""
    from apex.services.api import ApexService

    unread, _ = _prediction(finished, '[[controls.variables]]\nname = "u"\nvalues = [1.5, -0.5]\n')
    unread.write_text(unread.read_text().replace('variables = ["x", "u"]', 'variables = ["x"]'))
    with pytest.raises(ConfigError, match=r"gives \['u'\], which the linear simulator does not"):
        prediction.check_prediction(prediction.load_prediction_config(unread))

    named, _ = _prediction(finished, '[[controls.variables]]\nname = "x"\nvalues = [1.0]\n',
                           name="clash")  # fmt: skip
    with pytest.raises(ConfigError, match=r"names \['x'\]"):
        prediction.check_prediction(prediction.load_prediction_config(named))

    # One the simulator reads passes, and nothing is written by the check.
    read, _ = _prediction(finished, '[[controls.variables]]\nname = "u"\nvalues = [1.5, -0.5]\n',
                          name="fine")  # fmt: skip
    prediction.check_prediction(prediction.load_prediction_config(read))
    assert not (Path(finished.directory) / "runs" / "case" / "predictions").exists()  # type: ignore[attr-defined]
    # Submission runs the same check (the REST API, the client and the tools).
    service = ApexService(Path(finished.directory) / "service")  # type: ignore[attr-defined]
    with pytest.raises(ConfigError, match="does not read"):
        service.submit_prediction(unread)
