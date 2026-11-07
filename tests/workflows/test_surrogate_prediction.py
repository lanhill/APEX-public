"""Predictions with a surrogate.

The case is linear (``y = G x``) with two report times beyond its four observed ones, and
the surrogates are test-only exact maps (``surrogate_cases.Affine`` from the inputs, and
``EarlyLinear`` from each member's own run up to time "2"), built on series that reach the
prediction's horizon. That is the independent reference: a prediction with them gives the
plain prediction's forecasts to round-off, its history check passes within the gate, the
true model is the simulator's, and each member's source is recorded. Refused: times beyond
the model's keys (its horizon), vectors it does not answer, another deck than the build's.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from apex.core.errors import ConfigError, SurrogateDomainError, WorkflowError
from apex.core.plugins import simulators, surrogate_models
from apex.services import composition, surrogates
from apex.services.prediction import predict
from apex.storage.surrogate_files import read_surrogate, write_surrogate


def _load_cases() -> ModuleType:
    path = Path(__file__).with_name("surrogate_cases.py")
    spec = importlib.util.spec_from_file_location("_surrogate_cases_prediction", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cases = _load_cases()
AFFINE, EARLY, STOPPING = "test_affine_pred", "test_early_pred", "test_linear_stops_pred"
#: The history's report times and two beyond it.
TIMES = (*cases.TIMES, "5", "6")
ROUND_OFF = 1e-9


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register(AFFINE)(cases.Affine)
    surrogate_models.register(EARLY)(cases.EarlyLinear)
    simulators.register(STOPPING)(cases.StoppingLinear)
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        surrogate_models.remove(AFFINE)
        surrogate_models.remove(EARLY)
        simulators.remove(STOPPING)


def _case(folder: Path, simulator: str = STOPPING) -> Path:
    """The linear case whose simulator answers two report times beyond its observations."""
    config: Path = cases.write_case(folder, name="hm", simulator=simulator, members=40)
    forward = cases.forward_matrix()
    observed = cases.keys_in_order()
    rng = np.random.default_rng(5)
    rows, outputs = [], []
    for datatype in cases.DATATYPES:
        for time in TIMES:
            key = (time, datatype)
            rows.append(forward[observed.index(key)] if key in observed else rng.normal(size=3))
            outputs.append(f'["{time}", "{datatype}"]')
    np.savez(folder / "forward.npz", matrix=np.array(rows))
    text = config.read_text()
    start = text.index("outputs = [")
    end = text.index("]\n", text.index("]]", start)) + 1
    config.write_text(text[:start] + f"outputs = [{', '.join(outputs)}]" + text[end:])
    return config


def _build(folder: Path, plugin: str, name: str) -> Path:
    vectors = ", ".join(f'"{d}"' for d in cases.DATATYPES)
    times = ", ".join(f'"{t}"' for t in TIMES)
    path = folder / f"build_{name}.toml"
    path.write_text(f"""
[surrogate]
name = "{name}"
model = "{plugin}"
seed = 7

[case]
config_file = "hm.toml"
outputs = {{ vectors = [{vectors}], times = [{times}] }}

[design]
plan = "lhs"
training = 30
held_out = 10
box = {{ x = [-6.0, 6.0] }}
""")
    result = surrogates.build(path)
    assert result.summary.gate is not None
    assert result.summary.gate.passed, result.summary.gate.failures()
    return result.path / "model" / "surrogate.json"


def _prediction(folder: Path, name: str, surrogate: str = "", times: str = '"report"') -> Path:
    path = folder / f"{name}.toml"
    path.write_text(f"""
[prediction]
name = "{name}"
run_dir = "runs/hm"
ensembles = ["posterior", "prior", "truth"]

[prediction.outputs]
vectors = ["WOPR:*", "WBHP:*"]
times = {times}

{surrogate}
""")
    return path


@pytest.fixture
def run(tmp_path: Path) -> Path:
    """A finished ES-MDA run of the case."""
    composition.start(_case(tmp_path))
    return tmp_path


@pytest.mark.parametrize("plugin", [AFFINE, EARLY])
def test_a_prediction_with_an_exact_surrogate_is_the_simulators(run: Path, plugin: str) -> None:
    """Members by the model (an early-start one after each member's own early run), the
    truth by the simulator; the forecasts are the plain prediction's to round-off, and the
    history check passes within the gate."""
    model = _build(run, plugin, f"m_{plugin}")
    times = ", ".join(f'"{t}"' for t in TIMES)
    plain = predict(_prediction(run, "plain", times=f"[{times}]"))
    cases.StoppingLinear.stopped.clear()
    section = f'[surrogate]\nmodel_file = "{model}"'
    result = predict(_prediction(run, "fast", section))
    for name, forecast in result.forecasts.items():
        assert forecast.times == TIMES
        np.testing.assert_allclose(
            forecast.values, plain.forecasts[name].values, rtol=0, atol=ROUND_OFF
        )
    assert result.truth is not None
    assert plain.truth is not None
    np.testing.assert_array_equal(result.truth, plain.truth)
    folder = run / "runs" / "hm" / "predictions" / "fast"
    state = json.loads((folder / "state.json").read_text())
    sources = state["member_sources"]
    assert set(sources) == {"posterior", "prior"}
    assert all(s == "surrogate" for by in sources.values() for s in by.values())
    from apex.services.summaries import summarize

    [summary] = summarize([folder])
    assert any("40 member(s) by the surrogate" in note for note in summary.notes)
    provenance = json.loads((folder / "provenance.json").read_text())
    assert provenance["surrogate"]["identity"] == read_surrogate(model).identity
    stopped = cases.StoppingLinear.stopped
    if plugin == EARLY:
        assert len(stopped) == 2 * 40  # each member of both ensembles, once
        assert {date for _, date, _ in stopped} == {cases.EARLY_UNTIL}
    else:
        assert not stopped


def test_what_a_surrogate_prediction_refuses(run: Path) -> None:
    model = _build(run, AFFINE, "m")
    section = f'[surrogate]\nmodel_file = "{model}"'
    with pytest.raises(ConfigError, match=r"7 is not a report time of the surrogate"):
        predict(_prediction(run, "far", section, times='["5", "7"]'))
    path = _prediction(run, "gas", section)
    path.write_text(path.read_text().replace('"WBHP:*"', '"WGPR:*"'))
    with pytest.raises(ConfigError, match="no vector matches 'WGPR:\\*'"):
        predict(path)
    # Another deck than the build's: a changed forward model.
    np.savez(run / "forward_other.npz", matrix=2.0 * np.load(run / "forward.npz")["matrix"])
    other = _prediction(run, "other", section)
    other.write_text(
        other.read_text() + '\n[simulator.options]\nforward_file = "forward_other.npz"\n'
    )
    with pytest.raises(ConfigError, match="built for another simulator stack"):
        predict(other)


def test_members_outside_the_domain_stop_or_run_by_the_simulator(run: Path) -> None:
    """A model built on a narrow box: the prior's members lie outside it."""
    path = _build_narrow(run)
    with pytest.raises(SurrogateDomainError):
        predict(_prediction(run, "stop", f'[surrogate]\nmodel_file = "{path}"'))
    section = f'[surrogate]\nmodel_file = "{path}"\noutside = "simulator"'
    result = predict(_prediction(run, "mixed", section))
    folder = run / "runs" / "hm" / "predictions" / "mixed"
    sources = json.loads((folder / "state.json").read_text())["member_sources"]["prior"]
    assert "simulator" in sources.values()
    assert len(result.forecasts["prior"].member_ids) == 40


def _build_narrow(folder: Path, half_width: float = 0.3) -> Path:
    path = folder / "build_narrow.toml"
    _build(folder, AFFINE, "wide")  # writes build_wide.toml, the template
    path.write_text(
        (folder / "build_wide.toml").read_text()
        .replace('name = "wide"', 'name = "narrow"')
        .replace("box = { x = [-6.0, 6.0] }", f"box = {{ x = [{-half_width}, {half_width}] }}")
    )  # fmt: skip
    result = surrogates.build(path)
    return result.path / "model" / "surrogate.json"


def test_the_history_check_fails_a_surrogate_that_does_not_hold(run: Path) -> None:
    """A model biased by 3 sigma, its gate recorded as passed: the history check within the
    gate fails the prediction."""
    model = _build(run, AFFINE, "m")
    stored = read_surrogate(model)
    fitted = cases.Affine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[:, -1] += 3.0 * np.sqrt(cases.VARIANCE)
    biased = model.parent.parent / "model_biased" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        biased, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    with pytest.raises(WorkflowError, match="within its gate"):
        predict(_prediction(run, "biased", f'[surrogate]\nmodel_file = "{biased}"'))


def test_a_resumed_prediction_knows_the_surrogates_members(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped while the simulator runs the members outside the domain: the surrogate's
    members, saved already, keep their source on resume (review), so the history check
    gates them instead of comparing them exactly."""
    from apex.workflows.prediction import PredictionWorkflow

    # A box the posterior lies in and the prior only partly: the prior is mixed.
    path = _build_narrow(run, half_width=1.5)
    section = f'[surrogate]\nmodel_file = "{path}"\noutside = "simulator"'
    config = _prediction(run, "resumed", section)
    original = PredictionWorkflow._forecast_columns

    def interrupted(*args: object, **kwargs: object) -> dict[int, str]:
        raise KeyboardInterrupt

    monkeypatch.setattr(PredictionWorkflow, "_forecast_columns", interrupted)
    with pytest.raises(KeyboardInterrupt):
        predict(config)
    folder = run / "runs" / "hm" / "predictions" / "resumed"
    saved = json.loads((folder / "state.json").read_text())["member_sources"]["prior"]
    assert saved
    assert set(saved.values()) == {"surrogate"}
    monkeypatch.setattr(PredictionWorkflow, "_forecast_columns", original)
    predict(config)
    final = json.loads((folder / "state.json").read_text())["member_sources"]["prior"]
    assert {k: v for k, v in final.items() if k in saved} == saved
    assert "simulator" in final.values()
    assert len(final) == 40


def test_non_finite_surrogate_values_stop_the_prediction(run: Path) -> None:
    model = _build(run, AFFINE, "m")
    stored = read_surrogate(model)
    fitted = cases.Affine.from_arrays(stored.manifest.model, stored.arrays)
    fitted.weights = fitted.weights.copy()
    fitted.weights[0, -1] = np.nan
    broken = model.parent.parent / "model_nan" / "surrogate.json"
    manifest = stored.manifest
    write_surrogate(
        broken, plugin=manifest.plugin, model=fitted, domain=manifest.training_domain(),
        provenance=manifest.provenance, options=manifest.options, gate=manifest.gate,
        snapshots=manifest.snapshots, fit=manifest.fit, apex_commit=manifest.apex_commit,
    )  # fmt: skip
    with pytest.raises(WorkflowError, match="non-finite"):
        predict(_prediction(run, "nan", f'[surrogate]\nmodel_file = "{broken}"'))


def test_the_mean_model_is_the_surrogates_too(run: Path) -> None:
    model = _build(run, AFFINE, "m")
    times = ", ".join(f'"{t}"' for t in TIMES)
    plain_path = _prediction(run, "plain_mean", times=f"[{times}]")
    plain_path.write_text(plain_path.read_text().replace(
        "[prediction.outputs]", "include_mean_model = true\n\n[prediction.outputs]"))  # fmt: skip
    path = _prediction(run, "fast_mean", f'[surrogate]\nmodel_file = "{model}"')
    path.write_text(path.read_text().replace(
        "[prediction.outputs]", "include_mean_model = true\n\n[prediction.outputs]"))  # fmt: skip
    plain, fast = predict(plain_path), predict(path)
    for name, forecast in fast.forecasts.items():
        mine, theirs = forecast.mean_model, plain.forecasts[name].mean_model
        assert mine is not None
        assert theirs is not None
        assert float(np.max(np.abs(mine - theirs))) <= ROUND_OFF
    state = json.loads(
        (run / "runs" / "hm" / "predictions" / "fast_mean" / "state.json").read_text()
    )
    assert set(state["member_sources"]["posterior:mean"].values()) == {"surrogate"}


def test_report_times_are_in_time_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Step labels "1".."12": "10" after "9", not after "1" (review)."""
    from types import SimpleNamespace
    from typing import Any, cast

    from apex.core.simulation import SeriesRequest
    from apex.domain.observations import DataKey
    from apex.domain.surrogate import GateTolerances

    labels = [str(i) for i in range(1, 13)]
    model = SimpleNamespace(keys=tuple(DataKey(t, "WOPR:P1") for t in sorted(labels)))
    manifest = SimpleNamespace(
        gate=SimpleNamespace(passed=True, tolerances=GateTolerances(), datatypes=[]),
        provenance=SimpleNamespace(stack_mismatches=lambda other: [], case_digest=None),
        control_box=None,
    )
    loaded = SimpleNamespace(manifest=manifest, model=model, identity="i", domain=None,
                             simulator=None)  # fmt: skip
    monkeypatch.setattr(surrogates, "active_model", lambda path: loaded)
    monkeypatch.setattr(surrogates, "last_check", lambda path, identity: None)
    monkeypatch.setattr(surrogates, "stack_provenance", lambda case, simulator: None)
    monkeypatch.setattr(
        surrogates, "check_request", lambda domain, keys: SimpleNamespace(inside=True)
    )
    section = SimpleNamespace(model_file=Path("m.json"), workers=None, outside="stop")
    _, request = surrogates.prediction_surrogate(
        cast(Any, section), cast(Any, None), cast(Any, None), None,
        cast(Any, SimpleNamespace(identity=True)), SeriesRequest(("WOPR:*",), None), "serial",
    )  # fmt: skip
    assert request.times == tuple(labels)


def test_a_prediction_killed_while_it_saves_knows_every_saved_members_source(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The surrogate's members' sources are recorded before any is saved (coder's review):
    killed after the first save, the resumed prediction gates them all within the gate."""
    from apex.storage.prediction_directory import PredictionDirectory

    model = _build(run, AFFINE, "m")
    config = _prediction(run, "killed", f'[surrogate]\nmodel_file = "{model}"')
    original = PredictionDirectory.save_member
    saved: list[int] = []

    def dies(self: PredictionDirectory, ensemble: str, member_id: int, values: object) -> None:
        if ensemble == "posterior" and len(saved) == 1:
            raise KeyboardInterrupt  # as a SIGKILL would, between two saves
        original(self, ensemble, member_id, values)  # type: ignore[arg-type]
        if ensemble == "posterior":
            saved.append(member_id)

    monkeypatch.setattr(PredictionDirectory, "save_member", dies)
    with pytest.raises(KeyboardInterrupt):
        predict(config)
    folder = run / "runs" / "hm" / "predictions" / "killed"
    sources = json.loads((folder / "state.json").read_text())["member_sources"]["posterior"]
    assert sources[str(saved[0])] == "surrogate"
    monkeypatch.setattr(PredictionDirectory, "save_member", original)
    result = predict(config)  # its history check gates the saved member, which passes
    assert len(result.forecasts["posterior"].member_ids) == 40
