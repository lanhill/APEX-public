"""History matching with a model-error model (MEC) on a biased linear case.

The observations come from ``G x + b`` with an offset ``b`` the simulator ``G x`` lacks.
A model-error model ``r = theta`` estimated with ``x`` should take up the offset: the
effective data mismatch falls below the run without it, and theta approaches ``b``.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.core.plugins import model_error_models, simulators
from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey, ObservationSet
from apex.services import composition
from apex.simulation.model_error import RESIDUAL_PREFIX
from apex.storage.inputs import read_observation_section

N_STATE, N_DATA, OFFSET = 3, 8, 3.0
RNG = np.random.default_rng(7)
G = RNG.standard_normal((N_DATA, N_STATE))
TRUTH = np.array([0.5, -1.0, 0.8])
VARIANCE = 0.05


class FileLinear:
    """``G x`` written to and read back from the member's work directory."""

    needs_work_dir = True
    #: Member ids whose runs fail, to check that residual columns follow the survivors.
    fail_members: frozenset[int] = frozenset()

    def simulate(self, member: MemberInput) -> Path:
        assert member.work_dir is not None
        if member.member_id in self.fail_members:
            raise SimulationError(f"member {member.member_id} fails (on purpose)")
        assert set(member.state) == {"x"}  # the model-error parameters stay out of the deck
        np.save(member.work_dir / "MODEL.npy", G @ np.asarray(member.state["x"]))
        return member.work_dir / "MODEL"

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        values = np.load(case.with_suffix(".npy"))
        return np.array([values[int(k.time[1:])] for k in keys])

    def clean(self, case: Path) -> None:
        case.with_suffix(".npy").unlink()

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        case = self.simulate(member)
        values = self.read(case, keys)
        self.clean(case)
        return values


class Offset:
    """``r = theta``: one constant offset on every ``y`` datum."""

    n_params = 1

    def __init__(self, state_variable: str = "nn_params") -> None:
        self.state_variable = state_variable

    def fit(self, observations: ObservationSet) -> None:
        pass

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        return [DataKey(k.time, RESIDUAL_PREFIX + k.datatype) for k in keys]

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        assert case.with_suffix(".npy").exists()  # read before the simulator cleans up
        return np.full(len(keys), theta[0])


@pytest.fixture(autouse=True)
def plugins() -> Iterator[None]:
    simulators.register("test_file_linear")(FileLinear)
    model_error_models.register("test_offset")(Offset)
    try:
        yield
    finally:
        simulators._factories.pop("test_file_linear")
        model_error_models._factories.pop("test_offset")


def _case(directory: Path, *, mec: bool, size: int = 1) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(3).standard_normal(N_DATA) * np.sqrt(VARIANCE)
    observed = G @ TRUTH + OFFSET + noise
    rows = ["time,datatype,value,variance"] + [
        f"t{i},y,{float(observed[i])!r},{VARIANCE}" for i in range(N_DATA)
    ]
    (directory / "observations.csv").write_text("\n".join(rows) + "\n")
    np.savez(directory / "truth.npz", x=TRUTH)  # no true value of the offset
    theta = f"""
[[state.variables]]
name = "nn_params"
size = {size}

[prior.variables.nn_params]
name = "gaussian"
options = {{ mean = 0.0, std = 2.0 }}

[model_error]
name = "test_offset"
"""
    config = directory / "config.toml"
    config.write_text(
        f"""
[run]
name = "run"
seed = 11

[[state.variables]]
name = "x"
size = {N_STATE}
{theta if mec else ""}
[prior]
ensemble_size = 60
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = 1.0 }} }}

[observations]
file = "observations.csv"

[case]
truth_file = "truth.npz"

[simulator]
name = "test_file_linear"

[method]
name = "gies"
strategy = {{ name = "rlmmac", options = {{ energy = 1.0 }} }}
max_iterations = 8
"""
    )
    return config


def test_mec_takes_up_a_bias_the_simulator_lacks(tmp_path: Path) -> None:
    with_mec = composition.start(_case(tmp_path / "mec", mec=True))
    without = composition.start(_case(tmp_path / "plain", mec=False))

    final = [r.data_mismatch_mean for r in with_mec.history if r.accepted][-1]
    plain = [r.data_mismatch_mean for r in without.history if r.accepted][-1]
    assert final < 0.2 * plain
    theta = with_mec.posterior[N_STATE]
    assert abs(theta.mean() - OFFSET) < 0.5
    # The metrics that compare with the truth score x only.
    metrics = [r for r in with_mec.history if r.accepted][-1].metrics
    assert metrics is not None
    rmse = metrics["rmse"]
    assert set(rmse) == {"x", "all"}


def test_the_residuals_are_saved_with_each_ensemble(tmp_path: Path) -> None:
    composition.start(_case(tmp_path, mec=True))

    saved = np.load(tmp_path / "runs" / "run" / "checkpoints" / "iter_0.npz")
    assert list(saved["auxiliary_keys"]) == [f"MEC:y@t{i}" for i in range(N_DATA)]
    residuals = saved["auxiliary_predicted"]
    assert residuals.shape == (N_DATA, 60)
    np.testing.assert_array_equal(residuals, np.repeat(saved["state"][None, N_STATE], N_DATA, 0))
    # The run reader gives them back (for the residual figures).
    iterate = composition.load_results(tmp_path / "runs" / "run").iterate(0)
    assert iterate.auxiliary_predicted is not None
    np.testing.assert_array_equal(iterate.auxiliary_predicted, residuals)
    assert iterate.auxiliary_keys[0] == "MEC:y@t0"
    # The effective forecast is g + r.
    x = saved["state"][:N_STATE]
    np.testing.assert_allclose(saved["predicted"], G @ x + residuals)


def test_a_state_variable_of_the_wrong_size_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="'nn_params' has 2 values, the model 1"):
        composition.start(_case(tmp_path, mec=True, size=2))


def test_residual_columns_follow_the_members_that_succeeded(tmp_path: Path) -> None:
    config = _case(tmp_path, mec=True)
    config.write_text(config.read_text() + "\n[failures]\nmin_success_fraction = 0.5\n")
    FileLinear.fail_members = frozenset({3, 17})
    try:
        composition.start(config)
    finally:
        FileLinear.fail_members = frozenset()

    saved = np.load(tmp_path / "runs" / "run" / "checkpoints" / "iter_0.npz")
    assert 3 not in saved["member_ids"]
    assert saved["auxiliary_predicted"].shape == (N_DATA, 58)
    np.testing.assert_array_equal(saved["auxiliary_predicted"][0], saved["state"][N_STATE])


def test_what_mec_cannot_support_is_refused(tmp_path: Path) -> None:
    config = composition.load_config(_case(tmp_path, mec=True))
    observations = read_observation_section(config.observations)
    layout = composition.build_layout(config)
    levels = config.model_copy(
        update={"method": config.method.model_copy(update={"options": {"levels": []}})}
    )
    with pytest.raises(ConfigError, match="cannot be combined with fidelity levels"):
        composition.build_model_error(levels, layout, observations)

    from apex.integrations.ert.export import _check_exportable

    with pytest.raises(ConfigError, match="model-error model"):
        _check_exportable(config)


def test_a_prediction_of_a_mec_run_is_refused(tmp_path: Path) -> None:
    from apex.services.prediction import predict

    composition.start(_case(tmp_path, mec=True))
    prediction = tmp_path / "future.toml"
    prediction.write_text(
        '[prediction]\nname = "future"\nrun_dir = "runs/run"\nensembles = ["posterior"]\n\n'
        '[prediction.outputs]\nvectors = ["y"]\n'
    )
    with pytest.raises(ConfigError, match=r"model-error model .* that does not forecast"):
        predict(prediction)


def _forecasting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Offset`` a model whose residuals apply at any time."""
    monkeypatch.setattr(Offset, "forecasts", True, raising=False)


def test_a_prediction_with_a_forecasting_model_error_is_g_plus_r(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose model-error model forecasts is predicted with it,
    every member's forecast ``G x + theta`` (the run's own, so the history check holds);
    with ``[surrogate]`` it is refused (a surrogate forecasts g alone)."""
    from apex.services.prediction import predict
    from apex.storage.run_results import RunResults

    _forecasting(monkeypatch)
    composition.start(_case(tmp_path, mec=True))
    times = ", ".join(f'"t{i}"' for i in range(N_DATA))
    prediction = tmp_path / "future.toml"
    prediction.write_text(
        '[prediction]\nname = "future"\nrun_dir = "runs/run"\n'
        'ensembles = ["posterior", "truth"]\ntruth_is_simulator = true\n\n'
        f'[prediction.outputs]\nvectors = ["y"]\ntimes = [{times}]\n'
    )
    result = predict(prediction)
    recorded = json.loads((tmp_path / "runs/run/predictions/future/provenance.json").read_text())
    assert recorded["model_error"] == {"name": "test_offset", "truth_is_simulator": True}
    # The truth's parameters are not stored (auxiliary: zero), so its forecast is g.
    assert result.truth is not None
    np.testing.assert_allclose(result.truth[0], G @ TRUTH, rtol=1e-12)
    config = composition.load_config(tmp_path / "config.toml")
    results = RunResults(tmp_path / "runs" / "run", composition.build_layout(config))
    state = results.iterate(results.posterior_iteration).state
    expected = G @ np.asarray(state["x"]) + np.asarray(state["nn_params"])[0]
    np.testing.assert_allclose(result.forecasts["posterior"].values[0], expected, rtol=1e-12)
    with_surrogate = tmp_path / "fast.toml"
    with_surrogate.write_text(
        prediction.read_text().replace('name = "future"', 'name = "fast"')
        + '\n[surrogate]\nmodel_file = "missing/surrogate.json"\n'
    )
    with pytest.raises(ConfigError, match="cannot use \\[surrogate\\]"):
        predict(with_surrogate)


class _SeriesFile(FileLinear):
    """``FileLinear`` that also reads whole series from its case (as the deck adapters do)."""

    def read_series(self, case: Path, request: object) -> object:
        from apex.core.simulation import MemberSeries

        values = np.load(case.with_suffix(".npy"))
        times = tuple(f"t{i}" for i in range(N_DATA))
        return MemberSeries(times, ("y",), values[None, :])


def test_the_effective_simulator_forecasts_series_only_for_a_forecasting_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_series`` is the inner simulator's series of the case plus the residuals at
    every vector and time; offered only when the model forecasts and the inner simulator
    reads series."""
    from apex.core.simulation import SeriesRequest
    from apex.simulation.model_error import ModelErrorSimulator

    assert not hasattr(ModelErrorSimulator(_SeriesFile(), Offset()), "run_series")
    _forecasting(monkeypatch)
    assert not hasattr(ModelErrorSimulator(FileLinear(), Offset()), "run_series")
    effective = ModelErrorSimulator(_SeriesFile(), Offset())
    x = np.array([0.3, -0.2, 1.1])
    member = MemberInput(member_id=0, state={"x": x, "nn_params": np.array([2.5])},
                         work_dir=tmp_path)  # fmt: skip
    request = SeriesRequest(("y",), tuple(f"t{i}" for i in range(N_DATA)))
    series = effective.run_series(member, request)
    np.testing.assert_allclose(series.values[0], G @ x + 2.5, rtol=1e-14)
    assert not (tmp_path / "MODEL.npy").exists()  # the case cleaned


def test_the_remote_configuration_keeps_a_forecasting_model_error(tmp_path: Path) -> None:
    """Remote members rebuild the prediction's simulator from it: with a forecasting model
    the model goes too (the members are g + r there as well), otherwise not."""
    from apex.services.prediction import remote_config

    config = composition.load_config(_case(tmp_path, mec=True))
    assert remote_config(config, config.simulator, tmp_path).model_error is None
    kept = remote_config(config, config.simulator, tmp_path, model_error=True)
    assert kept.model_error == config.model_error


def test_a_guest_needs_the_model_error_plugin_allowed() -> None:
    from apex.domain.config import HistoryMatchConfig
    from apex.services.access import Access, UIAccess

    config = HistoryMatchConfig.model_validate({
        "run": {"name": "r", "seed": 1},
        "state": {"variables": [{"name": "x", "size": 2}, {"name": "nn_params", "size": 1}]},
        "prior": {"ensemble_size": 4},
        "observations": {"file": "obs.csv"},
        "simulator": {"name": "linear"},
        "method": {"name": "gies"},
        "model_error": {"name": "pg_lstm"},
        "case": {"kind": "real"},
    })  # fmt: skip
    rules = UIAccess.model_validate({
        "plugins": {"methods": ["gies"], "simulators": ["linear"], "executors": ["serial"],
                    "priors": ["gaussian"]},
    })  # fmt: skip

    assert Access(rules, [], "alice").hidden_plugins(config) == ["model_error_models:pg_lstm"]


def test_a_model_refusing_the_simulator_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model-error model may refuse the simulator it is given (``check_simulator``, e.g. one
    whose outputs it cannot read): composition names the model and its reason, and a model
    without the method accepts every simulator."""
    config = composition.load_config(_case(tmp_path, mec=True))
    observations = read_observation_section(config.observations)
    layout = composition.build_layout(config)
    seen: list[tuple[str, dict[str, object]]] = []

    def refuse(self: Offset, name: str, options: dict[str, object]) -> None:
        seen.append((name, options))
        raise ValueError(f"it cannot read what {name!r} writes")

    monkeypatch.setattr(Offset, "check_simulator", refuse, raising=False)
    with pytest.raises(ConfigError, match="test_offset: it cannot read what 'test_file_linear'"):
        composition.build_model_error(config, layout, observations)
    assert seen == [("test_file_linear", dict(config.simulator.options))]
    monkeypatch.delattr(Offset, "check_simulator")
    assert composition.build_model_error(config, layout, observations) is not None


def test_a_prediction_resolves_its_axis_through_the_effective_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a simulator that reads whole series, a prediction of a forecasting model's run
    resolves its pattern and report times through the effective ``run_series`` (the
    pilot), every member ``G x + theta``."""
    from apex.services.prediction import predict
    from apex.storage.run_results import RunResults

    _forecasting(monkeypatch)
    simulators.register("test_series_file")(_SeriesFile)
    try:
        config = _case(tmp_path, mec=True)
        config.write_text(config.read_text().replace("test_file_linear", "test_series_file"))
        composition.start(config)
        prediction = tmp_path / "series.toml"
        prediction.write_text(
            '[prediction]\nname = "series"\nrun_dir = "runs/run"\nensembles = ["posterior"]\n\n'
            '[prediction.outputs]\nvectors = ["*"]\ntimes = "report"\n'
        )
        result = predict(prediction)
        results = RunResults(tmp_path / "runs" / "run",
                             composition.build_layout(composition.load_config(config)))  # fmt: skip
        state = results.iterate(results.posterior_iteration).state
        expected = G @ np.asarray(state["x"]) + np.asarray(state["nn_params"])[0]
        np.testing.assert_allclose(result.forecasts["posterior"].values[0], expected, rtol=1e-12)
    finally:
        simulators._factories.pop("test_series_file")


def test_a_failed_series_releases_the_case_and_fit_errors_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A residual that fails leaves the case to the inner simulator's ``release`` (its files
    kept, not cleaned), as ``run`` does; a model whose ``fit`` refuses the observations is
    a ConfigError naming it (the review: a traceback before)."""
    from apex.core.simulation import SeriesRequest
    from apex.simulation.model_error import ModelErrorSimulator

    _forecasting(monkeypatch)
    released: list[Path] = []

    class Releasing(_SeriesFile):
        def release(self, case: Path) -> None:
            released.append(case)

    def broken(self: Offset, case: Path, theta: object, keys: object) -> object:
        raise SimulationError("no restart at a forecast step")

    monkeypatch.setattr(Offset, "residuals", broken)
    effective = ModelErrorSimulator(Releasing(), Offset())
    member = MemberInput(member_id=0, state={"x": np.ones(3), "nn_params": np.zeros(1)},
                         work_dir=tmp_path)  # fmt: skip
    with pytest.raises(SimulationError, match="no restart"):
        effective.run_series(member, SeriesRequest(("y",), ("t0",)))
    assert released == [tmp_path / "MODEL"]
    assert (tmp_path / "MODEL.npy").exists()  # kept for diagnosis

    config = composition.load_config(_case(tmp_path / "fit", mec=True))
    observations = read_observation_section(config.observations)

    def refuse(self: Offset, observations: object) -> None:
        raise ValueError("no observation of WOPR:P9")

    monkeypatch.setattr(Offset, "fit", refuse)
    with pytest.raises(ConfigError, match="test_offset: no observation of WOPR:P9"):
        composition.build_model_error(config, composition.build_layout(config), observations)


def _future(tmp_path: Path, name: str, body: str = "", times: int = N_DATA) -> Path:
    listed = ", ".join(f'"t{i}"' for i in range(times))
    path = tmp_path / f"{name}.toml"
    path.write_text(
        f'[prediction]\nname = "{name}"\nrun_dir = "runs/run"\nensembles = ["posterior"]\n'
        f'{body}\n[prediction.outputs]\nvectors = ["y"]\ntimes = [{listed}]\n'
    )
    return path


def test_the_truth_and_residual_vectors_are_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's review: 'truth' is refused without ``truth_is_simulator`` (a model error
    means the data came from another model, which scores against the simulator's truth
    would ignore), and the residuals' own vectors are refused up front."""
    from apex.services.prediction import predict

    _forecasting(monkeypatch)
    composition.start(_case(tmp_path, mec=True))
    truth = (
        _future(tmp_path, "truth").read_text().replace('["posterior"]', '["posterior", "truth"]')
    )
    (tmp_path / "truth.toml").write_text(truth)
    with pytest.raises(ConfigError, match="refuses 'truth' by default"):
        predict(tmp_path / "truth.toml")
    residuals = _future(tmp_path, "mec").read_text().replace('["y"]', '["y", "MEC:y"]')
    (tmp_path / "mec.toml").write_text(residuals)
    with pytest.raises(ConfigError, match="cannot request the model-error residuals"):
        predict(tmp_path / "mec.toml")
    assert not (tmp_path / "runs/run/predictions").exists() or not any(
        (tmp_path / "runs/run/predictions").glob("*/members/*")
    )


def test_the_history_check_holds_for_a_causal_model_and_fails_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A residual that depends on a key's time alone (causal: ``theta (i + 1)`` at time
    ``i``) gives the run's own values when the prediction asks for fewer times, so the
    history check holds; one that depends on what else is asked (``theta`` times the
    number of keys) does not, and the check says so."""
    from apex.core.errors import WorkflowError
    from apex.services.prediction import predict

    def causal(self: Offset, case: Path, theta: NDArray[np.float64],
               keys: Sequence[DataKey]) -> NDArray[np.float64]:  # fmt: skip
        return np.array([theta[0] * (int(k.time[1:]) + 1) for k in keys])

    def non_causal(self: Offset, case: Path, theta: NDArray[np.float64],
                   keys: Sequence[DataKey]) -> NDArray[np.float64]:  # fmt: skip
        return np.full(len(keys), theta[0] * len(keys))

    _forecasting(monkeypatch)
    for name, residuals, holds in (("causal", causal, True), ("other", non_causal, False)):
        monkeypatch.setattr(Offset, "residuals", residuals)
        folder = tmp_path / name
        composition.start(_case(folder, mec=True))
        prediction = _future(folder, "fewer", times=4)
        if holds:
            assert predict(prediction).forecasts["posterior"].values.shape[1] == 4
        else:
            with pytest.raises(WorkflowError, match="does not reproduce the history match"):
                predict(prediction)


def test_a_model_error_predictions_fingerprint_holds_its_files(tmp_path: Path) -> None:
    """The coder's review: the model's files and the observations enter a model-error
    prediction's fingerprint (a changed one must not resume from the old model's members);
    a plain prediction's fingerprint and configuration are unchanged."""
    from apex.domain.prediction import PredictionConfig
    from apex.services.prediction import _fingerprint, _model_error_files
    from apex.storage.run_directory import RunDirectory

    config = composition.load_config(_case(tmp_path, mec=True))
    composition.start(_case(tmp_path, mec=True))
    run = RunDirectory(tmp_path / "runs" / "run")
    prediction = PredictionConfig.model_validate({
        "prediction": {"name": "p", "run_dir": str(run.path), "outputs": {"vectors": ["y"]}},
    })  # fmt: skip
    assert "truth_is_simulator" not in prediction.model_dump_json()
    files = _model_error_files(config)
    assert set(files) == {"observations.file"}
    plain = _fingerprint(prediction, run, config.simulator)
    before = _fingerprint(prediction, run, config.simulator, "", files)
    (tmp_path / "observations.csv").write_text(
        (tmp_path / "observations.csv").read_text().replace("t0,", "t0,1")
    )
    after = _fingerprint(prediction, run, config.simulator, "", files)
    assert len({plain, before, after}) == 3
    assert _fingerprint(prediction, run, config.simulator) == plain
