"""History matching with a model-error model (MEC) on a biased linear case.

The observations come from ``G x + b`` with an offset ``b`` the simulator ``G x`` lacks.
A model-error model ``r = theta`` estimated with ``x`` should take up the offset: the
effective data mismatch falls below the run without it, and theta approaches ``b``.
"""

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.core.options import InputFile
from apex.core.plugins import model_error_models, simulators
from apex.core.simulation import SEED_ENTROPY, SEED_SPAWN_KEY, MemberInput
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


class Halves(Offset):
    """``r = theta_0`` on the first half of the times, ``theta_1`` on the rest; it says so.

    A model with a parameter mask: each parameter acts on its half of
    the data only, exactly.
    """

    n_params = 2
    masks_parameters = True

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return np.array([theta[0] if int(k.time[1:]) < N_DATA // 2 else theta[1] for k in keys])

    def parameter_mask(self, keys: Sequence[DataKey]) -> NDArray[np.bool_]:
        early = np.array([int(k.time[1:]) < N_DATA // 2 for k in keys])
        return np.vstack([early, ~early])


def _halves_case(directory: Path, localization: str | None) -> Path:
    config = _case(directory, mec=True, size=2)
    text = config.read_text().replace('name = "test_offset"', 'name = "test_halves"')
    if localization is not None:
        text = text.replace(
            'strategy = { name = "rlmmac", options = { energy = 1.0 } }',
            'strategy = { name = "rlmmac", options = { energy = 1.0 } }\n' + localization,
        )
    config.write_text(text)
    return config


def test_a_models_parameter_mask_reaches_the_runs_taper_and_resumes_bit_identically(
    tmp_path: Path,
) -> None:
    """Through a whole run: the taper of nn_params is the mask (with
    nn_params unlocalized), a run without a localization is refused before anything runs,
    and a resumed run equals an uninterrupted one bit for bit."""
    model_error_models.register("test_halves")(Halves)
    try:
        with pytest.raises(ConfigError, match="it has none"):
            composition.start(_halves_case(tmp_path / "none", None))
        localization = ('localization = { name = "adaptive_prior", options = { unlocalized = '
                        '["nn_params"] } }')  # fmt: skip
        whole = composition.start(_halves_case(tmp_path / "whole", localization), max_steps=4)
        checkpoints = tmp_path / "whole" / "runs" / "run" / "checkpoints"
        state = np.load(sorted(checkpoints.glob("algorithm_state_*.npz"))[-1])
        matrix = state["method_state:taper"]
        early = np.arange(N_DATA) < N_DATA // 2
        np.testing.assert_array_equal(matrix[N_STATE:], np.vstack([early, ~early]).astype(float))
        parts = _halves_case(tmp_path / "parts", localization)
        composition.start(parts, max_steps=2)
        resumed = composition.resume(tmp_path / "parts" / "runs" / "run", max_steps=2)
        np.testing.assert_array_equal(resumed.posterior, whole.posterior)
    finally:
        model_error_models._factories.pop("test_halves")


@pytest.mark.parametrize(
    ("method", "match"),
    [
        ('name = "chop"\nmax_iterations = 3\n'
         'strategy = { name = "rlmmac", options = { energy = 0.99 } }\n'
         'options = { inner = "gies", components = [{ name = "length_scale", mode = "data" }] }',
         "ChopTaper cannot apply one"),
        ('name = "es"\nstrategy = { name = "adaptive_rows" }', "it has none"),
    ],
)  # fmt: skip
def test_methods_that_cannot_apply_a_parameter_mask_are_refused_by_name(
    tmp_path: Path, method: str, match: str
) -> None:
    """CHOP tapers per member (its own ChopTaper); adaptive_rows has no gain taper."""
    model_error_models.register("test_halves")(Halves)
    try:
        config = _halves_case(tmp_path, None)
        text = config.read_text()
        start = text.index("[method]")
        config.write_text(text[:start] + "[method]\n" + method + "\n")
        with pytest.raises(ConfigError, match=match):
            composition.build_run(composition.load_config(config))
    finally:
        model_error_models._factories.pop("test_halves")


def test_a_parameter_mask_with_local_analysis_and_held_out_data(tmp_path: Path) -> None:
    """local_rlmmac selects its data by adaptive_prior's taper, the mask applied; with cv
    data held out, the mask is evaluated on the assimilated keys only."""
    model_error_models.register("test_halves")(Halves)
    try:
        localization = ('localization = { name = "adaptive_prior", options = { unlocalized = '
                        '["nn_params"] } }')  # fmt: skip
        local = _halves_case(tmp_path / "local", localization)
        local.write_text(local.read_text().replace('name = "rlmmac"', 'name = "local_rlmmac"'))
        composition.start(local, max_steps=2)
        held = _halves_case(tmp_path / "held", localization)
        held.write_text(
            held.read_text() + '\n[validation]\ncv = { datatypes = ["y"], times = ["t6", "t7"] }\n'
        )
        composition.start(held, max_steps=2)
        checkpoints = tmp_path / "held" / "runs" / "run" / "checkpoints"
        state = np.load(sorted(checkpoints.glob("algorithm_state_*.npz"))[-1])
        matrix = state["method_state:taper"]
        assert matrix.shape == (N_STATE + 2, N_DATA - 2)
        early = np.arange(N_DATA - 2) < N_DATA // 2
        np.testing.assert_array_equal(matrix[N_STATE:], np.vstack([early, ~early]).astype(float))
    finally:
        model_error_models._factories.pop("test_halves")


# ------------------------------------------- extrapolated parameters

HISTORY = 5  # the observations cover t0..t4; t5..t7 are past the history


class Steps(Offset):
    """``r = theta_i`` at history time ``i``; past it, ``theta_last + level + draw``.

    ``level`` is fitted from the run's ensembles (the posterior's mean last block minus the
    prior's); the draw is ``N(0, 1)`` from the prediction's seed and the member's key.
    """

    n_params = HISTORY
    forecasts = True
    extrapolates = True
    forecast_options = frozenset({"shift", "table_file"})

    def __init__(
        self,
        state_variable: str = "nn_params",
        forecast_state_file: InputFile | None = None,
        shift: float = 0.0,
        table_file: InputFile | None = None,
    ) -> None:
        super().__init__(state_variable)
        self.shift = float(shift)
        self.level: float | None = None
        self.seed: np.random.SeedSequence | None = None
        if forecast_state_file is not None:
            with np.load(forecast_state_file, allow_pickle=False) as data:
                self.level = float(data["level"])
                self.seed = np.random.SeedSequence(
                    int(str(data[SEED_ENTROPY])),
                    spawn_key=tuple(int(k) for k in data[SEED_SPAWN_KEY]),
                )

    def forecast_state(
        self, prior: NDArray[np.floating], posterior: NDArray[np.floating],
        keys: Sequence[DataKey],
    ) -> dict[str, NDArray[np.generic]]:  # fmt: skip
        # The assimilated keys are the history's (t0..t4) and the members are paired.
        assert [k.time for k in keys] == [f"t{i}" for i in range(HISTORY)]
        assert prior.shape == posterior.shape
        return {"level": np.array(float(posterior[-1].mean() - prior[-1].mean())),
                "members": np.array(prior.shape[1])}  # fmt: skip

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        if any(int(k.time[1:]) >= HISTORY for k in keys):
            raise SimulationError("past the history: only member_residuals answers there")
        return np.array([theta[int(k.time[1:])] for k in keys])

    def member_residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey],
        member: Sequence[int],
    ) -> NDArray[np.float64]:  # fmt: skip
        assert self.seed is not None
        assert self.level is not None
        out = []
        for key in keys:
            i = int(key.time[1:])
            if i < HISTORY:
                out.append(theta[i])
                continue
            stream = np.random.SeedSequence(
                self.seed.entropy, spawn_key=(*self.seed.spawn_key, *member, i)
            )
            draw = np.random.default_rng(stream).standard_normal()
            out.append(theta[-1] + self.level + self.shift + draw)
        return np.array(out)


def _steps_case(folder: Path) -> Path:
    config = _case(folder, mec=True, size=HISTORY)
    observations = folder / "observations.csv"
    observations.write_text("\n".join(observations.read_text().splitlines()[: 1 + HISTORY]) + "\n")
    config.write_text(config.read_text().replace('name = "test_offset"', 'name = "test_steps"'))
    return config


@pytest.fixture
def steps() -> Iterator[None]:
    model_error_models.register("test_steps")(Steps)
    try:
        yield
    finally:
        model_error_models._factories.pop("test_steps")


def _expected(
    folder: Path, ensemble: str, iteration: int, seed: np.random.SeedSequence, level: float
) -> NDArray[np.float64]:
    from apex.storage.run_results import RunResults
    from apex.workflows.prediction import stream_key

    config = composition.load_config(folder / "config.toml")
    results = RunResults(folder / "runs" / "run", composition.build_layout(config))
    iterate = results.iterate(iteration)
    x, theta = np.asarray(iterate.state["x"]), np.asarray(iterate.state["nn_params"])
    columns = []
    for column, member in enumerate(iterate.member_ids):
        r = list(theta[:, column])
        for i in range(HISTORY, N_DATA):
            stream = np.random.SeedSequence(
                seed.entropy, spawn_key=(*seed.spawn_key, *stream_key(ensemble, int(member)), i)
            )
            r.append(theta[-1, column] + level + np.random.default_rng(stream).standard_normal())
        columns.append(G @ x[:, column] + np.array(r))
    return np.stack(columns, axis=-1)


def test_a_prediction_extrapolates_each_members_parameters_from_its_own_key(
    tmp_path: Path, steps: None
) -> None:
    """The model's state is fitted once from the run's prior and
    posterior and written with the seed to a content-named file the model is built from;
    each member draws from its own key (its ensemble's and its id), so member 3 of the
    prior and of the posterior differ; the history check holds (the stored blocks); a
    resume of a dropped member and a second prediction with the same seed reproduce the
    draws bit for bit; the seed is recorded."""
    from apex.services.prediction import predict
    from apex.storage.run_results import RunResults

    composition.start(_steps_case(tmp_path))
    prediction = _future(tmp_path, "ahead", 'seed = 42\nensembles = ["posterior", "prior"]')
    prediction.write_text(prediction.read_text().replace('ensembles = ["posterior"]\n', "", 1))
    result = predict(prediction)
    folder = tmp_path / "runs/run/predictions/ahead"
    recorded = json.loads((folder / "provenance.json").read_text())["model_error"]
    assert recorded["seed"] == {"entropy": "42", "spawn_key": [], "given": True}
    state_file = folder / recorded["forecast_state"]["file"]
    assert state_file.name.startswith("forecast_state-")
    with np.load(state_file) as data:
        level = float(data["level"])
    config = composition.load_config(tmp_path / "config.toml")
    results = RunResults(tmp_path / "runs" / "run", composition.build_layout(config))
    seed = np.random.SeedSequence(42)
    posterior = _expected(tmp_path, "posterior", results.posterior_iteration, seed, level)
    prior = _expected(tmp_path, "prior", 0, seed, level)
    np.testing.assert_allclose(result.forecasts["posterior"].values[0], posterior, rtol=1e-12)
    np.testing.assert_allclose(result.forecasts["prior"].values[0], prior, rtol=1e-12)
    # A resumed member draws as before.
    saved = sorted((folder / "members").glob("posterior_*.npz"))
    saved[3].unlink()
    (folder / "forecast_posterior.npz").unlink()
    again = predict(prediction)
    np.testing.assert_array_equal(
        again.forecasts["posterior"].values, result.forecasts["posterior"].values
    )
    # Another prediction with the same seed and ensembles: the same file and the same draws.
    other = _future(tmp_path, "twin", 'seed = 42\nensembles = ["posterior", "prior"]')
    other.write_text(other.read_text().replace('ensembles = ["posterior"]\n', "", 1))
    twin = predict(other)
    np.testing.assert_array_equal(
        twin.forecasts["posterior"].values, result.forecasts["posterior"].values
    )
    twin_record = json.loads((tmp_path / "runs/run/predictions/twin/provenance.json").read_text())[
        "model_error"
    ]
    assert twin_record["forecast_state"] == recorded["forecast_state"]


def test_the_default_seed_follows_the_run_and_the_name_and_refusals(
    tmp_path: Path, steps: None
) -> None:
    """Without ``seed`` the run's seed and the prediction's name seed it (recorded); the
    truth is refused for an extrapolating model, and a seed where nothing draws."""
    from apex.services.prediction import predict

    composition.start(_steps_case(tmp_path))
    predict(_future(tmp_path, "plain"))
    recorded = json.loads((tmp_path / "runs/run/predictions/plain/provenance.json").read_text())[
        "model_error"
    ]["seed"]
    assert recorded["entropy"] == "11"
    assert recorded["given"] is False
    assert len(recorded["spawn_key"]) == 1
    truth = _future(tmp_path, "truthful", "truth_is_simulator = true")
    truth.write_text(truth.read_text().replace('["posterior"]', '["posterior", "truth"]'))
    with pytest.raises(ConfigError, match="cannot forecast 'truth'"):
        predict(truth)


def test_a_seed_is_refused_where_nothing_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.services.prediction import predict

    _forecasting(monkeypatch)
    composition.start(_case(tmp_path, mec=True))
    with pytest.raises(ConfigError, match="nothing in this prediction draws"):
        predict(_future(tmp_path, "seeded", "seed = 3"))


def test_members_without_a_key_get_the_history_residuals(tmp_path: Path) -> None:
    """History matching gives no ``stream_key``: the model's own residuals, as before."""
    from apex.simulation.model_error import ModelErrorSimulator

    effective = ModelErrorSimulator(FileLinear(), Steps())
    x = np.array([0.3, -0.2, 1.1])
    member = MemberInput(member_id=0, state={"x": x, "nn_params": np.arange(5.0)},
                         work_dir=tmp_path)  # fmt: skip
    keys = [DataKey(f"t{i}", "y") for i in range(HISTORY)]
    np.testing.assert_allclose(effective.run(member, keys), G[:HISTORY] @ x + np.arange(5.0))
    with pytest.raises(SimulationError, match="past the history"):
        effective.run(member, [DataKey("t6", "y")])


def test_dropped_members_pair_by_id_and_the_default_seed_reproduces(
    tmp_path: Path, steps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review: a member that failed in the run is dropped from its posterior; the fit
    pairs the rest by id (members recorded), and each keeps its original id's draws. Two
    predictions of one name (other folders) draw the same with the derived seed."""
    from apex.services.prediction import predict
    from apex.storage.run_results import RunResults

    monkeypatch.setattr(FileLinear, "fail_members", frozenset({2}))
    composition.start(_steps_case(tmp_path))
    monkeypatch.setattr(FileLinear, "fail_members", frozenset())
    config = composition.load_config(tmp_path / "config.toml")
    results = RunResults(tmp_path / "runs" / "run", composition.build_layout(config))
    assert 2 not in results.iterate(results.posterior_iteration).member_ids
    one = predict(_future(tmp_path, "same", 'output_dir = "out_a"'))
    folder = tmp_path / "out_a"
    recorded = json.loads((folder / "provenance.json").read_text())["model_error"]
    assert recorded["forecast_state"]["members"] == 59
    with np.load(folder / recorded["forecast_state"]["file"]) as data:
        assert int(data["members"]) == 59
        level = float(data["level"])
    seed = np.random.SeedSequence(
        int(recorded["seed"]["entropy"]), spawn_key=tuple(recorded["seed"]["spawn_key"])
    )
    expected = _expected(tmp_path, "posterior", results.posterior_iteration, seed, level)
    np.testing.assert_allclose(one.forecasts["posterior"].values[0], expected, rtol=1e-12)
    two = predict(_future(tmp_path, "same", 'output_dir = "out_b"'))
    np.testing.assert_array_equal(
        two.forecasts["posterior"].values, one.forecasts["posterior"].values
    )


def test_a_prediction_sets_only_the_models_forecast_options(tmp_path: Path, steps: None) -> None:
    """``[model_error] options`` in a prediction: a forecast option changes the forecast
    past the history only (the history check holds); any other option is refused."""
    from apex.services.prediction import predict

    composition.start(_steps_case(tmp_path))
    base = predict(_future(tmp_path, "base", "seed = 5"))
    shifted_config = _future(tmp_path, "shifted", "seed = 5")
    shifted_config.write_text(
        shifted_config.read_text() + "\n[model_error.options]\nshift = 10.0\n"
    )
    shifted = predict(shifted_config)
    difference = shifted.forecasts["posterior"].values - base.forecasts["posterior"].values
    np.testing.assert_allclose(difference[0, :HISTORY], 0.0, atol=1e-12)
    np.testing.assert_allclose(difference[0, HISTORY:], 10.0, rtol=1e-12)
    refused = _future(tmp_path, "refused", "seed = 5")
    refused.write_text(refused.read_text() + '\n[model_error.options]\nstate_variable = "x"\n')
    with pytest.raises(ConfigError, match="only the model's forecast options"):
        predict(refused)


def test_replacing_an_unfinished_prediction_keeps_only_the_forecast_state_in_use(
    tmp_path: Path,
) -> None:
    """Replacing a prediction that had no finished member keeps the forecast state the new
    configuration wrote or reused (named by its inputs) and removes the replaced
    configurations'; a configuration without one keeps none."""
    from apex.storage.prediction_directory import PredictionDirectory

    directory = PredictionDirectory(tmp_path / "p")
    directory.open(config_text="{}", fingerprint="one")
    old = directory.path / "forecast_state-0123456789abcdef.npz"
    new = directory.path / "forecast_state-fedcba9876543210.npz"
    old.write_bytes(b"old state")
    new.write_bytes(b"state")
    (directory.path / "forecast_state-0000000000000000.npz.tmp").write_bytes(b"crashed")
    (directory.path / "forecast_posterior.npz").write_bytes(b"old")
    directory.open(config_text="{}", fingerprint="two", forecast_state=new.name)
    assert new.read_bytes() == b"state"
    assert sorted(p.name for p in directory.path.glob("forecast_*")) == [new.name]
    directory.open(config_text="{}", fingerprint="three")
    assert not list(directory.path.glob("forecast_*"))


def test_forecast_options_where_nothing_extrapolates_are_refused_and_policy_sees_them(
    tmp_path: Path, steps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-review: options that would change nothing are refused, as a needless seed;
    the tool policy judges a prediction's model-error options by the run's model."""
    from apex.domain.prediction import PredictionConfig
    from apex.services import policy
    from apex.services.prediction import load_prediction_config, predict

    composition.start(_steps_case(tmp_path))
    config = _future(tmp_path, "inert")
    config.write_text(config.read_text() + "\n[model_error.options]\nshift = 1.0\n")
    named = policy.named(load_prediction_config(config))
    assert not [key for _, key, _ in named if key.startswith("model_error_models")]
    sections = policy._plugin_sections(load_prediction_config(config))
    assert ("model_error_models", load_prediction_config(config).model_error) in sections
    assert isinstance(load_prediction_config(config), PredictionConfig)
    monkeypatch.setattr(Steps, "extrapolates", False)
    with pytest.raises(ConfigError, match="would change nothing"):
        predict(config)


def test_validation_refuses_what_needs_no_fit(tmp_path: Path, steps: None) -> None:
    """The coder's review: the truth, a needless seed or options, a non-forecast option are
    refused by ``check_prediction`` (apex validate, the tools, submission), before a job."""
    from apex.services.prediction import check_prediction, load_prediction_config

    composition.start(_steps_case(tmp_path))
    check_prediction(load_prediction_config(_future(tmp_path, "fine", "seed = 1")))
    truth = _future(tmp_path, "truthful", "truth_is_simulator = true")
    truth.write_text(truth.read_text().replace('["posterior"]', '["posterior", "truth"]'))
    with pytest.raises(ConfigError, match="cannot forecast 'truth'"):
        check_prediction(load_prediction_config(truth))
    other = _future(tmp_path, "other")
    other.write_text(other.read_text() + '\n[model_error.options]\nstate_variable = "x"\n')
    with pytest.raises(ConfigError, match="only the model's forecast options"):
        check_prediction(load_prediction_config(other))


def test_a_needless_seed_is_refused_by_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apex.services.prediction import check_prediction, load_prediction_config

    _forecasting(monkeypatch)
    composition.start(_case(tmp_path, mec=True))
    with pytest.raises(ConfigError, match="nothing in this prediction draws"):
        check_prediction(load_prediction_config(_future(tmp_path, "seeded", "seed = 3")))


def test_the_policy_judges_a_path_in_the_forecast_options(tmp_path: Path, steps: None) -> None:
    """The coder's review: a forecast option naming a file is reported to the tool policy."""
    from apex.services import policy
    from apex.services.prediction import load_prediction_config

    composition.start(_steps_case(tmp_path))
    config = _future(tmp_path, "tabled")
    config.write_text(
        config.read_text() + '\n[model_error.options]\ntable_file = "/elsewhere/t.npz"\n'
    )
    named = policy.named(load_prediction_config(config))
    assert ("input", "model_error_models.test_steps: table_file", "/elsewhere/t.npz") in named


def test_each_input_of_the_fit_names_its_own_state_file(
    tmp_path: Path, steps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coder's review: a changed option, seed or posterior iterate gives a new file; the
    same inputs reuse it. A finished folder of another configuration is refused before a
    fit lands in it."""
    from apex.services.prediction import predict
    from apex.storage.run_results import RunResults

    composition.start(_steps_case(tmp_path))

    def state_file(name: str, body: str) -> str:
        predict(_future(tmp_path, name, f'output_dir = "out"\n{body}'))
        record = json.loads((tmp_path / "out" / "provenance.json").read_text())
        return str(record["model_error"]["forecast_state"]["file"])

    first = state_file("p", "seed = 1")
    assert state_file("p", "seed = 1") == first  # the same inputs: reused (a resume)
    with pytest.raises(ConfigError, match="another configuration and finished members"):
        predict(_future(tmp_path, "p", 'output_dir = "out"\nseed = 2'))
    assert sorted(p.name for p in (tmp_path / "out").glob("forecast_state-*")) == [first]
    names = {first}
    for index, body in enumerate(("seed = 2", "seed = 1\n[model_error.options]\nshift = 1.0")):
        folder = f"out{index}"
        predict(_future(tmp_path, "p", f'output_dir = "{folder}"\n{body}'))
        record = json.loads((tmp_path / folder / "provenance.json").read_text())
        names.add(record["model_error"]["forecast_state"]["file"])
    monkeypatch.setattr(RunResults, "posterior_iteration", property(lambda self: 1))
    predict(_future(tmp_path, "p", 'output_dir = "out_iterate"\nseed = 1'))
    record = json.loads((tmp_path / "out_iterate" / "provenance.json").read_text())
    names.add(record["model_error"]["forecast_state"]["file"])
    assert len(names) == 4


def test_a_refused_extrapolating_prediction_leaves_no_folder(tmp_path: Path, steps: None) -> None:
    """The coder's review: a prediction refused after its forecast state was written (here
    an unknown score) removes the new folder, the state file included."""
    from apex.services.prediction import predict

    composition.start(_steps_case(tmp_path))
    config = _future(tmp_path, "refused", 'output_dir = "fresh"\nscores = ["no_such_score"]')
    with pytest.raises(ConfigError, match="no_such_score"):
        predict(config)
    assert not (tmp_path / "fresh").exists()


def test_validation_accepts_a_run_to_come_and_a_plain_run(tmp_path: Path) -> None:
    """The review: a study's prediction of a run not yet written is not checked; a run
    without a model-error model passes unchanged (a seed there is still refused)."""
    from apex.services.prediction import check_prediction, load_prediction_config

    check_prediction(load_prediction_config(_future(tmp_path, "later", "seed = 1")))
    composition.start(_case(tmp_path, mec=False))
    check_prediction(load_prediction_config(_future(tmp_path, "plain")))
    with pytest.raises(ConfigError, match="nothing in this prediction draws"):
        check_prediction(load_prediction_config(_future(tmp_path, "seeded", "seed = 1")))


def test_a_bad_forecast_option_is_a_configuration_error(tmp_path: Path, steps: None) -> None:
    """The coder's re-review: the model's constructor refusing a client's forecast option is
    a ConfigError: validation reports it, and a prediction leaves no folder."""
    from apex.services.prediction import check_prediction, load_prediction_config, predict

    composition.start(_steps_case(tmp_path))
    config = _future(tmp_path, "bad", 'output_dir = "fresh"')
    config.write_text(config.read_text() + '\n[model_error.options]\nshift = "a lot"\n')
    with pytest.raises(ConfigError, match="test_steps"):
        check_prediction(load_prediction_config(config))
    with pytest.raises(ConfigError, match="test_steps"):
        predict(config)
    assert not (tmp_path / "fresh").exists()


def _state_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.glob("forecast_state-*"))


def _used_state(folder: Path) -> str:
    record = json.loads((folder / "provenance.json").read_text())
    return str(record["model_error"]["forecast_state"]["file"])


def test_a_replacement_through_predict_keeps_the_state_it_uses(
    tmp_path: Path, steps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction that failed after opening its folder, before any member, is replaced
    by another seed: the replacement keeps the state file it has just fitted (the service
    names it to the folder; dropping it would leave the rebuilt model without its file)."""
    from apex.services import prediction as service

    composition.start(_steps_case(tmp_path))
    real = composition.configure_logging
    failures = iter([RuntimeError("stopped after opening")])

    def configure_logging(folder: Path) -> None:
        error = next(failures, None)
        if error is not None:
            raise error
        real(folder)

    monkeypatch.setattr(composition, "configure_logging", configure_logging)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="stopped after opening"):
        service.predict(_future(tmp_path, "p", 'output_dir = "out"\nseed = 1'))
    first = _state_files(out)
    assert len(first) == 1
    assert (out / "state.json").exists()
    assert not list((out / "members").glob("*.npz"))
    service.predict(_future(tmp_path, "p", 'output_dir = "out"\nseed = 2'))
    assert _state_files(out) == [_used_state(out)]
    assert _state_files(out) != first


def test_a_folder_made_by_another_start_keeps_its_state_when_this_one_is_refused(
    tmp_path: Path, steps: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race: another start makes the folder (with its state file) after this one began
    but before its lock. This start's refusal must not remove that folder as its own, as it
    did when it judged the folder new by an exists() taken before the lock."""
    from apex.services import prediction as service
    from apex.storage import locks

    composition.start(_steps_case(tmp_path))
    out = tmp_path / "out"

    @contextmanager
    def other_start_first(directory: Path, what: str = "run") -> Iterator[bool]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "forecast_state-0123456789abcdef.npz").write_bytes(b"theirs")
        with locks.exclusive(directory, what) as made:
            yield made

    def refused(*args: object, **kwargs: object) -> None:
        raise ConfigError("refused")

    monkeypatch.setattr(service, "exclusive", other_start_first)
    monkeypatch.setattr(service, "_predict", refused)
    with pytest.raises(ConfigError, match="refused"):
        service.predict(_future(tmp_path, "p", 'output_dir = "out"'))
    assert (out / "forecast_state-0123456789abcdef.npz").read_bytes() == b"theirs"


def test_a_start_refused_by_another_predictions_folder_leaves_no_state_there(
    tmp_path: Path, steps: None
) -> None:
    """A configuration refused by a folder holding another prediction (here: other files)
    removes the state file it fitted there; the folder is otherwise untouched."""
    from apex.services.prediction import predict

    composition.start(_steps_case(tmp_path))
    out = tmp_path / "out"
    predict(_future(tmp_path, "p", 'output_dir = "out"\nseed = 1'))
    kept = _state_files(out)
    (out / "notes.txt").write_text("mine\n")
    for member in (out / "members").glob("*.npz"):
        member.unlink()
    with pytest.raises(ConfigError, match="other files"):
        predict(_future(tmp_path, "p", 'output_dir = "out"\nseed = 2'))
    assert _state_files(out) == kept
    assert (out / "notes.txt").exists()


def test_a_folder_without_a_prediction_keeps_only_the_state_in_use(tmp_path: Path) -> None:
    """A start that ended between its fit and opening the folder left a state file in a
    folder with no prediction yet: the next start keeps only its own."""
    from apex.storage.prediction_directory import PredictionDirectory

    folder = tmp_path / "p"
    folder.mkdir()
    (folder / "forecast_state-0123456789abcdef.npz").write_bytes(b"old")
    (folder / "forecast_state-fedcba9876543210.npz").write_bytes(b"mine")
    (folder / "notes.txt").write_text("kept\n")
    PredictionDirectory(folder).open(
        config_text="{}", fingerprint="a", forecast_state="forecast_state-fedcba9876543210.npz"
    )
    assert _state_files(folder) == ["forecast_state-fedcba9876543210.npz"]
    assert (folder / "notes.txt").exists()
