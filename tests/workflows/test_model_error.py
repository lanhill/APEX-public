"""History matching with a model-error model (MEC) on a biased linear case.

The observations come from ``G x + b`` with an offset ``b`` the simulator ``G x`` lacks.
A model-error model ``r = theta`` estimated with ``x`` should take up the offset: the
effective data mismatch falls below the run without it, and theta approaches ``b``.
"""

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
    with pytest.raises(ConfigError, match="model-error model"):
        predict(prediction)


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
