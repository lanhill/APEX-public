"""Model error from the member's state: ``StateInputModel``.

A model whose residuals come from the member's state variables, not from a simulation,
wraps any simulator (here an analytic one that writes no files): ``g`` from it, ``r`` from
the state. Its inputs' standardisation is fitted once from the prior by the workflow, written
to the run's ``model_error_prior_fit.npz`` and read by every rebuild of the model; a resume
reads it and stays bit-identical.

The case: observations ``G x + c (mean(x) - mu) / sd`` with a term the simulator ``G x``
lacks; the model ``r = theta_0 + theta_1 z``, ``z`` the member's ``mean(x)`` standardised by
the prior's mean and spread of it, can take it up.
"""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, WorkflowError
from apex.core.plugins import model_error_models, simulators
from apex.core.simulation import PRIOR_FIT_FILE, MemberInput, StateInputModel
from apex.domain.observations import DataKey, ObservationSet
from apex.services import composition
from apex.simulation.model_error import RESIDUAL_PREFIX, ModelErrorSimulator

N_STATE, N_DATA, SLOPE = 3, 8, 2.0
RNG = np.random.default_rng(17)
G = RNG.standard_normal((N_DATA, N_STATE))
TRUTH = np.array([0.6, -0.9, 1.2])
VARIANCE = 0.02


class Analytic:
    """``G x``, no files: not a ``CaseFileSimulator`` and no block states."""

    needs_work_dir = False

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        assert set(member.state) == {"x"}  # theta stays out of the simulator
        values = G @ np.asarray(member.state["x"])
        return np.array([values[int(k.time[1:])] for k in keys])


class MeanInput:
    """``r = theta_0 + theta_1 z``, ``z = (mean(x) - mu) / sd`` with mu, sd from the prior."""

    n_params = 2
    fits = 0  # how often fit_prior ran, over every instance

    def __init__(self, state_variable: str = "nn_params") -> None:
        self.state_variable = state_variable
        self.prior_fit_file: Path | None = None
        self._fit: dict[str, NDArray[np.float64]] | None = None

    def fit(self, observations: ObservationSet) -> None:
        pass

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        return [DataKey(k.time, RESIDUAL_PREFIX + k.datatype) for k in keys]

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        raise AssertionError("a state-input model is never asked for a case's residuals")

    def fit_prior(self, prior: Mapping[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        assert set(prior) == {"x"}
        MeanInput.fits += 1
        means = prior["x"].mean(axis=0)
        return {"mu": np.array([means.mean()]), "sd": np.array([means.std()])}

    def use_prior_fit(self, arrays: Mapping[str, NDArray[np.float64]]) -> None:
        self._fit = {k: np.asarray(v) for k, v in arrays.items()}

    def input_residuals(
        self,
        state: Mapping[str, NDArray[np.float64]],
        theta: NDArray[np.float64],
        keys: Sequence[DataKey],
    ) -> NDArray[np.float64]:
        # The framework installs the fit (the workflow, or the effective simulator from
        # the run's file): the model never reads the file itself.
        assert self._fit is not None
        z = (np.mean(state["x"]) - self._fit["mu"][0]) / self._fit["sd"][0]
        return np.full(len(keys), theta[0] + theta[1] * z)


@pytest.fixture(autouse=True)
def plugins() -> Iterator[None]:
    simulators.register("test_analytic")(Analytic)
    model_error_models.register("test_mean_input")(MeanInput)
    MeanInput.fits = 0
    try:
        yield
    finally:
        simulators._factories.pop("test_analytic")
        model_error_models._factories.pop("test_mean_input")


def _case(directory: Path, *, mec: bool = True, execution: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(5).standard_normal(N_DATA) * np.sqrt(VARIANCE)
    # The prior of x is N(0, 1) per element, so mean(x) has mean 0 and spread 1/sqrt(3).
    z_true = TRUTH.mean() / (1 / np.sqrt(N_STATE))
    observed = G @ TRUTH + SLOPE * z_true + noise
    rows = ["time,datatype,value,variance"] + [
        f"t{i},y,{float(observed[i])!r},{VARIANCE}" for i in range(N_DATA)
    ]
    (directory / "observations.csv").write_text("\n".join(rows) + "\n")
    np.savez(directory / "truth.npz", x=TRUTH)
    theta = """
[[state.variables]]
name = "nn_params"
size = 2

[prior.variables.nn_params]
name = "gaussian"
options = { mean = 0.0, std = 2.0 }

[model_error]
name = "test_mean_input"
"""
    config = directory / "config.toml"
    config.write_text(
        f"""
[run]
name = "run"
seed = 23

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
name = "test_analytic"
{execution}
[method]
name = "gies"
strategy = {{ name = "rlmmac", options = {{ energy = 1.0 }} }}
max_iterations = 6
"""
    )
    return config


def test_the_stub_is_a_state_input_model() -> None:
    assert isinstance(MeanInput(), StateInputModel)


def test_a_state_input_model_wraps_a_simulator_without_files(tmp_path: Path) -> None:
    """g from the analytic simulator, r from the state; the MEC keys give r alone."""
    model = MeanInput()
    np.savez(tmp_path / PRIOR_FIT_FILE, mu=np.array([0.1]), sd=np.array([0.5]))
    model.prior_fit_file = tmp_path / PRIOR_FIT_FILE
    effective = ModelErrorSimulator(Analytic(), model)  # installs the file on first use
    x = np.array([0.3, 0.4, 0.5])
    member = MemberInput(member_id=0, state={"x": x, "nn_params": np.array([1.0, 2.0])})
    keys = [DataKey("t0", "y"), DataKey("t3", "y"), DataKey("t3", RESIDUAL_PREFIX + "y")]
    r = 1.0 + 2.0 * (x.mean() - 0.1) / 0.5
    np.testing.assert_allclose(effective.run(member, keys), [G[0] @ x + r, G[3] @ x + r, r])
    assert effective.needs_work_dir is False
    assert effective.state_names_read() is None


def test_the_model_takes_up_what_the_simulator_lacks(tmp_path: Path) -> None:
    with_mec = composition.start(_case(tmp_path / "mec"))
    without = composition.start(_case(tmp_path / "plain", mec=False))

    final = [r.data_mismatch_mean for r in with_mec.history if r.accepted][-1]
    plain = [r.data_mismatch_mean for r in without.history if r.accepted][-1]
    assert final < 0.3 * plain
    # The fit: the drawn prior's mean and spread of mean(x), written once (every member
    # simulated, so iter_0 holds the whole drawn prior).
    assert MeanInput.fits == 1
    run = tmp_path / "mec" / "runs" / "run"
    prior = np.load(run / "checkpoints" / "iter_0.npz")["state"][:N_STATE]
    assert prior.shape[1] == 60
    with np.load(run / PRIOR_FIT_FILE) as fit:
        assert fit["mu"][0] == pytest.approx(prior.mean(axis=0).mean())
        assert fit["sd"][0] == pytest.approx(prior.mean(axis=0).std())
    # The saved residuals are theta_0 + theta_1 z of each member.
    saved = np.load(run / "checkpoints" / "iter_0.npz")
    x, theta = saved["state"][:N_STATE], saved["state"][N_STATE:]
    z = (x.mean(axis=0) - prior.mean(axis=0).mean()) / prior.mean(axis=0).std()
    np.testing.assert_allclose(saved["auxiliary_predicted"][0], theta[0] + theta[1] * z)


def test_a_rebuilt_model_reads_the_runs_fit(tmp_path: Path) -> None:
    """A worker's or a prediction's model (composition) reads the file, never refits."""
    config_path = _case(tmp_path)
    composition.start(config_path, max_steps=1)
    config = composition.load_config(config_path)
    stack = composition.forecast_stack(config, checks=False)
    model = stack.model_error
    assert isinstance(model, MeanInput)
    assert model.prior_fit_file == tmp_path / "runs" / "run" / PRIOR_FIT_FILE
    member = MemberInput(member_id=0, state={"x": TRUTH, "nn_params": np.array([0.0, 1.0])})
    value = stack.simulator.run(member, [DataKey("t0", RESIDUAL_PREFIX + "y")])
    with np.load(model.prior_fit_file) as fit:
        assert value[0] == pytest.approx((TRUTH.mean() - fit["mu"][0]) / fit["sd"][0])
    assert MeanInput.fits == 1


def test_a_resumed_run_reads_the_fit_and_equals_an_uninterrupted_one(tmp_path: Path) -> None:
    whole = composition.start(_case(tmp_path / "whole"), max_steps=4)
    composition.start(_case(tmp_path / "parts"), max_steps=2)
    fits = MeanInput.fits
    resumed = composition.resume(tmp_path / "parts" / "runs" / "run", max_steps=2)
    assert MeanInput.fits == fits  # the resume fitted nothing
    np.testing.assert_array_equal(resumed.posterior, whole.posterior)


def test_a_resume_without_the_runs_fit_is_refused(tmp_path: Path) -> None:
    composition.start(_case(tmp_path), max_steps=1)
    run = tmp_path / "runs" / "run"
    (run / PRIOR_FIT_FILE).unlink()
    with pytest.raises(WorkflowError, match="a resume never refits"):
        composition.resume(run, max_steps=1)


def test_the_workers_of_a_process_executor_use_the_fit(tmp_path: Path) -> None:
    """The same answers as the serial run: the workers' models carry the run's fit."""
    serial = composition.start(_case(tmp_path / "serial"), max_steps=2)
    execution = '[execution]\nname = "process"\noptions = { workers = 2 }\n'
    parallel = composition.start(_case(tmp_path / "process", execution=execution), max_steps=2)
    np.testing.assert_array_equal(parallel.posterior, serial.posterior)


def test_remote_members_are_refused() -> None:
    class Remote:
        remote = True

    with pytest.raises(ConfigError, match="remote members do not have"):
        composition.refuse_remote_state_inputs(Remote(), MeanInput())
    composition.refuse_remote_state_inputs(Remote(), None)  # no model: nothing to refuse
