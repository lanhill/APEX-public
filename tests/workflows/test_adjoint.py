"""The adjoint update end to end on linear and tanh models."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey
from apex.services import composition
from apex.simulation.analytic import LinearModel


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path
    forward: np.ndarray


CaseFactory = Callable[..., Case]

ADJOINT = 'strategy = { name = "approx", options = { energy = 1.0, sensitivities = true } }'
PLAIN = 'strategy = { name = "approx", options = { energy = 1.0 } }'
ESMDA = 'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'


def test_for_a_linear_model_the_adjoint_update_is_the_ensemble_one(
    linear_case: CaseFactory,
) -> None:
    """g(m) = G m: the ensemble anomalies are exactly G dX, so both paths agree."""
    adjoint = linear_case(name="adjoint", method=ESMDA + ADJOINT, ensemble_size=20)
    plain = linear_case(name="plain", method=ESMDA + PLAIN, ensemble_size=20)

    a, b = composition.start(adjoint.config), composition.start(plain.config)

    np.testing.assert_allclose(a.posterior, b.posterior, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize(
    "method",
    [
        ESMDA + ADJOINT,
        'name = "lm_enrml"\n' + ADJOINT + "\nschedule = { initial = 1.0 }\nmax_iterations = 6",
    ],
    ids=["esmda", "lm_enrml"],
)
def test_the_adjoint_update_runs_on_a_nonlinear_model_and_resumes(
    linear_case: CaseFactory, method: str
) -> None:
    extra = 'nonlinearity = "tanh"'
    whole = linear_case(name="whole", method=method, ensemble_size=12, simulator_extra=extra)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=12, simulator_extra=extra)
    composition.start(case.config, max_steps=2)

    resumed = composition.resume(case.directory / "runs" / "parts")

    kept = [r.data_mismatch_mean for r in uninterrupted.history if r.accepted]
    assert kept[-1] < kept[0] / 2  # the tanh model saturates
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)


def test_the_linear_models_jacobian_is_exact() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((4, 3))
    outputs = [[f"t{i}", "y"] for i in range(4)]
    model = LinearModel(outputs, ["x"], matrix=matrix, nonlinearity="tanh")
    x = rng.standard_normal(3)
    keys = [DataKey("t1", "y"), DataKey("t3", "y")]

    values, jacobian = model.run_with_sensitivities(MemberInput(0, {"x": x}), keys)

    step = 1e-6
    numeric = np.column_stack(
        [(model.run(MemberInput(0, {"x": x + step * e}), keys) - values) / step for e in np.eye(3)]
    )
    np.testing.assert_allclose(jacobian, numeric, rtol=1e-4, atol=1e-6)


def test_a_simulator_without_gradients_is_refused(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(LinearModel, "run_with_sensitivities")
    case = linear_case(method=ESMDA + ADJOINT, ensemble_size=10)

    with pytest.raises(ConfigError, match="needs a simulator with gradients"):
        composition.start(case.config)


def _mean_jacobian(
    model: LinearModel, state: np.ndarray, keys: list[DataKey], ids: np.ndarray | None = None
) -> np.ndarray:
    ids = np.arange(state.shape[1]) if ids is None else ids
    jacobians = [
        model.run_with_sensitivities(MemberInput(int(m), {"x": state[:, j]}), keys)[1]
        for j, m in enumerate(ids)
    ]
    mean: np.ndarray = np.mean(jacobians, axis=0)
    return mean


def test_the_sensitivity_is_the_survivors_mean(linear_case: CaseFactory) -> None:
    """Members that fail do not enter the mean (tanh model: the Jacobians differ)."""
    case = linear_case(
        method=ESMDA + ADJOINT,
        ensemble_size=10,
        simulator_extra='nonlinearity = "tanh"\nfail_members = [2, 7]',
        failures="[failures]\nmin_success_fraction = 0.5",
    )
    run = composition.build_run(composition.load_config(case.config))
    workflow = run.workflow
    state = np.random.default_rng(3).standard_normal((3, 10))

    ok, _, _, sensitivity = workflow._forecast(state, np.arange(10), 0, sensitivities=True)

    assert ok.sum() == 8
    assert not ok[[2, 7]].any()
    assert sensitivity is not None
    model = workflow.simulator
    assert isinstance(model, LinearModel)
    keys = list(workflow.raw_observations.keys)
    survivors = _mean_jacobian(model, state[:, ok], keys, np.arange(10)[ok])
    np.testing.assert_allclose(sensitivity, survivors, rtol=1e-13)


def test_cross_validation_rows_leave_the_sensitivity(well_case: Callable[..., Case]) -> None:
    from apex.algorithms.data_assimilation.algorithm import ForecastRequest

    case = well_case(
        method='name = "lm_enrml"\n' + ADJOINT + "\nmax_iterations = 3",
        validation='[validation]\ncv = { wells = ["P7", "P8"] }\nstopping = { name = "select" }',
    )
    workflow = composition.build_run(composition.load_config(case.config)).workflow
    state = np.random.default_rng(4).standard_normal((30, 10))

    forecast = workflow._forecast_request(
        ForecastRequest("ensemble", 0, state, np.arange(10), with_sensitivities=True)
    )

    assert forecast.sensitivity is not None
    hm = [
        i
        for i, k in enumerate(workflow.raw_observations.keys)
        if k.datatype not in ("WOPR:P7", "WOPR:P8")
    ]
    assert forecast.sensitivity.shape == (len(hm), 30)
    model = workflow.simulator
    assert isinstance(model, LinearModel)
    full = _mean_jacobian(model, state, list(workflow.raw_observations.keys))
    np.testing.assert_allclose(forecast.sensitivity, full[hm], rtol=1e-13)


def test_the_checkpointed_sensitivity_is_the_accepted_iterates(linear_case: CaseFactory) -> None:
    method = 'name = "lm_enrml"\n' + ADJOINT + "\nschedule = { initial = 1.0 }\nmax_iterations = 6"
    case = linear_case(method=method, ensemble_size=12, simulator_extra='nonlinearity = "tanh"')
    composition.start(case.config, max_steps=3)
    checkpoints = case.directory / "runs" / "case" / "checkpoints"
    latest = sorted(checkpoints.glob("algorithm_state_*.npz"), key=lambda p: p.stat().st_mtime)[-1]

    with np.load(latest) as saved:
        state, sensitivity = saved["state"], saved["sensitivity"]
    workflow = composition.build_run(composition.load_config(case.config)).workflow
    model = workflow.simulator
    assert isinstance(model, LinearModel)
    expected = _mean_jacobian(model, state, list(workflow.raw_observations.keys))
    np.testing.assert_allclose(sensitivity, expected, rtol=1e-13)


def test_a_localized_adjoint_update_is_refused(linear_case: CaseFactory) -> None:
    method = ESMDA + ADJOINT + '\nlocalization = { name = "adaptive_prior" }'
    case = linear_case(method=method, ensemble_size=10)

    with pytest.raises(ConfigError, match="does not take a localization"):
        composition.start(case.config)


def test_a_transposed_jacobian_is_refused() -> None:
    from apex.core.errors import SimulationError
    from apex.workflows.history_matching import SensitivityRun

    class Transposed:
        def run_with_sensitivities(
            self, member: MemberInput, keys: list[DataKey]
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(len(keys)), np.zeros((3, len(keys)))

    run = SensitivityRun(Transposed(), 3)  # type: ignore[arg-type]
    with pytest.raises(SimulationError, match=r"Jacobian \(2, 3\)"):
        run.run(MemberInput(0, {"x": np.zeros(3)}), [DataKey("t0", "y"), DataKey("t1", "y")])
