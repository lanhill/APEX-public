"""Builds whose runs give their gradients (``uses_gradients``; gradient-enhanced models).

The user's decision (2026-11-11): a plugin that learns from gradients makes its build run
every member with its sensitivities (``run_with_sensitivities``), and the snapshot set keeps
them (``SnapshotSet.gradients``, ``(n_keys, n_inputs, n_runs)``). The reference is the linear
case of ``test_surrogate_workflow.py``, ``y = G x``: every run's gradients are ``G`` exactly.
Builds of plugins without the capability are those of before.
"""

import dataclasses
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import surrogate_models
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import SnapshotProvenance, SnapshotSet
from apex.services import surrogates
from apex.simulation.sensitivity import SensitivityRun
from apex.storage.surrogate_directory import load_snapshots


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}_for_gradients", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = _load("test_surrogate_workflow")


class _Gradiented(workflow._Affine):  # type: ignore[misc,name-defined]
    """The affine map with ``uses_gradients``; it keeps the gradients its fit was given."""

    uses_gradients = True

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        assert snapshots.gradients is not None
        self.seen = snapshots.gradients


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register("test_affine_build")(workflow._Affine)
    surrogate_models.register("test_gradiented")(_Gradiented)
    try:
        yield
    finally:
        surrogate_models.remove("test_affine_build")
        surrogate_models.remove("test_gradiented")


def _build(folder: Path, name: str = "grad", design: str | None = None) -> Path:
    path = workflow.write_build(
        folder, name=name, design=design or 'plan = "prior"\ntraining = 12\nheld_out = 4'
    )
    path.write_text(
        path.read_text().replace('model = "test_affine_build"', 'model = "test_gradiented"')
    )
    return cast("Path", path)


def test_every_run_keeps_its_gradients_and_a_resume_keeps_them(tmp_path: Path) -> None:
    workflow.write_case(tmp_path)
    forward = np.load(tmp_path / "forward.npz")["matrix"]
    path = _build(tmp_path)
    config = surrogates.load_surrogate_config(path)
    case = surrogates._build_case(config)
    assert case.gradients
    assert surrogates.fingerprint_parts(config, case)["gradients"] == "sensitivities"
    result = surrogates.build(path)
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.gradients is not None
    assert snapshots.gradients.shape == (len(snapshots.keys), 3, snapshots.n_runs)
    # y = G x: every run's gradients are G exactly, in the keys' and the state's order.
    expected = np.repeat(forward[:, :, None], snapshots.n_runs, axis=2)
    np.testing.assert_array_equal(snapshots.gradients, expected)
    # The values are those of a plain run.
    x = snapshots.inputs["x"]
    np.testing.assert_allclose(snapshots.outputs, forward @ x, atol=1e-12)
    again = surrogates.build(path)
    assert again.summary == result.summary


def test_without_the_capability_nothing_changes(tmp_path: Path) -> None:
    workflow.write_case(tmp_path)
    path = workflow.write_build(tmp_path)
    config = surrogates.load_surrogate_config(path)
    case = surrogates._build_case(config)
    assert not case.gradients
    assert "gradients" not in surrogates.fingerprint_parts(config, case)
    result = surrogates.build(path)
    assert load_snapshots(result.path / "snapshots.npz").gradients is None


class _NoSensitivity:
    """A simulator's stand-in without ``run_with_sensitivities``."""

    def __init__(self, simulator: object) -> None:
        self._simulator = simulator

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        if name == "run_with_sensitivities":
            raise AttributeError(name)
        return getattr(self._simulator, name)


def test_what_cannot_give_gradients_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow.write_case(tmp_path)
    config = surrogates.load_surrogate_config(_build(tmp_path))
    plain = surrogates.surrogate_case
    case = surrogates._build_case(config)

    def without(*args: Any, **kwargs: Any) -> surrogates.SurrogateCase:  # noqa: ANN401
        found = plain(*args, **kwargs)
        return dataclasses.replace(found, simulator=_NoSensitivity(found.simulator))

    monkeypatch.setattr(surrogates, "surrogate_case", without)
    with pytest.raises(ConfigError, match="gives none"):
        surrogates.plan_build(config)
    monkeypatch.setattr(surrogates, "surrogate_case", plain)
    run = config.model_copy(update={"design": config.design.model_copy(update={"plan": "run"})})
    assert isinstance(case.simulator, SensitivityRun)
    plain_case = dataclasses.replace(case, simulator=case.simulator.simulator, gradients=False)
    with pytest.raises(ConfigError, match=r"the plan 'run' \(prior, lhs or file only\)"):
        surrogates._with_gradients(run, plain_case)
    with pytest.raises(ConfigError, match="state fields"):
        surrogates._with_gradients(config, dataclasses.replace(plain_case, fields=("SWAT",)))
    monkeypatch.setattr(surrogates, "MAX_GRADIENT_VALUES", 100)
    with pytest.raises(ConfigError, match="over the limit of 100"):
        surrogates._with_gradients(config, plain_case)


def test_a_sets_gradients_are_checked_sliced_and_digested() -> None:
    layout = StateLayout((StateVariable("a", 2), StateVariable("b", 1)))
    keys = (DataKey("1", "y"), DataKey("2", "y"))
    rng = np.random.default_rng(4)
    fields: dict[str, Any] = {
        "layout": layout, "inputs": {"a": rng.standard_normal((2, 4)),
                                     "b": rng.standard_normal((1, 4))},
        "keys": keys, "outputs": rng.standard_normal((2, 4)), "run_ids": (0, 1, 2, 3),
        "provenance": SnapshotProvenance(simulator="test"),
    }  # fmt: skip
    plain = SnapshotSet(**fields)
    gradients = rng.standard_normal((2, 3, 4))
    full = SnapshotSet(**fields, gradients=gradients)
    assert full.digest() != plain.digest()
    assert SnapshotSet(**fields, gradients=gradients.copy()).digest() == full.digest()
    some = full.columns([3, 1])
    assert some.gradients is not None
    np.testing.assert_array_equal(some.gradients, gradients[:, :, [3, 1]])
    assert plain.columns([3, 1]).gradients is None
    with pytest.raises(ValueError, match=r"gradients must be \(2, 3, 4\)"):
        SnapshotSet(**fields, gradients=gradients[:, :2])


def test_a_partial_build_resumes_with_its_gradients_on_worker_processes(tmp_path: Path) -> None:
    """Five members on a process pool (the sensitivity wrapper pickled), the build stopped,
    then resumed: every run's gradients are G, the journalled ones and the new ones."""
    workflow.write_case(tmp_path, execution='name = "process"\noptions = { workers = 2 }')
    forward = np.load(tmp_path / "forward.npz")["matrix"]
    path = _build(tmp_path)
    first = surrogates.build(path, max_runs=5)
    assert not first.complete
    result = surrogates.build(path)
    assert result.complete
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.gradients is not None
    expected = np.repeat(forward[:, :, None], snapshots.n_runs, axis=2)
    np.testing.assert_array_equal(snapshots.gradients, expected)


def test_a_complete_build_is_found_without_its_plugin(tmp_path: Path) -> None:
    """As every complete build: its fingerprint (with its gradients part) is the saved one."""
    workflow.write_case(tmp_path)
    path = _build(tmp_path)
    result = surrogates.build(path)
    surrogate_models.remove("test_gradiented")
    try:
        again = surrogates.build(path)
    finally:
        surrogate_models.register("test_gradiented")(_Gradiented)
    assert again.summary == result.summary


def test_a_refused_option_stamps_nothing(tmp_path: Path) -> None:
    """A plugin refusing its options is refused before the build's folder is opened."""

    class _Picky(_Gradiented):
        def __init__(self, ridge: float = 0.0, fail: bool = False, strict: bool = False) -> None:
            if strict:
                raise ValueError("strict is refused")
            super().__init__(ridge=ridge, fail=fail)

    surrogate_models.register("test_picky")(_Picky)
    try:
        workflow.write_case(tmp_path)
        path = _build(tmp_path)
        path.write_text(path.read_text().replace('model = "test_gradiented"',
                                                 'model = "test_picky"')
                        .replace("options = {}", "options = { strict = true }"))  # fmt: skip
        with pytest.raises(ConfigError, match="strict is refused"):
            surrogates.build(path)
        assert not (tmp_path / "runs" / "grad" / "state.json").exists()
    finally:
        surrogate_models.remove("test_picky")


def test_a_coarse_stack_or_parameterizations_are_refused(tmp_path: Path) -> None:
    workflow.write_case(tmp_path)
    config = surrogates.load_surrogate_config(_build(tmp_path))
    case = surrogates._build_case(config)
    assert isinstance(case.simulator, SensitivityRun)
    plain = dataclasses.replace(case, simulator=case.simulator.simulator, gradients=False)
    coarse = cast(Any, SimpleNamespace(surrogate=config.surrogate, design=config.design,
                                       coarse=object()))  # fmt: skip
    with pytest.raises(ConfigError, match=r"a \[coarse\] stack"):
        surrogates._with_gradients(coarse, plain)
    mapped = dataclasses.replace(plain, properties=cast(Any, SimpleNamespace(identity=False)))
    with pytest.raises(ConfigError, match="parameterizations"):
        surrogates._with_gradients(config, cast(Any, mapped))


def test_a_wrongly_sized_sensitivity_output_fails_the_member(tmp_path: Path) -> None:
    from apex.core.errors import SimulationError
    from apex.core.simulation import MemberInput
    from apex.storage.surrogate_directory import SurrogateDirectory
    from apex.workflows.surrogate import _Journalled

    class _Short:
        def output_size(self, n_keys: int) -> int:
            return n_keys * 4

        def run(self, member: MemberInput, keys: Any) -> np.ndarray:  # noqa: ANN401
            return np.zeros(len(keys) * 3)

    wrapped = _Journalled(cast(Any, _Short()), SurrogateDirectory(tmp_path))
    member = MemberInput(member_id=0, state={"x": np.zeros(3)})
    keys = [DataKey("1", "y"), DataKey("2", "y")]
    with pytest.raises(SimulationError, match=r"returned 6 values for 2 keys \(8 expected\)"):
        wrapped.run(member, keys)


def test_a_history_match_does_not_refit_it() -> None:
    """A run's checks are plain runs: a refit set would have no gradients for it."""
    config = SimpleNamespace(surrogate=SimpleNamespace(on_check_failure="refit"))
    model = SimpleNamespace(model=_Gradiented(), manifest=None, path=Path("model.json"))
    with pytest.raises(ConfigError, match="learns from the runs' gradients, which a run's"):
        surrogates.refuse_model(cast(Any, config), cast(Any, model), cast(Any, None),
                                cast(Any, None))  # fmt: skip


def _two_variable_case(folder: Path) -> NDArray[np.float64]:
    """``y = G [b; a]``: the layout is ``a`` (2), then ``b`` (3), and the simulator stacks
    ``b`` first; ``G`` ``(8, 5)`` is asymmetric. Returns ``G`` in the layout's column order."""
    folder.mkdir(parents=True, exist_ok=True)
    forward = np.random.default_rng(3).standard_normal((8, 5))  # columns: b0 b1 b2 a0 a1
    np.savez(folder / "forward.npz", matrix=forward)
    keys = [(t, d) for d in ("WOPR:P1", "WBHP:P1") for t in workflow.TIMES]
    rows = ["time,datatype,value,variance"] + [f"{t},{d},1.0,0.01" for t, d in keys]
    (folder / "observations.csv").write_text("\n".join(rows) + "\n")
    outputs = ", ".join(f'["{t}", "{d}"]' for t, d in keys)
    (folder / "case.toml").write_text(f"""
[run]
name = "case"
seed = 1

[[state.variables]]
name = "a"
size = 2
lower = -4.0
upper = 4.0

[[state.variables]]
name = "b"
size = 3
lower = -4.0
upper = 4.0

[prior]
ensemble_size = 20
generator = {{ name = "gaussian", options = {{ mean = 0.0, std = 1.0 }} }}

[observations]
file = "observations.csv"

[simulator]
name = "linear"

[simulator.options]
forward_file = "forward.npz"
variables = ["b", "a"]
outputs = [{outputs}]

[execution]
name = "serial"

[method]
name = "esmda"
options = {{ inflation = [1.0] }}

[case]
kind = "real"
""")
    in_layout: NDArray[np.float64] = np.concatenate([forward[:, 3:], forward[:, :3]], axis=1)
    return in_layout


@pytest.mark.parametrize("plan", ["prior", "lhs"])
def test_the_gradients_follow_the_layouts_column_order(tmp_path: Path, plan: str) -> None:
    """Two variables, the simulator stacking them the other way round: every run's
    ``gradients[k, i, r]`` is G's entry for key ``k`` and the layout's input ``i`` (``a``
    first), not the simulator's order (an asymmetric G tells the two apart)."""
    expected = _two_variable_case(tmp_path)
    path = _build(tmp_path, name=f"two_{plan}",
                  design=f'plan = "{plan}"\ntraining = 8\nheld_out = 3')  # fmt: skip
    result = surrogates.build(path)
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.layout.names == ("a", "b")
    assert snapshots.gradients is not None
    for r in range(snapshots.n_runs):
        np.testing.assert_array_equal(snapshots.gradients[:, :, r], expected)
    assert not np.array_equal(expected, np.load(tmp_path / "forward.npz")["matrix"])


def test_gradients_with_calibration_runs_are_refused(tmp_path: Path) -> None:
    """A plugin with both capabilities would run its calibration runs with sensitivities
    never used: refused (no plugin has both yet)."""

    class _Both(_Gradiented):
        uses_calibration_runs = True

    surrogate_models.register("test_both")(_Both)
    try:
        workflow.write_case(tmp_path)
        path = _build(tmp_path)
        path.write_text(path.read_text().replace('model = "test_gradiented"',
                                                 'model = "test_both"'))  # fmt: skip
        with pytest.raises(ConfigError, match="calibrates a spread on calibration runs"):
            surrogates.plan_build(surrogates.load_surrogate_config(path))
    finally:
        surrogate_models.remove("test_both")


def test_a_complete_build_is_found_when_its_plugin_now_refuses_its_options(
    tmp_path: Path,
) -> None:
    """The folder's recorded fingerprint answers for a plugin that cannot (its gradients
    part included), as for one that is not installed."""

    class _Strict(_Gradiented):
        refuse = False

        def __init__(self, ridge: float = 0.0, fail: bool = False) -> None:
            if _Strict.refuse:
                raise ValueError("now refused")
            super().__init__(ridge=ridge, fail=fail)

    surrogate_models.register("test_strict")(_Strict)
    try:
        workflow.write_case(tmp_path)
        path = _build(tmp_path)
        path.write_text(path.read_text().replace('model = "test_gradiented"',
                                                 'model = "test_strict"'))  # fmt: skip
        result = surrogates.build(path)
        _Strict.refuse = True
        assert surrogates.build(path).summary == result.summary
    finally:
        surrogate_models.remove("test_strict")
