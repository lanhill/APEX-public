"""Optimization with a surrogate (validation 4).

The simulator is a toy reservoir in this process (each injector produces ``20 perm sqrt(x)``
of oil a year and injects ``x`` of water), and the surrogate a test-only least-squares map
on the features ``sqrt(x)``, ``x`` (and ``perm sqrt(x)``, ``perm``), which reproduces it
exactly: the surrogate's objective equals the simulator's to round-off, an independent
reference for the whole path (build over the controls, the journal's two contexts, checks,
the simulator's optimum, resume). A ``bias`` makes a deliberately wrong surrogate for the
checks' failure policies.
"""

import datetime as dt
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError, SurrogateDomainError
from apex.core.plugins import simulators, surrogate_models
from apex.core.simulation import MemberInput
from apex.core.surrogates import SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet
from apex.services import optimization, surrogates
from apex.storage import optimization_directory as layout
from apex.workflows.simulated_objective import evaluation_key
from apex.workflows.surrogate_optimization import STATE_FILE, SurrogateCheckFailed

TIMES = [f"{2001 + k}-01-01" for k in range(4)]


class ToyReservoir:
    """Field totals of the toy reservoir; counts its runs."""

    runs = 0

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        type(self).runs += 1
        inj = np.asarray(member.state["inj"], dtype=float)
        perm = float(np.asarray(member.state.get("perm", [1.0]))[0])
        totals = {"FOPT": 20.0 * perm * float(np.sum(np.sqrt(inj))), "FWIT": float(np.sum(inj))}
        start = dt.date(2000, 1, 1)
        return np.array([
            (dt.date.fromisoformat(k.time) - start).days if k.datatype == "TIME"
            else totals[k.datatype] * (TIMES.index(k.time) + 1)
            for k in keys
        ], dtype=float)  # fmt: skip


class _Features:
    """Least squares on ``sqrt(x)``, ``x`` (and ``perm sqrt(x)``, ``perm``), plus ``bias``."""

    version = "1"

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias
        self.keys: tuple[DataKey, ...] = ()
        self.weights = np.zeros((0, 0))

    @staticmethod
    def _features(inputs: SurrogateInputs) -> NDArray[np.float64]:
        x = np.asarray(inputs["inj"], dtype=np.float64)
        rows = [np.sqrt(x), x]
        if "perm" in inputs:
            perm = np.asarray(inputs["perm"], dtype=np.float64)[:1]
            rows += [perm * np.sqrt(x), perm]
        return np.vstack([*rows, np.ones((1, x.shape[1]))])

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        self.keys = snapshots.keys
        features = self._features(snapshots.inputs)
        solution, *_ = np.linalg.lstsq(features.T, snapshots.outputs.T, rcond=None)
        self.weights = np.ascontiguousarray(solution.T)

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.asarray(inputs["inj"], dtype=np.float64)
        return np.ascontiguousarray(x), np.zeros(x.shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        values = self.weights @ self._features(inputs)
        biased = [
            r for r in rows if self.keys[r].datatype == "FOPT"
        ]  # a wrong oil, never a wrong TIME
        values[biased] *= 1.0 + self.bias
        return np.ascontiguousarray(values[rows])

    def to_arrays(self) -> SurrogateParts:
        own = {"bias": self.bias, "keys": [[k.time, k.datatype, k.index] for k in self.keys]}
        return own, {"weights": self.weights}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls(bias=float(manifest["bias"]))  # type: ignore[arg-type]
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.weights = np.asarray(arrays["weights"], dtype=np.float64)
        return model


@pytest.fixture(autouse=True)
def _registered() -> Iterator[type[ToyReservoir]]:
    simulators.register("test_toy_surrogate_reservoir")(ToyReservoir)
    surrogate_models.register("test_toy_features")(_Features)
    ToyReservoir.runs = 0
    try:
        yield ToyReservoir
    finally:
        simulators._factories.pop("test_toy_surrogate_reservoir")
        surrogate_models.remove("test_toy_features")


CASE = """
[optimization]
name = "{name}"
seed = 5

[[controls.variables]]
name = "inj"
initial = [300.0, 250.0]
lower = 10.0
upper = 400.0
variance = 100.0

[objective]
name = "npv"
options = {{ start = "2000-01-01", times = {times}, wop = 1.0, wwi = 1.0, disc = 0.08, \
obj_scaling = -1.0 }}

[optimizer]
name = "enopt"
options = {{ ne = 6, maxiter = {maxiter}, alpha = 0.1, resample = 1 }}

[simulator]
name = "test_toy_surrogate_reservoir"

[execution]
name = "serial"
options = {{ retries = 0 }}
{extra}
"""

BUILD = """
[surrogate]
name = "{name}"
model = "test_toy_features"
options = {{ bias = {bias} }}
seed = 3

[case]
config_file = "{case}"
outputs = "responses"

[design]
plan = "lhs"
training = 10
held_out = 5
{box}
"""


def _case(folder: Path, name: str, *, maxiter: int = 6, extra: str = "") -> Path:
    path = folder / f"{name}.toml"
    path.write_text(
        CASE.format(name=name, times=json.dumps(TIMES), maxiter=maxiter, extra=extra),
        encoding="utf-8",
    )
    return path


def _build(folder: Path, *, bias: float = 0.0, box: str = "", case: str = "case.toml") -> Path:
    """Build the surrogate of ``case`` (written by the caller); return its manifest."""
    path = folder / "build.toml"
    path.write_text(BUILD.format(name="toy_build", bias=bias, case=case, box=box), "utf-8")
    result = surrogates.build(path)
    assert result.summary.status == "complete"
    return result.path / "model" / "surrogate.json"


def _with_surrogate(model: Path, **options: object) -> str:
    lines = [f'model_file = "{model}"']
    lines += [f"{k} = {json.dumps(v)}" for k, v in options.items()]
    return "[surrogate]\n" + "\n".join(lines)


def test_a_build_over_an_optimizations_controls_ranks_its_objective(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    gate = json.loads((model.parent.parent / "gate.json").read_text())
    assert gate["passed"]
    # The objective's ranking of the held-out runs is gated too.
    assert gate["rank"]["correlation"] == pytest.approx(1.0)
    assert {d["datatype"] for d in gate["datatypes"]} == {"FOPT", "FWIT", "TIME"}
    assert all(d["error"] < 1e-10 for d in gate["datatypes"])
    design = np.load(model.parent.parent / "design.npz")
    held = design["held_out_0"]
    assert held.shape == (2, 5)
    assert np.all((held >= 10.0) & (held <= 400.0))


def test_an_optimization_with_an_exact_surrogate_reports_the_simulators_optimum(
    tmp_path: Path,
) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    plain = optimization.optimize(_case(tmp_path, "plain"))
    built_runs = ToyReservoir.runs = 0
    run = optimization.optimize(
        _case(tmp_path, "fast", extra=_with_surrogate(model, check_every=2))
    )
    rows = layout.read_evaluations(run.run_dir)
    by_source = {s: [r for r in rows if r.get("source") == s] for s in ("surrogate", "simulator")}
    assert by_source["surrogate"]
    assert by_source["simulator"]
    assert len(by_source["surrogate"]) + len(by_source["simulator"]) == len(rows)
    # Only the checks and the optimum ran the simulator.
    assert ToyReservoir.runs - built_runs == len(by_source["simulator"])
    history = layout.read_history(run.run_dir)
    checks = [r["surrogate_check"] for r in history if "surrogate_check" in r]
    assert checks
    assert all(c["passed"] for c in checks)
    assert [c["iteration"] for c in checks] == [r["iteration"] for r in history
                                                if r["iteration"] % 2 == 0
                                                and r["kind"] == "step"]  # fmt: skip
    result = layout.read_result(run.run_dir)
    assert result["surrogate"]["objective_source"] == "simulator"
    assert result["objective"] == pytest.approx(result["surrogate_objective"], rel=1e-10)
    # An exact surrogate takes the plain run's path (to round-off in the objective).
    np.testing.assert_allclose(run.result.x, plain.result.x, rtol=1e-8)


def test_the_journal_keeps_the_two_sources_apart(tmp_path: Path) -> None:
    """The surrogate's rows are never a plain run's; the simulator's are."""
    _case(tmp_path, "case")
    model = _build(tmp_path)
    config_path = _case(tmp_path, "fast", extra=_with_surrogate(model, check_every=1))
    run = optimization.optimize(config_path)
    plain_context = optimization.evaluation_context(
        optimization.load_optimization_config(_case(tmp_path, "case"))
    )
    rows = layout.read_evaluations(run.run_dir)
    x = np.asarray(layout.read_result(run.run_dir)["x"])
    assert evaluation_key(x, 0, plain_context) in {
        r["key"] for r in rows if r["source"] == "simulator"
    }
    surrogate_keys = {r["key"] for r in rows if r["source"] == "surrogate"}
    assert evaluation_key(x, 0, plain_context) not in surrogate_keys
    # A plain objective on this journal reuses the simulator's rows and none of the
    # surrogate's: it simulates exactly the controls the simulator never evaluated.
    config = optimization.load_optimization_config(config_path).model_copy(
        update={"surrogate": None}
    )
    _, problem, _ = optimization.build(config)
    simulated = {r["key"] for r in rows if r["source"] == "simulator"}
    surrogate_only = [r for r in rows if r["source"] == "surrogate"]
    ToyReservoir.runs = 0
    problem.objective(x.reshape(-1, 1), np.zeros(1, dtype=np.int64))
    assert ToyReservoir.runs == 0
    assert evaluation_key(x, 0, plain_context) in simulated
    assert surrogate_only  # evaluated by the surrogate only: not reusable without it


def test_a_resume_after_a_check_step_is_bit_identical(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    extra = _with_surrogate(model, check_every=2)
    whole = optimization.optimize(_case(tmp_path, "whole", extra=extra))
    part = optimization.optimize(_case(tmp_path, "part", extra=extra), max_iterations=2)
    assert any("surrogate_check" in r for r in layout.read_history(part.run_dir))
    resumed = optimization.resume(part.run_dir)
    assert [r["x"] for r in layout.read_history(resumed.run_dir)] == [
        r["x"] for r in layout.read_history(whole.run_dir)
    ]
    np.testing.assert_array_equal(resumed.result.x, whole.result.x)
    assert (
        layout.read_result(resumed.run_dir)["objective"]
        == (layout.read_result(whole.run_dir)["objective"])
    )


def test_a_biased_surrogate_stops_or_switches_to_the_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    # The model passed its gate; it goes wrong afterwards (a stand-in for a model that is
    # wrong where the optimizer goes): its bias is changed in memory, not in its files.
    original = _Features.predict

    def biased(self: _Features, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> np.ndarray:
        self.bias = 0.2
        return original(self, inputs, keys)

    monkeypatch.setattr(_Features, "predict", biased)
    surrogates.fitted_model.cache_clear()
    stop = _case(tmp_path, "stop", extra=_with_surrogate(model, check_every=1))
    with pytest.raises(SurrogateCheckFailed, match="check at step 1 failed"):
        optimization.optimize(stop)
    history = layout.read_history(tmp_path / "runs" / "stop")
    assert [r["kind"] for r in history] == ["start"]  # the last checkpoint: the start

    switch = _case(
        tmp_path, "switch",
        extra=_with_surrogate(model, check_every=1, on_check_failure="simulator"),
    )  # fmt: skip
    run = optimization.optimize(switch)
    state = json.loads((run.run_dir / STATE_FILE).read_text())
    assert state["switched"]
    assert state["switched_at"] == 1
    assert [c["passed"] for c in state["checks"]] == [False]
    # From the switch on, every evaluation is the simulator's: the run is the plain one's.
    plain = optimization.optimize(_case(tmp_path, "plain"))
    np.testing.assert_array_equal(run.result.x, plain.result.x)
    assert layout.read_result(run.run_dir)["surrogate"]["switched_to_simulator"]
    # A resume of the switched run keeps the switch.
    assert optimization.resume(run.run_dir).finished


def test_controls_outside_the_domain_stop_or_go_to_the_simulator(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path, box="[design.box]\ninj = [295.0, 305.0]")
    with pytest.raises(SurrogateDomainError, match="outside the surrogate's validated domain"):
        optimization.optimize(_case(tmp_path, "stop", extra=_with_surrogate(model)))
    run = optimization.optimize(
        _case(tmp_path, "sim", extra=_with_surrogate(model, outside="simulator"))
    )
    rows = layout.read_evaluations(run.run_dir)
    assert any(r["source"] == "simulator" for r in rows)


def test_a_robust_optimization_gives_each_design_member_a_realization(tmp_path: Path) -> None:
    np.savez(tmp_path / "perm.npz", perm=np.array([[1.0, 0.5, 1.5]]))
    extra = '[model]\nrealizations_file = "perm.npz"'
    _case(tmp_path, "case", extra=extra)
    model = _build(tmp_path)
    design = np.load(model.parent.parent / "design.npz")
    names = [str(n) for n in design["variables"]]
    perm = design[f"held_out_{names.index('perm')}"]
    np.testing.assert_array_equal(perm[0], [1.0, 0.5, 1.5, 1.0, 0.5])  # ids 0..4 mod 3
    run = optimization.optimize(
        _case(tmp_path, "robust", extra=extra + "\n" + _with_surrogate(model, check_every=2))
    )
    result = layout.read_result(run.run_dir)
    assert len(result["objective_per_realization"]) == 3
    assert result["objective"] == pytest.approx(result["surrogate_objective"], rel=1e-10)


def test_validate_reports_an_unknown_plugin_and_a_bad_objective_option(tmp_path: Path) -> None:
    # The objective's options are given to its plugin while the keys are read: a refusal
    # there is said as text beside the unknown surrogate plugin, not as a traceback.
    case = _case(tmp_path, "case")
    case.write_text(case.read_text().replace("wop = 1.0", "no_such_option = 1.0"))
    text = BUILD.format(name="bad", bias=0.0, case="case.toml", box="")
    (tmp_path / "bad.toml").write_text(text.replace('"test_toy_features"', '"no_such_model"'))

    with pytest.raises(ConfigError) as error:
        surrogates.plan_build(surrogates.load_surrogate_config(tmp_path / "bad.toml"))
    assert "unknown surrogate_models plugin 'no_such_model'" in str(error.value)
    assert "no_such_option" in str(error.value)
    planned = CliRunner().invoke(app, ["validate", str(tmp_path / "bad.toml")])
    assert planned.exit_code == 1
    assert "no_such_model" in planned.output
    assert "no_such_option" in planned.output
    assert "Traceback" not in planned.output
    assert not isinstance(planned.exception, (TypeError, ValueError))


def test_what_an_optimization_with_a_surrogate_refuses(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    # Other inputs: the model arrays of a robust optimization the model never saw.
    np.savez(tmp_path / "perm.npz", perm=np.array([[1.0, 2.0]]))
    robust = '[model]\nrealizations_file = "perm.npz"\n'
    other = _case(tmp_path, "other", extra=robust + _with_surrogate(model))
    with pytest.raises(ConfigError, match="fitted on inputs"):
        optimization.optimize(other)
    text = BUILD.format(name="prior", bias=0.0, case="case.toml", box="")
    (tmp_path / "prior.toml").write_text(text.replace('"lhs"', '"prior"'))
    with pytest.raises(ConfigError, match="does not suit an optimization"):
        surrogates.build(tmp_path / "prior.toml")
    text = BUILD.format(name="obs", bias=0.0, case="case.toml", box="")
    (tmp_path / "obs.toml").write_text(text.replace('"responses"', '"observations"'))
    with pytest.raises(ConfigError, match="no observations"):
        surrogates.build(tmp_path / "obs.toml")


def test_a_ropt_run_with_a_surrogate_resumes_bit_identically(tmp_path: Path) -> None:
    pytest.importorskip("ropt")
    options = (
        '{ method = "scipy/slsqp", maxiter = 30, number_of_perturbations = 4, '
        "perturbation_magnitude = 0.5, max_batches = 200 }"
    )
    _case(tmp_path, "case")
    model = _build(tmp_path)
    extra = _with_surrogate(model, check_every=2)

    def ropt_case(name: str) -> Path:
        path = _case(tmp_path, name, extra=extra)
        text = path.read_text().replace(
            'name = "enopt"\noptions = { ne = 6, maxiter = 6, alpha = 0.1, resample = 1 }',
            f'name = "ropt"\noptions = {options}',
        )
        path.write_text(text)
        return path

    whole = optimization.optimize(ropt_case("rwhole"))
    part = optimization.optimize(ropt_case("rpart"), max_iterations=2)
    resumed = optimization.resume(part.run_dir)
    assert layout.read_history(resumed.run_dir) == layout.read_history(whole.run_dir)


def test_the_plan_counts_the_surrogates_evaluations_apart(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    plain = optimization.plan_optimization(
        optimization.load_optimization_config(_case(tmp_path, "plain"))
    )
    fast = optimization.plan_optimization(
        optimization.load_optimization_config(
            _case(tmp_path, "fast", extra=_with_surrogate(model, check_every=2))
        )
    )
    assert fast.surrogate_evaluations == plain.max_evaluations
    assert fast.max_simulator_runs == 6 // 2 + 1  # three checks and the optimum, one model
    either = optimization.plan_optimization(
        optimization.load_optimization_config(
            _case(tmp_path, "either", extra=_with_surrogate(model, outside="simulator"))
        )
    )
    assert plain.max_simulator_runs is not None
    assert either.max_simulator_runs == plain.max_simulator_runs + 6 // 3 + 1


MULTI = """
[[objectives]]
name = "npv"
options = {{ start = "2000-01-01", times = {times}, wop = 1.0, wwi = 0.02, disc = 0.08, \
obj_scaling = -1000.0 }}

[[objectives]]
name = "field_total"
label = "water_injected"
options = {{ vector = "FWIT", time = "{last}", scale = 0.001 }}

[optimizer]
name = "nsga2"
options = {{ pop_size = 8, maxiter = 4, transform = true }}
"""


def _multi(folder: Path, name: str, extra: str = "") -> Path:
    path = _case(folder, name, extra=extra)
    text = path.read_text()
    start = text.index("[objective]")
    end = text.index("[simulator]")
    path.write_text(
        text[:start] + MULTI.format(times=json.dumps(TIMES), last=TIMES[-1]) + "\n" + text[end:]
    )
    return path


def test_a_multi_objective_front_is_the_simulators(tmp_path: Path) -> None:
    """A front found with the surrogate is evaluated again by the simulator,
    member by member; with an exact surrogate its objectives, hypervolume and representative
    are the surrogate's to round-off, and the build ranks both objectives."""
    _multi(tmp_path, "case")
    model = _build(tmp_path)
    gate = json.loads((model.parent.parent / "gate.json").read_text())
    assert gate["rank"]["correlation"] == pytest.approx(1.0)
    # A population reaches past the design's box (its domain): those go to the simulator.
    extra = _with_surrogate(model, outside="simulator")
    run = optimization.optimize(_multi(tmp_path, "front", extra=extra))
    front = layout.read_pareto(run.run_dir)
    assert front is not None
    assert front["objective_source"] == "simulator"
    np.testing.assert_allclose(front["objectives"], front["surrogate_objectives"], rtol=1e-9)
    assert front["hypervolume"] == pytest.approx(front["surrogate_hypervolume"], rel=1e-9)
    assert front["dominated_by_simulator"] == []
    assert front["representative"] == front["surrogate_representative"]
    # By hand: the simulator's front points against the same reference point.
    from apex.algorithms.optimization.multiobjective import hypervolume

    points = np.asarray(front["objectives"]).T
    assert front["hypervolume"] == pytest.approx(
        hypervolume(points, front["reference_point"]), rel=1e-12
    )
    result = layout.read_result(run.run_dir)
    assert result["x"] == front["x"][front["representative"]]
    np.testing.assert_allclose(result["objectives"], result["surrogate_objectives"], rtol=1e-9)
    # Every front member, at every realization, was evaluated by the simulator.
    rows = layout.read_evaluations(run.run_dir)
    assert sum(r["source"] == "simulator" for r in rows) >= len(front["x"])


def test_a_crash_inside_the_switch_restarts_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch is recorded before the old checkpoint is removed: a process ending in
    between must not resume that checkpoint with the simulator (the review of 2026-11-03)."""
    _case(tmp_path, "case")
    model = _build(tmp_path)
    original = _Features.predict

    def biased(self: _Features, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> np.ndarray:
        self.bias = 0.2
        return original(self, inputs, keys)

    monkeypatch.setattr(_Features, "predict", biased)
    surrogates.fitted_model.cache_clear()
    extra = _with_surrogate(model, check_every=1, on_check_failure="simulator")
    whole = optimization.optimize(_case(tmp_path, "whole", extra=extra))

    from apex.workflows import optimization as workflow

    def crash(run_dir: Path) -> None:
        raise RuntimeError("the process ended here")

    monkeypatch.setattr(workflow, "_reset", crash)
    with pytest.raises(RuntimeError, match="ended here"):
        optimization.optimize(_case(tmp_path, "crashed", extra=extra))
    state = json.loads((tmp_path / "runs" / "crashed" / STATE_FILE).read_text())
    assert state["restart_pending"]
    monkeypatch.undo()
    monkeypatch.setattr(_Features, "predict", biased)
    resumed = optimization.resume(tmp_path / "runs" / "crashed")
    np.testing.assert_array_equal(resumed.result.x, whole.result.x)
    assert [r["x"] for r in layout.read_history(resumed.run_dir)] == [
        r["x"] for r in layout.read_history(whole.run_dir)
    ]
    assert not json.loads((resumed.run_dir / STATE_FILE).read_text())["restart_pending"]


def test_the_returned_result_is_the_simulators_and_plain_configs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    original = _Features.predict

    def slightly(self: _Features, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> np.ndarray:
        self.bias = 0.001  # within the checks (1 % for totals), not the simulator's value
        return original(self, inputs, keys)

    monkeypatch.setattr(_Features, "predict", slightly)
    surrogates.fitted_model.cache_clear()
    run = optimization.optimize(_case(tmp_path, "fast", extra=_with_surrogate(model)))
    report = layout.read_result(run.run_dir)
    assert report["objective"] != report["surrogate_objective"]
    # What apex optimize prints (the run's result) is result.json's: the simulator's.
    assert float(np.mean(run.result.fun)) == pytest.approx(report["objective"], rel=1e-12)
    plain = optimization.optimize(_case(tmp_path, "plain"))
    assert "surrogate" not in layout.read_config(plain.run_dir)


def test_a_check_of_the_model_draws_within_its_design_box(tmp_path: Path) -> None:
    """A check draws the controls within the box the build spanned, not the controls' wider
    bounds (whose runs would lie outside the domain and fail the check: the review of
    2026-11-03)."""
    _case(tmp_path, "case")
    model = _build(tmp_path, box="[design.box]\ninj = [295.0, 305.0]")
    checked = surrogates.check_surrogate(model, tmp_path / "case.toml", runs=4, seed=3)
    summary = checked.summary
    assert summary.passed, summary.notes
    assert summary.outside == []
    design = np.load(checked.path / "design.npz")
    drawn = design["held_out_0"]
    assert np.all((drawn >= 295.0) & (drawn <= 305.0))


def test_a_stopped_run_goes_on_with_the_simulator_after_its_section_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The way SurrogateCheckFailed's message gives: on_check_failure = "simulator" in the
    run's config.json, then resume. The run starts again with the simulator (its stream at
    the start, not a checkpoint's), and ends as the plain run does."""
    _case(tmp_path, "case")
    model = _build(tmp_path)
    original = _Features.predict

    def biased(self: _Features, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> np.ndarray:
        self.bias = 0.2
        return original(self, inputs, keys)

    monkeypatch.setattr(_Features, "predict", biased)
    surrogates.fitted_model.cache_clear()
    stopped = tmp_path / "runs" / "stopped"
    with pytest.raises(SurrogateCheckFailed):
        optimization.optimize(_case(tmp_path, "stopped", extra=_with_surrogate(model)))
    config = layout.read_config(stopped)
    config["surrogate"]["on_check_failure"] = "simulator"
    layout.write_config(stopped, config)
    resumed = optimization.resume(stopped)
    plain = optimization.optimize(_case(tmp_path, "plain"))
    np.testing.assert_array_equal(resumed.result.x, plain.result.x)
    assert [r["x"] for r in layout.read_history(stopped)] == [
        r["x"] for r in layout.read_history(plain.run_dir)
    ]
    result = layout.read_result(stopped)
    assert result["surrogate"]["switched_to_simulator"]
    assert "surrogate_objective" not in result  # every value was the simulator's


def test_an_interruption_between_a_checks_runs_and_its_checkpoint_resumes_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    extra = _with_surrogate(model, check_every=2)
    whole = optimization.optimize(_case(tmp_path, "whole", extra=extra))

    saved = layout.save_checkpoint
    calls = {"n": 0}

    def ends_at_the_second_step(run_dir: Path, state: Mapping[str, object]) -> None:
        calls["n"] += 1
        if calls["n"] == 3:  # the start, step 1, then step 2: after its check's runs
            raise RuntimeError("the process ended here")
        saved(run_dir, state)

    monkeypatch.setattr(layout, "save_checkpoint", ends_at_the_second_step)
    with pytest.raises(RuntimeError, match="ended here"):
        optimization.optimize(_case(tmp_path, "cut", extra=extra))
    monkeypatch.undo()
    cut = tmp_path / "runs" / "cut"
    simulated_before = sum(r["source"] == "simulator" for r in layout.read_evaluations(cut))
    assert simulated_before >= 1  # the check's runs were made and journalled
    resumed = optimization.resume(cut)
    assert [r["x"] for r in layout.read_history(cut)] == [
        r["x"] for r in layout.read_history(whole.run_dir)
    ]
    np.testing.assert_array_equal(resumed.result.x, whole.result.x)
    # The check's simulations were reused, not run again.
    rows = layout.read_evaluations(cut)
    assert len({r["key"] for r in rows}) == len(rows)


def test_a_surrogate_run_without_its_section_and_a_rank_tolerance_without_responses_are_refused(
    tmp_path: Path,
) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    run = optimization.optimize(
        _case(tmp_path, "fast", extra=_with_surrogate(model)), max_iterations=1
    )
    config = layout.read_config(run.run_dir)
    del config["surrogate"]
    layout.write_config(run.run_dir, config)
    with pytest.raises(ConfigError, match="no longer has \\[surrogate\\]"):
        optimization.resume(run.run_dir)
    text = BUILD.format(name="rank", bias=0.0, case="case.toml", box="")
    text = text.replace(
        'outputs = "responses"', 'outputs = { vectors = ["FOPT"], times = ["2001-01-01"] }'
    )
    (tmp_path / "rank.toml").write_text(text + "\n[gate]\nmin_rank_correlation = 0.9\n")
    with pytest.raises(ConfigError, match="min_rank_correlation"):
        surrogates.load_surrogate_config(tmp_path / "rank.toml")


def test_validating_an_optimization_with_a_surrogate_writes_nothing(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    model = _build(tmp_path)
    config = optimization.load_optimization_config(
        _case(tmp_path, "planned", extra=_with_surrogate(model))
    )
    optimization.plan_optimization(config)
    assert not (tmp_path / "runs" / "planned").exists()


def test_a_copied_model_is_checked_within_the_box_its_manifest_records(tmp_path: Path) -> None:
    """The box travels with the model (its manifest's control_box), so a model copied away
    from its build's design.npz can still be checked (the re-review of 2026-11-03)."""
    import shutil

    _case(tmp_path, "case")
    model = _build(tmp_path, box="[design.box]\ninj = [295.0, 305.0]")
    manifest = json.loads(model.read_text())
    low, high = manifest["control_box"]["inj"]
    assert np.all(np.asarray(low) >= 295.0)
    assert np.all(np.asarray(high) <= 305.0)
    copy = tmp_path / "copy"
    shutil.copytree(model.parent, copy / "model")
    shutil.copy(model.parent.parent / "snapshots.npz", copy / "snapshots.npz")
    assert not (copy / "design.npz").exists()
    checked = surrogates.check_surrogate(
        copy / "model" / "surrogate.json", tmp_path / "case.toml", runs=4, seed=5
    )
    assert checked.summary.passed, checked.summary.notes
    drawn = np.load(checked.path / "design.npz")["held_out_0"]
    assert drawn.shape[0] == len(low) == len(high) == 2
    for entry in range(drawn.shape[0]):
        assert np.all((drawn[entry] >= low[entry]) & (drawn[entry] <= high[entry]))


def test_a_box_of_other_controls_is_refused(tmp_path: Path) -> None:
    """A model built on other controls (another size) or a damaged box is refused with the
    control's name, never drawn misaligned (the review of 2026-11-03)."""
    _case(tmp_path, "case")
    model = _build(tmp_path)
    manifest = json.loads(model.read_text())
    low, high = manifest["control_box"]["inj"]
    for box, message in (
        ([[*low, 10.0], [*high, 20.0]], "3 lower and 3 upper bounds"),
        ([high, low], "finite and ordered"),
    ):
        manifest["control_box"]["inj"] = box
        model.write_text(json.dumps(manifest))
        with pytest.raises(ConfigError, match=message):
            surrogates.check_surrogate(model, tmp_path / "case.toml", runs=2, seed=1)


@pytest.mark.parametrize("extent", ["hull", "mahalanobis"])
def test_a_check_says_when_the_domain_is_narrower_than_its_box(
    tmp_path: Path, extent: str, caplog: pytest.LogCaptureFixture
) -> None:
    _case(tmp_path, "case")
    path = tmp_path / "build.toml"
    text = BUILD.format(name=extent, bias=0.0, case="case.toml", box="")
    path.write_text(text.replace("seed = 3", f'seed = 3\nextent = "{extent}"'), encoding="utf-8")
    built = surrogates.build(path)
    model = built.path / "model" / "surrogate.json"
    with caplog.at_level("WARNING"):
        checked = surrogates.check_surrogate(
            model, tmp_path / "case.toml", runs=6, seed=1, min_inside=0.0
        )
    assert any("narrower" in note for note in checked.summary.notes)
    warned = [
        r for r in caplog.records if r.getMessage() == "surrogate_check_box_wider_than_domain"
    ]
    assert warned
    assert warned[0].__dict__["extent"] == extent
    assert warned[0].__dict__["controls"] == 2


def test_a_manifest_without_a_control_box_keeps_its_bytes(tmp_path: Path) -> None:
    """Models built before control boxes existed, and every history-matching one, keep
    their manifests' bytes and so their identities: the key is written only when given."""
    from apex.storage.surrogate_files import write_surrogate

    _case(tmp_path, "case")
    model = _build(tmp_path)
    loaded = surrogates.load_surrogate(model)
    kwargs = {
        "plugin": loaded.manifest.plugin,
        "model": loaded.model,
        "domain": loaded.domain,
        "provenance": loaded.manifest.provenance,
        "options": loaded.manifest.options,
        "gate": loaded.manifest.gate,
        "snapshots": loaded.manifest.snapshots,
        "fit": loaded.manifest.fit,
        "apex_commit": loaded.manifest.apex_commit,
    }
    without = write_surrogate(tmp_path / "a" / "surrogate.json", **kwargs)  # type: ignore[arg-type]
    explicit = write_surrogate(
        tmp_path / "b" / "surrogate.json",
        control_box=None,
        input_sizes=None,
        control_inputs=None,
        **kwargs,  # type: ignore[arg-type]
    )
    text = without.path.read_text()
    assert "control_box" not in text
    assert "input_sizes" not in text
    assert "control_inputs" not in text
    assert text == explicit.path.read_text()
    assert without.identity == explicit.identity
    # The same model with its box recorded is another file, so another identity.
    assert "control_box" in model.read_text()
    assert loaded.identity != without.identity
