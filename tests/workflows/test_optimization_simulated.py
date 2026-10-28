"""Simulator-backed optimization: NPV, the evaluation journal and resume.

The simulator here is a toy reservoir in this process: each year it produces
``20 * sum(sqrt(inj))`` of oil (times the realization's ``perm``) and injects ``sum(inj)`` of
water, so with an oil price of 1 and an injection cost of 1 the yearly cash flow
``20 sqrt(x) - x`` of each injector peaks at ``x = 100``.
"""

import datetime as dt
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.optimization import NetPresentValue
from apex.core.errors import ConfigError, SimulationError, WorkflowError
from apex.core.plugins import simulators
from apex.core.simulation import MemberInput
from apex.domain.observations import DataKey
from apex.services import optimization
from apex.storage import optimization_directory as layout
from apex.workflows.simulated_objective import SimulatedObjective, evaluation_key

#: Ten yearly reports: NumPy sums 8 or more terms pairwise, so a value that depended on its
#: batch would show here (the numerics review of 2026-10-03).
TIMES = [f"{2001 + k}-01-01" for k in range(10)]


class ToyReservoir:
    """Field totals of the toy reservoir (see the module docstring); counts its runs."""

    runs = 0
    #: Fail the run with this number (1-based), once.
    fail_at: int | None = None

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> np.ndarray:
        type(self).runs += 1
        if type(self).fail_at == type(self).runs:
            type(self).fail_at = None
            raise SimulationError("the toy reservoir failed (on purpose)")
        inj = np.asarray(member.state["inj"])
        perm = float(np.asarray(member.state.get("perm", [1.0]))[0])
        oil, water = 20.0 * perm * np.sum(np.sqrt(inj)), float(np.sum(inj))
        rate = {"FOPT": oil, "FWIT": water}
        start = dt.date(2000, 1, 1)
        return np.array([
            (dt.date.fromisoformat(k.time) - start).days if k.datatype == "TIME"
            else rate[k.datatype] * (TIMES.index(k.time) + 1)
            for k in keys
        ])  # fmt: skip


@pytest.fixture
def toy() -> Iterator[type[ToyReservoir]]:
    simulators.register("test_toy_reservoir")(ToyReservoir)
    ToyReservoir.runs, ToyReservoir.fail_at = 0, None
    try:
        yield ToyReservoir
    finally:
        simulators._factories.pop("test_toy_reservoir")


CONFIG = """
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
name = "test_toy_reservoir"

[execution]
name = "serial"
options = {{ retries = 0 }}
{extra}
"""


def _config(tmp_path: Path, name: str, maxiter: int = 6, extra: str = "") -> Path:
    path = tmp_path / f"{name}.toml"
    text = CONFIG.format(name=name, times=json.dumps(TIMES), maxiter=maxiter, extra=extra)
    path.write_text(text, encoding="utf-8")
    return path


def test_npv_matches_a_hand_computation() -> None:
    """Independent reference: the discounted cash flow of PET's npv, written out by hand."""
    npv = NetPresentValue(
        start="2000-01-01", times=["2000-12-31", "2002-01-01"], wop=50.0, wgp=2.0, wwp=5.0,
        wwi=3.0, disc=0.1, obj_scaling=-1e3,
    )  # fmt: skip
    keys = list(npv.response_keys())
    assert [str(k) for k in keys][:2] == ["FOPT@2000-12-31", "FOPT@2002-01-01"]
    totals = {
        "FOPT": [10.0, 25.0],
        "FGPT": [100.0, 180.0],
        "FWPT": [1.0, 4.0],
        "FWIT": [12.0, 30.0],
    }
    assert str(keys[-1]) == "TIME@2000-12-31"
    responses = np.array([[totals[k.datatype][i % 2]] for i, k in enumerate(keys[:-1])] + [[365]])
    days = [365, 731]
    cash = [
        (10 * 50 + 100 * 2 - 1 * 5 - 12 * 3) / 1.1 ** (days[0] / 365),
        (15 * 50 + 80 * 2 - 3 * 5 - 18 * 3) / 1.1 ** (days[1] / 365),
    ]
    np.testing.assert_allclose(npv.evaluate(responses), [sum(cash) / -1e3], rtol=1e-14)


def test_npv_asks_only_for_priced_totals_and_checks_its_times() -> None:
    npv = NetPresentValue(start="2000-01-01", times=["2001-01-01"], wop=1.0, wwi=2.0)
    assert [k.datatype for k in npv.response_keys()] == ["FOPT", "FWIT", "TIME"]
    with pytest.raises(ValueError, match="increase"):
        NetPresentValue(start="2001-01-01", times=["2001-01-01"], wop=1.0)
    with pytest.raises(ValueError, match="responses"):
        npv.evaluate(np.ones((4, 2)))
    with pytest.raises(ValueError, match="nonzero price"):
        NetPresentValue(start="2000-01-01", times=["2001-01-01"], wop=0.0)
    with pytest.raises(ValueError, match="finite"):
        NetPresentValue(start="2000-01-01", times=["2001-01-01"], wop=float("nan"))
    # A start that is not the simulation's: TIME at the first report disagrees.
    with pytest.raises(ValueError, match="not the simulation's start"):
        npv.evaluate(np.array([[1.0], [1.0], [300.0]]))


def test_enopt_improves_the_toy_npv_and_journals_each_simulation_once(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    run = optimization.optimize(_config(tmp_path, "toy"))
    history = layout.read_history(run.run_dir)
    assert history[-1]["objective"] < history[0]["objective"]  # NPV up (scaled by -1)
    journal = layout.read_evaluations(run.run_dir)
    assert len(journal) == toy.runs  # every simulation journaled, none twice
    assert len({row["key"] for row in journal}) == len(journal)
    assert [row["member_id"] for row in journal] == list(range(len(journal)))
    assert not (run.run_dir / layout.SIMULATIONS).exists()  # the toy needs no folders


def test_a_resume_after_a_failed_simulation_replays_the_journal(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    whole = optimization.optimize(_config(tmp_path, "whole"))
    runs_whole = toy.runs

    toy.runs, toy.fail_at = 0, 12  # mid-iteration: the second iteration's ensemble
    with pytest.raises(WorkflowError, match=r"1 of \d+ simulations failed"):
        optimization.optimize(_config(tmp_path, "broken"))
    failed_rows = [r for r in layout.read_evaluations(tmp_path / "runs" / "broken")
                   if r["value"] is None]  # fmt: skip
    assert len(failed_rows) == 1
    resumed = optimization.resume(tmp_path / "runs" / "broken")

    # Only the failed simulation runs again: the rest of its batch is in the journal.
    assert toy.runs == runs_whole + 1
    assert [r["x"] for r in layout.read_history(resumed.run_dir)] == [
        r["x"] for r in layout.read_history(whole.run_dir)
    ]
    np.testing.assert_array_equal(resumed.result.x, whole.result.x)


def test_an_objective_simulates_repeated_controls_once(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    from apex.execution.local import SerialExecutor

    objective = SimulatedObjective(
        NetPresentValue(start="2000-01-01", times=TIMES, wop=1.0, wwi=1.0),
        ToyReservoir(), SerialExecutor(retries=0), tmp_path, [("inj", slice(0, 2))],
    )  # fmt: skip
    controls = np.array([[100.0, 100.0, 50.0], [100.0, 100.0, 60.0]])
    first = objective(controls, np.zeros(3, dtype=np.int64))
    assert toy.runs == 2
    assert first[0] == first[1]
    again = SimulatedObjective(
        NetPresentValue(start="2000-01-01", times=TIMES, wop=1.0, wwi=1.0),
        ToyReservoir(), SerialExecutor(retries=0), tmp_path, [("inj", slice(0, 2))],
    )  # fmt: skip
    np.testing.assert_array_equal(again(controls, np.zeros(3, dtype=np.int64)), first)
    assert toy.runs == 2
    assert evaluation_key(controls[:, 0], 0) != evaluation_key(controls[:, 0], 1)
    assert evaluation_key(controls[:, 0], 0) != evaluation_key(controls[:, 0], 0, "other deck")


def test_a_torn_journal_line_is_dropped_and_the_file_repaired(tmp_path: Path) -> None:
    layout.append_evaluations(tmp_path, [{"key": "a", "member_id": 0, "value": 1.0}])
    with (tmp_path / layout.EVALUATIONS).open("a") as handle:
        handle.write('{"key": "b", "memb')
    assert [r["key"] for r in layout.read_evaluations(tmp_path, repair=True)] == ["a"]
    layout.append_evaluations(tmp_path, [{"key": "c", "member_id": 1, "value": 2.0}])
    assert [r["key"] for r in layout.read_evaluations(tmp_path)] == ["a", "c"]


def test_robust_optimization_takes_the_realizations_from_the_model_file(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    np.savez(tmp_path / "models.npz", perm=np.array([[0.5, 1.0, 1.5]]))
    extra = '[model]\nrealizations_file = "models.npz"\n'
    config = optimization.load_optimization_config(_config(tmp_path, "robust", 2, extra))
    _, problem, _ = optimization.build(config)
    assert problem.n_realizations == 3
    run = optimization.optimize(_config(tmp_path, "robust", 2, extra))
    realizations = {r["realization"] for r in layout.read_evaluations(run.run_dir)}
    assert realizations == {0, 1, 2}


def test_simulated_objectives_are_refused_without_their_parts(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    path = _config(tmp_path, "no_sim")
    path.write_text(path.read_text().split("[simulator]")[0], encoding="utf-8")
    with pytest.raises(ConfigError, match="add \\[simulator\\]"):
        optimization.build(optimization.load_optimization_config(path))
    analytic = _config(tmp_path, "analytic")
    analytic.write_text(
        analytic.read_text()
        .replace('name = "npv"', 'name = "rosenbrock"', 1)
        .replace("options = { start", "#options = { start"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="analytic"):
        optimization.build(optimization.load_optimization_config(analytic))
    robust = _config(tmp_path, "robust", extra="")
    robust.write_text(
        robust.read_text().replace("[[controls.variables]]", "[controls]\nrealizations = 2\n\n"
                                   "[[controls.variables]]"),
        encoding="utf-8",
    )  # fmt: skip
    with pytest.raises(ConfigError, match="realizations_file"):
        optimization.build(optimization.load_optimization_config(robust))
    # A reported issue (#250): line search and trust region need the objective's exact
    # derivatives, which a simulated objective lacks; refused at build, so validation, plans
    # and studies see it before anything runs.
    for name in ("line_search", "trust_region"):
        derivative = _config(tmp_path, name)
        derivative.write_text(
            derivative.read_text().replace(
                'name = "enopt"\noptions = { ne = 6, maxiter = 6, alpha = 0.1, resample = 1 }',
                f'name = "{name}"\noptions = {{ maxiter = 3 }}',
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"exact gradient.*computed from simulations"):
            optimization.build(optimization.load_optimization_config(derivative))


SCALED = """
[[parameterizations]]
name = "control_scaling"
options = { variable = "u", lower = 10.0, upper = 400.0, output = "inj" }
"""


def _scaled_config(tmp_path: Path, name: str, upper: float = 400.0) -> Path:
    """Normalized controls ``u`` in [0, 1], mapped onto the toy's pressures [10, 400]."""
    text = CONFIG.format(name=name, times=json.dumps(TIMES), maxiter=4, extra="")
    text = text.replace('name = "inj"\ninitial = [300.0, 250.0]\nlower = 10.0\nupper = 400.0\n'
                        "variance = 100.0",
                        'name = "u"\ninitial = [0.74, 0.62]\nlower = 0.0\nupper = 1.0\n'
                        "variance = 0.0007")  # fmt: skip
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}.toml"
    path.write_text(text + SCALED.replace("400.0", str(upper)), encoding="utf-8")
    return path


def test_controls_reach_the_simulator_through_parameterizations(
    tmp_path: Path, toy: type[ToyReservoir], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optimizer moves u in [0, 1]; the simulator reads inj = 10 + 390 u
    (2026-10-04), and the optimization improves the toy's NPV as before.
    """
    from apex.workflows.parameterization import PropertyMap

    seen: list[np.ndarray] = []
    given: list[np.ndarray] = []
    original, mapping = ToyReservoir.run, PropertyMap.member

    def recording(self: ToyReservoir, member: MemberInput, keys: Sequence[DataKey]) -> np.ndarray:
        assert set(member.state) == {"inj"}  # the controls' block u is consumed
        seen.append(np.asarray(member.state["inj"]).copy())
        return original(self, member, keys)

    def mapped(self: PropertyMap, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        given.append(np.asarray(state["u"]).copy())  # u, as the optimizer sees it
        return mapping(self, state)

    monkeypatch.setattr(ToyReservoir, "run", recording)
    monkeypatch.setattr(PropertyMap, "member", mapped)
    run = optimization.optimize(_scaled_config(tmp_path, "scaled"))
    history = layout.read_history(run.run_dir)
    assert history[-1]["objective"] < history[0]["objective"]
    # The first mapping is the build's check of the initial controls (no simulation).
    np.testing.assert_array_equal(given[0], [0.74, 0.62])
    given = given[1:]
    assert len(seen) == len(given) > 0
    assert np.all((np.array(given) >= 0.0) & (np.array(given) <= 1.0))
    np.testing.assert_allclose(np.array(seen), 10.0 + 390.0 * np.array(given), rtol=1e-12)


def test_the_journal_key_knows_the_parameterizations(tmp_path: Path) -> None:
    """Changing a parameterization changes what a control means: no stale reuse."""
    one = optimization.load_optimization_config(_scaled_config(tmp_path, "a"))
    two = optimization.load_optimization_config(_scaled_config(tmp_path, "b", upper=500.0))
    assert optimization.evaluation_context(one) != optimization.evaluation_context(two)


def test_parameterizations_are_checked(tmp_path: Path, toy: type[ToyReservoir]) -> None:
    text = _scaled_config(tmp_path, "bad").read_text().replace('variable = "u"', 'variable = "x"')
    (tmp_path / "bad.toml").write_text(text)
    with pytest.raises(ConfigError, match="neither the state"):
        optimization.optimize(tmp_path / "bad.toml")


CHAINED = """
[[parameterizations]]
name = "control_scaling"
options = { variable = "u", lower = 10.0, upper = 400.0, output = "knots" }

[[parameterizations]]
name = "control_schedule"
options = { variable = "knots", knots = [0, 1], n_steps = 3, output = "inj" }
"""


def _chained_config(tmp_path: Path, name: str, maxiter: int = 4, knots: str = "[0, 1]") -> Path:
    """Two normalized values -> two pressures -> a schedule of three (the toy's inj)."""
    path = _scaled_config(tmp_path, name)
    text = path.read_text().split("[[parameterizations]]")[0]
    text = text.replace("maxiter = 4", f"maxiter = {maxiter}")
    path.write_text(text + CHAINED.replace("knots = [0, 1]", f"knots = {knots}"))
    return path


def test_chained_maps_and_a_resume_reuse_the_journal(
    tmp_path: Path, toy: type[ToyReservoir], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scaling then a held schedule reach the simulator; a resume of a stopped run
    simulates nothing twice and ends as the uninterrupted one.
    """
    seen: list[np.ndarray] = []
    original = ToyReservoir.run

    def recording(self: ToyReservoir, member: MemberInput, keys: Sequence[DataKey]) -> np.ndarray:
        seen.append(np.asarray(member.state["inj"]).copy())
        return original(self, member, keys)

    monkeypatch.setattr(ToyReservoir, "run", recording)
    whole = optimization.optimize(_chained_config(tmp_path / "w", "whole"))
    assert all(s.shape == (3,) and s[1] == s[2] for s in seen)  # [p0, p1, p1]
    runs_whole = toy.runs
    toy.runs = 0
    part = optimization.optimize(_chained_config(tmp_path / "p", "part"), max_iterations=2)
    resumed = optimization.resume(part.run_dir)
    assert toy.runs == runs_whole  # each evaluation simulated once, across the resume
    assert layout.read_history(resumed.run_dir) == layout.read_history(whole.run_dir)


def test_a_block_that_does_not_fit_its_map_is_refused_before_simulating(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    path = _chained_config(tmp_path, "misfit", knots="[0, 1, 2]")  # 3 knots, 2 values
    with pytest.raises(ConfigError, match="has shape"):
        optimization.optimize(path)
    assert toy.runs == 0


def test_an_unchanged_map_keeps_the_journal_key(tmp_path: Path) -> None:
    one = optimization.load_optimization_config(_scaled_config(tmp_path / "x", "same"))
    two = optimization.load_optimization_config(_scaled_config(tmp_path / "y", "same"))
    assert optimization.evaluation_context(one) == optimization.evaluation_context(two)


def test_npv_counts_from_a_time_after_the_start() -> None:
    """from_time counts only the steps after it, discounted from it (a forecast
    after a history); without it, npv is PET's (the parity test)."""
    npv = NetPresentValue(
        start="2000-01-01", times=["2001-01-01", "2002-01-01", "2003-01-01"], wop=10.0,
        wwi=2.0, disc=0.1, from_time="2001-01-01",
    )  # fmt: skip
    keys = [str(k) for k in npv.response_keys()]
    assert keys[:3] == ["FOPT@2001-01-01", "FOPT@2002-01-01", "FOPT@2003-01-01"]
    responses = np.array([[100.0], [150.0], [230.0], [40.0], [55.0], [80.0], [366.0]])
    # Steps after 2001-01-01: (150 - 100, 55 - 40) and (230 - 150, 80 - 55); days 365, 730.
    cash = [(50 * 10 - 15 * 2) / 1.1 ** (365 / 365), (80 * 10 - 25 * 2) / 1.1 ** (730 / 365)]
    np.testing.assert_allclose(npv.evaluate(responses), [sum(cash)], rtol=1e-14)
    with pytest.raises(ValueError, match="one of the report times"):
        NetPresentValue(start="2000-01-01", times=["2001-01-01"], wop=1.0, from_time="2000-06-01")
    with pytest.raises(ValueError, match="before the last"):
        NetPresentValue(start="2000-01-01", times=["2001-01-01"], wop=1.0, from_time="2001-01-01")


def test_ren_npv_counts_injection_from_the_same_time() -> None:
    """ren_npv's well-by-well injection costs follow from_time as its field part."""
    from apex.algorithms.optimization import RenewableNetPresentValue

    ren = RenewableNetPresentValue(
        start="2000-01-01", times=["2001-01-01", "2002-01-01", "2003-01-01"], wop=10.0,
        wwi=2.0, wrenwi=1.0, injectors=["I1"], renewable_injectors=["R1"], disc=0.1,
        from_time="2001-01-01",
    )  # fmt: skip
    values = {"FOPT": [100.0, 150.0, 230.0], "TIME": [366.0], "WWIT:I1": [40.0, 55.0, 80.0],
              "WWIT:R1": [7.0, 9.0, 14.0]}  # fmt: skip
    responses = np.array([[v] for k in ("FOPT", "TIME", "WWIT:I1", "WWIT:R1") for v in values[k]])
    assert [str(k).split("@")[0] for k in ren.response_keys()][3:5] == ["TIME", "WWIT:I1"]
    # Steps after 2001-01-01 only, discounted from it (365 and 730 days).
    cash = [(50 * 10 - 15 * 2 - 2 * 1) / 1.1, (80 * 10 - 25 * 2 - 5 * 1) / 1.1**2]
    np.testing.assert_allclose(ren.evaluate(responses), [sum(cash)], rtol=1e-14)


@pytest.mark.parametrize("optimizer", ["ga", "pso"])
def test_population_optimizers_improve_the_toy_npv_in_journaled_batches(
    tmp_path: Path, toy: type[ToyReservoir], optimizer: str
) -> None:
    """On a simulated objective: each generation is one batch of simulations, every
    simulation journaled once, the evaluations within the plan's bound, and the NPV up."""
    path = _config(tmp_path, optimizer, maxiter=4)
    text = path.read_text().replace(
        'name = "enopt"\noptions = { ne = 6, maxiter = 4, alpha = 0.1, resample = 1 }',
        f'name = "{optimizer}"\noptions = {{ pop_size = 6, maxiter = 4, transform = true }}',
    )
    assert f'name = "{optimizer}"' in text
    path.write_text(text)
    plan = optimization.plan_optimization(optimization.load_optimization_config(path))
    run = optimization.optimize(path)
    history = layout.read_history(run.run_dir)
    assert history[-1]["objective"] <= history[0]["objective"]  # NPV up (scaled by -1)
    journal = layout.read_evaluations(run.run_dir)
    assert len(journal) == toy.runs  # every simulation journaled, none twice
    assert plan.max_evaluations is not None
    assert len(journal) <= plan.max_evaluations


MULTI = """
[optimization]
name = "{name}"
seed = 5

[[controls.variables]]
name = "inj"
initial = [100.0, 100.0]
lower = 1.0
upper = 400.0

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

[simulator]
name = "test_toy_reservoir"

[execution]
name = "serial"
"""


def test_npv_against_water_injected_is_a_monotone_front_and_resumes_from_its_journal(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    """On a simulated objective: one simulation per member gives both objectives
    (the journal holds ``values``), the front trades NPV for water (more water, more NPV),
    and a resumed run equals the uninterrupted one, simulating nothing twice."""

    def config(name: str) -> Path:
        path = tmp_path / f"{name}.toml"
        path.write_text(MULTI.format(name=name, times=json.dumps(TIMES), last=TIMES[-1]))
        return path

    whole = optimization.optimize(config("whole"))
    runs_whole = toy.runs
    journal = layout.read_evaluations(whole.run_dir)
    assert len(journal) == runs_whole  # one simulation per member, both objectives from it
    assert all(len(row["values"]) == 2 and "value" not in row for row in journal)
    front = layout.read_pareto(whole.run_dir)
    assert front is not None
    assert front["objective_names"] == ["npv", "water_injected"]
    F = np.array(front["objectives"])
    order = np.argsort(F[:, 1])  # by water injected
    assert np.all(np.diff(F[order, 0]) <= 1e-12)  # more water: more NPV (-NPV down)

    toy.runs = 0
    first = optimization.optimize(config("split"), max_iterations=2)
    assert not first.finished
    resumed = optimization.resume(first.run_dir)
    assert resumed.finished
    assert toy.runs == runs_whole  # the resume simulated only what was not journaled
    assert layout.read_pareto(resumed.run_dir) == front
    result = layout.read_result(resumed.run_dir)
    assert result["objective_names"] == ["npv", "water_injected"]
    assert len(result["objectives"]) == 2
    # A finished run read back keeps every objective, as the live result (review nit).
    from apex.workflows.optimization import finished_result

    assert finished_result(resumed.run_dir).fun.shape == np.shape(resumed.result.fun) == (2, 1)


def test_a_list_of_objectives_is_checked(tmp_path: Path, toy: type[ToyReservoir]) -> None:
    from apex.core.errors import ConfigError

    text = MULTI.format(name="bad", times=json.dumps(TIMES), last=TIMES[-1])
    both = tmp_path / "both.toml"
    both.write_text(text.replace("[[objectives]]\nname = \"npv\"", "[objective]\nname = \"npv\"\n"
                                 "[[objectives]]\nname = \"npv\"", 1))  # fmt: skip
    with pytest.raises(ConfigError, match="either"):
        optimization.load_optimization_config(both)
    same = tmp_path / "same.toml"
    same.write_text(text.replace('label = "water_injected"', 'label = "npv"'))
    with pytest.raises(ConfigError, match="repeat"):
        optimization.load_optimization_config(same)
    single = tmp_path / "single.toml"
    single.write_text(text.replace('name = "nsga2"', 'name = "ga"'))
    with pytest.raises(ConfigError, match=r"optimizes one objective|several"):
        optimization.optimize(single)
    mixed = tmp_path / "mixed.toml"
    analytic = text.replace('name = "field_total"', 'name = "rosenbrock"')
    mixed.write_text(analytic.replace(f'options = {{ vector = "FWIT", time = "{TIMES[-1]}", '
                                      'scale = 0.001 }', "options = {}"))  # fmt: skip
    with pytest.raises(ConfigError, match="mixes simulated and analytic"):
        optimization.optimize(mixed)


def test_the_journal_key_knows_every_listed_objective(tmp_path: Path) -> None:
    """Review finding (2026-10-06): with ``[[objectives]]`` the journal's context hash left
    the objectives out, so a resume after their options (or files) changed reused stale
    values. Each entry's options are part of it now."""
    text = MULTI.format(name="key", times=json.dumps(TIMES), last=TIMES[-1])
    one, two = tmp_path / "one.toml", tmp_path / "two.toml"
    one.write_text(text)
    two.write_text(text.replace("wop = 1.0", "wop = 2.0"))
    first = optimization.load_optimization_config(one)
    second = optimization.load_optimization_config(two)
    assert optimization.evaluation_context(first) != optimization.evaluation_context(second)


def test_a_statistic_shares_its_simulations_and_names_itself(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    """NPV's mean and its standard deviation over 3 realizations from the same
    simulations (one per member and realization); unnamed entries show their statistic."""
    np.savez(tmp_path / "models.npz", perm=np.array([[0.5, 1.0, 1.5]]))
    text = MULTI.format(name="robust", times=json.dumps(TIMES), last=TIMES[-1])
    text = text.replace('name = "field_total"\nlabel = "water_injected"',
                        'name = "npv"\nstatistic = "std"')  # fmt: skip
    text = text.replace(f'options = {{ vector = "FWIT", time = "{TIMES[-1]}", scale = 0.001 }}',
                        "options = " + text.split("options = ", 2)[1].split("\n")[0])  # fmt: skip
    path = tmp_path / "robust.toml"
    path.write_text(text + '\n[model]\nrealizations_file = "models.npz"\n')
    run = optimization.optimize(path)
    rows = layout.read_evaluations(run.run_dir)
    assert toy.runs == len(rows)  # each member and realization simulated once, for both
    assert len({(r["member_id"], r["realization"]) for r in rows}) == len(rows)
    front = layout.read_pareto(run.run_dir)
    assert front is not None
    assert front["objective_names"] == ["npv", "npv_std"]
    assert front["objective_statistics"] == ["mean", "std"]
    assert layout.read_result(run.run_dir)["objective_statistics"] == ["mean", "std"]


def test_a_statistic_leaves_the_journal_key_alone(tmp_path: Path) -> None:
    """A statistic decides what is ranked, not what is simulated, so it is not in
    the journal's context: an explicit mean, and another statistic, keep the key."""
    text = MULTI.format(name="key", times=json.dumps(TIMES), last=TIMES[-1])
    keys = []
    for name, extra in (("plain", ""), ("mean", 'statistic = "mean"\n'),
                        ("p90", 'statistic = "p90"\n')):  # fmt: skip
        path = tmp_path / f"{name}.toml"
        path.write_text(text.replace('label = "water_injected"\n',
                                     f'label = "water_injected"\n{extra}'))  # fmt: skip
        keys.append(optimization.evaluation_context(optimization.load_optimization_config(path)))
    assert keys[0] == keys[1] == keys[2]


def test_what_a_run_would_refuse_is_refused_at_validation(
    tmp_path: Path, toy: type[ToyReservoir]
) -> None:
    """Review findings: std with one realization, and MOEA/D's constraints without
    EPF, passed validation and failed when the run started."""
    from apex.core.errors import ConfigError

    text = MULTI.format(name="std1", times=json.dumps(TIMES), last=TIMES[-1])
    path = tmp_path / "std1.toml"
    path.write_text(text.replace('label = "water_injected"\n',
                                 'label = "water_injected"\nstatistic = "std"\n'))  # fmt: skip
    with pytest.raises(ConfigError, match="std needs at least two realizations"):
        optimization.plan_optimization(optimization.load_optimization_config(path))
    worst = tmp_path / "worst1.toml"
    worst.write_text(text.replace('label = "water_injected"\n',
                                  'label = "water_injected"\nstatistic = "worst"\n'))  # fmt: skip
    plan = optimization.plan_optimization(optimization.load_optimization_config(worst))
    assert any("equal the mean" in w for w in plan.warnings)
    bnh = tmp_path / "bnh.toml"
    bnh.write_text(
        '[optimization]\nname = "bnh"\nseed = 1\n\n[[controls.variables]]\nname = "x1"\n'
        'initial = 2.5\nlower = 0.0\nupper = 5.0\n\n[[controls.variables]]\nname = "x2"\n'
        "initial = 1.5\nlower = 0.0\nupper = 3.0\n\n[objective]\n"
        'name = "bnh"\n\n[optimizer]\nname = "moead"\noptions = { maxiter = 2 }\n'
    )
    with pytest.raises(ConfigError, match="external penalty only"):
        optimization.plan_optimization(optimization.load_optimization_config(bnh))


@pytest.mark.parametrize("name", ["nsga3", "moead"])
def test_nsga3_and_moead_resume_through_the_service(tmp_path: Path, name: str) -> None:
    """A fresh build (which sizes the directions) then the checkpoint files: the same front."""

    def config(run: str) -> Path:
        path = tmp_path / f"{run}.toml"
        path.write_text(
            f'[optimization]\nname = "{run}"\nseed = 3\n\n[[controls.variables]]\nname = "x"\n'
            "initial = [0.5, 0.5, 0.5, 0.5, 0.5]\nlower = 0.0\nupper = 1.0\n\n[objective]\n"
            f'name = "dtlz2"\n\n[optimizer]\nname = "{name}"\n'
            "options = { maxiter = 4, transform = true }\n"
        )
        return path

    whole = optimization.optimize(config("whole"))
    first = optimization.optimize(config("split"), max_iterations=2)
    assert not first.finished
    resumed = optimization.resume(first.run_dir)
    assert resumed.finished
    assert layout.read_pareto(resumed.run_dir) == layout.read_pareto(whole.run_dir)
