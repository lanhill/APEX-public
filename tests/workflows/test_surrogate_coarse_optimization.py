"""Coarse-corrected surrogates in optimization.

The simulator is a toy reservoir in this process (each injector produces ``20 perm sqrt(x)``
of oil a year and injects ``x`` of water); its coarse deck, the same simulator with the
option ``coarse = [a, b]``, answers ``a y + b`` at every rate key, an exact two-fidelity
pair. The surrogate is a test-only coarse-corrected model, ``y = s_k y_coarse + c_k`` fitted
key by key on the training runs, which reproduces the fine simulator from the coarse one to
round-off: an independent reference for the whole path (the build and the check with a
coarse stack, the journal's three sources and contexts, the coarse objective, the budget,
the simulator standing in for a failed coarse run, resume).
"""

import datetime as dt
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.core.plugins import simulators, surrogate_models
from apex.core.simulation import MemberInput
from apex.core.surrogates import SurrogateCapabilities, SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet
from apex.services import optimization, surrogates
from apex.simulation.deck import FidelityLevels
from apex.storage import optimization_directory as layout
from apex.workflows.simulated_objective import BrokenStackError
from apex.workflows.surrogate_optimization import STANDS_IN, SurrogateObjective

TIMES = [f"{2001 + k}-01-01" for k in range(4)]


class TwoFidelityReservoir:
    """Field totals of the toy reservoir; ``coarse = [a, b]`` makes its coarse deck.

    Counts its runs by fidelity; a coarse run of a member id in ``fail_coarse`` fails, and
    one in ``crash_coarse`` raises an error no executor catches (a killed process).
    """

    runs: ClassVar[dict[str, int]] = {"fine": 0, "coarse": 0}
    fail_all_coarse: ClassVar[bool] = False
    fail_coarse: ClassVar[set[int]] = set()
    crash_coarse: ClassVar[set[int]] = set()

    upscaled: ClassVar[list[int]] = []

    def __init__(
        self, coarse: Sequence[float] | None = None, upscale: Mapping[str, object] | None = None
    ) -> None:
        self.coarse = None if coarse is None else (float(coarse[0]), float(coarse[1]))
        # The Eclipse adapters' upscale: COARSEN records for every member, whatever its id.
        self.levels = FidelityLevels(None, Path("FINE.mako"), upscale)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        if self.levels.upscale is not None:
            _, level_values = self.levels.for_member(member)
            assert level_values["coarse"], "every member is coarsened"
            type(self).upscaled.append(member.member_id)
            return TwoFidelityReservoir((0.8, 5.0)).run(member, keys)
        fidelity = "fine" if self.coarse is None else "coarse"
        type(self).runs[fidelity] += 1
        if self.coarse is not None and member.member_id in type(self).crash_coarse:
            raise KeyboardInterrupt(f"killed at member {member.member_id}")
        if self.coarse is not None and (
            type(self).fail_all_coarse or member.member_id in type(self).fail_coarse
        ):
            raise SimulationError(f"the coarse deck failed at member {member.member_id}")
        inj = np.asarray(member.state["inj"], dtype=float)
        totals = {"FOPT": 20.0 * float(np.sum(np.sqrt(inj))), "FWIT": float(np.sum(inj))}
        start = dt.date(2000, 1, 1)
        values = []
        for k in keys:
            if k.datatype == "TIME":
                values.append(float((dt.date.fromisoformat(k.time) - start).days))
                continue
            value = totals[k.datatype] * (TIMES.index(k.time) + 1)
            if self.coarse is not None:
                value = self.coarse[0] * value + self.coarse[1]
            values.append(value)
        return np.array(values, dtype=float)


class CoarseLinear:
    """``y_k = s_k y_coarse,k + c_k``, least squares per key (a coarse-corrected test model)."""

    capabilities = SurrogateCapabilities(
        inputs=frozenset({"state", "controls", "coarse_responses"}),
        outputs=frozenset({"responses"}),
    )
    version = "1"

    def __init__(self) -> None:
        self.keys: tuple[DataKey, ...] = ()
        self.slope = np.zeros((0, 2))

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        assert snapshots.coarse_outputs is not None
        self.keys = snapshots.keys
        rows = []
        for coarse, fine in zip(snapshots.coarse_outputs, snapshots.outputs, strict=True):
            if np.ptp(coarse) == 0.0:  # a TIME key: the same in every run
                rows.append([0.0, float(np.mean(fine))])
                continue
            design = np.column_stack([coarse, np.ones_like(coarse)])
            rows.append(list(np.linalg.lstsq(design, fine, rcond=None)[0]))
        self.slope = np.ascontiguousarray(rows, dtype=np.float64)

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.asarray(inputs["inj"], dtype=np.float64)
        return np.ascontiguousarray(x), np.zeros(x.shape[1])

    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = np.asarray(coarse).shape[1]
        return np.zeros((1, n)), np.zeros(n)

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise ValueError("a coarse-corrected surrogate needs the coarse forecast")

    def predict_corrected(
        self, inputs: SurrogateInputs, coarse: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        full = self.slope[:, :1] * np.asarray(coarse, dtype=np.float64) + self.slope[:, 1:]
        return np.ascontiguousarray(full[rows])

    def to_arrays(self) -> SurrogateParts:
        return {"keys": [[k.time, k.datatype, k.index] for k in self.keys]}, {"slope": self.slope}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls()
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.slope = np.asarray(arrays["slope"], dtype=np.float64)
        return model


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    simulators.register("test_two_fidelity_reservoir")(TwoFidelityReservoir)
    surrogate_models.register("test_coarse_linear")(CoarseLinear)
    TwoFidelityReservoir.runs = {"fine": 0, "coarse": 0}
    TwoFidelityReservoir.fail_coarse = set()
    TwoFidelityReservoir.fail_all_coarse = False
    TwoFidelityReservoir.crash_coarse = set()
    TwoFidelityReservoir.upscaled = []
    surrogates.fitted_model.cache_clear()
    try:
        yield
    finally:
        simulators._factories.pop("test_two_fidelity_reservoir")
        surrogate_models.remove("test_coarse_linear")
        surrogates.fitted_model.cache_clear()


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
name = "test_two_fidelity_reservoir"

[execution]
name = "serial"
options = {{ retries = 0 }}
{extra}
"""

BUILD = """
[surrogate]
name = "coarse_build"
model = "test_coarse_linear"
seed = 3

[case]
config_file = "case.toml"
outputs = "responses"

[design]
plan = "lhs"
training = 8
held_out = 4

[coarse]
options = {{ {coarse_options} }}
"""


def _case(folder: Path, name: str, *, maxiter: int = 5, extra: str = "") -> Path:
    path = folder / f"{name}.toml"
    path.write_text(
        CASE.format(name=name, times=json.dumps(TIMES), maxiter=maxiter, extra=extra),
        encoding="utf-8",
    )
    return path


#: An upscale table (uniform: the same records for every member).
UPSCALE = 'upscale = { method = "uniform", dim = [8, 8], max_trunc = 0.5 }'


def _build(folder: Path, coarse: str = "[0.8, 5.0]", *, options: str | None = None) -> Path:
    _case(folder, "case")
    path = folder / "build.toml"
    path.write_text(BUILD.format(coarse_options=options or f"coarse = {coarse}"), "utf-8")
    result = surrogates.build(path)
    assert result.summary.status == "complete"
    return result.path / "model" / "surrogate.json"


def _with_surrogate(model: Path, **options: object) -> str:
    lines = [f'model_file = "{model}"'] + [f"{k} = {json.dumps(v)}" for k, v in options.items()]
    return "[surrogate]\n" + "\n".join(lines)


def test_a_build_and_a_check_over_the_controls_run_the_coarse_stack(tmp_path: Path) -> None:
    model = _build(tmp_path)
    # Every design run on both decks (12 each).
    assert TwoFidelityReservoir.runs == {"fine": 12, "coarse": 12}
    gate = json.loads((model.parent.parent / "gate.json").read_text())
    assert gate["passed"]
    assert all(d["error"] < 1e-9 for d in gate["datatypes"])
    manifest = json.loads(model.read_text())
    assert manifest["coarse"]["options"] == {"coarse": [0.8, 5.0]}
    assert manifest["provenance"]["coarse"]["simulator"] == "test_two_fidelity_reservoir"
    check = surrogates.check_surrogate(model, tmp_path / "case.toml", runs=4)
    assert check.summary.passed
    assert TwoFidelityReservoir.runs == {"fine": 16, "coarse": 16}


def test_an_optimizations_coarse_stack_refuses_fidelity_levels(tmp_path: Path) -> None:
    _case(tmp_path, "case")
    path = tmp_path / "build.toml"
    options = "coarse = [0.8, 5.0], levels = [{ first = 0, size = 4 }]"
    path.write_text(BUILD.format(coarse_options=options), "utf-8")
    with pytest.raises(ConfigError, match=r"cannot take fidelity levels.*options\.upscale"):
        surrogates.build(path)
    assert TwoFidelityReservoir.runs == {"fine": 0, "coarse": 0}


def test_an_upscaled_coarse_stack_coarsens_every_run_and_resumes_byte_identically(
    tmp_path: Path,
) -> None:
    """Amended: the fine deck coarsened by ``upscale`` is the coarse
    stack; every coarse run is coarsened, at any journal id, and a resume is byte-identical."""
    model = _build(tmp_path, options=UPSCALE)
    assert len(TwoFidelityReservoir.upscaled) == 12  # the build's design, on the coarse stack
    manifest = json.loads(model.read_text())
    assert "upscale" in manifest["coarse"]["options"]
    extra = _with_surrogate(model, check_every=2)
    TwoFidelityReservoir.upscaled = []
    whole = optimization.optimize(_case(tmp_path, "whole", extra=extra))
    rows = layout.read_evaluations(whole.run_dir)
    coarse_ids = [r["member_id"] for r in rows if r.get("source") == "coarse"]
    assert coarse_ids
    # Every coarse row ran coarsened (ids well past any level a build would have set), and
    # no simulator row did.
    assert TwoFidelityReservoir.upscaled == coarse_ids
    assert max(coarse_ids) > 12
    TwoFidelityReservoir.crash_coarse = {coarse_ids[len(coarse_ids) // 2]}
    with pytest.raises(KeyboardInterrupt):
        optimization.optimize(_case(tmp_path, "part", extra=extra))
    TwoFidelityReservoir.crash_coarse = set()
    resumed = optimization.resume(tmp_path / "runs" / "part")
    for name in ("evaluations.jsonl", "history.jsonl"):
        assert (resumed.run_dir / name).read_bytes() == (whole.run_dir / name).read_bytes()
    assert layout.read_result(resumed.run_dir) == layout.read_result(whole.run_dir)


def test_an_upscaled_coarse_stack_is_a_new_context(tmp_path: Path) -> None:
    """Another upscale is another coarse stack: the model built with one is refused with the
    other (the stack's options digest)."""
    model = _build(tmp_path, options=UPSCALE)
    manifest = json.loads(model.read_text())
    manifest["coarse"]["options"]["upscale"]["max_trunc"] = 0.25
    model.write_text(json.dumps(manifest), "utf-8")
    surrogates.fitted_model.cache_clear()
    with pytest.raises(
        ConfigError, match=r"another simulator stack \(coarse\.options_digest differ\)"
    ):
        optimization.optimize(_case(tmp_path, "other", extra=_with_surrogate(model)))


@pytest.mark.parametrize(
    ("options", "simulator_options", "match"),
    [
        (
            'upscale = { method = "haar", dim = [8, 8], max_trunc = 0.5, state = "inj" }',
            "",
            r"follows 'inj', which is not a property the deck gets from the \[model\] arrays",
        ),
        (UPSCALE, UPSCALE, "plain stack is coarsened too"),
    ],
)
def test_an_upscaled_coarse_stack_is_checked(
    tmp_path: Path, options: str, simulator_options: str, match: str
) -> None:
    extra = f"options = {{ {simulator_options} }}" if simulator_options else ""
    path = tmp_path / "case.toml"
    path.write_text(
        CASE.format(name="case", times=json.dumps(TIMES), maxiter=5, extra="").replace(
            'name = "test_two_fidelity_reservoir"\n',
            f'name = "test_two_fidelity_reservoir"\n{extra}\n',
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build.toml"
    build.write_text(BUILD.format(coarse_options=options), "utf-8")
    with pytest.raises(ConfigError, match=match):
        surrogates.build(build)
    assert TwoFidelityReservoir.runs == {"fine": 0, "coarse": 0}


def test_an_optimization_runs_the_coarse_stack_before_the_surrogate(tmp_path: Path) -> None:
    model = _build(tmp_path)
    plain = optimization.optimize(_case(tmp_path, "plain"))
    TwoFidelityReservoir.runs = {"fine": 0, "coarse": 0}
    run = optimization.optimize(
        _case(tmp_path, "fast", extra=_with_surrogate(model, check_every=2))
    )
    rows = layout.read_evaluations(run.run_dir)
    by_source = {s: [r for r in rows if r.get("source") == s] for s in ("coarse", "surrogate",
                                                                         "simulator")}  # fmt: skip
    assert sum(len(v) for v in by_source.values()) == len(rows)
    assert by_source["coarse"]
    assert by_source["surrogate"]
    assert by_source["simulator"]
    # One coarse run per surrogate evaluation (checks' included), none per simulator one.
    assert len(by_source["coarse"]) == len(by_source["surrogate"])
    assert TwoFidelityReservoir.runs == {
        "fine": len(by_source["simulator"]), "coarse": len(by_source["coarse"]),
    }  # fmt: skip
    # Three contexts: no key is shared between sources.
    keys = {s: {r["key"] for r in v} for s, v in by_source.items()}
    assert not keys["coarse"] & keys["surrogate"]
    assert not keys["coarse"] & keys["simulator"]
    # Member ids count the journal's rows, whichever source added them.
    assert [r["member_id"] for r in rows] == list(range(len(rows)))
    history = layout.read_history(run.run_dir)
    steps = [r for r in history if r["kind"] == "step"]
    assert steps
    assert all(isinstance(r["coarse_objective"], float) for r in steps)
    assert all("coarse_objective" not in r for r in history if r["kind"] != "step")
    # The coarse objective is the coarse deck's npv, not the fine one's.
    assert all(r["coarse_objective"] != pytest.approx(r["objective"]) for r in steps)
    result = layout.read_result(run.run_dir)
    assert result["surrogate"]["coarse_evaluations"] == len(by_source["coarse"])
    assert result["surrogate"]["coarse_failed"] == 0
    assert result["surrogate"]["coarse_stand_ins"] == 0
    assert result["objective"] == pytest.approx(result["surrogate_objective"], rel=1e-10)
    np.testing.assert_allclose(run.result.x, plain.result.x, rtol=1e-8)
    # apex summary shows the coarse stack.
    from apex.services.summaries import OptimizationSummarizer

    summary = OptimizationSummarizer().summarize(run.run_dir)
    assert any(n.startswith("coarse stack: ") for n in summary.notes)
    assert any(m.metric == "coarse_objective" for m in summary.metrics)


def test_a_later_run_reuses_the_coarse_rows(tmp_path: Path) -> None:
    """Coarse rows are journaled under the plain context and the coarse stack's: a resumed
    or repeated batch never runs them again."""
    model = _build(tmp_path)
    path = _case(tmp_path, "fast", extra=_with_surrogate(model))
    run = optimization.optimize(path)
    config = optimization.load_optimization_config(path)
    _, problem, _ = optimization.build(config)
    objective = problem.objective
    assert isinstance(objective, SurrogateObjective)
    TwoFidelityReservoir.runs = {"fine": 0, "coarse": 0}
    x = np.asarray(layout.read_result(run.run_dir)["x"]).reshape(-1, 1)
    objective(x, np.zeros(1, dtype=np.int64))
    assert TwoFidelityReservoir.runs == {"fine": 0, "coarse": 0}


@pytest.mark.parametrize("failing", [False, True])
def test_a_resume_after_a_killed_run_is_byte_identical(tmp_path: Path, failing: bool) -> None:
    model = _build(tmp_path)
    extra = _with_surrogate(model, check_every=2)
    if failing:
        # The start's coarse run fails: the simulator stands in for it.
        TwoFidelityReservoir.fail_coarse = {0}
    whole = optimization.optimize(_case(tmp_path, "whole", extra=extra))
    rows = layout.read_evaluations(whole.run_dir)
    # A killed process mid-iteration: after a coarse batch was journaled, before its
    # surrogate evaluations (the middle coarse member of the whole run).
    coarse_ids = [r["member_id"] for r in rows if r.get("source") == "coarse"]
    TwoFidelityReservoir.crash_coarse = {coarse_ids[len(coarse_ids) // 2]}
    with pytest.raises(KeyboardInterrupt):
        optimization.optimize(_case(tmp_path, "part", extra=extra))
    TwoFidelityReservoir.crash_coarse = set()
    part = tmp_path / "runs" / "part"
    assert layout.read_history(part)  # killed after a checkpoint
    resumed = optimization.resume(part)
    for name in ("evaluations.jsonl", "history.jsonl"):
        assert (resumed.run_dir / name).read_bytes() == (whole.run_dir / name).read_bytes()
    result, expected = layout.read_result(resumed.run_dir), layout.read_result(whole.run_dir)
    assert result == expected
    if failing:
        assert result["surrogate"]["coarse_failed"] == 1
        assert result["surrogate"]["coarse_stand_ins"] == 1
        failed = [r for r in rows if r.get("source") == "coarse" and r["value"] is None]
        assert [r["member_id"] for r in failed] == [0]
        standing = [r for r in rows if r.get(STANDS_IN) is not None]
        assert [(r["source"], r[STANDS_IN]) for r in standing] == [("simulator", "coarse")]
        # The failed coarse run was not run again for the start's repeated evaluations.
        assert sum(1 for r in rows if r["member_id"] == 0) == 1


def test_the_plan_counts_the_coarse_runs(tmp_path: Path) -> None:
    model = _build(tmp_path)
    config = optimization.load_optimization_config(
        _case(tmp_path, "planned", extra=_with_surrogate(model, check_every=2))
    )
    plan = optimization.plan_optimization(config)
    assert plan.max_evaluations is not None
    capped = min(plan.max_evaluations, 50_000)
    checks = 5 // 2
    # Checks, the optimum and the last step's ensemble (ne = 6) by the
    # simulator; a coarse run per evaluation and per check; the simulator standing in for
    # every coarse run that may fail.
    assert plan.max_simulator_runs == (checks + 1 + 6) + 2 * (capped + checks)


def test_a_bound_on_evaluations_bounds_the_coarse_runs(tmp_path: Path) -> None:
    model = _build(tmp_path)
    from apex.core.errors import WorkflowError

    with pytest.raises(WorkflowError, match="coarse stack's runs would exceed"):
        optimization.optimize(
            _case(tmp_path, "bounded", extra=_with_surrogate(model, max_evaluations=3))
        )


def test_a_broken_coarse_deck_stops_the_run_until_it_is_fixed(tmp_path: Path) -> None:
    """A batch whose new coarse runs all fail stops the run, naming the coarse stack and the
    failures (the user, 2026-11-21); those rows are marked for retry, so a resume runs them
    again: an unchanged (still broken) stack stops again, a fixed one continues."""
    model = _build(tmp_path)
    TwoFidelityReservoir.fail_all_coarse = True
    config = _case(tmp_path, "broken", extra=_with_surrogate(model))
    with pytest.raises(BrokenStackError) as stopped:
        optimization.optimize(config)
    message = str(stopped.value)
    assert "the coarse stack (test_two_fidelity_reservoir)" in message
    assert "the coarse deck failed at member" in message
    run_dir = tmp_path / "runs" / "broken"
    rows = layout.read_evaluations(run_dir)
    # The start's single coarse failure falls back (remembered); the gradient's batch stops.
    assert [r[STANDS_IN] for r in rows if r.get(STANDS_IN)] == ["coarse"]
    retried = [r for r in rows if r.get("retry")]
    assert len(retried) >= 2
    assert all(r["source"] == "coarse" and r["value"] is None for r in retried)
    # Unchanged: the same batch fails again, deterministically.
    with pytest.raises(BrokenStackError):
        optimization.resume(run_dir)
    again = layout.read_evaluations(run_dir)
    assert sum(1 for r in again if r.get("retry")) == 2 * len(retried)
    # Fixed: the failures are run again and the run finishes.
    TwoFidelityReservoir.fail_all_coarse = False
    resumed = optimization.resume(run_dir)
    assert resumed.finished
    final = layout.read_evaluations(run_dir)
    succeeded = {r["key"] for r in final if r.get("source") == "coarse" and r["value"] is not None}
    assert {r["key"] for r in retried} <= succeeded
    assert layout.read_result(run_dir)["surrogate"]["coarse_stand_ins"] == 1


def test_several_objectives_report_each_coarse_objective(tmp_path: Path) -> None:
    """With several objectives the history holds ``coarse_objective:<name>`` per objective
    (each objective's statistic over the realizations), and ``apex summary`` shows each."""
    from types import SimpleNamespace

    from apex.services.summaries import OptimizationSummarizer

    objective = SurrogateObjective.__new__(SurrogateObjective)
    values = {("a", 0): (1.0, 10.0), ("a", 1): (3.0, 30.0)}
    objective.coarse = SimpleNamespace(  # type: ignore[assignment]
        keys_of=lambda controls, reals: [("a", int(r)) for r in reals],
        journal=SimpleNamespace(values=values),
    )
    objective.vector = True
    objective.n_realizations = 2
    objective.objective_names = ("npv", "co2")
    objective.objective_statistics = ("mean", "mean")
    extra = objective._coarse_objective(np.array([1.0, 2.0]))
    assert extra == {"coarse_objective:npv": 2.0, "coarse_objective:co2": 20.0}
    del values[("a", 1)]
    assert objective._coarse_objective(np.array([1.0, 2.0])) == {
        "coarse_objective:npv": None, "coarse_objective:co2": None,
    }  # fmt: skip

    run = optimization.optimize(_case(tmp_path, "plain", maxiter=2))
    history = run.run_dir / "history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "step":
            row.update(extra)
    history.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = OptimizationSummarizer().summarize(run.run_dir)
    labels = {m.label for m in summary.metrics if m.metric == "coarse_objective"}
    assert labels == {"npv", "co2"}
