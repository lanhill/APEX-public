"""The surrogate workflow: ``apex surrogate build|check`` end to end.

The cases are linear (the ``linear`` simulator, ``y = G x``) and the surrogate a test-only
affine least-squares map (registered here, never shipped), which reproduces them exactly:
every gate error is round-off, an independent reference for the whole path (design, runs,
snapshots, fit, domain, gate, files, summary).
"""

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from typer.testing import CliRunner

from apex.algorithms.surrogates.designs import repeated_columns
from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.core.plugins import surrogate_models
from apex.core.surrogates import SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotReference, SnapshotSet, SurrogateCheckSummary
from apex.services import composition, surrogates
from apex.services.summaries import summarize
from apex.storage.surrogate_directory import (
    JOURNAL,
    MemberRecord,
    SurrogateDirectory,
    load_snapshots,
)
from apex.storage.surrogate_files import read_surrogate
from apex.workflows.history_matching import RandomStreams

N_STATE = 3
TIMES = ("1", "2", "3", "4")
DATATYPES = ("WOPR:P1", "WBHP:P1")


class _Affine:
    """An affine least-squares map from the stacked inputs to the outputs (test only)."""

    version = "1"

    def __init__(self, ridge: float = 0.0, fail: bool = False) -> None:
        self.ridge = ridge
        self.fail = fail
        self.keys: tuple[DataKey, ...] = ()
        self.names: tuple[str, ...] = ()
        self.weights = np.zeros((0, 0))

    def _stack(self, inputs: SurrogateInputs) -> NDArray[np.float64]:
        return np.concatenate([np.asarray(inputs[n], dtype=np.float64) for n in self.names])

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        if self.fail:
            raise ValueError("asked to fail")
        self.names = snapshots.layout.names
        self.keys = snapshots.keys
        design = np.vstack([self._stack(snapshots.inputs), np.ones((1, snapshots.n_runs))])
        gram = design @ design.T + self.ridge * np.eye(design.shape[0])
        self.weights = np.ascontiguousarray(np.linalg.solve(gram, design @ snapshots.outputs.T).T)

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        stacked = self._stack(inputs)
        return stacked, np.zeros(stacked.shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        full = self.weights[:, :-1] @ self._stack(inputs) + self.weights[:, -1:]
        return np.ascontiguousarray(full[rows])

    def to_arrays(self) -> SurrogateParts:
        own = {"names": list(self.names), "ridge": self.ridge,
               "keys": [[k.time, k.datatype, k.index] for k in self.keys]}  # fmt: skip
        return own, {"weights": self.weights}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls(ridge=float(manifest["ridge"]))  # type: ignore[arg-type]
        model.names = tuple(manifest["names"])  # type: ignore[arg-type]
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        model.weights = np.asarray(arrays["weights"], dtype=np.float64)
        return model


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register("test_affine_build")(_Affine)
    try:
        yield
    finally:
        surrogate_models.remove("test_affine_build")


def write_case(
    folder: Path,
    *,
    simulator_extra: str = "",
    execution: str = 'name = "serial"',
    times: Sequence[str] = TIMES,
    bounds: str = "lower = -4.0\nupper = 4.0",
    prior: str = 'generator = { name = "gaussian", options = { mean = 0.0, std = 1.0 } }',
) -> Path:
    """Write a linear case (``y = G x``, 2 datatypes x 4 times) and return its configuration."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    forward = rng.standard_normal((len(DATATYPES) * len(times), N_STATE))
    np.savez(folder / "forward.npz", matrix=forward)
    keys = [(t, d) for d in DATATYPES for t in times]
    observed = forward @ np.array([0.5, -1.0, 0.8])
    rows = ["time,datatype,value,variance"] + [
        f"{t},{d},{float(v)!r},0.01" for (t, d), v in zip(keys, observed, strict=True)
    ]
    (folder / "observations.csv").write_text("\n".join(rows) + "\n")
    outputs = ", ".join(f'["{t}", "{d}"]' for t, d in keys)
    config = folder / "case.toml"
    config.write_text(f"""
[run]
name = "case"
seed = 1

[[state.variables]]
name = "x"
size = {N_STATE}
{bounds}

[prior]
ensemble_size = 20
{prior}

[observations]
file = "observations.csv"

[simulator]
name = "linear"

[simulator.options]
forward_file = "forward.npz"
variables = ["x"]
outputs = [{outputs}]
{simulator_extra}

[execution]
{execution}

[method]
name = "esmda"
options = {{ inflation = [1.0] }}

[case]
kind = "real"
""")
    return config


def write_build(
    folder: Path,
    *,
    name: str = "lin",
    design: str = 'plan = "prior"\ntraining = 20\nheld_out = 5',
    options: str = "{}",
    extra: str = "",
) -> Path:
    """Write a build of the case in ``folder``."""
    path = folder / f"{name}.toml"
    path.write_text(f"""
[surrogate]
name = "{name}"
model = "test_affine_build"
options = {options}
seed = 7

[case]
config_file = "case.toml"

[design]
{design}
{extra}
""")
    return path


def _gate_errors(path: Path) -> list[float]:
    gate = json.loads((path / "gate.json").read_text())
    return [float(d["error"]) for d in gate["datatypes"]]


def test_build_check_and_summary_through_the_cli(tmp_path: Path) -> None:
    write_case(tmp_path)
    build = write_build(tmp_path)
    runner = CliRunner()

    planned = runner.invoke(app, ["validate", str(build)])
    assert planned.exit_code == 0, planned.output
    assert "simulator runs: 25" in planned.output
    assert "gated by sigma ratio" in planned.output

    built = runner.invoke(app, ["surrogate", "build", str(build), "--no-progress"])
    assert built.exit_code == 0, built.output
    assert "gate passed" in built.output
    out = tmp_path / "runs" / "lin"
    for name in ("config.json", "state.json", "design.npz", "snapshots.npz", JOURNAL,
                 "gate.json", "summary.json", "provenance.json", "events.jsonl",
                 "model/surrogate.json", "model/surrogate.npz"):  # fmt: skip
        assert (out / name).exists(), name
    assert max(_gate_errors(out)) < 1e-10  # an exact surrogate: round-off
    stored = read_surrogate(out / "model" / "surrogate.json")
    manifest = stored.manifest
    assert manifest.plugin == "test_affine_build"
    assert manifest.provenance.plan == "prior"
    assert manifest.provenance.seed == 7
    assert manifest.provenance.simulator == "linear"
    assert manifest.gate is not None
    assert manifest.gate.passed
    assert len(manifest.gate.held_out) == 5
    snapshots = load_snapshots(out / "snapshots.npz")
    assert manifest.snapshots is not None
    assert manifest.snapshots.digest == snapshots.digest()
    assert snapshots.n_runs == 25
    assert manifest.domain.n_points == 25

    checked = runner.invoke(
        app,
        ["surrogate", "check", str(out / "model" / "surrogate.json"), "--against",
         str(tmp_path / "case.toml"), "--runs", "6", "--seed", "3", "--min-inside", "0",
         "--no-progress"],
    )  # fmt: skip
    assert checked.exit_code == 0, checked.output
    assert "check passed" in checked.output
    [check] = [p for p in (out / "checks").iterdir() if p.is_dir()]
    record = json.loads((check / "check.json").read_text())
    assert record["status"] == "complete"
    assert record["identity"] == stored.identity
    # Members outside the domain are reported, not gated: the model refuses them.
    assert len(record["gate"]["held_out"]) + len(record["outside"]) == 6
    assert read_surrogate(out / "model" / "surrogate.json").identity == stored.identity
    last = surrogates.last_check(out / "model" / "surrogate.json")
    assert last is not None
    assert last.passed
    assert last.succeeded == 6
    assert last.inside + len(last.outside) == 6
    assert not last.trained_on

    [summary] = summarize([tmp_path / "runs"])
    assert summary.workflow == "surrogate"
    assert summary.status == "complete"
    assert summary.forecasts == 25
    assert summary.selected_iteration == 1
    assert [s.n_members for s in summary.steps] == [20, 5]
    for datatype in DATATYPES:
        error = summary.value("gate_error", datatype, 1)
        assert error is not None
        assert error.mean < 1e-10
    passed = summary.value("gate_passed", "all", 1)
    assert passed is not None
    assert passed.mean == 1.0
    latest = summary.value("check_passed", "latest", 1)
    assert latest is not None
    assert latest.mean == 1.0
    assert any(note.startswith("latest check") for note in summary.notes)


def test_an_interrupted_build_resumes_to_identical_model_files(tmp_path: Path) -> None:
    write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole"))
    path = write_build(tmp_path, name="parts")

    first = surrogates.build(path, max_runs=7)
    assert not first.complete
    assert first.summary.pending == 18
    again = surrogates.resume_build(first.path, max_runs=0)
    assert again.summary.pending == 18
    finished = surrogates.build(path)

    assert finished.complete
    assert finished.summary.identity == whole.summary.identity
    for name in ("surrogate.json", "surrogate.npz"):
        assert (finished.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()
    lines = (finished.path / JOURNAL).read_text().splitlines()
    assert sorted(json.loads(line)["member"] for line in lines) == list(range(25))


def test_a_changed_plugin_option_refits_on_the_runs_made(tmp_path: Path) -> None:
    write_case(tmp_path)
    first = surrogates.build(write_build(tmp_path))
    journal = (first.path / JOURNAL).read_text()

    again = surrogates.build(write_build(tmp_path, options="{ ridge = 10.0 }"))

    assert (again.path / JOURNAL).read_text() == journal  # no new run
    assert again.summary.identity != first.summary.identity
    assert json.loads((again.path / "config.json").read_text())["surrogate"]["options"] == {
        "ridge": 10.0
    }
    with pytest.raises(ConfigError, match=r"of other runs \(design differ\)"):
        surrogates.build(
            write_build(tmp_path, design='plan = "prior"\ntraining = 21\nheld_out = 5')
        )


def test_failed_runs_are_recorded_and_too_many_stop_the_build(tmp_path: Path) -> None:
    write_case(tmp_path / "one", simulator_extra="fail_members = [6]")
    result = surrogates.build(write_build(tmp_path / "one"))
    assert result.complete
    assert result.summary.failed == [6]
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.failed == (6,)
    assert 6 not in snapshots.run_ids
    journal = SurrogateDirectory(result.path).journal()
    assert not journal[6].ok

    write_case(tmp_path / "many", simulator_extra="fail_members = [0, 1]")
    with pytest.raises(WorkflowError, match="2 of the 5 held-out runs failed"):
        surrogates.build(write_build(tmp_path / "many"))


def test_the_process_executor_builds_the_same_model(tmp_path: Path) -> None:
    write_case(tmp_path / "serial")
    write_case(tmp_path / "process", execution='name = "process"\noptions = { workers = 3 }')

    serial = surrogates.build(write_build(tmp_path / "serial"))
    process = surrogates.build(write_build(tmp_path / "process"))

    assert serial.summary.identity == process.summary.identity


def test_a_latin_hypercube_stays_in_its_box(tmp_path: Path) -> None:
    write_case(tmp_path)
    result = surrogates.build(
        write_build(tmp_path, design='plan = "lhs"\ntraining = 12\nheld_out = 4\n'
                    "box = { x = [-0.5, 1.5] }")
    )  # fmt: skip
    inputs = load_snapshots(result.path / "snapshots.npz").inputs["x"]
    assert inputs.min() >= -0.5
    assert inputs.max() <= 1.5
    # One point per stratum per coordinate, in each set (the held-out runs, the training).
    design = SurrogateDirectory(result.path).load_design()
    assert design is not None
    strata = np.floor((design.training["x"] + 0.5) / 2.0 * 12)
    assert all(sorted(row) == list(range(12)) for row in strata.astype(int))


def test_a_design_from_a_stored_ensemble(tmp_path: Path) -> None:
    write_case(tmp_path)
    ensemble = np.random.default_rng(5).uniform(-1, 1, (N_STATE, 30))
    np.savez(tmp_path / "ensemble.npz", x=ensemble)
    result = surrogates.build(
        write_build(tmp_path, design='plan = "file"\ntraining = 20\nheld_out = 6\n'
                    'ensemble_file = "ensemble.npz"')
    )  # fmt: skip
    snapshots = load_snapshots(result.path / "snapshots.npz")
    # Its ids are the ensemble's columns.
    np.testing.assert_array_equal(snapshots.inputs["x"], ensemble[:, list(snapshots.run_ids)])
    assert result.summary.simulator_runs == 26


def test_a_design_from_a_runs_forecasts_runs_nothing(tmp_path: Path) -> None:
    config = write_case(tmp_path)
    composition.start(config)
    run_dir = tmp_path / "runs" / "case"
    result = surrogates.build(
        write_build(tmp_path, name="free", design='plan = "run"\ntraining = 25\nheld_out = 10\n'
                    f'run_dir = "{run_dir}"')
    )  # fmt: skip

    assert result.complete
    assert result.summary.simulator_runs == 0
    assert not (result.path / JOURNAL).exists()
    assert max(_gate_errors(result.path)) < 1e-10
    stored = read_surrogate(result.path / "model" / "surrogate.json")
    assert stored.manifest.provenance.source == str(run_dir)
    # The run's posterior is a check's other source: its members run by the simulator, but
    # only those the design left out (35 of the 40 pooled members are the model's).
    check = surrogates.check_surrogate(
        result.path / "model" / "surrogate.json", run_dir, runs=2, seed=1, min_inside=0.0
    )
    assert check.summary.source == "posterior"
    assert check.summary.gate is not None
    assert len(check.summary.trained_on) == 18
    assert len(check.summary.gate.held_out) + len(check.summary.outside) == 2


def test_what_a_build_refuses(tmp_path: Path) -> None:
    write_case(tmp_path / "unbounded", bounds="")
    with pytest.raises(ConfigError, match="bounded box"):
        surrogates.build(
            write_build(tmp_path / "unbounded", design='plan = "lhs"\ntraining = 5\nheld_out = 2')
        )
    write_case(tmp_path)
    with pytest.raises(ConfigError, match="lambda: Extra inputs"):
        surrogates.plan_build(
            surrogates.load_surrogate_config(write_build(tmp_path, options="{ lambda = 1 }"))
        )
    optimization = write_build(tmp_path, name="optb").read_text().replace("case.toml", "opt.toml")
    (tmp_path / "opt_build.toml").write_text(optimization)
    (tmp_path / "opt.toml").write_text("[optimization]\nname='o'\n[optimizer]\nname='enopt'\n")
    with pytest.raises(ConfigError, match="phase 4"):
        surrogates.plan_build(surrogates.load_surrogate_config(tmp_path / "opt_build.toml"))
    labelled = write_case(tmp_path / "labels", times=("t0", "t1"))
    assert labelled.exists()
    with pytest.raises(ConfigError, match="step indices or ISO dates"):
        surrogates.plan_build(surrogates.load_surrogate_config(write_build(tmp_path / "labels")))
    with pytest.raises(ConfigError, match="needs run_dir"):
        surrogates.load_surrogate_config(
            write_build(tmp_path, name="norun", design='plan = "run"\ntraining = 5\nheld_out = 2')
        )


# ------------------------------------------------------------- review findings (phase 2)


def _stored_prior_case(folder: Path) -> Path:
    """A case whose prior is a stored ensemble of 40 members (``file``)."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "prior.npz", x=np.random.default_rng(3).standard_normal((N_STATE, 40)))
    return write_case(
        folder, prior='generator = { name = "file", options = { ensemble_file = "prior.npz" } }'
    )


def test_a_stored_prior_cannot_be_a_prior_design(tmp_path: Path) -> None:
    """Review blocker: plan "prior" on a file prior gave held-out runs equal to training
    runs (the generator returns its first columns whatever the stream)."""
    _stored_prior_case(tmp_path)
    with pytest.raises(ConfigError, match="use plan = 'file'"):
        surrogates.build(write_build(tmp_path))


def test_a_check_never_gates_on_the_models_own_runs(tmp_path: Path) -> None:
    """Review blocker: a check's members could be the model's training runs."""
    # A stored prior, every column of it in the design: whatever a check draws from that
    # prior is among the model's runs.
    _stored_prior_case(tmp_path)
    built = surrogates.build(
        write_build(tmp_path, design='plan = "file"\ntraining = 30\nheld_out = 10\n'
                    'ensemble_file = "prior.npz"')
    )  # fmt: skip
    model = built.path / "model" / "surrogate.json"
    with pytest.raises(ConfigError, match=r"only 0 of the 6 members .* new to the model"):
        surrogates.check_surrogate(model, tmp_path / "case.toml", runs=6)

    # A run's posterior, partly in a "run" design: those members are left out and reported.
    config = write_case(tmp_path / "run")
    composition.start(config)
    run_dir = tmp_path / "run" / "runs" / "case"
    free = surrogates.build(
        write_build(tmp_path / "run", name="free",
                    design=f'plan = "run"\ntraining = 20\nheld_out = 8\nrun_dir = "{run_dir}"')
    )  # fmt: skip
    trained = load_snapshots(free.path / "snapshots.npz")
    posterior = composition.load_results(run_dir)
    iterate = posterior.iterate(posterior.posterior_iteration)
    repeats = repeated_columns(dict(iterate.state), dict(trained.inputs))
    expected = sorted({int(iterate.member_ids[i]) for i, _ in repeats})
    assert expected  # the design took some of the posterior's members

    check = surrogates.check_surrogate(
        free.path / "model" / "surrogate.json", run_dir, runs=4, seed=2, min_inside=0.0
    )

    assert check.summary.trained_on == expected
    design = SurrogateDirectory(check.path).load_design()
    assert design is not None
    assert not set(design.held_out_ids) & set(expected)
    assert not repeated_columns(dict(design.held_out), dict(trained.inputs))
    with pytest.raises(ConfigError, match="new to the model"):
        surrogates.check_surrogate(
            free.path / "model" / "surrogate.json", run_dir, runs=20 - len(expected) + 1
        )


def test_a_check_fails_on_too_few_runs_inside_or_succeeded(tmp_path: Path) -> None:
    """Review major: a check could pass on almost nothing."""
    write_case(tmp_path, simulator_extra="fail_members = [22]")
    built = surrogates.build(
        write_build(tmp_path, design='plan = "lhs"\ntraining = 12\nheld_out = 4\n'
                    "box = { x = [-0.3, 0.3] }")
    )  # fmt: skip
    model = built.path / "model" / "surrogate.json"

    # Prior draws (std 1) mostly lie outside a box of +-0.3: the gate passes on the few
    # inside, the check does not.
    narrow = surrogates.check_surrogate(model, tmp_path / "case.toml", runs=8, seed=1)
    summary = narrow.summary
    assert summary.gate is not None
    assert summary.inside < 0.8 * summary.succeeded
    assert not summary.passed
    assert any("inside the model's domain" in note for note in summary.notes)
    last = surrogates.last_check(model)
    assert last is not None
    assert not last.passed
    result = CliRunner().invoke(
        app, ["surrogate", "check", str(model), "--against", str(tmp_path / "case.toml"),
              "--runs", "8", "--seed", "1", "--no-progress"],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "check FAILED" in result.output

    # Member 22 of a check's draws fails: 22 of 23 succeed, short of every run.
    with pytest.raises(WorkflowError, match="1 of the 23 check runs failed"):
        surrogates.check_surrogate(model, tmp_path / "case.toml", runs=23, seed=4,
                                   min_success_fraction=1.0, min_inside=0.0)  # fmt: skip


def test_a_torn_journal_line_is_ended_before_new_records(tmp_path: Path) -> None:
    directory = SurrogateDirectory(tmp_path)
    directory.record([MemberRecord(0, True, 1.0)])
    with (tmp_path / JOURNAL).open("a") as handle:
        handle.write('{"member": 1, "ok": tr')  # interrupted mid-line

    directory.record([MemberRecord(2, False, None, "boom"), MemberRecord(3, True, 0.5)])

    journal = directory.journal()
    assert sorted(journal) == [0, 2, 3]
    assert journal[2].message == "boom"


def test_a_failed_refit_keeps_the_last_models_configuration(tmp_path: Path) -> None:
    write_case(tmp_path)
    first = surrogates.build(write_build(tmp_path))
    saved = (first.path / "config.json").read_text()

    with pytest.raises(ConfigError, match="asked to fail"):
        surrogates.build(write_build(tmp_path, options="{ fail = true }"))

    assert (first.path / "config.json").read_text() == saved
    assert read_surrogate(first.path / "model" / "surrogate.json").identity == (
        first.summary.identity
    )


@pytest.mark.parametrize(
    ("path", "valid"),
    [("snapshots.npz", True), ("../snapshots.npz", True), (None, True),
     ("../../snapshots.npz", False), ("/etc/snapshots.npz", False), ("a/b.npz", False),
     ("..", False), ("../", False)],
)  # fmt: skip
def test_a_snapshot_reference_names_a_file_beside_or_above_the_manifest(
    path: str | None, valid: bool
) -> None:
    if valid:
        assert SnapshotReference(path=path, digest="x").path == path
    else:
        with pytest.raises(ValidationError, match="snapshot path"):
            SnapshotReference(path=path, digest="x")


def test_a_copied_case_folder_is_the_same_case(tmp_path: Path) -> None:
    """Review minor: folder options (a deck's case_dir) counted by path, not content."""
    for name in ("a", "b"):
        (tmp_path / name / "include").mkdir(parents=True)
        (tmp_path / name / "include" / "grid.inc").write_text("COORD\n1 2 3 /\n")
    a = surrogates._with_contents({"case_dir": str(tmp_path / "a" / "include")})
    b = surrogates._with_contents({"case_dir": str(tmp_path / "b" / "include")})
    assert a == b
    (tmp_path / "b" / "include" / "grid.inc").write_text("COORD\n1 2 30 /\n")
    assert surrogates._with_contents({"case_dir": str(tmp_path / "b" / "include")}) != a


def test_a_runs_members_with_forecasts_not_finite_are_recorded_as_failed(tmp_path: Path) -> None:
    config = write_case(tmp_path)
    composition.start(config)
    run_dir = tmp_path / "runs" / "case"
    path = run_dir / "checkpoints" / "iter_1.npz"
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["predicted"] = np.array(arrays["predicted"])
    arrays["predicted"][2, 3] = np.nan  # pooled column 20 + 3 (iteration 0 holds 20)
    np.savez(path, **arrays)

    result = surrogates.build(
        write_build(tmp_path, name="free",
                    design=f'plan = "run"\ntraining = 25\nheld_out = 10\nrun_dir = "{run_dir}"')
    )  # fmt: skip

    assert result.summary.failed == [23]
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.failed == (23,)
    assert 23 not in snapshots.run_ids


def test_draw_prior_is_the_prior_ensemble_drawn_from_the_prior_stream(tmp_path: Path) -> None:
    run = composition.build_run(composition.load_config(write_case(tmp_path)))
    expected = composition.draw_prior_ensemble(
        run.config, run.workflow.layout, run.config.prior.ensemble_size,
        RandomStreams.from_seed(run.config.run.seed)["prior"],
    )  # fmt: skip

    np.testing.assert_array_equal(composition.draw_prior(run), expected)


def test_a_check_failing_on_too_few_runs_becomes_the_last_check(tmp_path: Path) -> None:
    """Re-review major: a check that failed on min_success_fraction raised before it was
    recorded, so an earlier passing check stayed the model's last."""
    write_case(tmp_path, simulator_extra="fail_members = [22]")
    built = surrogates.build(write_build(tmp_path))
    model = built.path / "model" / "surrogate.json"
    passing = surrogates.check_surrogate(model, tmp_path / "case.toml", runs=6, seed=3,
                                         min_inside=0.0)  # fmt: skip
    assert passing.summary.passed
    last = surrogates.last_check(model)
    assert last is not None
    assert last.id == passing.summary.id

    with pytest.raises(WorkflowError, match="1 of the 23 check runs failed"):
        surrogates.check_surrogate(model, tmp_path / "case.toml", runs=23, seed=4,
                                   min_success_fraction=1.0, min_inside=0.0)  # fmt: skip

    last = surrogates.last_check(model)
    assert last is not None
    assert last.id != passing.summary.id
    assert not last.passed
    assert last.sequence == 2
    assert "1 of the 23 check runs failed" in last.notes[0]
    [summary] = summarize([built.path])
    latest = summary.value("check_passed", "latest", 1)
    assert latest is not None
    assert latest.mean == 0.0


def test_a_plugins_fit_error_names_the_plugin_and_its_options(tmp_path: Path) -> None:
    write_case(tmp_path)
    with pytest.raises(ConfigError, match=r"'test_affine_build' \(ridge = 'x'\) cannot be fitted: "
                       r".*TypeError"):  # fmt: skip
        surrogates.build(write_build(tmp_path, options='{ ridge = "x" }'))


def test_checks_completing_at_once_get_distinct_order_numbers(tmp_path: Path) -> None:
    """Final review: two checks of one model, each holding only its own folder's lock, could
    take the same sequence."""
    write_case(tmp_path)
    built = surrogates.build(write_build(tmp_path))
    model = built.path / "model" / "surrogate.json"
    base = SurrogateCheckSummary(
        id="", identity="i", model=str(model), against="a", source="prior",
        status="complete", runs=1, created="2026-11-01T00:00:00+00:00",
    )  # fmt: skip
    n = 16
    barrier = threading.Barrier(n)
    numbers: list[int] = []

    def complete(index: int) -> None:
        folder = surrogates.checks_folder(model) / f"c{index:02d}"
        folder.mkdir(parents=True)
        barrier.wait()
        recorded = surrogates.record_check(
            model, folder, base.model_copy(update={"id": f"c{index:02d}"})
        )
        numbers.append(recorded.sequence)

    threads = [threading.Thread(target=complete, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(numbers) == list(range(1, n + 1))
    last = surrogates.last_check(model)
    assert last is not None
    assert last.sequence == n
    # A check completed again takes a new number: the first one numbered (whichever thread
    # was numbered last keeps n, being the latest already).
    first = min(surrogates._complete_checks(model), key=lambda c: c.sequence)
    again = surrogates.record_check(
        model, surrogates.checks_folder(model) / first.id, base.model_copy(update={"id": first.id})
    )
    assert again.sequence == n + 1


def test_a_plugins_file_options_are_resolved_against_the_configuration(tmp_path: Path) -> None:
    """Options named *_file, *_dir, *_path name paths relative to the build's file."""
    folder = tmp_path / "builds"
    folder.mkdir()
    path = write_build(folder, options='{ case_file = "data/case.npz", ridge = 0.5, label = "x" }')
    config = surrogates.load_surrogate_config(path)
    assert config.surrogate.options["case_file"] == str((folder / "data/case.npz").resolve())
    assert config.surrogate.options["ridge"] == 0.5
    assert config.surrogate.options["label"] == "x"


def test_a_file_an_option_names_counts_by_its_content(tmp_path: Path) -> None:
    """An edited file behind an option refits the model; the same content does not."""
    (tmp_path / "case.npz").write_bytes(b"first")
    config = surrogates.load_surrogate_config(
        write_build(tmp_path, options='{ case_file = "case.npz" }')
    )
    before = surrogates._fit_fingerprint(config)
    assert surrogates._fit_fingerprint(config) == before
    (tmp_path / "case.npz").write_bytes(b"second")
    assert surrogates._fit_fingerprint(config) != before


def test_file_options_are_written_relative_to_a_model_and_read_back(tmp_path: Path) -> None:
    """Relative to the model's folder, links not followed; older absolute paths still read."""
    model = tmp_path / "runs" / "lin" / "model"
    target = tmp_path / "elsewhere" / "case.npz"
    target.parent.mkdir()
    target.write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "case.npz").symlink_to(target)
    options = {"case_file": str(tmp_path / "data" / "case.npz"), "ridge": 0.5, "label": "x"}
    stored = surrogates.options_relative_to(options, model)
    assert stored == {"case_file": "../../../data/case.npz", "ridge": 0.5, "label": "x"}
    assert surrogates.options_resolved_in(stored, model) == options
    old = {"case_file": str(target)}
    assert surrogates.options_resolved_in(old, model) == old


def test_a_complete_build_without_its_model_files_refits_them(tmp_path: Path) -> None:
    """Deleted model files are fitted again on the runs made, to the same bytes (2026-11-02)."""
    write_case(tmp_path)
    first = surrogates.build(write_build(tmp_path))
    journal = (first.path / JOURNAL).read_text()
    model = {
        n: (first.path / "model" / n).read_bytes() for n in ("surrogate.json", "surrogate.npz")
    }
    for name in model:
        (first.path / "model" / name).unlink()
    (first.path / "summary.json").unlink()

    again = surrogates.build(write_build(tmp_path))

    assert (again.path / JOURNAL).read_text() == journal  # no new run
    assert again.summary.identity == first.summary.identity
    assert (again.path / "summary.json").is_file()
    for name, data in model.items():
        assert (again.path / "model" / name).read_bytes() == data
