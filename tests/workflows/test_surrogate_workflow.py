"""The surrogate workflow: ``apex surrogate build|check`` end to end.

The cases are linear (the ``linear`` simulator, ``y = G x``) and the surrogate a test-only
affine least-squares map (registered here, never shipped), which reproduces them exactly:
every gate error is round-off, an independent reference for the whole path (design, runs,
snapshots, fit, domain, gate, files, summary).
"""

import json
import shutil
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError
from typer.testing import CliRunner

from apex.algorithms.surrogates.designs import repeated_columns
from apex.cli.main import app
from apex.core.errors import ConfigError, WorkflowError
from apex.core.plugins import surrogate_models
from apex.core.surrogates import SurrogateInputs, SurrogateModel, SurrogateParts
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


def test_a_stopped_build_ends_between_waves_and_resumes_to_identical_files(
    tmp_path: Path,
) -> None:
    """A job's stop (decision 4) is checked before each wave of member runs
    (four times the executor's parallel width, here one): the build ends as stopped with the
    rest pending, and its resume writes the uninterrupted build's model files."""
    from apex.workflows.surrogate import WAVE_WIDTHS

    write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole"))
    path = write_build(tmp_path, name="parts")
    checks: list[int] = []

    def stopped() -> bool:
        checks.append(1)
        return len(checks) > 1  # the first wave runs, the second is refused

    first = surrogates.build_config(surrogates.load_surrogate_config(path), stopped=stopped)
    assert not first.complete
    assert first.summary.pending == 25 - WAVE_WIDTHS
    assert len(checks) == 2
    finished = surrogates.resume_build(first.path, stopped=lambda: False)

    assert finished.complete
    for name in ("surrogate.json", "surrogate.npz"):
        assert (finished.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()


def test_the_service_places_a_build_by_its_content(tmp_path: Path) -> None:
    """Unplaced (decision 2), a build keeps no folder; its id ignores the
    name and the folder and follows the case's configuration and deck."""
    from apex.services.identity import surrogate_build_id

    write_case(tmp_path)
    path = write_build(tmp_path, name="one")
    unplaced = surrogates.load_surrogate_config(path, place=False)
    assert unplaced.surrogate.output_dir is None
    assert surrogates.load_surrogate_config(path).surrogate.output_dir is not None
    renamed = unplaced.model_copy(update={"surrogate": unplaced.surrogate.model_copy(
        update={"name": "two", "output_dir": tmp_path / "elsewhere"})})  # fmt: skip
    assert surrogate_build_id(renamed) == surrogate_build_id(unplaced)
    other = unplaced.model_copy(update={"surrogate": unplaced.surrogate.model_copy(
        update={"seed": unplaced.surrogate.seed + 1})})  # fmt: skip
    assert surrogate_build_id(other) != surrogate_build_id(unplaced)
    case = unplaced.case.config_file
    before = surrogate_build_id(unplaced)
    case.write_text(case.read_text() + "\n# changed\n")
    assert surrogate_build_id(unplaced) != before


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
    # Read as an optimization since the phase 4 (tests/workflows/
    # test_surrogate_optimization.py), and validated as one.
    with pytest.raises(ConfigError, match="invalid optimization configuration"):
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
    # Options of the wrong type are refused before any run now (test_a_build_refuses_an_
    # unknown_model_or_bad_options_before_any_run); an error inside the fit names the options.
    with pytest.raises(ConfigError, match=r"'test_affine_build' \(fail = True\) cannot be "
                       r"fitted: ValueError: asked to fail"):  # fmt: skip
        surrogates.build(write_build(tmp_path, options="{ fail = true }"))


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


class _FarthestFirst:
    """A test adaptive plan: the candidates of largest norm in the surrogate's coordinates."""

    adaptive = True

    def __init__(self, initial: int = 6, batch: int = 4, candidates: int = 30) -> None:
        self.initial, self.batch, self.candidates = initial, batch, candidates
        self.seen: list[int] = []

    def initial_runs(self, training: int) -> int:
        return self.initial

    def batch_runs(self) -> int:
        return self.batch

    def candidate_runs(self) -> int:
        return self.candidates

    def choose(
        self,
        model: SurrogateModel,
        chosen: Mapping[str, NDArray[np.float64]],
        candidates: Mapping[str, NDArray[np.float64]],
        count: int,
        *,
        keys: Sequence[DataKey],
        variances: NDArray[np.float64] | None,
    ) -> list[int]:
        points, _ = model.coordinates(candidates)
        self.seen.append(next(iter(chosen.values())).shape[1])
        return [int(i) for i in np.argsort(-np.linalg.norm(points, axis=0), kind="stable")[:count]]


@pytest.fixture
def _adaptive() -> Iterator[None]:
    from apex.core.plugins import surrogate_designs

    surrogate_designs.register("test_farthest")(_FarthestFirst)
    try:
        yield
    finally:
        surrogate_designs.remove("test_farthest")


ADAPTIVE = 'plan = "test_farthest"\ntraining = 20\nheld_out = 5\noptions = { batch = 4 }'


@pytest.mark.usefixtures("_adaptive")
def test_an_adaptive_design_runs_its_batches_and_resumes_to_identical_files(
    tmp_path: Path,
) -> None:
    """Held-out and initial runs from the prior, then batches the plan chooses."""
    write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole", design=ADAPTIVE))
    assert whole.complete
    state = json.loads((whole.path / "state.json").read_text())
    batches = state["adaptive"]["batches"]
    assert [len(b) for b in batches] == [4, 4, 4, 2]  # 6 initial runs, then to 20
    assert len({i for b in batches for i in b}) == 14  # never a candidate twice
    assert len((whole.path / JOURNAL).read_text().splitlines()) == 25
    # Interrupted after each few runs, resumed: the same choices, the same model files.
    path = write_build(tmp_path, name="parts", design=ADAPTIVE)
    result = surrogates.build(path, max_runs=7)
    while not result.complete:
        result = surrogates.build(path, max_runs=5)
    for name in ("surrogate.json", "surrogate.npz"):
        assert (result.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()
    assert json.loads((result.path / "state.json").read_text())["adaptive"]["batches"] == batches


@pytest.mark.usefixtures("_adaptive")
def test_an_adaptive_build_stopped_before_its_next_fit_resumes_to_identical_files(
    tmp_path: Path,
) -> None:
    """A job's stop between an adaptive design's batches (before the fit that chooses the
    next one): the build ends stopped, and its resume chooses the same
    batches and writes the same model files."""
    write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole", design=ADAPTIVE))
    config = surrogates.load_surrogate_config(write_build(tmp_path, name="parts", design=ADAPTIVE))
    assert config.surrogate.output_dir is not None
    journal = config.surrogate.output_dir / JOURNAL

    def after_the_initial_runs() -> bool:  # 5 held-out and 6 initial training runs
        return journal.exists() and len(journal.read_text().splitlines()) >= 11

    stopped = surrogates.build_config(config, stopped=after_the_initial_runs)
    assert stopped.summary.status == "stopped"
    assert len(journal.read_text().splitlines()) == 11
    resumed = surrogates.resume_build(stopped.path, stopped=lambda: False)
    assert resumed.complete
    for name in ("surrogate.json", "surrogate.npz"):
        assert (resumed.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()


@pytest.mark.usefixtures("_adaptive")
def test_an_adaptive_designs_runs_follow_its_surrogate(tmp_path: Path) -> None:
    """The plugin chose the runs: a changed plugin option is refused, not refitted."""
    write_case(tmp_path)
    surrogates.build(write_build(tmp_path, design=ADAPTIVE))
    with pytest.raises(ConfigError, match="surrogate"):
        surrogates.build(write_build(tmp_path, design=ADAPTIVE, options="{ ridge = 10.0 }"))


def test_core_plans_keep_their_fingerprints_and_unknown_plans_are_refused(
    tmp_path: Path,
) -> None:
    write_case(tmp_path)
    config = surrogates.load_surrogate_config(write_build(tmp_path))
    case = surrogates._build_case(config)
    parts = surrogates.fingerprint_parts(config, case)
    # The design's digest is over the section as it was before plans had options.
    before = json.loads(config.design.model_dump_json())
    before.pop("options")
    assert parts["design"] == surrogates._digest(before)
    assert "surrogate" not in parts
    with pytest.raises(ConfigError, match=r"unknown design plan 'nope'.*prior"):
        surrogates.build(write_build(tmp_path, design='plan = "nope"\ntraining = 4\nheld_out = 2'))
    with pytest.raises(ConfigError, match="takes no options"):
        surrogates.load_surrogate_config(
            write_build(
                tmp_path,
                design='plan = "prior"\ntraining = 4\nheld_out = 2\noptions = { batch = 2 }',
            )
        )


@pytest.mark.usefixtures("_adaptive")
def test_an_adaptive_build_survives_a_crash_between_its_writes(tmp_path: Path) -> None:
    """The state is the record: a design saved without its state is drawn again, and a design
    saved behind its state is rebuilt from it; either way the same model files."""
    write_case(tmp_path)
    whole = surrogates.build(write_build(tmp_path, name="whole", design=ADAPTIVE))
    reference = (whole.path / "model" / "surrogate.npz").read_bytes()
    # The design saved, its adaptive state never written.
    path = write_build(tmp_path, name="lost", design=ADAPTIVE)
    first = surrogates.build(path, max_runs=3)
    state = json.loads((first.path / "state.json").read_text())
    del state["adaptive"]
    (first.path / "state.json").write_text(json.dumps(state))
    assert surrogates.build(path).complete
    assert (first.path / "model" / "surrogate.npz").read_bytes() == reference
    # Two batches recorded, the design saved with only the first runs (behind its state).
    path = write_build(tmp_path, name="behind", design=ADAPTIVE)
    early = surrogates.build(path, max_runs=8)
    snapshot = (early.path / "design.npz").read_bytes()
    later = surrogates.build(path, max_runs=10)
    assert len(json.loads((later.path / "state.json").read_text())["adaptive"]["batches"]) >= 2
    (later.path / "design.npz").write_bytes(snapshot)
    assert surrogates.build(path).complete
    assert (later.path / "model" / "surrogate.npz").read_bytes() == reference


class _NeedsStd(_FarthestFirst):
    uses = ("predict_std",)


def test_validate_reports_bad_keys_and_an_unknown_plugin_together(tmp_path: Path) -> None:
    # The release tester's case: the linear-Gaussian example's keys are labels (t0, ...),
    # and the model is not installed. `apex validate` stopped at the plugin, so the keys'
    # error only came from `apex surrogate build`; both are said at once now.
    example = Path(__file__).parents[2] / "examples" / "linear_gaussian"
    shutil.copytree(example, tmp_path / "case")
    build = tmp_path / "build.toml"
    build.write_text(
        '[surrogate]\nname = "lg"\nmodel = "no_such_model"\nseed = 1\n'
        '[case]\nconfig_file = "case/config_esmda.toml"\n'
        '[design]\nplan = "prior"\ntraining = 4\nheld_out = 2\n'
    )

    with pytest.raises(ConfigError) as error:
        surrogates.plan_build(surrogates.load_surrogate_config(build))
    assert "unknown surrogate_models plugin 'no_such_model'" in str(error.value)
    assert "step indices or ISO dates" in str(error.value)
    assert "'t0'" in str(error.value)
    planned = CliRunner().invoke(app, ["validate", str(build)])
    assert planned.exit_code == 1
    assert "no_such_model" in planned.output
    assert "step indices or ISO dates" in planned.output


def test_validate_reports_an_unknown_design_plan_with_bad_keys(tmp_path: Path) -> None:
    write_case(tmp_path, times=("t0", "t1"))
    build = write_build(tmp_path, design='plan = "nope"\ntraining = 4\nheld_out = 2')

    with pytest.raises(ConfigError) as error:
        surrogates.plan_build(surrogates.load_surrogate_config(build))
    assert "unknown design plan 'nope'" in str(error.value)
    assert "step indices or ISO dates" in str(error.value)


def test_a_build_refuses_an_unknown_model_or_bad_options_before_any_run(tmp_path: Path) -> None:
    # The model used to be created only at the fit, after the whole design had run.
    write_case(tmp_path)
    for model, options, refusal in (
        ("no_such_model", "{}", "unknown surrogate_models plugin 'no_such_model'"),
        ("test_affine_build", "{ lambda = 1 }", "lambda: Extra inputs"),
    ):
        path = write_build(tmp_path, name="refused", options=options)
        path.write_text(path.read_text().replace('"test_affine_build"', f'"{model}"'))
        with pytest.raises(ConfigError, match=refusal):
            surrogates.build(path)
        # Its folder holds at most the build's lock: no design, no run, no journal.
        held = {p.name for p in (tmp_path / "runs" / "refused").glob("*")}
        assert held <= {"run.lock"}, held
    # A resume is checked as well, before it runs the rest of the design.
    first = surrogates.build(write_build(tmp_path, name="resumed"), max_runs=3)
    journal = (first.path / JOURNAL).read_text()
    saved = json.loads((first.path / "config.json").read_text())
    saved["surrogate"]["model"] = "no_such_model"
    (first.path / "config.json").write_text(json.dumps(saved))
    with pytest.raises(ConfigError, match="no_such_model"):
        surrogates.resume_build(first.path)
    assert (first.path / JOURNAL).read_text() == journal


def test_a_complete_build_returns_its_summary_after_its_plugin_is_uninstalled(
    tmp_path: Path,
) -> None:
    # The refusal of an unknown plugin before any run must not hide a finished build: its
    # summary needs no plugin. (Its configuration unchanged: a build of another model, as a
    # config.json switched to one, is another fit, refitted or refused, not this build.)
    write_case(tmp_path)
    path = write_build(tmp_path, name="gone")
    path.write_text(path.read_text().replace('"test_affine_build"', '"test_affine_gone"'))
    surrogate_models.register("test_affine_gone")(_Affine)
    try:
        built = surrogates.build(path)
    finally:
        surrogate_models.remove("test_affine_gone")
    assert built.complete

    def contents() -> dict[Path, bytes]:
        # run.lock records when each call took the folder's lock: not the build's content.
        return {
            p: p.read_bytes() for p in built.path.rglob("*") if p.is_file() and p.name != "run.lock"
        }

    files = contents()
    assert surrogates.build(path).summary == built.summary
    assert surrogates.resume_build(built.path).summary == built.summary
    assert contents() == files
    # Not complete (its model files gone): the plugin is needed, so it is refused.
    (built.path / "model" / "surrogate.npz").unlink()
    with pytest.raises(ConfigError, match="unknown surrogate_models plugin 'test_affine_gone'"):
        surrogates.build(path)


def test_validation_refuses_a_plan_or_surrogate_before_any_run(tmp_path: Path) -> None:
    from apex.core.plugins import surrogate_designs

    write_case(tmp_path)
    with pytest.raises(ConfigError, match="unknown design plan"):
        surrogates.plan_build(surrogates.load_surrogate_config(
            write_build(tmp_path, design='plan = "nope"\ntraining = 4\nheld_out = 2')
        ))  # fmt: skip
    surrogate_designs.register("test_needs_std")(_NeedsStd)
    try:
        with pytest.raises(ConfigError, match="needs a surrogate with predict_std"):
            surrogates.plan_build(surrogates.load_surrogate_config(
                write_build(tmp_path, design='plan = "test_needs_std"\ntraining = 8\nheld_out = 2')
            ))  # fmt: skip
    finally:
        surrogate_designs.remove("test_needs_std")


class _CoarseAffine(_Affine):
    """A coarse-corrected test plugin: each key an affine map of its coarse value."""

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        assert snapshots.coarse_outputs is not None
        coarse, fine = snapshots.coarse_outputs, snapshots.outputs
        self.slope = np.array([np.polyfit(coarse[k], fine[k], 1) for k in range(len(fine))])

    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.asarray(coarse, dtype=np.float64), np.zeros(np.asarray(coarse).shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise ValueError("needs the coarse forecast")

    def predict_corrected(
        self, inputs: SurrogateInputs, coarse: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        rows = [self.keys.index(k) for k in keys]
        full = self.slope[:, :1] * coarse + self.slope[:, 1:]
        return np.ascontiguousarray(full[rows])

    def to_arrays(self) -> SurrogateParts:
        own, arrays = super().to_arrays()
        return own, {**arrays, "slope": self.slope}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = super().from_arrays(manifest, arrays)
        model.slope = np.asarray(arrays["slope"], dtype=np.float64)
        return model


@pytest.fixture
def _coarse_plugin() -> Iterator[None]:
    surrogate_models.register("test_coarse_affine")(_CoarseAffine)
    try:
        yield
    finally:
        surrogate_models.remove("test_coarse_affine")


def _coarse_case(folder: Path) -> str:
    """A coarse matrix whose fine response is ``d * coarse`` exactly, key by key."""
    write_case(folder)
    forward = np.load(folder / "forward.npz")["matrix"]
    d = np.linspace(0.5, 2.0, forward.shape[0])
    np.savez(folder / "coarse.npz", matrix=forward / d[:, None])
    return '[coarse]\noptions = { forward_file = "coarse.npz" }'


def _build_with(folder: Path, model: str, extra: str, **kwargs: str) -> Path:
    path = write_build(folder, extra=extra, **kwargs)
    path.write_text(path.read_text().replace('model = "test_affine_build"', f'model = "{model}"'))
    return path


@pytest.mark.usefixtures("_coarse_plugin")
def test_a_coarse_stack_runs_every_design_run_and_the_gate_judges_the_correction(
    tmp_path: Path,
) -> None:
    """Every run on both stacks; a fine response affine in the coarse one is
    corrected exactly, and the gate judges the corrected prediction."""
    extra = _coarse_case(tmp_path)
    path = _build_with(tmp_path, "test_coarse_affine", extra)
    plan = surrogates.plan_build(surrogates.load_surrogate_config(path))
    assert plan.simulator_runs == 50  # 25 runs on each stack
    result = surrogates.build(path)
    assert result.complete
    assert result.summary.simulator_runs == 50
    assert max(_gate_errors(result.path)) < 1e-6
    assert len((result.path / "coarse" / JOURNAL).read_text().splitlines()) == 25
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.coarse_outputs is not None
    assert snapshots.coarse_outputs.shape == snapshots.outputs.shape
    manifest = read_surrogate(result.path / "model" / "surrogate.json").manifest
    assert manifest.provenance.coarse is not None
    assert manifest.provenance.coarse.options_digest != manifest.provenance.options_digest
    config = surrogates.load_surrogate_config(path)
    assert "coarse" in surrogates.fingerprint_parts(config, surrogates._build_case(config))
    # The manifest names its coarse stack, relative to the model (step 2 runs it).
    assert manifest.coarse is not None
    assert manifest.coarse.options["forward_file"] == "../../../coarse.npz"
    # `apex surrogate check` runs both stacks and gates the corrected prediction.
    check = surrogates.check_surrogate(
        result.path / "model" / "surrogate.json",
        tmp_path / "case.toml",
        runs=6,
        seed=3,
        min_inside=0.0,  # the coarse coordinates' box, from 25 runs, is narrow
    )
    assert check.summary.passed, check.summary.notes
    assert check.summary.inside >= 1
    assert len((check.path / "coarse" / JOURNAL).read_text().splitlines()) == 6
    assert check.summary.gate is not None
    assert max(d.error or 0.0 for d in check.summary.gate.datatypes) < 1e-6
    # Bounded by max_runs: the fine runs first, then the coarse ones, across sessions.
    model = result.path / "model" / "surrogate.json"
    first = surrogates.check_surrogate(
        model, tmp_path / "case.toml", runs=6, seed=4, min_inside=0.0, max_runs=6
    )
    assert first.summary.status == "stopped"
    second = surrogates.check_surrogate(
        model, tmp_path / "case.toml", runs=6, seed=4, min_inside=0.0, max_runs=6
    )
    assert second.summary.status == "complete"
    assert len((second.path / "coarse" / JOURNAL).read_text().splitlines()) == 6


@pytest.mark.usefixtures("_coarse_plugin", "_adaptive")
def test_what_a_coarse_stack_refuses(tmp_path: Path) -> None:
    extra = _coarse_case(tmp_path)
    with pytest.raises(ConfigError, match="needs the runs' coarse outputs"):
        surrogates.build(_build_with(tmp_path, "test_coarse_affine", "", name="bare"))
    with pytest.raises(ConfigError, match="not supported yet"):
        surrogates.build(_build_with(tmp_path, "test_coarse_affine", extra, name="adaptive",
                                     design=ADAPTIVE))  # fmt: skip
    config = surrogates.load_surrogate_config(write_build(tmp_path))
    assert "coarse" not in surrogates.fingerprint_parts(config, surrogates._build_case(config))
    with pytest.raises(ConfigError, match="does not correct a coarse simulation"):
        surrogates.build(_build_with(tmp_path, "test_affine_build", extra, name="plain"))


@pytest.mark.usefixtures("_coarse_plugin")
def test_a_build_stopped_in_its_coarse_runs_resumes_to_identical_files(tmp_path: Path) -> None:
    extra = _coarse_case(tmp_path)
    whole = surrogates.build(_build_with(tmp_path, "test_coarse_affine", extra, name="whole"))
    path = _build_with(tmp_path, "test_coarse_affine", extra, name="parts")
    first = surrogates.build(path, max_runs=30)  # the 25 fine runs and 5 coarse ones
    assert not first.complete
    assert first.summary.pending == 20
    assert len((first.path / "coarse" / JOURNAL).read_text().splitlines()) == 5
    finished = surrogates.build(path)
    assert finished.complete
    for name in ("surrogate.json", "surrogate.npz"):
        assert (finished.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()


# ------------------------------------------- coarse-only design members (decision 13)

COARSE_ONLY = 'plan = "prior"\ntraining = 20\nheld_out = 5\ncoarse_only = 7'


class _CoarseOnlyAffine(_CoarseAffine):
    """The coarse-corrected test plugin, recording the coarse-only members its fit saw.

    It declares ``trains_on_coarse``: coarse-only members are refused for a plugin whose fit
    would not read them.
    """

    trains_on_coarse = True
    seen: ClassVar[list[int]] = []

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        type(self).seen.append(snapshots.n_coarse_only)


@pytest.fixture
def _coarse_only_plugin() -> Iterator[None]:
    _CoarseOnlyAffine.seen = []
    surrogate_models.register("test_coarse_only")(_CoarseOnlyAffine)
    try:
        yield
    finally:
        surrogate_models.remove("test_coarse_only")


@pytest.mark.usefixtures("_coarse_only_plugin")
def test_coarse_only_members_run_on_the_coarse_stack_only_and_reach_the_fit(
    tmp_path: Path,
) -> None:
    extra = _coarse_case(tmp_path)
    path = _build_with(tmp_path, "test_coarse_only", extra, design=COARSE_ONLY)
    plain = _build_with(tmp_path, "test_coarse_only", extra, name="plain")
    plan = surrogates.plan_build(surrogates.load_surrogate_config(path))
    assert plan.simulator_runs == 57  # 25 runs on each stack and 7 on the coarse one only
    assert plan.coarse_only == 7
    from apex.cli.surrogate_commands import surrogate_plan_lines

    assert "simulator runs: 57, 7 of them coarse-only" in "\n".join(surrogate_plan_lines(plan))

    result = surrogates.build(path)
    assert result.complete
    assert result.summary.simulator_runs == 57
    assert result.summary.coarse_only_runs == 7
    assert result.summary.training_runs == 20
    assert len((result.path / JOURNAL).read_text().splitlines()) == 25
    assert len((result.path / "coarse" / JOURNAL).read_text().splitlines()) == 32
    assert _CoarseOnlyAffine.seen == [7]
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.n_runs == 25
    assert snapshots.coarse_only_inputs is not None
    assert snapshots.coarse_only_outputs is not None
    assert snapshots.coarse_only_inputs["x"].shape == (N_STATE, 7)
    # The coarse stack's response (linear) at the coarse-only inputs, from the paired runs.
    assert snapshots.coarse_outputs is not None
    response = snapshots.coarse_outputs @ np.linalg.pinv(snapshots.inputs["x"])
    np.testing.assert_allclose(
        snapshots.coarse_only_outputs, response @ snapshots.coarse_only_inputs["x"], atol=1e-9
    )
    # The other members are a design's without coarse-only ones, ids after theirs.
    design = SurrogateDirectory(result.path).load_design()
    assert design is not None
    assert design.coarse_only_ids == tuple(range(25, 32))
    whole = surrogates.build(plain)
    other = SurrogateDirectory(whole.path).load_design()
    assert other is not None
    assert other.coarse_only_ids == ()
    assert (design.held_out_ids, design.training_ids) == (other.held_out_ids, other.training_ids)
    np.testing.assert_array_equal(design.training["x"], other.training["x"])
    np.testing.assert_array_equal(design.held_out["x"], other.held_out["x"])
    # The gate is the fine runs': the same as without coarse-only members.
    assert (result.path / "gate.json").read_bytes() == (whole.path / "gate.json").read_bytes()
    assert '"coarse_only": 7' in (result.path / "config.json").read_text()


class _CoarseTrained(_Affine):
    """A plugin that trains on coarse runs and predicts without them (``trains_on_coarse``)."""

    trains_on_coarse = True
    seen: ClassVar[list[tuple[int, int]]] = []

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        assert snapshots.coarse_outputs is not None
        type(self).seen.append((snapshots.n_runs, snapshots.n_coarse_only))
        super().fit(snapshots, rng)


@pytest.fixture
def _coarse_trained_plugin() -> Iterator[None]:
    _CoarseTrained.seen = []
    surrogate_models.register("test_coarse_trained")(_CoarseTrained)
    try:
        yield
    finally:
        surrogate_models.remove("test_coarse_trained")


@pytest.mark.usefixtures("_coarse_trained_plugin")
def test_a_plugin_that_trains_on_coarse_runs_takes_a_coarse_stack(tmp_path: Path) -> None:
    """``trains_on_coarse``: [coarse] and coarse-only members in training, a plain predict."""
    extra = _coarse_case(tmp_path)
    with pytest.raises(ConfigError, match="trains on coarse runs: it needs a"):
        surrogates.build(_build_with(tmp_path, "test_coarse_trained", "", name="bare"))
    result = surrogates.build(
        _build_with(tmp_path, "test_coarse_trained", extra, design=COARSE_ONLY)
    )
    assert result.complete
    assert _CoarseTrained.seen == [(20, 7)]
    assert result.summary.coarse_only_runs == 7


def test_a_model_that_trains_on_coarse_runs_is_not_refitted_in_a_run() -> None:
    """A run makes no coarse runs for it, and a refit set would need them (numerics review)."""
    from types import SimpleNamespace

    config = SimpleNamespace(surrogate=SimpleNamespace(on_check_failure="refit"))
    model = SimpleNamespace(model=_CoarseTrained(), manifest=None, path=Path("model.json"))
    with pytest.raises(ConfigError, match="trains on coarse runs, which a run does not make"):
        surrogates.refuse_model(cast(Any, config), cast(Any, model), cast(Any, None),
                                cast(Any, None))  # fmt: skip


@pytest.mark.usefixtures("_coarse_plugin")
def test_coarse_only_members_are_refused_for_a_plugin_that_does_not_read_them(
    tmp_path: Path,
) -> None:
    """A coarse-corrected plugin's fit reads no coarse-only members: their runs are refused."""
    extra = _coarse_case(tmp_path)
    with pytest.raises(ConfigError, match="does not train on coarse runs"):
        surrogates.build(_build_with(tmp_path, "test_coarse_affine", extra, design=COARSE_ONLY))


@pytest.mark.usefixtures("_coarse_only_plugin")
def test_a_failed_coarse_only_run_is_reported_apart_from_the_runs(tmp_path: Path) -> None:
    """``failed`` keeps the runs the fit and gate lost; coarse-only failures have their own."""
    extra = _coarse_case(tmp_path).replace(
        'forward_file = "coarse.npz" }', 'forward_file = "coarse.npz", fail_members = [27] }'
    )
    design = COARSE_ONLY + "\nmin_success_fraction = 0.8"  # 6 of 7
    result = surrogates.build(_build_with(tmp_path, "test_coarse_only", extra, design=design))
    assert result.complete
    assert result.summary.failed == []
    assert result.summary.coarse_only_failed == [27]
    assert result.summary.coarse_only_runs == 6
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert snapshots.failed == ()
    gate = json.loads((result.path / "gate.json").read_text())
    assert gate["failed"] == []
    # Counted on their own against min_success_fraction: 1 of 7 fails the default 0.9.
    with pytest.raises(WorkflowError, match="1 of the 7 coarse-only runs failed"):
        surrogates.build(_build_with(tmp_path, "test_coarse_only", extra, design=COARSE_ONLY,
                                     name="strict"))  # fmt: skip


@pytest.mark.usefixtures("_coarse_only_plugin")
def test_coarse_only_zero_is_a_build_without_them(tmp_path: Path) -> None:
    """``coarse_only = 0`` changes nothing: config, fingerprint, design, snapshots, model."""
    extra = _coarse_case(tmp_path)
    absent = _build_with(tmp_path, "test_coarse_only", extra, name="absent")
    zero = _build_with(
        tmp_path,
        "test_coarse_only",
        extra,
        name="zero",
        design='plan = "prior"\ntraining = 20\nheld_out = 5\ncoarse_only = 0',
    )
    configs = [surrogates.load_surrogate_config(p) for p in (absent, zero)]
    parts = [surrogates.fingerprint_parts(c, surrogates._build_case(c)) for c in configs]
    assert parts[0] == parts[1]
    first, second = surrogates.build(absent), surrogates.build(zero)
    assert '"coarse_only"' not in (second.path / "config.json").read_text()
    for name in ("design.npz", "snapshots.npz", "model/surrogate.json", "model/surrogate.npz"):
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes(), name
    assert load_snapshots(second.path / "snapshots.npz").coarse_only_outputs is None
    assert _CoarseOnlyAffine.seen == [0, 0]
    assert second.summary.coarse_only_runs == 0
    # A build made without the key resumes with coarse_only = 0 written out.
    absent.write_text(zero.read_text().replace('name = "zero"', 'name = "absent"'))
    again = surrogates.build(absent)
    assert again.summary.identity == first.summary.identity


@pytest.mark.usefixtures("_coarse_only_plugin", "_adaptive")
def test_what_coarse_only_members_refuse(tmp_path: Path) -> None:
    extra = _coarse_case(tmp_path)
    with pytest.raises(ConfigError, match=r"needs|add a \[coarse\] stack"):
        surrogates.load_surrogate_config(
            _build_with(tmp_path, "test_coarse_only", "", name="bare", design=COARSE_ONLY)
        )
    with pytest.raises(ConfigError, match="plan = 'run' reuses"):
        surrogates.load_surrogate_config(_build_with(
            tmp_path, "test_coarse_only", extra, name="run",
            design=f'plan = "run"\nrun_dir = "{tmp_path}"\ntraining = 4\nheld_out = 2\n'
            "coarse_only = 3",
        ))  # fmt: skip
    with pytest.raises(ConfigError, match="not supported yet"):
        surrogates.build(_build_with(tmp_path, "test_coarse_only", extra, name="adaptive",
                                     design=ADAPTIVE + "\ncoarse_only = 3"))  # fmt: skip
    with pytest.raises(ConfigError, match="greater than or equal to 0"):
        surrogates.load_surrogate_config(_build_with(
            tmp_path, "test_coarse_only", extra, name="negative",
            design='plan = "prior"\ntraining = 20\nheld_out = 5\ncoarse_only = -1',
        ))  # fmt: skip
    # Plan file: the next columns, refused when the file has too few.
    np.savez(tmp_path / "ensemble.npz", x=np.random.default_rng(0).standard_normal((N_STATE, 30)))
    stored = 'plan = "file"\nensemble_file = "ensemble.npz"\ntraining = 20\nheld_out = 5\n'
    with pytest.raises(ConfigError, match="20 training and 6 coarse-only runs"):
        surrogates.build(_build_with(tmp_path, "test_coarse_only", extra, name="few",
                                     design=stored + "coarse_only = 6"))  # fmt: skip
    built = surrogates.build(_build_with(tmp_path, "test_coarse_only", extra, name="file",
                                         design=stored + "coarse_only = 5"))  # fmt: skip
    design = SurrogateDirectory(built.path).load_design()
    assert design is not None
    used = {*design.held_out_ids, *design.training_ids}
    assert set(design.coarse_only_ids) == set(range(30)) - used
    # Another number of coarse-only members is another build.
    path = _build_with(tmp_path, "test_coarse_only", extra, name="file",
                       design=stored + "coarse_only = 4")  # fmt: skip
    with pytest.raises(ConfigError, match="design differ"):
        surrogates.build(path)


@pytest.mark.usefixtures("_coarse_only_plugin")
def test_a_build_stopped_in_its_coarse_only_runs_resumes_to_identical_files(
    tmp_path: Path,
) -> None:
    extra = _coarse_case(tmp_path)
    whole = surrogates.build(
        _build_with(tmp_path, "test_coarse_only", extra, name="whole", design=COARSE_ONLY)
    )
    path = _build_with(tmp_path, "test_coarse_only", extra, name="parts", design=COARSE_ONLY)
    first = surrogates.build(path, max_runs=53)  # 25 fine, 25 coarse and 3 coarse-only runs
    assert not first.complete
    assert first.summary.pending == 4
    assert len((first.path / "coarse" / JOURNAL).read_text().splitlines()) == 28
    finished = surrogates.build(path)
    assert finished.complete
    assert finished.summary.simulator_runs == 57
    assert len((finished.path / "coarse" / JOURNAL).read_text().splitlines()) == 32
    for name in ("snapshots.npz", "model/surrogate.json", "model/surrogate.npz"):
        assert (finished.path / name).read_bytes() == (whole.path / name).read_bytes(), name


# ------------------------------------------------------------- state fields


class _AffineFields(_Affine):
    """The affine test plugin that also maps the inputs to state fields (``predict_fields``)."""

    field_weights: dict[str, NDArray[np.float64]]
    field_times: tuple[str, ...]

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        design = np.vstack([self._stack(snapshots.inputs), np.ones((1, snapshots.n_runs))])
        self.field_times = snapshots.field_times
        self.field_weights = {}
        for name, values in snapshots.fields.items():
            flat = values.reshape(-1, snapshots.n_runs)  # (cells * times, n)
            solved = np.linalg.lstsq(design.T, flat.T, rcond=None)[0].T
            self.field_weights[name] = np.ascontiguousarray(solved)

    def predict_fields(
        self, inputs: SurrogateInputs, names: Sequence[str], times: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        stacked = self._stack(inputs)
        found = {}
        for name in names:
            w = self.field_weights[name]
            flat = w[:, :-1] @ stacked + w[:, -1:]
            cube = flat.reshape(-1, len(self.field_times), stacked.shape[1])
            found[name] = cube[:, [self.field_times.index(t) for t in times]]
        return found

    def to_arrays(self) -> SurrogateParts:
        own, arrays = super().to_arrays()
        own = {**own, "field_times": list(self.field_times), "fields": sorted(self.field_weights)}
        return own, {**arrays, **{f"field_{n}": w for n, w in self.field_weights.items()}}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = super().from_arrays(manifest, arrays)
        model.field_times = tuple(manifest["field_times"])  # type: ignore[arg-type]
        model.field_weights = {
            str(n): np.asarray(arrays[f"field_{n}"], np.float64)
            for n in manifest["fields"]  # type: ignore[attr-defined]
        }
        return model


def _field_case(folder: Path) -> None:
    """The linear case whose simulator also answers PRESSURE over 5 cells at times 1 and 2."""
    write_case(folder, simulator_extra='fields = [["1", "PRESSURE", 5], ["2", "PRESSURE", 5]]')
    rng = np.random.default_rng(2)
    scalar = np.load(folder / "forward.npz")["matrix"]
    np.savez(folder / "forward.npz", matrix=np.vstack([scalar, rng.standard_normal((10, 3))]))


FIELDS = 'fields = ["PRESSURE"]\nfield_times = ["1", "2"]\nfield_cells = 5'


def test_a_build_saves_state_fields_and_gates_their_prediction(tmp_path: Path) -> None:
    """[case] fields snapshot every run's fields; the gate judges them."""
    surrogate_models.register("test_affine_fields")(_AffineFields)
    try:
        _field_case(tmp_path)
        path = write_build(tmp_path, name="fields")
        path.write_text(
            path.read_text()
            .replace('model = "test_affine_build"', 'model = "test_affine_fields"')
            .replace('config_file = "case.toml"', f'config_file = "case.toml"\n{FIELDS}')
        )
        result = surrogates.build(path)
        assert result.complete
        snapshots = load_snapshots(result.path / "snapshots.npz")
        assert snapshots.field_times == ("1", "2")
        assert snapshots.fields["PRESSURE"].shape == (5, 2, 25)  # training and held out
        gate = json.loads((result.path / "gate.json").read_text())
        assert any(d["datatype"] == "PRESSURE" for d in gate["datatypes"])
        assert gate["passed"]
        manifest = read_surrogate(result.path / "model" / "surrogate.json").manifest
        assert manifest.fields == ["PRESSURE"]
        assert manifest.field_times == ["1", "2"]
        config = surrogates.load_surrogate_config(path)
        assert "fields" in surrogates.fingerprint_parts(config, surrogates._build_case(config))
        check = surrogates.check_surrogate(
            result.path / "model" / "surrogate.json", tmp_path / "case.toml", runs=4, seed=9
        )
        assert check.summary.passed, check.summary.notes
        assert check.summary.gate is not None
        assert any(d.datatype == "PRESSURE" for d in check.summary.gate.datatypes)
    finally:
        surrogate_models.remove("test_affine_fields")


def test_what_state_fields_refuse(tmp_path: Path) -> None:
    _field_case(tmp_path)
    path = write_build(tmp_path, name="plain")
    path.write_text(
        path.read_text().replace(
            'config_file = "case.toml"', f'config_file = "case.toml"\n{FIELDS}'
        )
    )
    with pytest.raises(ConfigError, match="predicts no fields"):
        surrogates.build(path)
    bad = tmp_path / "untimed.toml"
    bad.write_text(path.read_text().replace('field_times = ["1", "2"]\n', ""))
    with pytest.raises(ConfigError, match="field_times"):
        surrogates.load_surrogate_config(bad)


FILE_DESIGN = 'plan = "file"\ntraining = 30\nheld_out = 10\nensemble_file = "{}"'


def test_a_design_file_counts_by_its_content_not_its_path(tmp_path: Path) -> None:
    """2026-11-03: a build whose ensemble moves with the same content is the same build;
    another content is refused, naming the design."""
    _stored_prior_case(tmp_path)
    path = write_build(tmp_path, name="moved", design=FILE_DESIGN.format("prior.npz"))
    first = surrogates.build(path, max_runs=10)
    assert not first.complete
    (tmp_path / "elsewhere").mkdir()
    shutil.copy(tmp_path / "prior.npz", tmp_path / "elsewhere" / "prior.npz")
    path.write_text(path.read_text().replace('"prior.npz"', '"elsewhere/prior.npz"'))
    # Resumed while incomplete (no fit yet rewrites config.json): the new path recorded.
    assert not surrogates.build(path, max_runs=5).complete
    stored = json.loads((first.path / "config.json").read_text())
    assert stored["design"]["ensemble_file"].endswith("elsewhere/prior.npz")
    assert surrogates.build(path).complete  # resumed: the same runs
    with np.load(tmp_path / "prior.npz") as data:
        changed = {k: data[k] + 0.5 for k in data.files}
    np.savez(tmp_path / "elsewhere" / "prior.npz", **changed)
    with pytest.raises(ConfigError, match="design differ"):
        surrogates.build(path)


def test_a_build_fingerprinted_by_its_design_path_resumes_and_is_upgraded(
    tmp_path: Path,
) -> None:
    _stored_prior_case(tmp_path)
    path = write_build(tmp_path, name="old", design=FILE_DESIGN.format("prior.npz"))
    first = surrogates.build(path, max_runs=10)
    config = surrogates.load_surrogate_config(path)
    case = surrogates._build_case(config)
    legacy = surrogates.fingerprint_parts(config, case, legacy=True)
    assert legacy["design"] != surrogates.fingerprint_parts(config, case)["design"]
    # As a build made before 2026-11-03 recorded itself.
    directory = SurrogateDirectory(first.path)
    directory.write_state(
        surrogate_fingerprint=surrogates._fingerprint(legacy), fingerprint_parts=legacy
    )
    assert surrogates.build(path).complete
    state = directory.read_state()
    parts = surrogates.fingerprint_parts(config, case)
    assert state["surrogate_fingerprint"] == surrogates._fingerprint(parts)
    assert state["fingerprint_parts"] == parts


def test_a_legacy_build_of_other_runs_is_refused_not_upgraded(tmp_path: Path) -> None:
    _stored_prior_case(tmp_path)
    path = write_build(tmp_path, name="other", design=FILE_DESIGN.format("prior.npz"))
    first = surrogates.build(path, max_runs=5)
    config = surrogates.load_surrogate_config(path)
    legacy = surrogates.fingerprint_parts(config, surrogates._build_case(config), legacy=True)
    stale = {**legacy, "seed": "8"}  # a legacy build of another seed
    directory = SurrogateDirectory(first.path)
    directory.write_state(
        surrogate_fingerprint=surrogates._fingerprint(stale), fingerprint_parts=stale
    )
    with pytest.raises(ConfigError, match="seed differ"):
        surrogates.build(path)
    assert directory.read_state()["surrogate_fingerprint"] == surrogates._fingerprint(stale)


def test_drawn_designs_keep_their_fingerprints(tmp_path: Path) -> None:
    """prior and lhs plans name no file: their design part is the same as before."""
    write_case(tmp_path)
    for name, design in (("p", 'plan = "prior"\ntraining = 8\nheld_out = 2'),
                         ("l", 'plan = "lhs"\ntraining = 8\nheld_out = 2')):  # fmt: skip
        config = surrogates.load_surrogate_config(write_build(tmp_path, name=name, design=design))
        case = surrogates._build_case(config)
        assert surrogates.fingerprint_parts(config, case) == surrogates.fingerprint_parts(
            config, case, legacy=True
        )


class _EarlyLinear(_Affine):
    """An early-start test plugin: every key a linear map of the keys up to time "2".

    The case is linear (``y = G x``, three inputs) and four early keys determine ``x``, so
    the map is exact. Its early keys are listed in the reverse of the case's order, so a
    check that sliced the early data or the gated rows by position would fail.
    """

    early_until = "2"

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        early = snapshots.outputs[[snapshots.keys.index(k) for k in self.early_keys()]]
        self.map = np.ascontiguousarray(snapshots.outputs @ np.linalg.pinv(early))

    def early_keys(self) -> tuple[DataKey, ...]:
        return tuple(k for k in reversed(self.keys) if k.time <= self.early_until)

    def early_coordinates(
        self, early: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.ascontiguousarray(early), np.zeros(early.shape[1])

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise ValueError("needs the members' early data")

    def predict_from_early(
        self, inputs: SurrogateInputs, early: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        full = self.map @ early
        for row, key in enumerate(self.early_keys()):
            full[self.keys.index(key)] = early[row]  # the members' own values, unchanged
        return np.ascontiguousarray(full[[self.keys.index(k) for k in keys]])

    def to_arrays(self) -> SurrogateParts:
        own, arrays = super().to_arrays()
        return own, {**arrays, "map": self.map}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = super().from_arrays(manifest, arrays)
        model.map = np.asarray(arrays["map"], dtype=np.float64)
        return model


@pytest.fixture
def _early_plugin() -> Iterator[None]:
    surrogate_models.register("test_early_linear")(_EarlyLinear)
    try:
        yield
    finally:
        surrogate_models.remove("test_early_linear")


@pytest.mark.usefixtures("_early_plugin")
def test_a_check_gates_an_early_start_model_on_its_runs_own_early_data(tmp_path: Path) -> None:
    """``apex surrogate check`` takes each run's early
    data from its own keys (by key, whatever the model's order), bounds them in the domain,
    predicts from them, and gates only the keys after the early date."""
    write_case(tmp_path)
    result = surrogates.build(_build_with(tmp_path, "test_early_linear", ""))
    assert result.complete
    assert max(_gate_errors(result.path)) < 1e-8
    model = result.path / "model" / "surrogate.json"
    check = surrogates.check_surrogate(model, tmp_path / "case.toml", runs=4, seed=3,
                                       min_inside=0.0)  # fmt: skip
    gate = check.summary.gate
    assert gate is not None
    assert gate.held_out, "every checked run fell outside the domain"
    assert len(gate.held_out) + len(check.summary.outside) == 4
    assert gate.passed, gate.failures()
