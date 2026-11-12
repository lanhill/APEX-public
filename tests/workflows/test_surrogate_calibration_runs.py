"""Calibration runs of a surrogate's spread (``[design] calibration``).

The user's decision (2026-11-10): a surrogate with ``uncertainty = "laplace"`` gets extra full
runs, drawn after every other run and never trained or gated on, for its spread's two
scales; automatically 30 for a case of up to 20,000 grid cells and 20 above, stated in the
plan, changed by ``[design] calibration`` (0: none, the validation runs instead). Builds of
plugins without the capability are those of before. The cases are the linear ones of
``test_surrogate_workflow.py`` (loaded from its file: ``tests/`` is not a package) with its
test-only affine map, given the capability by an option here.
"""

import functools
import importlib.util
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from apex.algorithms.surrogates.designs import (
    Design,
    ensemble_design,
    lhs_design,
    sampled_design,
)
from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.core.plugins import surrogate_models
from apex.core.surrogates import SurrogateParts
from apex.domain.surrogate import SnapshotSet
from apex.services import composition, surrogates
from apex.services.surrogates import SurrogateCase
from apex.storage.surrogate_directory import JOURNAL, load_snapshots


def _load(name: str) -> ModuleType:
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_{name}_for_calibration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


workflow = _load("test_surrogate_workflow")
write_case = workflow.write_case
write_build = workflow.write_build


class _Calibrated(workflow._Affine):  # type: ignore[misc,name-defined]
    """The affine map with the capability (``uncertainty = "laplace"``); it records the
    calibration runs its fit was given, and fits only on the training runs."""

    def __init__(
        self,
        ridge: float = 0.0,
        fail: bool = False,
        uncertainty: str = "none",
        refuse: bool = False,
    ) -> None:
        if refuse:
            raise ValueError("refuse is set: this option is refused")
        super().__init__(ridge=ridge, fail=fail)
        self.uncertainty = uncertainty
        self.calibration_ids: tuple[int, ...] = ()

    @property
    def uses_calibration_runs(self) -> bool:
        return self.uncertainty == "laplace"

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        super().fit(snapshots, rng)
        runs = snapshots.calibration_runs()
        self.calibration_ids = () if runs is None else runs.run_ids
        if runs is not None:
            # Its runs are the simulator's: the affine map reproduces them (linear case).
            np.testing.assert_allclose(self.predict(runs.inputs, runs.keys), runs.outputs,
                                       atol=1e-8)  # fmt: skip

    def to_arrays(self) -> SurrogateParts:
        own, arrays = super().to_arrays()
        return {**own, "uncertainty": self.uncertainty,
                "calibration_ids": list(self.calibration_ids)}, arrays  # fmt: skip

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = super().from_arrays(manifest, arrays)
        model.uncertainty = str(manifest["uncertainty"])
        model.calibration_ids = tuple(cast("list[int]", manifest["calibration_ids"]))
        return cast("Self", model)


@pytest.fixture(autouse=True)
def _registered() -> Iterator[None]:
    surrogate_models.register("test_affine_build")(workflow._Affine)
    surrogate_models.register("test_calibrated")(_Calibrated)
    try:
        yield
    finally:
        surrogate_models.remove("test_affine_build")
        surrogate_models.remove("test_calibrated")


def _build(folder: Path, name: str, *, laplace: bool, design: str | None = None) -> Path:
    path = write_build(
        folder, name=name,
        design=design or 'plan = "prior"\ntraining = 20\nheld_out = 5',
        options='{ uncertainty = "laplace" }' if laplace else "{}",
    )  # fmt: skip
    text = path.read_text().replace('model = "test_affine_build"', 'model = "test_calibrated"')
    path.write_text(text)
    return cast("Path", path)


def test_designs_draw_the_calibration_runs_last() -> None:
    """Every core plan: the held-out and training runs (and the coarse-only ones) are those
    of the design without calibration runs; the calibration runs come after them."""

    def draw(n: int, rng: np.random.Generator) -> dict[str, NDArray[np.float64]]:
        return {"x": rng.standard_normal((3, n))}

    def sampled(k: int) -> Design:
        return sampled_design(draw, 6, 3, np.random.default_rng(1), 2, k)

    def hypercube(k: int) -> Design:
        box = {"x": (np.zeros(3), np.ones(3))}
        return lhs_design(box, 6, 3, np.random.default_rng(1), 2, k)

    def columns(k: int) -> Design:
        ensemble = draw(20, np.random.default_rng(9))
        return ensemble_design("file", ensemble, 3, np.random.default_rng(1), 6, 2, k)

    for make in (sampled, hypercube, columns):
        plain, calibrated = make(0), make(4)
        assert plain.calibration is None
        assert calibrated.calibration is not None
        assert calibrated.calibration["x"].shape == (3, 4)
        for part in ("training", "held_out", "coarse_only"):
            for name, value in getattr(plain, part).items():
                np.testing.assert_array_equal(getattr(calibrated, part)[name], value)


def test_a_snapshot_sets_calibration_runs() -> None:
    data = workflow_snapshots()
    runs = data.calibration_runs()
    assert runs is not None
    assert runs.run_ids == (10, 11)
    assert data.n_calibration == 2
    # Not runs of the set: columns keep them whole; the digest counts them.
    part = data.columns([0, 1])
    assert part.calibration_ids == (10, 11)
    plain = workflow_snapshots(calibration=False)
    assert plain.digest() != data.digest()
    assert plain.calibration_runs() is None
    with pytest.raises(ValueError, match="cannot also be a run"):
        workflow_snapshots(ids=(3, 11))
    with pytest.raises(ValueError, match="together"):
        SnapshotSet(**{**_fields(), "calibration_ids": ()})


def _fields(calibration: bool = True, ids: tuple[int, ...] = (10, 11)) -> dict[str, Any]:
    from apex.domain.observations import DataKey
    from apex.domain.state import StateLayout, StateVariable
    from apex.domain.surrogate import SnapshotProvenance

    rng = np.random.default_rng(0)
    fields: dict[str, Any] = {
        "layout": StateLayout((StateVariable("x", 3),)),
        "inputs": {"x": rng.standard_normal((3, 5))},
        "keys": (DataKey("1", "WOPR:P1"), DataKey("2", "WOPR:P1")),
        "outputs": rng.standard_normal((2, 5)),
        "run_ids": (0, 1, 2, 3, 4),
        "provenance": SnapshotProvenance(simulator="test"),
    }
    if calibration:
        fields |= {
            "calibration_inputs": {"x": rng.standard_normal((3, 2))},
            "calibration_outputs": rng.standard_normal((2, 2)),
            "calibration_ids": ids,
        }
    return fields


def workflow_snapshots(calibration: bool = True, ids: tuple[int, ...] = (10, 11)) -> SnapshotSet:
    return SnapshotSet(**_fields(calibration, ids))


def test_without_the_capability_nothing_changes(tmp_path: Path) -> None:
    """A plugin without uses_calibration_runs: no runs, no plan line, the configuration and
    fingerprint of a build of before (calibration left out of the serialized section)."""
    write_case(tmp_path)
    config = surrogates.load_surrogate_config(write_build(tmp_path))
    assert "calibration" not in json.loads(config.design.model_dump_json())
    plan = surrogates.plan_build(config)
    assert plan.simulator_runs == 25
    assert plan.calibration == 0
    assert plan.calibration_rule is None
    assert not any("calibration" in w for w in plan.warnings)
    case = surrogates._build_case(config)
    assert "calibration" not in surrogates.fingerprint_parts(config, case)
    with pytest.raises(ConfigError, match="does not calibrate a spread"):
        surrogates.plan_build(surrogates.load_surrogate_config(
            write_build(tmp_path, name="extra",
                        design='plan = "prior"\ntraining = 20\nheld_out = 5\ncalibration = 4')
        ))  # fmt: skip


def test_laplace_adds_the_runs_automatically_and_says_so(tmp_path: Path) -> None:
    write_case(tmp_path)
    path = _build(tmp_path, "auto", laplace=True)
    config = surrogates.load_surrogate_config(path)
    plan = surrogates.plan_build(config)
    # No deck: the largest state variable's 3 entries, a small case.
    assert plan.calibration == 30
    assert plan.simulator_runs == 55
    assert plan.calibration_rule == (
        "the largest state variable, 3 cells <= 20,000: 30 runs (small case)"
    )
    line = next(w for w in plan.warnings if "calibration runs" in w)
    # The number, the rule and how to change it, in one line.
    assert line == (
        "30 extra calibration runs for the surrogate's spread (the largest state variable, 3 "
        "cells <= 20,000: 30 runs (small case)); set [design] calibration = K to change, 0 to "
        "calibrate on the validation runs instead"
    )
    assert "set [design] calibration = K to change, 0 to calibrate on the validation runs" in line
    # Never in the fingerprint (the saved design is reconciled with the count instead).
    case = surrogates._build_case(config)
    assert "calibration" not in surrogates.fingerprint_parts(config, case)

    shown = CliRunner().invoke(app, ["validate", str(path)])
    assert shown.exit_code == 0, shown.output
    assert "simulator runs: 55" in shown.output
    assert "30 of them for the spread's calibration" in shown.output


def test_the_count_is_changed_or_turned_off_in_the_design(tmp_path: Path) -> None:
    write_case(tmp_path)
    given = surrogates.plan_build(surrogates.load_surrogate_config(_build(
        tmp_path, "five", laplace=True,
        design='plan = "prior"\ntraining = 20\nheld_out = 5\ncalibration = 5',
    )))  # fmt: skip
    assert (given.calibration, given.simulator_runs) == (5, 30)
    assert given.calibration_rule == "[design] calibration = 5"
    off = surrogates.plan_build(surrogates.load_surrogate_config(_build(
        tmp_path, "off", laplace=True,
        design='plan = "prior"\ntraining = 20\nheld_out = 5\ncalibration = 0',
    )))  # fmt: skip
    assert (off.calibration, off.simulator_runs) == (0, 25)
    assert any(w.startswith("no calibration runs: the surrogate's spread is calibrated on its "
                            "validation runs") for w in off.warnings)  # fmt: skip


def test_a_build_runs_them_apart_and_resumes_to_identical_files(tmp_path: Path) -> None:
    """The calibration runs are run and saved with the snapshots, after the others (ids
    25..54), never trained or gated on: the model is the plugin's without them, the
    held-out and training runs those of the build without them; a resume finishes to the
    same files."""
    write_case(tmp_path)
    plain = surrogates.build(_build(tmp_path, "plain", laplace=False))
    whole = surrogates.build(_build(tmp_path, "whole", laplace=True))
    assert whole.complete
    assert whole.summary.simulator_runs == 55
    assert whole.summary.calibration_runs == 30
    assert whole.summary.training_runs == 20
    assert whole.summary.held_out_runs == 5
    snapshots = load_snapshots(whole.path / "snapshots.npz")
    assert snapshots.calibration_ids == tuple(range(25, 55))
    assert snapshots.run_ids == load_snapshots(plain.path / "snapshots.npz").run_ids
    manifest = json.loads((whole.path / "model" / "surrogate.json").read_text())
    assert manifest["model"]["calibration_ids"] == list(range(25, 55))
    for name in ("held_out_0", "training_0"):
        with np.load(whole.path / "design.npz") as a, np.load(plain.path / "design.npz") as b:
            np.testing.assert_array_equal(a[name], b[name])
    with (
        np.load(whole.path / "model" / "surrogate.npz") as a,
        np.load(plain.path / "model" / "surrogate.npz") as b,
    ):
        np.testing.assert_array_equal(a["weights"], b["weights"])

    path = _build(tmp_path, "parts", laplace=True)
    first = surrogates.build(path, max_runs=40)
    assert not first.complete
    assert first.summary.pending == 15
    finished = surrogates.build(path)
    assert finished.complete
    assert finished.summary.identity == whole.summary.identity
    for name in ("surrogate.json", "surrogate.npz"):
        assert (finished.path / "model" / name).read_bytes() == (
            whole.path / "model" / name
        ).read_bytes()
    lines = (finished.path / JOURNAL).read_text().splitlines()
    assert sorted(json.loads(line)["member"] for line in lines) == list(range(55))


def test_a_runs_forecasts_give_the_calibration_runs_free(tmp_path: Path) -> None:
    config = write_case(tmp_path)
    composition.start(config)
    run_dir = tmp_path / "runs" / "case"
    result = surrogates.build(_build(
        tmp_path, "free", laplace=True,
        design=f'plan = "run"\ntraining = 25\nheld_out = 10\ncalibration = 4\n'
        f'run_dir = "{run_dir}"',
    ))  # fmt: skip
    assert result.complete
    assert result.summary.simulator_runs == 0
    assert result.summary.calibration_runs == 4
    snapshots = load_snapshots(result.path / "snapshots.npz")
    assert len(snapshots.calibration_ids) == 4
    assert not set(snapshots.calibration_ids) & set(snapshots.run_ids)
    # Automatic: capped at the run's 5 members left after the others.
    capped = surrogates.plan_build(surrogates.load_surrogate_config(_build(
        tmp_path, "short", laplace=True,
        design=f'plan = "run"\ntraining = 25\nheld_out = 10\nrun_dir = "{run_dir}"',
    )))  # fmt: skip
    assert (capped.calibration, capped.simulator_runs) == (5, 0)


def test_the_rule_reads_the_decks_grid(tmp_path: Path) -> None:
    """A deck of up to 20,000 cells is small (30 runs), a larger one large (20); without a
    readable grid, the largest state variable decides."""
    from apex.domain.state import StateLayout, StateVariable

    def case(dims: str | None, size: int) -> SurrogateCase:
        options: dict[str, object] = {}
        if dims is not None:
            deck = tmp_path / f"deck_{dims.replace(' ', '_')}.DATA"
            deck.write_text(f"RUNSPEC\nDIMENS\n {dims} /\nOIL\nWATER\n")
            options["deck_template_file"] = str(deck)
        # Only what the rule reads of a case: its simulator's options and its layout.
        return cast("SurrogateCase", SimpleNamespace(
            config=SimpleNamespace(simulator=SimpleNamespace(options=options)),
            layout=StateLayout((StateVariable("permx", size),)),
        ))  # fmt: skip

    write_case(tmp_path)
    config = surrogates.load_surrogate_config(_build(tmp_path, "rule", laplace=True))
    small = surrogates.calibration_count(config, case("100 200 1", 5))
    assert (small.runs, small.rule) == (
        30,
        "grid 100 x 200 x 1 = 20,000 cells <= 20,000: 30 runs (small case)",
    )
    large = surrogates.calibration_count(config, case("139 48 9", 5))
    assert (large.runs, large.rule) == (
        20,
        "grid 139 x 48 x 9 = 60,048 cells > 20,000: 20 runs (large case)",
    )
    by_state = surrogates.calibration_count(config, case(None, 44_550))
    assert by_state.runs == 20
    assert by_state.rule is not None
    assert by_state.rule.startswith("the largest state variable, 44,550 cells")


def test_turning_laplace_on_adds_the_runs_and_a_changed_count_is_refused(
    tmp_path: Path,
) -> None:
    """A build made without the option gets its calibration runs added in the same folder
    (the other runs reused, the model that of a fresh build); a count changed after its
    runs were made is refused; 0 drops them; a complete build is found without its plugin."""
    write_case(tmp_path)
    fresh = surrogates.build(_build(tmp_path, "fresh", laplace=True))
    first = surrogates.build(_build(tmp_path, "same", laplace=False))
    assert first.complete
    turned = surrogates.build(_build(tmp_path, "same", laplace=True))
    assert turned.complete
    assert turned.summary.calibration_runs == 30
    lines = (turned.path / JOURNAL).read_text().splitlines()
    assert len(lines) == 55  # the 25 runs made, then the 30 calibration runs only
    np.testing.assert_array_equal(
        load_snapshots(turned.path / "snapshots.npz").calibration_outputs,
        load_snapshots(fresh.path / "snapshots.npz").calibration_outputs,
    )
    with pytest.raises(ConfigError, match="5 would reuse their outputs for other inputs"):
        surrogates.build(_build(
            tmp_path, "same", laplace=True,
            design='plan = "prior"\ntraining = 20\nheld_out = 5\ncalibration = 5',
        ))  # fmt: skip
    zero = surrogates.build(_build(
        tmp_path, "same", laplace=True,
        design='plan = "prior"\ntraining = 20\nheld_out = 5\ncalibration = 0',
    ))  # fmt: skip
    assert zero.complete
    assert zero.summary.calibration_runs == 0
    again = surrogates.build(_build(tmp_path, "fresh", laplace=True))
    surrogate_models.remove("test_calibrated")
    try:
        found = surrogates.build(_build(tmp_path, "fresh", laplace=True))
    finally:
        surrogate_models.register("test_calibrated")(_Calibrated)
    assert found.summary == again.summary


def test_a_finite_ensemble_caps_the_automatic_count(tmp_path: Path) -> None:
    """Plan file: the automatic count takes the columns left after the other runs (the plan
    says so); a given count beyond them is refused when validated."""
    write_case(tmp_path)
    np.savez(tmp_path / "ensemble.npz", x=np.random.default_rng(3).standard_normal((3, 33)))
    design = 'plan = "file"\ntraining = 20\nheld_out = 5\nensemble_file = "ensemble.npz"'
    plan = surrogates.plan_build(surrogates.load_surrogate_config(
        _build(tmp_path, "capped", laplace=True, design=design)
    ))  # fmt: skip
    assert plan.calibration == 8
    assert plan.calibration_rule is not None
    assert plan.calibration_rule.endswith("capped at the 8 column(s) the design's ensemble leaves")
    built = surrogates.build(_build(tmp_path, "capped", laplace=True, design=design))
    assert built.summary.calibration_runs == 8
    with pytest.raises(ConfigError, match="leaves 8 column"):
        surrogates.plan_build(surrogates.load_surrogate_config(
            _build(tmp_path, "over", laplace=True, design=design + "\ncalibration = 9")
        ))  # fmt: skip


def _design(extra: str = "") -> str:
    return 'plan = "prior"\ntraining = 20\nheld_out = 5' + extra


def test_a_recount_never_reuses_runs_made_for_other_inputs(tmp_path: Path) -> None:
    """30 -> 0 -> 5 on a drawn plan: the 5 would have the ids of 5 of the 30 made, with other
    inputs (draw(5) is not the first 5 of draw(30)): refused, as is 30 -> 5. A member saved
    by its worker but not journalled (a stop) counts as made too."""
    write_case(tmp_path)
    assert surrogates.build(_build(tmp_path, "same", laplace=True)).complete
    dropped = surrogates.build(_build(tmp_path, "same", laplace=True,
                                      design=_design("\ncalibration = 0")))  # fmt: skip
    assert dropped.complete
    with pytest.raises(ConfigError, match="would reuse their outputs for other inputs"):
        surrogates.build(_build(tmp_path, "same", laplace=True,
                                design=_design("\ncalibration = 5")))  # fmt: skip
    # Back to the count they were made with: the same ids and inputs, accepted and reused.
    back = surrogates.build(_build(tmp_path, "same", laplace=True))
    assert back.complete
    assert back.summary.calibration_runs == 30
    assert len((back.path / JOURNAL).read_text().splitlines()) == 55

    # Saved, not journalled: the journal lost, a member's file kept.
    second = tmp_path / "second"
    write_case(second)
    path = _build(second, "stop", laplace=True)
    first = surrogates.build(path, max_runs=30)
    assert not first.complete
    journal = first.path / JOURNAL
    kept = [line for line in journal.read_text().splitlines()
            if json.loads(line)["member"] < 25]  # fmt: skip
    journal.write_text("".join(f"{line}\n" for line in kept))
    assert (first.path / "members" / "member_25.npz").is_file()
    with pytest.raises(ConfigError, match="members 25, 26, 27, 28, 29"):
        surrogates.build(_build(second, "stop", laplace=True,
                                design=_design("\ncalibration = 3")))  # fmt: skip


def test_a_plugin_refusing_its_options_is_said_not_hidden(tmp_path: Path) -> None:
    """The capability is asked of the plugin: options it refuses are a ConfigError with its
    message, never a silent count of 0 (which would undercharge the approval)."""
    write_case(tmp_path)
    path = write_build(tmp_path, name="refused",
                       options='{ uncertainty = "laplace", refuse = true }')  # fmt: skip
    path.write_text(path.read_text().replace('"test_affine_build"', '"test_calibrated"'))
    with pytest.raises(ConfigError, match="this option is refused"):
        surrogates.plan_build(surrogates.load_surrogate_config(path))


def test_failed_calibration_runs_fall_back_to_the_validation_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Too many calibration runs failed: the build is not lost after its good runs; the
    plugin gets none (it calibrates on its validation runs) and the failures are reported,
    the fallback said in the summary and the log."""
    write_case(tmp_path, simulator_extra=f"fail_members = {list(range(25, 40))}")
    with caplog.at_level("WARNING"):
        result = surrogates.build(_build(tmp_path, "lossy", laplace=True))
    assert "surrogate_calibration_runs_failed" in {getattr(r, "event", "") for r in caplog.records}
    assert result.complete
    assert result.summary.calibration_fallback == (
        "15 of the 30 calibration runs failed: the spread is calibrated on the validation runs"
    )
    assert result.summary.calibration_runs == 0
    assert result.summary.calibration_failed == list(range(25, 40))
    assert result.summary.failed == []
    manifest = json.loads((result.path / "model" / "surrogate.json").read_text())
    assert manifest["model"]["calibration_ids"] == []


def test_the_resume_charge_counts_runs_of_the_current_design(tmp_path: Path) -> None:
    write_case(tmp_path)
    built = surrogates.build(_build(tmp_path, "charge", laplace=True))
    assert surrogates.runs_made(built.path) == 55
    surrogates.build(_build(tmp_path, "charge", laplace=True, design=_design("\ncalibration = 0")))
    assert surrogates.runs_made(built.path) == 25  # the 30 dropped are not counted


def test_a_runs_forecasts_are_said_to_cost_no_simulator_runs(tmp_path: Path) -> None:
    config = write_case(tmp_path)
    composition.start(config)
    run_dir = tmp_path / "runs" / "case"
    plan = surrogates.plan_build(surrogates.load_surrogate_config(_build(
        tmp_path, "free", laplace=True,
        design=f'plan = "run"\ntraining = 25\nheld_out = 10\ncalibration = 4\n'
        f'run_dir = "{run_dir}"',
    )))  # fmt: skip
    line = next(w for w in plan.warnings if "calibration runs" in w)
    assert line.startswith("4 calibration runs from the run's stored forecasts (no simulator runs)")


def test_a_build_made_without_calibration_runs_gets_them_next(tmp_path: Path) -> None:
    """A complete Laplace build whose design has none (made before they existed, or with
    calibration = 0) is not complete under the automatic count: its next build adds them and
    refits, as its plan counts them."""
    write_case(tmp_path)
    old = surrogates.build(
        _build(tmp_path, "old", laplace=True, design=_design("\ncalibration = 0"))
    )
    assert old.summary.calibration_runs == 0
    plan = surrogates.plan_build(
        surrogates.load_surrogate_config(_build(tmp_path, "old", laplace=True))
    )
    assert plan.calibration == 30
    new = surrogates.build(_build(tmp_path, "old", laplace=True))
    assert new.summary.calibration_runs == 30
    manifest = json.loads((new.path / "model" / "surrogate.json").read_text())
    assert manifest["model"]["calibration_ids"] == list(range(25, 55))


def test_a_complete_build_without_its_plugin(tmp_path: Path) -> None:
    """Uninstalled, the plugin cannot be asked: any automatic count is taken as made; a given
    count other than the design's is not (the build then needs its plugin)."""
    write_case(tmp_path)
    made = surrogates.build(_build(tmp_path, "gone", laplace=True))
    surrogate_models.remove("test_calibrated")
    try:
        assert surrogates.build(_build(tmp_path, "gone", laplace=True)).summary == made.summary
        same = _build(tmp_path, "gone", laplace=True, design=_design("\ncalibration = 30"))
        assert surrogates.build(same).summary == made.summary
        with pytest.raises(ConfigError, match="test_calibrated"):
            surrogates.build(
                _build(tmp_path, "gone", laplace=True, design=_design("\ncalibration = 4"))
            )
    finally:
        surrogate_models.register("test_calibrated")(_Calibrated)


def test_a_recount_on_lhs_is_refused_and_on_a_stored_ensemble_allowed(tmp_path: Path) -> None:
    write_case(tmp_path)
    lhs = 'plan = "lhs"\ntraining = 20\nheld_out = 5'
    assert surrogates.build(_build(tmp_path, "lhs", laplace=True, design=lhs)).complete
    with pytest.raises(ConfigError, match="would reuse their outputs for other inputs"):
        surrogates.build(_build(tmp_path, "lhs", laplace=True, design=lhs + "\ncalibration = 4"))
    np.savez(tmp_path / "ensemble.npz", x=np.random.default_rng(3).standard_normal((3, 60)))
    file = 'plan = "file"\ntraining = 20\nheld_out = 5\nensemble_file = "ensemble.npz"'
    first = surrogates.build(_build(tmp_path, "file", laplace=True, design=file))
    assert first.summary.calibration_runs == 30
    fewer = surrogates.build(
        _build(tmp_path, "file", laplace=True, design=file + "\ncalibration = 4")
    )
    assert fewer.summary.calibration_runs == 4
    # An id is the ensemble's column: the 4 are the first 4 columns of the 30, same inputs.
    before = load_snapshots(first.path / "snapshots.npz")
    after = load_snapshots(fewer.path / "snapshots.npz")
    assert after.calibration_ids == before.calibration_ids[:4]
    assert after.calibration_inputs is not None
    assert before.calibration_inputs is not None
    np.testing.assert_array_equal(
        after.calibration_inputs["x"], before.calibration_inputs["x"][:, :4]
    )


def test_a_complete_build_is_found_when_its_plugin_now_refuses_its_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plugin, updated since the build, refuses the options it was built with: asking it
    for the capability fails, but the complete build is still returned (its runs and model
    are as they were)."""
    write_case(tmp_path)
    made = surrogates.build(_build(tmp_path, "strict", laplace=True))

    original = _Calibrated.__init__

    @functools.wraps(original)  # the same signature: its options still check
    def refusing(self: _Calibrated, *args: object, **kwargs: object) -> None:
        raise ValueError("an option this version refuses")

    monkeypatch.setattr(_Calibrated, "__init__", refusing)
    assert surrogates.build(_build(tmp_path, "strict", laplace=True)).summary == made.summary
    with pytest.raises(ConfigError, match="an option this version refuses"):
        surrogates.plan_build(surrogates.load_surrogate_config(
            _build(tmp_path, "strict", laplace=True)
        ))  # fmt: skip


def test_a_stale_drop_never_accepts_other_inputs(tmp_path: Path) -> None:
    """30 -> 0 -> 30 -> 0 -> 5 and 30 -> 0 -> 30 -> 5: the digest kept is the last runs
    dropped, and 5 draws other inputs under ids already made: refused both times."""
    write_case(tmp_path)
    auto, zero = _design(), _design("\ncalibration = 0")
    five = _design("\ncalibration = 5")
    for design in (auto, zero, auto, zero):
        assert surrogates.build(_build(tmp_path, "stale", laplace=True, design=design)).complete
    with pytest.raises(ConfigError, match="would reuse their outputs for other inputs"):
        surrogates.build(_build(tmp_path, "stale", laplace=True, design=five))
    assert surrogates.build(_build(tmp_path, "stale", laplace=True, design=auto)).complete
    with pytest.raises(ConfigError, match="would reuse their outputs for other inputs"):
        surrogates.build(_build(tmp_path, "stale", laplace=True, design=five))


def test_a_plugin_bug_at_its_creation_is_loud_even_for_a_complete_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a missing plugin or options it refuses leave the count unknown; another error
    of its constructor (a bug) propagates from the completion check, as from a build."""
    write_case(tmp_path)
    surrogates.build(_build(tmp_path, "bug", laplace=True))
    original = _Calibrated.__init__

    @functools.wraps(original)
    def broken(self: _Calibrated, *args: object, **kwargs: object) -> None:
        raise RuntimeError("a bug in the plugin")

    monkeypatch.setattr(_Calibrated, "__init__", broken)
    with pytest.raises(RuntimeError, match="a bug in the plugin"):
        surrogates.build(_build(tmp_path, "bug", laplace=True))


def test_raising_training_on_a_file_design_reuses_the_calibration_runs(tmp_path: Path) -> None:
    """The Brugge build's case (2026-11-12): plan file, training raised in the same folder.
    The old calibration columns fall inside the new training runs and are reused (they are
    full runs); only the new training columns beyond them and the new calibration runs are
    simulated. The result equals a build made with the larger count from the start."""
    write_case(tmp_path)
    np.savez(tmp_path / "ensemble.npz", x=np.random.default_rng(3).standard_normal((3, 45)))
    small = (
        'plan = "file"\ntraining = 20\nheld_out = 5\ncalibration = 5\n'
        'ensemble_file = "ensemble.npz"'
    )
    first = surrogates.build(_build(tmp_path, "grow", laplace=True, design=small))
    made = len((first.path / JOURNAL).read_text().splitlines())
    assert made == 30
    larger = small.replace("training = 20", "training = 30")
    grown = surrogates.build(_build(tmp_path, "grow", laplace=True, design=larger))
    assert grown.complete
    # 30 training (20 old + the 5 old calibration + 5 new) and 5 new calibration runs.
    assert len((grown.path / JOURNAL).read_text().splitlines()) == made + 10
    fresh = surrogates.build(_build(tmp_path, "fresh", laplace=True, design=larger))
    assert fresh.summary.gate == grown.summary.gate
    assert fresh.summary.calibration_runs == grown.summary.calibration_runs == 5
