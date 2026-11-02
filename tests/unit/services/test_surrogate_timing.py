"""A surrogate run's forecast times against the full model's (decision 14)."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.domain.surrogate import BuildCost
from apex.services.summaries import _surrogate_timing
from apex.services.surrogates import model_cost
from apex.storage.surrogate_directory import BUILD_FINGERPRINT, MemberRecord, SurrogateDirectory
from apex.storage.surrogate_files import cost_path, read_cost, write_cost
from apex.workflows.history_matching import HistoryMatchingWorkflow
from apex.workflows.surrogate_forecasts import forecast_timing

MODEL = "a" * 64
COST = BuildCost(
    identity=MODEL, runs=4, mean_seconds=30.0, median_seconds=28.0, total_seconds=120.0
)


def _done(label: str, seconds: float, n_ok: int, workers: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "event": "forecast_done", "label": label, "seconds": seconds, "n_ok": n_ok, "n_failed": 0
    }  # fmt: skip
    if workers is not None:
        record["workers"] = workers
    return record


def test_timing_sums_each_source_in_worker_seconds() -> None:
    """Surrogate forecasts by label, the rest the simulator's; worker-seconds per member."""
    events = [
        {"event": "surrogate_forecast", "seconds": 99.0},  # not a forecast_done: ignored
        _done("surrogate", 2.0, 10, workers=2),
        _done("surrogate", 1.0, 10, workers=2),
        _done("ensemble (outside)", 60.0, 2, workers=2),
        _done("posterior (simulator)", 90.0, 10, workers=5),
    ]
    timing = forecast_timing(events, COST)
    surrogate, simulator = timing["surrogate"], timing["simulator"]
    assert (surrogate["forecasts"], surrogate["members"], surrogate["seconds"]) == (2, 20, 3.0)
    assert surrogate["worker_seconds_per_member"] == pytest.approx(6.0 / 20)
    assert (simulator["members"], simulator["worker_seconds"]) == (12, 60.0 * 2 + 90.0 * 5)
    assert timing["speedup"] == pytest.approx(30.0 / 0.3)
    assert timing["simulator_equivalent_worker_seconds"] == 20 * 30.0
    assert timing["run_seconds"] == 153.0
    assert timing["run_worker_seconds"] == 6.0 + 570.0
    assert timing["full_model"]["total_seconds"] == 120.0


def test_timing_without_workers_or_cost_gives_no_speedup() -> None:
    """An executor stating no workers: wall time only; no build cost: no comparison."""
    timing = forecast_timing([_done("surrogate", 2.0, 4, 1), _done("ensemble", 8.0, 4)], None)
    assert timing["simulator"]["worker_seconds"] is None
    assert timing["run_worker_seconds"] is None
    assert timing["speedup"] is None
    assert timing["full_model"] is None
    assert timing["simulator_equivalent_worker_seconds"] is None


def test_the_cost_lives_beside_the_manifest(tmp_path: Path) -> None:
    """Written and read back beside the manifest; absent is None (a refit, an older build)."""
    manifest = tmp_path / "model" / "surrogate.json"
    manifest.parent.mkdir()
    assert read_cost(manifest) is None
    write_cost(manifest, COST)
    assert cost_path(manifest) == tmp_path / "model" / "surrogate.cost.json"
    assert read_cost(manifest) == COST


def test_an_older_build_gives_its_cost_from_the_journal(tmp_path: Path) -> None:
    """Without the file, the build the model is in gives its successful runs' times."""
    (tmp_path / "state.json").write_text(json.dumps({BUILD_FINGERPRINT: "x", "identity": MODEL}))
    directory = SurrogateDirectory(tmp_path)
    directory.record([
        MemberRecord(0, True, 10.0), MemberRecord(1, True, 20.0),
        MemberRecord(2, False, None, "failed"), MemberRecord(3, True, float("nan")),
    ])  # fmt: skip
    cost = model_cost(tmp_path / "model" / "surrogate.json", MODEL)
    assert cost is not None
    assert (cost.runs, cost.mean_seconds, cost.total_seconds) == (2, 15.0, 30.0)
    assert model_cost(tmp_path / "elsewhere" / "x" / "surrogate.json", MODEL) is None
    # A model copied into another build's folder takes none of its journal.
    assert model_cost(tmp_path / "model" / "surrogate.json", "b" * 64) is None


def test_another_models_or_a_corrupt_cost_is_passed_over(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The cost only informs: another model's file or a torn one is logged, never fatal."""
    manifest = tmp_path / "model" / "surrogate.json"
    manifest.parent.mkdir()
    write_cost(manifest, COST)
    assert model_cost(manifest, MODEL) == COST
    with caplog.at_level(logging.WARNING):
        assert model_cost(manifest, "b" * 64) is None
        cost_path(manifest).write_text("{not json")
        assert model_cost(manifest, MODEL) is None
    assert [r.event for r in caplog.records].count("surrogate_cost_skipped") == 2  # type: ignore[attr-defined]


def test_a_sub_millisecond_surrogate_keeps_its_speedup() -> None:
    """Unrounded seconds: a fast surrogate's forecast still gives a speed-up and a note."""
    timing = forecast_timing([_done("surrogate", 2e-4, 100, 1)], COST)
    assert timing["speedup"] == pytest.approx(30.0 / 2e-6)
    note = _surrogate_timing(timing)
    assert note is not None
    assert "speed-up 1.5e+07x" in note


def test_concurrency_is_capped_by_the_simulators_licences() -> None:
    """Members at once: the executor's workers, the simulator's licences, the members."""
    workflow = SimpleNamespace(
        executor=SimpleNamespace(workers=10), simulator=SimpleNamespace(max_parallel_runs=3)
    )
    assert HistoryMatchingWorkflow._concurrency(workflow, 50) == {"workers": 3}  # type: ignore[arg-type]
    workflow.simulator = SimpleNamespace()
    assert HistoryMatchingWorkflow._concurrency(workflow, 4) == {"workers": 4}  # type: ignore[arg-type]
    workflow.executor = SimpleNamespace()  # an executor placing members elsewhere
    assert HistoryMatchingWorkflow._concurrency(workflow, 4) == {}  # type: ignore[arg-type]


def test_the_summary_note_states_the_speedup() -> None:
    """``apex summary``'s note: members, per-member time, the full model's, the speed-up."""
    note = _surrogate_timing(forecast_timing([_done("surrogate", 2.0, 20, 2)], COST))
    assert note is not None
    for part in ("20 members", "speed-up 150x", "120 s once"):
        assert part in note
    assert _surrogate_timing(None) is None
    # The simulator's executor states no workers: no worker-s against wall-s comparison.
    wall = _surrogate_timing(
        forecast_timing([_done("surrogate", 2.0, 20, 2), _done("ensemble", 9.0, 5)], COST)
    )
    assert wall is not None
    assert "speed-up" in wall
    assert "would take" not in wall
    # An untimed (remote) build: the surrogate's own times, no comparison.
    untimed = _surrogate_timing(forecast_timing([_done("surrogate", 2.0, 20, 2)], None))
    assert untimed is not None
    assert "full model" not in untimed
