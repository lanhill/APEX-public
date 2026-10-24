"""Run summaries (``apex summary``): performance metrics at every step of every run."""

import csv
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from typer.testing import CliRunner

from apex.cli.main import app
from apex.core.errors import ConfigError
from apex.reporting.run_summary import summary_csv, summary_markdown
from apex.services import composition
from apex.services.summaries import find_runs, summarize

LM = (
    'name = "lm_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
    "schedule = { initial = 100.0, data_mismatch_tolerance = 1e-6 }\nmax_iterations = 15"
)
TANH = 'nonlinearity = "tanh"'
MODEL_CHANGE = '[[metrics]]\nname = "rmse"\n\n[[metrics]]\nname = "model_change"'


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]


def _run(linear_case: CaseFactory, name: str, failures: str = "") -> Path:
    case = linear_case(
        name=name, method=LM, ensemble_size=20, simulator_extra=TANH, failures=failures
    )
    composition.start(case.config)
    return case.directory / "runs" / name


def _status(path: Path) -> str:
    summary = json.loads((path / "summary.json").read_text())
    return "converged" if summary["converged"] else "complete"


def test_every_accepted_step_and_metric_is_summarised(linear_case: CaseFactory) -> None:
    path = _run(linear_case, "lm")
    results = composition.load_results(path)

    (summary,) = summarize([path.parent.parent])

    assert summary.workflow == "history_matching"
    assert summary.method == "lm_enrml (approx)"
    assert summary.status == _status(path)
    assert [s.iteration for s in summary.steps] == results.accepted_iterations
    assert sum(s.rejected_before for s in summary.steps) == len(results.history) - len(
        results.accepted_iterations
    )
    assert summary.selected_iteration == results.final_iteration
    recorded = results.metric_summary()
    for row in recorded:
        value = summary.value(str(row["metric"]), str(row["label"]), int(str(row["iteration"])))
        assert value is not None
        assert (value.mean, value.p90) == (row["mean"], row["p90"])


def _forget_model_change(path: Path) -> None:
    """Make ``path`` look like a run from before model_change was always recorded."""
    for data_mismatch in (path / "checkpoints").glob("data_mismatch_*.npz"):
        with np.load(data_mismatch) as data:
            kept = {k: data[k] for k in data.files if not k.startswith("metric:model_change:")}
        np.savez(data_mismatch, **kept)
    history = json.loads((path / "history.json").read_text())
    for record in history:
        (record.get("metrics") or {}).pop("model_change", None)
    (path / "history.json").write_text(json.dumps(history))


def test_model_change_after_the_run_equals_the_recorded_one(linear_case: CaseFactory) -> None:
    """The same run twice: once as recorded, once with model_change removed (an older run),
    so the summary computes it from the iterates."""
    recorded = summarize([_run(linear_case, "with", failures=MODEL_CHANGE)])[0]
    older = _run(linear_case, "without", failures='[[metrics]]\nname = "rmse"')
    _forget_model_change(older)
    computed = summarize([older])[0]

    assert computed.notes
    assert not recorded.notes
    rows = [r for r in recorded.metrics if r.metric == "model_change"]
    assert rows
    assert {r.iteration for r in rows} == {s.iteration for s in recorded.steps[1:]}
    for row in rows:
        value = computed.value("model_change", row.label, row.iteration)
        assert value is not None
        np.testing.assert_allclose(value.mean, row.mean, rtol=1e-13)
        np.testing.assert_allclose(value.std, row.std, rtol=1e-12)


def test_markdown_and_csv(linear_case: CaseFactory, tmp_path: Path) -> None:
    path = _run(linear_case, "lm", failures=MODEL_CHANGE)
    summaries = summarize([path])
    (summary,) = summaries

    text = summary_markdown(summaries, title="T", root=path.parent)
    table = list(csv.DictReader(io.StringIO(summary_csv(summaries, root=path.parent))))

    assert text.startswith("# T\n")
    assert f"| lm | lm_enrml (approx) | {_status(path)} |" in text
    assert f"| {summary.step_of(summary.selected_iteration)} (result) |" in text
    assert "model_change" in text
    assert "rmse" in text
    assert len(table) == len(summary.metrics)
    assert {row["path"] for row in table} == {"lm"}
    assert sum(int(row["result"]) for row in table) == len(
        [r for r in summary.metrics if r.iteration == summary.selected_iteration]
    )


def test_the_search_finds_runs_and_skips_other_folders(
    linear_case: CaseFactory, tmp_path: Path
) -> None:
    first, second = _run(linear_case, "a"), _run(linear_case, "b")
    (tmp_path / "empty").mkdir()

    assert find_runs([tmp_path]) == [first, second]
    assert find_runs([first, first]) == [first]
    with pytest.raises(ConfigError, match="no run directory"):
        summarize([tmp_path / "empty"])


def test_the_command_writes_markdown_and_csv(linear_case: CaseFactory, tmp_path: Path) -> None:
    path = _run(linear_case, "lm")
    output = tmp_path / "out" / "summary.md"

    shown = CliRunner().invoke(app, ["summary", str(path)])
    written = CliRunner().invoke(app, ["summary", str(path), "--output", str(output)])

    assert shown.exit_code == 0
    assert "## lm" in shown.output
    assert written.exit_code == 0
    assert output.read_text() == shown.output
    assert output.with_suffix(".csv").read_text().startswith("workflow,run,path")
    missing = CliRunner().invoke(app, ["summary", str(tmp_path / "nothing")])
    assert missing.exit_code == 1
    assert "no run directory" in missing.output


def test_runs_with_the_same_name_are_labelled_by_case(tmp_path: Path) -> None:
    from apex.domain.summary import RunSummary
    from apex.reporting.run_summary import run_labels

    def run(path: str) -> RunSummary:
        return RunSummary("history_matching", "gies", tmp_path / path, "gies", "complete", (), ())

    labels = run_labels([run("5Spots/runs/gies"), run("M5/runs/gies"), run("x/gies")])

    assert labels == ["5Spots/gies", "M5/gies", "x/gies"]


def test_runs_of_one_configuration_are_labelled_by_their_folders(tmp_path: Path) -> None:
    from apex.domain.summary import RunSummary
    from apex.reporting.run_summary import run_labels

    def run(path: str) -> RunSummary:
        return RunSummary("history_matching", "linear_esmda", tmp_path / path, "esmda",
                          "complete", (), ())  # fmt: skip

    labels = run_labels([run("case/runs/first_full_run"), run("case/runs/linear_esmda"),
                         run("a/case/runs/x"), run("b/case/runs/x")])  # fmt: skip

    assert labels[:2] == ["case/first_full_run", "case/linear_esmda"]
    assert labels[2:] == [str(tmp_path / "a/case/runs/x"), str(tmp_path / "b/case/runs/x")]


def test_relative_run_paths_get_no_leading_slash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release tester: ``apex summary copy_of_run`` labelled a run ``/copy_of_run`` (the
    relative path's parent ``.`` has an empty name)."""
    from apex.domain.summary import RunSummary
    from apex.reporting.run_summary import run_labels

    monkeypatch.chdir(tmp_path)

    def run(path: str) -> RunSummary:
        return RunSummary("history_matching", "gies", Path(path), "gies", "complete", (), ())

    labels = run_labels([run("copy_of_run"), run("runs/gies")])

    assert labels == [f"{tmp_path.name}/copy_of_run", f"{tmp_path.name}/gies"]


def test_a_prediction_is_summarised_with_its_run(linear_case: CaseFactory) -> None:
    """One step per ensemble; the final values over the members, and the scores
    against the truth as single values (with their mean over vectors)."""
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    rng = np.random.default_rng(7)
    g_pred = np.vstack([case.forward, rng.standard_normal((4, case.forward.shape[1]))])  # type: ignore[attr-defined]
    np.savez(case.directory / "forward_future.npz", matrix=g_pred)
    outputs = ", ".join(f'["t{i}", "y"]' for i in range(12))
    (case.directory / "future.toml").write_text(
        f"""
[prediction]
name = "future"
run_dir = "runs/case"
ensembles = ["posterior", "prior", "truth"]

[prediction.outputs]
vectors = ["y"]

[simulator]
options = {{ forward_file = "forward_future.npz", outputs = [{outputs}] }}
"""
    )
    from apex.services import prediction

    predicted = prediction.predict(case.directory / "future.toml")
    run = case.directory / "runs" / "case"

    found = summarize([case.directory])

    assert [s.workflow for s in found] == ["history_matching", "prediction"]
    (summary,) = [s for s in found if s.workflow == "prediction"]
    assert [s.label for s in summary.steps] == ["posterior", "prior"]
    assert summary.selected_iteration == 0
    final = summary.value("final_value", "y", 0)
    posterior = predicted.forecasts["posterior"].values[0, -1, :]
    assert final is not None
    assert final.mean == pytest.approx(posterior.mean())
    crps = summary.value("crps", "y", 0)
    assert crps is not None
    assert crps.mean == pytest.approx(predicted.scores["posterior"]["y"]["crps"])
    assert summary.value("crps", "all", 0) is not None
    text = summary_markdown(found, root=case.directory)
    assert "| posterior (result) |" in text
    assert str(run) not in text  # paths relative to the root


# ------------------------------------------------------------------------- optimizations

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _optimize(tmp_path: Path, example: str, config: str, max_iterations: int | None) -> Path:
    from apex.services import optimization
    from apex.services.examples import copy_example, get_example

    copied = copy_example(get_example(EXAMPLES, example), tmp_path)
    [path] = [p for _, _, p in copied.configs if p.name == config]
    return optimization.optimize(path, max_iterations=max_iterations).run_dir


def test_an_optimization_is_summarised_step_by_step(tmp_path: Path) -> None:
    """The user (2026-10-10): ``apex summary`` covers optimizations too."""
    from apex.storage import optimization_directory as layout

    run_dir = _optimize(tmp_path, "optimization", "rosenbrock_enopt.toml", None)
    history = layout.read_history(run_dir)

    [summary] = summarize([tmp_path])

    assert summary.workflow == "optimization"
    assert summary.method == "enopt on rosenbrock"
    assert summary.status == "complete"
    assert len(summary.steps) == len(history)
    assert summary.steps[0].label == "start"
    assert summary.selected_iteration == len(history) - 1
    values = [summary.value("objective", "all", s.iteration) for s in summary.steps]
    assert [v.mean for v in values if v is not None] == [
        pytest.approx(r["objective"]) for r in history]  # fmt: skip
    assert any(note.startswith("stopped:") for note in summary.notes)
    assert "objective" in summary_markdown([summary])

    # A robust optimization's result: the objective over the realizations, at the last step.
    result = json.loads((run_dir / layout.RESULT).read_text())
    result["objective_per_realization"] = [1.0, 2.0, 3.0, 4.0]
    (run_dir / layout.RESULT).write_text(json.dumps(result))
    [again] = summarize([run_dir])
    last = again.steps[-1].iteration
    spread = again.value("objective", "realizations", last)
    assert spread is not None
    assert spread.mean == pytest.approx(2.5)
    assert spread.p10 == pytest.approx(1.3)


def test_a_multi_objective_optimization_reports_its_front(tmp_path: Path) -> None:
    _optimize(tmp_path, "toy/toy_reservoir_robust_npv", "robust_npv_nsga2.toml", 2)

    [summary] = summarize([tmp_path])

    assert summary.status == "stopped"
    assert summary.selected_iteration is None  # not finished: no result yet
    last = summary.steps[-1].iteration
    for metric in ("hypervolume", "front_size"):
        assert summary.value(metric, "all", last) is not None
    assert len(summary.labels()["objective"]) >= 3  # the representative and each objective
    assert summary.steps[-1].n_members > 0


def test_every_workflow_has_a_summarizer() -> None:
    """The user's rule (2026-10-10): ``apex summary`` is extended to every new workflow
    that writes a result directory (risk assessment, ...). A new module in
    ``apex.workflows`` needs a summarizer of its name, or a place among the helpers."""
    import apex.services.summaries  # noqa: F401  (registers the summarizers)
    from apex import workflows
    from apex.core.plugins import summaries

    helpers = {
        "data_split": "chooses a cross-validation split; writes no run",
        "geophysics": "the AVO forward model inside history-matching runs",
        "model_error": "the model-error (MEC) model inside history-matching runs",
        "parameterization": "maps state to properties inside other workflows",
        "simulated_objective": "evaluates an optimization's objective by simulation",
    }
    folder = Path(workflows.__file__).parent
    modules = {p.stem for p in folder.glob("*.py")} - {"__init__"}
    modules |= {p.parent.name for p in folder.glob("*/__init__.py")}  # packages too

    missing = modules - helpers.keys() - set(summaries.names())
    assert not missing, f"workflows without a summarizer (services/summaries.py): {missing}"
    assert set(summaries.names()) >= {"history_matching", "prediction", "optimization"}


def test_report_says_a_job_folder_holds_no_run_yet(tmp_path: Path) -> None:
    """Parked pages (2026-10-10): ``apex report`` on an optimization job's folder before
    the job started fell through to a history match's report, with a misleading error."""
    folder = tmp_path / "optimizations" / "x" / "abc"
    folder.mkdir(parents=True)

    result = CliRunner().invoke(app, ["report", str(folder)])

    assert result.exit_code == 1
    assert "holds no run yet" in result.output
    assert "data mismatch" not in result.output  # not a history match's table


def test_a_started_optimization_without_a_step_is_known_but_not_summarised(
    tmp_path: Path,
) -> None:
    """Review finding: an optimization evaluating its start has config.json but no history
    yet, and was taken for a history match's directory."""
    from apex.storage import optimization_directory as layout

    run_dir = _optimize(tmp_path, "optimization", "rosenbrock_enopt.toml", 1)
    (run_dir / layout.HISTORY).unlink()

    assert find_runs([tmp_path]) == [run_dir]
    assert summarize([tmp_path]) == []  # skipped with a warning: no step yet
    result = CliRunner().invoke(app, ["report", str(run_dir)])
    assert "data mismatch" not in result.output


def test_a_half_written_last_history_row_is_left_out(tmp_path: Path) -> None:
    """Review finding: a running optimization's torn last line stopped the summary."""
    from apex.storage import optimization_directory as layout

    run_dir = _optimize(tmp_path, "optimization", "rosenbrock_enopt.toml", 2)
    rows = len(layout.read_history(run_dir))
    with (run_dir / layout.HISTORY).open("a") as history:
        history.write('{"kind": "step", "obj')

    [summary] = summarize([run_dir])
    assert len(summary.steps) == rows


def test_a_deterministic_optimizers_steps_count_its_function_evaluations(
    tmp_path: Path,
) -> None:
    """Review finding: line search counts no member evaluations, so every step had 0."""
    _optimize(tmp_path, "optimization", "rosenbrock_trust_region.toml", 3)

    [summary] = summarize([tmp_path])

    assert summary.method.startswith("trust_region")
    assert sum(s.n_members for s in summary.steps) == summary.forecasts > 0


def test_steps_are_numbered_without_the_rejected_trials(tmp_path: Path) -> None:
    """A run whose records 2 and 3 were rejected trials: its accepted iterates are steps
    0, 1, 2, 3 in the report, the CSV and the REST document, not 0, 1, 4, 5 (the user,
    2026-10-17: a 10-iteration run ended at "step 12")."""
    from apex.domain.summary import MetricStatistics, RunSummary, StepSummary
    from apex.reporting.run_summary import summary_csv, summary_markdown
    from apex.services.api import steps_document

    steps = (StepSummary(0, 10), StepSummary(1, 10), StepSummary(4, 10, rejected_before=2),
             StepSummary(5, 10))  # fmt: skip
    metrics = tuple(MetricStatistics("rmse", "all", k, 1.0 / (k + 1), 0.1, 1.0, 0.5, 1.5)
                    for k in (0, 1, 4, 5))  # fmt: skip
    summary = RunSummary("history_matching", "gappy", tmp_path / "gappy", "gies (rlmmac)",
                         "complete", steps, metrics, selected_iteration=5)  # fmt: skip

    assert [summary.step_of(s.iteration) for s in steps] == [0, 1, 2, 3]
    assert summary.step_of(3) is None  # a rejected trial is no step
    document = steps_document(summary)
    assert [(s.iteration, s.step) for s in document.steps] == [(0, 0), (1, 1), (4, 2), (5, 3)]
    assert (document.selected_iteration, document.selected_step) == (5, 3)
    report = summary_markdown([summary])
    assert "result: step 3" in report
    assert "| 3 (result) |" in report
    assert "| 5 (result) |" not in report
    rows = summary_csv([summary]).splitlines()
    header = rows[0].split(",")
    assert [r.split(",")[header.index("step")] for r in rows[1:]] == ["0", "1", "2", "3"]


def test_the_summary_states_its_std_and_it_is_the_sample_std(linear_case: CaseFactory) -> None:
    """Release tester (2026-10-20): ``apex report`` footnotes the methods' population std
    (ddof = 0), and ``apex summary`` showed a slightly larger std without saying which. The
    summary's is the sample std (ddof = 1), as the API's and the figures', and says so."""
    path = _run(linear_case, "lm")
    (summary,) = summarize([path])
    with np.load(path / "checkpoints" / "data_mismatch_0.npz") as data:
        members = data["metric:data_mismatch:all"]

    value = summary.value("data_mismatch", "all", 0)

    assert value is not None
    np.testing.assert_allclose(value.std, members.std(ddof=1), rtol=1e-12)
    assert "sample std over the members (ddof = 1" in summary_markdown([summary])
    report = CliRunner().invoke(app, ["report", str(path)])
    assert "`apex summary` gives the sample std (ddof = 1)" in report.output
