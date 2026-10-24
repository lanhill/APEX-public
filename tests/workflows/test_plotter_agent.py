"""The plotter agent: its rules, the model's plans held to the run, and its
front ends (CLI, REST, the assistant's tool), on a whole linear-Gaussian run."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

pytest.importorskip("matplotlib")

from typer.testing import CliRunner

from apex.agents.plotter import FigureChoice, PlotPlan, Plotter
from apex.agents.providers.stub import StubProvider
from apex.cli.main import app
from apex.core.llm import LLMUnavailable
from apex.services import composition
from apex.services.api import ApexService
from apex.services.settings import PlotterSection


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]
#: Small figures: the tests check what is drawn, not the resolution.
SMALL = PlotterSection(dpi_scale=0.05)


def _run(linear_case: CaseFactory) -> Path:
    case = linear_case(ensemble_size=20)
    composition.start(case.config)
    return case.directory / "runs" / "case"


# --------------------------------------------------------------------- on a whole run


def test_the_plotter_draws_the_classic_set_of_a_run(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    run_dir = _run(linear_case)
    service = ApexService(tmp_path / "service")

    outcome = Plotter(service, None, settings=SMALL).plot(run_dir)

    assert outcome.decided_by == "rules"
    output = run_dir / "report" / "plots"
    summary = json.loads((output / "summary.json").read_text())
    drawn = {f["subdir"] for f in summary["figures"] if f["files"]}
    # The linear case has no grid, no ssim, no residuals and no network: those are
    # skipped, each with its reason; model_change is recorded but not drawn by default.
    assert drawn == {"obj", "data_mismatch", "rmse", "simData"}
    assert {s["subdir"] for s in summary["skipped"]} == {"mssim", "residual", "models", "nn_hist"}
    assert all(s["reason"] for s in summary["skipped"])
    assert (output / "obj" / "obj_func_boxplot.png").is_file()
    assert (output / "simData" / "y_initial.png").is_file()
    assert list((run_dir / "agents").glob("plotter-*-decision.json"))


def test_the_models_plan_is_used_once_it_holds_and_the_rules_otherwise(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    run_dir = _run(linear_case)
    service = ApexService(tmp_path / "service")
    wrong = PlotPlan(figures=[FigureChoice(figure="pie_chart")]).model_dump_json()
    right = PlotPlan(
        figures=[FigureChoice(figure="metric_distribution", options={"metric": "rmse"},
                              subdir="rmse")],
        reason="the user asked for the RMSE",
    ).model_dump_json()  # fmt: skip

    revised = Plotter(service, StubProvider([wrong, right]), settings=SMALL).plot(
        run_dir, request="rmse"
    )
    failed = Plotter(service, StubProvider([LLMUnavailable("down")]), settings=SMALL).plot(
        run_dir, request="the objective"
    )

    assert revised.decided_by == "model"
    assert [d.figure for d in revised.result.drawn] == ["metric_distribution"]
    assert revised.result.files[0].startswith("rmse/rmse_")
    assert failed.decided_by == "fallback"
    assert [d.figure for d in failed.result.drawn] == ["objective_distribution"]


def test_the_command_draws_a_preset(tmp_path: Path, linear_case: CaseFactory) -> None:
    run_dir = _run(linear_case)
    output = tmp_path / "figures"
    (tmp_path / "svc").mkdir()
    (tmp_path / "svc" / "service.toml").write_text("[agents.plotter]\ndpi_scale = 0.05\n")

    root = str(tmp_path / "svc")
    result = CliRunner().invoke(app, ["agent", "plot", str(run_dir), "--preset", "brugge_classic",
                                      "--output", str(output), "--root", root])  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "figures (PNG and PDF)" in result.output
    assert (output / "obj" / "obj_func_violinplot.png").is_file()
    assert not (output / "model_change").exists()  # recorded, drawn only when asked for
    asked = CliRunner().invoke(app, ["agent", "plot", str(run_dir), "--preset", "brugge_classic",
                                     "--with", "model_change", "--output", str(output),
                                     "--root", root])  # fmt: skip
    assert asked.exit_code == 0, asked.output
    assert (output / "model_change" / "model_change_all_boxplot.png").is_file()
    unknown = CliRunner().invoke(app, ["agent", "plot", str(run_dir), "--preset", "pie",
                                       "--root", root])  # fmt: skip
    assert unknown.exit_code == 1
    assert "unknown preset 'pie'" in unknown.output


def test_the_route_and_the_assistants_tool(tmp_path: Path, linear_case: CaseFactory) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from apex.agents.chat import ChatAssistant
    from apex.api.app import create_app
    from apex.services.auth import ui_token
    from apex.storage.chat_store import ChatStore

    case = linear_case(name="alpha", ensemble_size=20)
    root = tmp_path / "root"
    root.mkdir()
    (root / "service.toml").write_text(
        f'[workspace]\nroots = ["{tmp_path}"]\n[agents.plotter]\ndpi_scale = 0.05\n'
    )
    (root / "ui_access.toml").write_text('[workspace]\nfolders = ["alpha"]\n'
                                         '[agents]\nnames = ["plotter"]\n')  # fmt: skip
    service = ApexService(root)
    handle = service.submit_run(case.config)
    assert service.wait(handle.id, timeout_s=300, poll_s=0.2).state == "complete"
    http = TestClient(create_app(root, token="t"))
    owner = {"Authorization": "Bearer t"}
    guest = {"Authorization": f"Bearer {ui_token(root, create=True, guest='g')}",
             "X-APEX-Client": "ui"}  # fmt: skip

    started = http.post("/v1/agents/plot", headers=owner,
                        json={"run": handle.run_ids[0], "request": "the objective"})  # fmt: skip
    theirs = http.post("/v1/agents/plot", headers=guest, json={"run": handle.run_ids[0]})

    assert started.status_code == 200, started.text
    body = started.json()
    assert (body["state"], body["requests"]) == ("drawing", 1)
    summary = _finished(Path(service.store.find(handle.run_ids[0]).path))  # type: ignore[union-attr]
    assert summary["written"] == 2
    figures = http.get(f"/v1/runs/{handle.run_ids[0]}/figures", headers=owner).json()
    assert any("plots/obj/obj_func_boxplot.png" in json.dumps(f) for f in figures)
    assert theirs.status_code == 404  # not the guest's run
    state = http.get(f"/v1/agents/plot/{handle.run_ids[0]}", headers=guest)
    assert state.status_code == 404  # nor its drawing's state
    assert http.get(f"/v1/agents/plot/{handle.run_ids[0]}", headers=owner).json()["state"] == "done"

    assistant = ChatAssistant(service, ChatStore(root), None)
    tool = {t.name: t for t in assistant.local_tools(isolate=False)}["plot_run"]
    answer = tool.handler(tool.arguments(run=handle.run_ids[0], request="rmse"))
    assert answer["figures"] == ["metric_distribution (rmse)"]
    assert _finished(Path(service.store.find(handle.run_ids[0]).path))["written"] == 4  # type: ignore[union-attr]


def _finished(run_dir: Path, timeout_s: float = 120.0) -> dict[str, object]:
    """Wait for a background drawing to finish; return its summary."""
    import time

    from apex.services.figures import drawing_state

    output = run_dir / "report" / "plots"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = drawing_state(output)
        if found["state"] in ("done", "failed"):
            assert found["state"] == "done", found
            return found
        time.sleep(0.2)
    raise AssertionError(f"the drawing did not finish: {drawing_state(output)}")


def test_a_run_with_a_grid_active_cells_and_residuals_through_the_adaptor(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """The real view (``services/figure_view.py``): grids with ``actnum`` from
    ``[[metrics]]``, auxiliary forecasts read back as keys, the posterior iterate."""
    import numpy as np

    from apex.reporting.figures.classic import FieldLayers
    from apex.services.figure_view import _holding, _parse_key, run_view

    case = linear_case(ensemble_size=20)
    np.save(case.directory / "actnum.npy", np.array([1, 0, 1]))
    text = (
        case.config.read_text().replace(
            'name = "x"\nsize = 3\n', 'name = "x"\nsize = 3\ngrid = [3, 1, 1]\n'
        )
        + '\n[[metrics]]\nname = "rmse"\noptions = { actnum_file = "actnum.npy" }\n'
    )
    case.config.write_text(text)
    composition.start(case.config)
    run_dir = case.directory / "runs" / "case"
    view = run_view(run_dir)
    for k in (0, view.posterior_iteration):
        path = run_dir / "checkpoints" / f"iter_{k}.npz"
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        n = arrays["member_ids"].size
        arrays["auxiliary_predicted"] = np.full((2, n), float(k))
        arrays["auxiliary_keys"] = np.array(["MEC:WOPR:P/1@2008-01-02", "MEC:WOPR:P/1@2008-02-01"])
        np.savez(path, **arrays)
    view = run_view(run_dir)

    grid = view.grid("x")
    assert grid is not None
    assert grid.dims == (3, 1, 1)
    assert grid.actnum is not None
    assert grid.actnum.tolist() == [True, False, True]
    final = view.iterate(view.posterior_iteration)
    assert [str(k) for k in final.auxiliary_keys] == [
        "MEC:WOPR:P/1@2008-01-02", "MEC:WOPR:P/1@2008-02-01"]  # fmt: skip
    assert view.start is None  # no deck
    assert view.observations().keys[0].datatype == "y"
    assert _parse_key("WOPR:P1@2008-01-02[3]").index == 3
    assert _holding({"inner": {"deck_template_file": "d.mako"}}, "deck_template_file") == {
        "deck_template_file": "d.mako"}  # fmt: skip
    written = FieldLayers(kinds=["ref"]).draw(view, tmp_path / "maps")
    assert [p.name for p in written] == ["x_ref_layer1.png"]

    # A well named with a slash stays inside the output folder.
    from apex.reporting.figures.classic import DataBands

    residuals = DataBands(source="auxiliary").draw(view, tmp_path / "residual")
    assert sorted(p.name for p in residuals) == ["P_1_WOPR.png", "P_1_WOPR_initial.png"]
    assert all(p.parent == tmp_path / "residual" for p in residuals)


def test_a_prefix_cannot_name_a_folder() -> None:
    from apex.core.plugins import figures

    for bad in ("../up", "/etc/x", "a\\b"):
        with pytest.raises(ValueError, match="plain file-name prefix"):
            figures.create("metric_distribution", metric="rmse", prefix=bad)
    with pytest.raises(ValueError, match="bins must be 1 to 1000"):
        figures.create("state_histogram", variable="nn_params", bins=10**9)


def test_one_drawing_per_run_and_a_dead_or_failed_drawer_is_told(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Second review (2026-10-11): two starts raced, and a drawer killed before it wrote
    its end left the run 'drawing' for ever."""
    import os

    from apex.core.errors import ConfigError
    from apex.core.figures import FigureRequest
    from apex.services.figures import _try_lock, draw, drawing_state, start_drawing

    run_dir = _run(linear_case)
    output = run_dir / "report" / "plots"
    output.mkdir(parents=True)
    held = _try_lock(output)  # as a running drawer holds it
    assert held is not None
    (output / "summary.json").write_text('{"state": "drawing"}')
    with pytest.raises(ConfigError, match="being drawn"):
        start_drawing(run_dir, [FigureRequest("objective_distribution")], dpi_scale=0.05)
    with pytest.raises(ConfigError, match="being drawn"):  # nor in the foreground
        draw(run_dir, [FigureRequest("objective_distribution")], dpi_scale=0.05)
    assert drawing_state(output)["state"] == "drawing"
    os.close(held)  # the drawer died without writing its end
    assert drawing_state(output) == {
        "state": "failed", "error": "the drawing process ended before it finished"}  # fmt: skip

    # A drawing that fails says so, without the server's path.
    start_drawing(run_dir, [FigureRequest("objective_distribution", {"font_size": 0})],
                  dpi_scale=0.05)  # fmt: skip
    summary = _finished(run_dir)
    assert summary["skipped"] == [{"figure": "objective_distribution", "subdir": "",
                                   "reason": "objective_distribution: font_size must be in "
                                             "(0, 100], got 0"}]  # fmt: skip
    start_drawing(run_dir, [FigureRequest("objective_distribution")], dpi_scale=0.05)
    assert _finished(run_dir)["written"] == 2
    assert str(tmp_path) not in json.dumps(drawing_state(output))


def test_drawings_beyond_the_machines_slots_are_refused(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    """Fourth review (2026-10-12): a guest could start a drawing for each of their runs."""
    import os

    from apex.core.errors import ConfigError
    from apex.core.figures import FigureRequest
    from apex.execution.resources import default_root
    from apex.services.figures import SLOTS_DIR, _take_slot, start_drawing

    run_dir = _run(linear_case)
    held = _take_slot(1)  # as another run's drawing holds the only slot
    assert held is not None
    try:
        assert _take_slot(1) is None
        with pytest.raises(ConfigError, match="drawing 1 runs' figures already"):
            start_drawing(run_dir, [FigureRequest("objective_distribution")], dpi_scale=0.05,
                          max_drawings=1)  # fmt: skip
        # The run's own lock was given back: a drawing with a free slot starts.
        start_drawing(run_dir, [FigureRequest("objective_distribution")], dpi_scale=0.05,
                      max_drawings=2)  # fmt: skip
        assert _finished(run_dir)["written"] == 2
    finally:
        os.close(held)
    assert (default_root() / SLOTS_DIR).is_dir()


def test_rules_match_whole_words() -> None:
    from apex.agents.plotter import _KEYWORDS, _mentions

    def matched(text: str) -> set[str]:
        return {subdir for words, _, _, subdir in _KEYWORDS if _mentions(text, words)}

    assert matched("rmse of my history-matching run") == {"rmse"}
    assert matched("the maps and histograms") == {"models", "nn_hist"}
    assert matched("forecasts per well") == {"simData"}
    assert matched("show the rmse as well") == {"rmse"}
    assert matched("the model-change box plots") == {"model_change"}
    assert matched("the mec residuals") == {"residual"}


def test_a_removed_output_folder_is_not_recreated(tmp_path: Path, linear_case: CaseFactory) -> None:
    """Fourth review: a run deleted mid-drawing got its plot folders back from ``_save``."""
    import shutil

    from apex.core.figures import FigureRequest
    from apex.services.figure_view import run_view
    from apex.services.figures import draw

    run_dir = _run(linear_case)
    view = run_view(run_dir)
    output = tmp_path / "plots"

    class Removing:
        """The view, whose first metric read removes the output, as a deletion would."""

        def __getattr__(self, name: str) -> object:
            if name == "metric":
                shutil.rmtree(output, ignore_errors=True)
            return getattr(view, name)

    result = draw(run_dir, [FigureRequest("objective_distribution", subdir="obj"),
                            FigureRequest("metric_distribution", {"metric": "rmse"}, "rmse")],
                  output_dir=output, dpi_scale=0.05, view=Removing())  # type: ignore[arg-type]  # fmt: skip
    assert not (output / "obj").exists()
    assert not (output / "rmse").exists()
    assert result.drawn[0].skipped is not None  # its folder could not be made
    assert result.drawn[-1].skipped == "the output folder was removed"


def test_a_failed_start_releases_the_runs_lock(
    tmp_path: Path, linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fifth review (2026-10-12): a slot that could not be taken leaked the run's lock, and
    the run stayed 'drawing' for the service's lifetime."""
    import os

    from apex.core.figures import FigureRequest
    from apex.services import figures as service_figures

    run_dir = _run(linear_case)

    def unwritable(_max: int) -> int | None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(service_figures, "_take_slot", unwritable)
    with pytest.raises(PermissionError):
        service_figures.start_drawing(run_dir, [FigureRequest("objective_distribution")],
                                      dpi_scale=0.05)  # fmt: skip
    fd = service_figures._try_lock(run_dir / "report" / "plots")
    assert fd is not None  # free again
    os.close(fd)
