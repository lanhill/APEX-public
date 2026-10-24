"""Figure plugins and presets: files, sizes, titles and values.

The plugins reproduce the classic figure set of an ensemble history-matching study (the
Brugge results), less five defects of the reference (``reporting/figures/classic.py``).
They are checked on a small in-memory run, :class:`FakeView`: a 6 x 5 x 2 grid with
inactive cells, two state variables (one an ln-permeability), data of two wells at three
report dates, auxiliary forecasts and a network's parameters.
"""

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("matplotlib")

from matplotlib.colors import same_color
from PIL import Image

import apex.reporting.figures  # noqa: F401  (registers the plugins)
from apex.agents.plotter import FigureChoice, PlotPlan, plan_problems, requests_of, rules
from apex.core.figures import (
    FieldLayout,
    FieldLayoutError,
    FigureRequest,
    GridView,
    IterateView,
    ObservationsView,
)
from apex.core.plugins import figure_presets, figures
from apex.domain.observations import DataKey
from apex.reporting.figures import classic
from apex.services.figures import check_request, draw

DIMS = (6, 5, 2)
N_MEMBERS = 8
TIMES = ("2008-01-02", "2008-01-11", "2008-03-01")
TYPES = ("WOPR:P1", "WBHP:I1")


class FakeView:
    """A :class:`~apex.core.figures.RunView` over arrays made here."""

    def __init__(
        self, *, auxiliary: bool = True, nn: bool = True, truth: bool = True,
        normalized: bool = True,
    ) -> None:  # fmt: skip
        rng = np.random.default_rng(7)
        size = int(np.prod(DIMS))
        self.actnum = np.ones(size, dtype=bool)
        self.actnum[:3] = False
        self.keys = tuple(DataKey(t, d) for d in TYPES for t in TIMES)
        self.values = rng.uniform(100, 200, len(self.keys))
        self._accepted = [0, 1, 3, 4]  # 2 rejected
        self._state = {
            k: {"permx": rng.normal(4.0, 0.5 / (1 + k), (size, N_MEMBERS)),
                "poro": rng.uniform(0.1, 0.3, (size, N_MEMBERS)),
                **({"nn_params": rng.normal(0, 1, (40, N_MEMBERS))} if nn else {})}
            for k in self._accepted
        }  # fmt: skip
        self._predicted = {k: rng.uniform(80, 220, (len(self.keys), N_MEMBERS))
                           for k in self._accepted}  # fmt: skip
        self._aux = auxiliary
        self._truth = {"permx": rng.normal(4.0, 0.5, size), "poro": rng.uniform(0.1, 0.3, size)}
        self._has_truth = truth
        self._metrics: dict[str, dict[str, list[NDArray[np.float64]]]] = {
            "data_mismatch": {lab: [rng.uniform(100, 900, N_MEMBERS) for _ in range(5)]
                              for lab in ("all", *TYPES)},
            "normalized_data_mismatch": {lab: [rng.uniform(1, 9, N_MEMBERS) for _ in range(5)]
                                         for lab in ("all", *TYPES)},
            "rmse": {lab: [rng.uniform(0, 1, N_MEMBERS) for _ in range(5)]
                     for lab in ("permx", "all", "poro")},
            "model_change": {lab: [rng.uniform(0, 1, N_MEMBERS) for _ in range(4)]
                             for lab in ("permx", "all")},
        }  # fmt: skip
        if not normalized:  # e.g. a run whose [[metrics]] leave it out
            del self._metrics["normalized_data_mismatch"]

    path = Path("/nowhere")
    final_iteration = 4
    posterior_iteration = 4
    start: dt.date | None = dt.date(2008, 1, 1)

    @property
    def iterations(self) -> list[int]:
        return [0, 1, 2, 3, 4]

    @property
    def accepted_iterations(self) -> list[int]:
        return list(self._accepted)

    def iterate(self, iteration: int) -> IterateView:
        if iteration not in self._accepted:
            raise KeyError(iteration)
        aux_keys = tuple(DataKey(k.time, f"MEC:{k.datatype}") for k in self.keys)
        return IterateView(
            iteration, self._state[iteration], self._predicted[iteration],
            np.arange(N_MEMBERS), self._predicted[iteration] - 150 if self._aux else None,
            aux_keys if self._aux else (),
        )  # fmt: skip

    def metric_labels(self) -> dict[str, list[str]]:
        return {name: list(by) for name, by in self._metrics.items()}

    def metric(self, name: str, label: str = "all", *, accepted_only: bool = False
               ) -> tuple[list[int], list[NDArray[np.float64]], list[bool]]:  # fmt: skip
        values = self._metrics[name][label]
        # model_change has no value at the prior (iteration 0).
        iterations = self.iterations[len(self.iterations) - len(values) :]
        rows = list(zip(iterations, values, strict=True))
        if accepted_only:
            rows = [(k, v) for k, v in rows if k in self._accepted]
        return [k for k, _ in rows], [v for _, v in rows], [k in self._accepted for k, _ in rows]

    def truth(self) -> Mapping[str, NDArray[np.float64]] | None:
        return self._truth if self._has_truth else None

    def observations(self) -> ObservationsView:
        return ObservationsView(self.keys, self.values, np.ones(len(self.keys)))

    def variables(self) -> list[str]:
        return list(self._state[0])

    def grid(self, variable: str) -> GridView | None:
        return GridView(DIMS, "F", self.actnum) if variable in ("permx", "poro") else None

    def field_layout(self, datatype: str) -> FieldLayout | None:
        return None  # no geophysical field data


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Record each saved figure's title, labels, ticks, limits and lines, by file stem."""
    seen: dict[str, dict[str, Any]] = {}
    original = classic._save

    def save(plt: Any, fig: Any, stem: Path, dpi: float, *, crop: bool = False) -> Path:  # noqa: ANN401
        ax = fig.axes[0]
        image = next(iter(ax.collections), None)
        seen[stem.name] = {
            "title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
            "xticks": list(ax.get_xticks()), "ylim": ax.get_ylim(),
            "legend": ax.get_legend() is not None,
            "lines": [(line.get_linestyle(), line.get_color(), np.asarray(line.get_ydata()))
                      for line in ax.get_lines()],
            "clim": None if image is None or not hasattr(image, "get_clim") else image.get_clim(),
            "size": tuple(fig.get_size_inches()), "dpi": dpi, "crop": crop,
        }  # fmt: skip
        return original(plt, fig, stem, dpi, crop=crop)

    monkeypatch.setattr(classic, "_save", save)
    return seen


def _pixels(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


# ------------------------------------------------------------------------ distributions


def test_metric_plots_have_the_references_names_sizes_and_labels(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    view = FakeView()
    plot = figures.create("metric_distribution", metric="rmse")

    # An earlier version's misspelt files of these figures (release tester U4), and one of
    # another figure, which a redraw leaves alone.
    for name in (
        "rmse_permx_violionplot.png",
        "rmse_permx_violionplot.pdf",
        "other_violionplot.png",
    ):
        (tmp_path / name).write_bytes(b"old")

    written = plot.draw(view, tmp_path, dpi_scale=0.1)

    assert sorted(p.name for p in tmp_path.glob("*violion*")) == ["other_violionplot.png"]
    # The labels in the run's order, 'all' last; violin first, as the reference.
    assert [p.name for p in written] == [
        "rmse_permx_violinplot.png", "rmse_permx_boxplot.png",
        "rmse_poro_violinplot.png", "rmse_poro_boxplot.png",
        "rmse_all_violinplot.png", "rmse_all_boxplot.png",
    ]  # fmt: skip
    assert all(p.with_suffix(".pdf").is_file() for p in written)
    assert _pixels(written[0]) == (1000, 800)  # 20 x 16 in at 500 dpi x 0.1
    info = captured["rmse_permx_boxplot"]
    assert info["ylabel"] == "rmse (permx)"
    assert info["xlabel"] == "Iteration no."
    assert info["title"] == ""
    # The accepted iterates (0, 1, 3, 4) numbered 0..3, a tick each and none extra.
    assert info["xticklabels"] == ["0", "1", "2", "3"]
    assert info["xticks"] == [1, 2, 3, 4]


def test_model_change_is_labelled_by_its_own_iterate(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    """The reference labelled the change made by iteration 1 as 0."""
    figures.create("metric_distribution", metric="model_change", kinds=["box"]).draw(
        FakeView(), tmp_path, dpi_scale=0.05
    )
    assert captured["model_change_all_boxplot"]["xticklabels"] == ["1", "2", "3"]


def test_the_objective_and_the_data_mismatch_names(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    view = FakeView()
    objective = figures.create("objective_distribution").draw(view, tmp_path, dpi_scale=0.05)
    mismatch = figures.create(
        "metric_distribution", metric="normalized_data_mismatch", prefix="data_mismatch",
        kinds=["box"],
    ).draw(view, tmp_path, dpi_scale=0.05)  # fmt: skip

    assert [p.name for p in objective] == ["obj_func_boxplot.png", "obj_func_violinplot.png"]
    assert captured["obj_func_boxplot"]["ylabel"] == "Data mismatch"
    assert [p.name for p in mismatch] == [
        "data_mismatch_WOPR P1_boxplot.png", "data_mismatch_WBHP I1_boxplot.png",
        "data_mismatch_all_boxplot.png",
    ]  # fmt: skip
    assert captured["data_mismatch_WOPR P1_boxplot"]["ylabel"] == "data_mismatch (WOPR P1)"
    figures.create("objective_distribution", metric="normalized_data_mismatch").draw(
        view, tmp_path / "n", dpi_scale=0.05
    )
    assert captured["obj_func_boxplot"]["ylabel"] == "Normalized data mismatch"


def test_one_figure_at_the_references_resolution(tmp_path: Path) -> None:
    [png, *_] = figures.create("objective_distribution").draw(FakeView(), tmp_path)
    assert _pixels(png) == (10000, 8000)


# --------------------------------------------------------------------------------- bands


def test_forecast_bands_per_well_initial_and_final(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    view = FakeView()

    written = figures.create("data_bands").draw(view, tmp_path)

    assert sorted(p.name for p in written) == [
        "I1_WBHP.png",
        "I1_WBHP_initial.png",
        "P1_WOPR.png",
        "P1_WOPR_initial.png",
    ]
    assert _pixels(tmp_path / "P1_WOPR_initial.png") == (640, 480)
    initial, final = captured["P1_WOPR_initial"], captured["P1_WOPR"]
    assert initial["title"] == "WOPR initial forcast, at Well: P1"  # sic, the reference's
    assert final["title"] == "WOPR final forcast, at Well: P1"
    assert (initial["xlabel"], initial["ylabel"]) == ("Days", "Sm3/Day")
    assert captured["I1_WBHP"]["ylabel"] == "Bar"
    assert final["xlabel"] == "Days"  # kept (the reference's final PNG lost it)
    assert initial["legend"]
    assert not final["legend"]
    assert final["ylim"] == initial["ylim"]
    # The lines: observations, then p90, p100, p10, p0 and the mean of the members.
    rows = [k.datatype == "WOPR:P1" for k in view.keys]
    members = view.iterate(0).predicted[rows]
    p90, p100, p10, p0, obs, mean = initial["lines"]  # the reference's order
    np.testing.assert_allclose(obs[2], view.values[rows])
    np.testing.assert_allclose(p90[2], np.percentile(members, 90, axis=1))
    np.testing.assert_allclose(p0[2], members.min(axis=1))
    np.testing.assert_allclose(mean[2], members.mean(axis=1))
    np.testing.assert_allclose(p10[2], np.percentile(members, 10, axis=1))
    assert p100[0] == ":"
    assert same_color(p100[1], "k")
    assert same_color(mean[1], "orange")


def test_residual_bands_need_auxiliary_forecasts(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    residuals = figures.create("data_bands", source="auxiliary")
    assert residuals.applies(FakeView(auxiliary=False)) == "the run saved no auxiliary forecasts"

    residuals.draw(FakeView(), tmp_path)

    info = captured["P1_WOPR_initial"]
    assert info["title"] == "WOPR initial residual, at Well: P1"
    assert len(info["lines"]) == 5  # no observations
    with pytest.raises(ValueError, match="no observations"):
        figures.create("data_bands", source="auxiliary", observations=True)


# ---------------------------------------------------------------------------- layer maps


def test_layer_maps_share_the_references_colour_limits(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    view = FakeView()

    written = figures.create("field_layers", variables=["permx"]).draw(view, tmp_path)

    assert len(written) == 6 * 2
    assert {p.name for p in written} >= {"permx_ref_layer1.png", "permx_std_post_layer2.png"}
    true_layer = np.exp(view.truth()["permx"]).copy()  # type: ignore[index]
    true_layer[~view.actnum] = np.nan
    layer1 = true_layer.reshape(DIMS, order="F")[:, :, 0]
    limits = (np.nanmin(layer1) * 0.95, np.nanmax(layer1) * 1.05)
    for kind in ("ref", "prior", "post"):
        np.testing.assert_allclose(captured[f"permx_{kind}_layer1"]["clim"], limits)
    assert captured["permx_diff_layer1"]["clim"] != pytest.approx(limits)
    assert captured["permx_ref_layer1"]["title"] == "Reference permx at layer 1"
    assert captured["permx_diff_layer2"]["title"] == "Posterior - Prior, permx at layer 2"
    assert captured["permx_std_prior_layer1"]["title"] == "Prior std permx at layer 1"
    assert captured["permx_post_layer1"]["crop"]


def test_without_a_truth_the_posterior_sets_the_limits(tmp_path: Path) -> None:
    written = figures.create("field_layers", variables=["poro"], layers=[2]).draw(
        FakeView(truth=False), tmp_path
    )
    assert sorted(p.name for p in written) == [
        "poro_diff_layer2.png", "poro_post_layer2.png", "poro_prior_layer2.png",
        "poro_std_post_layer2.png", "poro_std_prior_layer2.png",
    ]  # fmt: skip


# ---------------------------------------------------------------------------- histograms


def test_the_histogram_of_a_network_s_parameters(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    written = figures.create("state_histogram", variable="nn_params").draw(FakeView(), tmp_path)

    assert [p.name for p in written] == ["hist_initial.png", "hist_final.png"]
    assert captured["hist_final"]["title"] == "final nn_params"
    assert captured["hist_final"]["ylabel"] == "Frequency"
    assert (
        figures.create("state_histogram", variable="nn_params").applies(FakeView(nn=False))
        == "the run has no state variable 'nn_params'"
    )


# ------------------------------------------------------------------------- the preset


def test_the_brugge_classic_preset_writes_the_references_folders(tmp_path: Path) -> None:
    view = FakeView()

    result = draw(view.path, preset="brugge_classic", output_dir=tmp_path, dpi_scale=0.05,
                  view=view)  # fmt: skip

    folders = {d.subdir for d in result.drawn if d.files}
    # By default (the user, 2026-10-12): data_mismatch, not normalized_data_mismatch;
    # model_change recorded but not drawn.
    assert folders == {
        "obj",
        "data_mismatch",
        "rmse",
        "simData",
        "residual",
        "models",
        "nn_hist",
    }  # fmt: skip
    assert result.drawn[0].options == {}
    assert [(d.subdir, d.options["metric"]) for d in result.drawn[1:2]] == [
        ("data_mismatch", "data_mismatch")]  # fmt: skip
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["written"] == len(result.files)
    # No ssim recorded: mssim/ is reported, with the reason, not left out silently.
    assert summary["skipped"] == [{"figure": "metric_distribution", "subdir": "mssim",
                                   "reason": "the run recorded no 'ssim'"}]  # fmt: skip
    assert (tmp_path / "rmse" / "rmse_all_violinplot.png").is_file()
    # The reference's misspelt 'violionplot' file names are not reproduced (release tester U4).
    assert not list(tmp_path.rglob("*violion*"))
    assert (tmp_path / "simData" / "P1_WOPR.png").is_file()


def test_the_preset_reports_what_the_run_cannot_give(tmp_path: Path) -> None:
    """The coder's first use (2026-10-12): a run without normalized_data_mismatch lost
    obj/ and data_mismatch/ silently, the preset dropping them before the drawing."""
    view = FakeView(auxiliary=False, nn=False, normalized=False)
    preset = figure_presets.create("brugge_classic", normalized=True, model_change=True)

    result = draw(view.path, preset.requests(view), output_dir=tmp_path, dpi_scale=0.05,
                  view=view)  # fmt: skip

    skipped = {d.subdir: d.skipped for d in result.drawn if d.skipped}
    assert skipped == {
        "obj_normalized": "the run recorded no 'normalized_data_mismatch'",
        "normalized_data_mismatch": "the run recorded no 'normalized_data_mismatch'",
        "mssim": "the run recorded no 'ssim'",
        "residual": skipped["residual"],
        "nn_hist": skipped["nn_hist"],
    }  # fmt: skip
    assert {d.subdir for d in result.drawn if d.files} == {
        "obj", "data_mismatch", "rmse", "model_change", "simData", "models"}  # fmt: skip


def test_requests_that_cannot_be_drawn_are_skipped_with_the_reason(tmp_path: Path) -> None:
    view = FakeView()
    assert "unknown figure" in str(check_request(FigureRequest("pie_chart"), view))
    assert "unexpected keyword" in str(check_request(FigureRequest("data_bands", {"x": 1}), view))
    assert "recorded no 'ssim'" in str(
        check_request(FigureRequest("metric_distribution", {"metric": "ssim"}), view)
    )
    assert "relative folder" in str(check_request(FigureRequest("data_bands", {}, "../up"), view))

    result = draw(view.path, [FigureRequest("pie_chart"), FigureRequest("objective_distribution")],
                  output_dir=tmp_path, dpi_scale=0.05, view=view)  # fmt: skip

    assert [d.skipped is None for d in result.drawn] == [False, True]


# ------------------------------------------------- the plotter agent's rules


def test_the_rules_draw_what_the_request_names() -> None:
    view = FakeView()

    plan = rules("the RMSE box plots and the layer maps, please", view)

    assert plan.preset is None
    assert [(c.figure, c.options.get("metric"), c.subdir) for c in plan.figures] == [
        ("metric_distribution", "rmse", "rmse"), ("field_layers", None, "models"),
    ]  # fmt: skip


def test_the_rules_draw_data_mismatch_and_the_normalized_one_only_when_named() -> None:
    view = FakeView()

    def drawn(request: str) -> list[tuple[str | None, str]]:
        return [(c.options.get("metric"), c.subdir) for c in rules(request, view).figures]

    assert drawn("the data mismatch per well") == [("data_mismatch", "data_mismatch"),
                                                   (None, "simData")]  # fmt: skip
    assert drawn("the normalized data mismatch") == [
        ("normalized_data_mismatch", "normalized_data_mismatch")]  # fmt: skip
    assert drawn("the model change") == [("model_change", "model_change")]


def test_without_a_request_or_with_nothing_drawable_the_rules_draw_the_classic_set() -> None:
    view = FakeView()
    assert rules(None, view).preset == "brugge_classic"
    # The view records no ssim: the matched figure does not apply, so the preset is drawn.
    assert rules("mean ssim", view).preset == "brugge_classic"


def test_a_plan_is_held_to_the_run() -> None:
    view = FakeView()
    plan = PlotPlan(figures=[
        FigureChoice(figure="pie_chart"),
        FigureChoice(figure="metric_distribution", options={"metric": "ssim"}),
        FigureChoice(figure="data_bands", options={"colour": "red"}),
    ])  # fmt: skip

    problems = plan_problems(plan, view, max_figures=40)

    assert len(problems) == 3
    assert "unknown figure 'pie_chart'" in problems[0]
    assert "recorded no 'ssim'" in problems[1]
    assert "colour" in problems[2]
    too_many = plan_problems(PlotPlan(preset="brugge_classic"), view, max_figures=2)
    assert too_many == ["8 figure requests: at most 2"]
    # A preset's figure the run cannot give (no ssim here) is no problem: it is skipped.
    assert plan_problems(PlotPlan(preset="brugge_classic"), view, max_figures=40) == []
    asked = PlotPlan(preset="brugge_classic", preset_options={"model_change": True})
    assert plan_problems(asked, view, max_figures=40) == []
    assert "model_change" in {r.subdir for r in requests_of(asked, view)}
    wrong = PlotPlan(preset="brugge_classic", preset_options={"colour": "red"})
    assert "colour" in " ".join(plan_problems(wrong, view, max_figures=40))
    with pytest.raises(ValueError, match="a preset or at least one figure"):
        PlotPlan()


def test_the_rules_switch_the_presets_options_on_when_named() -> None:
    """Second review (2026-10-12): without a model, the preset's options were unreachable."""
    view = FakeView()

    classic = rules("the classic set with the model change", view)
    both = rules("all figures, normalized too, and the model change", view)

    assert (classic.preset, classic.preset_options) == ("brugge_classic", {"model_change": True})
    assert both.preset_options == {"normalized": True, "model_change": True}
    assert rules("the classic set", view).preset_options == {}

    def drawn(request: str) -> list[str]:
        return [c.subdir for c in rules(request, view).figures]

    assert drawn("the normalized objective") == ["obj_normalized"]
    assert drawn("the data mismatch and the normalized one") == [
        "data_mismatch", "normalized_data_mismatch"]  # fmt: skip
    assert drawn("the normalized objective and the data mismatch") == [
        "obj_normalized", "data_mismatch"]  # fmt: skip
    for wrong in ({"model_change": "false"}, {"histogram_variable": True}):
        with pytest.raises(ValueError, match="must be"):
            figure_presets.create("brugge_classic", **wrong)
    with pytest.raises(ValueError, match="preset_options need a preset"):
        PlotPlan(figures=[FigureChoice(figure="data_bands")], preset_options={"normalized": True})


def test_the_models_facts_list_the_presets_options() -> None:
    from apex.agents.plotter import plot_facts

    facts = plot_facts(FakeView())
    assert {"normalized", "model_change"} <= set(facts["presets"]["brugge_classic"]["options"])
    assert facts["figures"]["metric_distribution"]["applies"] is None


class FieldView(FakeView):
    """FakeView with an AVO map (6 x 5 columns, 3 untraced) and a gravity survey (4
    stations), each at two vintages, the forecasts appended to the wells' data."""

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(11)
        columns = np.array([(i, j) for j in range(5) for i in range(6)][3:], dtype=float)
        stations = np.array([[0.0, 0.0], [500.0, 0.0], [0.0, 500.0], [500.0, 500.0]])
        self.layouts = {"rms_10": FieldLayout("columns", columns, (6, 5)),
                        "grav": FieldLayout("stations", stations)}  # fmt: skip
        extra = tuple(
            DataKey(t, name, i)
            for name, layout in self.layouts.items()
            for t in ("2010-01-01", "2012-01-01")
            for i in range(layout.positions.shape[0])
        )
        self.keys = self.keys + extra
        self.values = np.concatenate([self.values, rng.normal(1.0, 0.2, len(extra))])
        self._predicted = {
            k: np.vstack([p, rng.normal(1.0, 0.3, (len(extra), N_MEMBERS))])
            for k, p in self._predicted.items()
        }

    def field_layout(self, datatype: str) -> FieldLayout | None:
        return self.layouts.get(datatype)


def test_field_maps_draw_each_datum_and_vintage(
    tmp_path: Path, captured: dict[str, dict[str, Any]]
) -> None:
    """Observed, prior, posterior and mismatch maps per field datum and
    vintage; AVO maps on the columns (untraced ones blank), stations as points."""
    figure = figures.create("field_maps")
    assert figure.applies(FakeView()) == "the run has no field data"
    view = FieldView()
    assert figure.applies(view) is None
    written = figure.draw(view, tmp_path)
    assert len(written) == 2 * 2 * 4  # two data, two vintages, four maps
    names = {p.stem for p in written}
    assert "rms_10_2010-01-01_obs" in names
    assert "grav_2012-01-01_mismatch" in names
    assert all(p.with_suffix(".pdf").exists() for p in written)
    obs = captured["rms_10_2010-01-01_obs"]
    assert obs["title"] == "rms_10 observed at 2010-01-01"
    mismatch = captured["rms_10_2010-01-01_mismatch"]
    low, high = mismatch["clim"]
    assert low == pytest.approx(-high)  # symmetric about zero
    only = figures.create("field_maps", datatypes=["grav"], kinds=["obs"])
    assert [p.stem for p in only.draw(view, tmp_path / "only")] == [
        "grav_2010-01-01_obs",
        "grav_2012-01-01_obs",
    ]


def test_the_plotters_rules_choose_field_maps_for_seismic_requests() -> None:
    plan = rules("the seismic and gravity data", FieldView())
    assert [c.figure for c in plan.figures] == ["field_maps"]
    # A run without field data: the request falls back to the preset, and says why.
    fallback = rules("the seismic and gravity data", FakeView())
    assert fallback.figures == []
    assert "field_maps: the run has no field data" in fallback.reason
    # "maps" also names the layer maps.
    assert [c.figure for c in rules("the seismic maps", FieldView()).figures] == [
        "field_layers",
        "field_maps",
    ]


def test_field_maps_say_why_they_cannot_be_drawn(tmp_path: Path) -> None:
    """The review of phase 4, step E: a model that cannot be rebuilt, and a datum asked for
    by name that has no map, are reported with their cause."""

    class Broken(FieldView):
        def field_layout(self, datatype: str) -> FieldLayout | None:
            raise FieldLayoutError("the run's geophysical model could not be rebuilt: moved")

    assert figures.create("field_maps").applies(Broken()) == (
        "the run's geophysical model could not be rebuilt: moved"
    )
    asked = figures.create("field_maps", datatypes=["rms_10", "wavelet(rms_25)"])
    assert asked.applies(FieldView()) == (
        "no map of wavelet(rms_25) (AVO maps and station data have one)"
    )


def test_a_thinned_survey_fills_its_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Columns every 2nd along I: the map is drawn on their lattice, so every traced column
    fills a cell (not one cell in two), the axes in the grid's column indices."""
    view = FieldView()
    columns = np.array([(i, j) for j in range(5) for i in range(0, 6, 2)], dtype=float)
    view.layouts["rms_10"] = FieldLayout("columns", columns, (6, 5), (2, 1))
    view.keys = tuple(
        k for k in view.keys if k.datatype != "rms_10" or (k.index or 0) < len(columns)
    )
    keep = [n for n, k in enumerate(FieldView().keys) if k in set(view.keys)]
    view.values = view.values[keep]
    view._predicted = {k: p[keep] for k, p in view._predicted.items()}
    figure = figures.create("field_maps", datatypes=["rms_10"], kinds=["obs"])
    import matplotlib.pyplot as plt

    drawn: list[NDArray[np.float64]] = []
    original = plt.pcolormesh

    def record(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        drawn.append(np.asarray(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(plt, "pcolormesh", record)
    figure.draw(view, tmp_path)
    assert drawn
    assert drawn[0].shape == (3, 5)  # the lattice of kept columns
    assert np.isfinite(drawn[0]).all()  # every lattice cell traced


def test_field_maps_skip_a_view_without_field_layouts() -> None:
    """The coder's review of phase 4: a RunView written before field_layout existed."""

    class Older(FieldView):
        def __getattribute__(self, name: str) -> Any:  # noqa: ANN401
            if name == "field_layout":
                raise AttributeError(name)
            return super().__getattribute__(name)

    assert figures.create("field_maps").applies(Older()) == (
        "no field datum has a map (geophysical AVO maps and station data have; "
        "whole traces and compressed fields do not)"
    )
