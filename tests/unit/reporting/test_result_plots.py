"""Result figures: one figure per file, PNG and PDF, for synthetic and real cases."""

from pathlib import Path

import numpy as np
import pytest

from apex.reporting.result_plots import (
    field_layer_figures,
    fields_comparison_figure,
    histogram_figures,
    metric_comparison_figure,
    metric_figure,
    production_figures,
)

PNG = b"\x89PNG\r\n\x1a\n"


def _written(png: Path) -> bool:
    """A PNG and its PDF twin exist."""
    pdf = png.with_suffix(".pdf")
    return (
        png.read_bytes()[:8] == PNG
        and png.stat().st_size > 5_000
        and pdf.read_bytes()[:5] == b"%PDF-"
    )


@pytest.mark.parametrize("kind", ["box", "violin"])
def test_metric_figure(tmp_path: Path, kind: str) -> None:
    rng = np.random.default_rng(0)
    values = [rng.lognormal(3 - k, 0.3, 20) for k in range(4)]
    path = metric_figure(
        values,
        [0, 1, 2, 3],
        tmp_path / "m",
        title="rmse",
        ylabel="rmse",
        accepted=[True, True, False, True],
        kind=kind,  # type: ignore[arg-type]
        log=True,
    )
    assert path == tmp_path / "m.png"
    assert _written(path)


def test_font_size_changes_the_figure(tmp_path: Path) -> None:
    values = [np.arange(5.0), np.arange(5.0) + 1]
    small = metric_figure(values, [0, 1], tmp_path / "small", title="t", ylabel="y", font_size=8)
    large = metric_figure(values, [0, 1], tmp_path / "large", title="t", ylabel="y", font_size=24)
    assert small.read_bytes() != large.read_bytes()


def test_metric_comparison_figure(tmp_path: Path) -> None:
    series = {f"exp{i}": ([0, 1, 2], [3.0 - i * 0.1, 2.0, 1.0 + i * 0.1]) for i in range(4)}
    assert _written(metric_comparison_figure(series, tmp_path / "c", title="t", ylabel="y"))
    with pytest.raises(ValueError, match="1 to 4"):
        metric_comparison_figure(
            {**series, "five": ([0], [1.0])}, tmp_path / "x", title="t", ylabel="y"
        )


def test_production_figures_one_per_type_and_stage(tmp_path: Path) -> None:
    times = [f"2024-0{m}-01" for m in range(1, 7)] * 2
    datatypes = ["WOPR:P1"] * 6 + ["FOPR"] * 6
    rng = np.random.default_rng(1)
    truth = rng.uniform(10, 20, 12)
    observed = truth + rng.standard_normal(12)
    prior = truth[:, None] + 5 * rng.standard_normal((12, 10))
    posterior = truth[:, None] + rng.standard_normal((12, 10))
    figures = production_figures(
        times, datatypes, observed, np.ones(12), prior, posterior, tmp_path, truth=truth
    )
    assert sorted(p.name for p in figures) == [
        "FOPR_final.png",
        "FOPR_initial.png",
        "P1_WOPR_final.png",
        "P1_WOPR_initial.png",
    ]
    assert all(_written(p) for p in figures)
    real = production_figures(
        times, datatypes, observed, np.ones(12), prior, posterior, tmp_path / "r"
    )
    assert len(real) == 4


@pytest.mark.parametrize("grid", [(10, 10, 2), (8, 6), (5,)])
def test_field_figures_one_per_quantity_and_layer(tmp_path: Path, grid: tuple[int, ...]) -> None:
    n = int(np.prod(grid))
    n_layers = grid[2] if len(grid) == 3 else 1
    rng = np.random.default_rng(2)
    truth = rng.standard_normal(n)
    prior = rng.standard_normal((n, 12))
    posterior = truth[:, None] + 0.3 * rng.standard_normal((n, 12))

    synthetic = field_layer_figures("permx", grid, prior, posterior, tmp_path / "s", truth=truth)
    real = field_layer_figures("permx", grid, prior, posterior, tmp_path / "r")

    assert len(synthetic) == 7 * n_layers  # true, 2 means, 2 stds, 2 differences
    assert len(real) == 5 * n_layers  # no truth: no true field, no error
    assert (tmp_path / "s" / f"permx_final_mean_layer{n_layers}.png") in synthetic
    assert all(_written(p) for p in synthetic)
    with pytest.raises(ValueError, match="does not fit"):
        field_layer_figures("permx", (3, 3), prior, posterior, tmp_path / "bad")


def test_histograms_prior_and_final_apart(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    prior, posterior = rng.standard_normal((50, 10)), rng.standard_normal((50, 10))
    figures = histogram_figures("poro", prior, posterior, tmp_path, truth=rng.standard_normal(50))
    assert [p.name for p in figures] == ["poro_prior.png", "poro_final.png"]
    assert all(_written(p) for p in figures)


def test_fields_comparison_figure_on_request(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    fields = {name: rng.standard_normal(200) for name in ("true", "prior mean", "ES-MDA")}
    assert _written(fields_comparison_figure("ln permx", (10, 10, 2), fields, tmp_path / "c"))


def test_health_figures(tmp_path: Path) -> None:
    from apex.reporting.result_plots import (
        damping_figure,
        data_mismatch_change_figure,
        members_figure,
        spread_figures,
    )

    few = spread_figures([0, 1, 2], {"permx": [1.0, 0.5, 0.03], "poro": [1.0, 0.8, 0.6]}, tmp_path)
    many = spread_figures([0, 1], {f"v{i}": [1.0, 0.5] for i in range(5)}, tmp_path / "many")
    change = data_mismatch_change_figure([2, 3], [0.2, 0.004], tmp_path / "change", tolerance=0.01)
    damping = damping_figure([1, 2, 3], [1.0, 2.0, 1.0], [True, False, True], tmp_path / "damping")
    members = members_figure([0, 1, 2], [50, 48, 48], 50, tmp_path / "members")

    assert [p.name for p in few] == ["spread.png"]  # up to four variables in one figure
    assert sorted(p.name for p in many) == [f"spread_v{i}.png" for i in range(5)]
    for png in [*few, *many, change, damping, members]:
        assert png.is_file()
        assert png.with_suffix(".pdf").is_file()
