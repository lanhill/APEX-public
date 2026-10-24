"""Figures of predictions: forecasts, end-of-forecast distributions, scores.

One figure per file, written as ``<name>.png`` and ``<name>.pdf``; every function takes
``font_size`` and returns the PNG paths.

- **Per ensemble and vector** (``forecasts/<stem>_<ensemble>``): PET-fork's
  percentile-band style of the history-matching report (grey 0-100 and 10-90 bands with
  black lines, orange mean, red observations, green true forecast), extended over the
  prediction period, with a dashed line at the end of history.
- **Prior versus posterior** (``forecasts/<stem>_comparison``): the ensembles' 10-90 bands
  and medians in the validated palette (posterior blue, prior orange; further ensembles
  categorical slots 3 and 4), the truth as an ink line.
- **End-of-forecast distributions** of cumulative vectors (``...T``, e.g. FOPT):
  box plots per ensemble with the truth marked (``distributions/<stem>``).
- **Scores** (``scores/<score>``): each score per vector and ensemble.

Percentile labels follow the prediction's convention: ``p10`` is the 10th percentile, or,
with ``exceedance``, SPE-PRMS labels (``P90`` is the 10th percentile).
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.domain.prediction import EnsembleForecast
from apex.reporting.plots import FONT_SIZE, INK, POSTERIOR, PRIOR, _positions, _pyplot
from apex.reporting.result_plots import (
    BAND_INNER,
    BAND_LINE,
    BAND_OUTER,
    MEAN_COLOUR,
    OBSERVED_COLOUR,
    SERIES,
    TRUE_COLOUR,
    _save,
    _slug,
)

#: Band labels per percentile convention (the band is always the 10th-90th percentiles).
BAND_LABELS = {
    "percentile": "p10-p90",
    "exceedance": "P90-P10 (P90 = 10th pctl)",
}


def ensemble_colour(name: str, index: int) -> str:
    """Colour of an ensemble in comparisons: posterior blue, prior orange, then slots 3-4."""
    if name == "posterior":
        return POSTERIOR
    if name == "prior":
        return PRIOR
    return SERIES[2 + index % 2]


def _stem(vector: str) -> str:
    quantity, _, well = vector.partition(":")
    return _slug(f"{well}_{quantity}" if well else quantity)


def _observed(
    vector: str, observations: Sequence[tuple[str, str, float, float]]
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float64]]:
    rows = [(t, v, s) for t, d, v, s in observations if d == vector]
    times = [t for t, _, _ in rows]
    return times, np.array([v for _, v, _ in rows]), np.array([s for _, _, s in rows])


def forecast_figures(
    forecasts: Mapping[str, EnsembleForecast],
    output_dir: Path,
    *,
    truth: NDArray[np.float64] | None = None,
    first_prediction_time: int = 0,
    observations: Sequence[tuple[str, str, float, float]] = (),
    convention: str = "percentile",
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Plot every vector's forecast: one figure per ensemble, plus the comparison.

    ``observations`` are ``(time, vector, value, variance)`` of the run's scalar data;
    ``truth`` is ``(n_vectors, n_times)``.
    """
    plt = _pyplot(font_size)
    some = next(iter(forecasts.values()))
    x = _positions(list(some.times))
    history_end = x[first_prediction_time - 1] if 0 < first_prediction_time <= len(x) else None
    band = BAND_LABELS.get(convention, BAND_LABELS["percentile"])
    figures = []
    for v, vector in enumerate(some.vectors):
        quantity, _, well = vector.partition(":")
        where = f", at well {well}" if well else ""
        obs_times, obs_values, obs_variances = _observed(vector, observations)
        obs_x = _positions(obs_times) if obs_times else []
        drawn = []
        for name, forecast in forecasts.items():
            values = forecast.values[v]
            fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
            ax.fill_between(
                x, values.min(axis=1), values.max(axis=1), facecolor=BAND_OUTER, label="0-100 pctl"
            )
            ax.fill_between(
                x,
                np.percentile(values, 10, axis=1),
                np.percentile(values, 90, axis=1),
                facecolor=BAND_INNER,
                label=band,
            )
            for q, style in ((0, ":"), (10, "-"), (90, "-"), (100, ":")):
                ax.plot(
                    x,
                    np.percentile(values, q, axis=1),
                    color=BAND_LINE,
                    linestyle=style,
                    linewidth=0.9,
                )
            ax.plot(x, values.mean(axis=1), color=MEAN_COLOUR, linewidth=1.8, label="mean")
            if forecast.mean_model is not None:
                ax.plot(
                    x,
                    forecast.mean_model[v],
                    color=MEAN_COLOUR,
                    linestyle="--",
                    linewidth=1.2,
                    label="mean model",
                )
            if truth is not None:
                ax.plot(x, truth[v], color=TRUE_COLOUR, linewidth=1.6, label="true forecast")
            if obs_times:
                ax.errorbar(
                    obs_x,
                    obs_values,
                    yerr=2 * np.sqrt(obs_variances),
                    fmt=".",
                    markersize=7,
                    color=OBSERVED_COLOUR,
                    ecolor=OBSERVED_COLOUR,
                    elinewidth=0.8,
                    capsize=2,
                    label="observed ±2 sd",
                    zorder=4,
                )
            if history_end is not None:
                ax.axvline(
                    history_end, color=INK, linestyle="--", linewidth=0.9, label="end of history"
                )
            ax.set_title(f"{quantity} forecast, {name} ensemble{where}", loc="left")
            ax.set_xlabel("report time")
            ax.set_ylabel(quantity)
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(False)  # as the history-matching report: grid lines would cut the bands
            ax.legend(loc="upper left", ncols=2)
            drawn.append((fig, ax, f"{_stem(vector)}_{_slug(name)}"))
        if len(forecasts) > 1:
            fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
            for index, (name, forecast) in enumerate(forecasts.items()):
                values = forecast.values[v]
                colour = ensemble_colour(name, index)
                ax.fill_between(
                    x,
                    np.percentile(values, 10, axis=1),
                    np.percentile(values, 90, axis=1),
                    facecolor=colour,
                    alpha=0.25,
                    edgecolor="none",
                    label=f"{name} {band}",
                )
                ax.plot(
                    x,
                    np.percentile(values, 50, axis=1),
                    color=colour,
                    linewidth=2.0,
                    label=f"{name} median",
                )
            if truth is not None:
                ax.plot(x, truth[v], color=INK, linewidth=1.6, label="true forecast")
            if obs_times:
                ax.plot(
                    obs_x,
                    obs_values,
                    "o",
                    color=INK,
                    markersize=4,
                    fillstyle="none",
                    label="observed",
                )
            if history_end is not None:
                ax.axvline(
                    history_end, color=INK, linestyle="--", linewidth=0.9, label="end of history"
                )
            ax.set_title(f"{quantity} forecast{where}", loc="left")
            ax.set_xlabel("report time")
            ax.set_ylabel(quantity)
            ax.tick_params(axis="x", labelrotation=30)
            ax.legend(loc="upper left", ncols=2)
            drawn.append((fig, ax, f"{_stem(vector)}_comparison"))
        # The ensembles of a vector on one y range, so their spreads compare by eye.
        lows, highs = zip(*(ax.get_ylim() for _, ax, _ in drawn), strict=True)
        for fig, ax, stem in drawn:
            ax.set_ylim(min(lows), max(highs))
            figures.append(_save(fig, output_dir / "forecasts" / stem))
            plt.close(fig)
    return figures


def is_cumulative(vector: str) -> bool:
    """Whether a summary vector is a cumulative total (``FOPT``, ``WWPT:P1``, ...)."""
    keyword = vector.partition(":")[0]
    return len(keyword) >= 4 and keyword.endswith("T")


def distribution_figures(
    forecasts: Mapping[str, EnsembleForecast],
    output_dir: Path,
    *,
    truth: NDArray[np.float64] | None = None,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Box plots of each cumulative vector at the last forecast time, per ensemble."""
    plt = _pyplot(font_size)
    some = next(iter(forecasts.values()))
    figures = []
    for v, vector in enumerate(some.vectors):
        if not is_cumulative(vector):
            continue
        names = list(forecasts)
        data = [forecasts[name].values[v, -1, :] for name in names]
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3.0, 4.5), layout="constrained")
        boxes = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True)
        for index, (patch, name) in enumerate(zip(boxes["boxes"], names, strict=True)):
            patch.set_facecolor(ensemble_colour(name, index))
            patch.set_alpha(0.6)
        for median in boxes["medians"]:
            median.set_color(INK)
        ax.set_xticks(range(1, len(names) + 1), names)
        if truth is not None:
            ax.axhline(
                truth[v, -1], color=INK, linestyle="--", linewidth=1.2, label="true forecast"
            )
            ax.legend(loc="upper right")
        quantity, _, well = vector.partition(":")
        where = f", at well {well}" if well else ""
        ax.set_title(f"{quantity} on {some.times[-1]}{where}", loc="left")
        ax.set_ylabel(quantity)
        figures.append(_save(fig, output_dir / "distributions" / _stem(vector)))
        plt.close(fig)
    return figures


def score_figures(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    output_dir: Path,
    *,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """One dot plot per score: its value for each vector (rows) and ensemble (colours)."""
    if not scores:
        return []
    plt = _pyplot(font_size)
    names = list(scores)
    vectors = list(next(iter(scores.values())))
    score_names = list(next(iter(next(iter(scores.values())).values())))
    figures = []
    for score in score_names:
        fig, ax = plt.subplots(figsize=(7.0, 0.35 * len(vectors) + 1.8), layout="constrained")
        rows = np.arange(len(vectors))
        for index, name in enumerate(names):
            values = [scores[name][vector][score] for vector in vectors]
            ax.plot(values, rows, "o", markersize=8, color=ensemble_colour(name, index), label=name)
        ax.set_yticks(rows, vectors)
        ax.invert_yaxis()
        ax.set_xlabel(score)
        ax.set_title(f"{score} over the prediction period", loc="left")
        ax.legend(loc="lower right")
        figures.append(_save(fig, output_dir / "scores" / _slug(score)))
        plt.close(fig)
    return figures
