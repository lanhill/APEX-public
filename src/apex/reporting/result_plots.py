"""Result figures of history-matching runs: metrics, production data, parameter fields.

APEX's counterpart of PET's ``Plotting`` package (github.com/Python-Ensemble-Toolbox/
Plotting @ ``af0e0cf``: ``plot_objective_function``, ``plot_data.plot_prod``,
``plot_parameters.plot_layer``) and of PET-fork's extensions (``1348861``,
``Plotting/``: ``metric_analyses.metric_plots``, ``plot_parameters.plot_layer`` and
``plot_histogram``, ``plot_data.plot_prod_separate``), as used for its Brugge study.

- **One figure per file.** Fields get one figure per quantity and layer (true field, prior
  and final mean, prior and final standard deviation, final minus prior, final minus
  truth), production data one figure per data type for the initial and one for the final
  ensemble, histograms one per ensemble. Only the experiment comparison
  (:func:`metric_comparison_figure`) and, on request, :func:`fields_comparison_figure`
  combine several things in one figure.
- **Every figure is written twice**, ``<name>.png`` and ``<name>.pdf``; functions return
  the PNG paths.
- **Font size**: every function takes ``font_size`` (default 12 pt).
- **Colours**: maps and colour bars use PET-fork's Brugge style (``jet``, vertical
  colour bar, y axis downwards, a shared colour range for the true field and the means);
  production data too (grey 0-100 and 10-90 percentile bands with black lines, orange
  mean, red observations, green true response). Box plots and the experiment comparison
  use the validated reference palette of :mod:`apex.reporting.plots` (comparison lines:
  categorical slots 1-4, direct-labelled).
"""

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from apex.reporting.plots import (
    BASELINE,
    FONT_SIZE,
    INK,
    MUTED,
    POSTERIOR,
    TEXT_SECONDARY,
    _positions,
    _pyplot,
)

#: Categorical slots 1-4 of the reference palette (light mode), in fixed order; validated
#: as an adjacent set (CVD dE >= 9.1, normal-vision dE >= 22.9).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Colour map of parameter and data maps (PET-fork's Brugge figures).
FIELD_CMAP = "jet"

#: PET-fork's production-data colours (``plot_prod_separate``).
BAND_OUTER, BAND_INNER, BAND_LINE = "lightgrey", "grey", "k"
MEAN_COLOUR, OBSERVED_COLOUR, TRUE_COLOUR = "orange", "r", "g"

#: Resolution of the PNG files (PDF files are vector).
DPI = 200


def _save(fig: Any, path: Path) -> Path:  # noqa: ANN401 (matplotlib Figure, optional dep)
    """Write ``fig`` as ``<path>.png`` and ``<path>.pdf``; return the PNG path."""
    path = Path(path)
    png = path.with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    return png


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


# ------------------------------------------------------------------------------ metrics


def metric_figure(
    values: Sequence[NDArray[np.float64]],
    iterations: Sequence[int],
    path: Path,
    *,
    title: str,
    ylabel: str,
    accepted: Sequence[bool] | None = None,
    kind: Literal["box", "violin"] = "box",
    log: bool = False,
    selected: int | None = None,
    font_size: float = FONT_SIZE,
) -> Path:
    """Plot the distribution over members of a per-member quantity at each iteration.

    PET-fork's ``metric_plots`` (box and violin plots). ``values[i]`` is
    ``(n_members,)`` at ``iterations[i]``; rejected trials (``accepted`` false) are drawn
    muted and labelled. ``selected`` (an iteration) marks the posterior chosen by cross
    validation with a dashed line.
    """
    plt = _pyplot(font_size)
    accepted = [True] * len(values) if accepted is None else list(accepted)
    data = [np.maximum(v, np.finfo(float).tiny) if log else np.asarray(v) for v in values]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(values) + 3.0), 4.2), layout="constrained")
    positions = list(range(len(values)))
    if kind == "violin":
        parts = ax.violinplot(data, positions=positions, widths=0.7, showmeans=True)
        for body, ok in zip(parts["bodies"], accepted, strict=True):
            body.set_facecolor(POSTERIOR if ok else MUTED)
            body.set_edgecolor(POSTERIOR if ok else MUTED)
            body.set_alpha(0.35)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color(INK if key == "cmeans" else MUTED)
            parts[key].set_linewidth(1.0)
    else:
        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": INK, "linewidth": 1.2},
            whiskerprops={"color": MUTED},
            capprops={"color": MUTED},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "none",
                "markeredgecolor": MUTED,
            },
        )
        for patch, ok in zip(boxes["boxes"], accepted, strict=True):
            colour = POSTERIOR if ok else MUTED
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
            patch.set_edgecolor(colour)
    for position, ok in enumerate(accepted):
        if not ok:
            ax.annotate(
                "rejected",
                (position, 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=font_size * 0.65,
                color=TEXT_SECONDARY,
            )
    if selected is not None and selected in iterations:
        # Cross validation selects by the member mean (the paper's average mismatch), so
        # show the means: a box's median can sit lower than the selected iterate's.
        ax.plot(
            positions,
            [float(np.mean(v)) for v in data],
            linestyle="none",
            marker="D",
            markersize=4,
            color=INK,
            label="mean",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=font_size * 0.75)
        at = [p for p, i, ok in zip(positions, iterations, accepted, strict=True) if i == selected]
        ax.axvline(at[-1], color=INK, linestyle="--", linewidth=1.0)
        ax.annotate(
            "selected (lowest mean CV mismatch)",
            (at[-1], 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=font_size * 0.75,
            color=TEXT_SECONDARY,
        )
    if log:
        ax.set_yscale("log")
    ax.set_xticks(positions, [str(i) for i in iterations])
    ax.set_xlabel("iteration (0 = prior)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    png = _save(fig, path)
    plt.close(fig)
    return png


def metric_comparison_figure(
    series: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    path: Path,
    *,
    title: str,
    ylabel: str,
    log: bool = False,
    font_size: float = FONT_SIZE,
) -> Path:
    """Plot a member-mean metric against iteration for up to four experiments.

    ``series`` maps an experiment's name to ``(iterations, values)``; lines are
    direct-labelled.

    Raises
    ------
    ValueError
        For more than four experiments (the validated categorical set; facet instead).
    """
    if not 1 <= len(series) <= len(SERIES):
        raise ValueError(f"compare 1 to {len(SERIES)} experiments per figure, got {len(series)}")
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    ends = []
    for colour, (name, (iterations, values)) in zip(SERIES, series.items(), strict=False):
        x, y = list(iterations), list(values)
        ax.plot(x, y, color=colour, linewidth=2, marker="o", markersize=4, label=name)
        ends.append((name, x[-1], y[-1]))
    if log:
        ax.set_yscale("log")
    ax.margins(x=0.25)
    # Direct labels at the line ends, in text ink (the markers carry the colour), nudged
    # apart vertically where lines end close together.
    fig.canvas.draw()
    to_pixels = ax.transData.transform
    gap = 1.1 * font_size * 0.8 * fig.dpi / 72  # about one label height
    placed: list[tuple[float, float]] = []
    for name, end_x, end_y in sorted(ends, key=lambda e: to_pixels((e[1], e[2]))[1]):
        px, py = to_pixels((end_x, end_y))
        target = py
        for qx, qy in placed:
            if abs(qx - px) < 4 * gap and target < qy + gap:
                target = qy + gap
        placed.append((px, target))
        ax.annotate(
            name,
            (end_x, end_y),
            xytext=(6, (target - py) * 72 / fig.dpi),
            textcoords="offset points",
            va="center",
            fontsize=font_size * 0.8,
            color=TEXT_SECONDARY,
        )
    ax.set_xlabel("iteration (0 = prior)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    png = _save(fig, path)
    plt.close(fig)
    return png


# ----------------------------------------------------------------------- production data


def production_figures(
    times: Sequence[str],
    datatypes: Sequence[str],
    observed: NDArray[np.float64],
    variances: NDArray[np.float64],
    prior: NDArray[np.float64],
    posterior: NDArray[np.float64],
    output_dir: Path,
    *,
    truth: NDArray[np.float64] | None = None,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Plot the initial and final ensemble forecasts of each data type, one figure each.

    PET-fork's ``plot_prod_separate``: the 0-100 and 10-90 percentile bands of the
    ensemble (light grey and grey, with black lines, dotted for 0 and 100), its mean
    (orange), the observations (red, with +-2 sigma bars) and, in synthetic cases, the
    true model's noise-free response (green). The initial and final figures of a type
    share the y range. Row ``i`` of the ``(n_data, n_members)`` forecasts is
    ``datatypes[i]`` at ``times[i]``. Files: ``<well>_<type>_initial`` and
    ``<well>_<type>_final`` for ``TYPE:WELL`` names, else ``<type>_initial``/``_final``.
    """
    plt = _pyplot(font_size)
    rows_of: dict[str, list[int]] = {}
    for i, datatype in enumerate(datatypes):
        rows_of.setdefault(datatype, []).append(i)
    figures = []
    for datatype, rows in rows_of.items():
        x = _positions([times[i] for i in rows])
        quantity, _, well = datatype.partition(":")
        stem = _slug(f"{well}_{quantity}" if well else quantity)
        drawn = {}
        for stage, ensemble in (("initial", prior[rows]), ("final", posterior[rows])):
            fig, ax = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
            ax.fill_between(
                x,
                ensemble.min(axis=1),
                ensemble.max(axis=1),
                facecolor=BAND_OUTER,
                label="0-100 pctl",
            )
            ax.fill_between(
                x,
                np.percentile(ensemble, 10, axis=1),
                np.percentile(ensemble, 90, axis=1),
                facecolor=BAND_INNER,
                label="10-90 pctl",
            )
            for q, style in ((0, ":"), (10, "-"), (90, "-"), (100, ":")):
                ax.plot(
                    x,
                    np.percentile(ensemble, q, axis=1),
                    color=BAND_LINE,
                    linestyle=style,
                    linewidth=0.9,
                )
            ax.plot(x, ensemble.mean(axis=1), color=MEAN_COLOUR, linewidth=1.8, label="mean")
            if truth is not None:
                ax.plot(x, truth[rows], color=TRUE_COLOUR, linewidth=1.6, label="true response")
            ax.errorbar(
                x,
                observed[rows],
                yerr=2 * np.sqrt(variances[rows]),
                fmt=".",
                markersize=8,
                color=OBSERVED_COLOUR,
                ecolor=OBSERVED_COLOUR,
                elinewidth=0.8,
                capsize=2,
                label="observed ±2 sd",
                zorder=4,
            )
            where = f", at well {well}" if well else ""
            ax.set_title(f"{quantity} {stage} forecast{where}", loc="left")
            ax.set_xlabel("report time")
            ax.set_ylabel(quantity)
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(False)  # as PET-fork: grid lines would cut through the bands
            ax.legend(loc="upper right", ncols=2)
            drawn[stage] = (fig, ax)
        # Both stages on one y range (PET-fork uses the initial one), widened to show
        # observations and the truth.
        lows, highs = zip(*(ax.get_ylim() for _, ax in drawn.values()), strict=True)
        for stage, (fig, ax) in drawn.items():
            ax.set_ylim(min(lows), max(highs))
            figures.append(_save(fig, output_dir / f"{stem}_{stage}"))
            plt.close(fig)
    return figures


# ------------------------------------------------------------------------------- fields


def _layers(vector: NDArray[np.float64], grid: Sequence[int], order: str) -> NDArray[np.float64]:
    """Reshape a field ``(n_cells,)`` to ``(nx, ny, nz)`` (one layer for 1-D and 2-D grids)."""
    shape = tuple(int(n) for n in grid)
    shape = (*shape, 1, 1)[:3] if len(shape) < 3 else shape
    if math.prod(shape) != vector.shape[0]:
        raise ValueError(f"grid {tuple(grid)} does not fit {vector.shape[0]} cells")
    result: NDArray[np.float64] = np.reshape(vector, shape, order="F" if order == "F" else "C")
    return result


def _limits(arrays: Sequence[NDArray[np.float64]]) -> tuple[float, float]:
    """Colour range over ``arrays``, widened by 5% (PET-fork's ``plot_layer`` margin)."""
    lo = min(float(np.nanmin(a)) for a in arrays)
    hi = max(float(np.nanmax(a)) for a in arrays)
    margin = 0.05 * max(hi - lo, abs(hi), abs(lo), 1e-12)
    return lo - margin, hi + margin


def _map_figure(
    layer: NDArray[np.float64],
    title: str,
    label: str,
    limits: tuple[float, float] | None,
    path: Path,
    font_size: float,
) -> Path:
    """Draw one map of a layer, PET-fork style (``pcolormesh``, ``jet``, y axis down)."""
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(6.0, 4.8), layout="constrained")
    nx, ny = layer.shape
    x, y = np.mgrid[0:nx, 0:ny]
    mesh = ax.pcolormesh(x, y, layer, cmap=FIELD_CMAP, shading="auto")
    if limits is not None:
        mesh.set_clim(*limits)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_title(title, loc="left")
    bar = fig.colorbar(mesh, ax=ax)
    bar.set_label(label)
    png = _save(fig, path)
    plt.close(fig)
    return png


def field_layer_figures(
    name: str,
    grid: Sequence[int],
    prior: NDArray[np.float64],
    posterior: NDArray[np.float64],
    output_dir: Path,
    *,
    truth: NDArray[np.float64] | None = None,
    order: Literal["F", "C"] = "F",
    layers: Sequence[int] | None = None,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Map, per layer, the true field and the ensemble's means and spreads, one each.

    PET-fork's ``plot_layer``. For every layer (0-based ``layers``, default all):
    ``<name>_true_layer<k>`` (synthetic cases), ``<name>_prior_mean_layer<k>``,
    ``<name>_final_mean_layer<k>`` on one colour range (the true field's when known, as
    PET-fork), ``<name>_prior_std_layer<k>`` and ``<name>_final_std_layer<k>`` on one
    range, ``<name>_final_minus_prior_layer<k>`` and ``<name>_final_minus_true_layer<k>``.
    ``prior``/``posterior`` are ``(n_cells, n_members)`` in ``order`` on ``grid``
    (``(nx, ny)`` or ``(nx, ny, nz)``); ``k`` counts from 1.
    """
    prior_mean = _layers(prior.mean(axis=-1), grid, order)
    final_mean = _layers(posterior.mean(axis=-1), grid, order)
    prior_std = _layers(prior.std(axis=-1, ddof=1), grid, order)
    final_std = _layers(posterior.std(axis=-1, ddof=1), grid, order)
    true_field = None if truth is None else _layers(np.asarray(truth), grid, order)
    chosen = list(range(prior_mean.shape[2])) if layers is None else list(layers)
    references = (
        [true_field[:, :, chosen]]
        if true_field is not None
        else [prior_mean[:, :, chosen], final_mean[:, :, chosen]]
    )
    values = _limits(references)
    spreads = _limits([prior_std[:, :, chosen], final_std[:, :, chosen]])
    figures = []
    for k in chosen:
        at = f" at layer {k + 1}" if prior_mean.shape[2] > 1 else ""
        maps: list[tuple[str, NDArray[np.float64], str, tuple[float, float] | None]] = []
        if true_field is not None:
            maps.append(("true", true_field[:, :, k], f"True {name}{at}", values))
        maps += [
            ("prior_mean", prior_mean[:, :, k], f"Prior mean, {name}{at}", values),
            ("final_mean", final_mean[:, :, k], f"Final mean, {name}{at}", values),
            ("prior_std", prior_std[:, :, k], f"Prior std, {name}{at}", spreads),
            ("final_std", final_std[:, :, k], f"Final std, {name}{at}", spreads),
            (
                "final_minus_prior",
                final_mean[:, :, k] - prior_mean[:, :, k],
                f"Final - prior mean, {name}{at}",
                None,
            ),
        ]
        if true_field is not None:
            maps.append(
                (
                    "final_minus_true",
                    final_mean[:, :, k] - true_field[:, :, k],
                    f"Final mean - true, {name}{at}",
                    None,
                )
            )
        for quantity, layer, title, limits in maps:
            label = f"std of {name}" if quantity.endswith("_std") else name
            path = output_dir / f"{_slug(name)}_{quantity}_layer{k + 1}"
            figures.append(_map_figure(layer, title, label, limits, path, font_size))
    return figures


def histogram_figures(
    name: str,
    prior: NDArray[np.float64],
    posterior: NDArray[np.float64],
    output_dir: Path,
    *,
    truth: NDArray[np.float64] | None = None,
    bins: int = 40,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Plot the histogram of a parameter over all cells and members, prior and final apart.

    PET-fork's ``plot_histogram`` (one figure per ensemble): ``<name>_prior`` and
    ``<name>_final``, on the same bins; the true field's distribution (synthetic cases)
    is an ink outline in both.
    """
    plt = _pyplot(font_size)
    values = [prior.ravel(), posterior.ravel()]
    if truth is not None:
        values.append(np.ravel(truth))
    lo, hi = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    figures = []
    for stage, ensemble in (("prior", prior), ("final", posterior)):
        fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
        ax.hist(
            ensemble.ravel(),
            bins=edges,
            density=True,
            color=POSTERIOR,
            alpha=0.45,
            label=f"{stage} ensemble",
        )
        if truth is not None:
            ax.hist(
                np.ravel(truth),
                bins=edges,
                density=True,
                histtype="step",
                color=INK,
                linewidth=1.4,
                label="true field",
            )
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"{stage.capitalize()} distribution of {name}", loc="left")
        ax.legend()
        figures.append(_save(fig, output_dir / f"{_slug(name)}_{stage}"))
        plt.close(fig)
    return figures


def fields_comparison_figure(
    name: str,
    grid: Sequence[int],
    fields: Mapping[str, NDArray[np.float64]],
    path: Path,
    *,
    order: Literal["F", "C"] = "F",
    layers: Sequence[int] | None = None,
    value_label: str | None = None,
    font_size: float = FONT_SIZE,
) -> Path:
    """Map several fields side by side on one colour range, one row per layer (on request).

    Not used by the default reports, which draw each field alone
    (:func:`field_layer_figures`).
    """
    plt = _pyplot(font_size)
    arrays = {title: _layers(np.asarray(v), grid, order) for title, v in fields.items()}
    chosen = list(range(next(iter(arrays.values())).shape[2])) if layers is None else list(layers)
    limits = _limits([a[:, :, chosen] for a in arrays.values()])
    fig, axes = plt.subplots(
        len(chosen),
        len(arrays),
        figsize=(3.0 * len(arrays) + 1.0, 2.8 * len(chosen) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    mesh = None
    for row, k in enumerate(chosen):
        for col, (title, array) in enumerate(arrays.items()):
            ax = axes[row, col]
            nx, ny = array.shape[:2]
            x, y = np.mgrid[0:nx, 0:ny]
            mesh = ax.pcolormesh(
                x,
                y,
                array[:, :, k],
                cmap=FIELD_CMAP,
                shading="auto",
                vmin=limits[0],
                vmax=limits[1],
            )
            ax.invert_yaxis()
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, loc="left")
            if col == 0:
                ax.set_ylabel(f"layer {k + 1}")
    bar = fig.colorbar(mesh, ax=list(axes.flat))
    bar.set_label(value_label or name)
    png = _save(fig, path)
    plt.close(fig)
    return png


def field_map_figures(
    name: str,
    grid: Sequence[int],
    field: NDArray[np.float64],
    output_dir: Path,
    *,
    title: str | None = None,
    limits: tuple[float, float] | None = None,
    order: Literal["F", "C"] = "F",
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Map one field ``(n_cells,)``, one figure per layer: ``<name>_layer<k>``.

    For fields outside an ensemble summary, e.g. single realisations; ``limits`` fixes
    the colour range (default the field's own).
    """
    layers = _layers(np.asarray(field, dtype=np.float64), grid, order)
    limits = limits or _limits([layers])
    figures = []
    for k in range(layers.shape[2]):
        at = f" at layer {k + 1}" if layers.shape[2] > 1 else ""
        path = output_dir / f"{_slug(name)}_layer{k + 1}"
        figures.append(
            _map_figure(layers[:, :, k], f"{title or name}{at}", name, limits, path, font_size)
        )
    return figures


# ------------------------------------------------------------------------------ health


def _threshold(
    ax: Any,  # noqa: ANN401 (a matplotlib Axes)
    value: float,
    label: str,
    font_size: float,
    colour: str = INK,
) -> None:
    # Beneath the data, labelled at the left: these series start far from their threshold
    # and approach it on the right, where a label would collide with the last points.
    ax.axhline(value, color=colour, linestyle="--", linewidth=1.0, zorder=1)
    ax.annotate(
        label,
        (0, value),
        xycoords=("axes fraction", "data"),
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        fontsize=font_size * 0.75,
        color=TEXT_SECONDARY,
    )


def spread_figures(
    iterations: Sequence[int],
    ratios: Mapping[str, Sequence[float]],
    output: Path,
    *,
    threshold: float = 0.05,
    font_size: float = FONT_SIZE,
) -> list[Path]:
    """Plot each variable's ensemble spread relative to the prior's, per accepted iterate.

    ``ratios[name][i]`` is the members' standard deviation per cell, averaged over cells,
    at ``iterations[i]`` over the prior's (1 at iteration 0). ``threshold`` marks the run
    monitor's collapse threshold (``[agents.monitor] collapse_spread_ratio``). Up to four
    variables share one figure (``spread``), in the palette's fixed categorical order and
    labelled at their ends; more get one figure each (``spread_<name>``).
    """
    plt = _pyplot(font_size)
    groups = [dict(ratios)] if len(ratios) <= len(SERIES) else [{k: v} for k, v in ratios.items()]
    paths = []
    for group in groups:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
        for colour, (name, values) in zip(SERIES, group.items(), strict=False):
            ax.plot(iterations, values, color=colour, linewidth=2.0, marker="o", markersize=4)
            ax.annotate(
                name,
                (iterations[-1], values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=font_size * 0.8,
                color=TEXT_SECONDARY,
            )
        _threshold(ax, threshold, f"collapse threshold {threshold:g}", font_size)
        ax.set_yscale("log")
        ax.set_xticks(list(iterations), [str(i) for i in iterations])
        ax.set_xlabel("accepted iterate (0 = prior)")
        ax.set_ylabel("spread / prior spread")
        name = "spread" if len(groups) == 1 else f"spread_{_slug(next(iter(group)))}"
        ax.set_title("Ensemble spread relative to the prior", loc="left")
        paths.append(_save(fig, output / name))
        plt.close(fig)
    return paths


def data_mismatch_change_figure(
    iterations: Sequence[int],
    changes: Sequence[float],
    path: Path,
    *,
    tolerance: float | None = None,
    font_size: float = FONT_SIZE,
) -> Path:
    """Plot the relative change ``|1 - m_k / m_(k-1)|`` of the accepted mean data mismatch.

    ``changes[i]`` is the change reaching ``iterations[i]`` (the second accepted iterate
    onwards). ``tolerance`` marks the method's convergence tolerance, if it has one.
    """
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    floor = np.finfo(float).tiny
    ax.plot(iterations, [max(c, floor) for c in changes], color=POSTERIOR, linewidth=2.0,
            marker="o", markersize=4)  # fmt: skip
    if tolerance is not None:
        _threshold(ax, tolerance, f"convergence tolerance {tolerance:g}", font_size)
    ax.set_yscale("log")
    ax.set_xticks(list(iterations), [str(i) for i in iterations])
    ax.set_xlabel("accepted iterate")
    ax.set_ylabel("relative data mismatch change")
    ax.set_title("Convergence: change of the mean data mismatch", loc="left")
    png = _save(fig, path)
    plt.close(fig)
    return png


def damping_figure(
    iterations: Sequence[int],
    dampings: Sequence[float],
    accepted: Sequence[bool],
    path: Path,
    *,
    font_size: float = FONT_SIZE,
    label: str = "damping",
) -> Path:
    """Plot the damping of every trial of an iterative method; rejected trials hollow.

    A run of rejections with the damping climbing is the "damping runaway" the run
    monitor flags. ``label`` names the control (GN-EnRML: "step length").
    """
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    ax.plot(iterations, dampings, color=BASELINE, linewidth=1.0)
    ok = [i for i, a in enumerate(accepted) if a]
    bad = [i for i, a in enumerate(accepted) if not a]
    ax.plot([iterations[i] for i in ok], [dampings[i] for i in ok], linestyle="none",
            marker="o", markersize=7, color=POSTERIOR, label="accepted")  # fmt: skip
    if bad:
        ax.plot([iterations[i] for i in bad], [dampings[i] for i in bad], linestyle="none",
                marker="o", markersize=7, markerfacecolor="none", markeredgecolor=MUTED,
                label="rejected")  # fmt: skip
    ax.legend(loc="best", frameon=False, fontsize=font_size * 0.75)
    ax.set_yscale("log")
    ax.set_xlabel("trial")
    ax.set_ylabel(label)
    ax.set_title(f"{label[0].upper()}{label[1:]} of each trial", loc="left")
    png = _save(fig, path)
    plt.close(fig)
    return png


def members_figure(
    iterations: Sequence[int],
    members: Sequence[int],
    ensemble_size: int,
    path: Path,
    *,
    font_size: float = FONT_SIZE,
) -> Path:
    """Plot the members that simulated successfully at each forecast, against the size."""
    plt = _pyplot(font_size)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    ax.step(iterations, members, where="post", color=POSTERIOR, linewidth=2.0, zorder=3)
    # A reference, not a limit to watch: muted, so a run without failures still shows its
    # (coinciding) line.
    _threshold(ax, ensemble_size, f"ensemble size {ensemble_size}", font_size, BASELINE)
    ax.set_ylim(0, ensemble_size * 1.08)
    ax.set_xlabel("forecast (iteration or trial)")
    ax.set_ylabel("members simulated")
    ax.set_title("Members that simulated successfully", loc="left")
    png = _save(fig, path)
    plt.close(fig)
    return png
