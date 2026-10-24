"""The classic result figures of an ensemble history-matching study.

Origin: the figure set of PET-fork's Brugge study (``not public``
@ ``1348861``: ``Plotting/plot_objective_function.py`` ``boxplot_or_violinplot``,
``Plotting/metric_analyses.py`` ``metric_plots``, ``Plotting/plot_data.py``
``plot_prod_separate``, ``Plotting/plot_parameters.py`` ``plot_layer`` and ``plot_histogram``,
``Examples/.../Brugge/plot_mec_data.py``). The figures keep its file names, sizes, fonts,
colours, titles and legends, typos included (``forcast`` in the titles, WWCT labelled
``Sm3/Day``), on matplotlib's default style. The one exception is a file name: the reference
wrote the metric files' violin plots as ``*_violionplot``; they are ``*_violinplot`` here, as
the objective's always were (a release tester's finding, 2026-10-20; nothing reads them back).
Redrawing a violin plot removes that figure's ``*_violionplot.png`` and ``.pdf`` from the
folder it is drawn into, the run's plots folder or an ``--output`` folder alike: exactly
those names, nothing else (so a listing shows no duplicates).

Five behaviours of the reference are not reproduced (the user's decision, 2026-10-11):

1. GIES's mean-model forecast, kept there as a 104th column, counted as a member in the
   data figures: these figures use the members only.
2. One empty extra tick in every metric figure, and the model-change boxes labelled one
   iteration early: ticks are at the iterates, labelled by iterate.
3. The final band figure's ``tight_layout`` ran before its axis labels were set, and the
   histogram had none, so their PNGs lost the labels: they are kept.
4. Layer maps: the prior's PNG at 500 dpi uncropped and the others at 100 dpi cropped by
   ImageMagick's ``convert -trim``: all at 100 dpi, cropped with ``bbox_inches="tight"``.
5. Residual PDFs renamed by hand.

Also, a layer map's title says ``at layer L`` whenever the grid has more than one layer,
for every kind: the reference tested ``nz > 1`` for ``ref`` and ``prior`` but ``nx > 1``
for the others, so a one-layer grid's titles disagreed.

"Final" and "posterior" are the run's posterior: the minimum-CV iterate with cross
validation, else the last accepted one (the reference had no cross validation).

Accepted iterates are numbered 0..n in order, as the reference numbers them (APEX's iteration
numbers also count rejected trials).

What the set draws by default (the user, 2026-10-12): the objective and ``data_mismatch/``
show ``data_mismatch``, the objective the methods minimise (a sum against the perturbed
observations), not the reference's ``normalized_data_mismatch`` (a mean square against the
unperturbed observations, so not a mere rescaling), which is drawn only when asked for;
``model_change`` is recorded by every run but drawn only when asked for.
"""

import datetime as dt
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from apex.core.figures import FigureRequest, GridView, RunView
from apex.core.options import Text, TextList
from apex.core.plugins import figure_presets, figures
from apex.domain.observations import DataKey

#: Box and violin plots (``metric_plots``, ``boxplot_or_violinplot``).
DISTRIBUTION_SIZE = (20.0, 16.0)
DISTRIBUTION_DPI = 500.0
#: Matplotlib's default figure (bands, layer maps, histograms), saved at 100 dpi.
DEFAULT_DPI = 100.0
#: The file-name word for a violin plot (the reference's metric files spelt it ``violionplot``).
VIOLIN_WORD = "violinplot"
#: The word earlier versions wrote; a redraw removes those files (no duplicate figures).
OLD_VIOLIN_WORD = "violionplot"
#: Bounds of the options a request may set (a model's plan included).
MAX_BINS = 1000
MAX_FONT_SIZE = 100.0
#: The layer maps' kinds, in the reference's order.
LAYER_KINDS = ("ref", "prior", "post", "diff", "std_prior", "std_post")
_LAYER_TITLES = {
    "ref": "Reference {name}",
    "prior": "Prior, {name}",
    "post": "Posterior, {name}",
    "diff": "Posterior - Prior, {name}",
    "std_prior": "Prior std {name}",
    "std_post": "Posterior std {name}",
}


@contextmanager
def _pyplot() -> Iterator[Any]:
    """Yield ``matplotlib.pyplot`` on matplotlib's default style (the reference's look).

    APEX's own figures change the global rcParams (``reporting.plots``); these figures are
    drawn inside the default style whatever ran before them in the process.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.style.context("default"):
        yield plt


def _save(plt: Any, fig: Any, stem: Path, dpi: float, *, crop: bool = False) -> Path:  # noqa: ANN401
    """Write ``stem.png`` (``dpi``; cropped to its content if ``crop``) and ``stem.pdf``."""
    # Only the figure's own folder: a removed output (a deleted run) is not recreated.
    stem.parent.mkdir(exist_ok=True)
    png = stem.parent / f"{stem.name}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight" if crop else None)
    fig.savefig(stem.parent / f"{stem.name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _shown_label(label: str) -> str:
    """Return a metric label as the reference names it: ``WOPR:BR-P-1`` -> ``WOPR BR-P-1``."""
    return _file_part(label.replace(":", " "))


def _file_part(text: str) -> str:
    """Return ``text`` safe as part of a file name: no folder separators, no ``..``.

    Labels and well names come from the run's data, and a prefix from a request: none may
    place a file outside the output folder.
    """
    return text.replace("/", "_").replace("\\", "_").replace("..", "_") or "_"


def _checked_prefix(prefix: str) -> str:
    """Return a file prefix, refusing one that names a folder."""
    if any(c in prefix for c in "/\\") or ".." in prefix:
        raise ValueError(f"prefix {prefix!r} must be a plain file-name prefix (no folders)")
    return prefix


def _type_and_well(datatype: str) -> tuple[str, str]:
    """Split ``TYPE:WELL`` (with any prefix, e.g. ``MEC:WOPR:BR-P-5``) into type and well."""
    parts = datatype.split(":")
    if len(parts) < 2:
        return datatype, ""
    return parts[-2], parts[-1]


# ----------------------------------------------------------------------- distributions


def _distribution(
    plt: Any,  # noqa: ANN401
    view: RunView,
    metric: str,
    label: str,
    kind: Literal["box", "violin"],
    stem: Path,
    *,
    ylabel: str,
    font_size: float,
    dpi: float,
) -> Path | None:
    """Draw one metric's per-member values over the accepted iterates as a box or violin plot."""
    iterations, values, _ = view.metric(metric, label, accepted_only=True)
    index = {k: n for n, k in enumerate(view.accepted_iterations)}
    rows = [
        (index.get(k, n), v[np.isfinite(v)])
        for n, (k, v) in enumerate(zip(iterations, values, strict=True))
    ]
    rows = [(n, v) for n, v in rows if v.size]
    if not rows:
        return None
    data = [v for _, v in rows]
    positions = np.arange(1, len(rows) + 1)
    fig = plt.figure()
    fig.set_size_inches(DISTRIBUTION_SIZE)
    if kind == "violin":
        plt.violinplot(data, positions=positions, showmeans=True, showmedians=False)
    else:
        plt.boxplot(data, positions=positions, notch=True, showfliers=True, showmeans=True)
    # Ticks at the iterates, labelled by accepted iterate (the reference had one tick too
    # many and labelled the model change one iteration early).
    plt.xticks(positions, [str(n) for n, _ in rows])
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Iteration no.", size=font_size)
    plt.ylabel(ylabel, size=font_size)
    fig.tight_layout(pad=2.0)
    return _save(plt, fig, stem, dpi)


@figures.register("metric_distribution")
class MetricDistribution:
    """Box and violin plots of a per-member metric over the accepted iterates.

    Parameters
    ----------
    metric
        The recorded metric (``rmse``, ``ssim``, ``model_change``,
        ``normalized_data_mismatch``, ...).
    labels
        Its labels to draw (default: every one, ``all`` last).
    kinds
        ``violin`` and/or ``box``, in the order drawn.
    prefix
        The file prefix and y-label word (default: the metric): files
        ``<prefix>_<label>_violinplot`` and ``<prefix>_<label>_boxplot``, y label
        ``<prefix> (<label>)``; a label ``TYPE:WELL`` is written ``TYPE WELL``.
    font_size
        Tick and axis label size.
    """

    def __init__(
        self,
        metric: Text,
        labels: TextList = (),
        kinds: TextList = ("violin", "box"),
        prefix: Text = "",
        font_size: float = 28.0,
    ) -> None:
        unknown = sorted(set(kinds) - {"violin", "box"})
        if unknown or not kinds:
            raise ValueError(f"kinds must be 'violin' and/or 'box', got {list(kinds)}")
        if not 0 < font_size <= MAX_FONT_SIZE:
            raise ValueError(f"font_size must be in (0, {MAX_FONT_SIZE:g}], got {font_size}")
        self.metric = metric
        self.labels = tuple(labels)
        self.kinds = tuple(kinds)
        self.prefix = _checked_prefix(prefix or metric)
        self.font_size = font_size

    def applies(self, view: RunView) -> str | None:
        """Why the run lacks the metric or a label, or ``None``."""
        recorded = view.metric_labels()
        if self.metric not in recorded:
            return f"the run recorded no {self.metric!r}"
        missing = [label for label in self.labels if label not in recorded[self.metric]]
        if missing:
            return f"the run recorded no {self.metric!r} for {missing}"
        return None

    def _labels(self, view: RunView) -> list[str]:
        if self.labels:
            return list(self.labels)
        recorded = view.metric_labels()[self.metric]
        return [label for label in recorded if label != "all"] + (
            ["all"] if "all" in recorded else []
        )

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the plots (PNG at 500 dpi times ``dpi_scale``, and PDF)."""
        written: list[Path] = []
        with _pyplot() as plt:
            for label in self._labels(view):
                shown = _shown_label(label)
                for kind in self.kinds:
                    word = VIOLIN_WORD if kind == "violin" else "boxplot"
                    png = _distribution(
                        plt, view, self.metric, label, kind,  # type: ignore[arg-type]
                        output_dir / f"{self.prefix}_{shown}_{word}",
                        ylabel=f"{self.prefix} ({shown})", font_size=self.font_size,
                        dpi=DISTRIBUTION_DPI * dpi_scale,
                    )  # fmt: skip
                    if png is not None:
                        written.append(png)
                        if kind == "violin":
                            _remove_old_violin(output_dir / f"{self.prefix}_{shown}")
        return written


def _remove_old_violin(base: Path) -> None:
    """Remove ``<base>_violionplot.{png,pdf}``, an earlier version's name of a redrawn figure.

    In whatever folder the figure is drawn into (a run's plots, or ``--output``): only those
    two exact names.
    """
    for suffix in (".png", ".pdf"):
        base.with_name(f"{base.name}_{OLD_VIOLIN_WORD}{suffix}").unlink(missing_ok=True)


@figures.register("objective_distribution")
class ObjectiveDistribution:
    """The objective over all data, as box and violin plots.

    ``data_mismatch`` by default, the objective the methods minimise; or
    ``normalized_data_mismatch``, the mean square against the unperturbed observations, as
    the reference's objective figures. Files ``obj_func_boxplot`` and ``obj_func_violinplot``.

    Parameters
    ----------
    metric
        The objective's metric.
    font_size
        Tick and axis label size.
    """

    def __init__(
        self,
        metric: Literal["data_mismatch", "normalized_data_mismatch"] = "data_mismatch",
        font_size: float = 30.0,
    ) -> None:
        if metric not in ("data_mismatch", "normalized_data_mismatch"):
            raise ValueError(f"metric must be data_mismatch or normalized_data_mismatch, "
                             f"got {metric!r}")  # fmt: skip
        if not 0 < font_size <= MAX_FONT_SIZE:
            raise ValueError(f"font_size must be in (0, {MAX_FONT_SIZE:g}], got {font_size}")
        self.metric = metric
        self.font_size = font_size

    def applies(self, view: RunView) -> str | None:
        """Why the run lacks the objective, or ``None``."""
        if "all" not in view.metric_labels().get(self.metric, []):
            return f"the run recorded no {self.metric!r}"
        return None

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the box and the violin plot."""
        written: list[Path] = []
        with _pyplot() as plt:
            for kind, word in (("box", "boxplot"), ("violin", VIOLIN_WORD)):
                png = _distribution(
                    plt, view, self.metric, "all", kind,  # type: ignore[arg-type]
                    output_dir / f"obj_func_{word}",
                    ylabel="Data mismatch" if self.metric == "data_mismatch"
                    else "Normalized data mismatch",
                    font_size=self.font_size, dpi=DISTRIBUTION_DPI * dpi_scale,
                )  # fmt: skip
                if png is not None:
                    written.append(png)
        return written


# --------------------------------------------------------------------------- data bands


def _days(times: Sequence[str], start: dt.date | None) -> NDArray[np.float64]:
    """Return report times as days since ``start`` (else since the first time).

    Times that are not ISO dates are used as numbers (step indices), else by their order.
    """
    try:
        dates = [dt.datetime.fromisoformat(t) for t in times]
    except ValueError:
        try:
            return np.array([float(t) for t in times])
        except ValueError:
            return np.arange(len(times), dtype=float)
    if not dates:
        return np.zeros(0)
    origin = dt.datetime.combine(start, dt.time()) if start is not None else min(dates)
    return np.array([(d - origin).total_seconds() / 86400.0 for d in dates])


def _bands(
    ax: Any,  # noqa: ANN401
    x: NDArray[np.float64],
    ensemble: NDArray[np.float64],
    observed: NDArray[np.float64] | None,
) -> list[tuple[Any, str]]:
    """Draw the reference's bands of ``ensemble`` ``(n_times, n_members)``, in its order.

    The percentile lines, the observations, the two bands, then the mean; returns the
    legend's handles and texts.
    """
    p0, p10, p90, p100 = (np.percentile(ensemble, q, axis=1) for q in (0, 10, 90, 100))
    ax.plot(x, p90, "k")
    ax.plot(x, p100, ":k")
    ax.plot(x, p10, "k")
    ax.plot(x, p0, ":k")
    handles: list[tuple[Any, str]] = []
    if observed is not None:
        (dots,) = ax.plot(x, observed, ".r")
        handles.append((dots, "obs"))
    ax.fill_between(x, p100, p0, facecolor="lightgrey")
    ax.fill_between(x, p90, p10, facecolor="grey")
    outer = ax.fill(np.nan, np.nan, "lightgrey")
    inner = ax.fill(np.nan, np.nan, "grey")
    (mean,) = ax.plot(x, np.mean(ensemble, axis=1), "orange")
    return [*handles, (mean, "mean"), (outer[0], "0-100 pctl"), (inner[0], "10-90 pctl")]


@figures.register("data_bands")
class DataBands:
    """Per data type and well, the initial and final ensemble forecasts as percentile bands.

    Each scalar data type ``TYPE:WELL`` gets ``<WELL>_<TYPE>_initial`` (the prior forecast,
    with the legend) and ``<WELL>_<TYPE>`` (the final iterate's, on the initial figure's y
    limits): 0-100 percentiles light grey between dotted black lines, 10-90 grey between
    solid black lines, the mean orange, the observations as red dots. The x axis is days
    since the deck's ``START``.

    Parameters
    ----------
    source
        ``predicted``: the forecasts of the observations; ``auxiliary``: the forecasts saved
        beside them but never assimilated (a model-error correction's residuals, MEC).
    observations
        Draw the observations (default: for ``predicted`` only).
    datatypes
        Only these data types (``WOPR:BR-P-1``; default all scalar ones).
    """

    def __init__(
        self,
        source: Literal["predicted", "auxiliary"] = "predicted",
        observations: bool | None = None,
        datatypes: TextList = (),
    ) -> None:
        if source not in ("predicted", "auxiliary"):
            raise ValueError(f"source must be 'predicted' or 'auxiliary', got {source!r}")
        self.source = source
        self.observations = source == "predicted" if observations is None else observations
        if self.observations and source == "auxiliary":
            raise ValueError("auxiliary forecasts have no observations to draw")
        self.datatypes = tuple(datatypes)

    def applies(self, view: RunView) -> str | None:
        """Why the run has no such forecasts, or ``None``."""
        if not self._series(view, 0):
            what = "auxiliary forecasts" if self.source == "auxiliary" else "scalar data"
            return f"the run saved no {what}"
        return None

    def _series(self, view: RunView, iteration: int) -> dict[str, tuple[list[int], list[DataKey]]]:
        """Return each data type's rows and keys in the source's data space."""
        iterate = view.iterate(iteration)
        if self.source == "auxiliary":
            keys: Sequence[DataKey] = iterate.auxiliary_keys
            if iterate.auxiliary_predicted is None:
                return {}
        else:
            keys = view.observations().keys
        found: dict[str, tuple[list[int], list[DataKey]]] = {}
        for row, key in enumerate(keys):
            if key.index is not None or (self.datatypes and key.datatype not in self.datatypes):
                continue
            rows, kept = found.setdefault(key.datatype, ([], []))
            rows.append(row)
            kept.append(key)
        return found

    def _values(self, view: RunView, iteration: int) -> NDArray[np.float64]:
        iterate = view.iterate(iteration)
        if self.source == "auxiliary":
            assert iterate.auxiliary_predicted is not None  # checked by _series
            return np.asarray(iterate.auxiliary_predicted, dtype=float)
        return np.asarray(iterate.predicted, dtype=float)

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the initial and final figure of every data type."""
        series = self._series(view, 0)
        initial = self._values(view, 0)
        final = self._values(view, view.posterior_iteration)
        observed = view.observations() if self.observations else None
        word = "forcast" if self.source == "predicted" else "residual"  # sic: the reference's
        written: list[Path] = []
        with _pyplot() as plt:
            for datatype, (rows, keys) in series.items():
                # Rows forecast in both iterates (NaN: a role's data not forecast).
                both = [
                    r for r in rows if np.isfinite(initial[r]).all() and np.isfinite(final[r]).all()
                ]
                if not both:
                    continue
                kept = [keys[rows.index(r)] for r in both]
                x = _days([k.time for k in kept], view.start)
                order = np.argsort(x, kind="stable")
                x = x[order]
                chosen = np.asarray(both)[order]
                kind, well = _type_and_well(datatype)
                ylabel = "Bar" if "WBHP" in kind else "Sm3/Day"  # WWCT too, as the reference
                name = f"{well}_{kind}" if well else kind
                stem = output_dir / _file_part(name)
                limits = None
                for which, values in (("initial", initial), ("final", final)):
                    fig = plt.figure()
                    ax = plt.subplot(1, 1, 1)
                    dots = None if observed is None else observed.values[chosen]
                    handles = _bands(ax, x, values[chosen], dots)
                    if which == "initial":
                        ax.legend([(h,) for h, _ in handles], [t for _, t in handles],
                                  loc="upper right", prop={"size": 8}, ncol=2)  # fmt: skip
                        limits = ax.get_ylim()
                    ax.set_ylim(limits)
                    ax.set_title(f"{kind} {which} {word}, at Well: {well}")
                    ax.set_xlabel("Days")
                    ax.set_ylabel(ylabel)
                    if which == "final":
                        # After the labels (the reference's ran before them, losing them).
                        fig.tight_layout(pad=0.5)
                    suffix = "_initial" if which == "initial" else ""
                    written.append(_save(
                        plt, fig, stem.with_name(stem.name + suffix), DEFAULT_DPI * dpi_scale
                    ))  # fmt: skip
        return written


# --------------------------------------------------------------------------- layer maps


def _layers(vector: NDArray[np.float64], grid: GridView) -> NDArray[np.float64]:
    """Return a flat field as ``(nx, ny, nz)`` with inactive cells NaN."""
    field = np.array(vector, dtype=float)
    if grid.actnum is not None:
        field[~grid.actnum] = np.nan
    dims = tuple(grid.dims) + (1,) * (3 - len(grid.dims))
    return field.reshape(dims, order=grid.order)


@figures.register("field_layers")
class FieldLayers:
    """Per gridded state variable and layer: the true, prior, posterior, change and std maps.

    Files ``<variable>_<kind>_layer<L>`` with ``kind`` in ``ref``, ``prior``, ``post``,
    ``diff``, ``std_prior`` and ``std_post``: ``jet`` cell maps over the (i, j) indices, the
    j axis downwards, inactive cells blank. ``ref``, ``prior`` and ``post`` share the colour
    limits 0.95 x min to 1.05 x max of the true layer (of the posterior mean without a
    truth); the others scale on their own. The prior is the first iterate, the posterior the
    final one; means and standard deviations (``ddof=1``) are over the members.

    Parameters
    ----------
    variables
        The variables to draw (default: every one with a grid).
    layers
        1-based layers (default: all).
    kinds
        The maps to draw (default: all six; ``ref`` needs the true model).
    exponentiate
        Variables shown as ``exp`` of their values (ln-permeabilities; default those whose
        name contains ``perm``, as the reference). The means are of the exponentiated members.
    """

    def __init__(
        self,
        variables: TextList = (),
        layers: Sequence[int] = (),
        kinds: TextList = LAYER_KINDS,
        exponentiate: TextList | None = None,
    ) -> None:
        unknown = sorted(set(kinds) - set(LAYER_KINDS))
        if unknown:
            raise ValueError(f"unknown kinds {unknown}: use {list(LAYER_KINDS)}")
        if any(int(layer) < 1 for layer in layers):
            raise ValueError(f"layers are 1-based, got {list(layers)}")
        self.variables = tuple(variables)
        self.layers = tuple(int(layer) for layer in layers)
        self.kinds = tuple(k for k in LAYER_KINDS if k in kinds)
        self.exponentiate = None if exponentiate is None else tuple(exponentiate)

    def _chosen(self, view: RunView) -> list[str]:
        if self.variables:
            return list(self.variables)
        return [name for name in view.variables() if view.grid(name) is not None]

    def applies(self, view: RunView) -> str | None:
        """Why the variables cannot be drawn, or ``None``."""
        chosen = self._chosen(view)
        if not chosen:
            return "the run has no state variable with a grid"
        gridless = [name for name in chosen if view.grid(name) is None]
        if gridless:
            return f"no grid for {gridless}"
        return None

    def _exp(self, name: str) -> bool:
        if self.exponentiate is None:
            return "perm" in name.lower()
        return name in self.exponentiate

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the maps of every chosen variable and layer."""
        truth = view.truth()
        written: list[Path] = []
        for name in self._chosen(view):
            grid = view.grid(name)
            if grid is None:
                continue
            transform = np.exp if self._exp(name) else (lambda a: a)
            prior = transform(np.asarray(view.iterate(0).state[name], dtype=float))
            post = transform(
                np.asarray(view.iterate(view.posterior_iteration).state[name], dtype=float)
            )
            fields: dict[str, NDArray[np.float64]] = {
                "prior": _layers(prior.mean(axis=-1), grid),
                "post": _layers(post.mean(axis=-1), grid),
                "std_prior": _layers(prior.std(axis=-1, ddof=1), grid),
                "std_post": _layers(post.std(axis=-1, ddof=1), grid),
            }
            fields["diff"] = fields["post"] - fields["prior"]
            if truth is not None and name in truth:
                fields["ref"] = _layers(transform(np.asarray(truth[name], dtype=float)), grid)
            nz = fields["prior"].shape[2]
            layers = self.layers or tuple(range(1, nz + 1))
            if max(layers) > nz:
                raise ValueError(f"{name} has {nz} layers, asked for layer {max(layers)}")
            nx, ny = fields["prior"].shape[:2]
            x, y = np.mgrid[0:nx:1, 0:ny:1]
            with _pyplot() as plt:
                for layer in layers:
                    reference = fields.get("ref", fields["post"])[:, :, layer - 1]
                    limits = (np.nanmin(reference) * 0.95, np.nanmax(reference) * 1.05)
                    for kind in self.kinds:
                        if kind not in fields:
                            continue
                        values = fields[kind][:, :, layer - 1]
                        fig = plt.figure()
                        ax = plt.axes()
                        ax.set_facecolor("white")
                        plt.pcolormesh(x, y, values, cmap="jet", shading="auto")
                        plt.gca().invert_yaxis()
                        plt.colorbar()
                        if kind in ("ref", "prior", "post") and all(map(math.isfinite, limits)):
                            plt.clim(*limits)
                        title = _LAYER_TITLES[kind].format(name=name)
                        plt.title(title + (f" at layer {layer}" if nz > 1 else ""))
                        written.append(_save(
                            plt, fig, output_dir / _file_part(f"{name}_{kind}_layer{layer}"),
                            DEFAULT_DPI * dpi_scale, crop=True,
                        ))  # fmt: skip
        return written


# --------------------------------------------------------------------------- histograms


@figures.register("state_histogram")
class StateHistogram:
    """A state variable's values at the initial and the final iterate, one histogram per member.

    As the reference: ``plt.hist`` of the ``(n, n_members)`` array, so each member is a
    dataset of its own on shared bins, with black bar edges (many members look solid
    black). Files ``<prefix>_initial`` and ``<prefix>_final``, titles ``initial <variable>``
    and ``final <variable>``.

    Parameters
    ----------
    variable
        The state variable (e.g. ``nn_params``, a model-error network's parameters).
    bins
        Number of shared bins.
    font_size
        Title, axis label and tick size.
    prefix
        The file prefix.
    """

    def __init__(
        self, variable: Text, bins: int = 50, font_size: float = 16.0, prefix: Text = "hist"
    ) -> None:
        if not 1 <= bins <= MAX_BINS:
            raise ValueError(f"bins must be 1 to {MAX_BINS}, got {bins}")
        if not 0 < font_size <= MAX_FONT_SIZE:
            raise ValueError(f"font_size must be in (0, {MAX_FONT_SIZE:g}], got {font_size}")
        self.variable = variable
        self.bins = bins
        self.font_size = font_size
        self.prefix = _checked_prefix(prefix)

    def applies(self, view: RunView) -> str | None:
        """Why the run lacks the variable, or ``None``."""
        if self.variable not in view.variables():
            return f"the run has no state variable {self.variable!r}"
        return None

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the initial and final histograms."""
        written: list[Path] = []
        with _pyplot() as plt:
            for which, iteration in (("initial", 0), ("final", view.posterior_iteration)):
                values = np.asarray(view.iterate(iteration).state[self.variable], dtype=float)
                fig = plt.figure()
                plt.hist(values, bins=self.bins, edgecolor="black")
                plt.xlabel("Value", fontsize=self.font_size)
                plt.ylabel("Frequency", fontsize=self.font_size)
                plt.title(f"{which} {self.variable}", fontsize=self.font_size)
                plt.xticks(fontsize=self.font_size)
                plt.yticks(fontsize=self.font_size)
                fig.tight_layout()  # the reference had none and lost its y label
                written.append(_save(
                    plt, fig, output_dir / f"{self.prefix}_{which}", DEFAULT_DPI * dpi_scale
                ))  # fmt: skip
        return written


# ------------------------------------------------------------------------------ presets


@figure_presets.register("brugge_classic")
class BruggeClassic:
    """The classic figure set of an ensemble history-matching study, in its folders.

    ``obj/`` (the objective, ``data_mismatch`` over all data), ``data_mismatch/``
    (``data_mismatch`` per data type and well, then ``all``), ``rmse/``, ``mssim/``
    (``ssim``), ``simData/`` (forecast bands), ``residual/`` (auxiliary forecasts, MEC's
    residuals), ``models/`` (layer maps) and ``nn_hist/`` (the ``nn_params`` histogram).
    ``normalized_data_mismatch/`` and ``model_change/`` only when asked for. Every figure is
    requested: one the run cannot give (no truth, no residuals, no ``nn_params``, a metric
    not recorded) is reported as skipped, with the reason.

    Parameters
    ----------
    histogram_variable
        The state variable of ``nn_hist/``.
    normalized
        Also draw ``normalized_data_mismatch`` (the objective and per data type and well).
    model_change
        Also draw ``model_change``.
    """

    def __init__(
        self, histogram_variable: Text = "nn_params", normalized: bool = False,
        model_change: bool = False,
    ) -> None:  # fmt: skip
        # Checked strictly: a model's "false" (text) would otherwise switch a group on.
        if not isinstance(histogram_variable, str):
            raise ValueError(f"histogram_variable must be a variable's name, "
                             f"got {histogram_variable!r}")  # fmt: skip
        for name, value in (("normalized", normalized), ("model_change", model_change)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
        self.histogram_variable = histogram_variable
        self.normalized = normalized
        self.model_change = model_change

    def requests(self, view: RunView) -> list[FigureRequest]:
        """Return the set's figures; the drawing reports those the run cannot give."""
        recorded = view.metric_labels()
        metrics = [("data_mismatch", "data_mismatch", True)]
        wanted: list[FigureRequest] = [FigureRequest("objective_distribution", {}, "obj")]
        if self.normalized:
            wanted.append(FigureRequest(
                "objective_distribution", {"metric": "normalized_data_mismatch"},
                "obj_normalized",
            ))  # fmt: skip
            metrics.append(("normalized_data_mismatch", "normalized_data_mismatch", True))
        metrics += [("rmse", "rmse", True), ("ssim", "mssim", False)]
        if self.model_change:
            metrics.append(("model_change", "model_change", True))
        for metric, name, with_all in metrics:
            options: dict[str, Any] = {"metric": metric, "prefix": name}
            labels = [label for label in recorded.get(metric, []) if label != "all"]
            if with_all and "all" in recorded.get(metric, []):
                labels.append("all")
            if labels:  # else drawn as is, and skipped: "the run recorded no ..."
                options["labels"] = labels
            wanted.append(FigureRequest("metric_distribution", options, name))
        wanted += [
            FigureRequest("data_bands", {"source": "predicted"}, "simData"),
            FigureRequest("data_bands", {"source": "auxiliary"}, "residual"),
            FigureRequest("field_layers", {}, "models"),
            FigureRequest("state_histogram", {"variable": self.histogram_variable}, "nn_hist"),
        ]
        return wanted
