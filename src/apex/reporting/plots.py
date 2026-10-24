"""Figure chrome shared by APEX's report figures (PNG, needs the ``plots`` extra).

The figures themselves are in :mod:`apex.reporting.result_plots`.

Colours follow the validated reference palette (dataviz skill): posterior = categorical
slot 1 (blue), prior = slot 2 (orange), which pass the lightness, chroma, colour-vision
and contrast checks as a pair. Observations are ink markers with error bars, so identity
never depends on colour alone, and every multi-series figure has a legend.
"""

import datetime as dt
from collections.abc import Sequence
from types import ModuleType

POSTERIOR = "#2a78d6"
PRIOR = "#eb6834"
INK = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

#: At most this many data-type panels per data-match figure.
MAX_PANELS = 12


#: Default font size of report figures (points); every figure function takes ``font_size``.
FONT_SIZE = 12.0


def _pyplot(font_size: float = FONT_SIZE) -> ModuleType:
    """Return ``matplotlib.pyplot`` with APEX's chrome, text sized from ``font_size``.

    Titles use ``font_size``; axis labels, ticks and legends are a step smaller.
    """
    # Imported lazily: matplotlib is optional, and the Agg backend needs no display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": INK,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size * 0.9,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": font_size * 0.8,
            "ytick.labelsize": font_size * 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": font_size * 0.8,
        }
    )
    return plt


def _positions(times: Sequence[str]) -> list[dt.date] | list[int]:
    """Report times as dates when they are ISO dates, otherwise as their order."""
    try:
        return [dt.date.fromisoformat(t) for t in times]
    except ValueError:
        return list(range(len(times)))
