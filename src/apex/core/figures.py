"""Figure plugins and presets, and the view of a run they draw.

A :class:`Figure` draws one kind of figure from a :class:`RunView` and writes each as PNG
and PDF twins. A :class:`FigurePreset` is a named list of figure requests, each with its
options and subfolder. Both are plugins (:data:`apex.core.plugins.figures`,
:data:`apex.core.plugins.figure_presets`), chosen by name.

``reporting`` may import only ``domain`` and ``core``, so the plugins read a run through
:class:`RunView`, which ``services`` implements over the stored results. Arrays follow APEX's
convention: the ensemble axis is the last, ``(n, n_members)``.
"""

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import DataKey


@dataclass(frozen=True, slots=True)
class IterateView:
    """One saved ensemble and its forecasts.

    Attributes
    ----------
    iteration
        The run's iteration number (0 = prior; rejected trials count too).
    state
        ``{variable: (size, n_members)}``.
    predicted
        The forecast of every observation of :meth:`RunView.observations`, in the analysis
        data space, ``(n_data, n_members)``; NaN where none was forecast.
    member_ids
        Member id of each column.
    auxiliary_predicted
        Forecasts that were never assimilated (e.g. a model-error correction's residuals),
        ``(n_aux, n_members)``, or ``None``.
    auxiliary_keys
        Their keys, one per row of ``auxiliary_predicted``.
    """

    iteration: int
    state: Mapping[str, NDArray[np.float64]]
    predicted: NDArray[np.float64]
    member_ids: NDArray[np.int64]
    auxiliary_predicted: NDArray[np.float64] | None = None
    auxiliary_keys: tuple[DataKey, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservationsView:
    """A run's observations in its analysis data space: keys, values and error variances."""

    keys: tuple[DataKey, ...]
    values: NDArray[np.float64]
    variances: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GridView:
    """A state variable's grid: dimensions, cell order and active cells.

    Attributes
    ----------
    dims
        ``(nx, ny, nz)`` (1-3 dimensions).
    order
        How the flat vector maps onto the grid (``"F"``: Eclipse's, i fastest).
    actnum
        Active cells of the flat vector, ``(size,)`` booleans, or ``None`` (all active).
    """

    dims: tuple[int, ...]
    order: Literal["F", "C"] = "F"
    actnum: NDArray[np.bool_] | None = None


@dataclass(frozen=True, slots=True)
class FieldLayout:
    """Where the values of a field datum sit (geophysics).

    Attributes
    ----------
    kind
        ``"columns"``: one value per grid column (AVO maps), ``positions`` its ``(i, j)``,
        0-based, on a grid of ``lateral`` = ``(nx, ny)`` columns; ``"stations"``: one value
        per station (gravity, displacement), ``positions`` its ``(x, y)`` in m.
    positions
        ``(n_values, 2)``, in the field's order.
    lateral
        ``(nx, ny)`` for columns, else ``None``.
    steps
        The columns' spacing ``(step_x, step_y)``: the map is drawn on their lattice.
    """

    kind: Literal["columns", "stations"]
    positions: NDArray[np.float64]
    lateral: tuple[int, int] | None = None
    steps: tuple[int, int] = (1, 1)


class FieldLayoutError(Exception):
    """A run's field layouts cannot be given: its geophysical model could not be rebuilt."""


class RunView(Protocol):
    """What a figure may read of a history-matching run."""

    @property
    def path(self) -> Path:
        """The run directory."""
        ...

    @property
    def iterations(self) -> list[int]:
        """Every saved iteration, in order."""
        ...

    @property
    def accepted_iterations(self) -> list[int]:
        """The accepted iterates (the prior included), in order."""
        ...

    @property
    def final_iteration(self) -> int:
        """The last accepted iterate."""
        ...

    @property
    def posterior_iteration(self) -> int:
        """The posterior: with cross validation the minimum-CV iterate, else the last."""
        ...

    @property
    def start(self) -> dt.date | None:
        """The date days are counted from (the deck's ``START``), or ``None`` if unknown."""
        ...

    def iterate(self, iteration: int) -> IterateView:
        """Return a saved iterate.

        Raises
        ------
        KeyError
            If it was not saved.
        """
        ...

    def metric_labels(self) -> dict[str, list[str]]:
        """Return every recorded metric and its labels (``"all"`` first)."""
        ...

    def metric(
        self, name: str, label: str = "all", *, accepted_only: bool = False
    ) -> tuple[list[int], list[NDArray[np.float64]], list[bool]]:
        """Return a metric's iterations, per-member values and whether each was accepted.

        The values are one ``(n_members,)`` array per iteration. Leading iterations where
        the metric is undefined (the prior's ``model_change``) are left out.

        Raises
        ------
        KeyError
            If the metric or label was not recorded.
        """
        ...

    def truth(self) -> Mapping[str, NDArray[np.float64]] | None:
        """Return the true model (synthetic cases), ``{variable: (size,)}``, or ``None``."""
        ...

    def observations(self) -> ObservationsView:
        """Return the observations in the analysis data space."""
        ...

    def variables(self) -> list[str]:
        """Return the state variables, in the state's order."""
        ...

    def grid(self, variable: str) -> GridView | None:
        """Return a state variable's grid, or ``None`` if it has none."""
        ...

    def field_layout(self, datatype: str) -> FieldLayout | None:
        """Return where a field datum's values sit, or ``None`` if they have no map.

        Geophysical maps and station data have one; whole traces, compressed
        fields and fields of other origins do not.

        Raises
        ------
        FieldLayoutError
            If the run's geophysical model cannot be rebuilt (its grid file moved, say).
        """
        ...


class Figure(Protocol):
    """Draws one kind of figure from a run."""

    def applies(self, view: RunView) -> str | None:
        """Return why the figure cannot be drawn for this run, or ``None`` if it can."""
        ...

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the figures into ``output_dir`` (PNG and PDF) and return the PNG paths.

        ``dpi_scale`` multiplies the plugin's resolution (1: its own). The caller makes
        ``output_dir``'s parent; the plugin makes ``output_dir`` only, so a folder removed
        meanwhile (a deleted run) is not recreated.
        """
        ...


@dataclass(frozen=True, slots=True)
class FigureRequest:
    """One figure plugin to draw, with its options, into a subfolder of the output."""

    figure: str
    options: Mapping[str, object] = field(default_factory=dict)
    subdir: str = ""


class FigurePreset(Protocol):
    """A named set of figure requests."""

    def requests(self, view: RunView) -> list[FigureRequest]:
        """Return the figures to draw for this run (those that apply to it)."""
        ...
