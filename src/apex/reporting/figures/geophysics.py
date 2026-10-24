"""Maps of geophysical field data (figure plugins).

``field_maps`` draws, for each field datum of a run's ``[geophysics]`` and each of its
vintages, the observed map, the prior and posterior ensemble means of the forecast, and the
posterior mean's mismatch (posterior mean minus observed): AVO maps over the grid's
``(i, j)`` columns in the Brugge style of :mod:`apex.reporting.figures.classic` (``jet``,
the j axis downwards, a vertical colour bar, untraced columns blank), gravity and seafloor
displacement as coloured stations at their ``(x, y)``. One figure per file, PNG and PDF.

Whole traces and data compressed by a transform have no map (their analysis-space values
are samples or wavelet coefficients), and are left out with the reason.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.figures import FieldLayout, FieldLayoutError, RunView
from apex.core.options import TextList
from apex.core.plugins import figures
from apex.reporting.figures import classic
from apex.reporting.figures.classic import DEFAULT_DPI, _file_part, _pyplot

#: The maps of each datum and vintage, in drawing order.
MAP_KINDS = ("obs", "prior", "post", "mismatch")
_TITLES = {
    "obs": "{name} observed at {time}",
    "prior": "{name} prior mean at {time}",
    "post": "{name} posterior mean at {time}",
    "mismatch": "{name} posterior mean minus observed at {time}",
}


def _fields(view: RunView) -> dict[str, list[str]]:
    """Return each field datatype the run observed and its times, in key order."""
    out: dict[str, list[str]] = {}
    for key in view.observations().keys:
        if key.index is not None and key.time not in out.setdefault(key.datatype, []):
            out[key.datatype].append(key.time)
    return out


@figures.register("field_maps")
class FieldMaps:
    """Per geophysical field datum and vintage: observed, prior, posterior and mismatch maps.

    Files ``<datatype>_<time>_<kind>`` with ``kind`` in ``obs``, ``prior``, ``post`` and
    ``mismatch``. ``obs``, ``prior`` and ``post`` share the observed map's range as colour
    limits (a forecast far from the data saturates them, so its difference shows);
    ``mismatch`` is symmetric about zero. Column maps have the j axis downwards, station
    maps y upwards. The prior is the first iterate, the posterior
    the run's posterior (the minimum-CV iterate with cross validation).

    Parameters
    ----------
    datatypes
        The field data to draw (default: every one with a map).
    kinds
        The maps to draw (default: all four).
    """

    def __init__(self, datatypes: TextList = (), kinds: TextList = MAP_KINDS) -> None:
        unknown = sorted(set(kinds) - set(MAP_KINDS))
        if unknown:
            raise ValueError(f"unknown kinds {unknown}: use {list(MAP_KINDS)}")
        self.datatypes = tuple(datatypes)
        self.kinds = tuple(k for k in MAP_KINDS if k in kinds)

    def _mapped(self, view: RunView) -> dict[str, tuple[FieldLayout, list[str]]]:
        fields = _fields(view)
        chosen = self.datatypes or tuple(fields)
        # A view written may lack field_layout: no maps, then.
        layout_of = getattr(view, "field_layout", None)
        out = {}
        for name in chosen:
            layout = layout_of(name) if layout_of is not None and name in fields else None
            if layout is not None:
                out[name] = (layout, fields[name])
        return out

    def applies(self, view: RunView) -> str | None:
        """Why the maps cannot be drawn, or ``None``."""
        fields = _fields(view)
        if not fields:
            return "the run has no field data"
        try:
            mapped = self._mapped(view)
        except FieldLayoutError as error:
            return str(error)
        unmapped = [name for name in self.datatypes if name not in mapped]
        if unmapped:
            # Named, so not drawing them must be said (a whole trace, a compressed field,
            # or no such datum).
            return f"no map of {', '.join(unmapped)} (AVO maps and station data have one)"
        if not mapped:
            return (
                "no field datum has a map (geophysical AVO maps and station data have; "
                "whole traces and compressed fields do not)"
            )
        return None

    def draw(self, view: RunView, output_dir: Path, *, dpi_scale: float = 1.0) -> list[Path]:
        """Write the maps of every mapped field datum and vintage."""
        observations = view.observations()
        keys = observations.keys
        prior = view.iterate(0).predicted
        post = view.iterate(view.posterior_iteration).predicted
        written: list[Path] = []
        with _pyplot() as plt:
            for name, (layout, times) in self._mapped(view).items():
                for time in times:
                    rows = sorted(
                        (k.index, n)
                        for n, k in enumerate(keys)
                        if k.datatype == name and k.time == time and k.index is not None
                    )
                    index = np.array([i for i, _ in rows], dtype=np.intp)
                    order = [n for _, n in rows]
                    # A value no member forecast (all NaN) stays blank, without a warning.
                    with np.errstate(all="ignore"), warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        maps = {
                            "obs": np.asarray(observations.values, dtype=float)[order],
                            "prior": np.nanmean(prior[order], axis=-1),
                            "post": np.nanmean(post[order], axis=-1),
                        }
                        maps["mismatch"] = maps["post"] - maps["obs"]
                        finite = maps["obs"][np.isfinite(maps["obs"])]
                        limits = (float(finite.min()), float(finite.max())) if finite.size else None
                        mismatch = np.abs(maps["mismatch"])
                        span = float(np.nanmax(mismatch)) if np.isfinite(mismatch).any() else 0.0
                    for kind in self.kinds:
                        fig = plt.figure()
                        ax = plt.axes()
                        ax.set_facecolor("white")
                        _plot(plt, ax, layout, index, maps[kind])
                        plt.colorbar()
                        if kind == "mismatch" and span > 0:
                            plt.clim(-span, span)
                        elif kind != "mismatch" and limits is not None and limits[1] > limits[0]:
                            plt.clim(*limits)
                        plt.title(_TITLES[kind].format(name=name, time=time))
                        stem = output_dir / _file_part(f"{name}_{time}_{kind}")
                        written.append(
                            classic._save(plt, fig, stem, DEFAULT_DPI * dpi_scale, crop=True)
                        )
        return written


def _plot(
    plt: Any,  # noqa: ANN401 (matplotlib's pyplot module)
    ax: Any,  # noqa: ANN401
    layout: FieldLayout,
    index: NDArray[np.intp],
    values: NDArray[np.float64],
) -> None:
    """Draw ``values`` (the field's components ``index``) where they sit."""
    positions = layout.positions[index]
    if layout.kind == "stations":
        plt.scatter(positions[:, 0], positions[:, 1], c=values, cmap="jet", s=36)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        return
    assert layout.lateral is not None
    # On the lattice of kept columns (every step_x-th and step_y-th), so that a thinned
    # survey's columns fill the map; the axes are still the grid's column indices.
    (nx, ny), (sx, sy) = layout.lateral, layout.steps
    lattice = positions.astype(np.intp) // np.array([sx, sy])
    grid = np.full((-(-nx // sx), -(-ny // sy)), np.nan)
    grid[lattice[:, 0], lattice[:, 1]] = values
    x, y = np.meshgrid(np.arange(grid.shape[0]) * sx, np.arange(grid.shape[1]) * sy, indexing="ij")
    plt.pcolormesh(x, y, grid, cmap="jet", shading="auto")
    plt.gca().invert_yaxis()
