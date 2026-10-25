"""Hard data for the geomodeler: read, and estimated from.

The data are a CSV with the columns ``i,j,k,property,value``: 1-based Eclipse cells and a
value there, e.g. well logs upscaled to the cells they cross. ``property`` names a state
variable (``poro``, ``permx``), in the deck's units: a log variable's values (those the
deck template exponentiates, e.g. ``permx`` in mD) are taken ``ln``, the state's space.
The property ``facies`` gives integer facies codes ``0..K-1``.

:func:`estimate_from_data` turns them into the facts the geomodeler holds its choices to:

- per continuous property, the data's mean and standard deviation, and a variogram from a
  ``variogram_estimators`` plugin (``wls`` by default: model, ranges, azimuth, nugget);
- for facies, the proportions (the codes' counts), the latent field's length scale from
  the indicator variograms mapped back through the truncated Gaussian, and the cells as
  conditioning data.

What the data cannot tell (too few, all equal, one column only) is left out with a note.
The continuous properties' estimates carry the data themselves, which the geomodeler's
continuous skills condition their priors to.

:func:`estimate_from_reference` estimates the same statistics and variograms from a
reference field (``--reference``: whole arrays in state space), for priors that share them
without being conditioned to it, and :func:`merge_estimates` combines the two: the
reference's statistics, the hard data's conditioning.
"""

import csv
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.geostatistics.variogram import (
    MAX_RANGE_LAGS,
    experimental_variogram,
    fit_variogram,
    latent_variogram,
)
from apex.core.errors import ConfigError
from apex.core.geomodeling import (
    FaciesEstimate,
    GeomodelFacts,
    PropertyEstimate,
    VariogramEstimate,
)
from apex.core.plugins import variogram_estimators

#: The property naming facies codes.
FACIES = "facies"
COLUMNS = ("i", "j", "k", "property", "value")
#: The fewest data a property needs for its statistics.
MIN_DATA = 3
#: The reference's cells a variogram is estimated from, drawn at random per layer with a
#: seed of the variable's own (the estimator pairs cells within a layer only, so each layer
#: gets its share, at least :data:`REFERENCE_LAYER_MIN`): 3000 cells in one layer are 4.5
#: million pairs, a second or two. The mean and std use every cell.
REFERENCE_SAMPLE = 3000
REFERENCE_LAYER_MIN = 200
REFERENCE_SEED = 20261025
#: Lag classes of the facies' latent variograms (as ``wls``'s default).
LATENT_LAGS = 20
#: An APEX ``gaussian`` variogram's practical range ``r`` is GSTools' Gaussian length
#: scale ``l`` times ``sqrt(12 / pi)``: ``exp(-3 (h / r)^2) = exp(-pi / 4 (h / l)^2)``.
GAUSSIAN_RANGE_PER_LEN_SCALE = math.sqrt(12.0 / math.pi)

HardData = dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]]


def load_hard_data(path: Path) -> HardData:
    """Return the data by property: ``(cells (n, 3), values (n,))`` (see the module).

    Raises
    ------
    ConfigError
        If the file cannot be read, lacks a column, or has a row that is not a cell and a
        number.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ConfigError(f"{path.name}: no column {', '.join(missing)} (columns: "
                                  f"{', '.join(COLUMNS)})")  # fmt: skip
            rows: dict[str, list[tuple[int, int, int, float]]] = defaultdict(list)
            seen: dict[str, set[tuple[int, int, int]]] = {}
            for number, row in enumerate(reader, start=2):
                try:
                    cell = (int(row["i"]), int(row["j"]), int(row["k"]))
                    value = float(row["value"])
                except (TypeError, ValueError):
                    raise ConfigError(f"{path.name}, line {number}: i, j, k must be integers "
                                      "and value a number") from None  # fmt: skip
                if min(cell) < 1 or not math.isfinite(value):
                    raise ConfigError(f"{path.name}, line {number}: cells are 1-based and "
                                      "values finite")  # fmt: skip
                name = str(row["property"]).strip()
                if cell in seen.setdefault(name, set()):
                    raise ConfigError(f"{path.name}, line {number}: cell {list(cell)} has a "
                                      f"second {name} value: give one a cell (e.g. the logs' "
                                      "average over it)")  # fmt: skip
                seen[name].add(cell)
                rows[name].append((*cell, value))
    except OSError as error:
        raise ConfigError(f"cannot read {path.name}: {error}") from None
    if not rows:
        raise ConfigError(f"{path.name} has no data")
    return {name: (np.array([r[:3] for r in found], dtype=np.int64),
                   np.array([r[3] for r in found], dtype=np.float64))
            for name, found in rows.items()}  # fmt: skip


def estimate_from_data(
    data: HardData, facts: GeomodelFacts, estimator: str = "wls"
) -> tuple[dict[str, PropertyEstimate], FaciesEstimate | None, list[str]]:
    """Return the continuous properties' and the facies' estimates, and notes.

    Properties that are no state variable (nor ``facies``) are noted and left out; cells
    outside a variable's grid are refused.

    Raises
    ------
    ConfigError
        If a cell lies outside its variable's grid, or a facies code is not an integer.
    """
    notes: list[str] = []
    found: dict[str, PropertyEstimate] = {}
    facies: FaciesEstimate | None = None
    for name, (cells, values) in sorted(data.items()):
        if name == FACIES:
            facies = _facies(cells, values, facts)
            continue
        fact = facts.variables.get(name)
        if fact is None:
            notes.append(f"hard data of {name!r} left out: no such state variable")
            continue
        if fact.grid and np.any(cells > np.asarray(fact.grid)):
            raise ConfigError(f"hard data of {name}: a cell is outside its grid {fact.grid}")
        if name in facts.log_variables:
            if np.any(values <= 0):
                raise ConfigError(f"hard data of {name}: the deck takes exp({name}), so its "
                                  "values must be positive (e.g. mD)")  # fmt: skip
            values = np.log(values)
        found[name] = _continuous(name, cells, values, estimator)
    return found, facies, notes


def estimate_from_reference(
    path: Path, facts: GeomodelFacts, estimator: str = "wls"
) -> tuple[dict[str, PropertyEstimate], list[str]]:
    """Return each gridded variable's statistics and variogram from a reference field.

    ``path`` is an ``.npz`` of arrays in state space, one per variable (as ``--reference``);
    cells that are 0 or not finite (inactive) are left out. The variogram is estimated
    from :data:`REFERENCE_SAMPLE` cells drawn with a fixed seed, so the same reference
    gives the same estimate; the mean and std from every cell. Nothing is conditioned to
    the reference: the estimates carry no cells. Inactive cells are left out: non-finite
    values, and zeros of variables that are not log variables (a log variable's 0 is
    ln 1, a value; an inactive cell of it is ln 0, not finite).

    Raises
    ------
    ConfigError
        If the file cannot be read.
    """
    try:
        with np.load(path) as data:
            arrays = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot read the reference {path.name}: {error}") from None
    found: dict[str, PropertyEstimate] = {}
    notes: list[str] = []
    for name, fact in sorted(facts.variables.items()):
        array = arrays.get(name)
        if array is None or not fact.grid or len(fact.grid) != 3:
            continue
        values = array.ravel()
        if values.size != math.prod(fact.grid):
            notes.append(f"reference {name}: {values.size} values for a grid of "
                         f"{math.prod(fact.grid)} cells, left out")  # fmt: skip
            continue
        nx, ny, nz = fact.grid
        active = np.flatnonzero(active_cells(values, name in facts.log_variables))
        if active.size < MIN_DATA:
            continue
        # A variable's own stream: adding or removing another leaves its sample alone.
        rng = np.random.default_rng([REFERENCE_SEED, zlib.crc32(name.encode())])
        share = max(REFERENCE_LAYER_MIN, REFERENCE_SAMPLE // nz)
        layer_of = active // (nx * ny)
        chosen = np.sort(np.concatenate([
            rng.choice(here, size=min(share, here.size), replace=False)
            for here in (active[layer_of == k] for k in range(nz)) if here.size
        ]))  # fmt: skip
        k, rest = np.divmod(chosen, nx * ny)
        j, i = np.divmod(rest, nx)
        cells = np.column_stack([i + 1, j + 1, k + 1]).astype(np.int64)
        estimate = _continuous(name, cells, values[chosen], estimator)
        # The statistics of every active cell; the sample only for the variogram.
        every = values[active]
        sampled = f"{chosen.size} of {active.size} cells" if chosen.size < active.size else ""
        found[name] = estimate.model_copy(update={
            "n_data": int(active.size), "mean": float(every.mean()), "std": float(every.std()),
            "source": "reference", "cells": [], "values": [],
            "notes": [*estimate.notes, *([f"variogram from {sampled}"] if sampled else [])],
        })  # fmt: skip
    return found, notes


def active_cells(values: NDArray[np.float64], log_variable: bool) -> NDArray[np.bool_]:
    """Return which reference values are the field's (see :func:`estimate_from_reference`)."""
    finite = np.isfinite(values)
    return finite if log_variable else finite & (values != 0)


def merge_estimates(
    reference: dict[str, PropertyEstimate], data: dict[str, PropertyEstimate]
) -> dict[str, PropertyEstimate]:
    """Return the estimates of both: the reference's statistics, the data's conditioning.

    A whole field tells the statistics and variogram better than a few wells, so where a
    variable has both, its estimate is the reference's, conditioned to the hard data.
    """
    merged = dict(reference)
    for name, estimate in data.items():
        found = reference.get(name)
        if found is None:
            merged[name] = estimate
            continue
        merged[name] = found.model_copy(update={
            "source": "reference+data", "cells": estimate.cells, "values": estimate.values,
            "notes": [*found.notes, f"conditioned to {len(estimate.values)} hard data"],
        })  # fmt: skip
    return merged


def _fit(
    estimator: str, cells: NDArray[np.int64], values: NDArray[np.float64], **options: object
) -> tuple[VariogramEstimate | None, str | None]:
    try:
        fitted = variogram_estimators.create(estimator).estimate(cells, values, **options)  # type: ignore[arg-type]
    except ValueError as error:
        return None, str(error)
    return fitted, None


def _continuous(
    name: str, cells: NDArray[np.int64], values: NDArray[np.float64], estimator: str
) -> PropertyEstimate:
    notes = []
    data = {"cells": cells.tolist(), "values": values.tolist()}
    if values.size < MIN_DATA:
        return PropertyEstimate(variable=name, n_data=int(values.size), mean=float(values.mean()),
                                std=float(values.std()),
                                notes=[f"{values.size} data: no variogram"], **data)  # fmt: skip
    fitted, why = _fit(estimator, cells, values)
    if fitted is None:
        notes.append(f"no variogram: {why}")
        return PropertyEstimate(variable=name, n_data=int(values.size), mean=float(values.mean()),
                                std=float(values.std()), notes=notes, **data)  # fmt: skip
    total = fitted.nugget + fitted.sill
    nugget = fitted.nugget / total if total > 0 else None
    if nugget is not None and nugget > 0.3:
        notes.append(f"nugget {nugget:.0%} of the sill: the generators have none, so the "
                     "prior is smoother than the data")  # fmt: skip
    return PropertyEstimate(
        variable=name, n_data=fitted.n_data, mean=fitted.mean, std=math.sqrt(fitted.variance),
        variogram=fitted.model, major_range=fitted.major_range, minor_range=fitted.minor_range,
        azimuth=fitted.azimuth, nugget_fraction=nugget, notes=[*notes, *fitted.notes], **data,
    )  # fmt: skip


def _facies(
    cells: NDArray[np.int64], values: NDArray[np.float64], facts: GeomodelFacts
) -> FaciesEstimate:
    """Return the facies' proportions, latent length scale and conditioning data.

    The length scale is the latent Gaussian field's, for ``facies_pluri``'s rule (facies
    ``k`` where the field lies between the ``k``-th thresholds): each boundary's indicator
    ``code < k`` is a thresholded Gaussian, its variogram is mapped back to the field's
    (:func:`~apex.algorithms.geostatistics.variogram.latent_variogram`), and a gaussian
    model fitted to that. The median over the boundaries whose fit is bounded by the data.
    """
    codes = np.rint(values).astype(np.int64)
    if np.any(np.abs(values - codes) > 1e-9) or np.any(codes < 0):
        raise ConfigError("hard data of facies: codes must be integers 0, 1, ...")
    grids = {tuple(v.grid) for v in facts.variables.values() if v.grid}
    if len(grids) == 1 and np.any(cells > np.asarray(next(iter(grids)))):
        raise ConfigError(f"hard data of facies: a cell is outside the grid {list(grids.pop())}")
    present = sorted(int(c) for c in np.unique(codes))
    proportions = [float(np.mean(codes == c)) for c in present]
    notes = []
    if present != list(range(len(present))):
        notes.append(f"facies codes {present}: some code below the largest is never seen")
    xy = cells[:, :2].astype(np.float64)
    max_lag = float(np.hypot(*np.ptp(xy, axis=0))) / 2
    ranges: dict[int, float] = {}
    if max_lag > 0:
        edges = np.linspace(0.0, max_lag, LATENT_LAGS + 1)
        top = MAX_RANGE_LAGS * max_lag
        for k in present[1:]:
            below = (codes < k).astype(np.float64)
            indicator = experimental_variogram(xy, below, edges, layers=cells[:, 2])
            try:
                fitted = fit_variogram([latent_variogram(indicator, float(below.mean()))],
                                       "gaussian", max_range=top)  # fmt: skip
            except ValueError as error:
                notes.append(f"boundary below facies {k}: no variogram ({error})")
                continue
            if fitted.major_range > 0.99 * top:
                notes.append(f"boundary below facies {k}: the range reached its bound "
                             f"({top:.3g} cells): left out")  # fmt: skip
                continue
            ranges[k] = fitted.major_range
    len_scale = None
    if ranges:
        len_scale = float(np.median(list(ranges.values()))) / GAUSSIAN_RANGE_PER_LEN_SCALE
    else:
        notes.append("no length scale: too few data, or none bounded")
    return FaciesEstimate(
        n_data=int(values.size), codes=present, proportions=proportions,
        indicator_ranges=ranges, len_scale=len_scale,
        conditioning_cells=cells.tolist(), conditioning_facies=codes.tolist(), notes=notes,
    )  # fmt: skip
