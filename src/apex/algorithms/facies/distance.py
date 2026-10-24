"""Facies as signed distances: the representation and ``facies_distance``.

A categorical realisation with ``K`` facies becomes ``K`` continuous fields, the level-set
representation of Chang, Zhang & Lu (2010) and Lorentzen, Flornes & Nævdal (2012):

    d_k(x) =  dist(x, cells not of facies k)    if x is of facies k   (> 0 inside)
    d_k(x) = -dist(x, cells of facies k)        otherwise             (< 0 outside)

with Euclidean distances between cell centres (scaled by the cell size), per layer (each
``k`` plane of an Eclipse grid separately, as facies bodies are mapped in plan view).
``argmax_k d_k(x)`` recovers the realisation exactly: a cell's own facies has
``d >= min cell size > 0`` and every other ``d <= -min cell size < 0``. The methods update
the distance fields; ``facies_distance`` maps them back to facies, and to properties by
facies, before every forecast. Any categorical prior can feed it (SNESIM,
``mpslib_snesim``; stored facies).

References
----------
Chang, H., Zhang, D. & Lu, Z. (2010). History matching of facies distribution with the
EnKF and level set technique. Journal of Petroleum Science and Engineering, 73, 7-21.
Lorentzen, R. J., Flornes, K. M. & Nævdal, G. (2012). History matching channelized
reservoirs using the ensemble Kalman filter. SPE Journal, 17(1), 137-151.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from apex.core.options import TEXT
from apex.core.plugins import parameterizations


def signed_distances(
    codes: NDArray[np.integer],
    grid: Sequence[int],
    n_facies: int,
    cell_size: Sequence[float] = (1.0, 1.0, 1.0),
    band: float | None = None,
) -> NDArray[np.float64]:
    """Return each facies' signed distance field ``(n_facies, n_cells)`` (module docstring).

    Parameters
    ----------
    codes
        Facies codes ``(n_cells,)`` in ``0..n_facies-1``, Eclipse order (``i`` fastest).
    grid
        ``[nx, ny, nz]``.
    n_facies
        The number of facies ``K``.
    cell_size
        ``[dx, dy, dz]``; distances are in these units (per layer, so ``dz`` is unused).
    band
        Clip the distances to ``[-band, band]`` (a narrow band, as level-set methods use):
        far from a boundary the value only says "inside" or "outside". ``argmax`` still
        recovers the facies exactly (only a cell's own facies is positive).

    Raises
    ------
    ValueError
        If the codes do not fit the grid or the facies, or ``dx``, ``dy`` or ``band`` is
        not positive.

    Notes
    -----
    A facies absent from a layer has ``d = -D`` there, and one that fills a layer
    ``d = +D``, with ``D`` the layer's diagonal (or ``band``): no boundary is nearer. In
    an ensemble, a member missing a facies in a layer then sits at ``-D`` while the others
    are tens of cells away: a minor facies in small layers can let that member dominate the
    anomalies (the numerics review of 2026-10-03), which a ``band`` bounds.
    """
    nx, ny, nz = (int(n) for n in grid)
    codes = np.asarray(codes)
    if codes.shape != (nx * ny * nz,):
        raise ValueError(f"codes must have {nx * ny * nz} cells, got {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= n_facies):
        raise ValueError(f"codes must be in 0..{n_facies - 1}")
    dx, dy = float(cell_size[0]), float(cell_size[1])
    if dx <= 0 or dy <= 0:
        raise ValueError(f"cell_size must be positive in x and y, got {cell_size}")
    if band is not None and band <= 0:
        raise ValueError(f"band must be positive, got {band}")
    far = math.hypot(nx * dx, ny * dy) if band is None else float(band)
    volume = codes.reshape((nx, ny, nz), order="F")
    result = np.empty((n_facies, nx, ny, nz))
    for k in range(nz):
        layer = volume[:, :, k]
        for facies in range(n_facies):
            inside = layer == facies
            if inside.all():
                result[facies, :, :, k] = far
            elif not inside.any():
                result[facies, :, :, k] = -far
            else:
                # EDT gives each True cell's distance to the nearest False one.
                into = distance_transform_edt(inside, sampling=(dx, dy))
                out = distance_transform_edt(~inside, sampling=(dx, dy))
                result[facies, :, :, k] = np.asarray(into) - np.asarray(out)
    if band is not None:
        np.clip(result, -band, band, out=result)
    return np.stack([result[f].ravel(order="F") for f in range(n_facies)])


def facies_from_distances(distances: NDArray[np.float64]) -> NDArray[np.int64]:
    """Return each cell's facies ``(n_cells,)``: the largest of ``(n_facies, n_cells)``."""
    return np.asarray(np.argmax(distances, axis=0), dtype=np.int64)


class FaciesProperties:
    """Properties by facies: constants per facies, or per-facies state variables.

    Parameters
    ----------
    facies
        Facies names; code ``k`` is ``facies[k]``.
    values
        ``{facies: {property: value}}``: constants.
    property_fields
        ``{property: {facies: state variable}}``: cell by cell from that facies' variable.
    """

    def __init__(
        self,
        facies: Sequence[str],
        values: Mapping[str, Mapping[str, float]] | None,
        property_fields: Mapping[str, Mapping[str, str]] | None,
    ) -> None:
        self.facies = tuple(facies)
        values = {k: dict(v) for k, v in (values or {}).items()}
        property_fields = {k: dict(v) for k, v in (property_fields or {}).items()}
        unknown = sorted(set(values) - set(self.facies))
        if unknown:
            raise ValueError(f"values names unknown facies: {', '.join(unknown)}")
        constant = sorted({p for row in values.values() for p in row})
        both = sorted(set(constant) & set(property_fields))
        if both:
            raise ValueError(f"properties {', '.join(both)} are in values and property_fields")
        for name in constant:
            missing = [f for f in self.facies if name not in values.get(f, {})]
            if missing:
                raise ValueError(f"property {name!r} has no value for facies {', '.join(missing)}")
        for name, by_facies in property_fields.items():
            if set(by_facies) != set(self.facies):
                raise ValueError(f"property_fields {name!r} must name a variable for every facies")
        self.values = values
        self.property_fields = property_fields
        self.variables = tuple(
            sorted({v for row in property_fields.values() for v in row.values()})
        )
        self.names: tuple[str, ...] = (*constant, *sorted(property_fields))

    def of(
        self, codes: NDArray[np.int64], state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{property: (n_cells,)}`` for the facies ``codes`` ``(n_cells,)``."""
        result: dict[str, NDArray[np.float64]] = {}
        for name in self.names:
            if name in self.property_fields:
                stacked = np.stack(
                    [
                        np.asarray(state[self.property_fields[name][f]], dtype=np.float64)
                        for f in self.facies
                    ]
                )
                if stacked.shape[1] != codes.size:
                    raise ValueError(f"property_fields of {name!r} must have {codes.size} cells")
                result[name] = stacked[codes, np.arange(codes.size)]
            else:
                table = np.array([self.values[f][name] for f in self.facies], dtype=np.float64)
                result[name] = table[codes]
        return result


@parameterizations.register("facies_distance")
class FaciesDistance:
    """Facies from signed distance fields (``argmax``), and properties by facies.

    Parameters
    ----------
    fields
        One distance state variable per facies, in code order (``fields[k]`` is facies
        ``facies[k]``'s).
    facies
        Facies names.
    values, property_fields
        Properties by facies, as for ``pgs_truncation``.

    Outputs: ``facies`` (the codes, as floats) and every property.
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {}
    #: Outputs that are codes, not quantities: scored by ``facies_mismatch``, not RMSE.
    categorical: tuple[str, ...] = ("facies",)

    def __init__(
        self,
        fields: Annotated[Sequence[str], TEXT],
        facies: Annotated[Sequence[str], TEXT],
        values: Annotated[Mapping[str, Mapping[str, float]] | None, TEXT] = None,
        property_fields: Annotated[Mapping[str, Mapping[str, str]] | None, TEXT] = None,
    ) -> None:
        if len(fields) != len(facies) or len(facies) < 2:
            raise ValueError(
                f"facies_distance needs one field per facies (at least 2); got {len(fields)} "
                f"fields for {len(facies)} facies"
            )
        self.fields = tuple(fields)
        self.mapping = FaciesProperties(facies, values, property_fields)
        self.variables: tuple[str, ...] = (*self.fields, *self.mapping.variables)
        self.outputs: tuple[str, ...] = ("facies", *self.mapping.names)

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{"facies": codes, property: values}`` for one member, ``(n_cells,)`` each.

        Raises
        ------
        ValueError
            If a variable is missing or the sizes differ.
        """
        missing = [v for v in self.variables if v not in state]
        if missing:
            raise ValueError(f"facies_distance needs {', '.join(missing)}")
        distances = np.stack([np.asarray(state[f], dtype=np.float64) for f in self.fields])
        codes = facies_from_distances(distances)
        return {"facies": codes.astype(np.float64), **self.mapping.of(codes, state)}
