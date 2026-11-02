"""Property fields averaged onto a coarser grid: the parameterization ``block_average``.

A fine grid ``(nx, ny, nz)`` in Eclipse order (``i`` fastest) is split into blocks of
``factors = (fx, fy, fz)`` cells (the last block along an axis takes what is left when the
factor does not divide it), each a coarse cell, in Eclipse order of the coarse grid
``(ceil(nx / fx), ceil(ny / fy), ceil(nz / fz))``. Each variable is averaged over its
block's cells by its own mean:

- ``arithmetic``: porosity, net-to-gross, and ``ln k`` (whose arithmetic mean is ``k``'s
  geometric mean, the usual estimate of a random field's effective permeability);
- ``geometric``: a positive variable such as ``k`` itself;
- ``harmonic``: a positive variable across which flow is in series (``k`` normal to layers).

Unweighted: every fine cell counts alike (the grid's cells are taken as equal in volume;
for unequal cells, average in the deck), inactive cells (``ACTNUM`` 0) included. Used for a
coarse simulation of a fine state, e.g. the coarse runs (the user chose core and
public, 2026-11-02). Its outputs live on the coarse grid (``output_grid``): metrics sized for
the fine grid (``rmse``'s ``actnum``, ``ssim``'s ``grid``) do not apply to them.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray

from apex.core.options import TEXT
from apex.core.plugins import parameterizations

Mean = Literal["arithmetic", "geometric", "harmonic"]


def block_index(dims: Sequence[int], factors: Sequence[int]) -> tuple[NDArray[np.intp], int]:
    """Return each fine cell's coarse cell ``(nx ny nz,)`` (Eclipse order) and their count."""
    nx, ny, nz = (int(d) for d in dims)
    fx, fy, fz = (int(f) for f in factors)
    cx, cy, cz = -(-nx // fx), -(-ny // fy), -(-nz // fz)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    coarse = (i // fx) + cx * ((j // fy) + cy * (k // fz))
    return np.ascontiguousarray(coarse.reshape(-1).astype(np.intp)), cx * cy * cz


def block_average(
    values: NDArray[np.float64], index: NDArray[np.intp], n_coarse: int, mean: Mean
) -> NDArray[np.float64]:
    """Return ``values`` ``(n_fine,)`` averaged per coarse cell, ``(n_coarse,)``.

    Raises
    ------
    ValueError
        If a geometric or harmonic mean meets a value that is not positive.
    """
    x = np.asarray(values, dtype=np.float64)
    counts = np.bincount(index, minlength=n_coarse).astype(np.float64)
    if mean == "arithmetic":
        return np.bincount(index, weights=x, minlength=n_coarse) / counts
    if not np.all(x > 0):
        raise ValueError(f"a {mean} mean needs positive values")
    if mean == "geometric":
        return np.exp(np.bincount(index, weights=np.log(x), minlength=n_coarse) / counts)
    return counts / np.bincount(index, weights=1.0 / x, minlength=n_coarse)


@parameterizations.register("block_average")
class BlockAverage:
    """Property fields averaged in blocks onto a coarser grid.

    Parameters
    ----------
    dims
        The fine grid, ``[nx, ny, nz]``.
    factors
        Fine cells per coarse cell along each axis, ``[fx, fy, fz]``.
    means
        Per variable its mean: `arithmetic` (porosity, ln k), `geometric` or `harmonic`
        (positive values). The variables averaged; other state variables pass through.
    """

    status = "stable"

    def __init__(
        self,
        dims: Sequence[int],
        factors: Sequence[int],
        means: Mapping[Annotated[str, TEXT], Mean],
    ) -> None:
        if len(dims) != 3 or len(factors) != 3:
            raise ValueError("dims and factors are three integers each, [x, y, z]")
        if any(int(d) < 1 for d in dims) or any(int(f) < 1 for f in factors):
            raise ValueError("dims and factors must be positive")
        if not means:
            raise ValueError("means names no variable to average")
        bad = sorted(
            {m for m in means.values() if m not in ("arithmetic", "geometric", "harmonic")}
        )
        if bad:
            raise ValueError(f"unknown means {bad}: arithmetic, geometric or harmonic")
        self.dims = tuple(int(d) for d in dims)
        self.factors = tuple(int(f) for f in factors)
        self.means: dict[str, Mean] = {str(k): v for k, v in means.items()}
        self.index, self.n_coarse = block_index(self.dims, self.factors)
        self.variables: tuple[str, ...] = tuple(self.means)
        self.outputs: tuple[str, ...] = tuple(self.means)

    @property
    def output_grid(self) -> tuple[int, int, int]:
        """The grid of its outputs (``workflows.parameterization.PropertyMap`` reads it)."""
        return self.coarse_dims

    @property
    def coarse_dims(self) -> tuple[int, int, int]:
        """The coarse grid, ``(cx, cy, cz)``."""
        nx, ny, nz = self.dims
        fx, fy, fz = self.factors
        return (-(-nx // fx), -(-ny // fy), -(-nz // fz))

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return each averaged variable on the coarse grid, ``(n_coarse,)``.

        Raises
        ------
        ValueError
            If a variable is missing or not the fine grid's size.
        """
        n_fine = self.index.size
        out = {}
        for name, mean in self.means.items():
            if name not in state:
                raise ValueError(f"block_average: the variable {name!r} is missing")
            values = np.asarray(state[name], dtype=np.float64).reshape(-1)
            if values.size != n_fine:
                raise ValueError(
                    f"block_average: {name!r} has {values.size} values, the grid {n_fine}"
                )
            out[name] = block_average(values, self.index, self.n_coarse, mean)
        return out
