"""Grid coarsening for multilevel ensembles: Eclipse ``COARSEN`` records.

Ported from SimulatorWrap
-------------------------
Module     : subsurface/multphaseflow/eclipse.py (eclipse.coarsen, write_coarse,
             _nodeIndex, _Haar, _haarTrunc, _unif)
Commit     : d29dc574d06f8f617dc2796a63c1e240e7f500ee (github.com/Python-Ensemble-Toolbox/
             SimulatorWrap, GPL-3.0)
Reference  : Fossum, K., Mannseth, T. & Stordal, A. S. (2020). Assessment of multilevel
             ensemble-based data assimilation for reservoir history matching.
             Computational Geosciences, 24(1), 217-239.
Differences: SimulatorWrap's Haar path cannot run on Python 3: it slices with float
             indices (``image[:image.shape[0] / 2]``); here the halves are integers
             (``//``), the only change to its arithmetic. Its bare ``except`` around a merge
             test (out-of-box cells) is ``except IndexError``. Its unreachable 1-D branch
             of ``_Haar`` (every NumPy shape is a tuple) is not ported. SimulatorWrap
             coarsens one layer (``K1 = K2 = 1``; its reshape to ``dim`` fails for more);
             with ``layers = nz`` the same x-y blocks are merged in every layer, each kept
             (``K1 = 1, K2 = NZ = nz``), the pattern following the layer mean of the field.
             SimulatorWrap's ``eclipse.coarsen`` cannot run end to end at ``d29dc57``
             (``inv_state`` is never set, and ``run_fwd_sim`` passes the state dict as the
             member index); APEX coarsens on the member's own field.

A field is coarsened in the x-y plane only. ``dim = (n_rows, n_cols)`` is SimulatorWrap's:
the field (one value per column of cells, Eclipse order, x fastest) is reshaped to
``(dim[0], dim[1])``, so rows run along y and columns along x, and a record ``[i1, i2, j1,
j2, 1, 1, 1, 1, 1]`` merges columns ``i1..i2`` (x) and rows ``j1..j2`` (y), 1-based, into one
cell. Cells within ``radius`` of a listed well cell ``[x, y]`` (1-based) are never merged.

- ``uniform`` (SimulatorWrap ``unif``): dyadic ``2^k x 2^k`` blocks, ``k`` up to
  ``floor(max_levels * max_trunc)`` with ``max_levels`` the number of halvings of the
  smaller dimension.
- ``haar``: merges where an unbalanced 2-D Haar transform of the field has small details:
  a block is merged when its smooth coefficient is below ``max_trunc * max(field)`` and its
  detail coefficients are at most ``max_diff * (max(field) - min(field))``.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: One ``COARSEN`` record: ``i1, i2, j1, j2, k1, k2, nx, ny, nz`` (1-based).
Record = tuple[int, int, int, int, int, int, int, int, int]


def coarsen(
    field: ArrayLike,
    dim: Sequence[int],
    *,
    method: Literal["uniform", "haar"] = "uniform",
    max_trunc: float,
    max_diff: float = 0.0,
    wells: Sequence[Sequence[int]] = (),
    radius: Sequence[int] = (),
    layers: int = 1,
) -> list[Record]:
    """Return the ``COARSEN`` records for one member (see the module docstring).

    Parameters
    ----------
    field
        The parameter the coarsening follows (``haar``), ``(dim[0] * dim[1],)``; unused by
        ``uniform`` except for its size.
    dim
        ``(n_rows, n_cols)`` of the field (y, x).
    method
        ``"uniform"`` or ``"haar"``.
    max_trunc
        ``uniform``: fraction of the possible dyadic levels; ``haar``: smooth-value cut as a
        fraction of the field's maximum. 0 gives no records.
    max_diff
        ``haar``: detail cut as a fraction of the field's range.
    wells, radius
        Well cells ``[x, y]`` (1-based) and the radius, in cells, kept fine around each.
    layers
        Layers ``nz`` of the grid: every layer gets the same x-y blocks, and keeps its own
        cells vertically. The field has ``dim[0] * dim[1] * nz`` values (layers slowest);
        its layer mean sets the pattern.

    Raises
    ------
    ValueError
        For a field of the wrong size, an unknown method, or wells and radii of different
        lengths.
    """
    rows, cols, nz = int(dim[0]), int(dim[1]), int(layers)
    image = np.asarray(field, dtype=np.float64).reshape(-1)
    if nz < 1 or image.size != rows * cols * nz:
        raise ValueError(f"the field has {image.size} values, not {rows} x {cols} x {nz}")
    image = image.reshape(nz, rows * cols).mean(axis=0)
    if len(wells) != len(radius):
        raise ValueError(f"{len(wells)} wells but {len(radius)} radii")
    grid = image.reshape((rows, cols))
    if max_trunc <= 0:
        # SimulatorWrap's run_fwd_sim does not coarsen at all then (write_coarse alone
        # would still merge 2 x 2 blocks).
        return []
    if method == "uniform":
        weights = _uniform((rows, cols), max_trunc)
    elif method == "haar":
        weights = _haar(grid, np.ones((rows, cols)), max_trunc, max_diff)
    else:
        raise ValueError(f"method must be 'uniform' or 'haar', got {method!r}")
    records = _records(weights, grid.shape, wells, radius)
    if nz == 1:
        return records
    return [(r[0], r[1], r[2], r[3], 1, nz, 1, 1, nz) for r in records]


def _records(
    weights: list[NDArray[np.float64]],
    shape: tuple[int, int],
    wells: Sequence[Sequence[int]],
    radius: Sequence[int],
) -> list[Record]:
    """SimulatorWrap's ``write_coarse``: merge blocks from the coarsest level down."""
    fine = np.ones(shape, dtype=bool)  # True: may still be merged
    flat = fine.reshape(-1)
    flat[_node_index(shape[1], shape[0], wells, radius)] = False
    fine = flat.reshape(shape)
    records: list[Record] = []
    for level in range(len(weights) - 1, -1, -1):
        at_level = weights[level]
        x_dim, y_dim = at_level.shape[0], at_level.shape[1]
        merged = at_level[x_dim // 2 :, : y_dim // 2]
        size = 2 ** (level + 1)
        for i in range(merged.shape[0]):
            for j in range(merged.shape[1]):
                if merged[i, j] != 1:
                    continue
                i_end = min((i + 1) * size, shape[0])
                j_end = min((j + 1) * size, shape[1])
                if i * size >= shape[0] or j * size >= shape[1]:
                    continue
                block = fine[i * size : i_end, j * size : j_end]
                # The four cases of SimulatorWrap (square, and cut by either or both edges)
                # all merge the block clipped to the field.
                if block.all():
                    fine[i * size : i_end, j * size : j_end] = False
                    records.append((j * size + 1, j_end, i * size + 1, i_end, 1, 1, 1, 1, 1))
    return records


def _node_index(
    x_dir: int, y_dir: int, cells: Sequence[Sequence[int]], radii: Sequence[int]
) -> list[int]:
    """Flat indices (x fastest) of the cells within each well's radius (``_nodeIndex``)."""
    index = []
    for (x_cell, y_cell, *_), radius in zip(cells, radii, strict=True):
        cx, cy = int(x_cell) - 1, int(y_cell) - 1  # 1-based to 0-based
        r = int(radius)
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                x, y = cx + i, cy + j
                if 0 <= x < x_dir and 0 <= y < y_dir and np.sqrt(i**2 + j**2) <= r:
                    index.append(x + y * x_dir)
    return index


def _uniform(dim: tuple[int, int], max_trunc: float) -> list[NDArray[np.float64]]:
    """SimulatorWrap's ``_unif``: the dyadic levels to merge."""
    min_dim, max_levels = float(min(dim)), 0
    while min_dim / 2 >= 1:
        min_dim, max_levels = min_dim / 2, max_levels + 1
    levels = int(np.floor(max_levels * max_trunc))  # conservative
    weights = [np.ones(dim)]
    for i in range(levels):
        weights.append(np.ones(tuple(int(np.ceil(n / 2)) for n in weights[i].shape)))
    return weights


def _haar(
    image: NDArray[np.float64], weights: NDArray[np.float64], max_trunc: float, max_diff: float
) -> list[NDArray[np.float64]]:
    """SimulatorWrap's ``_Haar``: the unbalanced 2-D Haar transform, level by level."""
    max_value = max_trunc * np.max(image)
    max_detail = max_diff * (np.max(image) - np.min(image))
    total_weights = []
    allow_merge: NDArray[Any] = np.ones(image.shape)
    while image.shape[0] > 1:
        for _axis in range(2):
            half = int(np.ceil(image.shape[0] / 2))
            diffs = np.empty((half, image.shape[1]))
            smooths = np.empty((half, image.shape[1]))
            new_weights = np.empty((half, image.shape[1]))
            for i in range(image.shape[1]):
                column, w = image[:, i], weights[:, i]
                if column.size % 2 == 0:
                    diff = column[1::2] - column[::2]
                    merged = w[::2] + w[1::2]
                    smooth = column[::2] + (w[1::2] / merged) * diff
                else:
                    head, head_w = column[:-1], w[:-1]
                    diff = head[1::2] - head[::2]
                    merged = np.append(head_w[::2] + head_w[1::2], w[-1])
                    smooth = np.append(head[::2] + (w[1::2] / merged[:-1]) * diff, column[-1])
                    diff = np.append(diff, 0)
                diffs[:, i], new_weights[:, i], smooths[:, i] = diff, merged, smooth
            image = np.vstack((smooths, diffs)).T
            weights = np.vstack((new_weights, new_weights)).T
        image, weights, allow_merge, done = _haar_truncate(
            image, weights, max_value, max_detail, allow_merge
        )
        total_weights.append(weights)
        if done:
            break
        # SimulatorWrap: image.shape[0] / 2 (a float index on Python 3).
        image = image[: image.shape[0] // 2, : image.shape[1] // 2]
        weights = weights[: weights.shape[0] // 2, : weights.shape[1] // 2]
    return total_weights


def _haar_truncate(
    image: NDArray[np.float64],
    weights: NDArray[np.float64],
    max_value: float,
    max_detail: float,
    merge: NDArray[Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], bool]:
    """SimulatorWrap's ``_haarTrunc``: zero small details where all four cells may merge."""
    x_dir, y_dir = image.shape[0] // 2, image.shape[1] // 2
    smooth = image[:x_dir, :y_dir]
    smooth_diff, diff_smooth = image[x_dir:, :y_dir], image[:x_dir, y_dir:]
    diff_diff, weights_diff = image[x_dir:, y_dir:], weights[x_dir:, :y_dir]
    allow_merge = np.zeros((x_dir, y_dir), dtype=bool)
    # Keep the last row and column during non-dyadic upscaling.
    if image.shape[0] > merge.shape[0]:
        merge = np.insert(merge, -1, merge[-1, :], axis=0)
    if image.shape[1] > merge.shape[1]:
        merge = np.insert(merge, -1, merge[:, -1], axis=1)
    done = True
    for element in zip(*np.where(smooth < max_value), strict=True):
        r, c = int(element[0]), int(element[1])
        try:
            mergeable = (
                abs(smooth_diff[r, c]) <= max_detail
                and abs(diff_smooth[r, c]) <= max_detail
                and merge[2 * r, 2 * c]
                and merge[2 * r + 1, 2 * c]
                and merge[2 * r + 1, 2 * c + 1]
                and merge[2 * r, 2 * c + 1]
            )
        except IndexError:  # a merge reaching outside the field is not made
            continue
        if mergeable:
            diff_smooth[r, c] = smooth_diff[r, c] = diff_diff[r, c] = 0
            weights_diff[r, c] = True
            allow_merge[r, c] = True
            done = False
    image[x_dir:, :y_dir], image[:x_dir, y_dir:], image[x_dir:, y_dir:] = (
        smooth_diff,
        diff_smooth,
        diff_diff,
    )
    weights[x_dir:, :y_dir] = weights_diff
    return image, weights, allow_merge, done
