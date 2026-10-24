"""Reference directions on the unit simplex, for NSGA-III and MOEA/D.

Ported from pymoo
-----------------
pymoo      : github.com/anyoptimization/pymoo, version 0.6.2, commit 217af15
             (Apache-2.0; LICENSES/Apache-2.0-pymoo.txt, THIRD_PARTY_NOTICES.md)
Module     : pymoo/util/reference_direction.py (``das_dennis``, ``das_dennis_recursion``)
Reference  : Das, I. & Dennis, J. E. (1998). Normal-boundary intersection: a new method for
             generating the Pareto surface in nonlinear multicriteria optimization problems.
             SIAM J. Optim., 8, 631-657.
             The inner layer as pymoo's ``MultiLayerReferenceDirectionFactory`` with a
             Das-Dennis layer scaled by 0.5 (Deb & Jain 2014's two layers).
Differences: the defaults (:func:`default_layers`) are APEX's. pymoo has none (the user
             passes the directions), and the papers' benchmarks use about three times as many
             directions for three objectives (12 partitions, 91 directions, where
             APEX takes 6, 28), as an evaluation here is a reservoir simulation.

From 5 objectives a single layer with fewer partitions than objectives has no interior
direction (every one has a zero coordinate), so the defaults there are two layers (the user's
choice, 2026-10-06): the boundary with 2 partitions and an inner layer with 1, shrunk halfway
to the centre (``0.5 d + 0.5 / M``): 20 directions for 5 objectives, 27 for 6, 44 for 8.

The directions are rows ``(n_directions, n_obj)`` (each one a point of the simplex: its
coordinates sum to 1); a layer is in pymoo's ``das_dennis`` order, the boundary first.
"""

import math

import numpy as np

from apex.algorithms.optimization.protocols import FloatArray

#: APEX's default partitions by number of objectives: about 20-35 directions, a
#: third of the papers' density for three objectives. From 5 objectives: two layers.
DEFAULT_PARTITIONS = {2: 19, 3: 6, 4: 4}
#: Two layers from this many objectives: the boundary's and the inner layer's partitions.
TWO_LAYERS_FROM, BOUNDARY_PARTITIONS, INNER_PARTITIONS = 5, 2, 1
#: How far the inner layer is shrunk towards the simplex's centre (Deb & Jain 2014: 0.5).
INNER_SCALING = 0.5


def default_layers(n_obj: int) -> tuple[int, int]:
    """Return APEX's default partitions ``(boundary, inner)`` (inner 0: one layer)."""
    if n_obj < 2:
        raise ValueError(f"reference directions need at least two objectives, got {n_obj}")
    if n_obj >= TWO_LAYERS_FROM:
        return BOUNDARY_PARTITIONS, INNER_PARTITIONS
    return DEFAULT_PARTITIONS[n_obj], 0


def default_partitions(n_obj: int) -> int:
    """Return the default boundary partitions (see :func:`default_layers`)."""
    return default_layers(n_obj)[0]


def layered_directions(n_partitions: int, inner_partitions: int, n_obj: int) -> FloatArray:
    """Return a boundary layer and, if ``inner_partitions``, an inner layer ``(k, n_obj)``.

    The inner layer is Das-Dennis's with ``inner_partitions``, shrunk to
    ``INNER_SCALING d + (1 - INNER_SCALING) / n_obj`` (pymoo's scaling): interior points.
    """
    boundary = das_dennis(n_partitions, n_obj)
    if not inner_partitions:
        return boundary
    inner = das_dennis(inner_partitions, n_obj) * INNER_SCALING + (1.0 - INNER_SCALING) / n_obj
    both = np.vstack([boundary, inner])
    # Chosen layers can share a direction (below 5 objectives), which would count its niche
    # twice: each once, in order (numerics review, 2026-10-06).
    _, first = np.unique(np.round(both, 12), axis=0, return_index=True)
    return np.ascontiguousarray(both[np.sort(first)])


def directions_for(
    n_obj: int, n_partitions: int | None = None, inner_partitions: int | None = None
) -> FloatArray:
    """Return the directions an optimizer uses: the defaults, or the partitions given.

    With ``n_partitions`` given and no ``inner_partitions``, one layer (as pymoo's
    Das-Dennis); without either, :func:`default_layers`.
    """
    if n_partitions is None and inner_partitions is None:
        boundary, inner = default_layers(n_obj)
    else:
        boundary = n_partitions if n_partitions is not None else default_partitions(n_obj)
        inner = inner_partitions or 0
    return layered_directions(boundary, inner, n_obj)


def number_of_directions(n_partitions: int, n_obj: int) -> int:
    """Return how many directions Das-Dennis gives: ``C(n_obj + p - 1, p)``."""
    return math.comb(n_obj + n_partitions - 1, n_partitions)


def das_dennis(n_partitions: int, n_obj: int) -> FloatArray:
    """Return Das-Dennis's uniform directions ``(n_directions, n_obj)``, in pymoo's order.

    Every point whose coordinates are multiples of ``1 / n_partitions`` and sum to 1; with no
    partitions, the simplex's centre.
    """
    if n_partitions < 0 or n_obj < 1:
        raise ValueError(f"n_partitions must be >= 0 and n_obj >= 1, got {n_partitions}, {n_obj}")
    if n_partitions == 0:
        return np.full((1, n_obj), 1.0 / n_obj)
    rows: list[FloatArray] = []
    _recurse(rows, np.full(n_obj, np.nan), n_partitions, n_partitions, 0)
    return np.ascontiguousarray(np.vstack(rows))


def _recurse(
    rows: list[FloatArray], point: FloatArray, n_partitions: int, beta: int, depth: int
) -> None:
    # pymoo's das_dennis_recursion: the last coordinate takes what is left.
    if depth == len(point) - 1:
        point[depth] = beta / (1.0 * n_partitions)
        rows.append(point[None, :])
        return
    for i in range(beta + 1):
        point[depth] = 1.0 * i / (1.0 * n_partitions)
        _recurse(rows, np.copy(point), n_partitions, beta - i, depth + 1)
