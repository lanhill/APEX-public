"""A case's cells and the faces between them, as data.

What a case-state reader with the capability ``connectivity`` gives
(:class:`~apex.core.case_state.ConnectivityReader`): :class:`CellGraph`, the pairs of cells
that share a face, over the cells the reader's ``state()`` returns, in its order (a
structured grid's active cells in Eclipse (F) order, an unstructured mesh's cells in the
simulator's own), with each face's area and the distance between the two cells' centres.
No I/O; the helpers find the faces of a structured grid (:func:`grid_faces`) and of a mesh
of cells given by their corner nodes (:func:`shared_faces`, VTK's cell types), and convert a
graph to plain arrays and back (:func:`cell_graph_arrays`, :func:`cell_graph_from_arrays`).

Shapes: ``pairs`` ``(n_faces, 2)``; ``areas``, ``distances`` ``(n_faces,)``; ``centres``
``(n_cells, 3)``. Lengths in m, areas in m² (SI, as :class:`~apex.domain.case_state.CaseMesh`).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from apex.domain.case_state import CaseMesh

__all__ = [
    "CELL_FACES",
    "CellGraph",
    "cell_graph_arrays",
    "cell_graph_from_arrays",
    "centre_distances",
    "grid_faces",
    "shared_faces",
]


@dataclass(frozen=True, eq=False)
class CellGraph:
    """The faces between a case's cells, in canonical order.

    Each face is one row of ``pairs``, its lower cell first, the rows sorted by the lower
    cell, then the upper (:meth:`__post_init__` sorts them, reordering ``areas`` and
    ``distances`` with them): one case gives one graph whatever order its files list the
    faces in, so whatever is computed over its faces is reproducible.

    Attributes
    ----------
    cells
        The number of cells: the reader's ``CaseMesh.cells``, the cells its ``state()``
        returns (a structured grid's active cells, an unstructured mesh's every cell).
    pairs
        ``(n_faces, 2)`` int64: two cells (indices into the reader's cell order) that share
        a face, each pair once; a cell is never its own neighbour.
    areas
        ``(n_faces,)``: each face's area (m²), ``None`` when the reader cannot give it. A
        2-D mesh's faces are its cells' edges and a 1-D mesh's its nodes; their areas
        follow the mesh's measure (an edge's length per metre of the axis the mesh does not
        span; a node's 1 m²), as its cells' volumes do.
    distances
        ``(n_faces,)``: the distance between the two cells' centres (m), ``None`` when not
        given.

    Raises
    ------
    ValueError
        If a pair is out of range, a cell's own, or listed twice, or an array's shape does
        not fit the pairs.
    """

    cells: int
    pairs: NDArray[np.int64]
    areas: NDArray[np.float64] | None = None
    distances: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate, and put the faces in canonical order (lower cell, then upper)."""
        cells = int(self.cells)
        if cells < 0:
            raise ValueError(f"a cell graph's cells must not be negative, got {cells}")
        pairs = np.asarray(self.pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size and (pairs.min() < 0 or pairs.max() >= cells):
            raise ValueError(f"a face names a cell outside 0..{cells - 1}")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("a face joins a cell to itself")
        low = np.minimum(pairs[:, 0], pairs[:, 1])
        high = np.maximum(pairs[:, 0], pairs[:, 1])
        order = np.lexsort((high, low))
        canonical = np.ascontiguousarray(np.column_stack([low, high])[order])
        if len(canonical) > 1 and np.any(np.all(np.diff(canonical, axis=0) == 0, axis=1)):
            raise ValueError("a pair of cells is listed twice (each face once)")
        canonical.setflags(write=False)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "pairs", canonical)
        for name in ("areas", "distances"):
            values = getattr(self, name)
            if values is None:
                continue
            found = np.asarray(values, dtype=np.float64).reshape(-1)
            if found.shape != (len(pairs),):
                raise ValueError(f"{name} must be ({len(pairs)},), one per face; got {found.shape}")
            if not np.all(np.isfinite(found)) or np.any(found < 0):
                raise ValueError(f"{name} must be finite and not negative")
            ordered = np.ascontiguousarray(found[order])
            ordered.setflags(write=False)
            object.__setattr__(self, name, ordered)

    @property
    def faces(self) -> int:
        """The number of faces."""
        return int(self.pairs.shape[0])

    def same_faces(self, other: "CellGraph") -> bool:
        """Whether two graphs join the same cells (areas and distances not compared)."""
        return self.cells == other.cells and np.array_equal(self.pairs, other.pairs)


def centre_distances(centres: NDArray[np.float64], pairs: NDArray[np.int64]) -> NDArray[np.float64]:
    """Return ``(n_faces,)``: the distance between each pair's cell centres ``(cells, 3)``."""
    c = np.asarray(centres, dtype=np.float64)
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    out: NDArray[np.float64] = np.linalg.norm(c[p[:, 1]] - c[p[:, 0]], axis=1)
    return out


def grid_faces(
    dims: tuple[int, int, int], active: NDArray[np.bool_] | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return a structured grid's faces between active cells, and each face's axis.

    ``active`` is ``(nx * ny * nz,)`` in Eclipse (F) order (``None``: every cell). Cells are
    numbered as a reader's state gives them: the active cells in that order. The faces are
    the logical neighbours (i±1, j±1, k±1); a structured grid's non-neighbour connections
    (faults, pinch-outs) are not among them.

    Returns
    -------
    tuple
        ``pairs`` ``(n_faces, 2)`` (lower cell first, in canonical order) and ``axis``
        ``(n_faces,)``: 0, 1 or 2 for a face across i, j or k.
    """
    nx, ny, nz = (int(v) for v in dims)
    total = nx * ny * nz
    mask = np.ones(total, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != (total,):
        raise ValueError(f"active must be ({total},) for a {dims} grid, got {mask.shape}")
    node = np.full(total, -1, dtype=np.int64)
    cells = np.flatnonzero(mask)
    node[cells] = np.arange(cells.size)
    i, j, k = cells % nx, (cells // nx) % ny, cells // (nx * ny)
    pairs, axes = [], []
    for axis, inside, offset in ((0, i < nx - 1, 1), (1, j < ny - 1, nx), (2, k < nz - 1, nx * ny)):
        source = cells[inside]
        target = node[source + offset]
        keep = target >= 0
        pairs.append(np.column_stack([node[source[keep]], target[keep]]))
        axes.append(np.full(int(keep.sum()), axis, dtype=np.int64))
    stacked = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    axis_of = np.concatenate(axes) if axes else np.zeros(0, dtype=np.int64)
    order = np.lexsort((stacked[:, 1], stacked[:, 0]))
    return np.ascontiguousarray(stacked[order]), np.ascontiguousarray(axis_of[order])


#: The faces of VTK's cell types (corner nodes, in VTK's order and orientation), by
#: meshio's names: a 3-D cell's faces are polygons, a 2-D cell's its edges, a 1-D cell's
#: its two nodes. Higher-order types take their first-order shape's (their corners come
#: first, in the same order).
CELL_FACES: Mapping[str, tuple[tuple[int, ...], ...]] = {
    "line": ((0,), (1,)),
    "triangle": ((0, 1), (1, 2), (2, 0)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
    "tetra": ((0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)),
    "pyramid": ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
    "wedge": ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    "hexahedron": (
        (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7),
    ),
}  # fmt: skip

#: Higher-order types and their first-order shapes.
_FIRST_ORDER = {
    "line3": "line", "triangle6": "triangle", "quad8": "quad", "quad9": "quad",
    "tetra10": "tetra", "pyramid13": "pyramid", "pyramid14": "pyramid", "wedge15": "wedge",
    "hexahedron20": "hexahedron", "hexahedron27": "hexahedron",
}  # fmt: skip

_DIMENSION = {"line": 1, "triangle": 2, "quad": 2, "tetra": 3, "pyramid": 3, "wedge": 3,
              "hexahedron": 3}  # fmt: skip


def shared_faces(
    points: NDArray[np.float64],
    blocks: Sequence[tuple[str, NDArray[np.int64]]],
    cells: NDArray[np.int64] | None = None,
) -> CellGraph:
    """Return the faces two cells of a conforming mesh share, with their areas.

    ``points`` ``(n_points, 3)``; ``blocks`` the mesh's cells by type, each ``(type,
    connectivity (n, nodes))`` with meshio's type names (:data:`CELL_FACES`), the cells
    numbered across the blocks in order. ``cells`` picks the cells the graph is over, in
    the graph's order (a reader's state cells, a subset of the mesh's); ``None``: every cell
    in the mesh's order. A face is a cell's face whose corner nodes, as a set, are another
    cell's face's; a face of a picked cell and one that is not is a boundary face. Its area
    is its polygon's vector area (half the norm of the sum of its edges' cross products about
    its corners' mean: exact for a planar face, the mean normal's for a warped one); a 2-D
    mesh's edges have their lengths, a 1-D mesh's nodes 1. Distances are not computed (the
    reader's cell centres give them: :func:`centre_distances`).

    Raises
    ------
    ValueError
        If a cell type is not known, the picked cells mix dimensions (lines with quads, a
        fracture network in a matrix), or a face is shared by more than two cells (not a
        conforming mesh of one dimension).
    """
    pts = np.asarray(points, dtype=np.float64)
    counts = [int(np.asarray(c).shape[0]) for _, c in blocks]
    total = sum(counts)
    picked = np.arange(total, dtype=np.int64) if cells is None else np.asarray(cells, np.int64)
    if picked.size and (picked.min() < 0 or picked.max() >= total):
        raise ValueError(f"a picked cell is outside the mesh's {total} cells")
    position = np.full(total, -1, dtype=np.int64)
    position[picked] = np.arange(picked.size)
    owners: list[NDArray[np.int64]] = []
    keys: list[NDArray[np.int64]] = []
    polygons: list[tuple[int, NDArray[np.int64]]] = []  # (corners per face, (m, corners))
    dimensions: set[int] = set()
    start = 0
    width = 4  # the widest face: a quadrilateral
    for kind, connectivity in blocks:
        conn = np.asarray(connectivity, dtype=np.int64)
        n = int(conn.shape[0])
        base = _FIRST_ORDER.get(kind, kind)
        if base not in CELL_FACES:
            raise ValueError(
                f"cells of type {kind!r} have no face table (known: {', '.join(CELL_FACES)})"
            )
        mine = position[start : start + n]
        start += n
        keep = mine >= 0
        if not np.any(keep):
            continue
        dimensions.add(_DIMENSION[base])
        conn, mine = conn[keep], mine[keep]
        for face in CELL_FACES[base]:
            nodes = conn[:, list(face)]
            owners.append(mine)
            key = np.full((len(nodes), width), -1, dtype=np.int64)
            key[:, : len(face)] = np.sort(nodes, axis=1)
            keys.append(key)
            polygons.append((len(face), nodes))
    if len(dimensions) > 1:
        raise ValueError(
            f"the cells mix dimensions ({', '.join(str(d) for d in sorted(dimensions))}-D): "
            "faces between cells of different dimensions are not a mesh's faces"
        )
    if not keys:
        return CellGraph(int(picked.size), np.zeros((0, 2), dtype=np.int64), np.zeros(0))
    owner = np.concatenate(owners)
    key = np.concatenate(keys)
    area = np.concatenate([_face_areas(pts, nodes, corners) for corners, nodes in polygons])
    _, inverse, multiplicity = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if np.any(multiplicity > 2):
        raise ValueError(
            f"{int(np.sum(multiplicity > 2))} faces are shared by more than two cells: not a "
            "conforming mesh of one dimension"
        )
    shared = multiplicity[inverse] == 2
    # Each shared face twice: once from each cell. Ordering by the face groups the two.
    rows = np.flatnonzero(shared)
    rows = rows[np.argsort(inverse[rows], kind="stable")]
    first, second = rows[0::2], rows[1::2]
    pairs = np.column_stack([owner[first], owner[second]])
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("a cell has two faces on the same nodes (a degenerate cell)")
    return CellGraph(int(picked.size), pairs, area[first])


def _face_areas(
    points: NDArray[np.float64], nodes: NDArray[np.int64], corners: int
) -> NDArray[np.float64]:
    """Return each face's area: a polygon's vector area, an edge's length, a node's 1."""
    if corners == 1:
        return np.ones(len(nodes))
    p = points[nodes]  # (m, corners, 3)
    if corners == 2:
        out: NDArray[np.float64] = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
        return out
    centre = p.mean(axis=1)
    total = np.zeros((len(p), 3))
    for a in range(corners):
        total += np.cross(p[:, a] - centre, p[:, (a + 1) % corners] - centre)
    area: NDArray[np.float64] = 0.5 * np.linalg.norm(total, axis=1)
    return area


#: The arrays :func:`cell_graph_arrays` writes.
_ARRAYS = ("cells", "pairs", "centres", "volumes")


def cell_graph_arrays(mesh: CaseMesh, graph: CellGraph) -> dict[str, NDArray[np.generic]]:
    """Return a mesh and its cell graph as plain arrays (a deterministic ``.npz``'s).

    Raises
    ------
    ValueError
        If the graph is not over the mesh's cells.
    """
    if graph.cells != mesh.cells:
        raise ValueError(f"the graph has {graph.cells} cells, the mesh {mesh.cells}")
    found: dict[str, NDArray[np.generic]] = {
        "cells": np.asarray(mesh.cells, dtype=np.int64),
        "pairs": graph.pairs,
        "centres": np.asarray(mesh.centres, dtype=np.float64),
        "volumes": np.asarray(mesh.volumes, dtype=np.float64),
    }
    if graph.areas is not None:
        found["areas"] = graph.areas
    if graph.distances is not None:
        found["distances"] = graph.distances
    if mesh.dims is not None:
        found["dims"] = np.asarray(mesh.dims, dtype=np.int64)
        if mesh.active is not None:
            found["active"] = np.asarray(mesh.active, dtype=np.bool_)
    return found


def cell_graph_from_arrays(
    arrays: Mapping[str, NDArray[np.generic]],
) -> tuple[CaseMesh, CellGraph]:
    """Return the mesh and graph :func:`cell_graph_arrays` wrote.

    Raises
    ------
    ValueError
        If an array is missing or does not fit the others.
    """
    missing = [name for name in _ARRAYS if name not in arrays]
    if missing:
        raise ValueError(f"the cell graph's arrays lack {', '.join(missing)}")
    cells = int(np.asarray(arrays["cells"]))
    centres = np.asarray(arrays["centres"], dtype=np.float64).reshape(-1, 3)
    volumes = np.asarray(arrays["volumes"], dtype=np.float64).reshape(-1)
    if centres.shape[0] != cells or volumes.shape != (cells,):
        raise ValueError(f"the cell graph's centres and volumes are not its {cells} cells'")
    dims: tuple[int, int, int] | None = None
    active = None
    if "dims" in arrays:
        found = [int(v) for v in np.asarray(arrays["dims"]).reshape(-1)]
        if len(found) != 3 or min(found) < 1:
            raise ValueError(f"the cell graph's dims {found} are not three positive integers")
        dims = (found[0], found[1], found[2])
        if "active" in arrays:
            active = np.asarray(arrays["active"], dtype=np.bool_).reshape(-1)
            if active.shape != (math.prod(dims),) or int(active.sum()) != cells:
                raise ValueError("the cell graph's active cells are not its grid's")
    mesh = CaseMesh(cells, centres, volumes, dims=dims, active=active)
    graph = CellGraph(
        cells,
        np.asarray(arrays["pairs"], dtype=np.int64),
        None if "areas" not in arrays else np.asarray(arrays["areas"], dtype=np.float64),
        None if "distances" not in arrays else np.asarray(arrays["distances"], dtype=np.float64),
    )
    return mesh, graph
