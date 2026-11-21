"""A case's cell graph: canonical order, grids' and meshes' faces.

Independent references: a structured grid's faces against the definition (each active cell's
i+, j+ and k+ neighbour, when active), and a mesh's shared faces against the grid's on the
same boxes written as hexahedra, triangles and lines, with areas from their sizes.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.domain.case_state import CaseMesh
from apex.domain.cell_graph import (
    CellGraph,
    cell_graph_arrays,
    cell_graph_from_arrays,
    centre_distances,
    grid_faces,
    shared_faces,
)


def test_faces_are_put_in_canonical_order_with_their_values() -> None:
    graph = CellGraph(4, np.array([[3, 2], [0, 1], [2, 0]]), np.array([1.0, 2.0, 3.0]),
                      np.array([4.0, 5.0, 6.0]))  # fmt: skip
    np.testing.assert_array_equal(graph.pairs, [[0, 1], [0, 2], [2, 3]])
    assert graph.areas is not None
    assert graph.distances is not None
    np.testing.assert_array_equal(graph.areas, [2.0, 3.0, 1.0])
    np.testing.assert_array_equal(graph.distances, [5.0, 6.0, 4.0])
    assert graph.faces == 3
    assert graph.same_faces(CellGraph(4, np.array([[1, 0], [2, 3], [0, 2]])))
    assert not graph.pairs.flags.writeable


@pytest.mark.parametrize(
    ("pairs", "areas", "match"),
    [
        ([[0, 4]], None, "outside"),
        ([[1, 1]], None, "to itself"),
        ([[0, 1], [1, 0]], None, "twice"),
        ([[0, 1]], [1.0, 2.0], "one per face"),
        ([[0, 1]], [-1.0], "not negative"),
    ],
)
def test_bad_graphs_are_refused(pairs: list[list[int]], areas: list[float] | None,
                                match: str) -> None:  # fmt: skip
    with pytest.raises(ValueError, match=match):
        CellGraph(4, np.array(pairs), None if areas is None else np.array(areas))


def _expected_grid_faces(
    dims: tuple[int, int, int], active: NDArray[np.bool_]
) -> set[tuple[int, int, int]]:
    nx, ny, nz = dims
    node = np.cumsum(active) - 1
    found = set()
    for cell in np.flatnonzero(active):
        i, j, k = cell % nx, (cell // nx) % ny, cell // (nx * ny)
        for axis, inside, offset in ((0, i < nx - 1, 1), (1, j < ny - 1, nx),
                                     (2, k < nz - 1, nx * ny)):  # fmt: skip
            if inside and active[cell + offset]:
                found.add((int(node[cell]), int(node[cell + offset]), axis))
    return found


def test_a_grids_faces_are_its_active_cells_neighbours() -> None:
    dims = (4, 3, 2)
    active = np.ones(24, dtype=bool)
    active[[5, 18]] = False
    pairs, axis = grid_faces(dims, active)
    assert {(int(a), int(b), int(c)) for (a, b), c in zip(pairs, axis, strict=True)} == (
        _expected_grid_faces(dims, active)
    )
    # Already canonical: a cell graph keeps the order.
    np.testing.assert_array_equal(CellGraph(22, pairs).pairs, pairs)
    every, _ = grid_faces(dims)
    assert len(every) == 3 * 3 * 2 + 4 * 2 * 2 + 4 * 3 * 1


def _box_mesh(dims: tuple[int, int, int], size: tuple[float, float, float]) -> tuple[
    NDArray[np.float64], NDArray[np.int64]
]:  # fmt: skip
    """Hexahedra of a box grid in Eclipse order, VTK's corner order."""
    nx, ny, nz = dims
    xs, ys, zs = (np.arange(n + 1) * s for n, s in zip(dims, size, strict=True))
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs])

    def p(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    cells = [
        [p(i, j, k), p(i + 1, j, k), p(i + 1, j + 1, k), p(i, j + 1, k),
         p(i, j, k + 1), p(i + 1, j, k + 1), p(i + 1, j + 1, k + 1), p(i, j + 1, k + 1)]
        for k in range(nz) for j in range(ny) for i in range(nx)
    ]  # fmt: skip
    return points, np.asarray(cells, dtype=np.int64)


def test_hexahedra_share_the_grids_faces_with_their_areas() -> None:
    dims, size = (3, 2, 2), (10.0, 20.0, 5.0)
    points, cells = _box_mesh(dims, size)
    graph = shared_faces(points, [("hexahedron", cells)])
    pairs, axis = grid_faces(dims)
    np.testing.assert_array_equal(graph.pairs, pairs)
    assert graph.areas is not None
    np.testing.assert_allclose(graph.areas, np.choose(axis, [100.0, 50.0, 200.0]))
    # A subset of the cells, in another order: the graph over the picked cells.
    picked = np.array([11, 0, 1, 5, 2])
    sub = shared_faces(points, [("hexahedron", cells)], picked)
    assert sub.cells == 5
    expected = {(0, 3), (1, 2), (2, 4), (3, 4)}  # 11-5 (k), 0-1 (i), 1-2 (i), 5-2 (j)
    assert {tuple(int(v) for v in pair) for pair in sub.pairs} == expected


def test_a_2d_meshs_faces_are_edges_and_a_1d_meshs_nodes() -> None:
    points = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
    triangles = np.array([[0, 1, 3], [0, 3, 2]])
    graph = shared_faces(points, [("triangle", triangles)])
    np.testing.assert_array_equal(graph.pairs, [[0, 1]])
    assert graph.areas is not None
    np.testing.assert_allclose(graph.areas, [5.0])  # the diagonal's length
    line = shared_faces(points, [("line", np.array([[0, 1], [1, 3], [3, 2]]))])
    np.testing.assert_array_equal(line.pairs, [[0, 1], [1, 2]])
    assert line.areas is not None
    np.testing.assert_array_equal(line.areas, [1.0, 1.0])
    # Second-order cells share by their corners.
    quads = np.array([[0, 1, 3, 2, 9, 9, 9, 9, 9], [1, 4, 5, 3, 9, 9, 9, 9, 9]])
    more = np.vstack([points, [[6.0, 0.0, 0.0], [6.0, 4.0, 0.0]], np.zeros((4, 3))])
    assert shared_faces(more, [("quad9", quads)]).pairs.tolist() == [[0, 1]]


def test_mixed_dimensions_and_three_cells_on_a_face_are_refused() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="mix dimensions"):
        shared_faces(points, [("triangle", np.array([[0, 1, 2]])), ("line", np.array([[0, 1]]))])
    with pytest.raises(ValueError, match="more than two"):
        shared_faces(points, [("line", np.array([[0, 1], [1, 2], [1, 3]]))])
    with pytest.raises(ValueError, match="no face table"):
        shared_faces(points, [("vertex", np.array([[0]]))])


def test_a_mesh_and_its_graph_round_trip_as_arrays() -> None:
    dims = (2, 2, 1)
    active = np.array([True, True, False, True])
    pairs, _ = grid_faces(dims, active)
    centres = np.array([[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0]])
    mesh = CaseMesh(3, centres, np.ones(3), dims=dims, active=active)
    graph = CellGraph(3, pairs, np.ones(len(pairs)), centre_distances(centres, pairs))
    back_mesh, back = cell_graph_from_arrays(cell_graph_arrays(mesh, graph))
    assert back.same_faces(graph)
    assert back.distances is not None
    np.testing.assert_array_equal(back.distances, [1.0, 1.0])
    assert back_mesh.dims == dims
    assert back_mesh.active is not None
    np.testing.assert_array_equal(back_mesh.active, active)
    loose = cell_graph_arrays(CaseMesh(3, centres, np.ones(3)), CellGraph(3, pairs))
    assert "dims" not in loose
    assert "areas" not in loose
    with pytest.raises(ValueError, match="lack"):
        cell_graph_from_arrays({"cells": np.array(3)})
    with pytest.raises(ValueError, match="graph has 3 cells"):
        cell_graph_arrays(CaseMesh(4, np.zeros((4, 3)), np.ones(4)), graph)
