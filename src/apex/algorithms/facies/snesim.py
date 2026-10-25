"""The ``mpslib_snesim`` prior: SNESIM facies from a training image, as signed distances.

SNESIM (Strebelle, 2002) simulates facies sequentially along a random
path, reading each cell's conditional facies probabilities from a training image's search
tree, on multiple grids. MPSlib (Hansen, Vu & Bach, 2016; ``scikit-mps``, the ``facies``
extra, LGPL) provides the program ``mps_snesim_tree``; this plugin writes its parameter
file, training image, hard data and soft data (GSLIB/EAS formats, as MPSlib's own
writers), runs it in
a temporary folder, and reads the realisations back. MPSlib's Python wrapper is not used
(it needs NumPy < 2).

Distances are per layer (``dz`` unused): with a 3-D training image or ``tz > 1`` the state
loses the vertical continuity SNESIM simulated, and with the default ``tz = 1`` each layer
is simulated independently (MPSlib's behaviour). Each realisation becomes the signed
distance fields of its facies
(:func:`apex.algorithms.facies.distance.signed_distances`), the state the methods update;
``facies_distance`` maps them back. Grids are Eclipse ``[nx, ny, nz]``, ``i`` fastest,
which is GSLIB's order too.

References
----------
Strebelle, S. (2002). Conditional simulation of complex geological structures using
multiple-point statistics. Mathematical Geology, 34(1), 1-21.
Hansen, T. M., Vu, L. T. & Bach, T. (2016). MPSLIB: A C++ class for sequential simulation of
multiple-point statistical models. SoftwareX, 5, 127-133.
"""

import importlib.util
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.facies.distance import signed_distances
from apex.core.options import TEXT, InputFile, Program, mapping
from apex.core.plugins import priors
from apex.domain.state import StateVariable

#: The program, as installed with ``scikit-mps`` (or found on the PATH).
DEFAULT_PROGRAM = "mps_snesim_tree"
#: What the conditioning's strings are: data, not paths.
CONDITIONING_ROLES = mapping()


def bundled_program(name: str = DEFAULT_PROGRAM) -> Path | None:
    """Return MPSlib's program shipped with ``scikit-mps``, or one on the PATH, or ``None``.

    The package is located, not imported: its wrapper needs NumPy < 2.
    """
    spec = importlib.util.find_spec("mpslib")
    if spec is not None and spec.origin is not None:
        bundled = Path(spec.origin).parent / "bin" / name
        if bundled.is_file():
            return bundled
    found = shutil.which(name)
    return None if found is None else Path(found)


def read_training_image(path: Path) -> NDArray[np.int64]:
    """Return a training image ``(nx, ny, nz)`` of integer facies codes.

    ``.npy`` (2-D or 3-D, ``[i, j(, k)]``) or a GSLIB/EAS grid file (first line
    ``nx ny nz``, then one variable).

    Raises
    ------
    ValueError
        If it is not integer codes on a grid.
    """
    if path.suffix == ".npy":
        array = np.load(path)
        if array.ndim == 2:
            array = array[:, :, None]
    else:
        array = read_gslib_grid(path)
    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 3 or not np.all(values == np.round(values)):
        raise ValueError(f"{path}: a training image is a 2-D or 3-D grid of integer codes")
    return values.astype(np.int64)


def read_gslib_grid(path: Path) -> NDArray[np.float64]:
    """Read a GSLIB/EAS grid file (``nx ny nz`` title, one variable) as ``(nx, ny, nz)``."""
    lines = path.read_text(encoding="ascii").split("\n")
    dims = [int(v) for v in lines[0].split()[:3]]
    if len(dims) != 3:
        raise ValueError(f"{path}: the first line must be 'nx ny nz'")
    n_vars = int(lines[1].split()[0])
    values = np.array(" ".join(lines[2 + n_vars :]).split(), dtype=np.float64)
    if values.size != math.prod(dims) * n_vars:
        raise ValueError(f"{path}: {values.size} values for a {dims} grid")
    first = values.reshape(-1, n_vars)[:, 0]
    return first.reshape(dims, order="F")


def write_gslib_grid(array: NDArray[np.int64], path: Path) -> None:
    """Write ``(nx, ny, nz)`` codes as a GSLIB/EAS grid file (``i`` fastest)."""
    nx, ny, nz = array.shape
    body = "\n".join(str(int(v)) for v in array.ravel(order="F"))
    path.write_text(f"{nx} {ny} {nz}\n1\nfacies\n{body}\n", encoding="ascii")


@priors.register("mpslib_snesim")
class MpslibSnesim:
    """SNESIM facies realisations (MPSlib), as one signed distance field per facies.

    The group's variables are the facies' distance fields in code order (variable ``k`` is
    facies ``k``'s), all on one grid; the training image's codes are ``0..K-1``.

    Parameters
    ----------
    training_image_file
        The training image: ``.npy`` codes ``[i, j(, k)]`` or a GSLIB/EAS grid file.
    conditioning
        ``{cells = [[i, j, k], ...], facies = [code, ...]}``: 1-based Eclipse cells and the
        facies observed there (hard data); every realisation honours them.
    soft_probability_file
        Soft data: a ``.npy`` grid of facies probabilities ``(nx, ny, nz, K)`` (e.g. from
        seismic), NaN in cells without soft data. MPSlib combines each cell's probabilities
        with the training image's conditional ones (``P ∝ P_TI · P_soft``), so they shift
        the facies rather than fix them, unless they are 0 or 1. MPSlib reads the first
        ``K - 1`` and takes the last as one minus their sum (each cell is renormalised
        before writing). With the ``preferential`` path MPSlib also copies them to nearby
        coarse-grid nodes; with ``random`` they act at their own cell only. A realisation
        whose facies has soft probability 0 somewhere is refused: where the soft data
        contradict the training image MPSlib returns a probability as the facies code
        (``SNESIM.cpp``, its conflict branch; right only by chance for two facies). Hard
        data take precedence. A pdf is certain or has no zeros (refused otherwise). Needs a
        program that reads every soft category (checked).
    template_size
        The search template ``[tx, ty, tz]``.
    n_multiple_grids
        Multiple grids (coarse to fine; MPSlib counts from 0).
    n_cond
        Most conditioning data per cell (0: all).
    n_min_node_count
        Least count of a pattern for its conditional probability (0: no limit).
    random_path
        ``random`` (1) or ``preferential`` (2, MPSlib's default, with ``entropy_factor``).
    entropy_factor
        The preferential path's entropy factor.
    cell_size
        ``[dx, dy, dz]``, for the hard data's coordinates and the distances.
    distance_band
        Clip the signed distances to ``[-band, band]``, in the units of ``cell_size``
        (default: unclipped; a facies absent from a layer is at minus the layer's diagonal).
    program
        MPSlib's ``mps_snesim_tree`` (default: the one installed with ``scikit-mps``).
    timeout_s
        Seconds the program may take.
    """

    #: Signed distances are not bounded state: nothing to clip.
    clip = False

    def __init__(
        self,
        training_image_file: InputFile,
        conditioning: Annotated[Mapping[str, Sequence[Any]] | None, CONDITIONING_ROLES] = None,
        template_size: Sequence[int] = (8, 7, 1),
        n_multiple_grids: int = 3,
        n_cond: int = 16,
        n_min_node_count: int = 0,
        random_path: Annotated[str, TEXT] = "random",
        entropy_factor: float = 4.0,
        cell_size: Sequence[float] = (1.0, 1.0, 1.0),
        distance_band: float | None = None,
        soft_probability_file: InputFile | None = None,
        program: Program | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        self.training_image = read_training_image(Path(training_image_file))
        codes = np.unique(self.training_image)
        if codes.min() < 0 or not np.array_equal(codes, np.arange(codes.size)):
            raise ValueError(f"the training image's codes must be 0..K-1, got {codes.tolist()}")
        self.n_facies = int(codes.size)
        if len(template_size) != 3 or min(template_size) < 1:
            raise ValueError(f"template_size must be [tx, ty, tz] >= 1, got {template_size}")
        if random_path not in ("random", "preferential"):
            raise ValueError("random_path must be 'random' or 'preferential'")
        if n_multiple_grids < 0 or n_cond < 0 or n_min_node_count < 0:
            raise ValueError("n_multiple_grids, n_cond and n_min_node_count must be >= 0")
        self.template_size = tuple(int(t) for t in template_size)
        self.n_multiple_grids = int(n_multiple_grids)
        self.n_cond = int(n_cond)
        self.n_min_node_count = int(n_min_node_count)
        self.path_kind = 1 if random_path == "random" else 2
        self.entropy_factor = float(entropy_factor)
        self.cell_size = tuple(float(s) for s in cell_size)
        if len(self.cell_size) != 3 or min(self.cell_size) <= 0:
            raise ValueError(f"cell_size must be [dx, dy, dz] > 0, got {cell_size}")
        resolved = Path(program) if program is not None else bundled_program()
        if resolved is None or not resolved.is_file():
            raise ValueError(
                "mpslib_snesim needs MPSlib's mps_snesim_tree: install the 'facies' extra "
                "(scikit-mps) or give program"
            )
        self.program = resolved
        self.timeout_s = float(timeout_s)
        if distance_band is not None and distance_band <= 0:
            raise ValueError(f"distance_band must be positive, got {distance_band}")
        self.distance_band = distance_band
        self.soft: NDArray[np.float64] | None = None
        self._soft_checked = False
        if soft_probability_file is not None:
            self.soft = self._read_soft(Path(soft_probability_file))
        self.cells: NDArray[np.int64] | None = None
        self.observed: NDArray[np.int64] | None = None
        if conditioning is not None and len(conditioning.get("cells", [])) > 0:
            cells = np.asarray(conditioning["cells"], dtype=np.int64)
            observed = np.asarray(conditioning.get("facies", []), dtype=np.int64)
            if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] != observed.size:
                raise ValueError("conditioning needs cells [[i, j, k], ...] and one facies each")
            if observed.min() < 0 or observed.max() >= self.n_facies:
                raise ValueError(f"conditioning facies must be 0..{self.n_facies - 1}")
            self.cells, self.observed = cells, observed

    def _read_soft(self, path: Path) -> NDArray[np.float64]:
        """Return the soft probabilities ``(nx, ny, nz, K)``, validated (NaN: no soft data)."""
        soft = np.asarray(np.load(path), dtype=np.float64)
        if soft.ndim != 4 or soft.shape[3] != self.n_facies:
            raise ValueError(
                f"{path}: soft probabilities must be (nx, ny, nz, {self.n_facies}), got "
                f"{soft.shape}"
            )
        missing = np.isnan(soft)
        given = ~missing.any(axis=3)
        if np.any(missing.any(axis=3) & ~missing.all(axis=3)):
            raise ValueError(f"{path}: a cell's probabilities must be all given or all NaN")
        values = soft[given]  # (n_soft, K)
        if values.size == 0:
            raise ValueError(f"{path}: no cell has soft data")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{path}: soft probabilities must be in [0, 1]")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"{path}: each cell's soft probabilities must sum to 1")
        # A zero in an uncertain pdf can meet the training image's certainty in MPSlib's
        # conflict branch, which then returns p_1 as the code: an integer, a facies of
        # positive soft probability, and wrong (three or more facies; review 2026-10-04).
        uncertain_with_zero = (values == 0).any(axis=1) & (values.max(axis=1) < 1)
        if uncertain_with_zero.any():
            raise ValueError(
                f"{path}: {int(uncertain_with_zero.sum())} cells have a zero probability in "
                "an uncertain pdf; give every facies a positive probability unless one is "
                "certain (MPSlib's conflict branch misreads such cells)"
            )
        return soft

    def generate_joint(
        self,
        variables: Sequence[StateVariable],
        n_ensemble: int | None,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[np.float64]]:
        """Return each facies' signed distances, ``{name: (n_cells, n_ensemble)}``.

        Raises
        ------
        ValueError
            If the variables are not one per facies on one grid, ``n_ensemble`` is missing,
            or a conditioning cell is off the grid.
        RuntimeError
            If MPSlib fails or does not honour the hard data.
        """
        if n_ensemble is None:
            raise ValueError("mpslib_snesim needs [prior] ensemble_size")
        if len(variables) != self.n_facies:
            raise ValueError(
                f"the training image has {self.n_facies} facies; the group draws "
                f"{len(variables)} variables (one distance field per facies)"
            )
        grids = {v.grid for v in variables}
        if len(grids) != 1 or None in grids:
            raise ValueError("the distance fields must all be grid fields of the same [nx, ny, nz]")
        grid = tuple(int(n) for n in next(iter(grids)))  # type: ignore[union-attr]
        if self.soft is not None and self.soft.shape[:3] != grid:
            raise ValueError(
                f"the soft probabilities' grid {list(self.soft.shape[:3])} is not the "
                f"variables' {list(grid)}"
            )
        if self.cells is not None:
            for cell in self.cells:
                if not all(1 <= c <= n for c, n in zip(cell, grid, strict=True)):
                    raise ValueError(
                        f"conditioning cell {cell.tolist()} is outside the grid {grid}"
                    )
        # One run for the ensemble (the search tree is built once), seeded from the stream;
        # MPSlib's seed 0 means "from the clock", so it is never drawn.
        if self.soft is not None:
            self._check_soft_support()
        seed = int(rng.integers(1, 2**31 - 1))
        realisations = self._simulate(grid, n_ensemble, seed)
        blocks = [np.empty((math.prod(grid), n_ensemble)) for _ in range(self.n_facies)]
        for j, codes in enumerate(realisations):
            distances = signed_distances(
                codes, grid, self.n_facies, self.cell_size, self.distance_band
            )
            for k in range(self.n_facies):
                blocks[k][:, j] = distances[k]
        return {v.name: np.ascontiguousarray(b) for v, b in zip(variables, blocks, strict=True)}

    def _simulate(self, grid: tuple[int, ...], n_real: int, seed: int) -> list[NDArray[np.int64]]:
        """Run MPSlib; return each realisation's codes ``(n_cells,)``, Eclipse order."""
        with tempfile.TemporaryDirectory(prefix="apex_snesim_") as folder:
            work = Path(folder)
            write_gslib_grid(self.training_image, work / "ti.dat")
            hard = ""
            if self.cells is not None and self.observed is not None:
                rows = "\n".join(
                    self._coordinates([c - 1 for c in cell]) + f" {int(code)}"
                    for cell, code in zip(self.cells.tolist(), self.observed.tolist(), strict=True)
                )
                (work / "hard.dat").write_text(f"hard data\n4\nx\ny\nz\nfacies\n{rows}\n", "ascii")
                hard = "hard.dat"
            if self.soft is not None:
                self._write_soft(work / "soft.dat", self.soft)
            (work / "mps.txt").write_text(self._parameters(grid, n_real, seed, hard), "ascii")
            try:
                completed = subprocess.run(
                    [str(self.program), "mps.txt"], cwd=work, capture_output=True, text=True,
                    timeout=self.timeout_s, check=False,
                )  # fmt: skip
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"MPSlib exceeded {self.timeout_s} s") from error
            outputs = [work / f"ti.dat_sg_{i}.gslib" for i in range(n_real)]
            missing = [p.name for p in outputs if not p.exists()]
            if completed.returncode != 0 or missing:
                tail = (completed.stdout + completed.stderr)[-1000:]
                raise RuntimeError(
                    f"MPSlib failed (exit {completed.returncode}; missing {missing[:3]}):\n{tail}"
                )
            codes = [read_gslib_grid(p).ravel(order="F") for p in outputs]
        for j, values in enumerate(codes):
            # A fraction is MPSlib's conflict branch returning a probability as the code.
            if not np.all(values == np.round(values)):
                raise RuntimeError(
                    f"MPSlib returned non-integer facies codes in realisation {j}: soft data "
                    "that contradict the training image (MPSlib's conflict branch)"
                )
        result = [np.asarray(c, dtype=np.int64) for c in codes]
        for j, realisation in enumerate(result):
            # MPSlib writes -997799 for nodes it could not simulate.
            unsimulated = int(np.sum((realisation < 0) | (realisation >= self.n_facies)))
            if unsimulated:
                raise RuntimeError(
                    f"MPSlib left {unsimulated} cells of realisation {j} unsimulated (or "
                    f"outside the codes 0..{self.n_facies - 1})"
                )
        self._check_hard_data(result, grid)
        self._check_soft_data(result)
        return result

    def _coordinates(self, cell: Sequence[int]) -> str:
        """Return a 0-based cell's centre in MPSlib's world coordinates (origin 0).

        MPSlib finds the cell as ``(int)((x - 0) / dx)`` in float32: a corner can land
        just below its index for a fractional ``dx`` and move to the previous cell, while a
        centre is half a cell from either side (review of 2026-10-04). Full precision.
        """
        return " ".join(repr((int(c) + 0.5) * s) for c, s in zip(cell, self.cell_size, strict=True))

    def _check_soft_data(self, realisations: list[NDArray[np.int64]]) -> None:
        """Refuse a realisation with a facies whose soft probability is 0 at its cell."""
        if self.soft is None:
            return
        soft = self.soft.reshape(-1, self.n_facies, order="F")  # (n_cells, K), Eclipse order
        for j, codes in enumerate(realisations):
            impossible = np.flatnonzero(soft[np.arange(codes.size), codes] == 0)
            if impossible.size:
                raise RuntimeError(
                    f"MPSlib realisation {j} has facies of soft probability 0 at "
                    f"{impossible.size} cells: the soft data contradict the training image "
                    "there (MPSlib then returns a probability as the code); soften them"
                )

    def _check_soft_support(self) -> None:
        """Refuse a program that does not read every soft category.

        The ``mps_snesim_tree`` built into ``scikit-mps`` 0.6.0 parses one soft category
        only, so each cell's soft pdf becomes ``P(facies 0) = 1`` and the file's
        probabilities are ignored (found 2026-10-04; MPSlib's source, ``a47718f``, reads
        them all). A one-realisation run on a small grid at debug level 3 prints the soft
        pdf MPSlib uses; it must have one entry per facies. Checked once per prior.
        """
        if self._soft_checked:
            return
        k = self.n_facies
        probe = np.full((4, 4, 1, k), np.nan)
        expected = np.arange(1, k + 1) / (k * (k + 1) / 2)  # distinct probabilities
        probe[1, 1, 0] = expected
        with tempfile.TemporaryDirectory(prefix="apex_snesim_probe_") as folder:
            work = Path(folder)
            write_gslib_grid(self.training_image, work / "ti.dat")
            self._write_soft(work / "soft.dat", probe)
            parameters = self._parameters((4, 4, 1), 1, 1, "", debug=3)
            (work / "mps.txt").write_text(parameters, "ascii")
            try:
                completed = subprocess.run(
                    [str(self.program), "mps.txt"], cwd=work, capture_output=True, text=True,
                    timeout=self.timeout_s, check=False,
                )  # fmt: skip
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"MPSlib's soft-data check exceeded {self.timeout_s} s"
                ) from error
        if completed.returncode != 0:
            tail = (completed.stdout + completed.stderr)[-1000:]
            raise RuntimeError(
                f"MPSlib's soft-data check failed (exit {completed.returncode}):\n{tail}"
            )
        # MPSlib's debug lines: "SOFT:  P[0]=0.166667 P[1]=0.333333 P[2]=0.5".
        used = [
            [float(v) for v in re.findall(r"P\[[^\]]*\]=([-+0-9.eE]+)", line)]
            for line in completed.stdout.splitlines()
            if line.startswith("SOFT:")
        ]
        if not used or any(
            len(pdf) != k or not np.allclose(pdf, expected, atol=1e-5) for pdf in used
        ):
            raise RuntimeError(
                f"{self.program} does not use the soft data's {k} facies probabilities "
                "(the mps_snesim_tree of scikit-mps 0.6.0 reads one and ignores the rest; "
                "the check reads MPSlib's debug output, 'SOFT: P[k]=...'): build MPSlib "
                "from source (github.com/ergosimulation/mpslib, `make CXX=g++`) and give "
                "its mps_snesim_tree as program"
            )
        self._soft_checked = True

    def _write_soft(self, path: Path, soft: NDArray[np.float64]) -> None:
        """Write soft data ``(nx, ny, nz, K)`` as MPSlib's point set: ``x y z p_0 .. p_K-1``."""
        given = np.argwhere(~np.isnan(soft[..., 0]))  # (n_soft, 3) 0-based i, j, k
        names = "\n".join(f"p{k}" for k in range(self.n_facies))
        # MPSlib takes the last category as 1 - sum(others): renormalise so that it is
        # the given one (an exact 0 stays 0, up to float32).
        pdfs = soft[tuple(given.T)]  # (n_soft, K)
        pdfs = pdfs / pdfs.sum(axis=1, keepdims=True)
        rows = "\n".join(
            self._coordinates(cell) + " " + " ".join(repr(float(p)) for p in pdf)
            for cell, pdf in zip(given.tolist(), pdfs, strict=True)
        )
        header = f"soft data\n{3 + self.n_facies}\nx\ny\nz\n{names}\n"
        path.write_text(header + rows + "\n", "ascii")

    def _check_hard_data(
        self, realisations: list[NDArray[np.int64]], grid: tuple[int, ...]
    ) -> None:
        if self.cells is None or self.observed is None:
            return
        nx, ny, _ = grid
        flat = (self.cells[:, 0] - 1) + nx * ((self.cells[:, 1] - 1) + ny * (self.cells[:, 2] - 1))
        for j, codes in enumerate(realisations):
            wrong = np.flatnonzero(codes[flat] != self.observed)
            if wrong.size:
                raise RuntimeError(
                    f"MPSlib realisation {j} does not honour the facies at cells "
                    f"{self.cells[wrong].tolist()[:5]}"
                )

    def _parameters(
        self, grid: tuple[int, ...], n_real: int, seed: int, hard: str, debug: int = -1
    ) -> str:
        """Return MPSlib's SNESIM parameter file (the format of ``mps_snesim_par_write``)."""
        tx, ty, tz = self.template_size
        dx, dy, dz = self.cell_size
        categories = ";".join(str(k) for k in range(self.n_facies))
        lines = [
            f"Number of realizations # {n_real}",
            f"Random Seed (0 `random` seed) # {seed}",
            f"Number of mulitple grids (start from 0) # {self.n_multiple_grids}",
            f"Min Node count (0 if not set any limit)# {self.n_min_node_count}",
            f"Maximum number condtitional data (0: all) # {self.n_cond}",
            f"Search template size X # {tx} {tx}",
            f"Search template size Y # {ty} {ty}",
            f"Search template size Z # {tz} {tz}",
            f"Simulation grid size X # {grid[0]}",
            f"Simulation grid size Y # {grid[1]}",
            f"Simulation grid size Z # {grid[2]}",
            "Simulation grid world/origin X # 0",
            "Simulation grid world/origin Y # 0",
            "Simulation grid world/origin Z # 0",
            f"Simulation grid grid cell size X # {dx!r}",
            f"Simulation grid grid cell size Y # {dy!r}",
            f"Simulation grid grid cell size Z # {dz!r}",
            "Training image file (spaces not allowed) # ti.dat",
            "Output folder (spaces in name not allowed) # .",
            "Shuffle Simulation Grid path (0: sequential, 1: random, 2: preferential, EF) # "
            f"{self.path_kind} {self.entropy_factor:g}",
            "Shuffle Training Image path (1 : random, 0 : sequential) # 1",
            f"HardData filename  (same size as the simulation grid)# {hard or 'hard.dat'}",
            f"HardData seach radius (world units) # {max(dx, dy, dz):g}",
            f"Softdata categories (separated by ;) # {categories}",
            "Soft datafilenames (separated by ; only need (number_categories - 1) grids) "
            "# soft.dat",
            "Number of threads (not currently used) # 1",
            f"Debug mode(2: write to file, 1: show preview, 0: show counters, -1: no ) # {debug}",
            "Mask grid filename (same size as the simulation grid)# mask.dat",
            "do Entropy # 0",
            "do Estimation# 0",
        ]
        return "\n".join(lines) + "\n"
