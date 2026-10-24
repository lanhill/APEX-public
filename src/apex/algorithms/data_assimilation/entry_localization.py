"""Upstream PET's distance-based localization: one taper per (data type, time, parameter).

Ported from PET
---------------
PET module : src/pipt/localization/distance_localization.py (DistanceLocalization,
             GaspariCohnKernel, FurrerBengtssonKernel, RegionKernel, _resolve_mask,
             _place_kernel, _zero_mask) and its use in
             src/pipt/update_schemes/analysis/approx.py (``distance_loc`` branch)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Gaspari, G. & Cohn, S. E. (1999). Construction of correlation functions in two
             and three dimensions. Q. J. R. Meteorol. Soc., 125, 723-757. Furrer, R. &
             Bengtsson, T. (2007). Estimation of high-dimensional prior and posterior
             covariance matrices in Kalman filter variants. J. Multivariate Anal., 98,
             227-255.
Differences: see :class:`EntryDistanceLocalization`.

Each entry localizes the data it names (``data``, ``time``, each a name or ``"*"``) for the
state variable it names (``parameter``) with a kernel around one grid cell. A datum and a
variable that no entry pairs get a taper of 0: that variable is not updated by that datum
(PET's zero mask). The taper ``T`` ``(n_state, n_data)`` multiplies the gain, as for the
other distance localization: ``(T o (dX G)) r`` (PET: ``T.multiply(A @ X) @ D_anom``).
"""

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.localization import _GainTaper, gaspari_cohn
from apex.algorithms.data_assimilation.registries import localizations
from apex.core.options import INPUT, TEXT, InputFile, mapping
from apex.domain.observations import DataKey

logger = logging.getLogger(__name__)

Kernel = Literal["gaspari_cohn", "furrer_bengtsson", "region", "import"]
_KERNELS = ("gaspari_cohn", "furrer_bengtsson", "region", "import")


class Entry(TypedDict, total=False):
    """One localization entry (PET: one row of its entries CSV, or one dict of ``entries``).

    ``data``, ``time`` and ``parameter`` name what it localizes (``"*"``: all, the default);
    ``taper`` is the kernel. ``position`` ``[i, j, k]`` is the centre cell, 1-based as in
    Eclipse decks; ``radius`` the kernel's length in cells; ``z_range`` ``"all"`` (PET
    ``":"``, the default) puts the kernel in every layer, an integer ``r`` in layers
    ``k - r .. k + r``; ``anisotropy`` is ``[ratio, angle in degrees]``: the rotated offset's
    first (``j``) component is divided by the ratio, so a ratio of 2 doubles the reach along
    ``j`` (PET's docstring says it halves it; its code divides, as here). The ``import``
    kernel reads its taper from ``taper_file`` instead (``.npy``/``.npz``: one value per
    grid cell, flat in Eclipse order or shaped ``(nx, ny, nz)``; relative to the
    configuration's folder).
    """

    data: str
    time: str
    parameter: str
    taper: Kernel
    position: Sequence[int]
    radius: float
    z_range: str | int
    anisotropy: Sequence[float]
    taper_file: str


#: What the entries' strings name: an ``import`` entry's file is read.
ENTRY_ROLES = mapping(
    data=TEXT, time=TEXT, parameter=TEXT, taper=TEXT, z_range=TEXT, taper_file=INPUT
)


def planar_taper(
    shape: tuple[int, int],
    centre: tuple[int, int],
    kernel: Kernel,
    radius: float,
    anisotropy: tuple[float, float],
    n_ensemble: int,
) -> NDArray[np.float64]:
    """Return a kernel's taper over one layer, ``(ny, nx)``, around the 0-based ``centre``.

    ``centre`` is ``(i, j)``. The offset ``(dj, di)`` is rotated by the angle and its first
    component divided by the ratio (PET's ``_build_transform`` on its ``(x, y)`` axes, which
    are Eclipse's ``(j, i)`` for a state in Eclipse order). ``gaspari_cohn`` vanishes beyond
    ``2 radius``; ``furrer_bengtsson`` turns the spherical correlation ``rho`` (zero beyond
    ``radius``) into ``N rho^2 / ((N + 1) rho^2 + 1)``; ``region`` is 1 at the centre cell.
    """
    ny, nx = shape
    ci, cj = centre
    taper = np.zeros((ny, nx))
    if kernel == "region":
        taper[cj, ci] = 1.0
        return taper
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ratio, angle = anisotropy
    theta = math.radians(angle)
    dj, di = j - cj, i - ci
    first = (math.cos(theta) * dj + math.sin(theta) * di) / ratio
    second = -math.sin(theta) * dj + math.cos(theta) * di
    distance = np.hypot(first, second)
    if kernel == "gaspari_cohn":
        return gaspari_cohn(distance / radius)
    if kernel == "furrer_bengtsson":
        d = distance / radius
        rho = np.where(distance < radius, 1 - (1.5 * d - 0.5 * d**3), 0.0)
        result: NDArray[np.float64] = n_ensemble * rho**2 / (rho**2 * (n_ensemble + 1) + 1)
        return result
    raise ValueError(f"kernel {kernel!r} has no planar form")


@localizations.register("distance_entries")
class EntryDistanceLocalization(_GainTaper):
    """Upstream PET's distance localization (``distance_loc``), one entry per data and variable.

    Parameters
    ----------
    grid
        ``[nx, ny, nz]``; each state variable named by an entry is a grid property in
        Eclipse order (``i`` fastest) over all cells, or over the active cells of
        ``actnum_file``.
    entries
        The entries (see :class:`Entry`), in order: a later entry replaces an earlier one
        for the (data, time, variable) it names, as in PET.
    variable_names, block_sizes
        The state variables' names and sizes (set from the state layout).
    actnum_file
        Optional ``.npy``/``.npz`` of active cells (one boolean per grid cell, Eclipse
        order); the tapers are then taken over the active cells (PET ``actnum``).

    Differences from PET: positions are 1-based Eclipse ``(i, j, k)`` (PET: 0-based along
    its ``[nz, nx, ny]`` field, so its ``x`` is Eclipse ``j`` and its ``y`` is ``i``); the
    kernel is evaluated directly at each cell's offset instead of from a cropped and
    re-centred stencil (the same values, except where PET's crop is asymmetric and shifts the
    kernel by a cell: a rotated, elongated kernel reaching past the stencil on one axis; see
    docs/migration/pet_parity.md); the radius is a float (PET truncates
    it to an integer); Furrer-Bengtsson uses the actual ensemble size; an entry naming data,
    a time or a variable that does not exist is an error (PET ignores it), and data no entry
    localizes are logged; the taper itself is checkpointed (a resume does not re-read
    ``taper_file``); the ``import`` kernel works (PET's ``_resolve_mask`` reads the
    entry's positions, which are ``None`` for it, and fails).
    """

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "taper": "per-entry distance gain taper, (n_state, n_data); dense"
    }
    #: Upstream PET localizes LM-EnRML's and GN-EnRML's approx update this way.
    enrml_localization: ClassVar[bool] = True

    def __init__(
        self,
        grid: Sequence[int],
        entries: Annotated[Sequence[Entry], ENTRY_ROLES],
        variable_names: Sequence[str],
        block_sizes: Sequence[int],
        actnum_file: InputFile | None = None,
    ) -> None:
        sizes = tuple(int(n) for n in grid)
        if len(sizes) != 3 or min(sizes) < 1:
            raise ValueError(f"grid must be [nx, ny, nz] with positive sizes, got {grid}")
        self.grid: tuple[int, int, int] = (sizes[0], sizes[1], sizes[2])
        if len(variable_names) != len(block_sizes):
            raise ValueError("variable_names and block_sizes must have one entry per variable")
        self.variable_names = tuple(str(name) for name in variable_names)
        self.block_sizes = tuple(int(size) for size in block_sizes)
        cells = math.prod(self.grid)
        self.active: NDArray[np.bool_] | None = None
        if actnum_file is not None:
            self.active = _load_actnum(Path(actnum_file), self.grid)
        self.n_cells = cells if self.active is None else int(self.active.sum())
        if not entries:
            raise ValueError("distance_entries needs at least one entry")
        self.entries = [self._checked(dict(entry), n) for n, entry in enumerate(entries)]
        lowered = {name.lower(): name for name in self.variable_names}
        for n, entry in enumerate(self.entries):
            name = str(entry["parameter"])
            if name == "*":
                targets = list(self.variable_names)
            elif name.lower() in lowered:
                targets = [lowered[name.lower()]]
            else:
                raise ValueError(
                    f"entry {n}: no state variable {name!r}; the state has "
                    f"{', '.join(self.variable_names)}"
                )
            for target in targets:
                size = self.block_sizes[self.variable_names.index(target)]
                if size != self.n_cells:
                    raise ValueError(
                        f"entry {n}: variable {target!r} has {size} entries, but the grid has "
                        f"{self.n_cells} {'active ' if self.active is not None else ''}cells"
                    )

    def _checked(self, entry: dict[str, object], n: int) -> dict[str, object]:
        unknown = set(entry) - set(Entry.__annotations__)
        if unknown:
            raise ValueError(f"entry {n}: unknown keys {sorted(unknown)}")
        kernel = entry.get("taper")
        if kernel not in _KERNELS:
            raise ValueError(
                f"entry {n}: taper must be one of {', '.join(_KERNELS)}, got {kernel!r}"
            )
        for key in ("data", "time", "parameter"):
            entry[key] = str(entry.get(key, "*"))
        nx, ny, nz = self.grid
        if kernel == "import":
            if "taper_file" not in entry:
                raise ValueError(f"entry {n}: the import taper needs a taper_file")
            values = _grid_values(
                np.asarray(_load_array(Path(str(entry["taper_file"])), "taper"), dtype=np.float64),
                self.grid,
                f"entry {n}: {entry['taper_file']}",
            )
            entry["values"] = values
            return entry
        position = [int(v) for v in entry.get("position", ())]  # type: ignore[attr-defined]
        if len(position) != 3 or not (
            1 <= position[0] <= nx and 1 <= position[1] <= ny and 1 <= position[2] <= nz
        ):
            raise ValueError(f"entry {n}: position must be [i, j, k] inside the grid {self.grid}")
        entry["position"] = position
        if kernel != "region":
            radius = float(entry.get("radius", 0))  # type: ignore[arg-type]
            if not radius > 0:
                raise ValueError(f"entry {n}: radius must be positive")
            entry["radius"] = radius
        z_range = entry.get("z_range", "all")
        if z_range in ("all", ":"):
            entry["z_range"] = None
        else:
            try:
                levels = int(z_range)  # type: ignore[call-overload]
            except (TypeError, ValueError) as error:
                raise ValueError(f"entry {n}: z_range must be 'all' or a layer count") from error
            if levels < 0:
                raise ValueError(f"entry {n}: z_range must be non-negative")
            entry["z_range"] = levels
        ratio, angle = (float(v) for v in entry.get("anisotropy", (1.0, 0.0)))  # type: ignore[attr-defined]
        if ratio <= 0:
            raise ValueError(f"entry {n}: anisotropy ratio must be positive")
        entry["anisotropy"] = (ratio, angle)
        return entry

    def prepare(
        self,
        prior_state: NDArray[np.float64],
        prior_predicted: NDArray[np.float64],
        keys: Sequence[DataKey],
        rng: np.random.Generator,
    ) -> None:
        """Build the taper ``(n_state, n_data)`` from the entries and the data keys.

        Raises
        ------
        ValueError
            If an entry names data or a time that the observations do not have: a misspelt
            name would otherwise silently leave its data out of the update.
        """
        datatypes = {k.datatype.lower() for k in keys}
        times = {k.time for k in keys}
        for n, entry in enumerate(self.entries):
            if entry["data"] != "*" and str(entry["data"]).lower() not in datatypes:
                raise ValueError(
                    f"entry {n}: no assimilated data {entry['data']!r} (misspelt, or held "
                    "out for cross validation?)"
                )
            if entry["time"] != "*" and entry["time"] not in times:
                raise ValueError(f"entry {n}: no assimilated data at time {entry['time']!r}")
        self._build(list(keys), prior_state.shape[-1])

    def _build(self, keys: list[DataKey], n_ensemble: int) -> None:
        cache: dict[int, NDArray[np.float64]] = {}
        blocks = []
        for name, size in zip(self.variable_names, self.block_sizes, strict=True):
            block = np.zeros((size, len(keys)))
            for column, key in enumerate(keys):
                n = self._entry_for(key, name)
                if n is not None:
                    if n not in cache:
                        cache[n] = self.entry_taper(self.entries[n], n_ensemble)
                    block[:, column] = cache[n]
            blocks.append(block)
        self.taper_matrix = np.vstack(blocks)
        unlocalized = sorted(
            {str(k) for k, c in zip(keys, self.taper_matrix.T, strict=True) if not c.any()}
        )
        if unlocalized:
            logger.warning(
                "distance_entries localization: %d data have no entry with a nonzero taper "
                "and do not update the state: %s",
                len(unlocalized),
                ", ".join(unlocalized[:10]) + (" ..." if len(unlocalized) > 10 else ""),
            )

    def _entry_for(self, key: DataKey, variable: str) -> int | None:
        """Return the last entry naming ``key`` and ``variable`` (PET's overwrite order)."""
        found = None
        for n, entry in enumerate(self.entries):
            if (
                (
                    entry["data"] in ("*", key.datatype)
                    or str(entry["data"]).lower() == key.datatype.lower()
                )
                and (entry["time"] in ("*", key.time))
                and (
                    entry["parameter"] == "*" or str(entry["parameter"]).lower() == variable.lower()
                )
            ):
                found = n
        return found

    def entry_taper(self, entry: Mapping[str, object], n_ensemble: int) -> NDArray[np.float64]:
        """Return one entry's taper over the grid (or its active cells), Eclipse order."""
        nx, ny, nz = self.grid
        if entry["taper"] == "import":
            values = np.asarray(entry["values"], dtype=np.float64)
        else:
            i, j, k = (int(v) - 1 for v in entry["position"])  # type: ignore[attr-defined]
            layer = planar_taper(
                (ny, nx),
                (i, j),
                entry["taper"],  # type: ignore[arg-type]
                float(entry.get("radius", 1.0)),  # type: ignore[arg-type]
                entry["anisotropy"],  # type: ignore[arg-type]
                n_ensemble,
            )
            z_range = entry["z_range"]
            if z_range is None:
                layers = range(nz)
            else:
                levels = int(z_range)  # type: ignore[call-overload]
                layers = range(max(0, k - levels), min(nz - 1, k + levels) + 1)
            grid = np.zeros((nz, ny, nx))
            for z in layers:
                grid[z] = layer
            values = grid.ravel()
        return values if self.active is None else values[self.active]


def _load_array(path: Path, key: str) -> NDArray[np.generic]:
    """Load a ``.npy``, or the array ``key`` of an ``.npz`` (its only array if it has one)."""
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if key in loaded.files:
                return np.asarray(loaded[key])
            if len(loaded.files) == 1:
                return np.asarray(loaded[loaded.files[0]])
            raise ValueError(f"{path} has arrays {loaded.files}; name the one to use {key!r}")
    return np.asarray(loaded)


def _grid_values(
    values: NDArray[np.generic], grid: tuple[int, int, int], what: str
) -> NDArray[np.generic]:
    """Return per-cell values in Eclipse order from a flat array or an ``(nx, ny, nz)`` one.

    Any other shape is refused: an array in another layout (PET's ``(nz, nx, ny)``, or a
    2-D ``(nx, ny)``) has the right size but would be read in the wrong cell order.
    """
    cells = math.prod(grid)
    if values.ndim == 1 and values.size == cells:
        return values
    if values.shape == grid:
        return values.ravel(order="F")  # i fastest
    raise ValueError(
        f"{what} has shape {values.shape}; give one value per cell, flat in Eclipse order "
        f"({cells},) or shaped (nx, ny, nz) = {grid}"
    )


def _load_actnum(path: Path, grid: tuple[int, int, int]) -> NDArray[np.bool_]:
    return _grid_values(np.asarray(_load_array(path, "actnum")), grid, str(path)).astype(bool)
