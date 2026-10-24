"""Prior (initial-ensemble) generators that sample in memory.

``gaussian`` draws independent normals; ``gaussian_field`` is PET's variogram-based
Gaussian field (Cholesky per grid layer, see :mod:`apex.algorithms.geostatistics` for
provenance and differences); ``geostatspy_sgsim`` wraps GeostatsPy's sequential Gaussian
simulation. Stored ensembles are read by ``storage.priors`` (``file``), since reading files
is not an algorithm's job.
"""

import contextlib
import io
import math
from collections.abc import Sequence
from typing import Annotated

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.geostatistics import correlated_sample, covariance_factor, layer_covariance
from apex.core.options import TEXT
from apex.core.plugins import priors
from apex.domain.state import StateVariable

PerLayer = float | Sequence[float]


def _need_size(n_ensemble: int | None) -> int:
    if n_ensemble is None or n_ensemble < 2:
        raise ValueError(f"this prior needs [prior] ensemble_size >= 2, got {n_ensemble}")
    return n_ensemble


def _per_layer(value: PerLayer | Sequence[str] | str, nz: int, name: str) -> list:  # type: ignore[type-arg]
    values = [value] if isinstance(value, (int, float, str)) else list(value)
    if len(values) == 1:
        return values * nz
    if len(values) != nz:
        raise ValueError(f"{name} has {len(values)} entries; give 1 or one per layer ({nz})")
    return values


@priors.register("gaussian")
class IidGaussianPrior:
    """Independent normals ``N(mean, std^2)`` for every entry (no spatial correlation).

    Parameters
    ----------
    mean
        Scalar, or one value per state entry.
    std
        Standard deviation, positive.
    """

    def __init__(self, mean: ArrayLike = 0.0, std: float = 1.0) -> None:
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = float(std)

    def generate(
        self, variable: StateVariable, n_ensemble: int | None, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw ``(variable.size, n_ensemble)`` independent normals."""
        size = _need_size(n_ensemble)
        mean = np.broadcast_to(self.mean, (variable.size,)) if self.mean.ndim == 0 else self.mean
        if mean.shape != (variable.size,):
            raise ValueError(f"mean must be a scalar or have {variable.size} entries")
        draws: NDArray[np.float64] = mean[:, None] + self.std * rng.standard_normal(
            (variable.size, size)
        )
        return draws


@priors.register("gaussian_field")
class GaussianFieldPrior:
    """Gaussian random field from a variogram, sampled layer by layer (PET's prior sampler).

    Each layer of the ``[nx, ny, nz]`` grid is an independent 2-D field with covariance
    from the variogram (as PET's ``StateLayout.from_prior_info``), sampled as
    ``mean + L z`` with the Cholesky factor ``L``. Cells are in Eclipse order.

    Cost: the dense layer covariance takes ``8 (nx ny)^2`` bytes and its factorisation
    ``O((nx ny)^3)`` time; at 10^4 cells per layer each dense matrix is 0.8 GB and building
    it peaks at several GB, so larger layers need another generator (e.g.
    ``geostatspy_sgsim``).
    Layers with identical variogram parameters share one factor.

    Parameters
    ----------
    grid
        ``[nx, ny, nz]``; ``nx * ny * nz`` must equal the variable's size.
    mean
        Scalar, or one value per cell (e.g. PET's ``priormean.npz``, loaded by the
        composition from ``mean_file``).
    variance
        Sill, scalar or per layer.
    major_range, minor_range
        Ranges in cells along and across the major axis, scalar or per layer;
        ``minor_range`` defaults to ``major_range``. A range of 0 means uncorrelated.
    azimuth
        Direction of the major axis, degrees counter-clockwise from the i-axis, scalar
        or per layer.
    variogram
        One of ``spherical``, ``exponential``, ``gaussian``, ``cubic``, ``stable``, scalar
        or per layer. PET's ``sph``/``cub`` are ``spherical``/``cubic``; PET's ``exp`` is
        ``stable`` with ``exponent = 1.9``.
    exponent
        Exponent of the ``stable`` model.
    """

    def __init__(
        self,
        grid: Sequence[int],
        mean: ArrayLike = 0.0,
        variance: PerLayer = 1.0,
        major_range: PerLayer = 10.0,
        minor_range: PerLayer | None = None,
        azimuth: PerLayer = 0.0,
        variogram: Annotated[str | Sequence[str], TEXT] = "spherical",
        exponent: PerLayer = 1.0,
    ) -> None:
        self.grid = tuple(int(n) for n in grid)
        if len(self.grid) != 3 or min(self.grid) < 1:
            raise ValueError(f"grid must be [nx, ny, nz] with positive sizes, got {grid}")
        nz = self.grid[2]
        self.mean = np.asarray(mean, dtype=np.float64)
        self.variance = [float(v) for v in _per_layer(variance, nz, "variance")]
        self.major_range = [float(v) for v in _per_layer(major_range, nz, "major_range")]
        self.minor_range = (
            self.major_range
            if minor_range is None
            else [float(v) for v in _per_layer(minor_range, nz, "minor_range")]
        )
        self.azimuth = [float(v) for v in _per_layer(azimuth, nz, "azimuth")]
        self.variogram = [str(v) for v in _per_layer(variogram, nz, "variogram")]
        self.exponent = [float(v) for v in _per_layer(exponent, nz, "exponent")]

    def generate(
        self, variable: StateVariable, n_ensemble: int | None, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw ``(variable.size, n_ensemble)`` realisations, layer by layer."""
        size = _need_size(n_ensemble)
        nx, ny, nz = self.grid
        cells = nx * ny
        if variable.size != cells * nz:
            raise ValueError(
                f"grid {list(self.grid)} has {cells * nz} cells but {variable.name!r} has "
                f"{variable.size} entries"
            )
        mean = np.broadcast_to(self.mean, (variable.size,)) if self.mean.ndim == 0 else self.mean
        if mean.shape != (variable.size,):
            raise ValueError(f"mean must be a scalar or have {variable.size} entries")
        layers = []
        # Layers with the same variogram share one factor (built once, O(cells^3)).
        factors: dict[tuple[str, float, float, float, float, float], NDArray[np.float64]] = {}
        for k in range(nz):
            params = (
                self.variogram[k],
                self.variance[k],
                self.major_range[k],
                self.minor_range[k],
                self.azimuth[k],
                self.exponent[k],
            )
            if params not in factors:
                model, sill, major, minor, azimuth, exponent = params
                factors[params] = covariance_factor(
                    layer_covariance(
                        nx,
                        ny,
                        model=model,
                        sill=sill,
                        major_range=major,
                        minor_range=minor,
                        azimuth=azimuth,
                        exponent=exponent,
                    )
                )
            # One draw per layer, in layer order (as PET), so injected normals line up.
            z = rng.standard_normal((cells, size))
            layers.append(correlated_sample(mean[k * cells : (k + 1) * cells], factors[params], z))
        return np.vstack(layers)


#: GeostatsPy variogram structure codes.
_GSLIB_MODELS = {"spherical": 1, "exponential": 2, "gaussian": 3}


@priors.register("geostatspy_sgsim")
class GeostatsPySgsimPrior:
    """Unconditional sequential Gaussian simulation with GeostatsPy (``geostatspy`` extra).

    Each layer is simulated independently with ``geostats.sgsim`` (GSLIB's SGSIM),
    standardised (``itrans = 0``), then scaled to ``mean + sqrt(variance) * field``.
    Parameters are as for :class:`GaussianFieldPrior`, with ``variogram`` one of
    ``spherical``, ``exponential``, ``gaussian``, ranges positive with
    ``major_range >= minor_range`` (GSLIB's convention), and ``max_neighbours`` the number
    of previously simulated nodes SGSIM conditions on (search radius: the major range).

    Randomness: GeostatsPy seeds NumPy's global generator from its ``seed`` argument. Each
    layer's seed is drawn from the run's prior stream, and the global generator's state is
    restored afterwards, so runs stay reproducible and nothing else is affected.
    GeostatsPy's progress output is suppressed. Unconditional simulation uses a single
    dummy datum far outside the grid, because GeostatsPy needs at least one datum.
    """

    def __init__(
        self,
        grid: Sequence[int],
        mean: ArrayLike = 0.0,
        variance: PerLayer = 1.0,
        major_range: PerLayer = 10.0,
        minor_range: PerLayer | None = None,
        azimuth: PerLayer = 0.0,
        variogram: Annotated[str | Sequence[str], TEXT] = "spherical",
        max_neighbours: int = 20,
    ) -> None:
        self.field = GaussianFieldPrior(
            grid, mean, variance, major_range, minor_range, azimuth, variogram
        )
        unknown = sorted(set(self.field.variogram) - set(_GSLIB_MODELS))
        if unknown:
            raise ValueError(f"GeostatsPy supports {sorted(_GSLIB_MODELS)}, not {unknown}")
        for major, minor in zip(self.field.major_range, self.field.minor_range, strict=True):
            if not major >= minor > 0:
                raise ValueError(
                    f"GeostatsPy needs major_range >= minor_range > 0, got {major}, {minor}"
                )
        if max_neighbours < 1:
            raise ValueError(f"max_neighbours must be positive, got {max_neighbours}")
        self.max_neighbours = max_neighbours

    def generate(
        self, variable: StateVariable, n_ensemble: int | None, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw ``(variable.size, n_ensemble)`` SGSIM realisations, layer by layer."""
        import geostatspy.geostats as gs  # optional dependency
        import pandas as pd

        size = _need_size(n_ensemble)
        field = self.field
        nx, ny, nz = field.grid
        cells = nx * ny
        if variable.size != cells * nz:
            raise ValueError(
                f"grid {list(field.grid)} has {cells * nz} cells but {variable.name!r} has "
                f"{variable.size} entries"
            )
        mean = np.broadcast_to(field.mean, (variable.size,)) if field.mean.ndim == 0 else field.mean
        dummy = pd.DataFrame({"X": [1e7], "Y": [1e7], "V": [0.0]})
        layers = []
        # GeostatsPy seeds the global generator; save and restore it around the calls.
        saved_state = np.random.get_state()  # noqa: NPY002
        try:
            for k in range(nz):
                vario = {
                    "nug": 0.0,
                    "nst": 1,
                    "it1": _GSLIB_MODELS[field.variogram[k]],
                    "cc1": 1.0,
                    # GSLIB azimuth: degrees clockwise from north (+j); APEX: counter-clockwise
                    # from +i.
                    "azi1": 90.0 - field.azimuth[k],
                    "hmaj1": field.major_range[k],
                    "hmin1": field.minor_range[k],
                    "it2": 1,
                    "cc2": 0.0,
                    "azi2": 0.0,
                    "hmaj2": 0.0,
                    "hmin2": 0.0,
                }
                seed = int(rng.integers(1, 2**31 - 1))
                with contextlib.redirect_stdout(io.StringIO()):
                    sims = gs.sgsim(
                        dummy, "X", "Y", "V", wcol=-1, scol=-1, tmin=-1e21, tmax=1e21,
                        itrans=0, ismooth=0, dftrans=0, tcol=0, twtcol=0, zmin=-10.0,
                        zmax=10.0, ltail=1, ltpar=0, utail=1, utpar=0, nsim=size, nx=nx,
                        xmn=0.5, xsiz=1.0, ny=ny, ymn=0.5, ysiz=1.0, seed=seed, ndmin=0,
                        ndmax=self.max_neighbours, nodmax=self.max_neighbours, mults=0,
                        nmult=2, noct=-1, ktype=0, colocorr=0.0, sec_map=0, vario=vario,
                    )  # fmt: skip
                # sgsim returns (nsim, ny, nx) with row 0 the *top* row (largest y); flip
                # to j increasing before the Eclipse (i fastest) reshape.
                standard = np.asarray(sims, dtype=np.float64)[:, ::-1, :].reshape(size, cells).T
                layer_mean = mean[k * cells : (k + 1) * cells]
                layers.append(layer_mean[:, None] + math.sqrt(field.variance[k]) * standard)
        finally:
            np.random.set_state(saved_state)  # noqa: NPY002
        return np.vstack(layers)
