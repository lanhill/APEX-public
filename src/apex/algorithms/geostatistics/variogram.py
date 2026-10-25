"""Variogram estimation from hard data: Matheron's estimator and a weighted least-squares fit.

The plugin ``wls`` of ``variogram_estimators``, a method PET lacks:

1. **The experimental semivariogram** (Matheron 1962), per direction:
   ``gamma(h_k) = 1 / (2 N_k) * sum over the N_k pairs in lag class k of (z_i - z_j)^2``,
   with lag classes ``[edge_k, edge_{k+1})``. Only pairs in the same layer count: the
   priors APEX draws are fields per layer (``gaussian_field``, SGSIM), so their variogram is
   horizontal. A direction keeps the pairs whose separation lies within ``angle_tolerance``
   of the azimuth or its opposite (and within ``bandwidth`` of its axis, if given).
2. **The fit** (Cressie 1985): ``spherical``, ``exponential`` or ``gaussian`` models
   ``gamma(h) = nugget + sill (1 - rho(h'))``, with ``rho`` APEX's covariance models
   (:func:`~apex.algorithms.geostatistics.covariance.variogram_covariance`, ranges
   practical) and ``h'`` the lag rescaled by geometric anisotropy (major and minor ranges, an
   azimuth). The weighted residual ``sum_k N_k (gamma_k / gamma(h_k) - 1)^2`` is minimised
   (scipy's ``least_squares``, from several starts); among models, the least residual wins
   unless the user fixes one. Classes with fewer than ``MIN_PAIRS`` pairs are left out
   (Journel & Huijbregts 1978's rule of thumb is 30).

Distances are in cells (the generators' unit), the azimuth in degrees counter-clockwise
from the i axis, as :func:`~apex.algorithms.geostatistics.covariance.layer_covariance`.

References
----------
- Matheron, G. (1962). Traité de géostatistique appliquée. Éditions Technip.
- Cressie, N. (1985). Fitting variogram models by weighted least squares. Mathematical
  Geology 17(5), 563-586.
- Journel, A. G. & Huijbregts, C. J. (1978). Mining Geostatistics. Academic Press.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from apex.algorithms.geostatistics.covariance import variogram_covariance
from apex.core.geomodeling import VariogramEstimate
from apex.core.plugins import variogram_estimators

#: The models fitted when none is fixed.
FIT_MODELS = ("spherical", "exponential", "gaussian")
#: Lag classes with fewer pairs are left out of the fit.
MIN_PAIRS = 30
#: The most data points: the pairs are formed in full (``n^2 / 2``), in chunks.
MAX_POINTS = 20_000
#: Pairs formed per chunk (memory: about 100 bytes each, for the indices, separations,
#: distances and masks; 0.25 GB at the peak).
CHUNK_PAIRS = 2_500_000
#: The directions of an anisotropic estimate (degrees), each with a tolerance of half
#: their spacing, so that every pair is in exactly one. Eight, not four: the fit takes each
#: sector's pairs to lie along its centre, and 45-degree sectors lengthened the major range
#: of a field at 30 degrees by half (16 cells read as 24, median of 10 seeds; 8: 18.6).
DIRECTIONS = tuple(22.5 * k for k in range(8))
#: The longest range fitted, in largest lags: beyond, the data cannot tell it.
MAX_RANGE_LAGS = 2.0


@dataclass(frozen=True)
class ExperimentalVariogram:
    """An experimental semivariogram: per lag class, its centre, value and pair count.

    ``lags``, ``gamma`` and ``counts`` are ``(n_classes,)``; a class without pairs has
    ``gamma`` NaN. ``azimuth`` is the direction (degrees), ``None`` for all directions.
    ``distances`` are the classes' mean pair distances (NaN when empty), which the fit
    uses: on a grid the distances are few (1, sqrt 2, 2, ...), and a class's centre can be
    a quarter off them. ``None``: the centres.
    """

    lags: NDArray[np.float64]
    gamma: NDArray[np.float64]
    counts: NDArray[np.int64]
    azimuth: float | None = None
    distances: NDArray[np.float64] | None = None


def experimental_variogram(
    coordinates: NDArray[np.float64],
    values: NDArray[np.float64],
    bin_edges: Sequence[float] | NDArray[np.float64],
    *,
    layers: NDArray[np.int64] | None = None,
    azimuth: float | None = None,
    angle_tolerance: float = 22.5,
    bandwidth: float | None = None,
) -> ExperimentalVariogram:
    """Return Matheron's semivariogram of ``values`` at ``coordinates`` (see the module).

    Parameters
    ----------
    coordinates
        ``(n, 2)``: each point's ``(i, j)`` position, in cells.
    values
        ``(n,)``: the values there.
    bin_edges
        ``(n_classes + 1,)``: increasing lag class edges, in cells; class ``k`` is
        ``[edge_k, edge_{k+1})``. The lags are the classes' centres.
    layers
        ``(n,)``: each point's layer; only pairs in one layer count (default: all one layer).
    azimuth
        The direction, degrees counter-clockwise from the i axis; ``None``: every direction.
    angle_tolerance
        Degrees either side of the direction (and of its opposite).
    bandwidth
        The largest distance from the direction's axis (cells); ``None``: no limit.

    Raises
    ------
    ValueError
        If the shapes disagree, there are more than ``MAX_POINTS`` points, or the edges do
        not increase.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    z = np.asarray(values, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or z.shape != (xy.shape[0],):
        raise ValueError(f"coordinates must be (n, 2) and values (n,), got {xy.shape} and "
                         f"{z.shape}")  # fmt: skip
    if xy.shape[0] > MAX_POINTS:
        raise ValueError(f"{xy.shape[0]} points: at most {MAX_POINTS} (thin the data)")
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be at least two increasing values")
    group = np.zeros(xy.shape[0], dtype=np.int64) if layers is None else np.asarray(layers)
    if group.shape != z.shape:
        raise ValueError(f"layers must be (n,), got {group.shape}")
    n_classes = edges.size - 1
    sums = np.zeros(n_classes)
    lengths = np.zeros(n_classes)
    counts = np.zeros(n_classes, dtype=np.int64)
    if azimuth is not None:
        theta = math.radians(azimuth)
        direction = np.array([math.cos(theta), math.sin(theta)])
        cos_tolerance = math.cos(math.radians(angle_tolerance))
    rows = max(1, CHUNK_PAIRS // max(1, xy.shape[0]))
    for start in range(0, xy.shape[0], rows):
        stop = min(start + rows, xy.shape[0])
        # Pairs (i, j) with i in the chunk and j > i: each pair once.
        i, j = np.nonzero(np.arange(start, stop)[:, None] < np.arange(xy.shape[0])[None, :])
        i = i + start
        keep = group[i] == group[j]
        i, j = i[keep], j[keep]
        separation = xy[j] - xy[i]
        distance = np.hypot(separation[:, 0], separation[:, 1])
        # Coincident points (distance 0) have no direction and no lag: left out in every
        # mode (GSTools keeps them in an omnidirectional first class; hard data have one
        # value a cell, so none arise there).
        keep = distance > 0
        if azimuth is not None:
            along = separation @ direction
            # Within the tolerance of the direction or of its opposite (|cos| >= cos tol).
            keep &= np.abs(along) >= cos_tolerance * distance
            if bandwidth is not None:
                across = np.abs(separation[:, 0] * direction[1] - separation[:, 1] * direction[0])
                keep &= across <= bandwidth
        i, j, distance = i[keep], j[keep], distance[keep]
        index = np.searchsorted(edges, distance, side="right") - 1
        inside = (index >= 0) & (index < n_classes)
        squared = (z[i[inside]] - z[j[inside]]) ** 2
        sums += np.bincount(index[inside], weights=squared, minlength=n_classes)
        lengths += np.bincount(index[inside], weights=distance[inside], minlength=n_classes)
        counts += np.bincount(index[inside], minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(counts > 0, sums / (2.0 * counts), np.nan)
        mean_distance = np.where(counts > 0, lengths / counts, np.nan)
    return ExperimentalVariogram(0.5 * (edges[:-1] + edges[1:]), gamma, counts, azimuth,
                                 mean_distance)  # fmt: skip


def model_variogram(
    lags: NDArray[np.float64],
    directions: NDArray[np.float64],
    model: str,
    nugget: float,
    sill: float,
    major_range: float,
    minor_range: float,
    azimuth: float,
) -> NDArray[np.float64]:
    """Return ``nugget + sill (1 - rho(h'))`` at lags ``(n,)`` in directions ``(n,)`` (degrees).

    ``h'`` rescales each lag by the major range along ``azimuth`` and the minor across it.
    """
    angle = np.radians(directions - azimuth)
    u = np.hypot(lags * np.cos(angle) / major_range, lags * np.sin(angle) / minor_range)
    return nugget + sill - variogram_covariance(u, model, sill)


@dataclass(frozen=True)
class VariogramFit:
    """A fitted variogram and its weighted residual (``sum N (gamma / model - 1)^2``)."""

    model: str
    nugget: float
    sill: float
    major_range: float
    minor_range: float
    azimuth: float
    residual: float


def fit_variogram(
    variograms: Sequence[ExperimentalVariogram],
    model: str,
    *,
    anisotropic: bool = False,
    max_range: float | None = None,
) -> VariogramFit:
    """Fit ``model`` to experimental variograms by Cressie's weighted least squares.

    ``variograms`` are one omnidirectional or several directional ones (their azimuths
    used); ``anisotropic`` fits a minor range and an azimuth too (directional ones only).
    Ranges are bounded by ``max_range`` (default: 4 times the largest lag).

    Raises
    ------
    ValueError
        If fewer than three lag classes have ``MIN_PAIRS`` pairs, or anisotropy is asked of
        an omnidirectional variogram.
    """
    lags, gammas, counts, directions = [], [], [], []
    for variogram in variograms:
        usable = (variogram.counts >= MIN_PAIRS) & np.isfinite(variogram.gamma)
        at = variogram.lags if variogram.distances is None else variogram.distances
        lags.append(at[usable])
        gammas.append(variogram.gamma[usable])
        counts.append(variogram.counts[usable])
        directions.append(np.full(int(usable.sum()), variogram.azimuth or 0.0))
    h, g, n, d = (np.concatenate(a) for a in (lags, gammas, counts, directions))
    if h.size < 3:
        raise ValueError(f"only {h.size} lag classes have {MIN_PAIRS} pairs or more: too few "
                         "data to fit a variogram")  # fmt: skip
    if anisotropic and any(v.azimuth is None for v in variograms):
        raise ValueError("an anisotropic fit needs directional variograms")
    if g.max() <= 0:
        raise ValueError("the values never differ within a layer: no horizontal variogram "
                         "(they vary only between layers)")  # fmt: skip
    top = float(max_range if max_range is not None else 4.0 * h.max())
    level = float(np.average(g, weights=n))
    weights = np.sqrt(n.astype(np.float64))

    def residuals(p: NDArray[np.float64]) -> NDArray[np.float64]:
        nugget, sill, major = p[0], p[1], p[2]
        minor, azimuth = (major * p[3], p[4]) if anisotropic else (major, 0.0)
        fitted = model_variogram(h, d, model, nugget, sill, major, minor, azimuth)
        # Cressie's weights N / gamma_model^2, as sqrt(N) (gamma / model - 1) residuals;
        # the floor keeps a zero model (no nugget, the first lag at 0) finite.
        return np.asarray(weights * (g / np.maximum(fitted, 1e-12 * level) - 1.0))

    lower = [0.0, 1e-9 * level, 1e-3]
    upper = [2.0 * g.max(), 4.0 * g.max(), top]
    starts_range = [h.max() / 4, h.max() / 2, h.max()]
    starts_angle: Sequence[float] = [0.0]
    if anisotropic:
        lower += [0.05, -90.0]
        upper += [1.0, 90.0]
        starts_angle = sorted({((v.azimuth or 0.0) + 90.0) % 180.0 - 90.0 for v in variograms})
    best: scipy.optimize.OptimizeResult | None = None
    for a in starts_range:
        for angle in starts_angle:
            start = [0.05 * level, level, min(a, top)]
            if anisotropic:
                start += [0.5, angle]
            found = scipy.optimize.least_squares(residuals, start, bounds=(lower, upper),
                                                 x_scale="jac")  # fmt: skip
            if best is None or found.cost < best.cost:
                best = found
    assert best is not None
    p = best.x
    major = float(p[2])
    minor, azimuth = (major * float(p[3]), float(p[4])) if anisotropic else (major, 0.0)
    return VariogramFit(model, float(p[0]), float(p[1]), major, minor, azimuth,
                        float(2.0 * best.cost))  # fmt: skip


#: Points of the table inverting the bivariate normal orthant probability.
ORTHANT_TABLE = 4001


def orthant_probability(threshold: float, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``P(Y1 < t, Y2 < t)`` for standard normals of correlation ``rho`` in [0, 1].

    With ``rho = sin(theta)`` the classic integral (Sheppard 1898; Owen 1956)
    ``Phi2(t, t; rho) = Phi(t)^2 + 1 / (2 pi) * int_0^theta exp(-t^2 / (1 + sin u)) du``
    has a smooth integrand, integrated here by the trapezoid rule on ``ORTHANT_TABLE``
    points (error ~1e-8 at |t| < 3).
    """
    from scipy.special import ndtr

    theta = np.linspace(0.0, math.pi / 2, ORTHANT_TABLE)
    density = np.exp(-(threshold**2) / (1.0 + np.sin(theta))) / (2.0 * math.pi)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(theta))]
    )
    wanted = np.arcsin(np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0))
    return np.asarray(float(ndtr(threshold)) ** 2 + np.interp(wanted, theta, cumulative))


def latent_variogram(indicator: ExperimentalVariogram, proportion: float) -> ExperimentalVariogram:
    """Return the latent Gaussian field's variogram ``1 - rho(h)`` behind an indicator's.

    For a facies boundary at ``t = Phi^-1(p)`` (the indicator is ``Y < t``, ``p`` its
    proportion), ``gamma_I(h) = p - Phi2(t, t; rho(h))``; each class's ``rho`` is the
    inverse of :func:`orthant_probability`, clipped to [0, 1] (an indicator's variogram
    beyond its sill ``p (1 - p)`` is read as uncorrelated). Fitting this, not the indicator
    variogram, gives the latent field's range: a truncated Gaussian's indicator variogram is
    linear at the origin whatever the latent model, so a direct fit is misspecified.
    """
    from scipy.special import ndtri

    if not 0.0 < proportion < 1.0:
        raise ValueError(f"proportion must be in (0, 1), got {proportion}")
    threshold = float(ndtri(proportion))
    rho = np.linspace(0.0, 1.0, ORTHANT_TABLE)
    table = orthant_probability(threshold, rho)  # increasing in rho, p^2 ... p
    target = np.clip(proportion - indicator.gamma, table[0], table[-1])
    latent = 1.0 - np.interp(target, table, rho)
    gamma = np.where(np.isfinite(indicator.gamma), latent, np.nan)
    return ExperimentalVariogram(indicator.lags, gamma, indicator.counts, indicator.azimuth,
                                 indicator.distances)  # fmt: skip


@variogram_estimators.register("wls")
class WeightedLeastSquares:
    """Matheron's semivariogram fitted by Cressie's weighted least squares (see the module).

    Parameters
    ----------
    n_lags
        Lag classes, of equal width up to ``max_lag``. Twenty, not the usual dozen: the
        models differ most at short lags (an exponential's slope at the origin, a
        gaussian's flat start), which a coarse first class hides (on 20 exponential fields
        the right model won 9 times with 12 classes, 11 with 20).
    max_lag
        The largest lag (cells); default: half the data's horizontal extent.
    """

    name = "wls"

    def __init__(self, n_lags: int = 20, max_lag: float | None = None) -> None:
        if n_lags < 3:
            raise ValueError(f"n_lags must be at least 3, got {n_lags}")
        if max_lag is not None and max_lag <= 0:
            raise ValueError(f"max_lag must be positive, got {max_lag}")
        self.n_lags = n_lags
        self.max_lag = max_lag

    def estimate(
        self,
        cells: NDArray[np.int64],
        values: NDArray[np.float64],
        *,
        model: str | None = None,
        anisotropic: bool = True,
    ) -> VariogramEstimate:
        """Return the variogram of ``values`` at ``cells`` (``(n, 3)``, 1-based ``i, j, k``).

        An anisotropic fit when asked and every direction has enough pairs, else an
        isotropic one (said in the notes). ``model`` fixes the model; else the best of
        :data:`FIT_MODELS`.

        Raises
        ------
        ValueError
            If the data are too few or constant, or the model is not one of FIT_MODELS.
        """
        cells = np.asarray(cells)
        z = np.asarray(values, dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != 3 or z.shape != (cells.shape[0],):
            raise ValueError("cells must be (n, 3) and values (n,)")
        if model is not None and model not in FIT_MODELS:
            raise ValueError(f"model must be one of {FIT_MODELS}, got {model!r}")
        if z.size < 3 or float(np.var(z)) == 0.0:
            raise ValueError("too few data, or all equal: nothing to fit")
        xy = cells[:, :2].astype(np.float64)
        extent = float(np.hypot(*np.ptp(xy, axis=0)))
        max_lag = self.max_lag or extent / 2
        if max_lag <= 0:
            raise ValueError("the data lie in one column: no horizontal lag to fit")
        edges = np.linspace(0.0, max_lag, self.n_lags + 1)
        layers = cells[:, 2]
        notes: list[str] = []
        variograms: list[ExperimentalVariogram] = []
        if anisotropic:
            tolerance = 90.0 / len(DIRECTIONS)  # half the spacing
            variograms = [experimental_variogram(xy, z, edges, layers=layers, azimuth=a,
                                                 angle_tolerance=tolerance)
                          for a in DIRECTIONS]  # fmt: skip
            short = [v.azimuth for v in variograms
                     if int(np.sum(v.counts >= MIN_PAIRS)) < 3]  # fmt: skip
            if short:
                notes.append(f"too few pairs in the directions {short} (degrees): isotropic fit")
                anisotropic = False
        if not anisotropic:
            variograms = [experimental_variogram(xy, z, edges, layers=layers)]
        top = MAX_RANGE_LAGS * max_lag
        fits = [fit_variogram(variograms, m, anisotropic=anisotropic, max_range=top)
                for m in ([model] if model else FIT_MODELS)]  # fmt: skip
        best = min(fits, key=lambda f: f.residual)
        if best.major_range > 0.99 * top:
            notes.append(f"the major range reached its bound, {top:.3g} cells ({MAX_RANGE_LAGS:g} "
                         "times the largest lag): the data do not show where the variogram "
                         "levels off; give a range if you know it")  # fmt: skip
        return VariogramEstimate(
            model=best.model, nugget=best.nugget, sill=best.sill,
            major_range=best.major_range, minor_range=best.minor_range, azimuth=best.azimuth,
            residual=best.residual, n_data=int(z.size), mean=float(z.mean()),
            variance=float(z.var()), residuals={f.model: f.residual for f in fits},
            notes=notes,
        )  # fmt: skip


__all__ = [
    "DIRECTIONS",
    "FIT_MODELS",
    "MIN_PAIRS",
    "ExperimentalVariogram",
    "VariogramFit",
    "WeightedLeastSquares",
    "experimental_variogram",
    "fit_variogram",
    "latent_variogram",
    "model_variogram",
    "orthant_probability",
]
