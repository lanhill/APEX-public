"""Wavelet sparse representation of field data (e.g. 4D-seismic vintages).

Ported from PET
---------------
PET module : src/pipt/misc_tools/wavelet_tools.py (SparseRepresentation.compress,
             SparseRepresentation.reconstruct), with the option handling of
             src/misc/read_input_csv.py (DataReader._wavelet_compression)
PET commit : e65fb1058411c8dad7ca70a09d501e0b18075e75
Reference  : Luo, X., Bhakta, T., Jakobsen, M. & Nævdal, G. (2017). An ensemble
             4D-seismic history-matching framework with sparse representation based on
             wavelet multiresolution analysis. SPE Journal, 22(3), 985-1010,
             doi:10.2118/180025-PA.
             Donoho, D. L. & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet
             shrinkage. Biometrika, 81(3), 425-455 (MAD noise estimate, universal rule).
Differences: - Soft thresholding shrinks the predicted coefficients with the observed
               thresholds as well; PET shrinks only the observations, which biases every
               retained observed coefficient by its threshold (pet_parity.md, D1).
             - The mask is flattened in the same ``order`` as the data; PET flattens it in
               C order whatever ``order`` is (D2).
             - The retained coefficients and their noise are one selection, so they
               always have the same length; PET's can differ when a threshold is 0 and a
               coefficient is exactly 0 (D6).
             - The Bayesian rule is only accepted with coloured noise; PET silently uses
               the universal rule with white noise (D4). Its formula is PET's (D3).
             - A 1-D grid is supported; PET handles 2-D and 3-D only.
             The author's earlier MATLAB implementation (MATLAB Brugge_4DSeis
             ``wavelet3D_sparse_representation.m``) retains the same coefficients; its
             differences (uncentred MAD noise estimate, |c| <= t ties, n over the whole
             cube) are listed in docs/migration/pet_parity.md.

Notation (PET name -> here): ``wdec`` -> the coefficient vector ``c`` (approximation band
first, then the detail bands from the coarsest level to the finest, each flattened in C
order, which is PET's ``ca_in_vec`` followed by ``cd_in_vec``); ``ca/cd_leading_index`` ->
``retained``; ``est_noise`` -> ``noise_std``; ``th_mult`` -> ``threshold_multiplier``.
"""

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

#: Detail sub-band keys per dimension, in PET's (Matlab) order.
SUBBANDS: dict[int, tuple[str, ...]] = {
    1: ("d",),
    2: ("da", "ad", "dd"),
    3: ("daa", "ada", "dda", "aad", "dad", "add", "ddd"),
}

#: Scales the median absolute deviation to the standard deviation of a normal variable.
MAD_TO_STD = 0.6745

Noise = Literal["white", "coloured"]
ThresholdRule = Literal["universal", "bayesian"]
Thresholding = Literal["hard", "soft"]


@dataclass(frozen=True, slots=True)
class WaveletOptions:
    """How one field is decomposed and thresholded.

    Attributes
    ----------
    grid
        Grid shape the field lives on, e.g. ``(nx, ny, nz)`` (1 to 3 dimensions).
    wavelet
        A discrete PyWavelets wavelet name, e.g. ``"haar"``, ``"db2"``.
    level
        Number of decomposition levels (>= 1).
    order
        Cell order of the flattened field and mask: ``"F"`` (default; Fortran, the Eclipse
        convention with i fastest when ``grid = (nx, ny, nz)``) or ``"C"``.
    noise
        ``"white"``: one noise level from the finest diagonal sub-band, used everywhere.
        ``"coloured"``: one noise level per sub-band.
    threshold_rule
        ``"universal"`` (``sqrt(2 ln n) sigma``), or ``"bayesian"`` (coloured noise only).
    threshold_multiplier
        Scales every threshold (PET's ``th_mult``); 0 keeps every non-zero coefficient.
    thresholding
        ``"hard"`` keeps retained coefficients as they are; ``"soft"`` shrinks them.
    keep_approximation
        Keep every approximation coefficient, unthresholded (PET's ``keep_ca``).
    min_noise
        Floor on estimated noise standard deviations.
    """

    grid: tuple[int, ...]
    wavelet: str
    level: int
    order: Literal["C", "F"] = "F"
    noise: Noise = "white"
    threshold_rule: ThresholdRule = "universal"
    threshold_multiplier: float = 1.0
    thresholding: Thresholding = "hard"
    keep_approximation: bool = False
    min_noise: float = 1e-9

    def __post_init__(self) -> None:
        """Validate the options.

        Raises
        ------
        ValueError
            If an option is out of range or an unsupported combination.
        """
        if not 1 <= len(self.grid) <= 3 or any(int(n) < 1 for n in self.grid):
            raise ValueError(f"grid must have 1 to 3 positive dimensions, got {self.grid}")
        if self.wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(f"unknown discrete wavelet {self.wavelet!r}")
        if self.level < 1:
            raise ValueError(f"level must be >= 1, got {self.level}")
        if self.order not in ("C", "F"):
            raise ValueError(f"order must be 'C' or 'F', got {self.order!r}")
        if self.noise not in ("white", "coloured"):
            raise ValueError(f"noise must be 'white' or 'coloured', got {self.noise!r}")
        if self.threshold_rule not in ("universal", "bayesian"):
            raise ValueError(f"unknown threshold rule {self.threshold_rule!r}")
        if self.threshold_rule == "bayesian" and self.noise == "white":
            # PET silently falls back to the universal rule here (pet_parity.md, D4).
            raise ValueError("the bayesian threshold rule needs noise = 'coloured'")
        if self.thresholding not in ("hard", "soft"):
            raise ValueError(f"thresholding must be 'hard' or 'soft', got {self.thresholding!r}")
        if not math.isfinite(self.threshold_multiplier) or self.threshold_multiplier < 0:
            raise ValueError(f"threshold_multiplier must be >= 0, got {self.threshold_multiplier}")
        if not self.min_noise > 0:
            raise ValueError(f"min_noise must be positive, got {self.min_noise}")

    @property
    def n_cells(self) -> int:
        """Number of grid cells."""
        return math.prod(self.grid)

    @property
    def max_level(self) -> int:
        """Highest level without boundary effects in every coefficient (PyWavelets' rule)."""
        return int(pywt.dwtn_max_level(self.grid, self.wavelet))


@dataclass(frozen=True, slots=True)
class CoefficientLayout:
    """Where each sub-band sits in the flat coefficient vector.

    Attributes
    ----------
    shapes
        Shape of each band: the approximation first, then per level (coarsest first) the
        detail bands in :data:`SUBBANDS` order.
    keys
        ``""`` for the approximation, otherwise the detail key of each band.
    band
        Band number of every coefficient, shape ``(n_coefficients,)``.
    """

    shapes: tuple[tuple[int, ...], ...]
    keys: tuple[str, ...]
    band: NDArray[np.intp] = field(repr=False)

    @property
    def n_coefficients(self) -> int:
        """Length of the coefficient vector."""
        return int(self.band.size)

    def slices(self) -> list[slice]:
        """Slice of each band in the coefficient vector."""
        sizes = [math.prod(shape) for shape in self.shapes]
        ends = np.cumsum(sizes).tolist()
        return [slice(end - size, end) for size, end in zip(sizes, ends, strict=True)]


def decompose(
    signal: NDArray[np.float64], options: WaveletOptions
) -> tuple[NDArray[np.float64], CoefficientLayout]:
    """Wavelet-decompose fields on the grid.

    Parameters
    ----------
    signal
        Fields, shape ``(*grid, n_ensemble)``.
    options
        Wavelet, level and grid.

    Returns
    -------
    coefficients
        ``(n_coefficients, n_ensemble)``, in :class:`CoefficientLayout` order.
    layout
        Where each band sits.
    """
    ndim = len(options.grid)
    if signal.ndim != ndim + 1 or signal.shape[:-1] != tuple(options.grid):
        raise ValueError(f"signal must be (*{options.grid}, n_ensemble), got {signal.shape}")
    with warnings.catch_warnings():
        # Levels above max_level are allowed (as in PET); the caller warns once.
        warnings.simplefilter("ignore", UserWarning)
        wdec = pywt.wavedecn(
            signal, options.wavelet, mode="symmetric", level=options.level, axes=range(ndim)
        )
    bands: list[NDArray[np.float64]] = [np.asarray(wdec[0])]
    keys = [""]
    for details in wdec[1:]:
        for key in SUBBANDS[ndim]:
            bands.append(np.asarray(details[key]))
            keys.append(key)
    n_ensemble = signal.shape[-1]
    shapes = tuple(tuple(b.shape[:-1]) for b in bands)
    # Each band flattened in C order over the grid axes, as PET's ``coeffs.flatten()``.
    coefficients = np.concatenate([b.reshape(-1, n_ensemble) for b in bands], axis=0)
    band = np.concatenate(
        [np.full(math.prod(shape), i, dtype=np.intp) for i, shape in enumerate(shapes)]
    )
    return coefficients, CoefficientLayout(shapes, tuple(keys), band)


def recompose(
    coefficients: NDArray[np.float64], layout: CoefficientLayout, options: WaveletOptions
) -> NDArray[np.float64]:
    """Invert :func:`decompose`: ``(n_coefficients, n_ensemble)`` to ``(*grid, n_ensemble)``."""
    if coefficients.ndim != 2 or coefficients.shape[0] != layout.n_coefficients:
        raise ValueError(
            f"coefficients must be ({layout.n_coefficients}, n_ensemble), got {coefficients.shape}"
        )
    ndim = len(options.grid)
    n_ensemble = coefficients.shape[-1]
    bands = [
        coefficients[s].reshape(*shape, n_ensemble)
        for s, shape in zip(layout.slices(), layout.shapes, strict=True)
    ]
    n_details = len(SUBBANDS[ndim])
    wdec: list[NDArray[np.float64] | dict[str, NDArray[np.float64]]] = [bands[0]]
    for start in range(1, len(bands), n_details):
        wdec.append(
            dict(
                zip(
                    layout.keys[start : start + n_details],
                    bands[start : start + n_details],
                    strict=True,
                )
            )
        )
    signal = np.asarray(pywt.waverecn(wdec, options.wavelet, mode="symmetric", axes=range(ndim)))
    # Odd-sized axes come back one sample longer; crop as PET does.
    return signal[tuple(slice(0, n) for n in options.grid)]


def mad_noise_std(coefficients: NDArray[np.float64], min_noise: float) -> float:
    """Robust noise standard deviation ``median(|c - median(c)|) / 0.6745``, floored.

    PET's form: the median absolute deviation about the median. Donoho & Johnstone
    (1994), and the author's MATLAB code (MATLAB ``wavelet3D_sparse_representation.m``),
    use the uncentred ``median(|c|) / 0.6745``; the two agree to within a fraction of a
    percent for zero-mean noise (docs/migration/pet_parity.md). PET applies it to one
    sub-band at a time.
    """
    c = np.ravel(coefficients)
    return max(float(np.median(np.abs(c - np.median(c)))) / MAD_TO_STD, min_noise)


def soft_threshold(
    values: NDArray[np.float64], threshold: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``sign(c) max(|c| - t, 0)``, with ``threshold`` broadcast against ``values``."""
    result: NDArray[np.float64] = np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    return result


@dataclass(frozen=True, slots=True)
class FieldCompression:
    """The sparse representation fitted to one observed field.

    Build it with :func:`fit_field`. Attributes are the fitted quantities; arrays are
    over the retained coefficients unless stated.

    Attributes
    ----------
    options
        The wavelet options.
    active
        Active-cell mask over the grid, flattened in ``options.order``, ``(n_cells,)``.
    inactive_value
        Value written into inactive cells before decomposing (fixed at fit time).
    layout
        Band layout of the coefficient vector.
    retained
        Indices of the retained coefficients, ascending, ``(n_retained,)``.
    thresholds
        Threshold of each retained coefficient (0 for a kept approximation).
    observed
        The compressed observation.
    noise_std
        Estimated noise standard deviation of each retained coefficient.
    """

    options: WaveletOptions
    active: NDArray[np.bool_]
    inactive_value: float
    layout: CoefficientLayout
    retained: NDArray[np.intp]
    thresholds: NDArray[np.float64]
    observed: NDArray[np.float64]
    noise_std: NDArray[np.float64]

    @property
    def n_active(self) -> int:
        """Length of the raw field vector (active cells)."""
        return int(self.active.sum())

    @property
    def n_retained(self) -> int:
        """Length of the compressed vector."""
        return int(self.retained.size)

    def coefficients(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """All wavelet coefficients of raw fields ``(n_active, n_ensemble)``."""
        coefficients, _ = decompose(
            _embed(data, self.active, self.inactive_value, self.options), self.options
        )
        return coefficients

    def compress(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compress raw fields ``(n_active, n_ensemble)`` to ``(n_retained, n_ensemble)``.

        The retained coefficients of the observation are taken; with soft thresholding
        they are shrunk by the observed thresholds, so that observed and predicted
        coefficients go through the same operator (PET does not shrink these, D1).
        """
        selected = self.coefficients(data)[self.retained]
        if self.options.thresholding == "soft":
            selected = soft_threshold(selected, self.thresholds[:, None])
        return selected

    def reconstruct(self, compressed: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fields ``(n_active, n_ensemble)`` from compressed ``(n_retained, n_ensemble)``.

        The dropped coefficients are set to zero (PET's ``wdec_rec`` + ``reconstruct``).
        """
        if compressed.ndim != 2 or compressed.shape[0] != self.n_retained:
            raise ValueError(
                f"compressed must be ({self.n_retained}, n_ensemble), got {compressed.shape}"
            )
        full = np.zeros((self.layout.n_coefficients, compressed.shape[-1]))
        full[self.retained] = compressed
        grid = recompose(full, self.layout, self.options)
        cells = grid.reshape(self.options.n_cells, -1, order=self.options.order)
        result: NDArray[np.float64] = cells[self.active]
        return result


def fit_field(
    observed: ArrayLike,
    options: WaveletOptions,
    *,
    active: ArrayLike | None = None,
    inactive_value: float | None = None,
) -> FieldCompression:
    """Threshold an observed field and fix its retained coefficients and their noise.

    Parameters
    ----------
    observed
        The observed field over the active cells, ``(n_active,)``, in ``options.order``.
    options
        Wavelet and thresholding options.
    active
        Active-cell mask, grid-shaped or flat in ``options.order``; default all cells.
    inactive_value
        Value for inactive cells; default the mean of ``observed`` (as PET).

    Returns
    -------
    FieldCompression
        The fitted representation; its :attr:`~FieldCompression.observed` is the
        compressed observation and ``noise_std**2`` its error variances.

    Raises
    ------
    ValueError
        If shapes disagree with the grid and mask, values are not finite, or nothing
        survives the thresholding.
    """
    data = np.asarray(observed, dtype=np.float64)
    if active is None:
        mask = np.ones(options.n_cells, dtype=bool)
    else:
        mask_array = np.asarray(active)
        if mask_array.shape not in (tuple(options.grid), (options.n_cells,)):
            raise ValueError(
                f"mask must have shape {options.grid} or ({options.n_cells},), "
                f"got {mask_array.shape}"
            )
        if not np.all((mask_array == 0) | (mask_array == 1)):
            raise ValueError("mask must be boolean (0/1)")
        mask = mask_array.astype(bool).reshape(-1, order=options.order)
    if data.shape != (int(mask.sum()),):
        raise ValueError(
            f"observed field must have one value per active cell ({int(mask.sum())},), "
            f"got {data.shape}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("observed field must be finite")
    fill = float(np.mean(data)) if inactive_value is None else float(inactive_value)

    coefficients, layout = decompose(_embed(data[:, None], mask, fill, options), options)
    c = coefficients[:, 0]
    slices = layout.slices()
    n_bands = len(slices)

    # Noise level and threshold of every band.
    band_noise = np.empty(n_bands)
    band_threshold = np.empty(n_bands)
    if options.noise == "white" and not pywt.Wavelet(options.wavelet).orthogonal:
        # White noise in the field is white with one level in every band only for an
        # orthogonal transform; otherwise the single level mis-states the variances.
        logger.warning(
            "wavelet %r is not orthogonal: noise = 'white' gives every band the finest "
            "band's noise level, which is not correct for it; use noise = 'coloured'",
            options.wavelet,
        )
    if options.noise == "white":
        # One level from the finest diagonal band (the last band), with the universal
        # threshold over the number of data: t = th_mult sqrt(2 ln n) sigma.
        sigma = mad_noise_std(c[slices[-1]], options.min_noise)
        band_noise[:] = sigma
        band_threshold[:] = (
            options.threshold_multiplier * math.sqrt(2 * math.log(data.size)) * sigma
        )
    else:
        for b, s in enumerate(slices):
            cb = c[s]
            sigma = mad_noise_std(cb, options.min_noise)
            band_noise[b] = sigma
            if options.threshold_rule == "universal":
                t = math.sqrt(2 * math.log(cb.size)) * sigma
            else:
                # PET's form of BayesShrink, t = sigma^2 / sqrt(|s_y^2 - sigma^2|), with its
                # ``s_y = ||c||_2 / n`` (Chang, Yu & Vetterli 2000 use ||c||_2 / sqrt(n) and
                # max(., 0)); reproduced as PET computes it (pet_parity.md, D3).
                s_y = float(np.linalg.norm(cb, 2)) / cb.size
                with np.errstate(divide="ignore"):
                    t = float(sigma**2 / np.sqrt(np.abs(s_y**2 - sigma**2)))
            band_threshold[b] = options.threshold_multiplier * t
    if options.keep_approximation:
        band_threshold[0] = 0.0

    threshold = band_threshold[layout.band]
    if options.thresholding == "hard":
        # PET zeroes |c| < t and retains the non-zero survivors.
        keep = (np.abs(c) >= threshold) & (c != 0)
        values = c
    else:
        values = soft_threshold(c, threshold)
        keep = values != 0
    if options.keep_approximation:
        keep[slices[0]] = True  # PET keeps every approximation index, zeros included.
    retained = np.flatnonzero(keep)
    floored = sorted({int(b) for b in layout.band[retained] if band_noise[b] <= options.min_noise})
    if floored:
        # A band without measurable noise (e.g. a constant inactive region, or noise-free
        # synthetic data) gets variance min_noise**2: its coefficients are fitted almost
        # exactly. Legitimate, but worth knowing.
        logger.warning(
            "wavelet noise estimate hit min_noise = %g in band(s) %s; those coefficients "
            "get variance %g",
            options.min_noise,
            ", ".join(f"{b} ({layout.keys[b] or 'approximation'})" for b in floored),
            options.min_noise**2,
        )
    if retained.size == 0:
        raise ValueError(
            "no wavelet coefficient survives the thresholding; lower threshold_multiplier"
        )
    return FieldCompression(
        options=options,
        active=mask,
        inactive_value=fill,
        layout=layout,
        retained=retained,
        thresholds=threshold[retained],
        observed=values[retained],
        noise_std=band_noise[layout.band[retained]],
    )


def _embed(
    data: NDArray[np.float64], active: NDArray[np.bool_], fill: float, options: WaveletOptions
) -> NDArray[np.float64]:
    """Place fields ``(n_active, n_ensemble)`` on the grid: ``(*grid, n_ensemble)``."""
    n_active = int(active.sum())
    if data.ndim != 2 or data.shape[0] != n_active:
        raise ValueError(f"fields must be ({n_active}, n_ensemble), got {data.shape}")
    cells = np.full((options.n_cells, data.shape[-1]), fill)
    cells[active] = data
    # Row i of ``cells`` is cell i in ``order``; reshaping in that same order keeps each
    # column one field and the ensemble axis last, for C and F alike.
    result: NDArray[np.float64] = cells.reshape(*options.grid, data.shape[-1], order=options.order)
    return result


__all__: Sequence[str] = (
    "SUBBANDS",
    "CoefficientLayout",
    "FieldCompression",
    "WaveletOptions",
    "decompose",
    "fit_field",
    "mad_noise_std",
    "recompose",
    "soft_threshold",
)
