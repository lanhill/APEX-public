"""The ``wavelet`` data transform: sparse representation of field data.

Each field of one data type (one vintage per report time) is compressed on its own, as in
PET, where one ``SparseRepresentation`` is fitted per vintage; the kernels and their PET
provenance are in :mod:`apex.algorithms.data_compression.wavelet`. Other data pass
through unchanged, in place.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.algorithms.data_compression.wavelet import (
    CoefficientLayout,
    FieldCompression,
    Noise,
    Thresholding,
    ThresholdRule,
    WaveletOptions,
    decompose,
    fit_field,
)
from apex.core.options import Text, TextList
from apex.core.plugins import data_transforms
from apex.domain.observations import DataKey, ObservationSet

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Segment:
    """Rows ``source`` of the input become the next output rows (compressed if ``field``)."""

    source: slice
    field: FieldCompression | None


@data_transforms.register("wavelet")
class WaveletCompression:
    """Replace each observed field of ``datatype`` by its leading wavelet coefficients.

    The compressed observation of a field at time ``t`` gets the keys
    ``DataKey(t, "wavelet(<datatype>)", k)``, ``k = 0..n_retained-1``, and the estimated
    noise variance of each coefficient replaces the field's given variances (as in PET).

    Parameters
    ----------
    datatype
        Data type of the fields to compress.
    grid
        Shape of the grid the fields live on, e.g. ``[nx, ny, nz]``: 1 to 3 positive sizes.
    wavelet
        A discrete PyWavelets wavelet name, e.g. ``"haar"`` or ``"db2"``.
    level
        Number of decomposition levels, at least 1.
    order
        Cell order of the flattened fields and mask: ``"F"`` (default; the Eclipse order,
        i fastest) or ``"C"``.
    noise
        ``"white"`` (default): one noise level, estimated from the finest diagonal
        sub-band, for every coefficient. ``"coloured"``: one noise level per sub-band.
    threshold_rule
        ``"universal"`` (default; ``sqrt(2 ln n)`` times the noise level) or
        ``"bayesian"``, which needs ``noise = "coloured"``. Non-zero coefficients at least
        as large as the threshold in magnitude are retained.
    threshold_multiplier
        Scales every threshold (PET's ``th_mult``), at least 0; the default 1 uses the
        rule's threshold as is, and 0 keeps every non-zero coefficient.
    thresholding
        ``"hard"`` (default) keeps the retained coefficients as they are; ``"soft"``
        shrinks them towards zero by the threshold.
    keep_approximation
        Keep every approximation coefficient, unthresholded (PET's ``keep_ca``); off by
        default.
    min_noise
        Floor on the estimated noise standard deviations, positive (default 1e-9).
    times
        Only compress the fields at these times; default all.
    mask
        Active-cell mask (grid-shaped or flat in ``order``) shared by the fields; default
        all cells. Configurations give ``mask_file``, which the composition root loads.
    inactive_value
        Value for inactive cells; default each observed field's mean (as PET).

    Raises
    ------
    ValueError
        If an option is invalid.
    """

    def __init__(
        self,
        datatype: Text,
        grid: Sequence[int],
        wavelet: Text,
        level: int,
        *,
        times: TextList | None = None,
        mask: ArrayLike | None = None,
        inactive_value: float | None = None,
        order: Literal["C", "F"] = "F",
        noise: Noise = "white",
        threshold_rule: ThresholdRule = "universal",
        threshold_multiplier: float = 1.0,
        thresholding: Thresholding = "hard",
        keep_approximation: bool = False,
        min_noise: float = 1e-9,
    ) -> None:
        if order not in ("C", "F"):
            raise ValueError(f"order must be 'C' or 'F', got {order!r}")
        self.datatype = datatype
        self.options = WaveletOptions(
            grid=tuple(int(n) for n in grid),
            wavelet=wavelet,
            level=int(level),
            order=order,
            noise=noise,
            threshold_rule=threshold_rule,
            threshold_multiplier=float(threshold_multiplier),
            thresholding=thresholding,
            keep_approximation=bool(keep_approximation),
            min_noise=float(min_noise),
        )
        self.times = None if times is None else frozenset(str(t) for t in times)
        self.mask = None if mask is None else np.asarray(mask)
        self.inactive_value = inactive_value
        self.fields: dict[str, FieldCompression] = {}
        self._segments: list[_Segment] | None = None
        self._n_in = 0

    #: Compressing one member's fields gives, bit for bit, its column of the ensemble's
    #: compression: ``pywt.wavedecn`` filters along the grid axes only, and
    #: the retained coefficients and thresholds act element by element. Checked by
    #: ``tests/unit/algorithms/test_wavelet_state.py``, and on every host by the probe.
    member_wise = True

    def state(self) -> dict[str, NDArray[np.generic]]:
        """Return the fitted state (see :class:`~apex.core.simulation.PortableTransform`).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._segments is None:
            raise RuntimeError("wavelet: fit() must run before state()")
        out: dict[str, NDArray[np.generic]] = {
            "n_in": np.array([self._n_in]), "n_segments": np.array([len(self._segments)]),
            # The options it was fitted for: a host whose options differ (order, thresholding
            # change every member's coefficients) refuses the state.
            "options": np.array([_options_text(self.options)]),
        }  # fmt: skip
        times = {id(fitted): time for time, fitted in self.fields.items()}
        for i, segment in enumerate(self._segments):
            out[f"{i}.source"] = np.array([segment.source.start, segment.source.stop])
            fitted = segment.field
            if fitted is None:
                continue
            out[f"{i}.time"] = np.array([times[id(fitted)]])
            out[f"{i}.active"] = fitted.active.copy()
            out[f"{i}.inactive_value"] = np.array([fitted.inactive_value])
            out[f"{i}.shapes"] = np.array(fitted.layout.shapes, dtype=np.int64)
            out[f"{i}.keys"] = np.array(fitted.layout.keys)
            for name in ("retained", "thresholds", "observed", "noise_std"):
                out[f"{i}.{name}"] = np.array(getattr(fitted, name), copy=True)
        return out

    def load_state(self, state: Mapping[str, NDArray[np.generic]]) -> None:
        """Take a fitted state from :meth:`state`, for these options.

        Raises
        ------
        ValueError
            If the state was fitted for other options, or its arrays do not fit together
            (a state that would fail, or misread rows, only when applied).
        """
        given = str(np.asarray(state["options"])[0])
        if given != _options_text(self.options):
            raise ValueError(
                f"wavelet: the state was fitted for other options ({given}), these are "
                f"{_options_text(self.options)}"
            )
        segments: list[_Segment] = []
        fields: dict[str, FieldCompression] = {}
        for i in range(int(state["n_segments"][0])):
            start, stop = (int(v) for v in state[f"{i}.source"])
            if f"{i}.time" not in state:
                segments.append(_Segment(slice(start, stop), None))
                continue
            shapes = tuple(tuple(int(n) for n in shape) for shape in state[f"{i}.shapes"])
            band = np.concatenate(
                [np.full(math.prod(shape), k, dtype=np.intp) for k, shape in enumerate(shapes)]
            )
            layout = CoefficientLayout(shapes, tuple(str(k) for k in state[f"{i}.keys"]), band)
            expected = decompose(np.zeros((*self.options.grid, 1)), self.options)[1]
            if layout.shapes != expected.shapes or layout.keys != expected.keys:
                raise ValueError("wavelet: the state's bands do not fit these options")
            fitted = FieldCompression(
                options=self.options,
                active=np.asarray(state[f"{i}.active"], dtype=np.bool_),
                inactive_value=float(state[f"{i}.inactive_value"][0]),
                layout=layout,
                retained=np.asarray(state[f"{i}.retained"], dtype=np.intp),
                thresholds=np.asarray(state[f"{i}.thresholds"], dtype=np.float64),
                observed=np.asarray(state[f"{i}.observed"], dtype=np.float64),
                noise_std=np.asarray(state[f"{i}.noise_std"], dtype=np.float64),
            )
            _check_fitted(fitted, stop - start)
            fields[str(state[f"{i}.time"][0])] = fitted
            segments.append(_Segment(slice(start, stop), fitted))
        n_in = int(state["n_in"][0])
        position = 0
        for segment in segments:  # the segments tile the input rows, in order
            if segment.source.start != position or segment.source.stop < position:
                raise ValueError("wavelet: the state's segments do not tile the input rows")
            position = segment.source.stop
        if position != n_in:
            raise ValueError(f"wavelet: the state's segments cover {position} of {n_in} rows")
        self.fields = fields
        self._segments = segments
        self._n_in = n_in

    @property
    def output_datatype(self) -> str:
        """Data type of the compressed coefficients' keys."""
        return f"wavelet({self.datatype})"

    def fit(self, observations: ObservationSet) -> ObservationSet:
        """Compress the observed fields; return the transformed observations.

        Raises
        ------
        ValueError
            If no field of ``datatype`` is observed (at ``times``), a field's components
            are not ``0..n-1`` in order, or a field does not fit the grid and mask.
        """
        runs = _field_runs(observations.keys, self.datatype, self.times)
        if not runs:
            where = "" if self.times is None else f" at times {sorted(self.times)}"
            raise ValueError(f"wavelet: no field observations of {self.datatype!r}{where}")
        if self.options.level > self.options.max_level:
            logger.warning(
                "wavelet level %d exceeds %d for grid %s and %s: all coefficients have "
                "boundary effects",
                self.options.level,
                self.options.max_level,
                self.options.grid,
                self.options.wavelet,
            )
        keys: list[DataKey] = []
        values: list[NDArray[np.float64]] = []
        variances: list[NDArray[np.float64]] = []
        segments: list[_Segment] = []
        position = 0
        self.fields = {}
        for time, rows in runs:
            if position < rows.start:  # untouched data before this field
                segments.append(_Segment(slice(position, rows.start), None))
                keys.extend(observations.keys[position : rows.start])
                values.append(observations.values[position : rows.start])
                variances.append(observations.variances[position : rows.start])
            try:
                fitted = fit_field(
                    observations.values[rows],
                    self.options,
                    active=self.mask,
                    inactive_value=self.inactive_value,
                )
            except ValueError as error:
                raise ValueError(f"wavelet: field {self.datatype}@{time}: {error}") from error
            self.fields[time] = fitted
            segments.append(_Segment(rows, fitted))
            keys.extend(DataKey(time, self.output_datatype, k) for k in range(fitted.n_retained))
            values.append(fitted.observed)
            variances.append(fitted.noise_std**2)
            logger.info(
                "field compressed",
                extra={
                    "event": "field_compressed",
                    "datatype": self.datatype,
                    "time": time,
                    "n_values": fitted.n_active,
                    "n_coefficients": fitted.n_retained,
                },
            )
            position = rows.stop
        n = observations.n_data
        if position < n:
            segments.append(_Segment(slice(position, n), None))
            keys.extend(observations.keys[position:])
            values.append(observations.values[position:])
            variances.append(observations.variances[position:])
        self._segments = segments
        self._n_in = n
        return ObservationSet(keys, np.concatenate(values), np.concatenate(variances))

    def apply(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compress predictions ``(n_in, n_ensemble)`` to ``(n_out, n_ensemble)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``predicted`` does not have the fitted number of rows.
        """
        if self._segments is None:
            raise RuntimeError("wavelet: fit() must run before apply()")
        predicted = np.asarray(predicted, dtype=np.float64)
        if predicted.ndim != 2 or predicted.shape[0] != self._n_in:
            raise ValueError(f"predicted must be ({self._n_in}, n_ensemble), got {predicted.shape}")
        parts = [
            predicted[s.source] if s.field is None else s.field.compress(predicted[s.source])
            for s in self._segments
        ]
        return np.concatenate(parts, axis=0)


def _options_text(options: WaveletOptions) -> str:
    """Return the wavelet options a fitted state was made for (``repr`` of the dataclass).

    The datatype, times, mask and inactive value are not among them: the state carries
    what it took of them (its segments, active mask and inactive value).
    """
    return repr(options)


def _check_fitted(fitted: FieldCompression, n_rows: int) -> None:
    """Refuse a loaded field whose arrays do not fit together or its rows.

    Raises
    ------
    ValueError
        If the mask is not the grid's size, the field's rows are not its active cells, the
        retained coefficients are out of range or not ascending, or the per-coefficient
        arrays differ in length.
    """
    if fitted.active.shape != (fitted.options.n_cells,):
        raise ValueError(f"wavelet: the state's mask has {fitted.active.size} cells, the grid "
                         f"{fitted.options.n_cells}")  # fmt: skip
    if fitted.n_active != n_rows:
        raise ValueError(f"wavelet: {n_rows} rows for {fitted.n_active} active cells")
    retained = fitted.retained
    n = fitted.layout.n_coefficients
    if retained.size and (retained[0] < 0 or retained[-1] >= n or np.any(np.diff(retained) <= 0)):
        raise ValueError("wavelet: the state's retained coefficients are out of range or order")
    sizes = {retained.size, fitted.thresholds.size, fitted.observed.size, fitted.noise_std.size}
    if len(sizes) != 1:
        raise ValueError("wavelet: the state's per-coefficient arrays differ in length")


def _field_runs(
    keys: Sequence[DataKey], datatype: str, times: frozenset[str] | None
) -> list[tuple[str, slice]]:
    """Return the contiguous rows of each field of ``datatype``, by time, in key order.

    Raises
    ------
    ValueError
        If a field's components are split, out of order, or do not start at 0.
    """
    runs: list[tuple[str, slice]] = []
    i = 0
    while i < len(keys):
        key = keys[i]
        if (
            key.datatype != datatype
            or key.index is None
            or (times is not None and key.time not in times)
        ):
            i += 1
            continue
        start = i
        while (
            i < len(keys)
            and keys[i].datatype == datatype
            and keys[i].time == key.time
            and keys[i].index is not None
        ):
            i += 1
        indices = [k.index for k in keys[start:i]]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"wavelet: field {datatype}@{key.time} must list components 0..n-1 in order"
            )
        if any(t == key.time for t, _ in runs):
            raise ValueError(f"wavelet: field {datatype}@{key.time} is not contiguous")
        runs.append((key.time, slice(start, i)))
    if times is not None:
        missing = sorted(times - {t for t, _ in runs})
        if missing:
            raise ValueError(f"wavelet: no field {datatype} observed at {', '.join(missing)}")
    return runs
