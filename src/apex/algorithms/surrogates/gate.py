"""The accuracy gate: a surrogate against the simulator on held-out full runs.

One gate, in core, for the surrogate workflow, history matching's checks and any agent that
builds surrogates.
The held-out runs are never used in ``fit``. Every array is ``(n_keys, n_runs)`` (or
``(n_cells, n_times, n_runs)`` for fields), the run axis last.

**With observation errors** (history matching), per datatype ``d``::

    e_d = sqrt( mean_{k in d, r} ((s_kr - y_kr) / sigma_k)^2 ) <= sigma_ratio   (0.5)

``s``: the surrogate, ``y``: the simulator, ``sigma_k``: the observation error's standard
deviation at key ``k``. The surrogate's error must lie well inside the data noise the method
assumes. Both are compared in the space the method sees: a caller with data transforms
applies them to both and gives the transformed data's variances.

**Without them** (series, fields, objectives), relative errors by kind of quantity, as for
reduced simulation models against their fine reference: per series (one
datatype and index over time, in time order) and run, ``max_t |s - y| / scale``, where
``scale`` is the reference's largest change from its first value for a pressure (so that a
build-up is judged, not the absolute pressure) and its largest magnitude otherwise; ``e_d``
is the largest over the datatype's series and runs. Its tolerance is the kind's
(:attr:`GateTolerances.relative`). Each scale is floored at :data:`SCALE_FLOOR` of the
largest among the series of its vector (every well's ``WOPR``; for pressures, the largest
change) or field: a shut well's zero rates, a cell whose pressure changes only by solver
noise. A pressure is judged on its magnitude, a much looser measure, only if no series of its
vector changes at all, and the result says so (``scale``, ``notes``). The "first value" is the
series' first *gated* time: a pressure series that starts after the first time gated is
noted, since its change may be up to twice smaller than from the initial value.

**Fields** (state fields of field surrogates, the user, 2026-11-02): one
pooled relative RMS per field over its cells, times and held-out runs (the measure of the
R-U-Net literature), ``e = RMS(s - y) / scale`` against the kind's tolerance, where ``scale``
is a pressure's RMS change from the first time, and other fields' range (max - min). Cells
that are 0 in every run, time and both arrays (inactive) are left out. A cell-by-cell
relative error, as for series, would let a cell that barely changes fail any learned field
(a 10 % RMS error scored 22.9 against 0.05 in the review).

With observation errors the criterion is the RMS; the largest single ``|s - y| / sigma`` of a
datatype is recorded beside it (``max_sigma_ratio``), so that one key far off shows.

**Objectives** (optimization): Spearman's rank correlation of the held-out runs' objectives,
surrogate against simulator, at least ``min_rank_correlation`` (0.95). Ties take average
ranks; a constant set of objectives has no correlation and fails. With several objectives
each is ranked, and the lowest correlation is gated.

A prediction that is not finite fails its datatype (its error is ``None``). The verdict passes
only if something was gated and everything gated passed.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    DatatypeGate,
    GateResult,
    GateTolerances,
    QuantityKind,
    RankGate,
    time_position,
)


def quantity_kind(datatype: str) -> QuantityKind:
    """Return a datatype's kind from its vector mnemonic (summary vectors' conventions).

    Only the mnemonic before ``:`` counts (``WGIR:CO2INJ`` is a rate). Pressure: ``BHP``,
    ``THP``, ``PRESS``, ``PR`` vectors (``FPR``, ``BPR``), well-block ``WBP`` (``WBP4``);
    plume: ``PLUME``, ``AREA``, ``EXTENT``; mass:
    ``MASS``, ``CO2``, totals and in-place vectors (``..IT``, ``..PT``, ``..IP``); rate:
    production and injection rates (``WOPR``, ``FGIR``); anything else ``other``, ratios
    (``WGOR``, ``WWGR``) and water cuts (``WWCT``) among them.
    """
    vector = datatype.upper().split(":", 1)[0]
    if (
        any(word in vector for word in ("BHP", "THP", "PRESS"))
        or re.fullmatch(r"[FRBWC]PR[A-Z]?", vector)
        or re.fullmatch(r"WBP\d?", vector)
    ):
        return "pressure"
    if any(word in vector for word in ("PLUME", "AREA", "EXTENT")):
        return "plume"
    if "MASS" in vector or "CO2" in vector or re.fullmatch(r"[FGWR][A-Z]{1,2}(IT|PT|IP)", vector):
        return "mass"
    # Rates are production or injection (third letter P or I: WOPR, FGIR); ratios
    # (WGOR, WWGR, WOGR, WGLR) and water cuts (WWCT) are "other".
    if re.fullmatch(r"[FGWRC][A-Z][PI]R", vector):
        return "rate"
    return "other"


def _finite(value: float) -> float | None:
    return value if np.isfinite(value) else None


#: A series' scale is at least this share of the largest scale of its vector (the datatypes
#: sharing the mnemonic before ":", e.g. every well's WOPR) or field: a shut well's zero
#: rates, or a cell whose pressure changes only by solver noise, would otherwise turn a
#: surrogate's round-off into a huge relative error.
SCALE_FLOOR = 1e-3


@dataclass(frozen=True)
class _Part:
    """Series ``(n_times, n_series)`` of one datatype, summarised per column.

    The largest error, the reference's largest change from its first value, and its largest
    magnitude.
    """

    error: NDArray[np.float64]
    change: NDArray[np.float64]
    magnitude: NDArray[np.float64]


def _part(surrogate: NDArray[np.float64], simulator: NDArray[np.float64]) -> _Part:
    return _Part(
        error=np.max(np.abs(surrogate - simulator), axis=0),
        change=np.max(np.abs(simulator - simulator[:1]), axis=0),
        magnitude=np.max(np.abs(simulator), axis=0),
    )


def _largest(values: Sequence[NDArray[np.float64]]) -> float:
    return max((float(np.max(v)) for v in values if v.size), default=0.0)


def _pool_scale(kind: QuantityKind, pool: Sequence[_Part]) -> Literal["change", "magnitude"]:
    """Return what a vector's (or field's) series are judged against.

    A pressure by its change from its first value, so that a build-up is judged and not the
    absolute pressure; by its magnitude only if no series of the vector changes at all (a much
    looser measure, reported). Every other kind by its magnitude.
    """
    if kind == "pressure" and _largest([p.change for p in pool]) > 0:
        return "change"
    return "magnitude"


def _relative(parts: Sequence[_Part], pool: Sequence[_Part], scale: str) -> float:
    """Return the largest ``error / max(scale, SCALE_FLOOR * the pool's largest scale)``.

    Zero error against a zero scale is 0, any other error against it infinite; an error that
    is not a number (a prediction that is not finite) is infinite.
    """
    if not parts:
        return 0.0
    errors = np.concatenate([p.error for p in parts])
    chosen = [p.change if scale == "change" else p.magnitude for p in parts]
    reference = _largest([p.change if scale == "change" else p.magnitude for p in pool])
    scales = np.maximum(np.concatenate(chosen), SCALE_FLOOR * reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(scales > 0, errors / np.where(scales > 0, scales, 1.0),
                         np.where(errors == 0, 0.0, np.inf))  # fmt: skip
    ratio = np.where(np.isnan(ratio), np.inf, ratio)
    return float(np.max(ratio)) if ratio.size else 0.0


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float | None:
    """Return Spearman's rho (Pearson's correlation of average ranks), ``None`` if undefined."""
    if a.size < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    ra = rankdata(a) - (a.size + 1) / 2
    rb = rankdata(b) - (b.size + 1) / 2
    denominator = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    if denominator == 0:
        return None
    return float(np.sum(ra * rb) / denominator)


def _field_gate(
    name: str,
    kind: QuantityKind,
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    limit: float,
) -> DatatypeGate:
    """Return one field's pooled relative RMS gate (module docstring).

    ``predicted`` and ``reference`` are ``(n_cells, n_times, n_runs)``.
    """
    zero = np.all((reference == 0) & (predicted == 0), axis=(1, 2))
    s, y = predicted[~zero], reference[~zero]
    notes: list[str] = []
    if y.size == 0:
        return DatatypeGate(
            datatype=name, measure="field_rms", kind=kind, error=None, tolerance=limit,
            values=0, passed=False, notes=["no cell of the field is ever non-zero"],
        )  # fmt: skip
    error_rms = float(np.sqrt(np.mean((s - y) ** 2)))
    scale_name: Literal["change", "magnitude", "range"]
    if kind == "pressure":
        scale_name = "change"
        scale = float(np.sqrt(np.mean((y - y[:, :1, :]) ** 2)))
        if scale == 0:
            scale_name, scale = "magnitude", float(np.sqrt(np.mean(y**2)))
            notes.append("no cell changes: judged on its magnitude")
    else:
        scale_name, scale = "range", float(np.max(y) - np.min(y))
        if scale == 0:
            scale_name, scale = "magnitude", float(np.max(np.abs(y)))
            notes.append("the field is constant: judged on its magnitude")
    error = error_rms / scale if scale > 0 else float("inf")
    if not np.all(np.isfinite(s)):
        error = float("nan")
    return DatatypeGate(
        datatype=name, measure="field_rms", kind=kind, error=_finite(error), scale=scale_name,
        notes=notes, tolerance=limit, values=int(y.size), passed=bool(error <= limit),
    )  # fmt: skip


def gate(
    surrogate: NDArray[np.float64],
    simulator: NDArray[np.float64],
    keys: Sequence[DataKey],
    held_out: Sequence[int],
    *,
    tolerances: GateTolerances | None = None,
    variances: NDArray[np.float64] | None = None,
    kinds: Mapping[str, QuantityKind] | None = None,
    objectives: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    fields: Mapping[str, tuple[NDArray[np.float64], NDArray[np.float64]]] | None = None,
    failed: Sequence[int] = (),
) -> GateResult:
    """Gate a surrogate on held-out full runs.

    Parameters
    ----------
    surrogate, simulator
        ``(n_keys, n_runs)``: the surrogate's and the simulator's values at ``keys`` for the
        held-out runs (``n_keys`` may be 0 when only objectives or fields are gated).
    keys
        ``(n_keys,)``.
    held_out
        ``(n_runs,)``, the held-out runs' ids.
    tolerances
        The gate's tolerances (the defaults if ``None``).
    variances
        ``(n_keys,)``, the observation errors' variances (> 0), in the space compared; ``None``
        gates by relative errors.
    kinds
        The kind of quantity of a datatype or field, where :func:`quantity_kind` would guess
        wrong.
    objectives
        ``((n_runs,), (n_runs,))``, the surrogate's and the simulator's objective per run; or
        ``((n_obj, n_runs), (n_obj, n_runs))`` for several objectives, gated on the lowest
        of their correlations.
    fields
        ``{name: (surrogate, simulator)}``, each ``(n_cells, n_times, n_runs)``, gated by
        relative errors (each cell a series over time).
    failed
        The design's runs that failed, recorded with the verdict.

    Raises
    ------
    ValueError
        If the shapes disagree, the simulator's values are not finite or a variance is not
        positive.
    """
    tolerances = tolerances or GateTolerances()
    s = np.ascontiguousarray(surrogate, dtype=np.float64)
    y = np.ascontiguousarray(simulator, dtype=np.float64)
    n_runs = len(held_out)
    if s.shape != (len(keys), n_runs) or y.shape != s.shape:
        raise ValueError(
            f"surrogate and simulator must both be ({len(keys)}, {n_runs}), got {s.shape} and "
            f"{y.shape}"
        )
    if n_runs < 1:
        raise ValueError("the gate needs at least one held-out run")
    if not np.all(np.isfinite(y)):
        raise ValueError("the simulator's held-out values must be finite")
    kinds = kinds or {}
    groups: dict[str, list[int]] = {}
    for row, key in enumerate(keys):
        groups.setdefault(key.datatype, []).append(row)
    entries: list[DatatypeGate] = []
    if variances is not None:
        sigma2 = np.asarray(variances, dtype=np.float64).reshape(-1)
        if sigma2.shape != (len(keys),) or not np.all(np.isfinite(sigma2) & (sigma2 > 0)):
            raise ValueError(f"variances must be ({len(keys)},), finite and > 0")
        normalized = (s - y) / np.sqrt(sigma2)[:, None]
        for datatype, rows in groups.items():
            with np.errstate(invalid="ignore", over="ignore"):
                error = float(np.sqrt(np.mean(np.square(normalized[rows]))))
                largest = float(np.max(np.abs(normalized[rows]))) if normalized[rows].size else 0.0
            error = error if np.isfinite(error) else np.inf
            entries.append(
                DatatypeGate(
                    datatype=datatype,
                    measure="sigma_ratio",
                    error=_finite(error),
                    max_sigma_ratio=_finite(largest),
                    tolerance=tolerances.sigma_ratio,
                    values=len(rows) * n_runs,
                    passed=bool(error <= tolerances.sigma_ratio),
                )
            )
    else:
        first_time = min((k.time for k in keys), key=time_position, default=None)
        found: dict[str, tuple[QuantityKind, list[_Part], list[str]]] = {}
        for datatype, rows in groups.items():
            kind = kinds.get(datatype) or quantity_kind(datatype)
            series: dict[int | None, list[int]] = {}
            for row in rows:
                series.setdefault(keys[row].index, []).append(row)
            parts, notes = [], []
            for members in series.values():
                ordered = sorted(members, key=lambda r: time_position(keys[r].time))
                parts.append(_part(s[ordered], y[ordered]))
                start = keys[ordered[0]].time
                if (
                    kind == "pressure"
                    and first_time is not None
                    and time_position(start) > time_position(first_time)
                ):
                    notes.append(
                        f"its change is measured from {start}, after the first time gated "
                        f"({first_time}): the scale may be looser than from the initial value"
                    )
            found[datatype] = (kind, parts, sorted(set(notes)))
        # The floor's reference: the series of one vector (the mnemonic before ":"), so that a
        # shut well is judged against its open neighbours.
        pools: dict[str, list[_Part]] = {}
        for datatype, (_, parts, _) in found.items():
            pools.setdefault(datatype.split(":", 1)[0], []).extend(parts)
        for datatype, rows in groups.items():
            kind, parts, notes = found[datatype]
            pool = pools[datatype.split(":", 1)[0]]
            scale = _pool_scale(kind, pool)
            if kind == "pressure" and scale == "magnitude":
                notes = [*notes, "no series of its vector changes: judged on its magnitude"]
            error = _relative(parts, pool, scale)
            limit = tolerances.relative[kind]
            entries.append(
                DatatypeGate(
                    datatype=datatype,
                    measure="relative",
                    kind=kind,
                    error=_finite(error),
                    scale=scale,
                    notes=notes,
                    tolerance=limit,
                    values=len(rows) * n_runs,
                    passed=bool(error <= limit),
                )
            )
    for name, (field_s, field_y) in (fields or {}).items():
        fs = np.asarray(field_s, dtype=np.float64)
        fy = np.asarray(field_y, dtype=np.float64)
        if fs.ndim != 3 or fs.shape != fy.shape or fs.shape[2] != n_runs:
            raise ValueError(f"field {name!r} must be (n_cells, n_times, {n_runs}) in both")
        if not np.all(np.isfinite(fy)):
            raise ValueError(f"the simulator's field {name!r} must be finite")
        kind = kinds.get(name) or quantity_kind(name)
        entries.append(_field_gate(name, kind, fs, fy, tolerances.relative[kind]))
    rank = None
    if objectives is not None:
        a = np.atleast_2d(np.asarray(objectives[0], dtype=np.float64))
        b = np.atleast_2d(np.asarray(objectives[1], dtype=np.float64))
        if a.ndim != 2 or a.shape != b.shape or a.shape[1] != n_runs:
            raise ValueError(f"objectives must be two ({n_runs},) or (n_obj, {n_runs}) arrays")
        # Several objectives: each must keep its ranking, so the lowest
        # correlation is the one gated; an undefined one (a constant objective) fails.
        rhos = [_spearman(a[i], b[i]) for i in range(a.shape[0])]
        rho = None if any(r is None for r in rhos) else min(r for r in rhos if r is not None)
        rank = RankGate(
            correlation=rho,
            tolerance=tolerances.min_rank_correlation,
            runs=n_runs,
            passed=rho is not None and rho >= tolerances.min_rank_correlation,
        )
    gated = bool(entries) or rank is not None
    passed = gated and all(e.passed for e in entries) and (rank is None or rank.passed)
    return GateResult(
        held_out=[int(i) for i in held_out],
        failed=[int(i) for i in failed],
        datatypes=entries,
        rank=rank,
        tolerances=tolerances,
        passed=passed,
    )
