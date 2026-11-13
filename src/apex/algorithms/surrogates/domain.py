"""A surrogate's validated domain, built and checked by core, never by a plugin.

**Building.** :func:`training_domain` asks the plugin for the coordinates and projection
residuals of the training *and* held-out inputs (the points the gate validated) and bounds
them:

- a box per coordinate, ``[min, max]`` over the validated points;
- optionally (``extent``) the convex hull of the points, or their Mahalanobis ellipsoid
  ``(x - mean)^T C^+ (x - mean) <= r^2`` with ``C`` the points' sample covariance and ``r``
  the largest distance of a validated point;
- the largest projection residual of a validated point: an input far from the reduced
  subspace is outside even with its coordinates inside the box;
- the keys and fields the surrogate was trained and gated on, and the interval of their times
  (by default the keys' own times: a surrogate is valid up to the last time it was gated on,
  never beyond).

**Checking.** :func:`check_members` computes the members' coordinates with the same plugin
and returns a :class:`DomainCheck`: every member outside, and for each the coordinates
outside the box and by how much, its residual against the bound, its distance against the
radius, or that it is outside the hull; and the keys, fields and times asked for outside
the domain. :meth:`DomainCheck.raise_if_outside` raises
:class:`~apex.core.errors.SurrogateDomainError` (never a ``SimulationError``: a member a
surrogate cannot predict is not a failed member to drop).

**Round-off.** Coordinates computed in batches of different sizes round differently (BLAS),
so a validated point checked again may move by a few ulps. The box, the hull and the
ellipsoid have a slack of :data:`RELATIVE_SLACK` (``1e-9``) relative to each coordinate's
magnitude, and the residual bound relative to the inputs' size, far below any meaningful
extrapolation (the hull's linear programme may accept up to HiGHS's feasibility tolerance,
1e-7 of the box, beyond it). A hull or ellipsoid that would be flat (fewer independent
validated points than varying coordinates) is refused when the domain is built: every new
member would lie outside it.
"""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from apex.core.errors import SurrogateDomainError
from apex.core.surrogates import (
    CoarseCorrectedSurrogate,
    EarlyStartSurrogate,
    SurrogateInputs,
    SurrogateModel,
)
from apex.domain.observations import DataKey
from apex.domain.surrogate import (
    DomainExtent,
    MahalanobisBound,
    TrainingDomain,
    time_position,
)

#: Relative slack of the box, the residual bound and the ellipsoid's radius (round-off only).
RELATIVE_SLACK = 1e-9

#: The hull LP's own accuracy, in units scaled to the box (HiGHS's default tolerances).
_HULL_SOLVER_SLACK = 1e-7

#: Members named in full in an error message; the rest are counted.
_LISTED = 10


def _as_matrix(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be (n_coordinates, n_points), got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} holds values that are not finite")
    return array


def _time_range(times: Sequence[str]) -> tuple[str, str] | None:
    """Return ``(first, last)`` of the labels, or ``None`` if there are none.

    Raises
    ------
    ValueError
        If the labels mix step indices and dates.
    """
    if not times:
        return None
    ordered = sorted(set(times), key=time_position)
    if time_position(ordered[0])[0] != time_position(ordered[-1])[0]:
        raise ValueError("a surrogate's times mix step indices and dates")
    return (ordered[0], ordered[-1])


def _mahalanobis(points: NDArray[np.float64]) -> MahalanobisBound:
    """Return the ellipsoid of ``points`` ``(n_coordinates, n_points)``.

    Built on the coordinates that vary beyond round-off only (:func:`_varying`); the others
    are the box's, since a coordinate constant up to a few ulps would give a direction of
    scale ~1e-14 that no point evaluated again could stay within. The covariance's
    pseudo-inverse is taken through the SVD of the centred covered points
    ``X - mean = U S V^T``: along the ``rank`` directions ``U`` the standard deviations are
    ``s / sqrt(n_points - 1)``; off them a validated point has no extent, so a member may lie
    there only by round-off.
    """
    n_points = points.shape[1]
    covered = np.flatnonzero(_varying(points))
    sub = points[covered]
    mean = sub.mean(axis=-1)
    centred = sub - mean[:, None]
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    cutoff = (s[0] if s.size else 0.0) * max(centred.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(s > cutoff)) if n_points > 1 else 0
    basis = np.ascontiguousarray(u[:, :rank])
    scales = s[:rank] / math.sqrt(max(n_points - 1, 1))
    whitened = (basis.T @ centred) / scales[:, None]
    radius = float(np.max(np.linalg.norm(whitened, axis=0))) if rank else 0.0
    # Round-off off the subspace scales with the coordinates' size, not only their spread (a
    # coordinate of 1e8 moves by ulps of 1e8 when evaluated again).
    size = (
        max(float(np.max(np.linalg.norm(centred, axis=0))),
            float(np.max(np.linalg.norm(sub, axis=0))))
        if covered.size
        else 0.0
    )  # fmt: skip
    return MahalanobisBound(
        coordinates=tuple(int(i) for i in covered),
        mean=mean,
        basis=basis.reshape(covered.size, rank),
        scales=scales,
        radius=radius,
        orthogonal_tolerance=RELATIVE_SLACK * size,
    )


def _slack(
    lower: NDArray[np.float64], upper: NDArray[np.float64], relative: float
) -> NDArray[np.float64]:
    """Return each coordinate's absolute round-off slack: ``relative * max(span, |lo|, |hi|)``."""
    magnitude = np.maximum(upper - lower, np.maximum(np.abs(lower), np.abs(upper)))
    return np.asarray(relative * magnitude, dtype=np.float64)


def _varying(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return which coordinates of ``points`` vary beyond the box's round-off slack."""
    lower, upper = points.min(axis=-1), points.max(axis=-1)
    return np.asarray((upper - lower) > _slack(lower, upper, RELATIVE_SLACK))


def _refuse_flat(points: NDArray[np.float64], extent: DomainExtent) -> None:
    """Refuse a hull or ellipsoid of lower dimension than the coordinates that vary.

    With no more validated points than varying coordinates (or points on a lower-dimensional
    set), the hull and the ellipsoid are flat: every new member would lie outside them.
    Coordinates that do not vary (beyond round-off) are left to the box.
    """
    varying = _varying(points)
    n_varying = int(np.count_nonzero(varying))
    if n_varying == 0:
        return
    centred = points[varying] - points[varying].mean(axis=-1, keepdims=True)
    s = np.linalg.svd(centred, compute_uv=False)
    rank = int(np.count_nonzero(s > s[0] * max(centred.shape) * np.finfo(np.float64).eps))
    if rank < n_varying:
        raise ValueError(
            f"the {extent} of {points.shape[1]} validated points in {n_varying} varying "
            f"coordinates is flat (affine rank {rank}): every new member would lie outside "
            f"it; use more runs (at least {n_varying + 1}, spread in every coordinate), fewer "
            "coordinates, or extent 'box'"
        )


def domain_from_coordinates(
    coordinates: NDArray[np.float64],
    residuals: NDArray[np.float64],
    keys: Sequence[DataKey],
    *,
    extent: DomainExtent = "box",
    fields: Sequence[str] = (),
    field_times: Sequence[str] = (),
    time_range: tuple[str, str] | None = None,
    input_scale: float = 0.0,
) -> TrainingDomain:
    """Return the domain of the validated points.

    Parameters
    ----------
    coordinates
        ``(n_coordinates, n_points)``, the training and held-out inputs' coordinates.
    residuals
        ``(n_points,)``, their projection residuals (>= 0).
    keys
        The keys the surrogate was trained and gated on.
    extent
        ``box``, or also the convex ``hull`` or the ``mahalanobis`` ellipsoid of the points.
    fields, field_times
        The fields it was trained and gated on, and their times.
    time_range
        The validated interval of times; by default that of the keys' and fields' times.
    input_scale
        The inputs' magnitude (the largest norm of an input column), which scales the
        residual's round-off slack.

    Raises
    ------
    ValueError
        If the shapes disagree, a value is not finite or a residual is negative, or a hull or
        ellipsoid would be flat (fewer independent points than varying coordinates).
    """
    points = _as_matrix(coordinates, "coordinates")
    if extent != "box":
        _refuse_flat(points, extent)
    residual = np.asarray(residuals, dtype=np.float64)
    if residual.shape != (points.shape[1],) or points.shape[1] < 1:
        raise ValueError(
            f"residuals must be ({points.shape[1]},) for at least one point, got {residual.shape}"
        )
    if not np.all(np.isfinite(residual)) or np.any(residual < 0):
        raise ValueError("projection residuals must be finite and >= 0")
    lower = points.min(axis=-1)
    upper = points.max(axis=-1)
    bound = float(residual.max())
    if time_range is None:
        time_range = _time_range([k.time for k in keys] + list(field_times))
    return TrainingDomain(
        lower=lower,
        upper=upper,
        box_tolerance=RELATIVE_SLACK,
        residual_bound=bound,
        residual_tolerance=RELATIVE_SLACK * max(bound, float(input_scale)),
        keys=tuple(keys),
        n_points=points.shape[1],
        fields=tuple(fields),
        field_times=tuple(field_times),
        time_range=time_range,
        hull=points if extent == "hull" else None,
        mahalanobis=_mahalanobis(points) if extent == "mahalanobis" else None,
    )


def _input_scale(*inputs: SurrogateInputs) -> float:
    """Return the largest norm of an input column (all variables stacked)."""
    scale = 0.0
    for block in inputs:
        if not block:
            continue
        squares = sum(
            np.sum(np.square(np.asarray(v, dtype=np.float64)), axis=0) for v in block.values()
        )
        scale = max(scale, float(np.sqrt(np.max(squares))))
    return scale


def model_coordinates(
    model: SurrogateModel,
    inputs: SurrogateInputs,
    coarse: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a model's coordinates ``(d, n)`` and residuals ``(n,)`` of members.

    For a coarse-corrected model (``core.surrogates.CoarseCorrectedSurrogate``) with the
    members' coarse forecasts ``(n_keys, n)``: the inputs' coordinates, then the coarse
    forecast's, then its residual as one more coordinate (so that the box bounds it). For an
    early-start model (``core.surrogates.EarlyStartSurrogate``), ``coarse``
    carries the members' early data ``(n_early_keys, n)`` instead, bounded the same way.

    Raises
    ------
    ValueError
        If a coarse-corrected model is given no coarse forecast, or an early-start one no
        early data.
    """
    found, residual = model.coordinates(inputs)
    matrix = _as_matrix(found, "the model's coordinates")
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if isinstance(model, CoarseCorrectedSurrogate):
        if coarse is None:
            raise ValueError("a coarse-corrected surrogate needs the members' coarse forecasts")
        extra, extra_residual = model.coarse_coordinates(np.asarray(coarse, dtype=np.float64))
        matrix = np.concatenate([
            matrix, _as_matrix(extra, "the coarse forecast's coordinates"),
            np.asarray(extra_residual, dtype=np.float64).reshape(1, -1),
        ])  # fmt: skip
    elif isinstance(model, EarlyStartSurrogate):
        if coarse is None:
            raise ValueError("an early-start surrogate needs the members' early data")
        extra, extra_residual = model.early_coordinates(np.asarray(coarse, dtype=np.float64))
        matrix = np.concatenate([
            matrix, _as_matrix(extra, "the early data's coordinates"),
            np.asarray(extra_residual, dtype=np.float64).reshape(1, -1),
        ])  # fmt: skip
    return np.ascontiguousarray(matrix), residual


def training_domain(
    model: SurrogateModel,
    training: SurrogateInputs,
    held_out: SurrogateInputs,
    *,
    coarse: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
    extent: DomainExtent = "box",
    fields: Sequence[str] = (),
    field_times: Sequence[str] = (),
    time_range: tuple[str, str] | None = None,
) -> TrainingDomain:
    """Return a fitted model's validated domain: its coordinates of both sets of inputs.

    ``training`` and ``held_out`` are ``{variable: (size, n_runs)}``; the keys are the
    model's own (:attr:`SurrogateModel.keys`). ``coarse``: the two sets' coarse forecasts,
    for a coarse-corrected model (:func:`model_coordinates`).
    """
    coordinates, residuals = [], []
    pairs = coarse if coarse is not None else (None, None)
    for inputs, forecast in zip((training, held_out), pairs, strict=True):
        found, residual = model_coordinates(model, inputs, forecast)
        coordinates.append(found)
        residuals.append(residual)
    return domain_from_coordinates(
        np.concatenate(coordinates, axis=-1),
        np.concatenate(residuals),
        model.keys,
        extent=extent,
        fields=fields,
        field_times=field_times,
        time_range=time_range,
        input_scale=_input_scale(training, held_out),
    )


# -------------------------------------------------------------------------------- checks


@dataclass(frozen=True)
class CoordinateExcess:
    """One coordinate of a member outside the box."""

    index: int
    value: float
    bound: float
    side: Literal["below", "above"]

    @property
    def excess(self) -> float:
        """How far outside, in the coordinate's units."""
        return abs(self.value - self.bound)


@dataclass(frozen=True)
class MemberOutside:
    """A member outside the domain, and why."""

    member_id: int
    coordinates: tuple[CoordinateExcess, ...] = ()
    residual: float | None = None
    residual_bound: float = 0.0
    distance: float | None = None
    radius: float = 0.0
    outside_hull: bool = False

    def describe(self) -> str:
        """Return the reasons, in one line."""
        reasons = []
        if self.coordinates:
            worst = sorted(self.coordinates, key=lambda c: -c.excess)[:3]
            parts = ", ".join(
                f"#{c.index} = {c.value:.6g} {c.side} {c.bound:.6g} by {c.excess:.3g}"
                for c in worst
            )
            more = len(self.coordinates) - len(worst)
            reasons.append(
                f"{len(self.coordinates)} coordinate(s) outside the box ({parts}"
                + (f", and {more} more)" if more else ")")
            )
        if self.residual is not None:
            reasons.append(
                f"projection residual {self.residual:.6g} above {self.residual_bound:.6g}"
            )
        if self.distance is not None:
            reasons.append(f"Mahalanobis distance {self.distance:.6g} above {self.radius:.6g}")
        if self.outside_hull:
            reasons.append("outside the convex hull of the validated points")
        return f"member {self.member_id}: " + "; ".join(reasons)


@dataclass(frozen=True)
class DomainCheck:
    """What lies outside a surrogate's domain (all empty: everything is inside)."""

    outside: tuple[MemberOutside, ...] = ()
    unknown_keys: tuple[DataKey, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    times_outside: tuple[str, ...] = ()
    time_range: tuple[str, str] | None = None

    @property
    def inside(self) -> bool:
        """Whether everything asked for is inside."""
        return not (self.outside or self.unknown_keys or self.unknown_fields or self.times_outside)

    @property
    def members(self) -> tuple[int, ...]:
        """The ids of the members outside."""
        return tuple(m.member_id for m in self.outside)

    def merged(self, other: "DomainCheck") -> "DomainCheck":
        """Return both checks' findings."""
        return DomainCheck(
            outside=self.outside + other.outside,
            unknown_keys=self.unknown_keys + other.unknown_keys,
            unknown_fields=self.unknown_fields + other.unknown_fields,
            times_outside=self.times_outside + other.times_outside,
            time_range=self.time_range or other.time_range,
        )

    def message(self) -> str:
        """Return what is outside, for an error or a report."""
        lines = []
        if self.outside:
            lines.append(f"{len(self.outside)} member(s) outside the surrogate's domain:")
            lines += [f"  {m.describe()}" for m in self.outside[:_LISTED]]
            if len(self.outside) > _LISTED:
                lines.append(f"  ... and {len(self.outside) - _LISTED} more")
        if self.unknown_keys:
            shown = ", ".join(str(k) for k in self.unknown_keys[:_LISTED])
            more = len(self.unknown_keys) - _LISTED
            lines.append(
                f"{len(self.unknown_keys)} key(s) it was not trained and gated on: {shown}"
                + (f", and {more} more" if more > 0 else "")
            )
        if self.unknown_fields:
            lines.append(f"fields it was not trained on: {', '.join(self.unknown_fields)}")
        if self.times_outside:
            span = "none" if self.time_range is None else " to ".join(self.time_range)
            lines.append(
                f"times outside its validated range ({span}): {', '.join(self.times_outside)}"
            )
        return "\n".join(lines) if lines else "inside the surrogate's domain"

    def raise_if_outside(self) -> None:
        """Raise if anything is outside.

        Raises
        ------
        SurrogateDomainError
            Naming the members, keys, fields and times outside, and by how much.
        """
        if not self.inside:
            raise SurrogateDomainError(
                "the surrogate refuses to extrapolate; " + self.message(), self.members
            )


def _in_hull(
    hull: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    slack: NDArray[np.float64],
    point: NDArray[np.float64],
) -> bool:
    """Whether ``point`` is within ``slack`` of the convex hull of ``hull``'s columns (an LP).

    Find ``lambda >= 0`` with ``sum(lambda) = 1`` and ``|hull @ lambda - point| <= slack``
    coordinate by coordinate: the box's round-off slack carries over to the hull. The LP is
    posed on coordinates scaled to the box (``[-1, 1]``); HiGHS's own feasibility tolerance
    (1e-7 in those units) may accept a point that much further out. Coordinates whose span is
    within their slack (constant up to round-off) are left to the box check.
    """
    span = upper - lower
    varying = span > slack
    if not np.any(varying):
        return True
    centre = 0.5 * (upper + lower)[varying]
    half = 0.5 * span[varying]
    scaled = (hull[varying] - centre[:, None]) / half[:, None]
    target = (point[varying] - centre) / half
    band = slack[varying] / half
    n_points = scaled.shape[1]
    # Always feasible: minimise t subject to |hull @ lambda - point|_i <= t * band_i. A pure
    # feasibility problem with bands of 1e-9 left HiGHS declaring validated vertices
    # infeasible (its tolerances are 1e-7); the minimiser lambda* is the combination closest
    # to the point in units of each coordinate's slack.
    weights = band[:, None]
    a_ub = np.block([[scaled, -weights], [-scaled, -weights]])
    result = linprog(
        c=np.concatenate([np.zeros(n_points), [1.0]]),
        A_ub=a_ub,
        b_ub=np.concatenate([target, -target]),
        A_eq=np.concatenate([np.ones(n_points), [0.0]])[None, :],
        b_eq=np.ones(1),
        bounds=(0, None),
        method="highs",
    )
    if result.status == 0 and result.x is not None:
        # Judge the best combination coordinate by coordinate: within each one's own slack,
        # plus the solver's accuracy in the box's units (never one coordinate's slack lent to
        # another, as a single threshold on t would).
        weights_found = np.asarray(result.x[:n_points], dtype=np.float64)
        miss = np.abs(scaled @ weights_found - target)
        return bool(np.all(miss <= band + _HULL_SOLVER_SLACK))
    raise ValueError(f"the convex-hull check could not decide ({result.message})")


def check_coordinates(
    domain: TrainingDomain,
    coordinates: NDArray[np.float64],
    residuals: NDArray[np.float64],
    member_ids: Sequence[int],
) -> DomainCheck:
    """Return the members outside the domain's inputs.

    Parameters
    ----------
    coordinates
        ``(n_coordinates, n_members)``, from the model the domain was built with.
    residuals
        ``(n_members,)``, their projection residuals.
    member_ids
        ``(n_members,)``, to name them.
    """
    points = np.ascontiguousarray(coordinates, dtype=np.float64)
    residual = np.asarray(residuals, dtype=np.float64).reshape(-1)
    n_members = len(member_ids)
    if points.shape != (domain.n_coordinates, n_members) or residual.shape != (n_members,):
        raise ValueError(
            f"expected coordinates ({domain.n_coordinates}, {n_members}) and residuals "
            f"({n_members},), got {points.shape} and {residual.shape}"
        )
    slack = _slack(domain.lower, domain.upper, domain.box_tolerance)
    low = domain.lower - slack
    high = domain.upper + slack
    bound = domain.mahalanobis
    outside: list[MemberOutside] = []
    for j, member in enumerate(member_ids):
        x = points[:, j]
        excess: list[CoordinateExcess] = []
        # A value that is not finite is outside: NaN compares false both ways.
        for i in np.flatnonzero(~((x >= low) & (x <= high))):
            below = bool(x[i] < low[i]) or bool(np.isnan(x[i]))
            excess.append(
                CoordinateExcess(
                    index=int(i),
                    value=float(x[i]),
                    bound=float(domain.lower[i] if below else domain.upper[i]),
                    side="below" if below else "above",
                )
            )
        r = float(residual[j])
        residual_out = r if not r <= domain.residual_bound + domain.residual_tolerance else None
        distance = None
        if bound is not None:
            centred = x[list(bound.coordinates)] - bound.mean
            along = bound.basis.T @ centred
            off = float(np.linalg.norm(centred - bound.basis @ along))
            d = (
                math.inf
                if off > bound.orthogonal_tolerance or not np.all(np.isfinite(x))
                else float(np.linalg.norm(along / bound.scales))
            )
            if not d <= bound.radius * (1 + RELATIVE_SLACK) + RELATIVE_SLACK:
                distance = d
        outside_hull = (
            domain.hull is not None
            and bool(np.all(np.isfinite(x)))
            and not _in_hull(domain.hull, domain.lower, domain.upper, slack, x)
        )
        if excess or residual_out is not None or distance is not None or outside_hull:
            outside.append(
                MemberOutside(
                    member_id=int(member),
                    coordinates=tuple(excess),
                    residual=residual_out,
                    residual_bound=domain.residual_bound,
                    distance=distance,
                    radius=0.0 if bound is None else bound.radius,
                    outside_hull=outside_hull,
                )
            )
    return DomainCheck(outside=tuple(outside), time_range=domain.time_range)


def check_request(
    domain: TrainingDomain,
    keys: Sequence[DataKey] = (),
    *,
    fields: Sequence[str] = (),
    times: Sequence[str] = (),
) -> DomainCheck:
    """Return the keys, fields and times asked for that lie outside the domain.

    A key is inside only if the surrogate was trained and gated on it; a time only if it is
    within the validated range (on the same axis: a step index is never inside a range of
    dates). Labels that are not times are outside.
    """
    known = set(domain.keys)
    unknown_keys = tuple(k for k in keys if k not in known)
    unknown_fields = tuple(f for f in fields if f not in domain.fields)
    times_outside: list[str] = []
    if times:
        if domain.time_range is None:
            times_outside = list(times)
        else:
            first, last = (time_position(t) for t in domain.time_range)
            for time in times:
                try:
                    position = time_position(time)
                except ValueError:
                    times_outside.append(time)
                    continue
                if position[0] != first[0] or not first <= position <= last:
                    times_outside.append(time)
    return DomainCheck(
        unknown_keys=unknown_keys,
        unknown_fields=unknown_fields,
        times_outside=tuple(times_outside),
        time_range=domain.time_range,
    )


def check_members(
    domain: TrainingDomain,
    model: SurrogateModel,
    inputs: SurrogateInputs,
    member_ids: Sequence[int],
    keys: Sequence[DataKey] = (),
    *,
    fields: Sequence[str] = (),
    times: Sequence[str] = (),
    coarse: NDArray[np.float64] | None = None,
) -> DomainCheck:
    """Return what lies outside the domain: members (by the model's coordinates), keys, times.

    ``inputs`` is ``{variable: (size, n_members)}`` and ``member_ids`` names its columns;
    ``coarse`` their coarse forecasts, for a coarse-corrected model (:func:`model_coordinates`).
    """
    coordinates, residuals = model_coordinates(model, inputs, coarse)
    found = check_coordinates(domain, coordinates, residuals, member_ids)
    return found.merged(check_request(domain, keys, fields=fields, times=times))


def refuse_outside(
    domain: TrainingDomain,
    model: SurrogateModel,
    inputs: SurrogateInputs,
    member_ids: Sequence[int],
    keys: Sequence[DataKey] = (),
    coarse: NDArray[np.float64] | None = None,
) -> None:
    """Raise unless every member and key lies inside the domain (a surrogate simulator's guard).

    ``coarse``: the members' coarse forecasts, for a coarse-corrected model.

    Raises
    ------
    SurrogateDomainError
        Naming the members, keys and times outside, and by how much.
    """
    check_members(domain, model, inputs, member_ids, keys, coarse=coarse).raise_if_outside()


def refuse_keys_outside(
    domain: TrainingDomain,
    model: SurrogateModel,
    inputs: SurrogateInputs,
    member_ids: Sequence[int],
    keys: Sequence[DataKey] = (),
    coarse: NDArray[np.float64] | None = None,
) -> None:
    """Raise unless every key, field and time lies inside the domain; members may lie outside.

    The guard of a history match with ``outside = "surrogate"``: its members outside the
    domain are answered anyway by the user's explicit choice (the run logs and counts them),
    but a key, field or time the model was never gated on is still refused.

    Raises
    ------
    SurrogateDomainError
        Naming the keys, fields and times outside.
    """
    found = check_members(domain, model, inputs, member_ids, keys, coarse=coarse)
    dataclasses.replace(found, outside=()).raise_if_outside()
