"""Surrogate models' data: snapshots, the validated domain and the gate's verdict.

A surrogate stands in for one simulator stack on one case. It maps what the simulator
receives, a member's ``MemberInput.state`` (after any parameterization), to what the
simulator answers at :class:`~apex.domain.observations.DataKey` s. This module holds its typed
data, with no I/O:

- :class:`SnapshotSet`: the full runs a surrogate is fitted and gated on, the ensemble axis
  last;
- :class:`TrainingDomain`: where the surrogate was *validated* (inputs, keys, times), built by
  core from the plugin's coordinates (``algorithms/surrogates/domain.py``), never by the
  plugin;
- :class:`GateResult` and :class:`GateTolerances`: the accuracy gate's verdict and its
  tolerances, which a user of the surrogate may tighten but never loosen;
- :class:`SurrogateManifest`: the JSON half of a fitted surrogate's files (the other half is
  the plugin's arrays, ``storage/surrogate_files.py``);
- :class:`SurrogateConfig`: a build's configuration (``apex surrogate build``, §7).
"""

import datetime as dt
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from apex.domain.config import PluginSection
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout

#: The manifest's format; readers refuse any other (raise it when the meaning changes).
MANIFEST_FORMAT: Literal[1] = 1

#: The kinds of quantity the gate tells apart without observation errors (§3): pressures are
#: judged on their change from the initial value, the others on their magnitude.
QuantityKind = Literal["pressure", "rate", "plume", "mass", "other"]

#: How a datatype was gated: against the observation error (``sigma_ratio``) or relative to
#: the reference's scale (``relative``).
GateMeasure = Literal["sigma_ratio", "relative", "field_rms"]

#: The shape of a domain beyond its box: none, the convex hull of the validated coordinates,
#: or their Mahalanobis ellipsoid.
DomainExtent = Literal["box", "hull", "mahalanobis"]

#: The relative tolerances by kind of quantity when there are no observation errors: those of
#: APEX's accuracy gate for reduced simulation models (coarse, vertical-equilibrium, aquifer
#: models against the fine reference), as starting values.
DEFAULT_RELATIVE: Mapping[QuantityKind, float] = MappingProxyType(
    {"pressure": 0.05, "rate": 0.05, "plume": 0.1, "mass": 0.01, "other": 0.05}
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _array(value: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    """Return a read-only, C-contiguous float64 copy with ``ndim`` dimensions, all finite.

    A copy, so that the caller's array cannot change the set afterwards; C-contiguous, as
    every ensemble array APEX holds (NumPy and OpenBLAS round the layouts differently).
    """
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} holds values that are not finite")
    array.setflags(write=False)
    return array


def key_document(key: DataKey) -> tuple[str, str, int | None]:
    """Return a key as ``(time, datatype, index)``, as manifests store it."""
    return (key.time, key.datatype, key.index)


def time_position(label: str) -> tuple[int, float]:
    """Return a time label's place on its axis: ``(0, step)`` or ``(1, POSIX seconds)``.

    Time labels are step indices as text or ISO dates (date-times), as
    :class:`~apex.domain.observations.DataKey` says; naive date-times are taken as UTC. The
    first element tells the two axes apart, so a step is never compared with a date.

    Raises
    ------
    ValueError
        If the label is neither.
    """
    text = label.strip()
    if re.fullmatch(r"-?\d+", text):
        return (0, float(int(text)))
    try:
        moment = dt.datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"time {label!r} is neither a step index nor an ISO date") from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.UTC)
    return (1, moment.timestamp())


# ----------------------------------------------------------------------------- snapshots


class SnapshotProvenance(_Frozen):
    """Where snapshots came from (§1): enough to refuse a surrogate for another stack or case.

    The digests are the ones run identity hashes.
    """

    simulator: str = Field(description="The simulator plugin (or stack) that ran them.")
    simulator_version: str | None = None
    options_digest: str | None = Field(default=None, description="SHA-256 of its options.")
    deck_digest: str | None = Field(default=None, description="SHA-256 of the deck template.")
    case_digest: str | None = Field(default=None, description="SHA-256 of the case's inputs.")
    plan: str | None = Field(default=None, description="The design: prior, lhs, file or run.")
    seed: int | None = None
    apex_commit: str | None = None
    source: str | None = Field(default=None, description="The run directory, if any.")
    coarse: "SnapshotProvenance | None" = Field(
        default=None, description="The coarse stack's, for coarse-corrected surrogates."
    )

    @model_serializer(mode="wrap")
    def _without_absent_coarse(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Absent before coarse stacks existed: manifests and snapshot digests
        # of surrogates without one keep their bytes.
        document: dict[str, Any] = handler(self)
        if document.get("coarse") is None:
            document.pop("coarse", None)
        return document

    def stack_mismatches(self, other: "SnapshotProvenance") -> list[str]:
        """Return the fields naming the stack and case on which ``other`` differs.

        Only what both state is compared: an unknown digest is not a difference. The coarse
        stack's fields are prefixed ``coarse.``.
        """
        names = ("simulator", "options_digest", "deck_digest", "case_digest")
        found = [
            n
            for n in names
            if getattr(self, n) is not None
            and getattr(other, n) is not None
            and getattr(self, n) != getattr(other, n)
        ]
        if self.coarse is not None and other.coarse is not None:
            found += [f"coarse.{n}" for n in self.coarse.stack_mismatches(other.coarse)]
        return found


@dataclass(frozen=True, eq=False)
class SnapshotSet:
    """Full runs of one simulator stack: the data a surrogate is fitted and gated on.

    Arrays keep the run (ensemble) axis last and are stored as read-only C-contiguous
    float64 copies.

    Attributes
    ----------
    layout
        The input variables, in order.
    inputs
        ``{variable: (size, n_runs)}``: what the simulator received, one per variable of
        ``layout``.
    keys
        The scalar data answered, ``(n_keys,)``; a series is one key per report time.
    outputs
        ``(n_keys, n_runs)``, the simulator's values at ``keys``.
    run_ids
        ``(n_runs,)``, each run's member id in its design.
    provenance
        Where the runs came from.
    failed
        The member ids of the design whose runs failed: recorded, never silently dropped.
    fields
        Optional state snapshots, ``{name: (n_cells, n_times, n_runs)}``.
    field_times
        ``(n_times,)``, the times of ``fields``.
    coarse_outputs
        Optional ``(n_keys, n_runs)``: a coarse simulation's values at the same ``keys`` for
        the same runs, for surrogates that correct a coarse forecast.
    coarse_only_inputs
        Optional ``{variable: (size, n_coarse_only)}``: the inputs of design members run on
        the coarse stack only (``[design] coarse_only``), e.g. for a network pretrained on
        many coarse runs and fine-tuned on the fewer fine ones. They are not runs of the set:
        :meth:`columns` keeps them whole, and the gate never sees them.
    coarse_only_outputs
        ``(n_keys, n_coarse_only)``: their coarse values at ``keys``; given with
        ``coarse_only_inputs`` and only with ``coarse_outputs``.
    """

    layout: StateLayout
    inputs: Mapping[str, NDArray[np.float64]]
    keys: tuple[DataKey, ...]
    outputs: NDArray[np.float64]
    run_ids: tuple[int, ...]
    provenance: SnapshotProvenance
    failed: tuple[int, ...] = ()
    fields: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    field_times: tuple[str, ...] = ()
    coarse_outputs: NDArray[np.float64] | None = None
    coarse_only_inputs: Mapping[str, NDArray[np.float64]] | None = None
    coarse_only_outputs: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Check every shape against the layout, the keys and the runs; copy the arrays."""
        n_runs = len(self.run_ids)
        if n_runs < 1:
            raise ValueError("a snapshot set needs at least one run")
        if len(set(self.run_ids)) != n_runs:
            raise ValueError("snapshot run ids must be unique")
        if set(self.failed) & set(self.run_ids):
            raise ValueError("a failed run cannot also be a snapshot")
        if set(self.inputs) != set(self.layout.names):
            raise ValueError(
                f"snapshot inputs {sorted(self.inputs)} must be the layout's variables "
                f"{sorted(self.layout.names)}"
            )
        inputs: dict[str, NDArray[np.float64]] = {}
        for variable in self.layout.variables:
            array = _array(self.inputs[variable.name], f"input {variable.name!r}", 2)
            if array.shape != (variable.size, n_runs):
                raise ValueError(
                    f"input {variable.name!r} must be ({variable.size}, {n_runs}), "
                    f"got {array.shape}"
                )
            inputs[variable.name] = array
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("snapshot keys must be unique")
        outputs = _array(self.outputs, "outputs", 2)
        if outputs.shape != (len(self.keys), n_runs):
            raise ValueError(f"outputs must be ({len(self.keys)}, {n_runs}), got {outputs.shape}")
        coarse = None
        if self.coarse_outputs is not None:
            coarse = _array(self.coarse_outputs, "coarse outputs", 2)
            if coarse.shape != outputs.shape:
                raise ValueError(
                    f"coarse outputs must be the outputs' {outputs.shape}, got {coarse.shape}"
                )
        only_inputs, only_outputs = self._coarse_only(coarse, len(self.keys))
        if self.fields and not self.field_times:
            raise ValueError("field snapshots need their field_times")
        fields: dict[str, NDArray[np.float64]] = {}
        for name, value in self.fields.items():
            array = _array(value, f"field {name!r}", 3)
            if array.shape[1:] != (len(self.field_times), n_runs):
                raise ValueError(
                    f"field {name!r} must be (n_cells, {len(self.field_times)}, {n_runs}), "
                    f"got {array.shape}"
                )
            fields[name] = array
        for time in self.field_times:
            time_position(time)
        object.__setattr__(self, "inputs", MappingProxyType(inputs))
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "run_ids", tuple(int(i) for i in self.run_ids))
        object.__setattr__(self, "failed", tuple(int(i) for i in self.failed))
        object.__setattr__(self, "field_times", tuple(self.field_times))
        object.__setattr__(self, "coarse_outputs", coarse)
        object.__setattr__(self, "coarse_only_inputs", only_inputs)
        object.__setattr__(self, "coarse_only_outputs", only_outputs)

    def _coarse_only(
        self, coarse: NDArray[np.float64] | None, n_keys: int
    ) -> tuple[Mapping[str, NDArray[np.float64]] | None, NDArray[np.float64] | None]:
        """Check and copy the coarse-only members (both or neither, with coarse outputs)."""
        if (self.coarse_only_inputs is None) != (self.coarse_only_outputs is None):
            raise ValueError("coarse-only members need both their inputs and their outputs")
        if self.coarse_only_inputs is None or self.coarse_only_outputs is None:
            return None, None
        if coarse is None:
            raise ValueError("coarse-only members need the runs' coarse outputs")
        outputs = _array(self.coarse_only_outputs, "coarse-only outputs", 2)
        n_only = outputs.shape[1]
        if n_only < 1 or outputs.shape[0] != n_keys:
            raise ValueError(
                f"coarse-only outputs must be ({n_keys}, n) with n >= 1, got {outputs.shape}"
            )
        if set(self.coarse_only_inputs) != set(self.layout.names):
            raise ValueError(
                f"coarse-only inputs {sorted(self.coarse_only_inputs)} must be the layout's "
                f"variables {sorted(self.layout.names)}"
            )
        inputs: dict[str, NDArray[np.float64]] = {}
        for variable in self.layout.variables:
            name = variable.name
            array = _array(self.coarse_only_inputs[name], f"coarse-only input {name!r}", 2)
            if array.shape != (variable.size, n_only):
                raise ValueError(
                    f"coarse-only input {name!r} must be ({variable.size}, {n_only}), "
                    f"got {array.shape}"
                )
            inputs[name] = array
        return MappingProxyType(inputs), outputs

    @property
    def n_coarse_only(self) -> int:
        """Number of coarse-only members (0 without them)."""
        return 0 if self.coarse_only_outputs is None else self.coarse_only_outputs.shape[1]

    @property
    def n_runs(self) -> int:
        """Number of runs (the last axis of every array)."""
        return len(self.run_ids)

    def columns(self, positions: Sequence[int]) -> "SnapshotSet":
        """Return the runs at ``positions`` (e.g. the training or the held-out ones).

        ``failed`` is the design's, and kept, as are the coarse-only members (not runs of
        the set).
        """
        index = np.asarray(positions, dtype=np.intp)
        return SnapshotSet(
            layout=self.layout,
            inputs={name: array[:, index] for name, array in self.inputs.items()},
            keys=self.keys,
            outputs=self.outputs[:, index],
            run_ids=tuple(self.run_ids[i] for i in index),
            provenance=self.provenance,
            failed=self.failed,
            fields={name: array[:, :, index] for name, array in self.fields.items()},
            field_times=self.field_times,
            coarse_outputs=None if self.coarse_outputs is None else self.coarse_outputs[:, index],
            coarse_only_inputs=self.coarse_only_inputs,
            coarse_only_outputs=self.coarse_only_outputs,
        )

    def digest(self) -> str:
        """Return the SHA-256 of the set's content: arrays, keys, ids and provenance.

        Independent of how the set is stored, so that a manifest can name the snapshots a
        surrogate was fitted on and a refit can be checked against them.
        """
        digest = hashlib.sha256(b"apex-snapshot-set-1\n")

        def text(value: str) -> None:
            data = value.encode()
            digest.update(len(data).to_bytes(8, "little") + data)

        def array(value: NDArray[np.float64]) -> None:
            text(repr(value.shape))
            digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())

        for variable in self.layout.variables:
            text(f"{variable.name}|{variable.size}|{variable.lower!r}|{variable.upper!r}")
            text(repr(variable.grid))
            array(self.inputs[variable.name])
        for key in self.keys:
            text(repr(key_document(key)))
        array(self.outputs)
        text(repr(self.run_ids))
        text(repr(self.failed))
        for name in sorted(self.fields):
            text(name)
            array(self.fields[name])
        text(repr(self.field_times))
        text(self.provenance.model_dump_json())
        if self.coarse_outputs is not None:
            # Only when there are some, so that sets without them keep their digests.
            text("coarse_outputs")
            array(self.coarse_outputs)
        if self.coarse_only_inputs is not None and self.coarse_only_outputs is not None:
            # Likewise: sets without coarse-only members keep their digests.
            text("coarse_only")
            for variable in self.layout.variables:
                array(self.coarse_only_inputs[variable.name])
            array(self.coarse_only_outputs)
        return digest.hexdigest()


# ------------------------------------------------------------------------------- domain


@dataclass(frozen=True, eq=False)
class MahalanobisBound:
    """The ellipsoid of the validated coordinates: within ``radius`` of their mean.

    It covers only the coordinates that vary beyond round-off; the others are the box's (a
    coordinate constant up to a few ulps would otherwise give a direction of scale ~1e-14).

    Attributes
    ----------
    coordinates
        ``(n_covered,)``, the indices of the coordinates it covers.
    mean
        ``(n_covered,)``.
    basis
        ``(n_covered, rank)``, orthonormal: the covered coordinates' principal directions.
    scales
        ``(rank,)``, the standard deviation along each direction.
    radius
        The largest Mahalanobis distance of a validated point.
    orthogonal_tolerance
        How far off the principal directions a point may lie (round-off), absolute.
    """

    coordinates: tuple[int, ...]
    mean: NDArray[np.float64]
    basis: NDArray[np.float64]
    scales: NDArray[np.float64]
    radius: float
    orthogonal_tolerance: float

    def __post_init__(self) -> None:
        """Check the shapes and copy the arrays."""
        mean = _array(self.mean, "mean", 1)
        basis = _array(self.basis, "basis", 2)
        scales = _array(self.scales, "scales", 1)
        if basis.shape != (mean.size, scales.size) or np.any(scales <= 0):
            raise ValueError("a Mahalanobis bound needs (n, r) basis, (r,) positive scales")
        if len(self.coordinates) != mean.size:
            raise ValueError("a Mahalanobis bound needs one mean per covered coordinate")
        object.__setattr__(self, "coordinates", tuple(int(i) for i in self.coordinates))
        if not (math.isfinite(self.radius) and self.radius >= 0):
            raise ValueError("a Mahalanobis radius must be finite and >= 0")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "scales", scales)


class _MahalanobisDocument(_Frozen):
    coordinates: list[int]
    mean: list[float]
    basis: list[list[float]]
    scales: list[float]
    radius: float
    orthogonal_tolerance: float


class TrainingDomainDocument(_Frozen):
    """A :class:`TrainingDomain` as JSON (a manifest's ``domain``)."""

    lower: list[float]
    upper: list[float]
    box_tolerance: float
    residual_bound: float
    residual_tolerance: float
    keys: list[tuple[str, str, int | None]]
    fields: list[str] = Field(default_factory=list)
    field_times: list[str] = Field(default_factory=list)
    time_range: tuple[str, str] | None = None
    n_points: int
    hull: list[list[float]] | None = None
    mahalanobis: _MahalanobisDocument | None = None


@dataclass(frozen=True, eq=False)
class TrainingDomain:
    """Where a surrogate was validated (§1): its inputs, keys and times.

    Built by core from the plugin's coordinates of the training and held-out inputs; a
    plugin cannot widen it. A member is inside when its coordinates are in the box (and in
    the hull or ellipsoid, if any) and its projection residual within the bound: a field far
    from the training subspace is outside even if its coordinates are inside the box.

    Attributes
    ----------
    lower, upper
        ``(n_coordinates,)``, the box of the validated coordinates.
    box_tolerance
        Relative slack of the box, for round-off (coordinates computed in batches of other
        sizes round differently).
    residual_bound
        The largest projection residual of a validated input.
    residual_tolerance
        Absolute slack of ``residual_bound``, for round-off.
    keys
        The keys the surrogate was trained and gated on; any other is outside.
    fields
        The fields it was trained and gated on.
    field_times
        The times of those fields.
    time_range
        ``(first, last)``, the times the gate checked on held-out runs (``None``: only the
        times of ``keys``). A surrogate that extrapolates in time is valid up to ``last``.
    n_points
        How many validated points the domain was built from.
    hull
        ``(n_coordinates, n_points)``, the validated coordinates whose convex hull bounds the
        domain, or ``None``.
    mahalanobis
        The ellipsoid that bounds the domain, or ``None``.
    """

    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    box_tolerance: float
    residual_bound: float
    residual_tolerance: float
    keys: tuple[DataKey, ...]
    n_points: int
    fields: tuple[str, ...] = ()
    field_times: tuple[str, ...] = ()
    time_range: tuple[str, str] | None = None
    hull: NDArray[np.float64] | None = None
    mahalanobis: MahalanobisBound | None = None

    def __post_init__(self) -> None:
        """Check the bounds and copy the arrays."""
        lower = _array(self.lower, "lower", 1)
        upper = _array(self.upper, "upper", 1)
        if lower.shape != upper.shape or np.any(lower > upper):
            raise ValueError("a domain's box needs lower <= upper, of equal sizes")
        for name in ("box_tolerance", "residual_bound", "residual_tolerance"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"a domain's {name} must be finite and >= 0")
        if self.n_points < 1:
            raise ValueError("a domain needs at least one validated point")
        if self.time_range is not None:
            first, last = (time_position(t) for t in self.time_range)
            if first[0] != last[0] or first > last:
                raise ValueError(f"time range {self.time_range} is not an interval")
        if self.hull is not None:
            hull = _array(self.hull, "hull", 2)
            if hull.shape[0] != lower.size:
                raise ValueError("a domain's hull points need one row per coordinate")
            object.__setattr__(self, "hull", hull)
        if self.mahalanobis is not None and any(
            not 0 <= i < lower.size for i in self.mahalanobis.coordinates
        ):
            raise ValueError("a domain's ellipsoid covers coordinates it does not have")
        for time in self.field_times:
            time_position(time)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "field_times", tuple(self.field_times))

    @property
    def n_coordinates(self) -> int:
        """Number of input coordinates."""
        return int(self.lower.size)

    @property
    def extent(self) -> DomainExtent:
        """The shape beyond the box."""
        if self.hull is not None:
            return "hull"
        return "mahalanobis" if self.mahalanobis is not None else "box"

    def to_document(self) -> TrainingDomainDocument:
        """Return the domain as JSON (floats round-trip exactly)."""
        bound = self.mahalanobis
        return TrainingDomainDocument(
            lower=self.lower.tolist(),
            upper=self.upper.tolist(),
            box_tolerance=self.box_tolerance,
            residual_bound=self.residual_bound,
            residual_tolerance=self.residual_tolerance,
            keys=[key_document(k) for k in self.keys],
            fields=list(self.fields),
            field_times=list(self.field_times),
            time_range=self.time_range,
            n_points=self.n_points,
            hull=None if self.hull is None else self.hull.tolist(),
            mahalanobis=None
            if bound is None
            else _MahalanobisDocument(
                coordinates=list(bound.coordinates),
                mean=bound.mean.tolist(),
                basis=bound.basis.tolist(),
                scales=bound.scales.tolist(),
                radius=bound.radius,
                orthogonal_tolerance=bound.orthogonal_tolerance,
            ),
        )

    @classmethod
    def from_document(cls, document: TrainingDomainDocument) -> "TrainingDomain":
        """Return the domain a document describes."""
        bound = document.mahalanobis
        return cls(
            lower=np.asarray(document.lower, dtype=np.float64),
            upper=np.asarray(document.upper, dtype=np.float64),
            box_tolerance=document.box_tolerance,
            residual_bound=document.residual_bound,
            residual_tolerance=document.residual_tolerance,
            keys=tuple(DataKey(t, d, i) for t, d, i in document.keys),
            n_points=document.n_points,
            fields=tuple(document.fields),
            field_times=tuple(document.field_times),
            time_range=document.time_range,
            hull=None if document.hull is None else np.asarray(document.hull, dtype=np.float64),
            mahalanobis=None
            if bound is None
            else MahalanobisBound(
                coordinates=tuple(bound.coordinates),
                mean=np.asarray(bound.mean, dtype=np.float64),
                basis=np.asarray(bound.basis, dtype=np.float64).reshape(
                    len(bound.mean), len(bound.scales)
                ),
                scales=np.asarray(bound.scales, dtype=np.float64),
                radius=bound.radius,
                orthogonal_tolerance=bound.orthogonal_tolerance,
            ),
        )


# --------------------------------------------------------------------------------- gate


class GateTolerances(_Frozen):
    """The accuracy gate's tolerances (§3; the user's starting values of 2026-10-31).

    A user of a surrogate may tighten them (:meth:`tightened`), never loosen them.
    """

    sigma_ratio: float = Field(
        default=0.5,
        gt=0,
        description="Largest RMS of (surrogate - simulator) / sigma per datatype.",
    )
    min_rank_correlation: float = Field(
        default=0.95,
        ge=-1,
        le=1,
        description="Smallest Spearman correlation of the held-out runs' objectives.",
    )
    relative: dict[QuantityKind, float] = Field(
        default_factory=lambda: dict(DEFAULT_RELATIVE),
        description="Largest relative error by kind of quantity, without observation errors.",
    )

    @field_validator("relative")
    @classmethod
    def _complete(cls, value: dict[QuantityKind, float]) -> dict[QuantityKind, float]:
        """Fill the kinds not given with the defaults; refuse a tolerance that is not > 0."""
        for kind, tolerance in value.items():
            if not (math.isfinite(tolerance) and tolerance > 0):
                raise ValueError(f"the relative tolerance of {kind} must be > 0")
        return {**DEFAULT_RELATIVE, **value}

    def tightened(
        self,
        *,
        sigma_ratio: float | None = None,
        min_rank_correlation: float | None = None,
        relative: Mapping[QuantityKind, float] | None = None,
    ) -> "GateTolerances":
        """Return these tolerances made stricter by the values given.

        Raises
        ------
        ValueError
            If a value would loosen the gate (a larger ratio or relative error, a smaller
            correlation), or a relative tolerance is not positive.
        """
        looser: list[str] = []
        if sigma_ratio is not None and sigma_ratio > self.sigma_ratio:
            looser.append(f"sigma_ratio {sigma_ratio} > {self.sigma_ratio}")
        if min_rank_correlation is not None and min_rank_correlation < self.min_rank_correlation:
            looser.append(
                f"min_rank_correlation {min_rank_correlation} < {self.min_rank_correlation}"
            )
        merged = dict(self.relative)
        for kind, value in (relative or {}).items():
            if not value > 0:
                raise ValueError(f"the relative tolerance of {kind} must be > 0, got {value}")
            if value > self.relative[kind]:
                looser.append(f"relative {kind} {value} > {self.relative[kind]}")
            merged[kind] = value
        if looser:
            raise ValueError(
                "a surrogate's gate may only be tightened, never loosened: " + "; ".join(looser)
            )
        return GateTolerances(
            sigma_ratio=self.sigma_ratio if sigma_ratio is None else sigma_ratio,
            min_rank_correlation=self.min_rank_correlation
            if min_rank_correlation is None
            else min_rank_correlation,
            relative=merged,
        )


class DatatypeGate(_Frozen):
    """One datatype (or field) through the gate."""

    datatype: str
    measure: GateMeasure
    kind: QuantityKind | None = Field(
        default=None, description="The kind of quantity, for a relative measure."
    )
    error: float | None = Field(
        description="RMS error / sigma, the largest relative error, or a field's pooled "
        "relative RMS error; None: not finite (a prediction that is not finite, or an error "
        "against a reference of scale 0)."
    )
    max_sigma_ratio: float | None = Field(
        default=None,
        description="The largest |surrogate - simulator| / sigma of one value (sigma_ratio "
        "only; the RMS is the criterion, this shows a single key far off).",
    )
    scale: Literal["change", "magnitude", "range"] | None = Field(
        default=None,
        description="What relative errors were taken against: a pressure's change, the "
        "magnitude (every other kind of series, and pressures that never change), or a "
        "field's range (fields other than pressures).",
    )
    notes: list[str] = Field(
        default_factory=list, description="Warnings about how the datatype was judged."
    )
    tolerance: float
    values: int = Field(description="How many values were compared.")
    passed: bool


class RankGate(_Frozen):
    """The held-out runs' objectives: surrogate against simulator, by rank."""

    correlation: float | None = Field(description="Spearman's rho; None if undefined.")
    tolerance: float
    runs: int
    passed: bool


class GateResult(_Frozen):
    """The gate's verdict on held-out full runs (§3)."""

    held_out: list[int] = Field(description="The held-out runs' ids (never used in fit).")
    failed: list[int] = Field(
        default_factory=list, description="Design runs that failed (reported, not gated)."
    )
    datatypes: list[DatatypeGate]
    rank: RankGate | None = None
    tolerances: GateTolerances
    passed: bool

    def failures(self) -> list[str]:
        """Return what failed, one line each (empty if it passed)."""
        lines = [
            f"{d.datatype}: {d.measure} error "
            f"{'not finite' if d.error is None else format(d.error, '.4g')} > {d.tolerance:.4g}"
            for d in self.datatypes
            if not d.passed
        ]
        if self.rank is not None and not self.rank.passed:
            rho = "undefined" if self.rank.correlation is None else f"{self.rank.correlation:.4g}"
            lines.append(f"objective rank correlation {rho} < {self.rank.tolerance:.4g}")
        if not self.datatypes and self.rank is None:
            lines.append("nothing was gated")
        return lines


# ----------------------------------------------------------------------------- manifest


class SnapshotReference(_Frozen):
    """The snapshot set a surrogate was fitted on: where it is stored, and its digest.

    ``path`` is relative to the manifest's folder: a bare file name beside it, or one in
    its parent (``../snapshots.npz``, a build's layout). Nothing else: a manifest must not
    make APEX read a file elsewhere.
    """

    path: str | None = None
    digest: str

    @field_validator("path")
    @classmethod
    def _beside(cls, value: str | None) -> str | None:
        if value is None:
            return value
        name = value.removeprefix("../")
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(
                f"a snapshot path is a file name beside the manifest or in its parent "
                f"(../name), got {value!r}"
            )
        return value


class FitRecord(_Frozen):
    """How the fit drew: the seed and the generator's state before the fit."""

    seed: int | None = None
    stream_state: dict[str, JsonValue] | None = None


class BuildCost(_Frozen):
    """What a build's full runs cost, kept beside its model (decision 14).

    A history-matching run with the model compares its surrogate forecasts' time with
    these, after the build's directory may be gone. Not part of the model's identity
    (``<manifest stem>.cost.json``): times differ between builds that give identical
    models. Seconds are each full run's own (start to end of one member, whatever ran
    beside it), over the design's successful runs.
    """

    identity: str = Field(description="The identity of the model whose build it is.")
    runs: int = Field(ge=0, description="The successful full runs timed.")
    mean_seconds: float = Field(ge=0, description="Their mean run time.")
    median_seconds: float = Field(ge=0, description="Their median run time.")
    total_seconds: float = Field(ge=0, description="Their summed run times (one-off cost).")
    fit_seconds: float | None = Field(default=None, description="The fit's time.")


class ArraysReference(_Frozen):
    """The arrays file beside the manifest (a bare file name) and its SHA-256."""

    file: str
    sha256: str


class SurrogateManifest(_Frozen):
    """The JSON half of a fitted surrogate (§2): what it is, where it is valid, how it was made.

    Loading it executes no code: the plugin is chosen by name from the registry and given
    ``model`` and the arrays; nothing in it is unpickled.
    """

    format: Literal[1] = MANIFEST_FORMAT
    plugin: str = Field(description="The plugin's registered name (surrogate_models).")
    plugin_version: str | None = None
    options: dict[str, JsonValue] = Field(default_factory=dict)
    keys: list[tuple[str, str, int | None]]
    fields: list[str] = Field(default_factory=list)
    field_times: list[str] = Field(default_factory=list)
    domain: TrainingDomainDocument
    gate: GateResult | None = None
    provenance: SnapshotProvenance
    snapshots: SnapshotReference | None = None
    fit: FitRecord = Field(default_factory=FitRecord)
    apex_commit: str | None = None
    model: dict[str, JsonValue] = Field(
        default_factory=dict, description="The plugin's own manifest (``to_arrays``)."
    )
    arrays: ArraysReference
    coarse: "CoarseSection | None" = Field(
        default=None,
        description="The coarse stack a coarse-corrected surrogate runs, its "
        "file options relative to the manifest's folder.",
    )
    control_box: dict[str, tuple[list[float], list[float]]] | None = Field(
        default=None,
        description="An optimization's surrogate: each control's entrywise "
        "range over the design's runs, the box a check draws within.",
    )

    @model_serializer(mode="wrap")
    def _without_absent_coarse(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Absent before coarse stacks and control boxes
        # existed: manifests without them keep their bytes, so their identities do not change.
        document: dict[str, Any] = handler(self)
        for name in ("coarse", "control_box"):
            if document.get(name) is None:
                document.pop(name, None)
        return document

    def data_keys(self) -> tuple[DataKey, ...]:
        """Return the keys the surrogate answers."""
        return tuple(DataKey(t, d, i) for t, d, i in self.keys)

    def training_domain(self) -> TrainingDomain:
        """Return the validated domain."""
        return TrainingDomain.from_document(self.domain)


# ------------------------------------------------------------------------ configuration


class SurrogateBuildSection(_Frozen):
    """``[surrogate]`` of a build: the plugin, its options and the build's seed."""

    name: str = Field(min_length=1, description="The build's name: runs/<name> by default.")
    model: str = Field(min_length=1, description="The surrogate plugin (surrogate_models).")
    options: dict[str, Any] = Field(default_factory=dict, description="The plugin's options.")
    seed: int = Field(ge=0, description="Root seed of the build's streams (design, surrogate).")
    extent: DomainExtent = Field(
        default="box",
        description="The validated domain beyond its box: `box`, the convex `hull` of the "
        "validated coordinates, or their `mahalanobis` ellipsoid.",
    )
    output_dir: Path | None = Field(
        default=None, description="The build's directory; default `runs/<name>` next to it."
    )

    @field_validator("name")
    @classmethod
    def _one_folder(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in (".", ".."):
            raise ValueError("a build's name is one folder name (no separators, not . or ..)")
        return value


class SurrogateSeries(_Frozen):
    """Outputs named by summary vectors and times: one key ``(time, vector)`` each."""

    vectors: list[str] = Field(
        min_length=1, description="Summary vectors, e.g. 'WOPR:P1' (no patterns)."
    )
    times: list[str] = Field(
        min_length=1, description="Their times: step indices or ISO dates, in order."
    )

    @field_validator("vectors")
    @classmethod
    def _no_patterns(cls, value: list[str]) -> list[str]:
        patterns = [v for v in value if any(c in v for c in "*?[")]
        if patterns:
            raise ValueError(f"list the vectors explicitly (no patterns): {patterns}")
        if len(set(value)) != len(value):
            raise ValueError("vectors repeat")
        return value


class SurrogateCaseSection(_Frozen):
    """``[case]`` of a build: the configuration the surrogate stands in for, and its outputs."""

    config_file: Path = Field(
        description="The history-matching or optimization configuration whose simulator "
        "stack it stands in for."
    )
    outputs: Literal["observations", "responses"] | SurrogateSeries = Field(
        default="observations",
        description="`observations`: a history-matching case's observed keys (every role), "
        "gated against their observation errors; `responses`: an optimization's objective's "
        "response keys, gated by relative errors and by the rank correlation of the "
        "objective; or `{ vectors, times }`, gated by relative errors.",
    )
    fields: list[str] = Field(
        default_factory=list,
        description="State fields every run also saves (restart arrays such as PRESSURE, "
        "SWAT) for field surrogates, gated by relative errors.",
    )
    field_times: list[str] = Field(
        default_factory=list,
        description="The dates (ISO) the fields are saved at; required with `fields`.",
    )
    field_cells: int | None = Field(
        default=None,
        gt=0,
        description="The fields' cells, for a simulator that answers field keys itself; a "
        "deck's grid gives them otherwise.",
    )

    @model_validator(mode="after")
    def _fields_have_times(self) -> "SurrogateCaseSection":
        if self.fields and not self.field_times:
            raise ValueError("[case] fields need field_times (the dates they are saved at)")
        if self.field_times and not self.fields:
            raise ValueError("[case] field_times without fields")
        if len({f.upper() for f in self.fields}) != len(self.fields):
            raise ValueError("[case] fields repeat a name (restart arrays ignore case)")
        return self


class DesignSection(_Frozen):
    """``[design]``: the full runs a surrogate trains on and is gated on (held-out).

    ``training`` runs are fitted on, ``held_out`` runs only gated on.

    The two are counted separately, as the designs draw them (the held-out runs first, fixed
    by the seed whatever the number of training runs): a design of ``training = 80`` and
    ``held_out = 20`` runs the simulator 100 times. the sketch named them ``runs``
    and ``holdout``.

    Plans:

    - ``prior``: drawn from the case's prior generators (fields keep their geostatistics);
    - ``lhs``: a Latin hypercube over the state's box (``box`` per variable, else the state
      variables' bounds, which must then be finite); no prior plugin offers a reduction of
      its fields today, so a hypercube is over every entry of every variable, which suits
      a few scalar inputs: use ``prior`` for fields;
    - ``file``: columns of a stored ensemble (``ensemble_file``, one array per variable);
    - ``run``: the forecasts an existing history-matching run holds (``run_dir``, its
      ``iterations``): free snapshots, no simulator run;
    - another plan of the registry ``surrogate_designs``, with its
      ``options``: e.g. an adaptive plan that chooses training runs where the surrogate is
      uncertain (the held-out runs are drawn from the prior first, fixed by the seed).

    With ``[[parameterizations]]`` in the case, the design is drawn in state space and each
    member mapped to the properties its simulator reads, which are the surrogate's inputs.

    ``coarse_only`` (with a ``[coarse]`` stack; plans ``prior``, ``lhs`` and ``file``) adds
    members run on the coarse stack only, drawn after the held-out and the training runs (so
    those stay the same for a given seed; plan ``file``: the next columns), e.g. for a
    network pretrained on many cheap coarse runs and fine-tuned on the fewer fine ones.
    """

    plan: str = Field(
        default="prior",
        description="prior, lhs, file, run, or another plan of the registry surrogate_designs.",
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict, description="Another plan's options (core's plans take none)."
    )
    training: int = Field(ge=2, description="Training runs: the plugin is fitted on these.")
    held_out: int = Field(ge=1, description="Held-out runs: the gate's, never fitted on.")
    box: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="lhs: per state variable, the bounds of every entry.",
    )
    ensemble_file: Path | None = Field(
        default=None, description="file: an .npz with one (size, n) array per state variable."
    )
    run_dir: Path | None = Field(default=None, description="run: a history-matching run.")
    iterations: Literal["all"] | list[int] = Field(
        default="all", description="run: its saved iterates whose members are pooled."
    )
    min_success_fraction: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Fail unless at least this share of the training and of the held-out "
        "runs succeeds.",
    )
    coarse_only: int = Field(
        default=0,
        ge=0,
        description="Members run on the [coarse] stack only, drawn after the others (for "
        "surrogates pretrained on coarse runs).",
    )

    @model_serializer(mode="wrap")
    def _without_absent_coarse_only(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Absent before coarse-only members existed: configurations, and the fingerprints
        # made from them, of builds without any keep their bytes.
        document: dict[str, Any] = handler(self)
        if not document.get("coarse_only"):
            document.pop("coarse_only", None)
        return document

    @model_validator(mode="after")
    def _plan_inputs(self) -> Self:
        needs = {"file": "ensemble_file", "run": "run_dir"}
        for plan, option in needs.items():
            given = getattr(self, option) is not None
            if (self.plan == plan) != given:
                raise ValueError(
                    f"{option} is for plan = {plan!r}"
                    if given
                    else f"plan = {plan!r} needs {option}"
                )
        if self.options and self.plan in ("prior", "lhs", "file", "run"):
            raise ValueError(f"plan = {self.plan!r} takes no options")
        if self.box and self.plan != "lhs":
            raise ValueError("box is for plan = 'lhs'")
        if self.iterations != "all" and self.plan != "run":
            raise ValueError("iterations are for plan = 'run'")
        if self.coarse_only and self.plan == "run":
            raise ValueError(
                "coarse_only is for drawn designs (prior, lhs, file): plan = 'run' reuses a "
                "run's forecasts and has no coarse runs; remove coarse_only"
            )
        for name, (lower, upper) in self.box.items():
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError(f"box of {name!r} needs finite bounds, lower <= upper")
        return self


class GateSection(_Frozen):
    """``[gate]``: the build's accuracy gate (the user's starting values)."""

    sigma_ratio: float = Field(
        default=0.5,
        gt=0,
        description="With observation errors: the largest RMS of (surrogate - simulator) / "
        "sigma per datatype over the held-out runs.",
    )
    relative: dict[QuantityKind, float] = Field(
        default_factory=dict,
        description="Without them: the largest relative error by kind of quantity (pressure, "
        "rate, plume, mass, other); the kinds not given keep their defaults.",
    )
    kinds: dict[str, QuantityKind] = Field(
        default_factory=dict,
        description="The kind of a datatype where its vector's mnemonic would mislead.",
    )
    min_rank_correlation: float | None = Field(
        default=None,
        ge=-1,
        le=1,
        description="With outputs = 'responses' (an optimization): the smallest Spearman "
        "correlation of the held-out runs' objectives, surrogate against simulator "
        "(default 0.95).",
    )

    @model_serializer(mode="wrap")
    def _without_absent_rank(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Absent before optimizations had surrogates: the fit
        # fingerprints of existing builds, made from this section, keep their bytes.
        document: dict[str, Any] = handler(self)
        if document.get("min_rank_correlation") is None:
            document.pop("min_rank_correlation", None)
        return document

    def tolerances(self) -> GateTolerances:
        """Return the gate's tolerances (the defaults where not given)."""
        if self.min_rank_correlation is None:
            return GateTolerances(sigma_ratio=self.sigma_ratio, relative=dict(self.relative))
        return GateTolerances(
            sigma_ratio=self.sigma_ratio,
            relative=dict(self.relative),
            min_rank_correlation=self.min_rank_correlation,
        )


class CoarseSection(_Frozen):
    """``[coarse]``: a coarse simulator stack every design run also runs on.

    For surrogates that correct a coarse forecast (``coarse_corrected``). The coarse stack is
    the case's, with ``simulator`` (default: the case's) and ``options`` merged over the
    case's simulator options (e.g. another ``deck_template_file``), and ``parameterizations``
    applied after the case's (e.g. ``block_average`` onto the coarse grid). Its runs answer
    the same keys as the case's.
    """

    simulator: str | None = Field(default=None, description="Default: the case's simulator.")
    options: dict[str, JsonValue] = Field(
        default_factory=dict, description="Merged over the case's simulator options."
    )
    parameterizations: list[PluginSection] = Field(
        default_factory=list, description="Applied after the case's (e.g. block_average)."
    )


class SurrogateOutputSection(_Frozen):
    """``[output]`` of a build."""

    keep_simulations: Literal["all", "failed", "none"] = Field(
        default="failed",
        description="Simulator work directories kept: all, only failed members' (marked "
        "APEX_FAILED), or none.",
    )


class SurrogateConfig(_Frozen):
    """A surrogate build: ``apex surrogate build CONFIG``.

    Example::

        [surrogate]
        name = "box_surrogate"
        model = "PLUGIN"
        options = {}
        seed = 7

        [case]
        config_file = "../gies.toml"
        outputs = "observations"

        [design]
        plan = "prior"
        training = 80
        held_out = 20

        [gate]
        sigma_ratio = 0.5

    Relative paths are resolved against the configuration's directory. ``[execution]``
    overrides the case's (where the full runs go); the case's simulator stack, its
    parameterizations and its ``[geophysics]`` are the surrogate's.
    """

    surrogate: SurrogateBuildSection
    case: SurrogateCaseSection
    design: DesignSection
    gate: GateSection = Field(default_factory=GateSection)
    execution: PluginSection | None = Field(
        default=None, description="Where the full runs go; default: the case's [execution]."
    )
    output: SurrogateOutputSection = Field(default_factory=SurrogateOutputSection)
    coarse: CoarseSection | None = Field(
        default=None, description="A coarse stack every run also runs on."
    )

    @model_validator(mode="after")
    def _rank_needs_responses(self) -> Self:
        if self.gate.min_rank_correlation is not None and self.case.outputs != "responses":
            raise ValueError(
                "[gate] min_rank_correlation gates an optimization's objective, which only "
                '[case] outputs = "responses" gives: remove it, or use those outputs'
            )
        return self

    @model_validator(mode="after")
    def _coarse_only_needs_coarse(self) -> Self:
        if self.design.coarse_only and self.coarse is None:
            raise ValueError(
                f"[design] coarse_only = {self.design.coarse_only} runs members on the "
                "[coarse] stack only: add a [coarse] stack, or remove coarse_only"
            )
        return self


# ------------------------------------------------------------------------ the workflow's results


class SurrogateBuildPlan(_Frozen):
    """What a build will cost (``apex validate``): nothing is run to say it."""

    name: str
    model: str
    plan: str
    training: int
    held_out: int
    simulator_runs: int = Field(description="Full runs the design needs (0 for plan 'run').")
    coarse_only: int = Field(default=0, description="Of them, runs on the coarse stack only.")
    keys: int = Field(description="The keys the surrogate answers.")
    datatypes: list[str]
    measure: GateMeasure
    simulator: str
    executor: str
    inputs: int = Field(description="Entries of the surrogate's inputs (after parameterizations).")
    output_dir: str
    warnings: list[str] = Field(default_factory=list)


class DomainSummary(_Frozen):
    """The size of a validated domain."""

    coordinates: int
    points: int
    extent: DomainExtent
    residual_bound: float


class SurrogateBuildSummary(_Frozen):
    """What a build did (``summary.json``; ``apex summary``'s summarizer reads it)."""

    name: str
    plugin: str
    plan: str
    status: Literal["complete", "stopped"]
    training_runs: int
    held_out_runs: int
    failed: list[int] = Field(default_factory=list)
    pending: int = 0
    simulator_runs: int = Field(description="Full runs taken (0 for plan 'run').")
    coarse_only_runs: int = Field(
        default=0, description="Successful runs on the coarse stack only ([design] coarse_only)."
    )
    coarse_only_failed: list[int] = Field(
        default_factory=list,
        description="Coarse-only members whose run failed; ``failed`` lists only the "
        "training and held-out runs that failed.",
    )
    keys: int
    datatypes: list[str]
    gate: GateResult | None = None
    identity: str | None = None
    model: str | None = Field(default=None, description="The manifest, relative to the build.")
    domain: DomainSummary | None = None
    fit_seconds: float | None = None
    prediction_seconds: float | None = Field(
        default=None, description="One member's prediction, averaged over the held-out runs."
    )
    simulator_median_seconds: float | None = Field(
        default=None, description="The median full run of the design."
    )

    @property
    def passed(self) -> bool | None:
        """The gate's verdict (``None`` before the fit)."""
        return None if self.gate is None else self.gate.passed


class SurrogateCheckSummary(_Frozen):
    """A check of a model on new full runs (``apex surrogate check``, ``checks/<id>/check.json``).

    ``checks/<id>/`` is the build's (or the model's folder's).
    """

    id: str
    identity: str
    model: str
    against: str
    source: str = Field(description="'prior' (a configuration's), or a run's 'iteration:<k>'.")
    status: Literal["complete", "stopped"]
    runs: int
    outside: list[int] = Field(
        default_factory=list, description="Members outside the domain: not gated."
    )
    failed: list[int] = Field(default_factory=list)
    trained_on: list[int] = Field(
        default_factory=list,
        description="Members left out: equal to one of the model's own runs (training or "
        "held-out).",
    )
    succeeded: int = Field(default=0, description="Runs that succeeded.")
    inside: int = Field(default=0, description="Successful runs inside the domain (gated).")
    min_success_fraction: float = 0.9
    min_inside: float = Field(
        default=0.8, description="The smallest share of successful runs inside the domain."
    )
    gate: GateResult | None = None
    passed: bool = Field(
        default=False,
        description="The gate passed and enough runs lay inside the domain.",
    )
    notes: list[str] = Field(default_factory=list)
    sequence: int = Field(
        default=0, description="The check's order among the model's complete checks (1, 2, ...)."
    )
    created: str


# The manifest names the coarse stack, defined with the build's sections.
SurrogateManifest.model_rebuild()
