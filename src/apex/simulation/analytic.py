"""Analytic forward models with known answers, for tests, demos and method checks."""

import datetime as dt
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.core.errors import SimulationError, TransientSimulationError
from apex.core.options import LEVELS, TEXT, InputFile, TextList
from apex.core.plugins import simulators
from apex.core.simulation import MemberInput, MemberSeries, SeriesRequest, match_vectors
from apex.domain.observations import DataKey


@simulators.register("linear")
class LinearModel:
    """``y = f(G x)`` for a fixed matrix ``G``, with ``f`` the identity or ``tanh``.

    With ``f`` the identity and a Gaussian prior, the exact posterior is known, which makes
    this the reference case for the workflow.

    Parameters
    ----------
    outputs
        ``[time, datatype]`` label of each scalar row of ``G``, in row order.
    fields
        ``[time, datatype, size]`` of each field output: ``size`` further rows of ``G``,
        after the scalar ones, with keys ``(time, datatype, 0..size-1)``.
    variables
        State variables stacked, in this order, into ``x``.
    matrix, forward_file
        ``G`` itself ``(n_outputs, n_state)``, or an ``.npz`` holding it as ``matrix``.
    nonlinearity
        ``"none"`` or ``"tanh"``.
    fail_members
        Member ids whose runs raise :class:`SimulationError`, to exercise failure handling.
    fail_above
        Runs whose state has a value above this raise :class:`SimulationError`: failures
        that appear mid-run, as iterates move (deterministic, so resume-safe).
    flaky_members
        Member id -> how many times its run fails transiently (a
        :class:`~apex.core.errors.TransientSimulationError`, as a licence checkout) before
        it succeeds; to test the executors' retries. Keys may be strings (TOML tables).
    delay_s
        Seconds each run waits before answering, to test scheduling and recovery with
        runs that take time.
    resources
        Machine-wide tokens each run holds, e.g. ``{"cpu": 1}``, to test
        admission; none by default.
    write_files
        Give each run a work directory and write ``output.txt`` there (the values), to
        test work-directory handling such as retention (``[output] keep_simulations``).
    levels
        Fidelity levels of a multilevel ensemble (the composition passes them):
        per level ``first``, ``size`` and optionally its own ``G`` (``matrix`` or
        ``forward_file``), e.g. a biased coarse model; a level without one uses ``G``.
        Grid upscaling does not apply to this model.

    Raises
    ------
    ValueError
        If ``G`` is missing, given twice, or inconsistent with ``outputs``.
    """

    def __init__(
        self,
        outputs: Annotated[Sequence[Sequence[str]], TEXT],
        variables: TextList,
        fields: Annotated[Sequence[Sequence[str | int]], TEXT] = (),
        matrix: ArrayLike | None = None,
        forward_file: InputFile | None = None,
        nonlinearity: Literal["none", "tanh"] = "none",
        fail_members: Sequence[int] = (),
        fail_above: float | None = None,
        flaky_members: Annotated[Mapping[str | int, int] | None, TEXT] = None,
        delay_s: float = 0.0,
        resources: Annotated[Mapping[str, int] | None, TEXT] = None,
        levels: Annotated[Sequence[Mapping[str, object]] | None, LEVELS] = None,
        write_files: bool = False,
    ) -> None:
        if (matrix is None) == (forward_file is None):
            raise ValueError("give exactly one of matrix and forward_file")
        if matrix is None:
            with np.load(Path(str(forward_file))) as data:
                matrix = data["matrix"]
        self.matrix: NDArray[np.float64] = np.asarray(matrix, dtype=np.float64)
        field_keys = [DataKey(str(t), str(d), i) for t, d, size in fields for i in range(int(size))]
        self.outputs = tuple(DataKey(str(t), str(d)) for t, d in outputs) + tuple(field_keys)
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.outputs):
            raise ValueError(
                f"matrix must have one row per output value ({len(self.outputs)}), "
                f"got shape {self.matrix.shape}"
            )
        if nonlinearity not in ("none", "tanh"):
            raise ValueError(f"nonlinearity must be 'none' or 'tanh', got {nonlinearity!r}")
        self.variables = tuple(variables)
        self.nonlinearity = nonlinearity
        self.fail_members = frozenset(fail_members)
        self.fail_above = fail_above
        self.flaky_members = {int(k): int(v) for k, v in (flaky_members or {}).items()}
        self._flaky_failures: dict[int, int] = {}
        self.delay_s = delay_s
        if resources:
            self.resources = dict(resources)
        self._rows = {key: i for i, key in enumerate(self.outputs)}
        if write_files:
            self.needs_work_dir = True
        self._levels: list[tuple[int, int, NDArray[np.float64]]] = []
        for index, level in enumerate(levels or []):
            if level.get("upscale") is not None:
                raise ValueError("the linear model has no grid to upscale")
            operator = self.matrix
            if level.get("forward_file") is not None:
                with np.load(Path(str(level["forward_file"]))) as data:
                    operator = np.asarray(data["matrix"], dtype=np.float64)
            elif level.get("matrix") is not None:
                operator = np.asarray(level["matrix"], dtype=np.float64)
            if operator.shape != self.matrix.shape:
                raise ValueError(f"level {index}: G must have shape {self.matrix.shape}")
            first, size = int(str(level["first"])), int(str(level["size"]))
            self._levels.append((first, first + size, operator))

    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        """Evaluate the model and return its scalar outputs as series.

        The times are those of the scalar outputs, in order of first appearance; every
        requested vector must have an output at every one of them.
        """
        scalar = [key for key in self.outputs if key.index is None]
        try:
            vectors = match_vectors(request.vectors, sorted({k.datatype for k in scalar}))
        except ValueError as error:
            raise SimulationError(f"linear model: {error}") from error
        times = tuple(dict.fromkeys(k.time for k in scalar if k.datatype in vectors))
        if request.times is not None:
            missing = [t for t in request.times if t not in times]
            if missing:
                raise SimulationError(f"linear model has no output at {', '.join(missing)}")
            times = tuple(request.times)
        keys = [DataKey(t, v) for v in vectors for t in times]
        values = self.run(member, keys).reshape(len(vectors), len(times))
        return MemberSeries(times, vectors, values)

    def run_with_sensitivities(
        self, member: MemberInput, keys: Sequence[DataKey]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the model and its exact Jacobian at ``keys``.

        The Jacobian is ``G`` (``diag(1 - tanh^2) G`` for ``tanh``) over the member's whole
        state, in its variables' order; variables the model does not read get zeros.
        """
        values = self.run(member, keys)
        operator = next(
            (g for first, end, g in self._levels if first <= member.member_id < end), self.matrix
        )
        x = np.concatenate([np.asarray(member.state[n], dtype=np.float64) for n in self.variables])
        derivative = operator
        if self.nonlinearity == "tanh":
            derivative = (1.0 - np.tanh(operator @ x) ** 2)[:, None] * operator
        rows = [self._rows[key] for key in keys]
        columns, start = [], 0
        offsets = {}
        for name in self.variables:
            offsets[name] = start
            start += np.asarray(member.state[name]).size
        for name, value in member.state.items():
            size = np.asarray(value).size
            if name in offsets:
                columns.append(derivative[rows, offsets[name] : offsets[name] + size])
            else:
                columns.append(np.zeros((len(rows), size)))
        return values, np.hstack(columns)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Evaluate the model for one member at ``keys``."""
        if member.member_id in self.fail_members:
            raise SimulationError(f"member {member.member_id} configured to fail")
        failed = self._flaky_failures.get(member.member_id, 0)
        if failed < self.flaky_members.get(member.member_id, 0):
            self._flaky_failures[member.member_id] = failed + 1
            raise TransientSimulationError(
                f"member {member.member_id}: licence not available (flaky, {failed + 1})"
            )
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        x = np.concatenate(
            [np.asarray(member.state[name], dtype=np.float64) for name in self.variables]
        )
        if x.shape != (self.matrix.shape[1],):
            raise SimulationError(f"expected {self.matrix.shape[1]} state values, got {x.shape}")
        if self.fail_above is not None and np.any(x > self.fail_above):
            raise SimulationError(f"member {member.member_id}: state above {self.fail_above}")
        operator = next(
            (g for first, end, g in self._levels if first <= member.member_id < end), self.matrix
        )
        y = operator @ x
        if self.nonlinearity == "tanh":
            y = np.tanh(y)
        try:
            rows = [self._rows[key] for key in keys]
        except KeyError as error:
            raise SimulationError(f"model has no output {error.args[0]}") from None
        result: NDArray[np.float64] = y[rows]
        if member.work_dir is not None and getattr(self, "needs_work_dir", False):
            (member.work_dir / "output.txt").write_text(" ".join(map(repr, result.tolist())))
        return result


@simulators.register("toy_reservoir")
class ToyReservoir:
    """Field totals of a toy waterflood, for optimization tests and demos.

    Each injector ``i`` with control ``x_i`` (its pressure) sweeps
    ``oil_rate * sqrt(x_i)`` of oil a year and injects ``water_rate * x_i`` of water, scaled
    by the realization's ``perm`` (a model array, default 1); the totals grow linearly from
    ``start``. With an oil price ``p_o`` and an injection cost ``p_w`` the yearly cash flow
    of each injector, ``p_o * oil_rate * sqrt(x) - p_w * water_rate * x``, peaks at
    ``x = (p_o * oil_rate / (2 p_w * water_rate))^2``.

    Parameters
    ----------
    start
        The simulation's start (ISO date); ``TIME`` is the days since.
    control
        The name of the controls block (the member's state key).
    oil_rate, water_rate
        Yearly oil per square root of pressure, and water per unit of pressure.
    """

    status = "validated"

    def __init__(
        self,
        start: Annotated[str, TEXT] = "2000-01-01",
        control: Annotated[str, TEXT] = "inj",
        oil_rate: float = 20.0,
        water_rate: float = 1.0,
    ) -> None:
        self.start = dt.date.fromisoformat(start)
        self.control = control
        self.oil_rate = oil_rate
        self.water_rate = water_rate

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return FOPT, FWIT (FGPT and FWPT are zero) or TIME at each key's date."""
        if self.control not in member.state:
            raise SimulationError(f"member {member.member_id} has no controls {self.control!r}")
        x = np.asarray(member.state[self.control], dtype=float)
        if np.any(x < 0):
            raise SimulationError(f"member {member.member_id}: negative controls {x}")
        perm = float(np.asarray(member.state.get("perm", [1.0]), dtype=float).ravel()[0])
        yearly = {
            "FOPT": self.oil_rate * perm * float(np.sum(np.sqrt(x))),
            "FWIT": self.water_rate * float(np.sum(x)),
            "FGPT": 0.0,
            "FWPT": 0.0,
        }
        values = np.empty(len(keys))
        for i, key in enumerate(keys):
            days = (dt.date.fromisoformat(key.time) - self.start).days
            if key.datatype == "TIME":
                values[i] = days
            elif key.datatype in yearly:
                values[i] = yearly[key.datatype] * days / 365.0
            else:
                raise SimulationError(f"the toy reservoir has no vector {key.datatype}")
        return values
