"""C-GIES's soft constraints: constraint functions and their distance metrics.

The user's method: Luo, X. & Cruz, W. C. (2022). Data assimilation with soft constraints
(DASC) through a generalized iterative ensemble smoother. Computational Geosciences,
26(3), 571-594. doi:10.1007/s10596-022-10137-7 (equation labels of the manuscript
DASC_CG_R1.tex: eqs.
``HM_constrained_obj``-``GMAC_constrained_update_formula_mid``; MATLAB
``MATLAB .../FaciesStudy1/run_scripts/GIES_constraint_ensLoc.m``,
``.../Brugge_4DSeis/local_scripts/GIES_constraint_ensLoc.m``). PET has no such method.

Equality constraints ``f_eq(m) = 0`` and inequality constraints ``h_in(m) <= 0`` are
"perfect measurements" of ``0``: a constraint term adds ``w D_c[0 - c(m)]`` to GMAC's
data mismatch, with ``c`` a **constraint function** (registry ``gmac_constraints``) and
``D_c`` a **distance metric** (registry ``gmac_constraint_metrics``) of which the update
needs the gradient and the Hessian at ``x_j = 0 - c(m_j)``, never the constraint's
gradient with respect to ``m`` (the ensemble's anomalies of ``c`` stand in for it).

Arrays: constraint values ``(n_c, k)`` for ``k`` columns (members, or the mean model);
the ensemble axis is last.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.gmac.parts import Block, TVTransform, _select
from apex.algorithms.data_assimilation.registries import gmac_constraint_metrics, gmac_constraints
from apex.core.errors import ApexError
from apex.core.options import INPUT, TEXT, InputFile, Text, TextList, mapping

Array = NDArray[np.float64]
Kind = Literal["equality", "inequality"]


class ConstraintFunction(Protocol):
    """A constraint system ``c(m) = 0`` (equality) or ``c(m) <= 0`` (inequality)."""

    kind: Kind

    def values(self, x: Array, layout: Mapping[str, Block]) -> Array:
        """Return ``c`` ``(n_c, k)`` of the columns of ``x`` ``(n, k)`` (state or properties)."""
        ...

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` if the constraint does not fit the variables."""
        ...


class ConstraintMetric(Protocol):
    """A distance metric ``D(x)`` of a constraint's residual ``x = 0 - c``."""

    #: Its metric is the same for every member (a constant Hessian).
    constant: bool

    def cost(self, x: Array) -> Array:
        """Return ``D`` of each column of ``x`` ``(n_c, k)``, ``(k,)``."""
        ...

    def gradient(self, x: Array) -> Array:
        """Return ``grad D`` of each column, ``(n_c, k)``."""
        ...

    def metric(self, a: Array, x: Array) -> Array:
        """Return ``A^T H(x_j) A`` ``(N, N, N)`` for anomalies ``A`` ``(n_c, N)``."""
        ...


#: The roles of a constraint's ``function`` and ``metric`` tables, shared by
#: gmac's ``constraints`` and the metric ``constraint_cost``. Each new constraint function
#: or metric with other string options extends them here.
FUNCTION_ROLES = mapping(
    name=TEXT,
    options=mapping(bounds=TEXT, variable=TEXT, variables=TEXT, range=TEXT, bins=TEXT,
                    counts=TEXT, kind=TEXT, reference_file=INPUT, matrix_file=INPUT),
)  # fmt: skip
METRIC_ROLES = mapping(name=TEXT, options=mapping(curvature=TEXT))


def plugin_table(entry: object, what: str) -> tuple[str, dict[str, Any]]:
    """Return ``(name, options)`` of a ``{name, options}`` table.

    Raises
    ------
    ValueError
        If ``entry`` is not such a table.
    """
    if not isinstance(entry, Mapping) or "name" not in entry:
        raise ValueError(
            f"{what} must be a table {{ name = ..., options = {{ ... }} }}, got {entry!r}"
        )
    unknown = sorted(set(entry) - {"name", "options"})
    if unknown:
        raise ValueError(f"{what}: unknown keys {unknown}")
    options = entry.get("options") or {}
    if not isinstance(options, Mapping):
        raise ValueError(f"{what}: options must be a table, got {options!r}")
    return str(entry["name"]), dict(options)


class ConstraintViolation(ApexError, ValueError):
    """A member violates an inequality where its barrier is undefined."""


# ---------------------------------------------------------------------- functions


def _rows(layout: Mapping[str, Block], names: Sequence[str]) -> list[Block]:
    return _select(layout, list(names))


@gmac_constraints.register("box")
class Box:
    """Box constraints ``lower <= x <= upper`` per variable: ``h = [lower - x; x - upper]``.

    The paper's inequality system (2-D: PERMX in [100, 15000]; Brugge: log-permeabilities
    and porosity), MATLAB ``get_box_constraints.m``.

    Parameters
    ----------
    bounds
        ``{variable: [lower, upper]}`` (state variables, or properties with a
        parameterization).
    """

    kind: Kind = "inequality"

    def __init__(self, bounds: Annotated[Mapping[str, Sequence[float]], TEXT]) -> None:
        if not bounds:
            raise ValueError("box needs bounds = { variable = [lower, upper], ... }")
        self.bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        for name, (low, high) in self.bounds.items():
            if not low < high:
                raise ValueError(f"box bounds[{name!r}] must be [lower, upper] with lower < upper")

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` for unknown variables."""
        _rows(layout, list(self.bounds))

    def values(self, x: Array, layout: Mapping[str, Block]) -> Array:
        """Return ``[lower - x; x - upper]`` per variable, stacked ``(2 n, k)``."""
        parts = []
        for block in _rows(layout, list(self.bounds)):
            low, high = self.bounds[block.name]
            values = x[block.start : block.start + block.size]
            parts += [low - values, values - high]
        return np.concatenate(parts)


@gmac_constraints.register("histogram")
class Histogram:
    """The histogram of a variable equals a reference one: ``counts(x) - counts_ref = 0``.

    The paper's equality system (2-D: 50 bins of PERMX against the reference model's),
    MATLAB ``compare_hist.m``, but with fixed bins: ``bins`` equal bins over ``range``
    (values outside fall in the end bins), the same for every member and the reference
    (MATLAB's ``hist(x, 50)`` binned each sample over its own range).

    Parameters
    ----------
    variable
        The variable (state, or property with a parameterization).
    range
        ``[low, high]`` of the bins.
    bins
        Number of bins (MATLAB: 50).
    reference_file
        An ``.npy``, or an ``.npz`` with the variable's array (e.g. the truth), whose
        histogram is the target.
    counts
        Or the target counts per bin.
    """

    kind: Kind = "equality"

    def __init__(
        self,
        variable: Text,
        range: Sequence[float],
        bins: int = 50,
        reference_file: InputFile | None = None,
        counts: Sequence[float] | None = None,
    ) -> None:
        if bins < 2:
            raise ValueError(f"bins must be at least 2, got {bins}")
        self.variable, self.bins = variable, bins
        self.low, self.high = float(range[0]), float(range[1])
        if not self.low < self.high:
            raise ValueError("histogram range must be [low, high] with low < high")
        if (reference_file is None) == (counts is None):
            raise ValueError("histogram needs reference_file or counts (not both)")
        if counts is not None:
            target = np.asarray(counts, dtype=np.float64)
        else:
            target = self.counts(_read_reference(Path(str(reference_file)), variable)[:, None])[
                :, 0
            ]
        if target.shape != (bins,):
            raise ValueError(f"histogram counts must have {bins} entries, got {target.shape}")
        self.target = target

    def counts(self, values: Array) -> Array:
        """Return the counts per bin ``(bins, k)`` of each column of ``values``."""
        width = (self.high - self.low) / self.bins
        index = np.clip(np.floor((values - self.low) / width).astype(np.int64), 0, self.bins - 1)
        out = np.zeros((self.bins, values.shape[1]))
        for column in range(values.shape[1]):
            out[:, column] = np.bincount(index[:, column], minlength=self.bins)
        return out

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` for an unknown variable or a target of another size."""
        (block,) = _rows(layout, [self.variable])
        if block.size and not math.isclose(float(self.target.sum()), block.size):
            raise ValueError(
                f"histogram target counts sum to {self.target.sum():g}, but {self.variable!r} "
                f"has {block.size} values: it could never be met"
            )

    def values(self, x: Array, layout: Mapping[str, Block]) -> Array:
        """Return ``counts(x) - target`` ``(bins, k)``."""
        (block,) = _rows(layout, [self.variable])
        found: Array = self.counts(x[block.start : block.start + block.size]) - self.target[:, None]
        return found


def _read_reference(path: Path, variable: str) -> Array:
    data = np.load(path)
    if isinstance(data, np.ndarray):
        return np.asarray(data, dtype=np.float64).reshape(-1)
    with data:
        if variable not in data:
            raise ValueError(f"{path} has no array {variable!r} (has {sorted(data.files)})")
        return np.asarray(data[variable], dtype=np.float64).reshape(-1)


@gmac_constraints.register("tv")
class TotalVariation:
    """The total variation of gridded variables vanishes: ``f = TV(x) = 0``.

    MATLAB ``GIES_TVL12_constraint.m``, with the ``l1sq`` metric there.

    Parameters
    ----------
    variables
        The gridded variables; default every one with a grid.
    """

    kind: Kind = "equality"

    def __init__(self, variables: TextList | None = None) -> None:
        self.transform = TVTransform(variables)

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` for unknown or ungridded variables."""
        self.transform.check(layout)

    def values(self, x: Array, layout: Mapping[str, Block]) -> Array:
        """Return the first differences along each grid axis ``(n_f, k)``."""
        return self.transform.features(x, layout)


@gmac_constraints.register("linear")
class Linear:
    """A linear system ``A x - b`` (equality ``= 0`` or inequality ``<= 0``).

    Parameters
    ----------
    matrix_file
        An ``.npz`` with ``matrix`` ``(n_c, n)`` over the stacked ``variables`` and,
        optionally, ``rhs`` ``(n_c,)`` (default zeros).
    kind
        ``equality`` or ``inequality``.
    variables
        The variables ``A`` acts on, stacked in this order; default all.
    """

    def __init__(
        self,
        matrix_file: InputFile,
        kind: Kind = "equality",
        variables: TextList | None = None,
    ) -> None:
        if kind not in ("equality", "inequality"):
            raise ValueError(f"kind must be equality or inequality, got {kind!r}")
        self.kind = kind
        self.variables = tuple(variables or ())
        with np.load(Path(str(matrix_file))) as data:
            self.matrix = np.asarray(data["matrix"], dtype=np.float64)
            self.rhs = (
                np.asarray(data["rhs"], dtype=np.float64)
                if "rhs" in data
                else np.zeros(self.matrix.shape[0])
            )
        if self.matrix.ndim != 2 or self.rhs.shape != (self.matrix.shape[0],):
            raise ValueError("linear needs matrix (n_c, n) and rhs (n_c,)")

    def _rows(self, layout: Mapping[str, Block]) -> list[Block]:
        return _select(layout, list(self.variables) or None)

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` if the matrix does not fit the variables."""
        size = sum(b.size for b in self._rows(layout))
        if size and size != self.matrix.shape[1]:
            raise ValueError(f"linear matrix has {self.matrix.shape[1]} columns for {size} values")

    def values(self, x: Array, layout: Mapping[str, Block]) -> Array:
        """Return ``A x - b`` ``(n_c, k)``."""
        rows = np.concatenate([np.arange(b.start, b.start + b.size) for b in self._rows(layout)])
        out: Array = self.matrix @ x[rows] - self.rhs[:, None]
        return out


# ---------------------------------------------------------------------- metrics


def _diagonal_metric(a: Array, diagonals: Array) -> Array:
    """Return ``A^T diag(h_j) A`` for each column ``h_j`` of ``diagonals`` ``(n_c, N)``.

    For ``h_j >= 0`` as ``B^T B`` with ``B = A sqrt(h_j)`` (a symmetric product).
    """
    if np.all(diagonals >= 0):
        roots = np.sqrt(diagonals)
        scaled = [a * roots[:, [j]] for j in range(diagonals.shape[1])]
        return np.stack([b.T @ b for b in scaled])
    return np.stack([(a * diagonals[:, [j]]).T @ a for j in range(diagonals.shape[1])])


@gmac_constraint_metrics.register("quadratic")
class Quadratic:
    """``D(x) = 1/2 ||x||^2``: gradient ``x``, Hessian ``I`` (a perfect measurement)."""

    constant = True

    def cost(self, x: Array) -> Array:
        """Return ``1/2 ||x_j||^2``."""
        out: Array = 0.5 * np.sum(x**2, axis=0)
        return out

    def gradient(self, x: Array) -> Array:
        """Return ``x``."""
        return x

    def metric(self, a: Array, x: Array) -> Array:
        """Return ``A^T A`` for every member."""
        return np.broadcast_to(a.T @ a, (x.shape[1], a.shape[1], a.shape[1])).copy()


@gmac_constraint_metrics.register("l1sq")
class L1Squared:
    """``D(x) = ||x||_1^2``: gradient ``2 ||x||_1 sgn(x)``, Hessian ``2 sgn(x) sgn(x)^T``.

    MATLAB ``GIES_TVL12_constraint.m``'s TV term.
    """

    constant = False

    def cost(self, x: Array) -> Array:
        """Return ``||x_j||_1^2``."""
        out: Array = np.sum(np.abs(x), axis=0) ** 2
        return out

    def gradient(self, x: Array) -> Array:
        """Return ``2 ||x||_1 sgn(x)``."""
        out: Array = 2.0 * np.sum(np.abs(x), axis=0) * np.sign(x)
        return out

    def metric(self, a: Array, x: Array) -> Array:
        """Return the rank-one ``2 v_j v_j^T``, ``v_j = A^T sgn(x_j)``."""
        v = a.T @ np.sign(x)
        out: Array = 2.0 * np.einsum("ij,kj->jik", v, v)
        return out


@gmac_constraint_metrics.register("log_barrier")
class LogBarrier:
    """The barrier ``D(x) = -sum log(x + a)`` of inequalities, ``x = -h >= 0``.

    Eqs. ``GMAC_constrained_barrier_function``-``barrier_function_hessian``.

    Gradient ``-1/(x + a)``, Hessian ``diag(1/(x + a)^2)``. Undefined once a row is
    violated by ``a`` or more (``x + a <= 0``): the update is refused then.

    Parameters
    ----------
    a
        The offset (the paper: 0.1).
    """

    constant = False

    def __init__(self, a: float = 0.1) -> None:
        if not a > 0:
            raise ValueError(f"a must be positive, got {a}")
        self.a = float(a)

    def shifted(self, x: Array) -> Array:
        """Return ``x + a``; raise :class:`ConstraintViolation` where it is not positive."""
        shifted = x + self.a
        if np.any(shifted <= 0):
            raise ConstraintViolation(
                f"{int(np.sum(shifted <= 0))} rows violated by a = {self.a} or more"
            )
        return shifted

    def cost(self, x: Array) -> Array:
        """Return ``-sum log(x_j + a)`` (``inf`` where undefined)."""
        shifted = x + self.a
        with np.errstate(divide="ignore", invalid="ignore"):
            out: Array = -np.sum(np.log(np.where(shifted > 0, shifted, np.nan)), axis=0)
        return np.where(np.isnan(out), np.inf, out)

    def gradient(self, x: Array) -> Array:
        """Return ``-1/(x + a)``."""
        out: Array = -1.0 / self.shifted(x)
        return out

    def metric(self, a: Array, x: Array) -> Array:
        """Return ``A^T diag(1/(x_j + a)^2) A``."""
        return _diagonal_metric(a, 1.0 / self.shifted(x) ** 2)


@gmac_constraint_metrics.register("log_channel")
class LogChannel:
    """The channel ``D(x) = sum log(|x| + b)`` of equalities.

    Eqs. ``GMAC_constrained_channel_function``-``channel_function_hessian_epsilon``.

    It decreases toward 0, but is concave on each side: gradient ``sgn(x)/(|x| + b)``,
    Hessian ``-diag(1/(|x| + b)^2)`` (the scripts: ``1/(x + b sgn(x) + epsilon)`` and
    ``-diag(1/(x + b sgn(x) + epsilon)^2)``, used for ``paper``). ``curvature`` chooses
    the metric (the user's decision):

    - ``absolute_hessian`` (default; the user's choice after the facies study 1
      follow-up, 2026-10-06): ``|Hessian|`` with a positive weight;
    - ``absolute_trace``: the negative Hessian, a positive weight (``|tr|`` in the trace
      rule); the metric may become indefinite;
    - ``paper``: the negative Hessian and the paper's (negative) trace weight.

    Parameters
    ----------
    b, epsilon
        The offsets (the paper's scripts: 0.1, 0.001; ``epsilon`` is used by ``paper``).
    curvature
        See above.
    """

    constant = False

    def __init__(
        self,
        b: float = 0.1,
        epsilon: float = 0.001,
        curvature: Literal["absolute_hessian", "absolute_trace", "paper"] = "absolute_hessian",
    ) -> None:
        if not b > 0 or not epsilon > 0:
            raise ValueError(f"b and epsilon must be positive, got {b}, {epsilon}")
        if curvature not in ("absolute_hessian", "absolute_trace", "paper"):
            raise ValueError(
                f"curvature must be absolute_hessian, absolute_trace or paper, got {curvature!r}"
            )
        self.b, self.epsilon, self.curvature = float(b), float(epsilon), curvature

    def _denominator(self, x: Array) -> Array:
        """Return the scripts' ``x + b sgn(x) + epsilon`` (``paper``), else ``|x| + b``.

        With ``sgn(0) = 0`` the scripts' form gives a row already on target (``x = 0``,
        common for histogram counts) the gradient ``1/epsilon`` and the curvature
        ``1/epsilon^2``, which then take nearly all of the trace weight; the absolute
        curvatures use the cost's own derivatives, ``sgn(x)/(|x| + b)`` and
        ``1/(|x| + b)^2`` (0 and ``1/b^2`` at ``x = 0``).
        """
        if self.curvature == "paper":
            out: Array = x + self.b * np.sign(x) + self.epsilon
            return out
        return np.abs(x) + self.b

    def cost(self, x: Array) -> Array:
        """Return ``sum log(|x_j| + b)``."""
        out: Array = np.sum(np.log(np.abs(x) + self.b), axis=0)
        return out

    def gradient(self, x: Array) -> Array:
        """Return ``sgn(x)/(|x| + b)``, or the scripts' ``1/(x + b sgn(x) + epsilon)``."""
        if self.curvature == "paper":
            out: Array = 1.0 / self._denominator(x)
            return out
        signed: Array = np.sign(x) / self._denominator(x)
        return signed

    def metric(self, a: Array, x: Array) -> Array:
        """Return ``A^T H A``: ``H = 1/denominator^2``, negated unless ``absolute_hessian``."""
        hessian = 1.0 / self._denominator(x) ** 2
        sign = 1.0 if self.curvature == "absolute_hessian" else -1.0
        return _diagonal_metric(a, sign * hessian)


# ---------------------------------------------------------------------- terms


class ConstraintTerm:
    """One constraint of ``gmac``: a function, a metric and a weight.

    Parameters
    ----------
    function, metric
        The parts.
    rule, value
        ``trace``: ``w_j = value tr(M_D)/tr(M_c,j)`` (the paper's rule; ``|tr|`` unless
        the metric's curvature is ``paper``); ``fixed``: ``w = value``.
    name
        For messages and diagnostics.
    """

    def __init__(
        self,
        function: ConstraintFunction,
        metric: ConstraintMetric,
        rule: Literal["trace", "fixed"],
        value: float,
        name: str,
    ) -> None:
        if value < 0:
            raise ValueError(f"constraint {name}: its weight must be non-negative, got {value}")
        self.function, self.metric, self.rule = function, metric, rule
        self.value, self.name = float(value), name
        # The paper's rule keeps tr(M_c)'s sign: the negative weight of a concave metric.
        self.signed_trace = getattr(metric, "curvature", "") == "paper"

    def weights(self, data_trace: float, metrics: Array) -> Array:
        """Return each member's weight ``(N,)`` from its metric ``(N, N, N)``."""
        n = metrics.shape[0]
        if self.rule == "fixed":
            return np.full(n, self.value)
        traces = np.einsum("jii->j", metrics)
        scale = traces if self.signed_trace else np.abs(traces)
        out: Array = np.where(
            scale != 0, self.value * data_trace / np.where(scale != 0, scale, 1.0), 0.0
        )
        return out
