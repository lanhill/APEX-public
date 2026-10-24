"""GMAC's metric terms: model transforms S and norms.

A term of GMAC's step metric (Luo, *GIES*, eq. ``GMAC_M_R``) is
``M_k(j) = S_Sᵀ ∇²R_k[S(m̄) - S(m_j)] S_S``, an ``N x N`` matrix per member ``j``:

- a **transform** gives the ensemble's anomalies in its feature space, ``A = S_S``
  ``(n_f, N)``, scaled by ``1/sqrt(N-1)`` as the state anomalies are, and the points where the
  Hessian is taken, ``T[:, j] = S(m̄) - S(m_j)`` ``(n_f, N)``;
- a **norm** gives ``Aᵀ H(T[:, j]) A`` from its Hessian ``H``, for every member, without
  forming an ``n_f x n_f`` matrix (an l₁² Hessian is rank one).

Transforms see the state variables by name, with their sizes and grids (the composition
supplies them). Registered in :data:`~apex.algorithms.data_assimilation.registries.gmac_transforms`
and ``gmac_norms``; a new kind of regularisation is a new plugin there.

Sources: the user's manuscript (GIES_R1.tex: App. C, the l_p^q Hessians; eqs.
``total_PLH``, ``weights_rule``) and MATLAB ``MATLAB .../run_scripts/getTV2D.m`` and
``getHistEntropy.m`` (the TV and histogram-entropy transforms). Differences from the MATLAB,
recorded in ``docs/migration/gmac.md``: histogram bins are fixed across the ensemble and the
mean model (MATLAB's FaciesStudy1 took each sample's own range).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.registries import gmac_norms, gmac_transforms
from apex.core.options import TEXT, TextList

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Block:
    """One state variable as GMAC's transforms see it: its slice and grid."""

    name: str
    start: int
    size: int
    grid: tuple[int, ...] | None


def blocks(state_variables: Sequence[Mapping[str, object]]) -> dict[str, Block]:
    """Return the state's blocks by name, from the composition's ``state_variables``."""
    found: dict[str, Block] = {}
    start = 0
    for variable in state_variables:
        name, size = str(variable["name"]), int(variable["size"])  # type: ignore[call-overload]
        grid = variable.get("grid")
        found[name] = Block(name, start, size,
                            None if grid is None else tuple(int(n) for n in grid))  # type: ignore[attr-defined]  # fmt: skip
        start += size
    return found


class PropertyMapping(Protocol):
    """The parameterizations' map from the state to the properties, stacked.

    Supplied by the composition when ``[[parameterizations]]`` are configured, so that the
    terms see the properties the simulator reads.
    """

    #: The properties' names and the grids of those on one.
    names: tuple[str, ...]
    grids: Mapping[str, tuple[int, ...]]

    def __call__(self, state: Array) -> tuple[Array, list[dict[str, object]]]:
        """Return the properties ``(n_p, k)`` of ``state`` ``(n_state, k)`` and their layout.

        The layout is ``[{"name", "size", "grid"}]`` in the stacked order.
        """
        ...


class GMACTransform(Protocol):
    """A model transform S of a metric term (see the module docstring)."""

    #: ``S`` is linear in its own input (so ``S(mean x) = mean S(x)``; with a nonlinear
    #: parameterization ``S o P`` is not). Informative; the strategy does not rely on it.
    linear: bool

    def anomalies(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> tuple[Array, Array]:
        """Return ``(A, T)``: feature anomalies ``(n_f, N)`` and Hessian points ``(n_f, N)``.

        ``A = (S(x_j) - S(c))/sqrt(N-1)`` and ``T = S(c) - S(x_j)`` about ``centre`` ``c``
        ``(n,)``, the variables at the mean state (the properties ``P(m_mean)`` when a
        parameterization maps the state); default the mean of ``state``.
        """
        ...


class GMACNorm(Protocol):
    """A norm of a metric term: the action of its Hessian (see the module docstring)."""

    #: The Hessian is constant (l₂²): the metric is the same for every member.
    constant: bool

    def metric(self, a: Array, t: Array) -> Array:
        """Return ``Aᵀ H(T[:, j]) A``: ``(N, N)`` if constant, else ``(N, N, N)`` by member."""
        ...


# ---------------------------------------------------------------------- transforms


def _select(layout: Mapping[str, Block], variables: Sequence[str] | None) -> list[Block]:
    chosen = list(layout.values()) if not variables else [layout[v] for v in variables
                                                          if v in layout]  # fmt: skip
    unknown = sorted(set(variables or ()) - set(layout))
    if unknown:
        raise ValueError(f"no state variables {unknown}; the state has {sorted(layout)}")
    return chosen


def _member_anomalies(features: Array, mean_features: Array) -> tuple[Array, Array]:
    """Return ``A = (F - f̄)/sqrt(N-1)`` and ``T = f̄ - F`` for features ``F (n_f, N)``."""
    n = features.shape[-1]
    centred = features - mean_features[:, None]
    return centred / math.sqrt(n - 1), -centred


@gmac_transforms.register("identity")
class IdentityTransform:
    """The paper's I term: the ensemble-whitened state, so that ``S_Sᵀ S_S = I_N``.

    With ``l2sq`` (the trace weight) its metric is ``I``: RLM-MAC's (eq.
    ``update_formula_IES_alternative``). Its Hessian points are not used by a quadratic norm;
    a non-quadratic norm on the raw state is the ``state`` transform's.
    """

    linear = True

    def anomalies(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> tuple[Array, Array]:
        """Return ``(I_N, 0)``."""
        n = state.shape[-1]
        return np.eye(n), np.zeros((n, n))


@gmac_transforms.register("state")
class StateTransform:
    """The raw state (or some of its variables): MATLAB ``RLM_MAC_L1``'s l₁² on ``m``.

    Parameters
    ----------
    variables
        The state variables; default all.
    """

    linear = True

    def __init__(self, variables: TextList | None = None) -> None:
        self.variables = tuple(variables or ())

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` if a named variable is not in the state."""
        _select(layout, self.variables)

    def anomalies(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> tuple[Array, Array]:
        """Return the selected variables' anomalies about ``centre`` (default the mean)."""
        rows = np.concatenate([np.arange(b.start, b.start + b.size)
                               for b in _select(layout, self.variables)])  # fmt: skip
        x = state[rows]
        return _member_anomalies(x, x.mean(axis=-1) if centre is None else centre[rows])


@gmac_transforms.register("tv")
class TVTransform:
    """First differences along each grid axis, per variable, stacked (``S_V``, ``getTV2D``).

    A variable's cells are in Eclipse order (x fastest, ``order = "F"``), as APEX's grids.
    Differences are taken along every axis longer than one cell.

    Parameters
    ----------
    variables
        The gridded state variables; default every one with a grid.
    scale_per_variable
        Divide each variable's differences by their ensemble RMS, so that no variable
        dominates by its units (the 5Spot MATLAB stacked PORO and PERMX unscaled: off).
    """

    linear = True

    def __init__(self, variables: TextList | None = None,
                 scale_per_variable: bool = False) -> None:  # fmt: skip
        self.variables = tuple(variables or ())
        self.scale_per_variable = scale_per_variable

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` for unknown or ungridded named variables, or none gridded."""
        chosen = _select(layout, self.variables)
        ungridded = [b.name for b in chosen if b.grid is None and self.variables]
        if ungridded:
            raise ValueError(f"tv needs gridded variables; {ungridded} have no grid")
        if not any(b.grid is not None for b in chosen):
            raise ValueError("tv needs gridded state variables ([[state.variables]] grid)")

    def features(self, state: Array, layout: Mapping[str, Block]) -> Array:
        """Return the stacked differences ``(n_f, k)`` of ``state`` ``(n_state, k)``."""
        chosen = [b for b in _select(layout, self.variables) if b.grid is not None]
        if not chosen:
            raise ValueError("tv needs gridded state variables ([[state.variables]] grid)")
        parts = []
        for block in chosen:
            assert block.grid is not None
            field = state[block.start:block.start + block.size].reshape(
                (*block.grid, state.shape[-1]), order="F")  # fmt: skip
            for axis, n in enumerate(block.grid):
                if n > 1:
                    d = np.diff(field, axis=axis)
                    parts.append(d.reshape(-1, state.shape[-1], order="F"))
        return np.concatenate(parts)

    def anomalies(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> tuple[Array, Array]:
        """Return the differences' anomalies about ``S(centre)`` (default ``mean S(x)``)."""
        f = self.features(state, layout)
        mean = f.mean(axis=-1) if centre is None else self.features(centre[:, None], layout)[:, 0]
        a, t = _member_anomalies(f, mean)
        if self.scale_per_variable:
            a, t = self._scaled(a, t, state, layout)
        return a, t

    def _scaled(self, a: Array, t: Array, state: Array,
                layout: Mapping[str, Block]) -> tuple[Array, Array]:  # fmt: skip
        start = 0
        for block in (b for b in _select(layout, self.variables) if b.grid is not None):
            assert block.grid is not None
            rows = sum(math.prod(block.grid) // n * (n - 1) for n in block.grid if n > 1)
            rms = float(np.sqrt(np.mean(a[start : start + rows] ** 2))) or 1.0
            a[start : start + rows] /= rms
            t[start : start + rows] /= rms
            start += rows
        return a, t


@gmac_transforms.register("histogram_entropy")
class HistogramEntropyTransform:
    """A histogram's count-weighted entropy per bin, per variable (``S_H``, ``getHistEntropy``).

    Bin b's feature is ``c_b (ln n - ln c_b)`` for its count ``c_b > 0`` of a variable's ``n``
    cells (0 for an empty bin). The transform is nonlinear and piecewise constant: the term
    works through the ensemble's secants, ``A = (S(m_j) - S(m̄))/sqrt(N-1)`` with ``S(m̄)`` the
    features of the mean state.

    Parameters
    ----------
    variables
        The state variables; default all.
    bins
        Bins per variable (MATLAB: 100).
    ranges
        Fixed ``[low, high]`` per variable (5Spot MATLAB: PORO [0.1, 0.4], PERMX [6, 8]);
        default the ensemble's and the mean's range at each update. The bins are the same
        for every member and the mean, so the differences compare like with like. Values
        outside a fixed range fall in its end bins.
    """

    linear = False

    def __init__(
        self,
        variables: TextList | None = None,
        bins: int = 100,
        ranges: Annotated[Mapping[str, Sequence[float]], TEXT] | None = None,
    ) -> None:
        if bins < 2:
            raise ValueError(f"bins must be at least 2, got {bins}")
        self.variables = tuple(variables or ())
        self.bins = bins
        self.ranges = {k: (float(v[0]), float(v[1])) for k, v in (ranges or {}).items()}
        for name, (low, high) in self.ranges.items():
            if not low < high:
                raise ValueError(f"ranges[{name!r}] must be [low, high] with low < high")

    def check(self, layout: Mapping[str, Block]) -> None:
        """Raise ``ValueError`` for unknown variables or ``ranges`` of unselected ones."""
        chosen = {b.name for b in _select(layout, self.variables)}
        stray = sorted(set(self.ranges) - chosen)
        if stray:
            raise ValueError(f"ranges name {stray}, which the term does not use ({sorted(chosen)})")

    def _entropy(self, values: Array, low: float, high: float) -> Array:
        """Return the per-bin features ``(bins, k)`` of each column of ``values``."""
        width = (high - low) / self.bins
        index = np.clip(((values - low) / width).astype(np.int64), 0, self.bins - 1)
        n = values.shape[0]
        counts = np.zeros((self.bins, values.shape[1]))
        for column in range(values.shape[1]):
            counts[:, column] = np.bincount(index[:, column], minlength=self.bins)
        out = np.zeros_like(counts)
        nonzero = counts > 0
        out[nonzero] = counts[nonzero] * (math.log(n) - np.log(counts[nonzero]))
        return out

    def anomalies(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> tuple[Array, Array]:
        """Return the secant anomalies about the features of ``centre`` (default the mean)."""
        mean = state.mean(axis=-1, keepdims=True) if centre is None else centre[:, None]
        members, centres = [], []
        for block in _select(layout, self.variables):
            x = state[block.start : block.start + block.size]
            m = mean[block.start : block.start + block.size]
            low, high = self.ranges.get(block.name, (float(min(x.min(), m.min())),
                                                     float(max(x.max(), m.max()))))  # fmt: skip
            if not high > low:
                high = low + 1.0  # a constant variable: one occupied bin, no information
            members.append(self._entropy(x, low, high))
            centres.append(self._entropy(m, low, high)[:, 0])
        return _member_anomalies(np.concatenate(members), np.concatenate(centres))


# ---------------------------------------------------------------------- norms


@gmac_norms.register("l2sq")
class L2Squared:
    """``‖x‖₂²``: Hessian ``2I``, so the metric ``2AᵀA`` is the same for every member."""

    constant = True

    def metric(self, a: Array, t: Array) -> Array:
        """Return ``2AᵀA`` ``(N, N)``."""
        out: Array = 2.0 * (a.T @ a)
        return out


@gmac_norms.register("l1sq")
class L1Squared:
    """``‖x‖₁²``: Hessian ``2 sgn(x) sgn(x)ᵀ``, rank one, so ``M_j = 2 v_j v_jᵀ``.

    ``v_j = Aᵀ sgn(t_j)``.

    ``sgn(0) = 0``, as MATLAB's ``sign``. A term of this norm alone gives each member a one-
    dimensional step (its metric's pseudo-inverse has rank one).
    """

    constant = False

    def metric(self, a: Array, t: Array) -> Array:
        """Return ``(N, N, N)``: one rank-one metric per member."""
        v = a.T @ np.sign(t)  # (N, N): column j is v_j
        out: Array = 2.0 * np.einsum("ij,kj->jik", v, v)
        return out


@gmac_norms.register("lpq")
class LpQ:
    """``‖x‖_p^q`` with its exact Hessian (the manuscript's App. C).

    ``H(x) = q(q-p) ‖x‖_p^{q-2p} g gᵀ + q(p-1) ‖x‖_p^{q-p} diag(|x|^{p-2})``,
    ``g = |x|^{p-1} sgn(x)``. ``p = q = 2`` is ``l2sq``; ``p = 1, q = 2`` is ``l1sq``.

    Parameters
    ----------
    p, q
        At least 1. ``p = q = 1`` (``‖x‖₁``) is refused: its Hessian is 0 almost
        everywhere, so it adds nothing.
    floor
        For ``1 < p < 2``, ``|x|^{p-2}`` is taken at ``max(|x|, floor · max|x|)`` (it is
        infinite at 0).
    """

    def __init__(self, p: float = 2.0, q: float = 2.0, floor: float = 1e-6) -> None:
        if p < 1 or q < 1:
            raise ValueError(f"p and q must be at least 1, got p = {p}, q = {q}")
        if p == 1 and q == 1:
            raise ValueError("l1 (p = q = 1) has a zero Hessian almost everywhere: no metric")
        self.p, self.q, self.floor = float(p), float(q), float(floor)
        self.constant = self.p == 2 and self.q == 2

    def metric(self, a: Array, t: Array) -> Array:
        """Return ``(N, N)`` for ``p = q = 2``, else ``(N, N, N)`` by member."""
        p, q = self.p, self.q
        if self.constant:
            out: Array = 2.0 * (a.T @ a)
            return out
        n = t.shape[-1]
        result = np.zeros((n, a.shape[1], a.shape[1]))
        for j in range(n):
            x = t[:, j]
            norm = float(np.sum(np.abs(x) ** p) ** (1 / p))
            if norm == 0.0:
                continue  # at the mean model: no curvature to add
            g = np.abs(x) ** (p - 1) * np.sign(x)
            ag = a.T @ g
            result[j] = q * (q - p) * norm ** (q - 2 * p) * np.outer(ag, ag)
            if p != 1:
                ax = np.maximum(np.abs(x), self.floor * float(np.max(np.abs(x))))
                diag = ax ** (p - 2)
                result[j] += q * (p - 1) * norm ** (q - p) * ((a.T * diag) @ a)
        return result
