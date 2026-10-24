"""Plurigaussian truncation rules and their conditioning to facies observations.

The latent fields ``Y_1..Y_m`` are independent standard Gaussian random fields. A rule
partitions the latent space ``R^m`` into facies regions, and the facies at a cell is the
region its latent vector falls in (Galli et al., 1994; Armstrong et al., 2011).

Rules are **fixed in latent space**. GSTools' ``PGS`` lays its lithotype array over axes
spanning each realisation's own range and centroid, so the same latent value may map to
different facies in different members, or in one member after an update; an ensemble
smoother needs one map for all. Here an axis of ``n`` codes is cut into ``n`` equal bins of
``[-extent, extent]`` (standard deviations), the first and last also taking the tails. For
the same edges the indexing is GSTools' (``np.digitize``), which ``tests/unit`` checks
against GSTools with GSTools' own axes.

Reference  : Armstrong, M. et al. (2011). Plurigaussian Simulations in Geosciences (2nd
             ed.), Springer. Emery, X. (2007). Simulation of geological domains using the
             plurigaussian model. Computers & Geosciences, 33, 1189-1201. Pakman, A. &
             Paninski, L. (2014). Exact Hamiltonian Monte Carlo for truncated multivariate
             Gaussians. J. Comput. Graph. Stat., 23(2), 518-542.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats


@dataclass(frozen=True)
class LithotypeRule:
    """A lithotype array over fixed latent axes: ``codes[i_1, ..., i_m]`` is a facies code.

    Attributes
    ----------
    codes
        Integer facies codes, one axis per latent field, shape ``(n_1, ..., n_m)``.
    axes
        The bin edges of each latent axis, ``(n_d,)`` each, increasing: bin 0 is
        ``(-inf, axes[0])``, bin ``i`` is ``[axes[i-1], axes[i])`` and the last bin is
        ``[axes[-2], inf)``.
    """

    codes: NDArray[np.int64]
    axes: tuple[NDArray[np.float64], ...]

    def __post_init__(self) -> None:
        """Check that the codes and axes agree and the axes increase."""
        if self.codes.ndim != len(self.axes):
            raise ValueError(
                f"a rule of {self.codes.ndim} axes needs {self.codes.ndim} axis arrays"
            )
        for d, axis in enumerate(self.axes):
            if axis.shape != (self.codes.shape[d],) or axis.size < 2:
                raise ValueError(f"axis {d} must have {self.codes.shape[d]} >= 2 increasing edges")
            if np.any(np.diff(axis) <= 0):
                raise ValueError(f"axis {d} must increase")

    @classmethod
    def on_box(cls, codes: NDArray[np.integer], extent: float) -> "LithotypeRule":
        """Return the rule of ``codes`` laid over ``[-extent, extent]`` on every axis.

        Each axis of ``n`` codes is cut into ``n`` equal bins of ``[-extent, extent]``; the
        first and last also take the tails beyond (``-inf`` and ``+inf``).
        """
        array: NDArray[np.int64] = np.asarray(codes, dtype=np.int64)
        if extent <= 0:
            raise ValueError(f"extent must be positive, got {extent}")
        axes = []
        for n in array.shape:
            if n < 2:
                raise ValueError("every axis of a lithotype array needs at least 2 codes")
            interior = np.linspace(-extent, extent, n + 1)[1:-1]  # (n - 1,) bin boundaries
            # The class's layout: n edges, the last unused beyond the clip (see above).
            axes.append(np.append(interior, interior[-1] + 2 * extent / n))
        return cls(array, tuple(axes))

    @classmethod
    def from_proportions(cls, proportions: Sequence[float]) -> "LithotypeRule":
        """Return the truncated-Gaussian rule of one field for target facies proportions.

        Facies ``k`` is ``Y`` in ``[q_{k-1}, q_k)`` with ``q_k = Phi^-1(p_1 + ... + p_k)``,
        so facies ``k`` has prior probability ``p_k`` at every cell.
        """
        p = np.asarray(proportions, dtype=np.float64)
        if p.ndim != 1 or p.size < 2 or np.any(p <= 0) or not math.isclose(p.sum(), 1.0):
            raise ValueError(f"proportions must be >= 2 positive fractions adding up to 1, got {p}")
        thresholds = special.ndtri(np.cumsum(p)[:-1])  # (K - 1,)
        # Bins: (-inf, q_1), [q_1, q_2), ..., [q_{K-1}, inf): K bins need K edges, the
        # last one unused (it closes the second-to-last bin, see the class docstring).
        edges = np.append(thresholds, thresholds[-1] + 1.0)
        return cls(np.arange(p.size, dtype=np.int64), (edges,))

    @property
    def n_fields(self) -> int:
        """Return the number of latent fields."""
        return self.codes.ndim

    def bins(self, latent: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return the bin of each value on each axis, ``(m, n)``, from ``latent`` ``(m, n)``."""
        if latent.ndim != 2 or latent.shape[0] != self.n_fields:
            raise ValueError(f"latent must be ({self.n_fields}, n), got {latent.shape}")
        return np.stack(
            [
                np.clip(np.digitize(latent[d], self.axes[d]), 0, self.codes.shape[d] - 1)
                for d in range(self.n_fields)
            ]
        )

    def facies(self, latent: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return the facies code of each cell, ``(n,)``, from ``latent`` ``(m, n)``."""
        result: NDArray[np.int64] = self.codes[tuple(self.bins(latent))]
        return result

    def edges(self, d: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the lower and upper bounds of axis ``d``'s bins, ``(n_d,)`` each."""
        axis = self.axes[d]
        lower = np.concatenate([[-np.inf], axis[:-1]])
        upper = np.concatenate([axis[:-1], [np.inf]])
        return lower, upper

    def proportions(self) -> NDArray[np.float64]:
        """Return each facies code's probability for independent standard normal fields.

        The probability of a lithotype cell is the product over axes of its bin's normal
        mass; a facies' probability is the sum over its cells. Returns ``(max code + 1,)``.
        """
        result = np.zeros(int(self.codes.max()) + 1)
        np.add.at(result, self.codes.ravel(), self.cell_masses().ravel())
        return result

    def cell_masses(self) -> NDArray[np.float64]:
        """Return each lithotype cell's standard normal mass, the shape of ``codes``."""
        cell = np.ones(())
        for d in range(self.n_fields):
            lower, upper = self.edges(d)
            cell = np.multiply.outer(cell, special.ndtr(upper) - special.ndtr(lower))
        return np.asarray(cell, dtype=np.float64)

    def allowed(self, d: int, bins: NDArray[np.intp], code: int) -> list[tuple[float, float]]:
        """Return the intervals of axis ``d`` where the facies is ``code``, others fixed.

        ``bins`` ``(m,)`` holds the other axes' bins (its entry ``d`` is ignored).
        """
        index = list(bins)
        lower, upper = self.edges(d)
        intervals: list[tuple[float, float]] = []
        for b in range(self.codes.shape[d]):
            index[d] = b
            if self.codes[tuple(index)] == code:
                if intervals and intervals[-1][1] == lower[b]:
                    intervals[-1] = (intervals[-1][0], float(upper[b]))  # merge neighbours
                else:
                    intervals.append((float(lower[b]), float(upper[b])))
        return intervals


def _truncated_normal(
    mean: float, std: float, intervals: Sequence[tuple[float, float]], rng: np.random.Generator
) -> float:
    """Draw from ``N(mean, std^2)`` restricted to a union of disjoint intervals."""
    if not intervals:
        raise ValueError("the conditioning sampler found no feasible interval (report this)")
    a = np.array([(lo - mean) / std for lo, _ in intervals])
    b = np.array([(hi - mean) / std for _, hi in intervals])
    # Interval masses in log space: far tails underflow ndtr.
    log_mass = np.array([_log_mass(lo, hi) for lo, hi in zip(a, b, strict=True)])
    weights = np.exp(log_mass - log_mass.max())
    k = int(rng.choice(len(intervals), p=weights / weights.sum()))
    value = stats.truncnorm.rvs(a[k], b[k], random_state=rng)
    return float(mean + std * value)


def _log_mass(a: float, b: float) -> float:
    """Return ``log(Phi(b) - Phi(a))`` stably, for ``a < b``."""
    if a > 0:  # upper tail: use the symmetry Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
        a, b = -b, -a
    log_b, log_a = special.log_ndtr(b), special.log_ndtr(a)
    return float(log_b + np.log1p(-np.exp(log_a - log_b))) if log_a < log_b else -np.inf


#: Added to the conditioning covariances' diagonal (unit variances): a nugget of 1e-6 of
#: the variance keeps closely spaced cells of smooth models (Gaussian) factorable.
JITTER = 1e-6
#: Covariances still worse conditioned than this, after the jitter, are refused. With the
#: jitter the condition number is at most about ``n_c / JITTER``, so this guards against
#: non-finite or wildly scaled covariances, not against close cells (the jitter handles
#: those; duplicate cells are merged by the prior).
MAX_CONDITION = 1e10
#: Unconditional draws tried for a smooth feasible start (see ``condition_latent``).
START_TRIES = 2000


def condition_latent(
    rule: LithotypeRule,
    covariances: Sequence[NDArray[np.float64]],
    facies: NDArray[np.integer],
    rng: np.random.Generator,
    sweeps: int = 100,
) -> NDArray[np.float64]:
    """Draw latent values at the conditioning cells, given their facies, by Gibbs sampling.

    Each latent field ``Y_j`` is ``N(0, C_j)`` over the ``n_c`` conditioning cells
    (``covariances[j]`` ``(n_c, n_c)``, unit variance), the fields independent. The target is
    their joint distribution restricted to ``rule.facies(y[:, c]) == facies[c]`` at every
    cell (Armstrong et al., 2011; Emery, 2007).

    Each latent field is sampled in **whitened coordinates**, ``y_j = L_j x_j`` with
    ``L_j L_j^T = C_j + JITTER I`` (Cholesky), so that ``x_j`` is standard normal. One
    iteration, per field, given the other fields:

    1. **Exact Hamiltonian Monte Carlo** (Pakman & Paninski, 2014) of ``x_j`` within the box
       of the intervals each cell is in now (``lower_c <= (L x)_c <= upper_c``): the
       trajectories ``x(t) = v sin t + x cos t`` are analytic and reflect off the walls, so
       strongly correlated cells (a vertical well under a smooth model) mix in one step,
       where single-site Gibbs does not.
    2. A **coordinate sweep**: each ``x_jk`` given the others is ``N(0, 1)`` truncated to
       where every cell it moves stays in its facies' intervals on axis ``j`` (an
       intersection of unions of intervals), so a cell can move between disjoint intervals.

    Both moves leave the target invariant (the first given which interval each cell is in).
    The start is the first feasible draw among ``START_TRIES`` unconditional ones, else
    each cell drawn in a facies cell independently (feasible, but rough).

    Parameters
    ----------
    facies
        The observed codes, ``(n_c,)``.
    sweeps
        Iterations (an HMC step and a coordinate sweep per field); the last state is
        returned.

    Returns
    -------
    NDArray
        ``(m, n_c)`` latent values consistent with the observed facies.

    Raises
    ------
    ValueError
        If a code is not in the rule or its lithotype cells are not linked along the axes,
        the shapes disagree, or a covariance is not usable even with the jitter.

    Notes
    -----
    The values are drawn under ``C + JITTER I``; GSTools' ``CondSRF`` then kriges them with
    the nugget-free model, a difference of 1e-6 of the variance. The coordinate sweep costs
    ``O(n_c^2)`` interval operations per field and sweep in Python: seconds per member for
    tens of conditioning cells, minutes for hundreds (vectorise before large 3-D cases).
    """
    facies = np.asarray(facies, dtype=np.int64)
    m, n_c = rule.n_fields, facies.size
    if len(covariances) != m or any(c.shape != (n_c, n_c) for c in covariances):
        raise ValueError(f"need {m} covariances of shape ({n_c}, {n_c})")
    missing = sorted(set(facies.tolist()) - set(rule.codes.ravel().tolist()))
    if missing:
        raise ValueError(f"facies codes {missing} are not in the rule")
    for code in np.unique(facies):
        if _components(rule.codes == code) > 1:
            # Both moves change one latent field at a time: a facies whose lithotype cells
            # are not linked along the axes has parts the chain could never reach.
            raise ValueError(
                f"facies {int(code)}'s cells of the lithotype array are not connected along "
                "the axes, so conditioning to it cannot reach every part; split the facies "
                "into two codes"
            )
    factors = []
    for covariance in covariances:
        jittered = covariance + JITTER * np.eye(n_c)
        condition = np.linalg.cond(jittered)
        if not np.isfinite(condition) or condition > MAX_CONDITION:
            raise ValueError(
                f"the conditioning cells' covariance has condition number {condition:.3g} "
                f"(above {MAX_CONDITION:.0e} even with a {JITTER:g} nugget): are two "
                "conditioning cells at the same position?"
            )
        factors.append(np.linalg.cholesky(jittered))
    y = _start(rule, factors, facies, rng)
    z = np.stack([np.linalg.solve(factors[d], y[d]) for d in range(m)])
    for _ in range(sweeps):
        for d in range(m):
            factor = factors[d]
            # The other fields stay fixed during field d's sweep, so each cell's allowed
            # intervals on axis d are too.
            bins = rule.bins(y)
            allowed = [rule.allowed(d, bins[:, c], int(facies[c])) for c in range(n_c)]
            lower, upper = np.empty(n_c), np.empty(n_c)
            for c in range(n_c):  # the interval each cell is in now (nearest, for rounding)
                distance = [max(lo - y[d, c], y[d, c] - hi, 0.0) for lo, hi in allowed[c]]
                lower[c], upper[c] = allowed[c][int(np.argmin(distance))]
            z[d] = _hmc_box(factor, z[d], lower, upper, rng)
            y[d] = factor @ z[d]
            for k in range(n_c):
                column = factor[:, k]
                rest = y[d] - column * z[d, k]  # y without z_k's part
                feasible: list[tuple[float, float]] = [(-np.inf, np.inf)]
                for cell in np.flatnonzero(np.abs(column) > 1e-14):
                    scale, offset = float(column[cell]), float(rest[cell])
                    ends = [
                        ((lo - offset) / scale, (hi - offset) / scale) for lo, hi in allowed[cell]
                    ]
                    mapped = sorted((min(e), max(e)) for e in ends)  # a negative scale flips
                    feasible = _intersect(feasible, mapped)
                value = _truncated_normal(0.0, 1.0, feasible, rng)
                y[d] = rest + column * value
                z[d, k] = value
    if not np.array_equal(
        rule.facies(y), facies
    ):  # numerical guard; cannot happen in exact arithmetic
        raise ValueError("the conditioning sampler left a cell outside its facies (report this)")
    return y


def _start(
    rule: LithotypeRule,
    factors: Sequence[NDArray[np.float64]],
    facies: NDArray[np.integer],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return a feasible start ``(m, n_c)``: a feasible unconditional draw if one is found."""
    m, n_c = len(factors), facies.size
    draws = np.stack([factor @ rng.standard_normal((n_c, START_TRIES)) for factor in factors])
    feasible = np.all(
        rule.facies(draws.reshape(m, -1)).reshape(n_c, START_TRIES) == facies[:, None], axis=0
    )
    if feasible.any():
        return np.ascontiguousarray(draws[:, :, int(np.argmax(feasible))])
    y = np.empty((m, n_c))
    masses = rule.cell_masses()
    for c in range(n_c):
        weights = np.where(rule.codes == facies[c], masses, 0.0).ravel()
        cell = np.unravel_index(
            int(rng.choice(weights.size, p=weights / weights.sum())), rule.codes.shape
        )
        for d in range(m):
            lower, upper = rule.edges(d)
            y[d, c] = _truncated_normal(0.0, 1.0, [(lower[cell[d]], upper[cell[d]])], rng)
    return y


def _hmc_box(
    factor: NDArray[np.float64],
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    rng: np.random.Generator,
    duration: float = math.pi / 2,
    max_bounces: int = 10_000,
) -> NDArray[np.float64]:
    """One exact HMC step of ``x ~ N(0, I)`` restricted to ``lower <= factor @ x <= upper``.

    Pakman & Paninski (2014), "Exact Hamiltonian Monte Carlo for truncated multivariate
    Gaussians", J. Comput. Graph. Stat. 23(2), 518-542. Constraints are ``f_i . x + g_i >= 0``;
    along ``x(t) = v sin t + x cos t`` each ``f . x(t) = u cos(t - phi)``, so the first wall a
    trajectory meets is found in closed form, and the velocity is reflected off it.
    """
    rows, offsets = [], []
    for c in range(factor.shape[0]):
        if np.isfinite(lower[c]):
            rows.append(factor[c])
            offsets.append(-lower[c])
        if np.isfinite(upper[c]):
            rows.append(-factor[c])
            offsets.append(upper[c])
    if not rows:
        return rng.standard_normal(x.size)
    f, g = np.array(rows), np.array(offsets)
    norms = np.einsum("ij,ij->i", f, f)
    v = rng.standard_normal(x.size)
    remaining, last = duration, -1
    for _ in range(max_bounces):
        a, b = f @ v, f @ x  # f . x(t) = a sin t + b cos t
        u = np.hypot(a, b)
        phi = np.arctan2(a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            reach = np.where(u > np.abs(g), np.arccos(np.clip(-g / u, -1.0, 1.0)), np.nan)
        times = np.full(g.size, np.inf)
        for sign in (1.0, -1.0):  # t = phi +- arccos(-g/u), the next one after 0
            t = np.mod(phi + sign * reach, 2 * np.pi)
            t = np.where(np.isnan(t) | (t < 1e-10), np.inf, t)
            if last >= 0 and t[last] < 1e-8:
                t[last] = np.inf  # the wall just bounced off: drop only this root at t ~ 0
            times = np.minimum(times, t)
        hit = int(np.argmin(times))
        if times[hit] >= remaining:
            return x * math.cos(remaining) + v * math.sin(remaining)
        t = times[hit]
        x, v = x * math.cos(t) + v * math.sin(t), v * math.cos(t) - x * math.sin(t)
        v = v - 2.0 * (f[hit] @ v) / norms[hit] * f[hit]  # reflect off the wall
        remaining -= t
        last = hit
    raise ValueError("the conditioning sampler's HMC step bounced too often (report this)")


def _intersect(
    a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Return the intersection of two sorted unions of disjoint intervals."""
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            result.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _components(mask: NDArray[np.bool_]) -> int:
    """Count the groups of ``True`` cells linked by sharing all indices but one."""
    cells = [tuple(int(i) for i in idx) for idx in np.argwhere(mask)]
    parent = {cell: cell for cell in cells}

    def root(cell: tuple[int, ...]) -> tuple[int, ...]:
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    for d in range(mask.ndim):
        lines: dict[tuple[int, ...], tuple[int, ...]] = {}
        for cell in cells:
            key = cell[:d] + cell[d + 1 :]
            if key in lines:
                parent[root(cell)] = root(lines[key])
            else:
                lines[key] = cell
    return len({root(cell) for cell in cells})
