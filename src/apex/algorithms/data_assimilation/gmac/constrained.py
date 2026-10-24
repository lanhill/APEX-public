"""C-GIES's update: GMAC with constraint terms on the data side.

The user's method (Luo & Cruz (2022), Computational Geosciences 26(3), 571-594,
doi:10.1007/s10596-022-10137-7; equation labels of the manuscript DASC_CG_R1.tex: eqs.
``GMAC_constrained_update_formula``,
``matrix_factor``-``GMAC_constrained_update_formula_corrLoc``; MATLAB
``GIES_constraint_ensLoc.m``). For each member ``j``, with whitened data anomalies ``S``
about ``g(x_mean)`` and constraint anomalies ``S_c`` about ``c(x_mean)`` (both scaled by
``1/sqrt(N-1)``):

- ``M_eff,j = S^T S + sum_c w_c,j S_c^T H_c(x_c,j) S_c`` ``(N, N)``,
- source innovations ``r_d,j = S^T L^-1 (d_j - y_j)`` and
  ``r_c,j = w_c,j S_c^T grad D_c(x_c,j)``, ``x_c,j = 0 - c(m_j)``,
- ``coeff_j``: ``(M_eff,j + gamma_j M_R,j)^+`` with ``gamma_j = lambda tr(M_eff,j) /
  tr(M_R,j)`` (``direct``, the scripts' algorithm), or ``F_j V diag(1/(e + gamma_j)) V^T
  F_j^T`` with ``F_j`` from ``M_R,j``, ``F_j^T M_eff,j F_j = V diag(e) V^T`` truncated by
  ``energy`` and ``gamma_j = lambda mean(e)`` (``subspace``, as GMAC's),
- the projected innovations ``A_s,j = coeff_j r_s,j`` per source ``s``, and the step
  ``dX sum_s A_s,j``, or, localized, ``sum_s (T_s o dX) A_s,j`` with one taper per
  source (eq. ``GMAC_constrained_update_formula_corrLoc``).
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.gmac.constraints import (
    ConstraintTerm,
    ConstraintViolation,
    plugin_table,
)
from apex.algorithms.data_assimilation.gmac.parts import Block
from apex.algorithms.data_assimilation.registries import gmac_constraint_metrics, gmac_constraints
from apex.core.errors import PluginError

Array = NDArray[np.float64]

#: Directions with |e + gamma| below this fraction of max|e| are not inverted.
NEAR_SINGULAR = 1e-10

#: A constraint entry's default metric by its function's kind (the user's decision: the
#: log channel, with the absolute Hessian, for equalities; the barrier for inequalities).
DEFAULT_METRICS = {"equality": "log_channel", "inequality": "log_barrier"}


def build_constraints(entries: Sequence[Mapping[str, Any]]) -> list[ConstraintTerm]:
    """Return the configured constraint terms.

    Each entry is ``{function = {name, options}, metric = {name, options}, weight =
    {rule = "trace", w} | {rule = "fixed", value}}``; the metric defaults by the
    function's kind.

    Raises
    ------
    ValueError
        For unknown keys, parts or rules.
    """
    built = []
    for k, entry in enumerate(entries):
        try:
            built.append(_constraint(k, entry))
        except (KeyError, OSError, PluginError, TypeError) as error:
            raise ValueError(f"constraint {k}: {error}") from error
    return built


def _constraint(k: int, entry: object) -> ConstraintTerm:
    if not isinstance(entry, Mapping):
        raise ValueError(f"constraint {k} must be a table, got {entry!r}")
    unknown = sorted(set(entry) - {"function", "metric", "weight"})
    if unknown:
        raise ValueError(f"constraint {k}: unknown keys {unknown}")
    if "function" not in entry:
        raise ValueError(f"constraint {k}: no function")
    f_name, f_options = plugin_table(entry["function"], f"constraint {k}: function")
    function = gmac_constraints.create(f_name, **f_options)
    metric = entry.get("metric", {"name": DEFAULT_METRICS[function.kind]})
    m_name, m_options = plugin_table(metric, f"constraint {k}: metric")
    weight = entry.get("weight") or {}
    if not isinstance(weight, Mapping):
        raise ValueError(f"constraint {k}: weight must be a table {{ rule, w | value }}")
    rule = weight.get("rule", "trace")
    if rule not in ("trace", "fixed"):
        raise ValueError(f"constraint {k}: weight rule must be trace or fixed, got {rule!r}")
    value = float(weight.get("w" if rule == "trace" else "value", 1.0))
    stray = sorted(set(weight) - {"rule", "w" if rule == "trace" else "value"})
    if stray:
        raise ValueError(f"constraint {k}: unknown weight keys {stray}")
    return ConstraintTerm(
        function, gmac_constraint_metrics.create(m_name, **m_options), rule, value,
        f"{k}:{f_name}/{m_name}",
    )  # fmt: skip


class Sources:
    """The per-source pieces of one constrained update (data, then each constraint)."""

    def __init__(self, n: int) -> None:
        self.innovations: list[Array] = []  # (N, N) per source: column j is r_s,j
        self.costs: list[Array] = []
        self.weights: list[Array] = []
        self.violations: list[Array] = []
        self.n = n


def constraint_sources(
    constraints: Sequence[ConstraintTerm],
    seen: Array,
    centre: Array,
    layout: Mapping[str, Block],
    data_trace: float,
    metric: Array,
    sources: Sources,
) -> None:
    """Add each constraint's weighted metric to ``metric`` ``(N, N, N)`` and its source.

    ``seen`` ``(n, N)`` holds the variables the constraints see (state or properties),
    ``centre`` ``(n,)`` those of the mean state.

    Raises
    ------
    ConstraintViolation
        If a member violates an inequality where its barrier is undefined, with the
        cause and the fixes.
    """
    n = seen.shape[-1]
    for term in constraints:
        values = term.function.values(seen, layout)  # (n_c, N)
        at_mean = term.function.values(centre[:, None], layout)[:, 0]
        anomalies = (values - at_mean[:, None]) / math.sqrt(n - 1)  # S_c, (n_c, N)
        residuals = -values  # x = 0 - c
        try:
            gradient = term.metric.gradient(residuals)
            member_metrics = term.metric.metric(anomalies, residuals)  # (N, N, N)
        except ConstraintViolation as error:
            raise ConstraintViolation(_violation_message(term, values)) from error
        weights = term.weights(data_trace, member_metrics)
        metric += weights[:, None, None] * member_metrics
        sources.innovations.append(weights * (anomalies.T @ gradient))
        sources.costs.append(term.metric.cost(residuals))
        sources.weights.append(weights)
        violated = values > 0 if term.function.kind == "inequality" else np.zeros_like(values)
        sources.violations.append(np.sum(violated, axis=0).astype(np.float64))


def _violation_message(term: ConstraintTerm, values: Array) -> str:
    a = float(getattr(term.metric, "a", 0.0))
    bad = values >= a
    members = np.flatnonzero(bad.any(axis=0))
    return (
        f"constraint {term.name}: {int(bad.sum())} rows of members "
        f"{', '.join(str(j) for j in members[:10])}{' ...' if members.size > 10 else ''} "
        f"violate h <= 0 by {a} or more (worst: {float(values.max()):.4g}), where the log "
        f"barrier -log(-h + a) is undefined and would push them further out. Causes: the "
        f"state's bounds the loop clips members to ([[state.variables]] lower/upper) are "
        f"wider than the box, or absent, or the prior (or an earlier update) already "
        f"violates it. Fixes: set lower/upper within the box (the loop then clips members "
        f"into it, as the scripts do), raise the barrier's a above {float(values.max()):.4g}, "
        f"or correct the prior."
    )


def coefficient(
    form: Literal["subspace", "direct"],
    effective: Array,
    regularization: Array,
    damping: float,
    energy: float,
    factor: Array | None,
    indefinite: bool = False,
) -> tuple[Array, float, int]:
    """Return ``coeff_j`` ``(N, N)``, ``gamma_j`` and the rank kept (see the module).

    ``factor`` is ``F_j`` of ``M_R,j`` (subspace form; ``None`` for the direct form).

    Raises
    ------
    ValueError
        If the effective metric is zero.
    """
    n = effective.shape[0]
    if form == "direct":
        trace_r = float(np.trace(regularization))
        # tr(M_eff), or the sum of |eigenvalues| of an indefinite one (an absolute-trace
        # channel): gamma must stay a damping, never zero or negative.
        size = float(np.trace(effective))
        if indefinite:
            size = float(np.sum(np.abs(np.linalg.eigvalsh(0.5 * (effective + effective.T)))))
        gamma = damping * size / trace_r if trace_r > 0 else 0.0
        return np.linalg.pinv(effective + gamma * regularization), gamma, n
    assert factor is not None
    inner = factor.T @ effective @ factor
    inner = 0.5 * (inner + inner.T)
    values, vectors = np.linalg.eigh(inner)
    order = np.argsort(-np.abs(values))  # an absolute-trace channel may be indefinite
    values, vectors = values[order], vectors[:, order]
    if values.size == 0 or values[0] == 0.0:
        raise ValueError("the effective metric (data and constraints) is zero")
    # As rlmmac's TSVD of S F: s = sqrt|e|, the rank reaching `energy` of their sum, and
    # numerically null directions dropped.
    s = np.sqrt(np.abs(values))
    rank = s.size
    if energy < 1.0:  # truncated_svd's rule: the first rank whose share reaches `energy`
        share = np.cumsum(s) / float(np.sum(s))
        rank = int(np.searchsorted(share, energy) + 1)
    # Numerically null directions: |e| at the eigensolver's rounding (numpy's matrix_rank
    # tolerance for symmetric matrices), so that rank-deficient data drop as in rlmmac's TSVD.
    null = np.abs(values) <= np.finfo(np.float64).eps * inner.shape[0] * abs(values[0])
    rank = min(rank, int(np.sum(~null)))
    values, vectors = values[:rank], vectors[:, :rank]
    gamma = damping * float(np.mean(np.abs(values)))
    denominator = values + gamma
    # An indefinite metric (absolute-trace channel) may give e = -gamma: such directions are
    # dropped (counted in the diagnostics), not inverted.
    usable = np.abs(denominator) > NEAR_SINGULAR * float(np.max(np.abs(values)))
    inverse = np.where(usable, 1.0 / np.where(usable, denominator, 1.0), 0.0)
    projector = factor @ vectors  # (N, r)
    out: Array = (projector * inverse) @ projector.T
    return out, gamma, rank
