"""GMAC: GIES's generalised update with composable metric terms.

New in APEX; the method is the user's (Luo, *GIES*, manuscript GIES_R1.tex: eqs.
``GMAC_update_formula``, ``GMAC_effective_KGain``, ``GMAC_M_R``, ``GMAC_update_lpq``,
``GMAC_update_lpq_alternative``, ``total_PLH``, ``weights_rule``), and its MATLAB variants
(``MATLAB main/SyntheticStudies/FaciesStudy1/run_scripts/RLM_MAC_GIES.m``: the subspace
form; ``RLM_MAC_GIES_backup.m``: the direct form). PET has no such method.

For each member ``j`` the step metric is ``M_j = sum_k w_k(j) M_k(j)`` (``N x N``), from the
terms of :mod:`.parts`, and with the whitened data anomalies ``S = L^-1 dY`` (dY about
``g(x_mean)``, scaled by ``1/sqrt(N-1)``, as ``rlmmac``):

- **subspace** form (the paper's algorithm, MATLAB ``RLM_MAC_GIES.m``):
  ``M_j = U Lambda Uᵀ`` (eigenvalues below ``metric_floor · max`` dropped, or the energy rule
  ``metric_energy``), ``F_j = U_r Lambda_r^{-1/2}``, ``S F_j = U_j s_j V_jᵀ`` truncated by
  ``energy``, ``gamma_j = λ · mean(s_j²)``, and
  ``step_j = dX F_j V_j diag(s_j/(s_j² + gamma_j)) U_jᵀ L^-1 (d_j - y_j)``;
- **direct** form (MATLAB ``RLM_MAC_GIES_backup.m``): with ``M_D = Sᵀ S``,
  ``alpha_j = λ tr(M_D)/tr(M_j)`` and
  ``step_j = dX (M_D + alpha_j M_j)^+ Sᵀ L^-1 (d_j - y_j)``.

With one ``identity``/``l2sq`` term (its metric is exactly ``I``), the subspace form is
``rlmmac``: the same operations, bit for bit (``F_j = I`` is not decomposed).

A term's weight is ``trace`` (``alpha N / tr(M_k(j))``, the paper's rule; a term of zero trace
adds nothing, where MATLAB's 0/0 gave NaN) or ``fixed``. ``code`` is the paper's shorthand:
5 bits over (I l2², V l2², V l1², H l2², H l1²), or 3 bits over (I l2², V l2², V l1²) as
in its 5Spot appendix, with equal alpha over the bits set.

Localization: a metric that is the same for every member (quadratic norms) gives one gain,
localized as ``rlmmac``'s; a member-dependent one gives each member its own gain ``G_j``,
which the localization tapers through ``localized_member_steps``.

Parameterizations: with ``[[parameterizations]]`` the non-identity terms see the
simulator's properties ``P(m)`` (by their property names), in secants about ``P(m_mean)``:
the paper's transform ``S o P`` (revision 1).
"""

import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from apex.algorithms.data_assimilation.data_error import DataErrorCovariance
from apex.algorithms.data_assimilation.gmac.constrained import (
    Sources,
    build_constraints,
    coefficient,
    constraint_sources,
)
from apex.algorithms.data_assimilation.gmac.constraints import FUNCTION_ROLES, METRIC_ROLES
from apex.algorithms.data_assimilation.gmac.parts import (
    Block,
    GMACNorm,
    GMACTransform,
    IdentityTransform,
    PropertyMapping,
    blocks,
)
from apex.algorithms.data_assimilation.kernels import anomalies, truncated_svd
from apex.algorithms.data_assimilation.localization import (
    Localization,
    MemberLocalization,
    SourceLocalization,
    refuse_projected,
)
from apex.algorithms.data_assimilation.registries import (
    gmac_norms,
    gmac_transforms,
    update_strategies,
)
from apex.algorithms.data_assimilation.update_strategies import check_shapes
from apex.core.diagnostics import EmitsDiagnostics
from apex.core.options import TEXT, Text, TextList, mapping

Array = NDArray[np.float64]

#: The roles of ``constraints``' strings: plugin names and options; the files a
#: constraint reads are inputs.
CONSTRAINTS = mapping(function=FUNCTION_ROLES, metric=METRIC_ROLES, weight=TEXT)
logger = logging.getLogger(__name__)

#: The paper's energy rule for metrics with no identity term (MATLAB ``tsvdCut = 0.99``).
DEFAULT_METRIC_ENERGY = 0.99

#: Ensembles above this size get a warning: each trial costs N eigendecompositions of N x N.
LARGE_ENSEMBLE = 200

#: The paper's 5-bit code's terms, in order, and its 5Spot appendix's 3-bit code's.
CODE_TERMS: dict[int, tuple[tuple[str, str], ...]] = {
    5: (("identity", "l2sq"), ("tv", "l2sq"), ("tv", "l1sq"),
        ("histogram_entropy", "l2sq"), ("histogram_entropy", "l1sq")),
    3: (("identity", "l2sq"), ("tv", "l2sq"), ("tv", "l1sq")),
}  # fmt: skip


class Term:
    """One metric term: a transform, a norm and a weight rule (see the module docstring)."""

    def __init__(self, transform: GMACTransform, norm: GMACNorm,
                 rule: Literal["trace", "fixed"], value: float, name: str) -> None:  # fmt: skip
        if value < 0:
            raise ValueError(f"term {name}: its weight must be non-negative, got {value}")
        self.transform, self.norm, self.rule, self.value, self.name = (
            transform, norm, rule, float(value), name)  # fmt: skip
        # A quadratic norm's metric 2 A^T A is the same for every member, also on a nonlinear
        # transform (its secant anomalies A are shared): one gain serves all.
        self.constant = norm.constant
        if isinstance(transform, IdentityTransform) and not norm.constant:
            # Its Hessian points are all 0 (I_N about its own mean): the term would add nothing.
            raise ValueError(
                f"term {name}: identity takes l2sq only; a non-quadratic norm of the state is "
                "the state transform's (e.g. MATLAB RLM_MAC_L1: transform state, norm l1sq)"
            )

    def metric(
        self, state: Array, layout: Mapping[str, Block], centre: Array | None = None
    ) -> Array:
        """Return the weighted metric: ``(N, N)`` if constant, else ``(N, N, N)``.

        ``state`` ``(n, N)`` holds the variables the transform sees (the state, or the
        properties), ``centre`` ``(n,)`` those of the mean state (default their mean).
        """
        a, t = self.transform.anomalies(state, layout, centre)
        m = self.norm.metric(a, t)
        if self.rule == "fixed":
            # The paper's R = 1/2 sum_k w_k ||.||: w multiplies half the Hessian, so an
            # identity l2sq term of weight w is w I (MATLAB's NLS_coeff_weight * eye(ne)).
            return (0.5 * self.value) * m
        n = state.shape[-1]
        if m.ndim == 2:
            trace = float(np.trace(m))
            return (self.value * n / trace) * m if trace > 0 else np.zeros_like(m)
        traces = np.einsum("jii->j", m)
        scale = np.where(traces > 0, self.value * n / np.where(traces > 0, traces, 1.0), 0.0)
        out: Array = m * scale[:, None, None]
        return out


def by_member_terms(terms: Sequence[Term]) -> bool:
    """Return whether any term's metric depends on the member."""
    return not all(t.constant for t in terms)


def _plugin(entry: Mapping[str, Any] | str) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    return str(entry["name"]), dict(entry.get("options") or {})


def build_terms(terms: Sequence[Mapping[str, Any]] | None, code: str | None,
                variables: Sequence[str] | None) -> list[Term]:  # fmt: skip
    """Return the configured terms, from ``terms`` or the paper's ``code``.

    Raises
    ------
    ValueError
        If neither or both are given, a code is malformed, or a part is unknown.
    """
    if terms is not None and code is not None:
        raise ValueError("give terms or code, not both")
    if terms is None and code is None:
        raise ValueError("give terms or code")
    if terms is not None and variables is not None:
        raise ValueError("variables applies to code only; give each term's own variables")
    if code is not None:
        if len(code) not in CODE_TERMS or set(code) - {"0", "1"} or "1" not in code:
            raise ValueError(f"code must be 5 (or 3) bits with one set, e.g. '10100'; got {code!r}")
        chosen = [p for bit, p in zip(code, CODE_TERMS[len(code)], strict=True) if bit == "1"]
        alpha = 1.0 / len(chosen)
        named = list(variables or ())
        terms = [{"transform": {"name": t, "options": {} if t == "identity" or not named
                                else {"variables": named}},
                  "norm": {"name": n}, "weight": {"rule": "trace", "alpha": alpha}}
                 for t, n in chosen]  # fmt: skip
    assert terms is not None
    built = []
    for k, entry in enumerate(terms):
        t_name, t_options = _plugin(entry["transform"])
        n_name, n_options = _plugin(entry.get("norm", "l2sq"))
        weight = dict(entry.get("weight") or {})
        rule = weight.get("rule", "trace")
        if rule not in ("trace", "fixed"):
            raise ValueError(f"term {k}: weight rule must be trace or fixed, got {rule!r}")
        value = float(weight.get("alpha" if rule == "trace" else "value", 1.0))
        unknown = sorted(set(entry) - {"transform", "norm", "weight"})
        if unknown:
            raise ValueError(f"term {k}: unknown keys {unknown}")
        built.append(Term(gmac_transforms.create(t_name, **t_options),
                          gmac_norms.create(n_name, **n_options), rule, value,
                          f"{t_name}/{n_name}"))  # fmt: skip
    if not built:
        raise ValueError("gmac needs at least one term")
    return built


@update_strategies.register("gmac")
class GMACUpdate(EmitsDiagnostics):
    """GMAC: GIES's update with per-member step metrics from composable terms.

    Parameters
    ----------
    terms
        The metric terms: ``{transform = {name, options}, norm = {name, options},
        weight = {rule = "trace", alpha} | {rule = "fixed", value}}``.
    code
        The paper's shorthand instead of ``terms`` (e.g. ``"10100"``). Without ``terms`` and
        ``code`` the metric is one identity ``l2sq`` term (``code = "100"``): ``rlmmac``.
    variables
        The variables the code's TV and histogram terms use (the properties' names with a
        parameterization); default all (gridded, for TV).
    energy
        Fraction of the singular-value sum of each ``S F_j`` kept (as ``rlmmac``).
    form
        ``subspace`` (the paper's algorithm) or ``direct`` (``M_D + alpha M_j``).
    metric_floor
        Eigenvalues of ``M_j`` below this fraction of the largest are always dropped
        (subspace form).
    metric_energy
        Also keep only the eigenvalues up to the first whose cumulative share exceeds this
        fraction of their sum (the paper's energy rule), in ``(0, 1)``. Default: 0.99 when no
        term is ``identity`` (nothing bounds such a metric below, and its smallest
        eigenvalues would dominate the step), else off: with the I term the metric has many
        equal eigenvalues (all N with I alone), of which the rule would drop an arbitrary
        few once N > 100. The direct form uses none of ``energy``, ``metric_floor`` and
        ``metric_energy``: it inverts ``M_D + alpha M_j`` whole.
    state_variables
        Supplied by the composition: the state's variables (name, size, grid).
    property_map
        Supplied by the composition with ``[[parameterizations]]``: the map to the
        properties, which the non-identity terms then see (secants about ``P(m_mean)``).
    localization
        Any, but ``adaptive_projected``; with member-dependent terms, one with
        ``localized_member_steps``; with constraints, one with ``localized_source_steps``
        (``adaptive_per_update``: a taper per source).
    constraints
        C-GIES's soft constraints: ``{function = {name, options}, metric =
        {name, options}, weight = {rule = "trace", w} | {rule = "fixed", value}}``, with
        functions of ``gmac_constraints`` and metrics of ``gmac_constraint_metrics``
        (default ``log_barrier`` for inequalities, ``log_channel`` for equalities). The
        regularisation defaults to the identity (``code = "100"``). ``metric_rank`` then
        holds the rank kept of ``F^T M_eff F``.
    localize_constraints
        With a localization, taper the constraint terms too, each with its own taper (the
        paper's §2.3; default), or only the data's (the constraints observe the model
        itself: their correlation tapers cannot see a pull shared by every member, e.g. a
        box's toward its centre; the user's option, 2026-10-05).
    """

    requires_reference_prediction = True
    relative_damping = True

    available_diagnostics: ClassVar[Mapping[str, str]] = {
        "metric_rank": "rank of each member's metric M_j after its truncation, (N,)",
        "gamma": "each member's gamma_j = lambda * mean(s_j^2) (subspace) or alpha_j, (N,)",
        "constraint_cost": "each constraint's D_c(0 - c(m_j)) per member, (n_constraints, N)",
        "constraint_weight": "each constraint's weight w_c,j per member, (n_constraints, N)",
        "constraint_violations": "violated inequality rows per member, (n_constraints, N)",
    }

    def __init__(
        self,
        terms: Annotated[Sequence[Mapping[str, Any]], TEXT] | None = None,
        code: Text | None = None,
        variables: TextList | None = None,
        energy: float = 0.95,
        form: Literal["subspace", "direct"] = "subspace",
        metric_floor: float = 1e-10,
        metric_energy: float | None = None,
        state_variables: Sequence[Mapping[str, Any]] = (),
        property_map: PropertyMapping | None = None,
        localization: Localization | None = None,
        constraints: Annotated[Sequence[Mapping[str, Any]], CONSTRAINTS] | None = None,
        localize_constraints: bool = True,
    ) -> None:
        if not 0.0 < energy <= 1.0:
            raise ValueError(f"energy must be a fraction in (0, 1], got {energy}")
        if form not in ("subspace", "direct"):
            raise ValueError(f"form must be subspace or direct, got {form!r}")
        if not 0.0 <= metric_floor < 1.0:
            raise ValueError(f"metric_floor must be in [0, 1), got {metric_floor}")
        if metric_energy is not None and not 0.0 < metric_energy < 1.0:
            raise ValueError(f"metric_energy must be in (0, 1), got {metric_energy}")
        if terms is None and code is None and variables is not None:
            raise ValueError("variables applies to code only; give the code that uses them")
        if terms is None and code is None:
            # One identity l2sq term (trace weight N/tr(I) = 1, so M_j = I exactly): rlmmac
            # bit for bit, and C-GIES's regularisation, as the paper's. A bare
            # ``name = "gmac"`` is thus the documented minimal pairing (release tester).
            code = "100"
        self.terms = build_terms(terms, code, variables)
        self.constraints = build_constraints(constraints or ())
        self.localize_constraints = localize_constraints
        if (
            self.constraints
            and localization is not None
            and not hasattr(localization, "localized_source_steps")
        ):
            raise ValueError(
                "with constraints gmac localizes each source (data, each constraint) with its "
                "own correlation taper (the paper's scheme): use adaptive_per_update; "
                f"{type(localization).__name__} cannot"
            )
        if metric_energy is None and not any(
            isinstance(t.transform, IdentityTransform) for t in self.terms
        ):
            # Without the I term nothing bounds M_j below: its smallest eigenvalues mu get
            # S F_j columns scaled by mu^-1/2, which then dominate the TSVD and gamma_j.
            # The paper's 99 % energy rule (MATLAB tsvdCut) keeps the well-determined ones.
            metric_energy = DEFAULT_METRIC_ENERGY
        refuse_projected(localization, "gmac")
        if (
            localization is not None
            and not all(t.constant for t in self.terms)
            and not isinstance(localization, MemberLocalization)
        ):
            raise ValueError(
                f"gmac's terms here differ per member, and {type(localization).__name__} "
                "cannot localize per-member gains (it has no localized_member_steps)"
            )
        self.energy, self.form = energy, form
        self.metric_floor, self.metric_energy = metric_floor, metric_energy
        self.layout = blocks(state_variables)
        # With a parameterization the terms see the simulator's properties (the paper's TV
        # and histograms are of physical fields), not the latent state (revision 1).
        self.property_map = property_map
        seen = (
            blocks(
                [
                    {"name": n, "size": 0, "grid": property_map.grids.get(n)}
                    for n in property_map.names
                ]
            )
            if property_map is not None
            else self.layout
        )
        if seen:  # fail at configuration, not at the first update after a forecast
            for term in self.terms:
                check = getattr(term.transform, "check", None)
                if check is not None:
                    try:
                        check(seen)
                    except ValueError as error:
                        where = (
                            "with [[parameterizations]] the terms name properties: "
                            if property_map is not None
                            else ""
                        )
                        raise ValueError(f"term {term.name}: {where}{error}") from error
            for constraint in self.constraints:
                try:
                    constraint.function.check(seen)
                except ValueError as error:
                    where = (
                        "with [[parameterizations]] constraints name properties: "
                        if property_map is not None
                        else ""
                    )
                    raise ValueError(f"constraint {constraint.name}: {where}{error}") from error
        self.localization = localization
        self._warned = False

    # ------------------------------------------------------------------ the metric

    def metrics(self, state: Array) -> tuple[Array | None, Array | None]:
        """Return ``(shared, by_member)``: ``(N, N)`` and ``(N, N, N)`` parts (``None``: none)."""
        layout = self.layout or {"state": Block("state", 0, state.shape[0], None)}
        seen, centre = state, None
        if self.property_map is not None and any(
            not isinstance(t.transform, IdentityTransform) for t in self.terms
        ):
            # Secants about the properties of the mean state, P(m_mean) (the paper's S o P).
            seen, variables = self.property_map(state)
            centre = self.property_map(state.mean(axis=-1, keepdims=True))[0][:, 0]
            layout = blocks(variables)
        shared: Array | None = None
        by_member: Array | None = None
        for term in self.terms:
            m = term.metric(seen, layout, centre)
            if m.ndim == 2:
                shared = m if shared is None else shared + m
            else:
                by_member = m if by_member is None else by_member + m
        return shared, by_member

    def _factor(self, m: Array) -> Array:
        """Return ``F = U_r Lambda_r^{-1/2}`` of a symmetric positive semi-definite ``m``."""
        values, vectors = np.linalg.eigh(m)
        values, vectors = values[::-1], vectors[:, ::-1]  # largest first
        if values[0] <= 0:
            raise ValueError("a member's metric has no positive eigenvalue (all terms zero)")
        # The floor always applies: no zero or negative (rounding) eigenvalue is inverted.
        rank = int(np.sum(values > self.metric_floor * values[0]))
        if self.metric_energy is not None:
            share = np.cumsum(np.clip(values, 0, None)) / float(np.sum(np.clip(values, 0, None)))
            # the first rank whose share exceeds the cut, as MATLAB's "> tsvdCut"
            rank = min(rank, int(np.searchsorted(share, self.metric_energy, side="right") + 1))
        out: Array = vectors[:, :rank] / np.sqrt(values[:rank])
        return out

    # ------------------------------------------------------------------ the step

    def step(
        self,
        state: Array,
        predicted: Array,
        perturbed_observations: Array,
        data_error: DataErrorCovariance,
        *,
        damping: float = 0.0,
        reference_prediction: Array | None = None,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Return the state increment ``(n_state, N)``; see :class:`GMACUpdate`.

        Raises
        ------
        ValueError
            As ``rlmmac``'s, or if a member's metric is zero.
        """
        check_shapes(state, predicted, perturbed_observations, data_error)
        if reference_prediction is None or reference_prediction.shape != (predicted.shape[0],):
            raise ValueError(
                "gmac needs reference_prediction (n_data,), the forecast of mean(state)"
            )
        if not np.all(np.isfinite(reference_prediction)):
            raise ValueError("reference_prediction contains non-finite values")
        if damping < 0:
            raise ValueError(f"damping must be non-negative, got {damping}")
        n = state.shape[-1]
        if self.layout and sum(b.size for b in self.layout.values()) != state.shape[0]:
            raise ValueError(
                f"the state has {state.shape[0]} rows, its variables "
                f"{sum(b.size for b in self.layout.values())}"
            )
        if (
            n > LARGE_ENSEMBLE
            and (by_member_terms(self.terms) or self.constraints)
            and not self._warned
        ):
            self._warned = True  # once per strategy, not per trial
            logger.warning(
                "gmac: %d members; each trial forms %d metrics of %d x %d (cost ~N^4)", n, n, n, n
            )
        dx = anomalies(state)  # (n_state, N), / sqrt(N-1)
        dy = (predicted - reference_prediction[:, None]) / math.sqrt(n - 1)
        s_data = data_error.whiten(dy)  # S = L^-1 dY, (n_data, N)
        innovations = data_error.whiten(perturbed_observations - predicted)  # (n_data, N)
        if not np.any(s_data):
            raise ValueError(
                "data anomalies are all zero: every forecast equals the mean-model forecast"
            )
        shared, by_member = self.metrics(state)
        if self.constraints:
            return self._constrained(
                state, dx, s_data, innovations, shared, by_member, damping, rng
            )
        exact_identity = (
            by_member is None and shared is not None and np.array_equal(shared, np.eye(n))
        )
        ranks = np.zeros(n, dtype=np.int64)
        gammas = np.zeros(n)
        step = np.zeros_like(state)
        if self.form == "direct":
            md = s_data.T @ s_data  # (N, N)
            if self.localization is not None and by_member is None:  # one gain for all
                m = self._member(shared, None, 0, n)
                trace = float(np.trace(m))
                alpha = damping * float(np.trace(md)) / trace if trace > 0 else 0.0
                gain_factor = np.linalg.pinv(md + alpha * m) @ s_data.T  # (N, n_data)
                gammas[:] = alpha
                local = self.localization.localized_step(dx, gain_factor, innovations, rng)
                return self._done(local, None, gammas)
            if self.localization is not None:  # per-member gains (revision 1)
                return self._done(
                    self._localized_members(dx, s_data, innovations, shared, by_member,
                                            damping, gammas, None, rng),
                    None, gammas,
                )  # fmt: skip
            for j in range(n):
                m = self._member(shared, by_member, j, n)
                trace = float(np.trace(m))
                alpha = damping * float(np.trace(md)) / trace if trace > 0 else 0.0
                weights = np.linalg.pinv(md + alpha * m) @ (s_data.T @ innovations[:, j])
                step[:, j] = dx @ weights
                gammas[j] = alpha
            return self._done(step, None, gammas)
        if by_member is None:  # one metric for every member: one gain, as rlmmac
            factor = np.eye(n) if exact_identity else self._factor(self._member(shared, None, 0, n))
            u, s, vt, gamma = self._spectrum(s_data @ factor if not exact_identity else s_data,
                                             damping)  # fmt: skip
            ranks[:], gammas[:] = factor.shape[1], gamma
            if self.localization is not None:
                gain_factor = factor @ ((vt.T * (s / (s**2 + gamma))) @ u.T)  # (N, n_data)
                local = self.localization.localized_step(dx, gain_factor, innovations, rng)
                return self._done(local, ranks, gammas)
            projected = (u.T @ innovations) * (s / (s**2 + gamma))[:, None]
            if exact_identity:  # rlmmac's operations, in its order (the parity anchor)
                return self._done(dx @ (vt.T @ projected), ranks, gammas)
            return self._done(dx @ (factor @ (vt.T @ projected)), ranks, gammas)
        if self.localization is not None:  # per-member gains (revision 1)
            return self._done(
                self._localized_members(dx, s_data, innovations, shared, by_member,
                                        damping, gammas, ranks, rng),
                ranks, gammas,
            )  # fmt: skip
        for j in range(n):
            factor = self._factor(self._member(shared, by_member, j, n))
            u, s, vt, gamma = self._spectrum(s_data @ factor, damping)
            weights = factor @ (vt.T @ ((u.T @ innovations[:, j]) * (s / (s**2 + gamma))))
            step[:, j] = dx @ weights
            ranks[j], gammas[j] = factor.shape[1], gamma
        return self._done(step, ranks, gammas)

    def _localized_members(
        self, dx: Array, s_data: Array, innovations: Array, shared: Array | None,
        by_member: Array | None, damping: float, gammas: Array,
        ranks: NDArray[np.int64] | None, rng: np.random.Generator | None,
    ) -> Array:  # fmt: skip
        """Return the localized step with each member's own gain ``G_j`` ``(N, n_data)``.

        The gains are generated one at a time (never all ``N x N x n_data`` at once) for
        the localization's ``localized_member_steps``.
        """
        assert isinstance(self.localization, MemberLocalization)
        n = dx.shape[-1]
        md = s_data.T @ s_data

        def gains() -> Iterator[Array]:
            for j in range(n):
                m = self._member(shared, by_member, j, n)
                if self.form == "direct":
                    trace = float(np.trace(m))
                    alpha = damping * float(np.trace(md)) / trace if trace > 0 else 0.0
                    gammas[j] = alpha
                    yield np.linalg.pinv(md + alpha * m) @ s_data.T
                    continue
                factor = self._factor(m)
                u, s, vt, gamma = self._spectrum(s_data @ factor, damping)
                gammas[j] = gamma
                if ranks is not None:
                    ranks[j] = factor.shape[1]
                yield factor @ ((vt.T * (s / (s**2 + gamma))) @ u.T)

        drawn = 0

        def counted() -> Iterator[Array]:
            nonlocal drawn
            for gain in gains():
                drawn += 1
                yield gain

        step = self.localization.localized_member_steps(dx, counted(), innovations, rng)
        if drawn != n:
            raise ValueError(f"{type(self.localization).__name__} drew {drawn} of {n} member gains")
        return step

    def _constrained(
        self, state: Array, dx: Array, s_data: Array, innovations: Array,
        shared: Array | None, by_member: Array | None, damping: float,
        rng: np.random.Generator | None,
    ) -> Array:  # fmt: skip
        """Return C-GIES's step with the constraint terms (:mod:`.constrained`)."""
        n = dx.shape[-1]
        md = s_data.T @ s_data  # (N, N)
        seen, layout = state, self.layout or {"state": Block("state", 0, state.shape[0], None)}
        centre = state.mean(axis=-1)
        if self.property_map is not None:
            seen, variables = self.property_map(state)
            centre = self.property_map(state.mean(axis=-1, keepdims=True))[0][:, 0]
            layout = blocks(variables)
        effective = np.broadcast_to(md, (n, n, n)).copy()
        sources = Sources(n)
        sources.innovations.append(s_data.T @ innovations)  # the data's r_d,j columns
        constraint_sources(
            self.constraints, seen, centre, layout, float(np.trace(md)), effective, sources
        )
        indefinite = any(
            getattr(c.metric, "curvature", "") == "absolute_trace" for c in self.constraints
        )
        coefficients = np.empty((n, n, n))
        gammas = np.zeros(n)
        ranks = np.zeros(n, dtype=np.int64)
        for j in range(n):
            regularization = self._member(shared, by_member, j, n)
            factor = None if self.form == "direct" else self._factor(regularization)
            coefficients[j], gammas[j], ranks[j] = coefficient(
                self.form, effective[j], regularization, damping, self.energy, factor,
                indefinite,
            )  # fmt: skip
        # A_s,j = coeff_j r_s,j per source: (N, N) each, column j for member j.
        projected = [np.einsum("jab,bj->aj", coefficients, r) for r in sources.innovations]
        self._diagnostics.add("strategy", "constraint_cost", np.stack(sources.costs))
        self._diagnostics.add("strategy", "constraint_weight", np.stack(sources.weights))
        self._diagnostics.add("strategy", "constraint_violations", np.stack(sources.violations))
        if self.localization is not None:
            assert isinstance(self.localization, SourceLocalization)
            if self.localize_constraints:  # the paper's: a taper per source, data and each c
                step = self.localization.localized_source_steps(dx, projected, rng)
            else:  # the data's localized, the constraints' (model-space terms) global
                step = self.localization.localized_source_steps(dx, projected[:1], rng)
                step = step + dx @ np.sum(projected[1:], axis=0)
        else:
            step = dx @ np.sum(projected, axis=0)
        return self._done(step, None if self.form == "direct" else ranks, gammas)

    @staticmethod
    def _member(shared: Array | None, by_member: Array | None, j: int, n: int) -> Array:
        m = np.zeros((n, n)) if shared is None else shared.copy()
        if by_member is not None:
            m += by_member[j]
        return m

    def _spectrum(self, matrix: Array, damping: float) -> tuple[Array, Array, Array, float]:
        """Return ``rlmmac``'s truncated, null-dropped SVD of ``matrix`` and its ``gamma``."""
        u, s, vt = truncated_svd(matrix, self.energy)
        if s[0] == 0.0:
            raise ValueError("the projected data anomalies are all zero")
        keep = s > np.finfo(np.float64).eps * max(matrix.shape) * s[0]
        u, s, vt = u[:, keep], s[keep], vt[keep, :]
        return u, s, vt, damping * float(np.mean(s**2))

    def _done(self, step: Array, ranks: NDArray[np.int64] | None, gammas: Array) -> Array:
        if ranks is not None:  # the direct form inverts M_D + alpha M_j whole: no rank
            self._diagnostics.add("strategy", "metric_rank", ranks)
        self._diagnostics.add("strategy", "gamma", gammas)
        return step
