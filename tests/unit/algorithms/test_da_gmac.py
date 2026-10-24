"""GMAC, GIES's generalised update.

GMAC has no PET equivalent. Its independent references are:

- ``rlmmac`` (itself PET-fork's parity case): one identity l2sq term is RLM-MAC exactly;
- the closed forms ``dX (M_D + gamma M)^-1 S^T L^-1 (d - y)`` of the user's manuscript
  (GIES_R1.tex, eq. ``GMAC_update_formula``) and of MATLAB ``RLM_MAC_GIES_backup.m``;
- the l_p^q Hessians (the manuscript's App. C) against finite differences of the gradients.
"""

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    AdaptiveLocalization,
    DataErrorCovariance,
    DistanceLocalization,
    GMACUpdate,
    RLMMACUpdate,
)
from apex.algorithms.data_assimilation.gmac.parts import (
    HistogramEntropyTransform,
    L1Squared,
    L2Squared,
    LpQ,
    StateTransform,
    TVTransform,
    blocks,
)
from apex.algorithms.data_assimilation.gmac.strategy import build_terms
from apex.algorithms.data_assimilation.kernels import anomalies
from apex.domain import DataKey

Array = NDArray[np.float64]
GRID = (4, 3, 2)
N_STATE = math.prod(GRID)
VARIABLES = [{"name": "permx", "size": N_STATE, "grid": GRID}]


def _problem(
    n_ensemble: int = 12, n_data: int = 30, seed: int = 3
) -> tuple[Array, Array, Array, DataErrorCovariance, Array]:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((N_STATE, n_ensemble))
    forward = rng.standard_normal((n_data, N_STATE))
    predicted = np.tanh(forward @ state)
    reference = np.tanh(forward @ state.mean(axis=-1))
    error = DataErrorCovariance.diagonal(rng.uniform(0.1, 0.3, n_data))
    perturbed = predicted + error.correlate(rng.standard_normal((n_data, n_ensemble))) + 0.5
    return state, predicted, perturbed, error, reference


def _step(strategy: GMACUpdate | RLMMACUpdate, damping: float = 0.7) -> Array:
    state, predicted, perturbed, error, reference = _problem()
    return strategy.step(
        state, predicted, perturbed, error, damping=damping, reference_prediction=reference
    )


def _localization() -> DistanceLocalization:
    """A distance taper of radius 5 cells about cell 3, prepared for ``_problem``'s data."""
    state, predicted, *_ = _problem()
    localization = DistanceLocalization([N_STATE, 1, 1], {"P": [[3, 1, 1]]}, 5.0, [N_STATE])
    keys = [DataKey(f"t{i}", "WOPR:P") for i in range(predicted.shape[0])]
    localization.prepare(state, predicted, keys, np.random.default_rng(0))
    return localization


def _closed_form(metric: Array, gamma: float | Array) -> Array:
    """``dX (M_D + gamma_j M_j)^-1 S^T L^-1 (d_j - y_j)`` member by member."""
    state, predicted, perturbed, error, reference = _problem()
    n = state.shape[-1]
    s = error.whiten((predicted - reference[:, None]) / math.sqrt(n - 1))
    innovations = error.whiten(perturbed - predicted)
    metrics = np.broadcast_to(metric, (n, n, n)) if metric.ndim == 2 else metric
    gammas = np.broadcast_to(gamma, (n,))
    columns = [
        anomalies(state)
        @ np.linalg.solve(s.T @ s + gammas[j] * metrics[j], s.T @ innovations[:, j])
        for j in range(n)
    ]
    return np.stack(columns, axis=-1)


# ---------------------------------------------------------------------- RLM-MAC exactly


@pytest.mark.parametrize("energy", [1.0, 0.9])
@pytest.mark.parametrize("how", ["terms", "code5", "code3", "default"])
def test_one_identity_term_is_rlmmac_bit_for_bit(energy: float, how: str) -> None:
    """``default``: no terms and no code, gmac.md's minimal pairing (release tester)."""
    options = {
        "terms": {"terms": [{"transform": "identity", "norm": "l2sq"}]},
        "code5": {"code": "10000"},
        "code3": {"code": "100"},
        "default": {},
    }[how]
    gmac = GMACUpdate(**options, energy=energy, state_variables=VARIABLES)  # type: ignore[arg-type]

    np.testing.assert_array_equal(_step(gmac), _step(RLMMACUpdate(energy=energy)))


def test_one_identity_term_with_a_localization_is_rlmmac_with_it_bit_for_bit() -> None:
    gmac = GMACUpdate(code="10000", state_variables=VARIABLES, localization=_localization())

    np.testing.assert_array_equal(_step(gmac), _step(RLMMACUpdate(localization=_localization())))


def test_a_zero_trace_term_adds_nothing() -> None:
    """Members that differ by a constant have equal TV features: the TV l1sq term's Hessian
    points are 0 (sgn 0 = 0), its trace 0, its weight 0, not NaN (MATLAB's 0/0)."""
    state, predicted, perturbed, error, reference = _problem()
    # integers, so that the differences are exact (rounding would give sgn = +-1 noise)
    shifted = np.round(4 * state[:, :1]) + np.arange(state.shape[-1], dtype=float)
    gmac = GMACUpdate(
        terms=[
            {"transform": "identity", "norm": "l2sq"},
            {"transform": "tv", "norm": "l1sq"},
        ],
        state_variables=VARIABLES,
    )
    _, by_member = gmac.metrics(shifted)
    assert by_member is not None
    assert not np.any(by_member)

    step = gmac.step(
        shifted, predicted, perturbed, error, damping=0.7, reference_prediction=reference
    )

    expected = RLMMACUpdate().step(
        shifted, predicted, perturbed, error, damping=0.7, reference_prediction=reference
    )
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


def test_a_fixed_identity_weight_is_the_papers_half_hessian() -> None:
    """``R = 1/2 sum w_k ||.||``: an identity l2sq term of fixed weight 1 is I, so rlmmac."""
    gmac = GMACUpdate(
        terms=[{"transform": "identity", "weight": {"rule": "fixed", "value": 1.0}}],
        state_variables=VARIABLES,
    )

    np.testing.assert_array_equal(_step(gmac), _step(RLMMACUpdate()))


# ---------------------------------------------------------------------- the closed forms


@pytest.mark.parametrize("code", ["11000", "10100", "10001", "11111"])
def test_the_subspace_form_is_the_closed_form_when_nothing_is_truncated(code: str) -> None:
    """With the identity term (a term of centred features alone, e.g. ``01100``, is
    singular along the ones vector, so its inverse is a pseudo-inverse: see below).
    With a full-rank metric and no truncation, ``F (F^T M_D F + gamma I)^-1 F^T S^T`` is
    ``(M_D + gamma M)^-1 S^T``, with ``gamma_j = lambda mean(s_j^2)`` and ``s_j^2`` the
    generalised eigenvalues of ``(M_D, M_j)``."""
    gmac = GMACUpdate(code=code, energy=1.0, metric_floor=0.0, state_variables=VARIABLES)
    state, predicted, _, error, reference = _problem()
    n = state.shape[-1]
    shared, by_member = gmac.metrics(state)
    metric = (0 if shared is None else shared) + (0 if by_member is None else by_member)
    metrics = np.broadcast_to(metric, (n, n, n))
    s = error.whiten((predicted - reference[:, None]) / math.sqrt(n - 1))
    gammas = np.array(
        [0.7 * float(np.mean(np.linalg.eigvals(np.linalg.solve(m, s.T @ s)).real)) for m in metrics]
    )
    assert all(np.linalg.matrix_rank(m) == n for m in metrics)

    np.testing.assert_allclose(_step(gmac), _closed_form(np.asarray(metric), gammas), rtol=1e-8)


@pytest.mark.parametrize("code", ["10000", "11000", "10100"])
def test_the_direct_form_is_its_closed_form(code: str) -> None:
    """MATLAB ``RLM_MAC_GIES_backup.m``: ``alpha_j = lambda tr(M_D) / tr(M_j)``."""
    gmac = GMACUpdate(code=code, form="direct", state_variables=VARIABLES)
    state, predicted, _, error, reference = _problem()
    n = state.shape[-1]
    shared, by_member = gmac.metrics(state)
    metric = np.broadcast_to(
        (0 if shared is None else shared) + (0 if by_member is None else by_member), (n, n, n)
    )
    s = error.whiten((predicted - reference[:, None]) / math.sqrt(n - 1))
    alphas = np.array([0.7 * np.trace(s.T @ s) / np.trace(m) for m in metric])

    np.testing.assert_allclose(_step(gmac), _closed_form(np.asarray(metric), alphas), rtol=1e-8)


def test_a_singular_metric_is_inverted_on_its_range() -> None:
    """``01100`` (TV only): ``M_j`` has rank ``N - 1`` (centred features), and the subspace
    form is the closed form restricted to its range, ``P (P^T (M_D + gamma M) P)^-1 P^T``."""
    gmac = GMACUpdate(
        code="01100", energy=1.0, metric_floor=1e-12, metric_energy=1 - 1e-12,
        state_variables=VARIABLES,
    )  # fmt: skip
    state, predicted, perturbed, error, reference = _problem()
    n = state.shape[-1]
    shared, by_member = gmac.metrics(state)
    assert shared is not None
    assert by_member is not None
    s = error.whiten((predicted - reference[:, None]) / math.sqrt(n - 1))
    innovations = error.whiten(perturbed - predicted)
    columns = []
    for j in range(n):
        m = shared + by_member[j]
        values, vectors = np.linalg.eigh(m)
        p = vectors[:, values > 1e-12 * values.max()]
        assert p.shape[1] == n - 1
        projected_data = (s @ p).T @ (s @ p)
        generalised = np.linalg.eigvals(np.linalg.solve(p.T @ m @ p, projected_data)).real
        gamma = 0.7 * float(np.mean(generalised))
        inner = np.linalg.solve(projected_data + gamma * p.T @ m @ p, (s @ p).T @ innovations[:, j])
        columns.append(anomalies(state) @ (p @ inner))

    np.testing.assert_allclose(_step(gmac), np.stack(columns, axis=-1), rtol=1e-7, atol=1e-10)


def test_without_the_identity_term_the_energy_rule_bounds_the_conditioning() -> None:
    """TV alone on smooth fields: a floor of 1e-10 keeps eigenvalues ~1e-10 of the largest,
    whose S F_j columns (scaled by mu^-1/2) then dominate the spectrum and gamma; the default
    energy rule (0.99, MATLAB's tsvdCut) keeps the well-determined ones."""
    rng = np.random.default_rng(11)
    n = 40
    x = np.linspace(0, 1, N_STATE)[:, None]
    state = np.sin(3 * x * rng.uniform(0.5, 1.5, n) + rng.uniform(0, 6, n)) + 0.1 * x
    forward = rng.standard_normal((30, N_STATE))
    predicted = forward @ state
    reference = forward @ state.mean(axis=-1)
    error = DataErrorCovariance.diagonal(np.ones(30))
    s = error.whiten((predicted - reference[:, None]) / math.sqrt(n - 1))

    def spread(gmac: GMACUpdate) -> float:
        shared, by_member = gmac.metrics(state)
        assert shared is not None
        assert by_member is not None
        values = np.linalg.svd(s @ gmac._factor(shared + by_member[0]), compute_uv=False)
        return float(values[0] / np.median(values))

    default = GMACUpdate(code="01100", state_variables=VARIABLES)
    floor_only = GMACUpdate(
        code="01100", metric_floor=1e-10, metric_energy=1 - 1e-15, state_variables=VARIABLES
    )

    assert default.metric_energy == 0.99
    assert spread(default) < 100
    assert spread(floor_only) > 100 * spread(default)


def test_the_energy_rule_never_inverts_a_null_eigenvalue() -> None:
    """A TV metric is singular along the ones vector: however close the cut is to 1, the
    floor still drops that direction (the review's NaN at metric_energy = 1)."""
    gmac = GMACUpdate(code="01000", metric_energy=1 - 1e-16, state_variables=VARIABLES)
    state = _problem()[0]
    shared, _ = gmac.metrics(state)
    assert shared is not None

    factor = gmac._factor(shared)

    assert factor.shape[1] == state.shape[-1] - 1
    assert np.all(np.isfinite(factor))


def test_zero_damping_is_the_least_squares_step_whatever_the_metric() -> None:
    """At gamma = 0 the metric only selects the subspace: a full-rank one changes nothing."""
    gmac = GMACUpdate(code="11000", energy=1.0, metric_floor=0.0, state_variables=VARIABLES)

    np.testing.assert_allclose(
        _step(gmac, damping=0.0), _step(RLMMACUpdate(energy=1.0), damping=0.0), rtol=1e-7
    )


def test_the_metric_energy_rule_keeps_fewer_directions() -> None:
    floor = GMACUpdate(code="10100", state_variables=VARIABLES)
    energy = GMACUpdate(code="10100", metric_energy=0.5, state_variables=VARIABLES)
    state = _problem()[0]
    _, by_member = floor.metrics(state)
    assert by_member is not None

    assert energy._factor(by_member[0] + np.eye(12) * 0.5).shape[1] < 12
    assert floor._factor(by_member[0] + np.eye(12) * 0.5).shape[1] == 12


# ---------------------------------------------------------------------- the norms


def _lpq_value_gradient(x: Array, p: float, q: float) -> tuple[float, Array]:
    norm = float(np.sum(np.abs(x) ** p) ** (1 / p))
    return norm**q, q * norm ** (q - p) * np.abs(x) ** (p - 1) * np.sign(x)


@pytest.mark.parametrize(("p", "q"), [(2.0, 2.0), (1.5, 2.0), (3.0, 2.0), (2.0, 3.0), (1.3, 1.7)])
def test_the_lpq_hessian_is_the_derivative_of_its_gradient(p: float, q: float) -> None:
    rng = np.random.default_rng(5)
    x = rng.uniform(0.5, 2.0, 6) * rng.choice([-1, 1], 6)
    h = 1e-6
    numeric = np.stack(
        [(_lpq_value_gradient(x + h * e, p, q)[1] - _lpq_value_gradient(x - h * e, p, q)[1])
         / (2 * h) for e in np.eye(6)]
    )  # fmt: skip
    a = np.eye(6)  # A = I: the metric is the Hessian itself
    t = np.tile(x[:, None], (1, 6))

    metric = LpQ(p, q).metric(a, t)

    analytic = metric if metric.ndim == 2 else metric[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_l2sq_and_l1sq_are_lpq_at_their_exponents() -> None:
    rng = np.random.default_rng(6)
    a, t = rng.standard_normal((9, 5)), rng.standard_normal((9, 5))

    np.testing.assert_allclose(L2Squared().metric(a, t), LpQ(2, 2).metric(a, t), rtol=1e-14)
    np.testing.assert_allclose(L1Squared().metric(a, t), LpQ(1, 2).metric(a, t), rtol=1e-12)
    # and l1sq is the limit of l_p^2 as p -> 1+
    # (the diagonal term is O(p - 1) |x|^-1, so the entries near 0 converge in absolute terms)
    limit = L1Squared().metric(a, t)
    np.testing.assert_allclose(
        LpQ(1 + 1e-7, 2).metric(a, t), limit, atol=1e-5 * np.abs(limit).max()
    )


def test_l1sq_is_rank_one_per_member() -> None:
    rng = np.random.default_rng(7)
    metric = L1Squared().metric(rng.standard_normal((9, 5)), rng.standard_normal((9, 5)))

    assert metric.shape == (5, 5, 5)
    assert [np.linalg.matrix_rank(m) for m in metric] == [1] * 5


def test_l1_is_refused() -> None:
    with pytest.raises(ValueError, match="zero Hessian"):
        LpQ(1, 1)


# ---------------------------------------------------------------------- the transforms


def test_tv_takes_first_differences_along_each_axis_in_eclipse_order() -> None:
    state = np.arange(N_STATE, dtype=float)[:, None] ** 2
    field = state[:, 0].reshape(GRID, order="F")

    features = TVTransform().features(state, blocks(VARIABLES))

    expected = np.concatenate([np.diff(field, axis=k).ravel(order="F") for k in range(3)])
    np.testing.assert_array_equal(features[:, 0], expected)


def test_tv_stacks_the_variables_and_can_scale_each() -> None:
    """Two gridded variables of very different units: unscaled, PERMX dominates; scaled,
    each variable's differences have unit RMS."""
    rng = np.random.default_rng(12)
    layout = blocks([{"name": "poro", "size": N_STATE, "grid": GRID},
                     {"name": "permx", "size": N_STATE, "grid": GRID}])  # fmt: skip
    state = np.concatenate([0.01 * rng.standard_normal((N_STATE, 8)),
                            100 * rng.standard_normal((N_STATE, 8))])  # fmt: skip
    rows = 3 * 3 * 2 + 4 * 2 * 2 + 4 * 3 * 1  # differences along x, y and z per variable

    a, _ = TVTransform().anomalies(state, layout)
    scaled, _ = TVTransform(scale_per_variable=True).anomalies(state, layout)

    assert a.shape == (2 * rows, 8)
    assert np.sqrt(np.mean(a[rows:] ** 2)) > 1e3 * np.sqrt(np.mean(a[:rows] ** 2))
    np.testing.assert_allclose(
        [np.sqrt(np.mean(scaled[:rows] ** 2)), np.sqrt(np.mean(scaled[rows:] ** 2))], [1, 1]
    )
    poro_only, _ = TVTransform(variables=["poro"]).anomalies(state, layout)
    np.testing.assert_array_equal(poro_only, a[:rows])


def test_the_state_transform_selects_variables() -> None:
    layout = blocks([{"name": "a", "size": 3}, {"name": "b", "size": 2}])
    state = np.arange(15, dtype=float).reshape(5, 3) ** 2

    a, t = StateTransform(variables=["b"]).anomalies(state, layout)

    centred = state[3:] - state[3:].mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(a, centred / math.sqrt(2))
    np.testing.assert_allclose(t, -centred)


def test_a_layout_that_does_not_match_the_state_is_refused() -> None:
    gmac = GMACUpdate(code="100", state_variables=[{"name": "x", "size": 5}])

    with pytest.raises(ValueError, match="rows"):
        _step(gmac)


def test_tv_needs_a_grid() -> None:
    with pytest.raises(ValueError, match="gridded"):
        TVTransform().features(np.zeros((5, 2)), blocks([{"name": "x", "size": 5}]))


def test_histogram_entropy_weights_each_bin_by_its_count() -> None:
    """Counts (2, 1, 1) of 4 values: c (ln 4 - ln c) = (2 ln 2, ln 4, ln 4)."""
    transform = HistogramEntropyTransform(bins=3, ranges={"x": [0.0, 3.0]})

    features = transform._entropy(np.array([[0.5], [0.7], [1.5], [2.5]]), 0.0, 3.0)

    np.testing.assert_allclose(features[:, 0], [2 * math.log(2), math.log(4), math.log(4)])


def test_values_outside_fixed_ranges_fall_in_the_end_bins() -> None:
    transform = HistogramEntropyTransform(bins=4, ranges={"x": [0.0, 4.0]})
    values = np.array([[-5.0], [0.5], [3.5], [9.0]])

    features = transform._entropy(values, 0.0, 4.0)

    # counts (2, 0, 0, 2) of 4 values: 2 (ln 4 - ln 2) in each end bin
    np.testing.assert_allclose(features[:, 0], [2 * math.log(2), 0, 0, 2 * math.log(2)])


def test_histogram_bins_are_shared_by_the_members_and_the_mean() -> None:
    state = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    a, t = HistogramEntropyTransform(bins=2).anomalies(state, blocks([{"name": "x", "size": 2}]))

    assert a.shape == (2, 3)
    assert t.shape == (2, 3)
    np.testing.assert_allclose(t, -a * math.sqrt(2))


# ---------------------------------------------------------------------- the configuration


def test_the_code_expands_to_the_papers_terms_with_equal_alpha() -> None:
    terms = build_terms(None, "10101", ["permx"])

    assert [t.name for t in terms] == [
        "identity/l2sq",
        "tv/l1sq",
        "histogram_entropy/l1sq",
    ]
    assert [t.value for t in terms] == [pytest.approx(1 / 3)] * 3
    assert [t.name for t in build_terms(None, "011", None)] == ["tv/l2sq", "tv/l1sq"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"code": "100", "terms": [{"transform": "identity"}]}, "terms or code, not both"),
        ({"code": "1010"}, "5 \\(or 3\\) bits"),
        ({"code": "00000"}, "5 \\(or 3\\) bits"),
        ({"terms": [{"transform": "identity", "colour": 1}]}, "unknown keys"),
        ({"terms": [{"transform": "identity", "weight": {"rule": "max"}}]}, "trace or fixed"),
        ({"terms": [{"transform": "identity", "weight": {"rule": "fixed", "value": -1}}]},
         "non-negative"),
        ({"code": "100", "form": "dual"}, "subspace or direct"),
        ({"code": "100", "metric_energy": 1.0}, "metric_energy"),
        ({"terms": [{"transform": "identity"}], "variables": ["permx"]}, "code only"),
        ({"variables": ["permx"]}, "code only"),
        ({"constraints": [{"function": {"name": "box",
                                        "options": {"bounds": {"permx": [0.0, 1.0]}}}}],
          "variables": ["permx"]}, "code only"),  # C-GIES's default identity takes none
        ({"code": "110", "variables": ["nope"]}, "no state variables"),
        ({"terms": [{"transform": {"name": "tv", "options": {"variables": ["flat"]}}}]},
         "no grid"),
        ({"terms": [{"transform": {"name": "histogram_entropy",
                                   "options": {"ranges": {"poro": [0, 1]}}}}]}, "ranges name"),
        ({"terms": [{"transform": "identity", "norm": "l1sq"}]}, "identity takes l2sq only"),
    ],
)  # fmt: skip
def test_bad_configurations_are_refused(options: dict[str, object], message: str) -> None:
    """At construction, before any forecast spends simulations."""
    variables = [*VARIABLES, {"name": "flat", "size": 3}]
    with pytest.raises(ValueError, match=message):
        GMACUpdate(**options, state_variables=variables)  # type: ignore[arg-type]


class _WholeGainOnly:
    """A localization with no per-member form (``localized_step`` only)."""

    def prepare(self, *args: object) -> None: ...

    def localized_step(self, dx: Array, gain: Array, r: Array, rng: object) -> Array:
        return dx @ (gain @ r)

    def export_state(self) -> dict[str, Array]:
        return {}

    def restore_state(self, arrays: object) -> None: ...


def test_a_localization_without_a_per_member_form_is_refused_for_member_dependent_terms() -> None:
    GMACUpdate(code="11000", state_variables=VARIABLES, localization=_WholeGainOnly())

    with pytest.raises(ValueError, match="localized_member_steps"):
        GMACUpdate(code="10100", state_variables=VARIABLES, localization=_WholeGainOnly())


@pytest.mark.parametrize("form", ["subspace", "direct"])
def test_an_untapered_localization_of_per_member_gains_is_the_unlocalized_step(
    form: Literal["subspace", "direct"],
) -> None:
    """Revision 1: member-dependent metrics (``10100``: TV l1sq) localize each member's own
    gain; with a taper of 1 (radius 1e9 cells) that is the plain per-member step."""
    state, predicted, *_ = _problem()
    wide = DistanceLocalization([N_STATE, 1, 1], {"P": [[3, 1, 1]]}, 1e9, [N_STATE])
    keys = [DataKey(f"t{i}", "WOPR:P") for i in range(predicted.shape[0])]
    wide.prepare(state, predicted, keys, np.random.default_rng(0))
    local = GMACUpdate(code="10100", form=form, state_variables=VARIABLES, localization=wide)
    plain = GMACUpdate(code="10100", form=form, state_variables=VARIABLES)

    np.testing.assert_allclose(_step(local), _step(plain), rtol=1e-9, atol=1e-12)
    tight = GMACUpdate(
        code="10100", form=form, state_variables=VARIABLES, localization=_localization()
    )
    assert not np.allclose(_step(tight), _step(plain))


def test_adaptive_per_update_localizes_per_member_gains() -> None:
    """Its taper comes from the members' projected innovations ``G_j r_j``."""
    state, predicted, perturbed, error, reference = _problem()
    localization = AdaptiveLocalization([N_STATE])
    gmac = GMACUpdate(code="10100", state_variables=VARIABLES, localization=localization)

    step = gmac.step(
        state, predicted, perturbed, error, damping=0.7, reference_prediction=reference,
        rng=np.random.default_rng(4),
    )  # fmt: skip

    assert step.shape == state.shape
    assert np.all(np.isfinite(step))
    assert localization.last_taper is not None


class _Squares:
    """A fake parameterization map: the property ``q = x**2`` of the gridded state ``permx``."""

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ("q",)
        self.grids: Mapping[str, tuple[int, ...]] = {"q": GRID}

    def __call__(self, state: Array) -> tuple[Array, list[dict[str, object]]]:
        return state**2, [{"name": "q", "size": N_STATE, "grid": GRID}]


def test_with_a_parameterization_the_terms_see_the_properties() -> None:
    """Revision 1: the TV of ``q = x**2``, in secants about ``P(m_mean)`` (not mean(P(m)))."""
    state = _problem()[0]
    gmac = GMACUpdate(
        terms=[{"transform": {"name": "tv"}, "norm": "l2sq", "weight": {"rule": "fixed"}}],
        state_variables=VARIABLES,
        property_map=_Squares(),
    )
    tv = TVTransform()
    layout = blocks([{"name": "q", "size": N_STATE, "grid": GRID}])
    features = tv.features(state**2, layout)
    centre = tv.features(state.mean(axis=-1, keepdims=True) ** 2, layout)[:, 0]
    a = (features - centre[:, None]) / math.sqrt(state.shape[-1] - 1)

    shared, by_member = gmac.metrics(state)

    assert by_member is None
    assert shared is not None
    np.testing.assert_allclose(shared, a.T @ a, rtol=1e-12)


def test_with_a_parameterization_the_terms_name_properties() -> None:
    with pytest.raises(ValueError, match="no state variables"):
        GMACUpdate(
            terms=[{"transform": {"name": "tv", "options": {"variables": ["permx"]}}}],
            state_variables=VARIABLES,
            property_map=_Squares(),
        )


@pytest.mark.parametrize("form", ["subspace", "direct"])
def test_a_localization_changes_the_step_in_either_form(
    form: Literal["subspace", "direct"],
) -> None:
    """The review's blocker: the direct form dropped the localization silently."""
    plain = GMACUpdate(code="11000", form=form, state_variables=VARIABLES)
    local = GMACUpdate(
        code="11000", form=form, state_variables=VARIABLES, localization=_localization()
    )

    assert not np.allclose(_step(local), _step(plain))


@pytest.mark.parametrize("form", ["subspace", "direct"])
@pytest.mark.parametrize("code", ["11000", "10010"])
def test_an_untapered_localization_is_the_unlocalized_step(
    form: Literal["subspace", "direct"], code: str
) -> None:
    """Radius 1e9 cells: the taper is 1 in double precision (as rlmmac's localization test).
    ``10010``: the histogram l2sq metric is shared (secant anomalies), so it localizes too."""
    state, predicted, *_ = _problem()
    wide = DistanceLocalization([N_STATE, 1, 1], {"P": [[3, 1, 1]]}, 1e9, [N_STATE])
    keys = [DataKey(f"t{i}", "WOPR:P") for i in range(predicted.shape[0])]
    wide.prepare(state, predicted, keys, np.random.default_rng(0))
    local = GMACUpdate(code=code, form=form, state_variables=VARIABLES, localization=wide)

    np.testing.assert_allclose(
        _step(local), _step(GMACUpdate(code=code, form=form, state_variables=VARIABLES)),
        rtol=1e-9, atol=1e-12,
    )  # fmt: skip


def test_a_localization_works_with_member_independent_terms() -> None:
    localization = _localization()
    gmac = GMACUpdate(code="11000", state_variables=VARIABLES, localization=localization)

    assert np.all(np.isfinite(_step(gmac)))


def test_the_step_is_deterministic() -> None:
    gmac = GMACUpdate(code="11111", state_variables=VARIABLES)

    np.testing.assert_array_equal(_step(gmac), _step(gmac))


@pytest.mark.benchmark(group="gmac")
def test_one_trial_at_5spots_size(benchmark: pytest.FixtureRequest) -> None:
    """N = 100, n_d = 500, n_state = 2500 (a 50 x 50 grid): the benchmark."""
    rng = np.random.default_rng(8)
    n, n_data, grid = 100, 500, (50, 50, 1)
    state = rng.standard_normal((2500, n))
    predicted = rng.standard_normal((n_data, n))
    error = DataErrorCovariance.diagonal(np.ones(n_data))
    gmac = GMACUpdate(code="111", state_variables=[{"name": "permx", "size": 2500, "grid": grid}])

    step = benchmark(  # type: ignore[operator]
        gmac.step, state, predicted, predicted + 1.0, error, damping=1.0,
        reference_prediction=predicted.mean(axis=-1),
    )  # fmt: skip

    assert step.shape == (2500, n)


@pytest.mark.parametrize("kind", ["distance", "adaptive_per_update"])
def test_equal_member_gains_are_the_whole_gain_step(kind: str) -> None:
    """``localized_member_steps`` with every ``G_j = G`` is ``localized_step`` with ``G``."""
    state, predicted, perturbed, error, _ = _problem()
    rng = np.random.default_rng(9)
    gain = rng.standard_normal((state.shape[-1], predicted.shape[0]))
    innovations = error.whiten(perturbed - predicted)
    dx = anomalies(state)

    def build() -> DistanceLocalization | AdaptiveLocalization:
        return _localization() if kind == "distance" else AdaptiveLocalization([N_STATE])

    whole = build().localized_step(dx, gain, innovations, np.random.default_rng(1))
    members = build().localized_member_steps(
        dx, (gain for _ in range(state.shape[-1])), innovations, np.random.default_rng(1)
    )

    np.testing.assert_allclose(members, whole, rtol=1e-12, atol=1e-14)


def test_the_state_and_histogram_transforms_take_secants_about_the_centre() -> None:
    """``A = (S(x_j) - S(c))/sqrt(N-1)``, ``T = S(c) - S(x_j)`` with ``c`` the properties of
    the mean state, not the mean of the members' (revision 1)."""
    rng = np.random.default_rng(13)
    layout = blocks([{"name": "q", "size": 6}])
    members = rng.uniform(0, 1, (6, 5))
    centre = rng.uniform(0, 1, 6)

    a, t = StateTransform().anomalies(members, layout, centre)

    np.testing.assert_allclose(a, (members - centre[:, None]) / 2.0)
    np.testing.assert_allclose(t, centre[:, None] - members)
    histogram = HistogramEntropyTransform(bins=4, ranges={"q": [0.0, 1.0]})
    a, _ = histogram.anomalies(members, layout, centre)
    expected = histogram._entropy(members, 0, 1) - histogram._entropy(centre[:, None], 0, 1)
    np.testing.assert_allclose(a, expected / 2.0)


@pytest.mark.benchmark(group="gmac")
def test_one_localized_per_member_trial_at_5spots_size(benchmark: pytest.FixtureRequest) -> None:
    """Code 101 (TV l1sq: per-member gains) with a distance taper: N tapered gains."""
    rng = np.random.default_rng(14)
    n, n_data, grid = 100, 500, (50, 50, 1)
    state = rng.standard_normal((2500, n))
    predicted = rng.standard_normal((n_data, n))
    error = DataErrorCovariance.diagonal(np.ones(n_data))
    taper = DistanceLocalization([50, 50, 1], {"P": [[25, 25, 1]]}, 15.0, [2500])
    taper.prepare(
        state, predicted, [DataKey(f"t{i}", "WOPR:P") for i in range(n_data)],
        np.random.default_rng(0),
    )  # fmt: skip
    gmac = GMACUpdate(
        code="101",
        state_variables=[{"name": "permx", "size": 2500, "grid": grid}],
        localization=taper,
    )

    step = benchmark(  # type: ignore[operator]
        gmac.step, state, predicted, predicted + 1.0, error, damping=1.0,
        reference_prediction=predicted.mean(axis=-1), rng=np.random.default_rng(2),
    )  # fmt: skip

    assert step.shape == (2500, n)
