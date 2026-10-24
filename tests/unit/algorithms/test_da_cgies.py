"""C-GIES: GMAC with soft constraints.

The user's method (Luo & Cruz (2022), Computational Geosciences 26(3), 571-594,
doi:10.1007/s10596-022-10137-7; equation labels of the manuscript DASC_CG_R1.tex), which
PET lacks. Independent references:

- a linear constraint with the quadratic metric and a fixed weight ``w`` is a perfect
  measurement of error variance ``1/w``: the constrained step equals GMAC's on the
  observations augmented by ``(f, 0, I/w)`` (eq. ``GMAC_constrained_update_formula_mid``);
- the metrics' gradients and Hessians against finite differences;
- the sign of the channel's curvature on a toy (the user's decision).
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    AdaptiveLocalization,
    DataErrorCovariance,
    DistanceLocalization,
    GMACUpdate,
)
from apex.algorithms.data_assimilation.gmac.constraints import (
    Box,
    ConstraintViolation,
    Histogram,
    L1Squared,
    LogBarrier,
    LogChannel,
    Quadratic,
)
from apex.algorithms.data_assimilation.gmac.parts import blocks
from apex.algorithms.data_assimilation.kernels import anomalies

Array = NDArray[np.float64]
N_STATE, N_DATA, N = 6, 9, 12
VARIABLES = [{"name": "x", "size": N_STATE}]


def _problem(seed: int = 3) -> tuple[Array, Array, Array, DataErrorCovariance, Array, Array]:
    rng = np.random.default_rng(seed)
    state = 1.0 + 0.3 * rng.standard_normal((N_STATE, N))
    forward = rng.standard_normal((N_DATA, N_STATE))
    predicted = forward @ state + 0.05 * np.sin(state[: N_DATA % N_STATE + 1]).sum(axis=0)
    reference = forward @ state.mean(axis=-1)
    error = DataErrorCovariance.diagonal(rng.uniform(0.5, 1.5, N_DATA))
    perturbed = predicted + error.correlate(rng.standard_normal((N_DATA, N))) + 0.3
    return state, predicted, perturbed, error, reference, forward


def _step(gmac: GMACUpdate, damping: float = 0.7, seed: int = 3) -> Array:
    state, predicted, perturbed, error, reference, _ = _problem(seed)
    return gmac.step(
        state, predicted, perturbed, error, damping=damping, reference_prediction=reference,
        rng=np.random.default_rng(5),
    )  # fmt: skip


def _linear_file(tmp_path: Path, matrix: Array, rhs: Array) -> str:
    path = tmp_path / "constraint.npz"
    np.savez(path, matrix=matrix, rhs=rhs)
    return str(path)


# ---------------------------------------------------------------------- the analytic anchor


@pytest.mark.parametrize("form", ["direct", "subspace"])
def test_a_linear_quadratic_constraint_is_a_perfect_measurement(
    tmp_path: Path, form: Literal["direct", "subspace"]
) -> None:
    """``w D_c[0 - (A x - b)]`` with ``D_c = 1/2 ||.||^2`` is an observation ``0`` of
    ``A x - b`` with error variance ``1/w``: GMAC on the augmented data, step for step."""
    state, predicted, perturbed, error, reference, _ = _problem()
    rng = np.random.default_rng(11)
    a, b, w = rng.standard_normal((4, N_STATE)), rng.standard_normal(4), 2.5
    constraint = {
        "function": {"name": "linear", "options": {"matrix_file": _linear_file(tmp_path, a, b)}},
        "metric": {"name": "quadratic"},
        "weight": {"rule": "fixed", "value": w},
    }
    constrained = GMACUpdate(
        code="100", form=form, energy=1.0, state_variables=VARIABLES, constraints=[constraint]
    )
    augmented_error = DataErrorCovariance.diagonal(
        np.concatenate([_variances(error), np.full(4, 1.0 / w)])
    )
    plain = GMACUpdate(code="100", form=form, energy=1.0, state_variables=VARIABLES)

    expected = plain.step(
        state, np.vstack([predicted, a @ state - b[:, None]]),
        np.vstack([perturbed, np.zeros((4, N))]), augmented_error, damping=0.7,
        reference_prediction=np.concatenate([reference, a @ state.mean(axis=-1) - b]),
    )  # fmt: skip
    step = constrained.step(
        state, predicted, perturbed, error, damping=0.7, reference_prediction=reference
    )

    np.testing.assert_allclose(step, expected, rtol=1e-9, atol=1e-12)


def _variances(error: DataErrorCovariance) -> Array:
    """The diagonal of ``C_d`` (whitening the identity gives ``L^-1``)."""
    whitened = error.whiten(np.eye(N_DATA))
    out: Array = 1.0 / np.diag(whitened) ** 2
    return out


def test_without_constraints_nothing_changes() -> None:
    np.testing.assert_array_equal(
        _step(GMACUpdate(code="100", state_variables=VARIABLES, constraints=[])),
        _step(GMACUpdate(code="100", state_variables=VARIABLES)),
    )


def test_constraints_default_to_the_identity_regularisation() -> None:
    box = {"function": {"name": "box", "options": {"bounds": {"x": [-5.0, 5.0]}}}}

    np.testing.assert_array_equal(
        _step(GMACUpdate(state_variables=VARIABLES, constraints=[box])),
        _step(GMACUpdate(code="100", state_variables=VARIABLES, constraints=[box])),
    )


# ---------------------------------------------------------------------- the metrics


def _finite_difference(function: object, x: Array, h: float = 1e-6) -> Array:
    """Central differences of ``function`` ``(n,) -> (n,)`` at ``x``, ``(n, n)``."""
    rows = []
    for e in np.eye(x.size):
        plus = function(x[:, None] + h * e[:, None])  # type: ignore[operator]
        minus = function(x[:, None] - h * e[:, None])  # type: ignore[operator]
        rows.append((np.asarray(plus) - np.asarray(minus)).reshape(-1) / (2 * h))
    return np.stack(rows)


@pytest.mark.parametrize(
    "metric",
    [Quadratic(), L1Squared(), LogBarrier(0.1), LogChannel(0.1, 1e-3, "absolute_trace")],
    ids=["quadratic", "l1sq", "log_barrier", "log_channel"],
)
def test_the_gradient_is_the_derivative_of_the_cost(metric: object) -> None:
    x = np.array([0.7, -1.3, 0.4, -2.1])  # both signs: the channel's |x|
    if isinstance(metric, LogBarrier):
        x = np.abs(x)  # the barrier's domain, x + a > 0

    numeric: Array = _finite_difference(lambda v: metric.cost(v), x)[:, 0]  # type: ignore[attr-defined]
    analytic: Array = metric.gradient(x[:, None])[:, 0]  # type: ignore[attr-defined]

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize(
    "metric",
    [Quadratic(), LogBarrier(0.1), LogChannel(0.1, 1e-3, "absolute_trace")],
    ids=["quadratic", "log_barrier", "log_channel"],
)
def test_the_metric_is_the_hessian_of_the_cost(metric: object) -> None:
    """With ``A = I`` the metric is the Hessian itself (the channel's is negative)."""
    x = np.array([0.7, -1.3, 0.4, -2.1])
    if isinstance(metric, LogBarrier):
        x = np.abs(x)
    numeric = _finite_difference(lambda v: metric.gradient(v), x)  # type: ignore[attr-defined]

    analytic = metric.metric(np.eye(4), np.tile(x[:, None], (1, 4)))[0]  # type: ignore[attr-defined]

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


def test_at_its_minimum_the_absolute_channel_has_no_pull_and_bounded_curvature() -> None:
    """``x = 0`` (a histogram bin on target): gradient 0 and curvature ``1/b^2``, where the
    scripts' ``1/(x + b sgn(x) + epsilon)`` gives ``1/epsilon`` and ``1/epsilon^2``."""
    x = np.zeros((1, 1))
    absolute = LogChannel(b=0.1, epsilon=1e-3, curvature="absolute_hessian")
    paper = LogChannel(b=0.1, epsilon=1e-3, curvature="paper")

    assert absolute.gradient(x)[0, 0] == 0.0
    np.testing.assert_allclose(absolute.metric(np.eye(1), x)[0, 0, 0], 1 / 0.1**2)
    np.testing.assert_allclose(paper.gradient(x)[0, 0], 1e3)
    np.testing.assert_allclose(paper.metric(np.eye(1), x)[0, 0, 0], -1e6)


def test_the_channel_curvatures() -> None:
    """``absolute_hessian`` is ``|H|``; ``absolute_trace`` and ``paper`` keep ``H < 0``."""
    x = np.array([[0.5], [-1.0]])
    a = np.eye(2)

    absolute = LogChannel(curvature="absolute_hessian").metric(a, x)[0]
    negative = LogChannel(curvature="paper").metric(a, x)[0]
    np.testing.assert_allclose(LogChannel().metric(a, x)[0], absolute)  # the default

    assert np.all(np.diag(absolute) > 0)
    np.testing.assert_allclose(LogChannel(curvature="absolute_trace").metric(a, x)[0], -absolute)
    # the paper's keeps the scripts' denominator x + b sgn(x) + epsilon
    np.testing.assert_allclose(
        np.diag(negative), -1 / (x[:, 0] + 0.1 * np.sign(x[:, 0]) + 1e-3) ** 2
    )


def _channel_toward_zero(curvature: str) -> tuple[float, float]:
    """One step on ``f(m) = m`` alone (no data pull): mean |f| before and after."""
    state, predicted, _, error, reference, _ = _problem()
    gmac = GMACUpdate(
        form="direct", state_variables=VARIABLES,
        constraints=[{
            "function": {"name": "linear", "options": {"matrix_file": _eye_file}},
            "metric": {"name": "log_channel", "options": {"curvature": curvature}},
            "weight": {"rule": "trace", "w": 0.5},
        }],
    )  # fmt: skip
    step = gmac.step(
        state, predicted, predicted, error, damping=1.0, reference_prediction=reference
    )
    return float(np.abs(state).mean()), float(np.abs(state + step).mean())


_eye_file = ""


@pytest.fixture(autouse=True)
def _identity_constraint(tmp_path: Path) -> None:
    global _eye_file
    _eye_file = _linear_file(tmp_path, np.eye(N_STATE), np.zeros(N_STATE))


def test_the_channel_pulls_toward_the_constraint() -> None:
    """With ``|H|`` (``absolute_hessian``); ``absolute_trace``'s metric may be indefinite, so its
    direction is not guaranteed (the user's comparison on facies study 1 decides)."""
    before, after = _channel_toward_zero("absolute_hessian")

    assert after < before


def test_the_papers_channel_pushes_away_from_it() -> None:
    """The finding, kept as ``curvature = "paper"`` for comparisons."""
    before, after = _channel_toward_zero("paper")

    assert after > before


class _Sink:
    """Records every diagnostics item."""

    def __init__(self) -> None:
        self.items: dict[str, Array] = {}

    def wants(self, role: str, name: str) -> bool:
        return True

    def add(self, role: str, name: str, item: object, *, nbytes: int | None = None) -> None:
        self.items[name] = np.asarray(item() if callable(item) else item)


@pytest.mark.parametrize(
    ("curvature", "positive"),
    [("absolute_hessian", True), ("absolute_trace", True), ("paper", False)],
)
def test_trace_weights_are_signed_only_for_the_papers_curvature(
    curvature: str, positive: bool
) -> None:
    gmac = GMACUpdate(
        form="direct", state_variables=VARIABLES,
        constraints=[{"function": {"name": "linear", "options": {"matrix_file": _eye_file}},
                      "metric": {"name": "log_channel", "options": {"curvature": curvature}}}],
    )  # fmt: skip
    sink = _Sink()
    gmac.attach_diagnostics(sink)

    _step(gmac)

    weights = sink.items["constraint_weight"]
    assert weights.shape == (1, N)
    assert bool(np.all(weights > 0)) == positive
    assert sink.items["constraint_cost"].shape == (1, N)


# ---------------------------------------------------------------------- the functions


def test_box_rows_are_lower_minus_x_then_x_minus_upper() -> None:
    layout = blocks([{"name": "p", "size": 2}, {"name": "k", "size": 2}])
    x = np.array([[0.1], [0.5], [3.0], [8.0]])

    values = Box({"k": [2.0, 9.0], "p": [0.0, 0.4]}).values(x, layout)

    # in the order of ``bounds``: k's lower then upper rows, then p's
    np.testing.assert_allclose(values[:, 0], [-1.0, -6.0, -6.0, -1.0, -0.1, -0.5, -0.3, 0.1])


def test_histogram_counts_on_fixed_bins_against_the_reference(tmp_path: Path) -> None:
    np.savez(tmp_path / "truth.npz", x=np.array([0.5, 1.5, 1.5, 3.5]))
    histogram = Histogram("x", [0.0, 4.0], bins=4, reference_file=str(tmp_path / "truth.npz"))
    layout = blocks([{"name": "x", "size": 4}])

    values = histogram.values(np.array([[0.2], [0.3], [2.5], [9.0]]), layout)

    np.testing.assert_allclose(values[:, 0], [2 - 1, 0 - 2, 1 - 0, 1 - 1])


# ---------------------------------------------------------------------- refusals


def test_an_infeasible_member_is_refused_with_the_cause_and_the_fixes() -> None:
    gmac = GMACUpdate(
        state_variables=VARIABLES,
        constraints=[{"function": {"name": "box", "options": {"bounds": {"x": [0.0, 1.0]}}}}],
    )

    with pytest.raises(ConstraintViolation) as caught:
        _step(gmac)

    message = str(caught.value)
    assert "violate h <= 0" in message
    assert "set lower/upper within the box" in message
    assert "raise the barrier's a" in message


def test_distance_localization_is_refused_with_constraints() -> None:
    localization = DistanceLocalization([N_STATE, 1, 1], {"P": [[3, 1, 1]]}, 5.0, [N_STATE])
    box = {"function": {"name": "box", "options": {"bounds": {"x": [-5.0, 5.0]}}}}

    with pytest.raises(ValueError, match="adaptive_per_update"):
        GMACUpdate(state_variables=VARIABLES, constraints=[box], localization=localization)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"metric": {"name": "quadratic"}}, "no function"),
        ({"function": {"name": "box"}, "colour": 1}, "unknown keys"),
        ({"function": "box"}, "must be a table"),
        ({"function": {"options": {}}}, "must be a table"),
        ({"function": {"name": "nope"}}, "nope"),
        ({"function": {"name": "box", "options": {"bounds": {"x": [0, 1]}}},
          "metric": {"name": "nope"}}, "nope"),
        ({"function": {"name": "histogram", "options": {
            "variable": "x", "range": [0, 1], "reference_file": "/no/such.npz"}}}, "constraint 0"),
        ({"function": {"name": "linear", "options": {"matrix_file": "/no/such.npz"}}},
         "constraint 0"),
        ({"function": {"name": "box", "options": {"bounds": {"x": [0, 1]}}},
          "weight": "trace"}, "weight must be a table"),
        ({"function": {"name": "box", "options": {"bounds": {"y": [0, 1]}}}}, "no state variables"),
        ({"function": {"name": "box", "options": {"bounds": {"x": [0, 1]}}},
          "weight": {"rule": "max"}}, "trace or fixed"),
    ],
)  # fmt: skip
def test_bad_constraints_are_refused(entry: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GMACUpdate(state_variables=VARIABLES, constraints=[entry])


# ---------------------------------------------------------------------- localization, properties


def test_per_source_localization_runs_and_is_reproducible() -> None:
    box = {"function": {"name": "box", "options": {"bounds": {"x": [-5.0, 5.0]}}}}

    def run() -> Array:
        return _step(
            GMACUpdate(state_variables=VARIABLES, constraints=[box],
                       localization=AdaptiveLocalization([N_STATE]))
        )  # fmt: skip

    first = run()
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, run())


def test_one_source_is_the_whole_gain_step() -> None:
    state = _problem()[0]
    rng = np.random.default_rng(2)
    projected = rng.standard_normal((N, N))
    dx = anomalies(state)

    whole = AdaptiveLocalization([N_STATE]).localized_step(
        dx, projected, np.eye(N), np.random.default_rng(1)
    )
    sources = AdaptiveLocalization([N_STATE]).localized_source_steps(
        dx, [projected], np.random.default_rng(1)
    )

    np.testing.assert_allclose(sources, whole, rtol=1e-12, atol=1e-14)


class _Exp:
    """A fake parameterization: the property ``k = exp(x)``."""

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ("k",)
        self.grids: Mapping[str, tuple[int, ...]] = {}

    def __call__(self, state: Array) -> tuple[Array, list[dict[str, object]]]:
        return np.exp(state), [{"name": "k", "size": N_STATE, "grid": None}]


def test_with_a_parameterization_constraints_see_the_properties() -> None:
    """A box on ``k = exp(x)`` in ``[e^-5, e^5]`` is the box on ``x`` in ``[-5, 5]`` up to
    the barrier's rows: the steps differ, and the latent name is refused."""
    on_property = {
        "function": {"name": "box", "options": {"bounds": {"k": [math.exp(-5), math.exp(5)]}}}
    }
    gmac = GMACUpdate(state_variables=VARIABLES, property_map=_Exp(), constraints=[on_property])

    assert np.all(np.isfinite(_step(gmac)))
    with pytest.raises(ValueError, match="constraints name properties"):
        GMACUpdate(
            state_variables=VARIABLES, property_map=_Exp(),
            constraints=[{"function": {"name": "box", "options": {"bounds": {"x": [-5, 5]}}}}],
        )  # fmt: skip


@pytest.mark.benchmark(group="gmac")
def test_one_constrained_trial_at_scale(benchmark: pytest.FixtureRequest) -> None:
    """A box on 10 000 values (n_c = 20 000) and a 50-bin histogram, N = 100, n_d = 500:
    the IN metric is N products of n_c x N x N (0.8 to 4 s per trial measured)."""
    rng = np.random.default_rng(15)
    n, n_data, size = 100, 500, 10_000
    state = rng.uniform(0.2, 0.8, (size, n))
    predicted = rng.standard_normal((n_data, n))
    error = DataErrorCovariance.diagonal(np.ones(n_data))
    gmac = GMACUpdate(
        state_variables=[{"name": "x", "size": size}],
        constraints=[
            {"function": {"name": "box", "options": {"bounds": {"x": [0.0, 1.0]}}}},
            {"function": {"name": "histogram",
                          "options": {"variable": "x", "range": [0.0, 1.0], "bins": 50,
                                      "counts": [size / 50] * 50}}},
        ],
    )  # fmt: skip

    step = benchmark(  # type: ignore[operator]
        gmac.step, state, predicted, predicted + 1.0, error, damping=1.0,
        reference_prediction=predicted.mean(axis=-1),
    )  # fmt: skip

    assert step.shape == (size, n)


# ---------------------------------------------------------------------- directions


def test_the_box_reduces_the_barrier_cost_relative_to_unconstrained_gies() -> None:
    """A box the data pull toward: C-GIES's step ends with a lower barrier cost."""
    state, predicted, perturbed, error, reference, _ = _problem()
    box = {"function": {"name": "box", "options": {"bounds": {"x": [0.0, 2.0]}}},
           "weight": {"rule": "trace", "w": 2.0}}  # fmt: skip
    plain = GMACUpdate(code="100", state_variables=VARIABLES)
    constrained = GMACUpdate(state_variables=VARIABLES, constraints=[box])
    barrier, layout = LogBarrier(0.1), blocks(VARIABLES)

    def cost(step: Array) -> float:
        values = Box({"x": [0.0, 2.0]}).values(state + step, layout)
        return float(np.mean(barrier.cost(np.maximum(-values, -0.099))))

    kwargs = {"damping": 0.7, "reference_prediction": reference}
    assert cost(constrained.step(state, predicted, perturbed, error, **kwargs)) < cost(  # type: ignore[arg-type]
        plain.step(state, predicted, perturbed, error, **kwargs)  # type: ignore[arg-type]
    )


def test_the_histogram_channel_moves_the_counts_toward_the_target(tmp_path: Path) -> None:
    """With no data pull, the histogram term alone moves the members' counts toward the
    reference's (the default, absolute-Hessian channel; paper's pushes away)."""
    state, predicted, _, error, reference, _ = _problem()
    target = np.concatenate([np.full(N_STATE // 2, 0.6), np.full(N_STATE - N_STATE // 2, 1.4)])
    np.savez(tmp_path / "ref.npz", x=target)
    histogram = Histogram("x", [0.0, 2.0], bins=4, reference_file=str(tmp_path / "ref.npz"))
    gmac = GMACUpdate(
        form="direct", state_variables=VARIABLES,
        constraints=[{"function": {"name": "histogram", "options": {
            "variable": "x", "range": [0.0, 2.0], "bins": 4,
            "reference_file": str(tmp_path / "ref.npz")}},
            "metric": {"name": "log_channel", "options": {"curvature": "absolute_hessian"}}}],
    )  # fmt: skip
    layout = blocks(VARIABLES)

    step = gmac.step(
        state, predicted, predicted, error, damping=0.1, reference_prediction=reference
    )

    before = np.abs(histogram.values(state, layout)).sum()
    after = np.abs(histogram.values(state + step, layout)).sum()
    assert after < before


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_unlocalized_constraints_add_their_global_step(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    """``localize_constraints = false``: ``(T_d o dX) A_d + dX A_box``. A taper of ones is
    the unlocalized step either way; a taper of zeros leaves only the box's global part,
    which the paper's (every source tapered) zeroes too."""
    box = {"function": {"name": "box", "options": {"bounds": {"x": [-5.0, 5.0]}}}}
    monkeypatch.setattr(
        AdaptiveLocalization, "_taper", lambda self, dx, p, rng: np.full(dx.shape, value)
    )

    def run(localization: AdaptiveLocalization | None, localize: bool = True) -> Array:
        gmac = GMACUpdate(
            state_variables=VARIABLES, constraints=[box], localization=localization,
            localize_constraints=localize,
        )  # fmt: skip
        return _step(gmac)

    split = run(AdaptiveLocalization([N_STATE]), localize=False)
    both = run(AdaptiveLocalization([N_STATE]))
    plain = run(None)

    if value == 1.0:
        np.testing.assert_allclose(split, plain, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(both, plain, rtol=1e-12, atol=1e-14)
    else:
        np.testing.assert_array_equal(both, 0.0)
        assert np.abs(split).max() > 0  # the box's global part remains
        assert not np.allclose(split, plain)
