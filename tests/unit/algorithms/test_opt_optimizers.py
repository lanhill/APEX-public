"""The optimizers' loop: resume, refusals, robust optimization and convergence."""

from collections.abc import Callable

import numpy as np
import pytest

from apex.algorithms.optimization import (
    EnOpt,
    GaussianEnsemble,
    GeneratorDraws,
    GenOpt,
    LineSearch,
    OptimizerBase,
    Problem,
    Quadratic,
    Rosenbrock,
    SmcOpt,
    TrustRegion,
)
from apex.algorithms.optimization.protocols import FloatArray, IntArray


def _problem(objective: object = None, n: int = 3, n_realizations: int = 1) -> Problem:
    return Problem(
        objective or Rosenbrock(),  # type: ignore[arg-type]
        np.full(n, 0.5),
        lower=np.full(n, -2.0),
        upper=np.full(n, 2.0),
        n_realizations=n_realizations,
    )


FACTORIES: dict[str, Callable[[], OptimizerBase]] = {
    "enopt": lambda: EnOpt(ne=10, maxiter=12, resample=1, ensemble_hessian=True, alpha_cov=0.01),
    "enopt_adam_transform": lambda: EnOpt(ne=10, maxiter=12, optimizer="Adam", transform=True),
    "smcopt": lambda: SmcOpt(ne=12, maxiter=10, survival_factor=0.5, resample=1),
    "genopt": lambda: GenOpt(ne=12, maxiter=10, cma=True),
    "line_search": lambda: LineSearch(method="BFGS", maxiter=30),
    "trust_region": lambda: TrustRegion(maxiter=30, quasi_newton=True),
    "enopt_epf": lambda: EnOpt(
        ne=10, maxiter=4, epf={"r": 1.0, "conv_crit": 1e-6, "max_epf_iter": 3}
    ),
}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_resume_is_bit_identical(name: str) -> None:
    """Stopping after every accepted step and resuming gives the uninterrupted run."""
    objective = Quadratic() if name == "enopt_epf" else Rosenbrock()
    cov = 0.01 * np.eye(3)
    full = FACTORIES[name]().minimize(
        _problem(objective), GeneratorDraws(np.random.default_rng(7)), cov=cov
    )
    state = None
    resumed = None
    for _ in range(200):
        optimizer = FACTORIES[name]()
        resumed = optimizer.minimize(
            _problem(objective),
            GeneratorDraws(np.random.default_rng(7)),
            cov=cov,
            on_iteration=lambda _: False,
            state=state,
        )
        if resumed.message != "stopped by the caller":
            break
        state = optimizer.export_state()
    assert resumed is not None
    np.testing.assert_array_equal(resumed.x, full.x)
    np.testing.assert_array_equal(resumed.fun, full.fun)
    assert (resumed.nfev, resumed.njev, resumed.member_evaluations, resumed.message) == (
        full.nfev,
        full.njev,
        full.member_evaluations,
        full.message,
    )


def test_same_seed_same_run_other_seed_other_run() -> None:
    def run(seed: int) -> FloatArray:
        result = EnOpt(ne=10, maxiter=8).minimize(
            _problem(), GeneratorDraws(np.random.default_rng(seed)), cov=0.01 * np.eye(3)
        )
        return result.x

    np.testing.assert_array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_the_gradient_baseline_is_the_current_iterate() -> None:
    """APEX subtracts f(x_k); PET the last point evaluated, a rejected trial after a resample."""
    seen: list[tuple[FloatArray, FloatArray]] = []
    original = GaussianEnsemble.gradient

    def spy(self: GaussianEnsemble, x: FloatArray, cov: FloatArray, f_x: FloatArray) -> FloatArray:
        seen.append((x.copy(), np.array(f_x)))
        return original(self, x, cov, f_x)

    optimizer = EnOpt(ne=6, maxiter=6, resample=3, alpha=1.5)
    GaussianEnsemble.gradient = spy  # type: ignore[method-assign]
    try:
        optimizer.minimize(
            _problem(), GeneratorDraws(np.random.default_rng(3)), cov=0.04 * np.eye(3)
        )
    finally:
        GaussianEnsemble.gradient = original  # type: ignore[method-assign]
    assert len(seen) > optimizer.iteration  # some iterations resampled
    for x, f_x in seen:
        np.testing.assert_allclose(f_x, Rosenbrock()(x[:, None], np.zeros(1, np.int64)))


def test_robust_optimization_evaluates_each_realization() -> None:
    """Members cycle through the realizations; a point is evaluated on all of them."""
    calls: list[IntArray] = []

    class Shifted:
        def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
            calls.append(realizations.copy())
            return np.asarray(0.5 * np.sum((controls - 1.0 - realizations) ** 2, axis=0))

    result = EnOpt(ne=8, maxiter=30, alpha=0.2).minimize(
        Problem(Shifted(), np.zeros(2), n_realizations=2),
        GeneratorDraws(np.random.default_rng(0)),
        cov=0.01 * np.eye(2),
    )
    np.testing.assert_array_equal(calls[0], [0, 1])  # the initial point
    np.testing.assert_array_equal(calls[1], [0, 0, 0, 0, 1, 1, 1, 1])  # the first ensemble
    assert result.fun.shape == (2,)
    # The mean over the two realizations is minimal at 1.5 in every control.
    np.testing.assert_allclose(result.x, 1.5, atol=0.15)


def test_optimizers_converge_on_the_quadratic() -> None:
    """Independent reference: the quadratic's minimum is x = 1, f = 0."""
    for optimizer in (
        LineSearch(method="BFGS"),
        LineSearch(method="Newton-CG"),
        TrustRegion(),
        TrustRegion(method="CG-Steihaug", quasi_newton=True),
    ):
        result = optimizer.minimize(Problem(Quadratic(), np.array([3.0, -2.0, 0.0])))
        np.testing.assert_allclose(result.x, 1.0, atol=1e-6)
    result = EnOpt(ne=20, maxiter=200, alpha=0.05).minimize(
        Problem(Quadratic(), np.array([3.0, -2.0, 0.0])),
        GeneratorDraws(np.random.default_rng(1)),
        cov=0.01 * np.eye(3),
    )
    assert float(result.fun[0]) < 0.01


def test_epf_drives_the_constraints() -> None:
    """EPF on the quadratic: sum(x) = n + 1 and x_0 <= 0 pull the optimum off x = 1."""
    result = EnOpt(
        ne=20, maxiter=10, alpha=0.05, epf={"r": 10.0, "r_factor": 4.0, "max_epf_iter": 5}
    ).minimize(
        Problem(Quadratic(), np.array([0.0, 1.0, 1.0])),
        GeneratorDraws(np.random.default_rng(2)),
        cov=0.01 * np.eye(3),
    )
    c_eq, c_iq = Quadratic().constraints(result.x[:, None])
    # The constrained optimum is (0, 2, 2); the penalty leaves a small violation.
    np.testing.assert_allclose(result.x, [0.0, 2.0, 2.0], atol=0.1)
    assert abs(float(c_eq[0, 0])) < 0.05
    assert float(c_iq[0, 0]) > -0.05
    assert result.epf_iteration > 1


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: EnOpt(optimizer="Steihaug"), "ensemble_hessian"),
        (lambda: EnOpt(hessian=True), "ensemble_hessian"),
        (lambda: EnOpt(optimizer="nesterov"), "unknown step_rules"),
        (lambda: GenOpt(optimizer="AdaMax"), "GD or Adam"),
        (lambda: EnOpt(epf={"r_factor": 2.0}), "penalty factor r"),
        (lambda: EnOpt(epf={"r": 1.0, "speed": 2.0}), "unknown EPF"),
    ],
)
def test_bad_options_are_refused(make: Callable[[], OptimizerBase], match: str) -> None:
    with pytest.raises(Exception, match=match):
        make()


def test_bad_problems_are_refused() -> None:
    draws = GeneratorDraws(np.random.default_rng(0))
    with pytest.raises(ValueError, match="covariance"):
        EnOpt().minimize(_problem(), draws)
    with pytest.raises(ValueError, match="cannot use EPF"):
        LineSearch(epf={"r": 1.0}).minimize(_problem(Quadratic()))
    with pytest.raises(ValueError, match="constraints"):
        EnOpt(epf={"r": 1.0}).minimize(_problem(), draws, cov=np.eye(3))

    class Plain:
        def __call__(self, controls: FloatArray, realizations: IntArray) -> FloatArray:
            return np.zeros(controls.shape[1])

    with pytest.raises(ValueError, match="gradient"):
        TrustRegion().minimize(_problem(Plain()))
    with pytest.raises(ValueError, match="multiple"):
        EnOpt(ne=5).minimize(_problem(n_realizations=2), draws, cov=np.eye(3))


def test_genopt_keeps_theta_positive() -> None:
    """PET's normalised theta step can set sigma to 0 (then every trial fails); APEX halves."""
    for seed in range(5):
        optimizer = GenOpt(ne=16, marginal="TruncGaussian", maxiter=20)
        result = optimizer.minimize(
            _problem(), GeneratorDraws(np.random.default_rng(seed)), cov=0.01 * np.eye(3)
        )
        assert np.all(optimizer.theta > 0)
        assert float(result.fun[0]) < 5.0  # f(x0) = 13; PET's run stalled at 5.5-12.9


def test_betamc_mode_is_not_pinned_on_wide_bounds() -> None:
    """PET's mode clip is 0.1 sqrt(var) in physical units: on [0, 1000] with variance 100 it
    pins the mode to a bound. APEX scales it by the width of the bounds."""
    from apex.algorithms.optimization.generalized import GeneralizedEnsemble

    def evaluate(controls: FloatArray, realizations: IntArray) -> FloatArray:
        return np.zeros(controls.shape[1])

    x0 = np.full(2, 500.0)
    ensemble = GeneralizedEnsemble(
        evaluate,
        x0,
        100.0 * np.eye(2),
        50,
        GeneratorDraws(np.random.default_rng(0)),
        np.zeros(2),
        np.full(2, 1000.0),
    )
    members, _ = ensemble.sample(x0, ensemble.theta, ensemble.corr)
    assert abs(float(np.mean(members)) - 500.0) < 5.0


def test_unsupported_combinations_are_refused() -> None:
    draws = GeneratorDraws(np.random.default_rng(0))
    with pytest.raises(ValueError, match="robust"):
        SmcOpt(ne=4).minimize(_problem(n_realizations=2), draws, cov=np.eye(3))
    with pytest.raises(ValueError, match="transform"):
        EnOpt(ensemble_hessian=True, hessian=True, transform=True)


def test_a_singular_ensemble_hessian_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GaussianEnsemble, "hessian", lambda self: np.zeros((3, 3)))
    with pytest.raises(FloatingPointError, match="singular"):
        EnOpt(ne=6, ensemble_hessian=True, hessian=True).minimize(
            _problem(), GeneratorDraws(np.random.default_rng(0)), cov=0.01 * np.eye(3)
        )


@pytest.mark.parametrize(
    "make",
    [
        lambda: LineSearch(method="GD", maxiter=15, lsmaxiter=4),
        lambda: LineSearch(method="BFGS", maxiter=15, lsmethod=0, lsmaxiter=3, recompute_jac=1),
        lambda: LineSearch(method="Newton-CG", maxiter=15),
        lambda: TrustRegion(maxiter=15),
        lambda: TrustRegion(maxiter=15, quasi_newton=True, trust_radius_cuts=2, trust_radius=5.0),
    ],
)
def test_line_search_and_trust_region_keep_to_their_evaluation_bound(
    make: Callable[[], OptimizerBase],
) -> None:
    """A reported issue (#211, #214): studies of line_search and trust_region were refused as
    unbounded. Independent check: the evaluations counted (nfev, each on every realization)
    never exceed the bound, on Rosenbrock from a far start."""
    optimizer = make()
    problem = Problem(Rosenbrock(), np.array([-1.5, 1.8, -1.2]), n_realizations=2)
    bound = optimizer.max_evaluations(2)
    optimizer.minimize(problem)
    assert bound is not None
    assert optimizer.nfev * 2 <= bound
