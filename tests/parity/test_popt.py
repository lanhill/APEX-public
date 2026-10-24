"""Parity of APEX's optimizers with PET's popt, whole optimizations at a time.

Each reference runs a PET optimizer to the end (``tools/pet_reference/generate.py``, cases
``popt_*``). PET's ensembles draw through a recording ``RandomState``; the tests replay those
draws through APEX's ``Draws`` protocol, so every iterate, every objective value and every
evaluation count must agree, through backtracking, resampling, SMC resampling and the EPF
loop. Line search and trust region are deterministic.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.optimization import (
    CMA,
    EnOpt,
    GaussianEnsemble,
    GeneralizedEnsemble,
    GenOpt,
    LineSearch,
    OptimizerBase,
    Problem,
    Quadratic,
    RateTable,
    Rosenbrock,
    SmcOpt,
    Steihaug,
    TrustRegion,
)
from apex.algorithms.optimization.protocols import FloatArray, IntArray

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]


def _f(value: NDArray[np.generic] | list[float]) -> FloatArray:
    return np.asarray(value, dtype=float)


# Every run below repeats PET's arithmetic operation for operation, so iterates agree to
# rounding. Rounding differs between machines (CPU, BLAS) and grows along a path: on CI,
# 40 BFGS steps on Rosenbrock reached 3e-10 absolute, and BFGS trust region 2e-5 by step
# 50 (the curved valley amplifies it chaotically). Ensemble paths are compared in full at
# rtol 1e-8 / atol 1e-9; the deterministic ones over their first DETERMINISTIC_PREFIX
# iterates, then by their converged result (FINAL_ATOL) and exact step and evaluation counts.
RTOL = 1e-8
ATOL = 1e-9
DETERMINISTIC_PREFIX = 25
FINAL_ATOL = 1e-6


class ReplayDraws:
    """APEX's draw source replaying the draws PET recorded, in order."""

    def __init__(self, ref: Mapping[str, NDArray[np.generic]]) -> None:
        keys = sorted(k for k in ref if k.startswith("in_draw_"))
        self.draws = [(k.rsplit("_", 1)[1], np.asarray(ref[k])) for k in keys]
        self.ref = ref
        self.position = 0

    def _check_argument(self, name: str, value: FloatArray) -> None:
        # The draws carry PET's mean, covariance and weights; APEX must ask for the same,
        # else a wrong covariance (shrink, Nesterov, adaptation) would pass unseen.
        key = f"in_drawarg_{self.position - 1:03d}_{name}"
        if key in self.ref:
            # 1e-10: the adapted covariance differed by 2e-12 relative between machines.
            np.testing.assert_allclose(value, _f(self.ref[key]), rtol=1e-10, atol=1e-14)

    def _next(self, kind: str) -> NDArray[Any]:
        assert self.position < len(self.draws), "APEX drew more often than PET"
        recorded_kind, value = self.draws[self.position]
        assert recorded_kind == kind, f"draw {self.position}: PET drew {recorded_kind}"
        self.position += 1
        return value

    def multivariate_normal(self, mean: FloatArray, cov: FloatArray, size: int) -> FloatArray:
        value = self._next("mvn")
        assert value.shape == (size, mean.size)
        self._check_argument("mean", mean)
        self._check_argument("cov", cov)
        return np.asarray(value, dtype=float)

    def choice(self, n: int, size: int, p: FloatArray) -> IntArray:
        value = self._next("choice")
        assert value.shape == (size,)
        self._check_argument("p", p)
        return np.asarray(value, dtype=np.int64)

    def export_state(self) -> dict[str, object]:
        return {"position": self.position}

    def restore_state(self, state: Mapping[str, object]) -> None:
        self.position = int(state["position"])  # type: ignore[call-overload]


def _run(
    optimizer: OptimizerBase,
    ref: Mapping[str, NDArray[np.generic]],
    objective: Rosenbrock | Quadratic,
    *,
    bounded: bool = True,
    with_draws: bool = True,
    prefix: int | None = None,
) -> None:
    """Run ``optimizer`` on the reference's problem and compare the path with PET's."""
    x0 = np.asarray(ref["in_x0"], dtype=float)
    problem = Problem(
        objective,
        x0,
        lower=np.asarray(ref["in_lower"], dtype=float) if bounded else None,
        upper=np.asarray(ref["in_upper"], dtype=float) if bounded else None,
    )
    draws = ReplayDraws(ref) if with_draws else None
    rows: list[tuple[FloatArray, float, int, int]] = []

    def record(opt: OptimizerBase) -> None:
        # PET calls back after accepted steps only; APEX also after an EPF update
        # (iteration 0), which PET's path does not have.
        if opt.iteration > 0:
            rows.append((opt.xk.copy(), float(np.mean(opt.fk)), opt.nfev, opt.iteration))  # type: ignore[arg-type]

    cov = np.asarray(ref["in_cov"], dtype=float) if "in_cov" in ref else None
    result = optimizer.minimize(problem, draws, cov=cov, on_iteration=record)

    assert len(rows) == len(ref["out_xk"]), "a different number of accepted steps"
    path = slice(None, prefix)
    np.testing.assert_allclose(
        np.stack([r[0] for r in rows])[path], _f(ref["out_xk"])[path], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        _f([r[1] for r in rows])[path], _f(ref["out_fk"][:, 0])[path], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_array_equal([r[2] for r in rows], ref["out_nfev"])
    np.testing.assert_array_equal([r[3] for r in rows], ref["out_iteration"])
    final_atol = ATOL if prefix is None else FINAL_ATOL
    np.testing.assert_allclose(result.x, _f(ref["out_final_x"]), rtol=RTOL, atol=final_atol)
    assert result.nfev == int(ref["out_final_nfev"])
    assert result.njev == int(ref["out_final_njev"])
    assert result.nit == int(ref["out_final_iteration"])
    assert result.message == str(ref["out_message"])
    if draws is not None:
        assert draws.position == len(draws.draws), "APEX drew less often than PET"


def _use_pets_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the ensemble gradient subtract the last single-point evaluation, as PET does.

    After a rejected trial and a resample PET's baseline is the rejected trial's objective,
    not ``f(x_k)`` (docs/migration/pet_parity.md). The gradient barely notices (the
    perturbations are centred), but the ensemble Hessian's ``- C mean(dF)`` term does, and so
    does a Hessian-preconditioned direction. APEX's own baseline is unit-tested.
    """
    last: dict[str, FloatArray] = {}
    point = OptimizerBase.fun
    gradient = GaussianEnsemble.gradient

    def fun(self: OptimizerBase, x: FloatArray) -> FloatArray:
        last["values"] = point(self, x)
        return last["values"]

    def pet_gradient(
        self: GaussianEnsemble, x: FloatArray, cov: FloatArray, f_x: FloatArray
    ) -> FloatArray:
        return gradient(self, x, cov, last["values"])

    monkeypatch.setattr(OptimizerBase, "fun", fun)
    monkeypatch.setattr(GaussianEnsemble, "gradient", pet_gradient)


ENOPT_CASES: dict[str, tuple[Rosenbrock | Quadratic, dict[str, Any]]] = {
    "popt_enopt_gd": (Rosenbrock(), {"ne": 10, "maxiter": 15}),
    "popt_enopt_momentum": (
        Rosenbrock(),
        {
            "ne": 12,
            "maxiter": 15,
            "beta": 0.5,
            "nesterov": True,
            "resample": 2,
            "alpha": 0.2,
            "natural_gradient": False,
        },
    ),
    "popt_enopt_adam": (
        Rosenbrock(),
        {"ne": 10, "maxiter": 15, "optimizer": "Adam", "beta": 0.9, "alpha": 0.05},
    ),
    "popt_enopt_adamax": (
        Rosenbrock(),
        {"ne": 10, "maxiter": 15, "optimizer": "AdaMax", "beta": 0.9, "alpha": 0.05},
    ),
    "popt_enopt_hessian": (
        Quadratic(),
        {
            "ne": 60,
            "resample": 2,
            "maxiter": 10,
            "hessian": True,
            "ensemble_hessian": True,
            "alpha": 0.5,
            "alpha_cov": 0.01,
        },
    ),
    "popt_enopt_steihaug": (
        Quadratic(),
        {
            "ne": 60,
            "maxiter": 10,
            "optimizer": "Steihaug",
            "ensemble_hessian": True,
            "alpha_cov": 0.01,
        },
    ),
    "popt_enopt_transform": (
        Rosenbrock(),
        {"ne": 10, "maxiter": 12, "transform": True, "alpha_cov": 0.01, "ensemble_hessian": True},
    ),
    "popt_enopt_epf": (
        Quadratic(),
        {
            "ne": 12,
            "maxiter": 8,
            "alpha": 0.3,
            "epf": {"r": 1.0, "r_factor": 4.0, "conv_crit": 1e-3, "max_epf_iter": 4},
        },
    ),
}


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", sorted(ENOPT_CASES))
def test_enopt_matches_pet(
    pet_reference: ReferenceLoader, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if case == "popt_enopt_steihaug":
        # At negative curvature PET's CG-Steihaug measures the boundary step from the
        # residual r instead of the iterate p (a PET bug; APEX's own step is unit-tested in
        # tests/unit/algorithms/optimization). Reproduce PET's formula to compare the rest.
        def pet_step(self: Steihaug, p: FloatArray, r: FloatArray, d: FloatArray) -> FloatArray:
            return p + self._tau(r, d) * d

        monkeypatch.setattr(Steihaug, "_negative_curvature_step", pet_step)
    if case == "popt_enopt_hessian":
        _use_pets_baseline(monkeypatch)
    objective, options = ENOPT_CASES[case]
    _run(EnOpt(**options), pet_reference(case), objective)


@pytest.mark.pet_parity
def test_smcopt_matches_pet(pet_reference: ReferenceLoader) -> None:
    optimizer = SmcOpt(ne=20, maxiter=12, survival_factor=0.5, resample=1, alpha=0.3)
    _run(optimizer, pet_reference("popt_smcopt"), Rosenbrock())


@pytest.mark.pet_parity
def test_genopt_betamc_cma_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("popt_genopt_betamc_cma")
    optimizer = GenOpt(
        ne=16, maxiter=10, alpha=0.1, alpha_theta=0.1, cma=True, separate_mutation_ensemble=True
    )
    _run(optimizer, ref, Rosenbrock())
    np.testing.assert_allclose(optimizer.theta, _f(ref["out_theta"]), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(optimizer.corr, _f(ref["out_corr"]), rtol=RTOL, atol=ATOL)


@pytest.mark.pet_parity
def test_genopt_truncated_gaussian_matches_pet(pet_reference: ReferenceLoader) -> None:
    """With theta fixed: PET's theta gradient for this marginal is the one in the mean."""
    ref = pet_reference("popt_genopt_truncgauss")
    optimizer = GenOpt(
        ne=16,
        marginal="TruncGaussian",
        maxiter=10,
        alpha=0.1,
        alpha_theta=0.0,
        separate_mutation_ensemble=True,
    )
    _run(optimizer, ref, Rosenbrock())
    np.testing.assert_allclose(optimizer.theta, _f(ref["out_theta"]), rtol=RTOL, atol=ATOL)


DETERMINISTIC_CASES: dict[str, Callable[[], OptimizerBase]] = {
    "popt_linesearch_bfgs": lambda: LineSearch(method="BFGS", maxiter=60),
    "popt_linesearch_newton_cg": lambda: LineSearch(method="Newton-CG", maxiter=60),
    "popt_linesearch_gd_backtracking": lambda: LineSearch(
        method="GD", lsmethod=0, step_size_adapt=2, maxiter=40
    ),
    "popt_trust_region_iterative": lambda: TrustRegion(maxiter=60),
    "popt_trust_region_bfgs_steihaug": lambda: TrustRegion(
        maxiter=80, method="CG-Steihaug", quasi_newton=True
    ),
}


@pytest.mark.pet_parity
@pytest.mark.parametrize("case", sorted(DETERMINISTIC_CASES))
def test_deterministic_optimizers_match_pet(pet_reference: ReferenceLoader, case: str) -> None:
    _run(
        DETERMINISTIC_CASES[case](),
        pet_reference(case),
        Rosenbrock(),
        bounded=False,
        with_draws=False,
        prefix=DETERMINISTIC_PREFIX,
    )


@pytest.mark.pet_parity
@pytest.mark.parametrize("marginal", ["BetaMC", "Beta", "Logistic", "TruncGaussian", "Gaussian"])
def test_generalized_gradients_match_pet(pet_reference: ReferenceLoader, marginal: str) -> None:
    ref = pet_reference("popt_generalized_gradients")
    x0 = np.asarray(ref["in_x0"], dtype=float)
    bound = (0.0, 1.0) if marginal == "Beta" else (-2.0, 2.0)
    cov = 0.02 * np.eye(x0.size)
    calls: list[FloatArray] = []

    def evaluate(controls: FloatArray, realizations: IntArray) -> FloatArray:
        calls.append(controls)
        return Rosenbrock()(controls, realizations)

    replay = ReplayDraws({"in_draw_000_mvn": ref[f"in_{marginal}_z"]})
    ensemble = GeneralizedEnsemble(
        evaluate,
        x0,
        cov,
        25,
        replay,
        np.full(x0.size, bound[0]),
        np.full(x0.size, bound[1]),
        marginal=marginal,
    )
    # Default theta: BetaMC's concentration from NumPy's cubic, not PET's sympy solve.
    np.testing.assert_allclose(ensemble.theta, _f(ref[f"in_{marginal}_theta"]), rtol=1e-12)
    f_x = Rosenbrock()(x0[:, None], np.zeros(1, dtype=np.int64))
    sample = ensemble.draw_and_evaluate(x0, ensemble.theta, ensemble.corr)
    # Quantile functions of scipy.stats agree to rounding (same SciPy as PET's environment).
    np.testing.assert_allclose(sample["X"], _f(ref[f"out_{marginal}_members"]), rtol=1e-12)
    np.testing.assert_allclose(sample["F"], _f(ref[f"out_{marginal}_values"]), rtol=1e-12)
    gradient, hessian = ensemble.gradient(x0, ensemble.theta, ensemble.corr, sample, f_x)
    # The Hessian sums N rank-one terms in another order than PET's (N, d, d) stack.
    np.testing.assert_allclose(gradient, _f(ref[f"out_{marginal}_gradient"]), rtol=1e-10)
    np.testing.assert_allclose(hessian, _f(ref[f"out_{marginal}_hessian"]), rtol=1e-10, atol=1e-10)
    if marginal == "BetaMC":
        nat_grad, nat_hess = ensemble.mutation_gradient(x0, ensemble.theta, sample, f_x)
        np.testing.assert_allclose(nat_grad, _f(ref["out_BetaMC_mutation_gradient"]), rtol=1e-10)
        np.testing.assert_allclose(nat_hess, _f(ref["out_BetaMC_mutation_hessian"]), rtol=1e-10)


@pytest.mark.pet_parity
@pytest.mark.parametrize(
    ("label", "equal", "corr"), [("equal_cov", True, False), ("log_corr", False, True)]
)
def test_cma_matches_pet(
    pet_reference: ReferenceLoader, label: str, equal: bool, corr: bool
) -> None:
    ref = pet_reference("popt_cma")
    cma = CMA(12, 4, corr_update=corr, equal_weights=equal)
    cov = np.asarray(ref[f"in_{label}_cov"], dtype=float)
    for step in range(2):
        cov = cma(
            cov,
            np.asarray(ref[f"in_{label}_step_{step}"], dtype=float),
            np.asarray(ref[f"in_{label}_samples_{step}"], dtype=float),
            np.asarray(ref[f"in_{label}_values_{step}"], dtype=float),
        )
        np.testing.assert_allclose(cov, _f(ref[f"out_{label}_cov_{step}"]), rtol=1e-12, atol=1e-14)


@pytest.mark.pet_parity
def test_npv_matches_pet(pet_reference: ReferenceLoader) -> None:
    """PET's npv (fd17979, the case's in_npv_source) on the same field totals."""
    from apex.algorithms.optimization import NetPresentValue

    ref = pet_reference("popt_npv")
    wop, wgp, wwp, wwi, disc, scaling = (float(v) for v in _f(ref["in_prices"]))
    npv = NetPresentValue(
        start=str(ref["in_start"]),
        times=[str(t) for t in ref["in_times"]],
        wop=wop, wgp=wgp, wwp=wwp, wwi=wwi, disc=disc, obj_scaling=scaling,
    )  # fmt: skip
    # APEX also reads TIME at the first report, the days since the start (checked).
    days = np.datetime64(str(ref["in_times"][0])) - np.datetime64(str(ref["in_start"]))
    responses = _f(ref["in_responses"])
    responses = np.vstack([responses, np.full(responses.shape[1], days.astype(float))])
    # rtol: the same sums in another order (APEX discounts the steps as an array).
    np.testing.assert_allclose(npv.evaluate(responses), _f(ref["out_values"]), rtol=1e-13, atol=0)


def _days(ref: Mapping[str, NDArray[np.generic]]) -> float:
    """TIME at the first report: the days since the start."""
    gap = np.datetime64(str(ref["in_times"][0])) - np.datetime64(str(ref["in_start"]))
    return float(gap.astype(float))


# Any: the prices go in as keywords beside the objectives' text options (e.g. from_time).
def _prices(ref: Mapping[str, NDArray[np.generic]]) -> dict[str, Any]:
    return {
        str(k): float(v) for k, v in zip(ref["in_price_names"], _f(ref["in_prices"]), strict=True)
    }


@pytest.mark.pet_parity
def test_ren_npv_matches_pet(pet_reference: ReferenceLoader) -> None:
    """PET's ren_npv (fd17979): injectors inj1, inj2 at wwi and ren1 at wrenwi."""
    from apex.algorithms.optimization import RenewableNetPresentValue

    ref = pet_reference("popt_ren_npv")
    objective = RenewableNetPresentValue(
        start=str(ref["in_start"]), times=[str(t) for t in ref["in_times"]],
        injectors=["inj1", "inj2"], renewable_injectors=["ren1"], **_prices(ref),
    )  # fmt: skip
    n_times = len(ref["in_times"])
    totals = _f(ref["in_responses"]).reshape(6, n_times, -1)  # fopt, fgpt, fwpt, 3 wells
    field = totals[:3].reshape(3 * n_times, -1)
    time = np.full((1, field.shape[1]), _days(ref))
    responses = np.vstack([field, time, totals[3:].reshape(3 * n_times, -1)])
    # rtol: the same sums in another order.
    np.testing.assert_allclose(
        objective.evaluate(responses), _f(ref["out_values"]), rtol=1e-12, atol=0
    )


def _ecalc_model(folder: Path, start: str, end: str) -> Path:
    """A one-consumer eCalc model spanning the case (its emissions are stubbed here)."""
    (folder / "rates.csv").write_text(f"DATE,OIL_PROD,GAS_PROD,WATER_INJ\n{start},0,0,0\n")
    model = folder / "model.yaml"
    model.write_text(
        f"START: {start}\nEND: {end}\nTIME_SERIES:\n  - NAME: SIM\n    FILE: rates.csv\n"
        "    TYPE: DEFAULT\nFUEL_TYPES:\n  - NAME: fuel_gas\n    EMISSIONS:\n"
        "      - NAME: CO2\n        FACTOR: 2.0\nINSTALLATIONS:\n  - NAME: p\n"
        "    HCEXPORT: SIM;OIL_PROD\n    FUEL: fuel_gas\n    CATEGORY: FIXED\n"
        "    FUELCONSUMERS:\n      - NAME: t\n        CATEGORY: TURBINE-GENERATOR\n"
        "        ENERGY_USAGE_MODEL:\n          TYPE: DIRECT\n          FUELRATE: SIM;GAS_PROD\n"
    )
    return model


def _stubbed(cls: type[Any], ref: Mapping[str, NDArray[np.generic]]) -> type[Any]:
    """``cls`` with the generator's eCalc stand-in: CO2 (t/d) linear in the step rates."""
    stub = {str(k): float(v) for k, v in zip(ref["in_stub_names"], _f(ref["in_stub"]), strict=True)}

    class Stubbed(cls):  # type: ignore[misc]
        def _emissions(self, table: RateTable) -> NDArray[np.float64]:
            return np.asarray(
                sum(c * table.columns[k] for k, c in stub.items() if k in table.columns),
                dtype=np.float64,
            )

    return Stubbed


@pytest.mark.pet_parity
@pytest.mark.parametrize(
    ("case", "name"),
    [
        ("popt_ecalc_npv", "EcalcNetPresentValue"),
        ("popt_ecalc_pareto_npv_wemc", "EcalcParetoNetPresentValue"),
        ("popt_ecalc_pareto_npv_w", "EcalcParetoNetPresentValue"),
    ],
)
def test_ecalc_npv_matches_pet(
    pet_reference: ReferenceLoader, tmp_path: Path, case: str, name: str
) -> None:
    """PET's ecalc_npv and ecalc_pareto_npv (fd17979), eCalc stubbed alike on both sides
    (the generator's ECALC_STUB): the rates handed to eCalc and the economics around it.
    """
    pytest.importorskip("libecalc")
    import apex.algorithms.optimization as opt

    ref = pet_reference(case)
    times = [str(t) for t in ref["in_times"]]
    model = _ecalc_model(tmp_path, str(ref["in_start"]), times[-1])
    objective = _stubbed(getattr(opt, name), ref)(
        start=str(ref["in_start"]), times=times, ecalc_model_file=str(model), **_prices(ref)
    )
    responses = _f(ref["in_responses"])
    responses = np.vstack([responses, np.full((1, responses.shape[1]), _days(ref))])
    # rtol: the same sums in another order.
    np.testing.assert_allclose(
        objective.evaluate(responses), _f(ref["out_values"]), rtol=1e-12, atol=0
    )


@pytest.mark.pet_parity
def test_ren_npv_co2_matches_pet(pet_reference: ReferenceLoader, tmp_path: Path) -> None:
    """PET's ren_npv_co2 (fd17979) with eCalc stubbed alike (wind enters it); disc = 0 and
    a stub blind to dates and THP keep PET's two bugs and its THP timing out (see the
    module's Differences).
    """
    pytest.importorskip("libecalc")
    from apex.algorithms.optimization import RenewableCO2NetPresentValue

    ref = pet_reference("popt_ren_npv_co2")
    times = [str(t) for t in ref["in_times"]]
    np.save(tmp_path / "wind.npy", _f(ref["in_wind"]))
    model = _ecalc_model(tmp_path, str(ref["in_start"]), times[-1])
    objective = _stubbed(RenewableCO2NetPresentValue, ref)(
        start=str(ref["in_start"]), times=times, ecalc_model_file=str(model),
        injectors=["inj1", "inj2"], wind_power_file=str(tmp_path / "wind.npy"), **_prices(ref),
    )  # fmt: skip
    totals = _f(ref["in_responses"])
    wthp = _f(ref["in_wthp"]).reshape(-1, totals.shape[1])
    responses = np.vstack([totals, np.full((1, totals.shape[1]), _days(ref)), wthp])
    # rtol: the same sums in another order.
    np.testing.assert_allclose(
        objective.evaluate(responses), _f(ref["out_values"]), rtol=1e-12, atol=0
    )
