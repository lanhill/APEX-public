"""Reference outputs for C-GIES: a NumPy transcription of the user's MATLAB.

``gies_constraint_block`` transcribes, line by line, the update of
``MATLAB/main/SyntheticStudies/FaciesStudy1/run_scripts/GIES_constraint_ensLoc.m``
(lines 200-315, without ``autoAdaLoc``), with the constraint functions and metrics of its
``setupCase_iES_constraint.m``:

- EQ: ``compare_hist`` (histogram counts minus the reference's), channel gradient
  ``1/(x + 0.1 sign(x) + 0.001)``, Hessian ``-diag((1/(x + 0.1 sign(x) + 0.001))^2)``;
- IN: ``h = [lb - m; m - ub]``, barrier gradient ``-1/(x + 0.1)``, Hessian
  ``diag(1/(x + 0.1)^2)``;
- weights 0.5 (EQ) and 0.5 (IN) by the trace rule.

It imports nothing from APEX; ``tests/unit/algorithms/test_da_cgies_matlab.py`` compares
``gmac`` (``curvature = "paper"``, ``form = "direct"``, code ``100``) against it. The
one departure, as for GMAC (``docs/migration/gmac.md``): ``hist(x, 50)`` bins each
sample over its own range in MATLAB; here every sample uses the same fixed bins.

Run: ``uv run python tools/gmac_reference/generate_cgies.py``.
"""

import math
from pathlib import Path

import numpy as np
import scipy.io
from numpy.typing import NDArray

Array = NDArray[np.float64]
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "gmac"


def hist_counts(values: Array, low: float, high: float, bins: int) -> Array:
    """``hist`` with fixed edges ``linspace(low, high, bins + 1)``, per column."""
    edges = np.linspace(low, high, bins + 1)
    return np.stack(
        [
            np.histogram(np.clip(values[:, i], low, high), bins=edges)[0]
            for i in range(values.shape[1])
        ],
        axis=-1,
    ).astype(np.float64)


def gies_constraint_block(
    ensemble: Array, sim_data: Array, perturbed_data: Array, lam: float,
    lower: float, upper: float, reference_counts: Array, hist_range: tuple[float, float],
    eq_weight: float, in_weight: float, use_eq: bool, use_in: bool,
    tapers: list[Array] | None = None,
) -> Array:  # fmt: skip
    """Return the updated members; ``ensemble`` and ``sim_data`` end with the mean model."""
    ne = ensemble.shape[1] - 1
    bins = reference_counts.size

    def eq_func(m: Array) -> Array:
        return hist_counts(m, *hist_range, bins) - reference_counts[:, None]

    def in_func(m: Array) -> Array:
        return np.concatenate([lower - m, m - upper])

    def eq_grad(x: Array) -> Array:
        return 1.0 / (x + 0.1 * np.sign(x) + 0.001)

    def eq_hess(x: Array) -> Array:
        return 0 - np.diag((1.0 / (x + 0.1 * np.sign(x) + 0.001)) ** 2)

    def in_grad(x: Array) -> Array:
        return -1.0 / (x + 0.1)

    def in_hess(x: Array) -> Array:
        return np.diag(1.0 / ((x + 0.1) ** 2))

    members = ensemble[:, :ne]
    eq_plus_mean = np.concatenate(
        [eq_func(members), eq_func(members.mean(axis=1, keepdims=True))], axis=1
    )
    in_plus_mean = np.concatenate(
        [in_func(members), in_func(members.mean(axis=1, keepdims=True))], axis=1
    )
    eq_gradient = np.stack([eq_grad(0 - eq_plus_mean[:, i]) for i in range(ne)], axis=1)
    in_gradient = np.stack([in_grad(0 - in_plus_mean[:, i]) for i in range(ne)], axis=1)

    mean_en = members.mean(axis=1)
    delta_m = (members - mean_en[:, None]) / math.sqrt(ne - 1)
    sim_mean = sim_data[:, -1]
    delta_d = (sim_data[:, :ne] - sim_mean[:, None]) / math.sqrt(ne - 1)
    innovation = delta_d.T @ (perturbed_data - sim_data[:, :ne])
    delta_eq = (eq_plus_mean[:, :ne] - eq_plus_mean[:, [-1]]) / math.sqrt(ne - 1)
    projected_eq = delta_eq.T @ eq_gradient
    delta_in = (in_plus_mean[:, :ne] - in_plus_mean[:, [-1]]) / math.sqrt(ne - 1)
    projected_in = delta_in.T @ in_gradient

    init = delta_d.T @ delta_d
    trace_init = np.trace(init)
    products = np.empty((ne, ne, ne))
    innovation_eq = np.zeros((ne, ne))
    innovation_in = np.zeros((ne, ne))
    for i in range(ne):
        products[i] = init
        if use_eq:
            eq_product = delta_eq.T @ eq_hess(0 - eq_plus_mean[:, i]) @ delta_eq
            weight = eq_weight * trace_init / np.trace(eq_product)
            products[i] = products[i] + weight * eq_product
            innovation_eq[:, i] = weight * projected_eq[:, i]
        if use_in:
            in_product = delta_in.T @ in_hess(0 - in_plus_mean[:, i]) @ delta_in
            weight = in_weight * trace_init / np.trace(in_product)
            products[i] = products[i] + weight * in_product
            innovation_in[:, i] = weight * projected_in[:, i]

    coeff_times = [np.zeros((ne, ne)) for _ in range(3)]
    for i in range(ne):
        alpha = lam * np.trace(products[i]) / ne
        coeff = np.linalg.inv(products[i] + alpha * np.eye(ne))
        coeff_times[0][:, i] = coeff @ innovation[:, i]
        coeff_times[1][:, i] = coeff @ innovation_eq[:, i]
        coeff_times[2][:, i] = coeff @ innovation_in[:, i]
    # effective_deltaM(_EQ, _IN) = tapering_matrix(_EQ, _IN) .* deltaM with autoAdaLoc
    # (lines 279-302), given here instead of drawn (autoAdaLoc's shuffle is random).
    sources = [0] + ([1] if use_eq else []) + ([2] if use_in else [])
    model_change = np.zeros_like(members)
    for k, source in enumerate(sources):
        effective = delta_m if tapers is None else tapers[k] * delta_m
        model_change += effective @ coeff_times[source]
    return members + model_change


def case(use_eq: bool, use_in: bool, seed: int, localized: bool = False) -> dict[str, Array]:
    """One random case: 30 cells, 12 members, 40 data, values in [100, 15000]."""
    rng = np.random.default_rng(seed)
    nm, ne, nd = 30, 12, 40
    members = rng.uniform(600.0, 9000.0, (nm, ne))
    ensemble = np.concatenate([members, members.mean(axis=1, keepdims=True)], axis=1)
    forward = rng.standard_normal((nd, nm)) / (1000.0 * math.sqrt(nm))
    sim_data = np.tanh(forward @ ensemble) * 3.0  # whitened, the mean model's last
    perturbed = sim_data[:, :ne] + 0.4 + rng.standard_normal((nd, ne))
    truth = rng.choice([500.0, 10000.0], nm).astype(np.float64)
    hist_range, bins = (0.0, 15000.0), 10
    reference = hist_counts(truth[:, None], *hist_range, bins)[:, 0]
    lam = 0.8
    n_sources = 1 + use_eq + use_in
    tapers = list(rng.uniform(0.0, 1.0, (n_sources, nm, ne))) if localized else None
    posterior = gies_constraint_block(
        ensemble, sim_data, perturbed, lam, 100.0, 15000.0, reference, hist_range, 0.5, 0.5,
        use_eq, use_in, tapers,
    )  # fmt: skip
    return {
        "in_ensemble": ensemble, "in_sim_data": sim_data, "in_perturbed_data": perturbed,
        "in_lambda": np.array(lam), "in_bounds": np.array([100.0, 15000.0]),
        "in_truth": truth, "in_hist_range": np.array(hist_range), "in_bins": np.array(bins),
        "in_weights": np.array([0.5, 0.5]), "in_use": np.array([use_eq, use_in]),
        "in_tapers": np.zeros((0, nm, ne)) if tapers is None else np.stack(tapers),
        "out_iterated_ensemble": posterior, "meta_seed": np.array(seed),
    }  # fmt: skip


CASES = {
    "cgies_in": (False, True, 201),
    "cgies_eq": (True, False, 202),
    "cgies_in_eq": (True, True, 203),
    "cgies_in_eq_localized": (True, True, 204),
}


def main() -> None:
    """Write every case as ``.npz`` and ``.mat``."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (use_eq, use_in, seed) in CASES.items():
        arrays = case(use_eq, use_in, seed, localized=name.endswith("_localized"))
        np.savez(OUT / f"{name}.npz", **arrays)  # type: ignore[arg-type]
        scipy.io.savemat(OUT / f"{name}.mat", arrays)
        print(name, arrays["out_iterated_ensemble"].shape)


if __name__ == "__main__":
    main()
