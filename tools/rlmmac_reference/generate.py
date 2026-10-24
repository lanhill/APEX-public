r"""Reference outputs for ``rlmmac``: a NumPy transcription of the author's MATLAB RLM-MAC.

``rlm_mac_update`` transcribes, line by line, the analysis of one inner-loop trial of the
method author's MATLAB RLM-MAC, in its two versions:

- ``scaled = True``: ``MATLAB/main/Kalman/RLM_MAC.m`` (``$Id: .../RLM_average_cost.m#18``,
  2017), lines 253-261 (``deltaD``, ``deltaM`` scaled by ``1/sqrt(ne-1)``, ``deltaD``
  about the mean model's forecast, ``simData(:, end)``), 361-375 (SVD and ``tsvdData``
  cut), 396-405 (``alpha``, ``x1``), 436 and 470 (``coeff``, the update);
- ``scaled = False``: the older ``MATLAB/main/Kalman/RLM_average_cost.m`` (2014-10-10),
  lines 124-147 (``deltaM``, ``deltaD`` unscaled), 212-225 (SVD and cut), 248-257
  (``alpha``, ``x1``), 261 and 270 (``coeff``, the update).

Both scripts are transcribed for one data normalisation, ``C``: the data and perturbed data
divided by the standard deviations for variances, or ``chol(C)' \ data`` for a matrix
(``RLM_MAC.m``'s ``normalizeData(data, sqrtW(Cbase))``, lines 131-132). That is the
scripts' own normalisation under these settings:

- ``scale`` (``scaling``) empty or ones, and ``Wbase`` ones. ``RLM_MAC.m`` divides
  ``simData`` alone by ``sqrtW(Wbase)`` too (line 93), and ``RLM_average_cost.m``
  divides ``simData`` alone by ``Wbase`` (line 81) and never normalises the perturbed
  data, so other values mean pre-normalised measurements;
- ``useDataNorm`` off, so that ``Cbase = C`` (``RLM_MAC.m`` lines 96-124).

The ``dense_*`` case extends ``RLM_average_cost.m``, whose ``Wbase`` is a vector of
standard deviations, to a correlated ``C``. It is not APEX
code and imports nothing from APEX: it is the independent reference that
``tests/unit/algorithms/test_da_rlmmac_matlab.py`` compares ``rlmmac`` against. MATLAB is
not licensed on the generating host (licensing error -39,147, again on 2026-10-13), so
every case is also written as ``.mat`` to rerun the lines in MATLAB elsewhere.

Not transcribed (outside the analysis, or options these cases leave off): localization,
``ignoreUninformativeMeasurements``, bounds (``respectBoundsWithReport``), the
perturbation of the data, and the inner loop's damping rules (``DampingSchedule`` with
``reduction`` and ``GIES.max_trials_per_iteration``; see docs/migration/pet_parity.md).

Run: ``uv run python tools/rlmmac_reference/generate.py`` (NumPy and SciPy only).
"""

import math
from pathlib import Path

import numpy as np
import scipy.io
from numpy.typing import NDArray

Array = NDArray[np.float64]
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "rlmmac"
SEED = 20261013


def sqrt_w(c: Array) -> Array:
    """``sqrtW.m``: ``sqrt`` of variances, or the lower Cholesky factor ``chol(C)'``."""
    return np.sqrt(c) if c.ndim == 1 else np.linalg.cholesky(c)


def normalize_data(data: Array, s_w: Array) -> Array:
    r"""``normalizeData.m``: ``data ./ sW`` row-wise, or ``sW \ data``."""
    return data / s_w[:, None] if s_w.ndim == 1 else np.linalg.solve(s_w, data)


def rlm_mac_update(
    ensemble: Array,
    sim_data: Array,
    perturbed_data: Array,
    c: Array,
    lam: float,
    tsvd_data: float,
    *,
    scaled: bool,
) -> tuple[Array, int, float]:
    """Return the trial ensemble ``(n_m, ne)``, ``svdPd`` and ``alpha``.

    ``ensemble`` and ``sim_data`` have ``ne + 1`` columns, the last the mean model and its
    forecast (``ensemble(:, ne+1)``, ``simData(:, end)``); ``perturbed_data`` has ``ne``.
    """
    ne = ensemble.shape[1] - 1
    s_w = sqrt_w(c)
    sim_n = normalize_data(sim_data, s_w)
    pert_n = normalize_data(perturbed_data, s_w)
    scale = 1.0 / math.sqrt(ne - 1) if scaled else 1.0
    sim_mean = sim_n[:, -1]
    delta_d = np.empty((sim_n.shape[0], ne))
    for j in range(ne):
        delta_d[:, j] = (sim_n[:, j] - sim_mean) * scale
    mean_en = ensemble[:, :ne].mean(axis=1)
    delta_m = np.empty((ensemble.shape[0], ne))
    for j in range(ne):
        delta_m[:, j] = (ensemble[:, j] - mean_en) * scale
    ud, val, vdt = np.linalg.svd(delta_d, full_matrices=False)
    vd = vdt.T
    total = val.sum()
    svd_pd = ne
    for j in range(1, ne + 1):  # MATLAB's j = 1:ne, breaking at the first cut above tsvdData
        svd_pd = j
        if val[:j].sum() / total > tsvd_data:
            break
    vd, ud, wd = vd[:, :svd_pd], ud[:, :svd_pd], val[:svd_pd]
    alpha = lam * float(np.sum(wd**2)) / svd_pd
    alpha = max(alpha, 1e-3)  # "in case of ensemble collapse"
    x1 = np.zeros((ne, svd_pd))
    for j_vec in range(svd_pd):
        for j in range(ne):
            x1[j, j_vec] = vd[j, j_vec] * wd[j_vec] / (alpha + wd[j_vec] ** 2)
    coeff = delta_m @ x1 @ ud.T
    trial = ensemble[:, :ne] - coeff @ (sim_n[:, :ne] - pert_n)
    return trial, svd_pd, alpha


def case(rng: np.random.Generator, n_m: int, nd: int, ne: int, *, dense: bool) -> dict[str, Array]:
    """Draw a mildly nonlinear problem: forecasts of the members and of their mean."""
    ensemble = rng.standard_normal((n_m, ne))
    ensemble = np.column_stack([ensemble, ensemble.mean(axis=1)])
    g = rng.standard_normal((nd, n_m)) / math.sqrt(n_m)
    sim_data = g @ ensemble + 0.1 * (g @ ensemble) ** 2
    std = 0.05 + 0.2 * rng.random(nd)
    if dense:
        rho = np.exp(-np.abs(np.subtract.outer(np.arange(nd), np.arange(nd))) / 3.0)
        c = rho * np.outer(std, std)
    else:
        c = std**2
    truth = rng.standard_normal(n_m)
    measurement = g @ truth + 0.1 * (g @ truth) ** 2
    noise = (np.linalg.cholesky(c) if dense else np.diag(std)) @ rng.standard_normal((nd, ne))
    return {
        "in_ensemble": ensemble, "in_sim_data": sim_data,
        "in_perturbed_data": measurement[:, None] + noise, "in_c": c,
    }  # fmt: skip


CASES = {
    # name: (n_m, nd, ne, dense, lambda, tsvdData)
    "diag_lambda1": (40, 60, 25, False, 1.0, 0.99),
    "diag_lambda037_truncated": (40, 60, 25, False, 0.37, 0.9),
    "dense_lambda1": (30, 45, 20, True, 1.0, 0.99),
    "diag_lambda_large": (40, 60, 25, False, 24.0, 0.99),
}


def main() -> None:
    """Write ``tests/data/reference/rlmmac/<case>.npz`` and ``.mat``."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name, (n_m, nd, ne, dense, lam, tsvd) in CASES.items():
        inputs = case(rng, n_m, nd, ne, dense=dense)
        out: dict[str, Array] = dict(inputs)
        out["in_lambda"], out["in_tsvd_data"] = np.float64(lam), np.float64(tsvd)
        for version, scaled in (("rlm_mac", True), ("rlm_average_cost", False)):
            trial, svd_pd, alpha = rlm_mac_update(
                inputs["in_ensemble"], inputs["in_sim_data"], inputs["in_perturbed_data"],
                inputs["in_c"], lam, tsvd, scaled=scaled,
            )  # fmt: skip
            out[f"out_{version}_ensemble"] = trial
            out[f"out_{version}_svd_pd"] = np.int64(svd_pd)
            out[f"out_{version}_alpha"] = np.float64(alpha)
        np.savez(OUT / f"{name}.npz", **out)
        scipy.io.savemat(OUT / f"{name}.mat", out)
        print(name, {k: v for k, v in out.items() if np.ndim(v) == 0})


if __name__ == "__main__":
    main()
