"""Reference outputs for GMAC: a NumPy transcription of the user's MATLAB.

``rlm_mac_gies_block`` transcribes, line by line, the per-member update block of
``MATLAB/main/SyntheticStudies/FaciesStudy1/run_scripts/RLM_MAC_GIES.m`` (the loop
``for i = 1 : ne`` inside ``while iterLambda < kalmanOptions.maxInnerIter``), and
``get_tv_2d`` its ``getTV2D.m``. It is not APEX code and imports nothing from APEX: it is
the independent reference that ``tests/unit/algorithms/test_da_gmac_matlab.py`` compares
``gmac`` against. MATLAB is not licensed on the generating host (licensing error -39,147,
2026-10-05), so every case is also written as ``.mat`` to rerun the block in MATLAB
elsewhere (``in_*`` are the block's inputs, in the script's names below).

Two departures, both recorded in ``docs/migration/gmac.md``:

- ``getHistEntropy.m`` bins each sample by MATLAB's ``histogram(x, 100)`` (its own,
  "nice" edges per sample); the transcription bins every sample on the same fixed edges
  (``get_hist_entropy``), as APEX does, so that differences compare like with like;
- the inputs are already whitened (``simData``, ``perturbedData`` and the mean model's
  forecast normalised by the data errors' standard deviations, as the script's
  ``normalizeData``).

Run: ``uv run python tools/gmac_reference/generate.py`` (NumPy and SciPy only).
"""

import math
from pathlib import Path

import numpy as np
import scipy.io
from numpy.typing import NDArray

Array = NDArray[np.float64]
OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "reference" / "gmac"

#: The script's ``kalmanOptions`` weights, by the paper's 5-bit code (equal over the bits).
WEIGHTS = ("NLS_coeff_weight", "TV_coeff_weight_l22", "TV_coeff_weight_l12",
           "Hist_coeff_weight_l22", "Hist_coeff_weight_l12")  # fmt: skip


def get_tv_2d(input_vec: Array, dim: tuple[int, int]) -> Array:
    """``getTV2D.m``: ``[dx; dy]`` of each column reshaped column-major to ``dim``."""
    n_sample = input_vec.shape[1]
    n_row, n_col = dim
    mtx = input_vec.reshape((n_row, n_col, n_sample), order="F")
    dx = mtx[:, 1:, :] - mtx[:, :-1, :]
    dy = mtx[1:, :, :] - mtx[:-1, :, :]
    return np.concatenate(
        [dx.reshape((n_row * (n_col - 1), n_sample), order="F"),
         dy.reshape(((n_row - 1) * n_col, n_sample), order="F")]
    )  # fmt: skip


def get_hist_entropy(input_vec: Array, low: float, high: float, n_bin: int = 100) -> Array:
    """``getHistEntropy.m`` on fixed edges ``linspace(low, high, n_bin + 1)`` (see above)."""
    n_data, n_sample = input_vec.shape
    edges = np.linspace(low, high, n_bin + 1)
    out = np.zeros((n_bin, n_sample))
    for i in range(n_sample):
        values, _ = np.histogram(np.clip(input_vec[:, i], low, high), bins=edges)
        active = np.flatnonzero(values)
        out[active, i] = values[active] * (math.log(n_data) - np.log(values[active]))
    return out


def _cut(val: Array, tsvd_cut: float) -> int:
    """Return the rank of the script's ``if (sum(val(1:j))/total > tsvdCut) break`` loop."""
    total = float(np.sum(val))
    svd_pd = 0
    for j in range(1, val.size + 1):
        svd_pd = j
        if float(np.sum(val[:j])) / total > tsvd_cut:
            break
    return svd_pd


def rlm_mac_gies_block(
    ensemble: Array, sim_data: Array, perturbed_data: Array, weights: Array,
    lam: float, tsvd_cut: float, dim: tuple[int, int], hist_range: tuple[float, float],
) -> Array:  # fmt: skip
    """Return the updated members; ``ensemble`` and ``sim_data`` end with the mean model."""
    ne = ensemble.shape[1] - 1
    mean_en = ensemble[:, :ne].mean(axis=1)
    delta_m = ensemble[:, :ne] - mean_en[:, None]
    hist_delta_m = (get_hist_entropy(ensemble[:, :ne], *hist_range)
                    - get_hist_entropy(mean_en[:, None], *hist_range))  # fmt: skip
    tv_delta_m = get_tv_2d(ensemble[:, :ne], dim) - get_tv_2d(mean_en[:, None], dim)
    sim_mean = sim_data[:, -1]
    delta_d = sim_data[:, :ne] - sim_mean[:, None]

    m_r_tvl2 = tv_delta_m.T @ tv_delta_m
    m_r_histl2 = hist_delta_m.T @ hist_delta_m
    iterated = np.zeros((ensemble.shape[0], ne))
    for i in range(ne):
        m_r_tvl1 = tv_delta_m.T @ np.sign(np.outer(tv_delta_m[:, i], tv_delta_m[:, i])) @ tv_delta_m
        m_r_histl1 = (hist_delta_m.T @ np.sign(np.outer(hist_delta_m[:, i], hist_delta_m[:, i]))
                      @ hist_delta_m)  # fmt: skip
        terms = (np.eye(ne), m_r_tvl2, m_r_tvl1, m_r_histl2, m_r_histl1)
        # the script divides by every trace (0/0 is NaN there): only the weighted terms here
        m_r = weights[0] * terms[0]
        for w, m in zip(weights[1:], terms[1:], strict=True):
            if w != 0:
                m_r = m_r + w * m * ne / np.trace(m)
        um, wm, _ = np.linalg.svd(m_r)
        svd_pd = _cut(wm, tsvd_cut)
        sqrt_inv_m = um[:, :svd_pd] * (1.0 / wm[:svd_pd] ** 0.5)
        new_delta_d = delta_d @ sqrt_inv_m
        ud, wd, vdt = np.linalg.svd(new_delta_d, full_matrices=False)
        svd_pd = _cut(wd, tsvd_cut)
        new_delta_m = delta_m @ sqrt_inv_m
        vd, ud, wd = vdt.T[:, :svd_pd], ud[:, :svd_pd], wd[:svd_pd]
        alpha = lam * float(np.sum(wd**2)) / svd_pd
        x1 = vd * (wd / (wd**2 + alpha))
        k_gain = new_delta_m @ x1 @ ud.T
        iterated[:, i] = ensemble[:, i] - k_gain @ (sim_data[:, i] - perturbed_data[:, i])
    return iterated


def case(code: str, tsvd_cut: float, seed: int) -> dict[str, Array]:
    """One random case on a 6 x 5 grid, 12 members, 40 data."""
    rng = np.random.default_rng(seed)
    dim, ne, nd = (6, 5), 12, 40
    nm = dim[0] * dim[1]
    members = rng.standard_normal((nm, ne))
    forward = rng.standard_normal((nd, nm)) / math.sqrt(nm)
    ensemble = np.concatenate([members, members.mean(axis=1, keepdims=True)], axis=1)
    sim_data = np.tanh(forward @ ensemble) * 3.0  # whitened, with the mean model's last
    perturbed = sim_data[:, :ne] + 0.4 + rng.standard_normal((nd, ne))
    weights = np.array([float(b) for b in code]) / code.count("1")
    hist_range = (float(members.min()), float(members.max()))
    lam = 0.8
    posterior = rlm_mac_gies_block(ensemble, sim_data, perturbed, weights, lam, tsvd_cut,
                                   dim, hist_range)  # fmt: skip
    return {
        "in_ensemble": ensemble, "in_sim_data": sim_data, "in_perturbed_data": perturbed,
        "in_weights": weights, "in_lambda": np.array(lam), "in_tsvd_cut": np.array(tsvd_cut),
        "in_dim": np.array(dim), "in_hist_range": np.array(hist_range),
        "out_iterated_ensemble": posterior, "meta_seed": np.array(seed),
        "meta_code": np.array(code),
    }  # fmt: skip


CASES = {
    "gmac_10100": ("10100", 1 - 1e-12, 101),
    "gmac_11000": ("11000", 1 - 1e-12, 102),
    "gmac_10001": ("10001", 1 - 1e-12, 103),
    "gmac_01100": ("01100", 1 - 1e-12, 104),
    "gmac_01100_truncated": ("01100", 0.9, 105),
    "gmac_00010": ("00010", 0.9, 106),
}


def main() -> None:
    """Write every case as ``.npz`` and ``.mat``."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (code, cut, seed) in CASES.items():
        arrays = case(code, cut, seed)
        np.savez(OUT / f"{name}.npz", **arrays)  # type: ignore[arg-type]
        scipy.io.savemat(OUT / f"{name}.mat", arrays)
        print(name, arrays["out_iterated_ensemble"].shape)


if __name__ == "__main__":
    main()
