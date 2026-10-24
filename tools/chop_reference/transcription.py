"""A line-by-line NumPy transcription of the author's CHOP MATLAB code.

Independent of APEX (it imports nothing from it): the reference that APEX's CHOP kernels
are tested against when MATLAB cannot run (as ``tools/gmac_reference`` for GMAC).
Transcribed from ``5SpotExample/run_scripts`` (the user's MATLAB tree):

- ``getCorrMtx.m``: sample standard deviations (divisor ``N - 1``), the product divided
  by ``N`` (so ``(N - 1) / N`` times the sample correlation); zero-spread rows and
  columns (std < 1e-6) get correlation 0;
- ``autoAdaLoc/rationalFunction.m``: the Gaspari-Cohn function of ``|dist| ./ lc``
  (``lc`` a row: column ``s`` uses ``lc(s)``);
- ``RLM_MAC_CHOP.m`` lines 177-271: one trial's model update with each member's taper on
  the shared RLM-MAC gain;
- ``update_hyper.m`` and ``update_hyper_no_loc.m``: the hyper-parameter update.

Layout as MATLAB's: ensembles carry a mean column last; data are already whitened.
"""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def get_corr_mtx(ensemble: Array, sim_data: Array) -> Array:
    """``getCorrMtx(ensemble, simData)``, members only (``(n_a, ne)``, ``(n_b, ne)``)."""
    ne = sim_data.shape[1]
    if ensemble.shape[1] != ne:
        raise ValueError("Sizes of ensemble and simData are not equal!")
    std_model = ensemble.std(axis=1, ddof=1)
    std_data = sim_data.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_ensemble = (ensemble - ensemble.mean(axis=1, keepdims=True)) / std_model[:, None]
        normalized_sim = (sim_data - sim_data.mean(axis=1, keepdims=True)) / std_data[:, None]
        corr: Array = normalized_ensemble @ normalized_sim.T / ne
    corr[std_model < 1e-6, :] = 0.0
    corr[:, std_data < 1e-6] = 0.0
    return corr


def rational_function(dist: Array, lc: Array) -> Array:
    """``rationalFunction(dist, lc)``: Gaspari & Cohn (1999) Eq. (4.10), ``z = |dist| ./ lc``."""
    z = np.abs(dist) / lc
    y = np.zeros_like(z)
    one = z <= 1
    two = (z <= 2) & ~one
    y[one] = 1 - z[one] ** 5 / 4 + z[one] ** 4 / 2 + 5 * z[one] ** 3 / 8 - 5 * z[one] ** 2 / 3
    y[two] = (
        z[two] ** 5 / 12 - z[two] ** 4 / 2 + 5 * z[two] ** 3 / 8 + 5 * z[two] ** 2 / 3
        - 5 * z[two] - 2 / (3 * z[two]) + 4
    )  # fmt: skip
    return y


def _tsvd(delta_d: Array, tsvd_data: float) -> tuple[Array, Array, Array, int]:
    """Return the SVD truncated by MATLAB's rule: the first ``j`` with ``s(1:j) / sum > tsvd``."""
    ud, s, vdt = np.linalg.svd(delta_d, full_matrices=False)
    total = s.sum()
    rank = len(s)
    for j in range(1, len(s) + 1):
        rank = j
        if s[:j].sum() / total > tsvd_data:
            break
    return ud[:, :rank], s[:rank], vdt[:rank].T, rank


def _x1(vd: Array, wd: Array, lam: float) -> tuple[Array, float]:
    alpha = lam * float(np.sum(wd**2)) / len(wd)
    alpha_raw = alpha
    alpha = max(alpha, 1e-3)
    return vd * (wd / (alpha + wd**2))[None, :], alpha_raw


def chop_model_update(
    ensemble: Array, sim_data: Array, perturbed: Array, lam: float, tsvd_data: float, nm: int
) -> dict[str, Array]:
    """``RLM_MAC_CHOP.m`` lines 177-271; return the updated members and intermediates."""
    ne = ensemble.shape[1] - 1
    delta_m = ensemble[:, :ne] - ensemble.mean(axis=1, keepdims=True)
    delta_d = sim_data[:, :ne] - sim_data[:, -1:]
    corr = get_corr_mtx(ensemble[:nm, :ne], sim_data[:, :ne])
    ud, wd, vd, rank = _tsvd(delta_d, tsvd_data)
    x1, alpha_raw = _x1(vd, wd, lam)
    x = perturbed - sim_data[:, :ne]
    ens_bg, ens_hyper = ensemble[:nm].copy(), ensemble[nm:]
    coeff = delta_m[:nm] @ x1 @ ud.T
    change = np.zeros((nm, ne))
    factor = 2
    for i in range(ne):
        lc = np.abs((ens_hyper[:, i] + 1e-6) * factor)
        taper = rational_function(1 - np.abs(corr), lc[None, :])
        change[:, i] = (taper * coeff) @ x[:, i]
    ens_bg[:, :ne] += change
    return {"model": ens_bg[:, :ne], "corr": corr, "rank": np.array(float(rank)),
            "alpha": np.array(alpha_raw)}  # fmt: skip


def update_hyper(
    ens_hyper: Array, sim_data: Array, perturbed: Array, lam: float, tsvd_data: float,
    *, localized: bool,
) -> Array:  # fmt: skip
    """``update_hyper.m`` (``localized``) or ``update_hyper_no_loc.m``; members, floored."""
    ne = ens_hyper.shape[1] - 1
    delta_m = ens_hyper[:, :ne] - ens_hyper[:, -1:]
    delta_d = sim_data[:, :ne] - sim_data[:, -1:]
    ud, wd, vd, _ = _tsvd(delta_d, tsvd_data)
    x1, _ = _x1(vd, wd, lam)
    x = perturbed - sim_data[:, :ne]
    coeff = delta_m @ x1 @ ud.T
    updated = ens_hyper.copy()
    if localized:
        corr = get_corr_mtx(ens_hyper[:, :ne], sim_data[:, :ne])
        change = np.zeros((ens_hyper.shape[0], ne))
        for i in range(ne):
            lc = np.abs(ens_hyper[:, i] + 1e-6) * 2
            taper = rational_function(1 - np.abs(corr), lc[None, :])
            change[:, i] = (taper * coeff) @ x[:, i]
        updated[:, :ne] += change
    else:
        updated[:, :ne] += coeff @ x
    updated[:, ne] = updated[:, :ne].mean(axis=1)
    updated[updated < 1e-6] = 1e-6
    members: Array = updated[:, :ne]
    return members
