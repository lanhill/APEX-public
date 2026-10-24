"""GMAC against a NumPy transcription of the user's MATLAB ``RLM_MAC_GIES.m``.

The independent reference is ``tools/gmac_reference/generate.py``: a line-by-line
transcription of the script's per-member block (its explicit ``n_s x n_s`` sign matrices,
SVDs and energy cuts), with fixed histogram bins (``docs/migration/gmac.md``). Its stored
outputs are ``tests/data/reference/gmac/*.npz`` (also ``.mat`` for MATLAB).
"""

from pathlib import Path

import numpy as np
import pytest

from apex.algorithms.data_assimilation import DataErrorCovariance, GMACUpdate

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "gmac"


@pytest.mark.parametrize(
    "case",
    ["gmac_10100", "gmac_11000", "gmac_10001", "gmac_01100", "gmac_01100_truncated", "gmac_00010"],
)
def test_gmac_matches_the_matlab_block(case: str) -> None:
    ref = np.load(REFERENCE / f"{case}.npz")
    ensemble, sim_data = ref["in_ensemble"], ref["in_sim_data"]
    n = ensemble.shape[1] - 1
    nx, ny = (int(v) for v in ref["in_dim"])
    cut = float(ref["in_tsvd_cut"])
    gmac = GMACUpdate(
        code=str(ref["meta_code"]),
        energy=cut,
        metric_energy=cut,  # the script cuts M_R by the data's tsvdCut
        state_variables=[{"name": "facies", "size": nx * ny, "grid": (nx, ny, 1)}],
    )

    step = gmac.step(
        np.ascontiguousarray(ensemble[:, :n]),
        np.ascontiguousarray(sim_data[:, :n]),
        ref["in_perturbed_data"],
        DataErrorCovariance.diagonal(np.ones(sim_data.shape[0])),  # the inputs are whitened
        damping=float(ref["in_lambda"]),
        reference_prediction=sim_data[:, -1].copy(),
    )

    # rtol: the same operations up to eigh vs svd of M_R, the 1/sqrt(N-1) scalings (which
    # cancel) and the rank-one form of the sign matrices: measured 1.1e-15 to 4.7e-15 of the
    # largest step entry (rounding only).
    np.testing.assert_allclose(
        ensemble[:, :n] + step, ref["out_iterated_ensemble"], rtol=1e-10, atol=1e-10
    )
