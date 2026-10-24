"""``rlmmac`` against a NumPy transcription of the author's MATLAB RLM-MAC.

The independent reference is ``tools/rlmmac_reference/generate.py``: a line-by-line
transcription of the analysis of ``RLM_MAC.m`` (anomalies scaled by ``1/sqrt(ne-1)``) and of
the older ``RLM_average_cost.m`` (unscaled), stored in ``tests/data/reference/rlmmac/``
(also ``.mat`` for MATLAB). Reference: Luo, Stordal, Lorentzen & Naevdal (2015), SPE
Journal 20(5), 962-982.

Both MATLAB versions give the same step, and it is ``rlmmac``'s default (scaled data
anomalies) at the same ``lambda``: MATLAB's ``alpha = lambda * mean(Wd^2)`` is APEX's
``gamma``. APEX at ``lambda = N - 1`` is another, more strongly damped step, and the
unscaled ``rlmmac`` (PET-fork's) is the MATLAB step times ``1/sqrt(N - 1)``.
"""

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import DataErrorCovariance, RLMMACUpdate
from apex.algorithms.data_assimilation.gies import UNSCALED_DATA_ANOMALIES_WARNING

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "rlmmac"
CASES = ["diag_lambda1", "diag_lambda037_truncated", "dense_lambda1", "diag_lambda_large"]


def _apex_step(
    ref: dict[str, NDArray[np.float64]], damping: float, *, scaled: bool = True
) -> NDArray[np.float64]:
    ensemble, sim_data, c = ref["in_ensemble"], ref["in_sim_data"], ref["in_c"]
    n = ensemble.shape[1] - 1
    error = DataErrorCovariance.dense(c) if c.ndim == 2 else DataErrorCovariance.diagonal(c)
    update = RLMMACUpdate(energy=float(ref["in_tsvd_data"]), scale_data_anomalies=scaled)
    return update.step(
        np.ascontiguousarray(ensemble[:, :n]),
        np.ascontiguousarray(sim_data[:, :n]),
        ref["in_perturbed_data"],
        error,
        damping=damping,
        reference_prediction=sim_data[:, -1],
    )


def _load(case: str) -> dict[str, NDArray[np.float64]]:
    with np.load(REFERENCE / f"{case}.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("version", ["rlm_mac", "rlm_average_cost"])
def test_rlmmac_matches_the_matlab_analysis(case: str, version: str) -> None:
    ref = _load(case)
    n = ref["in_ensemble"].shape[1] - 1
    expected = ref[f"out_{version}_ensemble"] - ref["in_ensemble"][:, :n]

    step = _apex_step(ref, float(ref["in_lambda"]))

    # The same SVD, cut and weights up to rounding (MATLAB's loops against matrix products).
    np.testing.assert_allclose(step, expected, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case", CASES)
def test_lambda_n_minus_1_is_not_the_matlab_step_at_lambda_1(case: str) -> None:
    """The user's question of 2026-10-13: APEX at ``N - 1`` against MATLAB at ``lambda``."""
    ref = _load(case)
    n = ref["in_ensemble"].shape[1] - 1
    expected = ref["out_rlm_mac_ensemble"] - ref["in_ensemble"][:, :n]

    step = _apex_step(ref, float(ref["in_lambda"]) * (n - 1))

    assert np.linalg.norm(step) < 0.5 * np.linalg.norm(expected)


@pytest.mark.parametrize("case", CASES)
def test_unscaled_rlmmac_is_the_matlab_step_over_sqrt_n_minus_1(
    case: str, caplog: pytest.LogCaptureFixture
) -> None:
    ref = _load(case)
    n = ref["in_ensemble"].shape[1] - 1
    expected = ref["out_rlm_mac_ensemble"] - ref["in_ensemble"][:, :n]

    with caplog.at_level(logging.WARNING):
        step = _apex_step(ref, float(ref["in_lambda"]), scaled=False)

    np.testing.assert_allclose(step, expected / math.sqrt(n - 1), rtol=1e-9, atol=1e-11)
    assert UNSCALED_DATA_ANOMALIES_WARNING in caplog.messages
