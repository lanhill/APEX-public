"""Model RMSE and SSIM against PET-fork's ``Plotting/metric_analyses.py`` (@ 1348861).

The reference runs PET-fork's ``calculate_rmse_and_model_change`` and
``calculate_ssim`` (the Brugge study's metrics) on two ensembles of a masked 10x8x9 grid.
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.metrics import ModelRMSE, ModelSSIM
from apex.core.metrics import MetricContext
from apex.domain.observations import DataKey, ObservationSet

# tests/ is not an importable package, so spell out the fixture's type here.
type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

NAMES = ("permx", "poro")


def _context(ref: dict[str, NDArray[np.generic]], k: int) -> MetricContext:
    state = {n: np.asarray(ref[f"in_ens{k}_{n}"], dtype=np.float64) for n in NAMES}
    n = state["permx"].shape[-1]
    return MetricContext(
        state=state,
        predicted=np.zeros((1, n)),
        observations=ObservationSet([DataKey("t", "d")], [0.0], [1.0]),
        truth={n: np.asarray(ref[f"in_truth_{n}"], dtype=np.float64) for n in NAMES},
    )


@pytest.mark.pet_parity
def test_model_rmse_matches_pet_fork(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("metrics_rmse_ssim")
    metric = ModelRMSE(actnum=np.asarray(ref["in_actnum"]))
    for k in (0, 1):
        result = metric.evaluate(_context(ref, k))
        for label in (*NAMES, "all"):
            # rtol: the same sum of squares, reduced in a different order (norm vs mean).
            np.testing.assert_allclose(
                result[label], np.asarray(ref[f"out_rmse_{label}"], float)[k], rtol=1e-13
            )


@pytest.mark.pet_parity
def test_model_ssim_matches_pet_fork(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("metrics_rmse_ssim")
    metric = ModelSSIM(
        grid=np.asarray(ref["in_grid"]).tolist(), actnum=np.asarray(ref["in_actnum"])
    )
    for k in (0, 1):
        result = metric.evaluate(_context(ref, k))
        for label in NAMES:
            # rtol: native uniform-filter SSIM vs scikit-image's; measured ~1e-16.
            np.testing.assert_allclose(
                result[label], np.asarray(ref[f"out_ssim_{label}"], float)[k], rtol=1e-12
            )


@pytest.mark.pet_parity
def test_model_change_matches_pet_fork(pet_reference: ReferenceLoader) -> None:
    """Steps 1 and 2, each against the ensemble before it; the prior has none."""
    from apex.algorithms.metrics.performance import ModelChange

    ref = pet_reference("metrics_model_change")
    metric = ModelChange(actnum=np.asarray(ref["in_actnum"]))
    ensembles = [
        {n: np.asarray(ref[f"in_ens{k}_{n}"], dtype=np.float64) for n in NAMES} for k in range(3)
    ]

    def context(k: int) -> MetricContext:
        n = ensembles[k]["permx"].shape[-1]
        return MetricContext(
            state=ensembles[k],
            predicted=np.zeros((1, n)),
            observations=ObservationSet([DataKey("t", "d")], [0.0], [1.0]),
            previous_state=None if k == 0 else ensembles[k - 1],
        )

    assert metric.evaluate(context(0)) == {}
    for k in (1, 2):
        result = metric.evaluate(context(k))
        for label in (*NAMES, "all"):
            # rtol: the same norm (PET-fork's normalized_l2), vectorised over members.
            np.testing.assert_allclose(
                result[label], np.asarray(ref[f"out_change_{label}"], float)[k - 1], rtol=1e-13
            )
