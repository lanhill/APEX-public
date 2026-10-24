"""QA/QC diagnostics against PET's ``qaqc_tools.QAQC`` (upstream ``e65fb10``).

The reference (``qaqc_diagnostics``) drives PET's ``_gain`` (exact arrays) and
``calc_mahalanobis((1, None, 2, None, 1, "time"))`` and ``calc_da_stat`` (their logs, 4
significant digits). PET perturbs the forecasts with one ``default_rng(50)`` stream across
the three passes, in this order; APEX draws the same.
"""

import re
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms import qaqc

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

TYPES = 3
NAMES = ("permx", "mult")
#: PET logs scores with 4 significant digits.
LOG_RTOL = 5e-4


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


def _data(ref: dict[str, NDArray[np.generic]]) -> list[tuple[NDArray[np.float64], ...]]:
    return [
        (
            _floats(ref, f"in_forecast_{k}"),
            _floats(ref, f"in_observations_{k}"),
            _floats(ref, f"in_variances_{k}"),
        )
        for k in range(TYPES)
    ]


@pytest.mark.pet_parity
def test_the_kalman_gains_match_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("qaqc_diagnostics")
    for k, (forecast, observed, variances) in enumerate(_data(ref)):
        for name in NAMES:
            for damping, key in ((0.0, "gain"), (0.7, "gain_damped")):
                gain = qaqc.kalman_gain(
                    _floats(ref, f"in_state_{name}"), forecast, observed, variances, damping
                )
                expected = _floats(ref, f"out_{key}_{k}_{name}")
                assert gain is not None
                # rtol: the same subspace algebra; SVD and eigh order differ (~1e-15).
                np.testing.assert_allclose(
                    gain, expected, rtol=1e-10, atol=1e-12 * np.abs(expected).max()
                )


@pytest.mark.pet_parity
def test_the_mahalanobis_scores_match_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("qaqc_diagnostics")
    data = _data(ref)
    forecast = np.vstack([f for f, _, _ in data])
    observed = np.concatenate([o for _, o, _ in data])
    variances = np.concatenate([v for _, _, v in data])
    rng = np.random.default_rng(qaqc.MAHALANOBIS_SEED)

    level1 = qaqc.mahalanobis_scores(qaqc.perturbed_forecast(forecast, variances, rng), observed)
    level2 = qaqc.joint_mahalanobis(qaqc.perturbed_forecast(forecast, variances, rng), observed, 2)
    projected = [qaqc.leading_projection(qaqc.perturbed_forecast(f, v, rng), o) for f, o, v in data]
    by_type = qaqc.mahalanobis_scores(
        np.array([p for p, _ in projected]), np.array([o for _, o in projected])
    )

    top = np.sort(level1)[::-1][:10]
    np.testing.assert_allclose(top, _floats(ref, "out_level1_scores"), rtol=LOG_RTOL)
    top2 = sorted(level2.values(), reverse=True)[:10]
    np.testing.assert_allclose(top2, _floats(ref, "out_level2_scores"), rtol=LOG_RTOL)
    np.testing.assert_allclose(
        np.sort(by_type)[::-1], _floats(ref, "out_level1_type_scores"), rtol=LOG_RTOL
    )


@pytest.mark.pet_parity
def test_the_parameter_movement_matches_pet(pet_reference: ReferenceLoader) -> None:
    ref = pet_reference("qaqc_diagnostics")
    log = str(np.asarray(ref["out_da_stat_log"]))
    for name in NAMES:
        movement = qaqc.parameter_movement(
            _floats(ref, f"in_prior_{name}"), _floats(ref, f"in_state_{name}")
        )
        line = next(r for r in log.splitlines() if r.startswith(f"Group {name}:"))
        numbers = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]
        np.testing.assert_allclose([movement.prior_std, movement.std], numbers[:2], rtol=LOG_RTOL)
        assert list(movement.beyond) == pytest.approx(numbers[2:5], abs=0.05)
