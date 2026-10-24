"""Upstream PET's adaptive localization in the projected-data space (``e65fb10``).

The reference (``approx_autoadaloc_rankr``) runs ``approx_update`` with
``AutoAdaptiveLocalization`` (``projection = "rank-r"``) for each threshold and taper, at
``lambda = 0`` (ES, ES-MDA) and ``lambda = 1.5`` (LM-EnRML), and with ``emp_cov``; two
parameter blocks of 30 rows, 25 data, 20 members, truncating energy. APEX's
``adaptive_projected`` runs on the same inputs, with the shuffle drawn from a generator of
the same seed.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import (
    ApproxUpdate,
    DataErrorCovariance,
    ProjectedAdaptiveLocalization,
)
from apex.algorithms.data_assimilation.localization import Taper, Threshold

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

RUNS = (
    "adaptive_hard",
    "adaptive_soft",
    "adaptive_sigmoid",
    "fixed_hard",
    "universal_soft",
    "universal_hard_lm",
    "adaptive_hard_emp_cov",
)
#: PET's ``type`` names for APEX's tapers.
TAPERS = {"hard": "hard", "soft": "soft", "sigm": "sigmoid"}
#: SVD, eigh and summation order, and PET's np.corrcoef against APEX's normalised
#: products: measured 6e-15 to 1.5e-13 of the largest step (the most for the universal soft
#: taper, whose Gaspari-Cohn slope amplifies the rounding of the correlations). The hard
#: tapers agree exactly: no correlation lies within rounding of its threshold here.
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


@pytest.mark.pet_parity
@pytest.mark.parametrize("run", RUNS)
def test_the_projected_localized_step_matches_pet(pet_reference: ReferenceLoader, run: str) -> None:
    ref = pet_reference("approx_autoadaloc_rankr")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    perturbed = _floats(ref, "in_observations")[:, None] + error.correlate(_floats(ref, "in_z"))
    np.testing.assert_array_equal(perturbed, ref["out_perturbed_observations"])
    seed = int(np.asarray(ref[f"in_shuffle_seed_{run}"]))
    # APEX draws the shuffle as PET does; the same seed must give PET's permutation.
    np.testing.assert_array_equal(
        np.random.default_rng(seed).permutation(perturbed.shape[1]),
        ref[f"out_permutation_{run}"],
    )
    localization = ProjectedAdaptiveLocalization(
        [int(b) for b in np.asarray(ref["in_block_sizes"])],
        threshold=cast(Threshold, str(ref[f"in_threshold_{run}"])),
        cutoff=float(_floats(ref, f"in_cutoff_{run}")),
        taper=cast(Taper, TAPERS[str(ref[f"in_taper_{run}"])]),
    )
    strategy = ApproxUpdate(
        float(_floats(ref, "in_energy")),
        localization=localization,
        empirical_data_error=bool(np.asarray(ref[f"in_emp_cov_{run}"])),
    )

    step = strategy.step(
        _floats(ref, "in_state"),
        _floats(ref, "in_predicted"),
        perturbed,
        error,
        damping=float(_floats(ref, f"in_lambda_{run}")),
        rng=np.random.default_rng(seed),
    )

    expected = _floats(ref, f"out_step_{run}")
    np.testing.assert_allclose(step, expected, rtol=0, atol=STEP_RTOL * np.abs(expected).max())


@pytest.mark.pet_parity
@pytest.mark.parametrize("run", RUNS)
def test_the_reference_localizes(pet_reference: ReferenceLoader, run: str) -> None:
    """The case is informative: each taper changes PET's step from the unlocalized one."""
    ref = pet_reference("approx_autoadaloc_rankr")
    error = DataErrorCovariance.diagonal(_floats(ref, "in_variances"))
    unlocalized = ApproxUpdate(
        float(_floats(ref, "in_energy")),
        empirical_data_error=bool(np.asarray(ref[f"in_emp_cov_{run}"])),
    ).step(
        _floats(ref, "in_state"),
        _floats(ref, "in_predicted"),
        _floats(ref, "out_perturbed_observations"),
        error,
        damping=float(_floats(ref, f"in_lambda_{run}")),
    )
    assert not np.allclose(_floats(ref, f"out_step_{run}"), unlocalized)
