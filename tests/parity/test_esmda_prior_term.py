"""PET's ES-MDA with the full and subspace analyses: a PET defect, pinned (not ported).

PET (``e65fb10``) lets ES-MDA run the full and subspace analyses, and passes them the
*original* prior at every step (``prior = prior_enX``; the subspace weights accumulate
relative to it). ES-MDA must use the current ensemble as each step's prior: re-imposing
the original one counts the prior at every step and the data with weight ``1 / alpha``
only. The reference (``esmda_prior_term_linear``) runs PET's analyses exactly as its
``ESMDA.calc_analysis`` does on a linear-Gaussian case; this test checks that APEX's kernels
reproduce PET's procedure, and that the result is not the posterior. APEX's ES-MDA refuses
both analyses (tests/workflows/test_lm_enrml.py).
"""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.data_assimilation import DataErrorCovariance, FullUpdate, SubspaceUpdate

type ReferenceLoader = Callable[[str], dict[str, NDArray[np.generic]]]

#: As the other steps: SVD and summation order differ.
STEP_RTOL = 1e-12


def _floats(ref: dict[str, NDArray[np.generic]], key: str) -> NDArray[np.float64]:
    return np.asarray(ref[key], dtype=np.float64)


def _posterior_covariance(ref: dict[str, NDArray[np.generic]], alpha: float) -> NDArray[np.float64]:
    """The Kalman posterior covariance with ``alpha C_d``, from the prior's sample covariance."""
    g, c_d = _floats(ref, "in_forward"), np.diag(_floats(ref, "in_variances"))
    c_p = np.cov(_floats(ref, "in_prior"))
    gain = c_p @ g.T @ np.linalg.inv(g @ c_p @ g.T + alpha * c_d)
    covariance: NDArray[np.float64] = c_p - gain @ g @ c_p
    return covariance


@pytest.mark.pet_parity
@pytest.mark.parametrize("flavour", ["full", "subspace"])
def test_pets_es_mda_with_the_prior_term_is_not_the_posterior(
    pet_reference: ReferenceLoader, flavour: str
) -> None:
    ref = pet_reference("esmda_prior_term_linear")
    g, variances = _floats(ref, "in_forward"), _floats(ref, "in_variances")
    observations, prior = _floats(ref, "in_observations"), _floats(ref, "in_prior")
    strategy = (
        FullUpdate(1.0, state_scaling=np.full(3, float(_floats(ref, "in_prior_std"))))
        if flavour == "full"
        else SubspaceUpdate(1.0 - 1e-9)  # all but the null singular values, as the case
    )
    strategy.prepare(prior)
    state = prior
    for alpha, z in zip(_floats(ref, "in_alphas"), _floats(ref, "in_z"), strict=True):
        error = DataErrorCovariance.diagonal(alpha * variances)
        perturbed = observations[:, None] + error.correlate(z)
        state = state + strategy.step(state, g @ state, perturbed, error, prior_state=prior)
        if isinstance(strategy, SubspaceUpdate):
            strategy.accept_trial()
    pet = _floats(ref, f"out_posterior_{flavour}")

    # APEX's kernels, driven PET's way, reproduce PET's procedure.
    np.testing.assert_allclose(state, pet, rtol=0, atol=STEP_RTOL * np.abs(pet).max())
    # ...and it is not ES-MDA: the posterior variances are far too large (measured: full
    # 1.4-1.7x, subspace 2.4-3.1x the exact ones; 2000 members, so sampling error ~3%).
    exact = np.diag(_posterior_covariance(ref, 1.0))
    ratio = np.diag(np.cov(pet)) / exact
    assert ratio.min() > (1.3 if flavour == "full" else 2.0)
    if flavour == "subspace":
        # For a linear model each full Gauss-Newton step from the prior lands on the RML
        # solution with alpha C_d: the result is an ES with alpha_last C_d.
        es_last = np.diag(_posterior_covariance(ref, float(_floats(ref, "in_alphas")[-1])))
        np.testing.assert_allclose(np.diag(np.cov(pet)), es_last, rtol=0.05)
