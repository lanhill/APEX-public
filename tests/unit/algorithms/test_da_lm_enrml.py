"""LM-EnRML's damping schedule and construction (the rules' parity is in tests/parity/)."""

import math

import pytest

from apex.algorithms.data_assimilation import ApproxUpdate, iterative_methods
from apex.algorithms.data_assimilation.lm_enrml import LMDampingSchedule, LMEnRML


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"initial": 0.0}, "initial lambda"),
        ({"initial": "never"}, "initial lambda"),
        ({"factor": 1.0}, "factor"),
        ({"minimum": 5.0, "maximum": 1.0}, "minimum < maximum"),
        ({"data_mismatch_tolerance": 0.0}, "data_mismatch_tolerance"),
    ],
)
def test_invalid_schedules_are_refused(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LMDampingSchedule(**fields)  # type: ignore[arg-type]


def test_decide_refuses_non_finite_or_non_positive_data_mismatches() -> None:
    schedule = LMDampingSchedule()
    with pytest.raises(ValueError, match="finite"):
        schedule.decide(1.0, 10.0, math.nan, previous_spread=1.0, spread=1.0)
    with pytest.raises(ValueError, match="positive"):
        schedule.decide(1.0, 0.0, 1.0, previous_spread=1.0, spread=1.0)


def test_defaults_are_pets() -> None:
    method = iterative_methods.create("lm_enrml")

    assert isinstance(method, LMEnRML)
    assert isinstance(method.strategy, ApproxUpdate)
    assert method.strategy.energy == 0.95  # PET IterativeEnRML: iteration.energy
    assert method.lm_schedule == LMDampingSchedule(100.0, 5.0, 0.01, 1e10, 0.01)
    assert method.max_trials_per_iteration == 10  # max_inner_iter
    assert method.iterative
    assert not method.requires_reference_prediction  # no mean-model forecast, unlike GIES


def test_localization_and_gies_decisions_are_refused() -> None:
    from apex.algorithms.data_assimilation import PriorAdaptiveLocalization

    localized = ApproxUpdate(localization=PriorAdaptiveLocalization(block_sizes=[3]))
    with pytest.raises(ValueError, match="localize as upstream PET does"):
        LMEnRML(strategy=localized)
    with pytest.raises(TypeError, match="lm_schedule"):
        LMEnRML().decide(1.0, 10.0, 5.0)  # GIES's rule would ignore the spreads


def test_the_full_update_needs_its_prior_and_scales_per_variable() -> None:
    import numpy as np

    from apex.algorithms.data_assimilation import DataErrorCovariance, FullUpdate

    rng = np.random.default_rng(3)
    prior = np.vstack([rng.normal(0, 2.0, (4, 30)), rng.normal(0, 0.1, (2, 30))])
    predicted, error = rng.standard_normal((5, 30)), DataErrorCovariance.diagonal(np.ones(5))
    strategy = FullUpdate(block_sizes=[4, 2])
    with pytest.raises(ValueError, match="not prepared"):
        strategy.step(prior, predicted, predicted, error, prior_state=prior)

    strategy.prepare(prior)

    scaling = strategy.export_state()["scaling"]
    variances = np.var(prior, axis=-1, ddof=1)
    np.testing.assert_allclose(scaling[:4], np.sqrt(variances[:4].mean()))  # one per variable
    np.testing.assert_allclose(scaling[4:], np.sqrt(variances[4:].mean()))
    with pytest.raises(ValueError, match="prior state"):
        strategy.step(prior, predicted, predicted, error)
    with pytest.raises(ValueError, match="do not add up"):
        FullUpdate(block_sizes=[4]).prepare(prior)
    # At the prior itself the prior term vanishes: the full step is the approx step.
    full = strategy.step(prior, predicted, predicted + 1.0, error, damping=0.5, prior_state=prior)
    approx = ApproxUpdate(energy=0.98).step(prior, predicted, predicted + 1.0, error, damping=0.5)
    np.testing.assert_allclose(full, approx, rtol=1e-12, atol=1e-14)


def test_a_worse_trial_within_the_tolerance_is_rejected_not_converged() -> None:
    """PET stops on it; APEX rejects it."""
    from apex.algorithms.data_assimilation import LMDampingSchedule

    decision = LMDampingSchedule(initial=1.0).decide(
        1.0, 100.0, 100.5, previous_spread=1.0, spread=1.0
    )
    assert (decision.accepted, decision.converged, decision.stalled) == (False, False, True)
    pet = LMDampingSchedule(initial=1.0, converge_on_rejection=True).decide(
        1.0, 100.0, 100.5, previous_spread=1.0, spread=1.0
    )
    assert (pet.accepted, pet.converged) == (False, True)


def test_a_small_improvement_under_heavy_damping_is_not_yet_convergence() -> None:
    """Tester finding 34: with lambda = 100 the subspace step is 1/101 of Gauss-Newton's, so
    the first trial improves by less than the tolerance although the minimum is far away."""
    schedule = LMDampingSchedule()  # initial 100, factor 5, minimum 0.01, tolerance 1 %

    decision = schedule.decide(100.0, 252.0, 251.8, previous_spread=10.0, spread=10.0)

    assert (decision.accepted, decision.tentative, decision.converged) == (True, True, False)
    assert decision.damping == 20.0  # relaxed, whatever the spread did


def test_at_the_minimum_damping_a_small_improvement_converges() -> None:
    schedule = LMDampingSchedule()

    decision = schedule.decide(0.01, 252.0, 251.8, previous_spread=10.0, spread=10.0)

    assert (decision.accepted, decision.converged, decision.tentative) == (True, True, False)


def test_without_confirmation_the_first_small_improvement_converges_as_in_pet() -> None:
    schedule = LMDampingSchedule(confirm_convergence=False)

    decision = schedule.decide(100.0, 252.0, 251.8, previous_spread=10.0, spread=10.0)

    assert (decision.accepted, decision.converged, decision.tentative) == (True, True, False)
    assert decision.damping == 100.0
