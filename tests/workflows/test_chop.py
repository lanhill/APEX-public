"""CHOP through the loop: runs, saves, equals GIES with unit factors, resumes."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_directory import RunDirectory


class LinearCase(Protocol):
    config: Path
    directory: Path
    forward: np.ndarray


CaseFactory = Callable[..., LinearCase]


def _chop(components: str, extra: str = "", strategy: str = "") -> str:
    return (
        'name = "chop"\nmax_iterations = 4\n'
        f'options = {{ inner = "gies", components = [{components}]{extra} }}\n{strategy}'
    )


LENGTH_SCALES = '{ name = "length_scale", mode = "data" }'


def test_chop_runs_and_saves_every_iterates_hyperparameters(linear_case: CaseFactory) -> None:
    case = linear_case(ensemble_size=20, method=_chop(LENGTH_SCALES))

    result = composition.start(case.config)

    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert accepted[-1] < accepted[0]
    run = RunDirectory(case.directory / "runs" / "case")
    first = np.load(run.path / "checkpoints" / "iter_0.npz")["hyperparameters"]
    last_iteration = max(r.iteration for r in result.history if r.accepted)
    last = np.load(run.path / "checkpoints" / f"iter_{last_iteration}.npz")["hyperparameters"]
    n_data = first.shape[0]
    assert first.shape == (n_data, 20)
    assert np.all((first >= 0.23) & (first <= 0.43))  # the paper's prior
    assert not np.array_equal(first, last)  # estimated as the run went
    assert np.all(last >= 1e-6)  # the floor


def test_with_unit_factors_chop_is_gies(linear_case: CaseFactory) -> None:
    """Weights fixed at 1 (no spread): the tapers are 1, so the update is GIES's."""
    unit = '{ name = "data_weights", mode = "datum", low = 1.0, high = 1.0 }'
    chop = linear_case(name="chop", ensemble_size=20, method=_chop(unit))
    gies = linear_case(name="gies", ensemble_size=20, method='name = "gies"\nmax_iterations = 4\n')

    with_chop, without = composition.start(chop.config), composition.start(gies.config)

    np.testing.assert_allclose(with_chop.posterior, without.posterior, rtol=1e-10, atol=1e-12)
    assert [r.accepted for r in with_chop.history] == [r.accepted for r in without.history]


def test_a_resumed_chop_run_is_bit_identical(linear_case: CaseFactory) -> None:
    both = LENGTH_SCALES + ', { name = "gain_inflation", mode = "scalar" }'
    method = _chop(both, ', hyper_localization = "same"')
    whole = linear_case(name="whole", ensemble_size=20, method=method)
    uninterrupted = composition.start(whole.config)
    parts = linear_case(name="parts", ensemble_size=20, method=method)
    composition.start(parts.config, max_steps=2)

    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.data_mismatch_mean for r in resumed.history] == [
        r.data_mismatch_mean for r in uninterrupted.history
    ]


@pytest.mark.parametrize(
    ("method", "message"),
    [
        (_chop(LENGTH_SCALES, strategy='strategy = { name = "approx" }'), "RLM-MAC strategy"),
        (_chop(LENGTH_SCALES, strategy='strategy = { name = "local_rlmmac" }'), "RLM-MAC"),
        (
            _chop(
                LENGTH_SCALES,
                strategy='strategy = { name = "rlmmac" }\nlocalization = { name = "adaptive" }',
            ),
            "remove \\[method.localization\\]",
        ),
        (
            # PET-fork's unscaled dY would apply to the model step but not CHOP's
            # hyper-parameter step, which shares the damping.
            _chop(
                LENGTH_SCALES,
                strategy=(
                    'strategy = { name = "rlmmac", options = { scale_data_anomalies = false } }'
                ),
            ),
            "scale_data_anomalies = true",
        ),
        (
            _chop(LENGTH_SCALES, ', inflation = [2, 2], inner = "esmda"').replace(
                'inner = "gies", ', ""
            ),
            "GIES's loop",
        ),
        (_chop('{ name = "nope" }'), "nope"),
    ],
)
def test_what_chop_cannot_wrap_is_refused(
    linear_case: CaseFactory, method: str, message: str
) -> None:
    case = linear_case(ensemble_size=10, method=method)

    with pytest.raises(ConfigError, match=message):
        composition.start(case.config)


def test_failed_members_take_their_hyper_parameters_with_them(linear_case: CaseFactory) -> None:
    case = linear_case(
        ensemble_size=20, method=_chop(LENGTH_SCALES),
        simulator_extra="fail_members = [3]",
        failures="[failures]\nmin_success_fraction = 0.9",
    )  # fmt: skip

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    theta = np.load(run.path / "checkpoints" / "iter_0.npz")["hyperparameters"]
    assert theta.shape[1] == 19  # member 3 failed in the prior forecast
    assert result.posterior.shape[-1] == 19


def test_a_registered_localization_may_localize_the_hyper_parameter_update(
    linear_case: CaseFactory,
) -> None:
    method = _chop(LENGTH_SCALES, ', hyper_localization = "adaptive_per_update"')
    case = linear_case(ensemble_size=20, method=method)

    result = composition.start(case.config)

    accepted = [r.data_mismatch_mean for r in result.history if r.accepted]
    assert accepted[-1] < accepted[0]


def test_the_published_5spots_configuration_runs(linear_case: CaseFactory) -> None:
    """docs/chop.md's reproduction of the paper's 5Spots P-AutoAdaLoc-0 run's choices."""
    method = (
        'name = "chop"\nmax_iterations = 3\n'
        "schedule = { initial = 1.0, factor = 2.0, reduction = 0.9, "
        "data_mismatch_tolerance = 1e-4 }\n"
        'strategy = { name = "rlmmac", options = { energy = 0.99 } }\n'
        '[method.options]\ninner = "gies"\nmax_trials_per_iteration = 4\n'
        'components = [{ name = "length_scale", mode = "data", correlation = "current" }]\n'
        'hyper_localization = "none"'
    )
    case = linear_case(ensemble_size=20, method=method)

    result = composition.start(case.config)

    assert [r.data_mismatch_mean for r in result.history if r.accepted][-1] < (
        result.history[0].data_mismatch_mean
    )


def test_the_hyper_update_sees_the_accepted_trials_lambda_and_mean_model(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding: the acceptance-time wiring that makes CHOP's update after an
    acceptance equal the code's per-trial update (whose rejected ones are discarded)."""
    from apex.algorithms.data_assimilation.chop.method import Chop

    seen: list[dict[str, object]] = []
    original = Chop.after_accept

    def recording(self: Chop, state, predicted, reference, perturbed, data_error, damping) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
        seen.append({"state": state.copy(), "reference": reference.copy(), "damping": damping})
        original(self, state, predicted, reference, perturbed, data_error, damping)

    monkeypatch.setattr(Chop, "after_accept", recording)
    case = linear_case(ensemble_size=20, method=_chop(LENGTH_SCALES))

    result = composition.start(case.config)

    accepted = [r for r in result.history if r.accepted and r.iteration > 0]
    assert len(seen) == len(accepted)  # once per accepted trial, never for a rejected one
    assert [call["damping"] for call in seen] == [r.damping for r in accepted]  # the trial's
    forward = case.forward
    for call in seen:
        mean_model = forward @ np.asarray(call["state"]).mean(axis=1)
        np.testing.assert_allclose(
            np.asarray(call["reference"]), mean_model, rtol=1e-12, atol=1e-12
        )


def test_a_resumed_run_after_a_failure_is_bit_identical(linear_case: CaseFactory) -> None:
    """Review finding: the hyper taper's cutoff used the initial ensemble size."""
    both = LENGTH_SCALES + ', { name = "gain_inflation", mode = "scalar" }'
    method = _chop(both)
    failing = {"simulator_extra": "fail_members = [3]",
               "failures": "[failures]\nmin_success_fraction = 0.9"}  # fmt: skip
    whole = linear_case(name="whole", ensemble_size=20, method=method, **failing)
    uninterrupted = composition.start(whole.config)
    parts = linear_case(name="parts", ensemble_size=20, method=method, **failing)
    composition.start(parts.config, max_steps=2)

    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)


def test_chops_options_are_checked_with_its_inner_methods() -> None:
    from apex.services.plugins import check_options

    good = {"inner": "gies", "max_trials_per_iteration": 4,
            "components": [{"name": "length_scale"}]}  # fmt: skip

    assert check_options("methods", "chop", good) == []
    errors = check_options("methods", "chop", {**good, "max_trails_per_iteration": 4})
    assert any("max_trails_per_iteration" in e for e in errors)


def test_the_assistant_keeps_a_wrappers_max_iterations() -> None:
    """CHOP passes max_iterations to its iterative inner method, so it is not "not used".

    The assistant's summaries marked every method without a ``max_iterations`` parameter as
    run by its schedule, which hid CHOP's iteration budget.
    """
    from apex.agents.chat import _method_steps

    method: dict[str, Any] = {"name": "chop", "max_iterations": 5, "options": {"inner": "gies"}}
    _method_steps(method)
    assert method["max_iterations"] == 5
    assert "steps" not in method


def test_a_wrapper_is_judged_by_the_method_it_runs() -> None:
    """Whether max_iterations applies is the wrapped method's: a wrapper of ES-MDA shows its
    inflation steps, not an iteration limit (wrappers were all taken as iterative).

    CHOP itself refuses ES-MDA as its inner method (no hyper-parameter hook): this tests the
    generic helpers on a wrapper's options, not a configuration that would build.
    """
    from apex.agents.chat import _method_steps
    from apex.services.plugins import iterated_method, takes_max_iterations

    options = {"inner": "esmda", "inflation": [4.0, 4.0, 4.0, 4.0]}
    assert iterated_method("chop", options) == ("esmda", {"inflation": [4.0, 4.0, 4.0, 4.0]})
    assert not takes_max_iterations("chop", options)
    assert takes_max_iterations("chop", {"inner": "gies"})
    assert takes_max_iterations("chop", {})  # inner defaults to gies

    method: dict[str, Any] = {"name": "chop", "max_iterations": 5, "options": options}
    _method_steps(method)
    assert method["max_iterations"].startswith("not used")
    assert method["steps"] == 4
