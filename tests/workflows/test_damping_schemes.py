"""Damping schemes as an auxiliary technique.

GIES and LM-EnRML each adapt ``lambda`` by a damping scheme, ``[method.damping]``: ``rlmmac``
(GIES's rule, its default) or ``lm`` (PET's LM-EnRML rule, LM-EnRML's default). Either
method runs either scheme. Independent references: each run's ``lambda`` sequence is
recomputed from its own history by the scheme's rule, and the fixed points do not depend
on the scheme (the approximate form's least-squares fit on the linear-Gaussian case).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition
from apex.storage.run_directory import RunDirectory

STRATEGIES = {
    "approx": '{ name = "approx", options = { energy = 1.0 } }',
    "rlmmac": '{ name = "rlmmac", options = { energy = 1.0 } }',
}
DAMPING = {
    # A floor the default reduction (0.9) reaches on the tanh case within the 60 iterations.
    "rlmmac": '{ name = "rlmmac", options = { initial = 1.0, factor = 2.0, minimum = 0.1, '
    "data_mismatch_tolerance = 1e-12 } }",
    "lm": '{ name = "lm", options = { initial = 1.0, factor = 5.0, '
    "data_mismatch_tolerance = 1e-12 } }",
}


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path

    def least_squares(self, perturbed: np.ndarray) -> np.ndarray: ...


CaseFactory = Callable[..., Case]


def _method(method: str, strategy: str, damping: str | None, extra: str = "") -> str:
    lines = [f'name = "{method}"', f"strategy = {STRATEGIES[strategy]}", "max_iterations = 60"]
    if damping is not None:
        lines.append(f"damping = {DAMPING[damping]}")
    return "\n".join(lines) + extra


@pytest.mark.parametrize("method", ["gies", "lm_enrml"])
@pytest.mark.parametrize("damping", ["rlmmac", "lm"])
@pytest.mark.parametrize("strategy", ["approx", "rlmmac"])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_either_scheme_reaches_the_same_fit_and_follows_its_rule(
    linear_case: CaseFactory, method: str, damping: str, strategy: str, nonlinear: bool
) -> None:
    """Linear: the least-squares fit, whatever the scheme. Tanh: trials are rejected
    (21-26 of 54 measured), so both rules' branches run."""
    extra = 'nonlinearity = "tanh"' if nonlinear else ""
    case = linear_case(
        method=_method(method, strategy, damping), ensemble_size=30, simulator_extra=extra
    )

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    if not nonlinear:
        np.testing.assert_allclose(result.posterior, case.least_squares(perturbed), atol=1e-5)
    else:
        assert any(not r.accepted for r in result.history)
    # Recompute every lambda the trials ran with, by the scheme's rule.
    history = result.history
    lam = 1.0
    previous_mean, previous_std = history[0].data_mismatch_mean, history[0].data_mismatch_std
    held = 0
    for record in history[1:]:
        assert record.damping == pytest.approx(lam, rel=1e-12), (damping, record.iteration)
        # Both schemes accept exactly the trials that improved (within tolerance too).
        assert record.accepted == (record.data_mismatch_mean < previous_mean)
        if record.accepted:
            before = lam
            small = abs(1.0 - record.data_mismatch_mean / previous_mean) < 1e-12
            if damping == "rlmmac":
                lam = lam if lam <= 0.1 else lam * 0.9  # the default reduction
            elif small and lam > 0.01:  # lm: a tentative step relaxes, whatever the spread
                lam = max(lam / 5.0, 0.01)
            elif record.data_mismatch_std < previous_std and lam > 0.01:  # lm: spread-gated
                lam /= 5.0
            held += lam == before
            previous_mean, previous_std = record.data_mismatch_mean, record.data_mismatch_std
        else:
            lam *= 2.0 if damping == "rlmmac" else 5.0
    assert previous_mean <= history[0].data_mismatch_mean
    # An accepted trial that kept its lambda: lm's spread gate, or rlmmac's floor (which
    # the approx runs reach on the tanh case; the RLM-MAC ones stay above it).
    if nonlinear and (damping == "lm" or strategy == "approx"):
        assert held > 0


@pytest.mark.parametrize("method", ["gies", "lm_enrml"])
def test_without_a_damping_section_each_method_keeps_its_default(
    linear_case: CaseFactory, method: str
) -> None:
    from apex.algorithms.data_assimilation import DampingSchedule, LMDampingSchedule

    case = linear_case(method=_method(method, "approx", None), ensemble_size=10)
    run = composition.build_run(composition.load_config(case.config))

    expected = DampingSchedule if method == "gies" else LMDampingSchedule
    assert type(run.method.schedule) is expected  # type: ignore[attr-defined]


def test_damping_is_refused_where_it_does_not_apply(linear_case: CaseFactory) -> None:
    gn = linear_case(name="gn", method=_method("gn_enrml", "approx", "lm"), ensemble_size=10)
    with pytest.raises(ConfigError, match="adapts no damping"):
        composition.start(gn.config)
    both = linear_case(
        name="both", method=_method("gies", "approx", "lm", "\nschedule = { initial = 2.0 }"),
        ensemble_size=10,
    )  # fmt: skip
    with pytest.raises(ConfigError, match="not also"):
        composition.start(both.config)
    unknown = linear_case(name="unknown", ensemble_size=10, method=(
        'name = "gies"\nstrategy = { name = "approx" }\ndamping = { name = "nope" }'))  # fmt: skip
    with pytest.raises(ConfigError, match=r"\[method.damping\]"):
        composition.start(unknown.config)


def test_lm_enrml_with_rlm_mac_needs_a_floor_whatever_the_scheme(linear_case: CaseFactory) -> None:
    """The user's rule: rlmmac's scheme has no floor by default."""
    text = 'name = "lm_enrml"\nstrategy = { name = "rlmmac" }\ndamping = { name = "rlmmac" }'
    case = linear_case(method=text, ensemble_size=10)
    with pytest.raises(ConfigError, match="needs a positive damping minimum"):
        composition.start(case.config)


def test_a_study_variant_switches_the_damping_scheme(
    tmp_path: Path, linear_case: CaseFactory
) -> None:
    from apex.services.study import expand, load_study

    case = linear_case(method=_method("gies", "rlmmac", None), ensemble_size=10)
    study = case.directory / "study.toml"
    study.write_text(
        '[study]\nname = "damping"\n[study.cases]\nlinear = "config.toml"\n'
        '[[study.variants]]\nname = "rlmmac_damping"\n'
        '[[study.variants]]\nname = "lm_damping"\n'
        'patch = { method = { damping = { name = "lm", options = { minimum = 0.01 } } } }\n'
    )

    cells = expand(load_study(study), tmp_path, case.directory)

    assert cells[0].config.method.damping is None
    assert cells[1].config.method.damping is not None
    assert cells[1].config.method.damping.name == "lm"
    assert json.loads(cells[1].config.model_dump_json())["method"]["strategy"]["name"] == "rlmmac"


@pytest.mark.parametrize("nonlinear", [False, True])
def test_lm_enrml_starts_from_auto_with_rlm_mac(linear_case: CaseFactory, nonlinear: bool) -> None:
    """The roadmap item: ``initial = "auto"`` (half the prior's mean
    data mismatch per datum) was calibrated for the approx update's ``1 + lambda``; with
    RLM-MAC's relative damping it still starts as PET says and reaches the fit."""
    method = (
        'name = "lm_enrml"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
        'damping = { name = "lm", options = { initial = "auto", data_mismatch_tolerance = 1e-12 } }'
        "\nmax_iterations = 60"
    )
    extra = 'nonlinearity = "tanh"' if nonlinear else ""
    case = linear_case(method=method, ensemble_size=30, simulator_extra=extra)

    result = composition.start(case.config)

    run = RunDirectory(case.directory / "runs" / "case")
    perturbed = run.load_arrays("perturbed_observations")["perturbed_observations"]
    first = result.history[1].damping
    assert first == pytest.approx(0.5 * result.history[0].data_mismatch_mean / perturbed.shape[0])
    if nonlinear:
        assert result.history[-1].data_mismatch_mean < 0.4 * result.history[0].data_mismatch_mean
    else:
        np.testing.assert_allclose(result.posterior, case.least_squares(perturbed), atol=1e-5)


def test_a_run_with_another_scheme_resumes_identically(linear_case: CaseFactory) -> None:
    method = _method("gies", "rlmmac", "lm")
    whole = linear_case(name="whole", method=method, ensemble_size=20,
                        simulator_extra='nonlinearity = "tanh"')  # fmt: skip
    uninterrupted = composition.start(whole.config)
    parts = linear_case(name="parts", method=method, ensemble_size=20,
                        simulator_extra='nonlinearity = "tanh"')  # fmt: skip
    composition.start(parts.config, max_steps=4)

    resumed = composition.resume(parts.directory / "runs" / "parts")

    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)
    assert [r.damping for r in resumed.history] == [r.damping for r in uninterrupted.history]


def test_gies_decide_is_only_for_the_scheme_without_spreads() -> None:
    from apex.algorithms.data_assimilation import GIES, LMDampingSchedule

    assert GIES().decide(1.0, 10.0, 5.0).accepted
    with pytest.raises(TypeError, match="spreads"):
        GIES(schedule=LMDampingSchedule()).decide(1.0, 10.0, 5.0)


def test_an_auto_start_is_the_lm_schemes(linear_case: CaseFactory) -> None:
    text = (
        'name = "gies"\nstrategy = { name = "approx" }\n'
        'damping = { name = "rlmmac", options = { initial = "auto" } }'
    )
    case = linear_case(method=text, ensemble_size=10)
    with pytest.raises(ConfigError, match="the lm scheme"):
        composition.start(case.config)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ({"name": "gies"}, {"name": "gies", "schedule": {"reduction": None}}),
        (
            {"name": "gies", "schedule": {"factor": 3.0}},
            {"name": "gies", "schedule": {"reduction": None, "factor": 3.0}},
        ),
        (  # an explicit reduction is kept
            {"name": "gies", "schedule": {"reduction": 0.9}},
            {"name": "gies", "schedule": {"reduction": 0.9}},
        ),
        (
            {"name": "lm_enrml", "damping": {"name": "rlmmac", "options": {"initial": 2.0}}},
            {
                "name": "lm_enrml",
                "damping": {"name": "rlmmac", "options": {"reduction": None, "initial": 2.0}},
            },
        ),
        (  # CHOP: the wrapped method's schedule
            {"name": "chop", "options": {"inner": "gies"}},
            {"name": "chop", "options": {"inner": "gies"}, "schedule": {"reduction": None}},
        ),
        (  # CHOP around LM-EnRML: its LM rule, no reduction
            {"name": "chop", "options": {"inner": "lm_enrml"}},
            {"name": "chop", "options": {"inner": "lm_enrml"}},
        ),
        ({"name": "gies", "schedule": [1]}, {"name": "gies", "schedule": [1]}),  # malformed
        ({"name": "chop", "options": "x"}, {"name": "chop", "options": "x"}),  # malformed
        (
            {"name": "gies", "damping": {"name": "rlmmac", "options": "x"}},
            {"name": "gies", "damping": {"name": "rlmmac", "options": "x"}},
        ),
        ({"name": "lm_enrml"}, {"name": "lm_enrml"}),  # PET's LM rule, no reduction
        ({"name": "gies", "damping": {"name": "lm"}}, {"name": "gies", "damping": {"name": "lm"}}),
        ({"name": "esmda"}, {"name": "esmda"}),  # a fixed schedule
    ],
)
def test_snapshots_from_before_the_reduction_default_keep_the_division(
    method: dict[str, object], expected: dict[str, object]
) -> None:
    """Only the rlmmac scheme of an old snapshot is pinned."""
    raw = {"method": json.loads(json.dumps(method))}

    composition._migrate_damping_reduction(raw)

    assert raw["method"] == expected
