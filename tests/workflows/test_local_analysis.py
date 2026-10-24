"""Local analysis end to end: every method it supports, and resume."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from apex.core.errors import ConfigError
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


CaseFactory = Callable[..., Case]

LOCAL = 'strategy = { name = "local_analysis", options = { energy = 1.0 } }'
METHODS = {
    "es": f'name = "es"\n{LOCAL}',
    "esmda": f'name = "esmda"\noptions = {{ inflation = [4, 4, 4, 4] }}\n{LOCAL}',
    "lm_enrml": f'name = "lm_enrml"\n{LOCAL}\nschedule = {{ initial = 1.0 }}\nmax_iterations = 6',
    "gn_enrml": f'name = "gn_enrml"\n{LOCAL}\nmax_iterations = 6',
}


@pytest.mark.parametrize("method", list(METHODS))
def test_local_analysis_runs_with_every_approx_method(
    linear_case: CaseFactory, method: str
) -> None:
    """The default selection (adaptive_prior, hard taper) keeps the correlated data, so the
    fit improves; the run resumes bit-identically (the selection is checkpointed)."""
    whole = linear_case(name="whole", method=METHODS[method], ensemble_size=40)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=METHODS[method], ensemble_size=40)
    composition.start(case.config, max_steps=1)

    resumed = composition.resume(case.directory / "runs" / "parts")

    kept = [r.data_mismatch_mean for r in uninterrupted.history if r.accepted]
    assert kept[-1] < kept[0] / 5
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)


@pytest.mark.parametrize("taper", ["soft", "sigmoid"])
def test_a_soft_adaptive_taper_cannot_select(linear_case: CaseFactory, taper: str) -> None:
    """A soft or sigmoid adaptive taper is positive for nearly every pair: no subset."""
    method = (
        'name = "es"\nstrategy = { name = "local_analysis" }\n'
        f'localization = {{ name = "adaptive_prior", options = {{ taper = "{taper}" }} }}'
    )
    case = linear_case(method=method, ensemble_size=20)

    with pytest.raises(ConfigError, match="needs the hard taper"):
        composition.start(case.config)


def test_the_per_update_localization_cannot_select(linear_case: CaseFactory) -> None:
    method = (
        'name = "es"\nstrategy = { name = "local_analysis" }\n'
        'localization = { name = "adaptive_per_update" }'
    )
    case = linear_case(method=method, ensemble_size=20)

    with pytest.raises((ConfigError, ValueError), match="adaptive_per_update"):
        composition.start(case.config)


def test_gies_runs_the_local_rlmmac_and_resumes_identically(linear_case: CaseFactory) -> None:
    method = (
        'name = "gies"\nstrategy = { name = "local_rlmmac" }\n'
        "schedule = { data_mismatch_tolerance = 1e-6 }\nmax_iterations = 6"
    )
    whole = linear_case(name="whole", method=method, ensemble_size=40)
    uninterrupted = composition.start(whole.config)
    case = linear_case(name="parts", method=method, ensemble_size=40)
    composition.start(case.config, max_steps=2)

    resumed = composition.resume(case.directory / "runs" / "parts")

    kept = [r.data_mismatch_mean for r in uninterrupted.history if r.accepted]
    assert kept[-1] < kept[0] / 5
    np.testing.assert_array_equal(resumed.posterior, uninterrupted.posterior)


def test_gies_refuses_the_approx_local_analysis(linear_case: CaseFactory) -> None:
    case = linear_case(method='name = "gies"\nstrategy = { name = "local_analysis" }')

    with pytest.raises(ConfigError, match="local_rlmmac"):
        composition.start(case.config)
