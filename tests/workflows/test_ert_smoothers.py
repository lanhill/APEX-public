"""ERT's smoothers inside APEX's ES-MDA, on the linear-Gaussian case."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from apex.services import composition

pytest.importorskip("iterative_ensemble_smoother")


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    config: Path


CaseFactory = Callable[..., Case]


@pytest.mark.parametrize("strategy", ["ert_esmda", "ert_adaptive"])
def test_esmda_runs_with_erts_analysis(linear_case: CaseFactory, strategy: str) -> None:
    method = (
        'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        f'strategy = {{ name = "{strategy}" }}\n'
    )
    case = linear_case(method=method, ensemble_size=40, name=strategy)

    result = composition.start(case.config)

    assert result.complete
    assert len(result.history) == 5
    assert result.history[-1].data_mismatch_mean < 0.2 * result.history[0].data_mismatch_mean


def test_esmda_runs_with_erts_localized_analysis(linear_case: CaseFactory) -> None:
    import numpy as np

    method = (
        'name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "ert_localized", options = { taper_file = "taper.npz" } }\n'
    )
    case = linear_case(method=method, ensemble_size=40, name="localized")
    taper = case.config.parent / "taper.npz"
    np.savez(taper, taper=np.ones((1, 1)))  # building the run loads it: sized below
    config = composition.load_config(case.config)
    n_state = sum(v.size for v in config.state.variables)
    n_data = len(composition.build_run(config).workflow.analysis_observations.keys)
    np.savez(taper, taper=np.full((n_state, n_data), 0.8))

    result = composition.start(case.config)

    assert result.complete
    assert result.history[-1].data_mismatch_mean < 0.3 * result.history[0].data_mismatch_mean
