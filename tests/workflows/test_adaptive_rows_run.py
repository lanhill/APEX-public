"""ES-MDA with ``adaptive_rows`` runs end to end through the composition."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest

from apex.core.errors import ConfigError
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


def test_esmda_with_adaptive_rows_lowers_the_data_mismatch(
    linear_case: Callable[..., Case],
) -> None:
    case = linear_case(
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "adaptive_rows", options = { energy = 1.0 } }',
        ensemble_size=50,
    )
    config = composition.load_config(case.config)
    assert composition.start_config(config, None).complete
    assert config.run.output_dir is not None
    history = json.loads((config.run.output_dir / "history.json").read_text())
    assert history[-1]["data_mismatch_mean"] < history[0]["data_mismatch_mean"] / 10


@pytest.mark.parametrize("method", ["lm_enrml", "gn_enrml"])
def test_iterative_methods_with_adaptive_rows_lower_the_data_mismatch(
    linear_case: Callable[..., Case], method: str
) -> None:
    """The selection follows the current iterate (review 2026-10-04)."""
    case = linear_case(
        method=f'name = "{method}"\nmax_iterations = 4\n'
        'strategy = { name = "adaptive_rows", options = { energy = 1.0 } }',
        ensemble_size=50,
    )
    config = composition.load_config(case.config)
    assert composition.start_config(config, None).complete
    assert config.run.output_dir is not None
    history = json.loads((config.run.output_dir / "history.json").read_text())
    accepted = [h for h in history if h.get("accepted", True)]
    assert accepted[-1]["data_mismatch_mean"] < history[0]["data_mismatch_mean"] / 2


def test_gies_refuses_adaptive_rows(linear_case: Callable[..., Case]) -> None:
    case = linear_case(
        method='name = "gies"\nstrategy = { name = "adaptive_rows" }',
    )
    with pytest.raises((ConfigError, ValueError), match="no RLM-MAC variant"):
        composition.build_run(composition.load_config(case.config))


def test_adaptive_rows_takes_no_localization(linear_case: Callable[..., Case]) -> None:
    case = linear_case(
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "adaptive_rows" }\nlocalization = { name = "adaptive_prior" }',
    )
    with pytest.raises(ConfigError, match="localization"):
        composition.build_run(composition.load_config(case.config))


def test_too_few_members_for_the_default_threshold_are_refused_when_validated(
    linear_case: Callable[..., Case],
) -> None:
    """A reported issue (#216): a 7-member study passed validation and failed at its first
    update; 3/sqrt(7) > 1 keeps no data, which the configuration already shows."""
    case = linear_case(
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "adaptive_rows" }',
        ensemble_size=7,
    )
    with pytest.raises(ConfigError, match="keeps no data with 7 members"):
        composition.build_run(composition.load_config(case.config))

    explicit = linear_case(
        name="explicit",
        method='name = "esmda"\noptions = { inflation = [4, 4, 4, 4] }\n'
        'strategy = { name = "adaptive_rows", options = { threshold = 0.5 } }',
        ensemble_size=7,
    )
    composition.build_run(composition.load_config(explicit.config))  # a threshold given: fine
