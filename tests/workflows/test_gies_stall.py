"""GIES stops when its update stalls (the numerics review of 2026-10-04).

A worse trial within the tolerance is rejected, not converged; ``max_stalled_rejections``
of them in a row stop the run, so a starved update does not spend ``max_iterations``
forecasts. A test damping scheme reports every trial as such a rejection.
"""

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pytest

from apex.algorithms.data_assimilation.protocols import DampingDecision
from apex.algorithms.data_assimilation.registries import damping_schemes
from apex.services import composition


class Case(Protocol):
    """What the ``linear_case`` fixture (conftest.py) returns."""

    directory: Path
    config: Path


@dataclass(frozen=True)
class AlwaysStalled:
    """Every trial: worse, within the tolerance (a stalled update)."""

    initial: float = 1.0
    factor: float = 2.0
    max_stalled_rejections: int = 3

    def decide(
        self,
        damping: float,
        previous_data_mismatch: float,
        data_mismatch: float,
        *,
        previous_spread: float | None = None,
        spread: float | None = None,
    ) -> DampingDecision:
        return DampingDecision(False, False, damping * self.factor, stalled=True)


@pytest.fixture
def stalled_scheme() -> Iterator[None]:
    damping_schemes.register("test_always_stalled")(AlwaysStalled)
    try:
        yield
    finally:
        damping_schemes._factories.pop("test_always_stalled")


def test_gies_stops_after_consecutive_stalled_trials(
    linear_case: Callable[..., Case], stalled_scheme: None
) -> None:
    case = linear_case(
        method='name = "gies"\nstrategy = { name = "rlmmac", options = { energy = 1.0 } }\n'
        'damping = { name = "test_always_stalled" }\nmax_iterations = 10',
        ensemble_size=20,
    )
    config = composition.load_config(case.config)
    result = composition.start_config(config, None)
    assert result.complete
    run_dir = config.run.output_dir
    assert run_dir is not None
    history = json.loads((run_dir / "history.json").read_text())
    trials = [h for h in history if h["iteration"] > 0]
    assert len(trials) == 3  # three stalled rejections, not ten
    assert not any(h["accepted"] for h in trials)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "stalled" for e in events)


def test_lm_enrml_stops_after_its_schemes_stalled_trials(
    linear_case: Callable[..., Case], stalled_scheme: None
) -> None:
    """LM-EnRML's loop (with its inner-iteration count) honours the scheme's stall limit."""
    case = linear_case(
        method='name = "lm_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
        'damping = { name = "test_always_stalled", options = { max_stalled_rejections = 2 } }\n'
        "max_iterations = 10",
        ensemble_size=20,
    )
    config = composition.load_config(case.config)
    assert composition.start_config(config, None).complete
    assert config.run.output_dir is not None
    history = json.loads((config.run.output_dir / "history.json").read_text())
    assert len([h for h in history if h["iteration"] > 0]) == 2


def test_gn_enrml_reads_its_own_stall_limit(
    linear_case: Callable[..., Case], monkeypatch: pytest.MonkeyPatch
) -> None:
    """GN-EnRML's limit is its step control's, not the unused damping scheme's (review)."""
    from apex.algorithms.data_assimilation.gn_enrml import GNStepSchedule

    def stalled(
        self: GNStepSchedule, gamma: float, *args: object, **kwargs: object
    ) -> DampingDecision:
        return DampingDecision(False, False, gamma / 2, stalled=True)

    monkeypatch.setattr(GNStepSchedule, "decide", stalled)
    case = linear_case(
        method='name = "gn_enrml"\nstrategy = { name = "approx", options = { energy = 1.0 } }\n'
        "schedule = { max_stalled_rejections = 2 }\nmax_iterations = 10",
        ensemble_size=20,
    )
    config = composition.load_config(case.config)
    assert composition.start_config(config, None).complete
    assert config.run.output_dir is not None
    history = json.loads((config.run.output_dir / "history.json").read_text())
    assert len([h for h in history if h["iteration"] > 0]) == 2
    events = [
        json.loads(x) for x in (config.run.output_dir / "events.jsonl").read_text().splitlines()
    ]
    assert any(e.get("event") == "stalled" for e in events)
