"""The workflows refuse a surrogate by its capabilities, and the plugin
listing shows them.

Refusals through each workflow's service, with test-only models: an optimization refuses
an early-start model (also ``tests/unit/algorithms/test_surrogate_early_start.py``) and a
model of the state alone (the user's rule of 2026-11-21), a prediction a coarse-corrected
one, and a history match a model of the controls alone (the same day's rule).
"""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.errors import ConfigError
from apex.core.plugins import surrogate_models
from apex.core.surrogates import (
    SurrogateCapabilities,
    SurrogateInputs,
    capabilities_of,
    serves,
)
from apex.domain.observations import DataKey
from apex.services import surrogates
from apex.services.plugins import plugin_info


class _Plain:
    """A model with nothing optional, declaring nothing."""

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.zeros((len(keys), 1))


class _Early(_Plain):
    """An early-start model (``core.surrogates.EarlyStartSurrogate``)."""

    early_until = "2001-01-01"

    def early_keys(self) -> tuple[DataKey, ...]:
        return ()

    def early_coordinates(
        self, early: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return early, np.zeros(early.shape[1])

    def predict_from_early(
        self, inputs: SurrogateInputs, early: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return np.zeros((len(keys), early.shape[1]))


class _Coarse(_Plain):
    """A coarse-corrected model (``core.surrogates.CoarseCorrectedSurrogate``)."""

    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return coarse, np.zeros(coarse.shape[1])

    def predict_corrected(
        self, inputs: SurrogateInputs, coarse: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return coarse


class _StateOnly(_Plain):
    """A model of the member's state alone."""

    capabilities = SurrogateCapabilities(inputs=frozenset({"state"}))


class _ControlsOnly(_Plain):
    """A model of the wells' controls alone."""

    capabilities = SurrogateCapabilities(inputs=frozenset({"controls"}))


def test_a_prediction_refuses_a_coarse_corrected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = SimpleNamespace(model=_Coarse(), manifest=SimpleNamespace(plugin="coarse_like"))
    monkeypatch.setattr(surrogates, "active_model", lambda path: loaded)
    section = SimpleNamespace(model_file="model.json")
    with pytest.raises(ConfigError) as error:
        surrogates.prediction_surrogate(cast(Any, section), *(cast(Any, None),) * 5, "serial")
    message = str(error.value)
    assert "plugin 'coarse_like' cannot serve prediction" in message
    assert "inputs: coarse_responses" in message
    assert "predictions with it are not supported" in message


def test_an_optimization_refuses_an_early_start_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = SimpleNamespace(model=_Early(), manifest=SimpleNamespace(plugin="early_like"))
    monkeypatch.setattr(surrogates, "load_surrogate", lambda path: loaded)
    config = SimpleNamespace(surrogate=SimpleNamespace(model_file="m.json"), simulator=object())
    with pytest.raises(ConfigError) as error:
        surrogates.optimization_surrogate(cast(Any, config), cast(Any, None))
    message = str(error.value)
    assert "plugin 'early_like' cannot serve optimization: inputs: early_responses" in message
    assert "optimizations cannot use it" in message


def test_an_optimization_refuses_a_model_of_the_state_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(model=_StateOnly(), manifest=SimpleNamespace(plugin="state_like"))
    monkeypatch.setattr(surrogates, "load_surrogate", lambda path: loaded)
    config = SimpleNamespace(surrogate=SimpleNamespace(model_file="m.json"), simulator=object())
    with pytest.raises(ConfigError) as error:
        surrogates.optimization_surrogate(cast(Any, config), cast(Any, None))
    message = str(error.value)
    assert "plugin 'state_like' cannot serve optimization: inputs: controls (not taken" in message
    assert "use a surrogate that takes controls" in message
    assert "optimizations cannot use it" in message


def test_a_history_match_refuses_a_model_of_the_controls_alone() -> None:
    model = SimpleNamespace(
        model=_ControlsOnly(), manifest=SimpleNamespace(plugin="controls_like"),
        path=Path("m.json"),
    )  # fmt: skip
    config = SimpleNamespace(surrogate=SimpleNamespace())
    with pytest.raises(ConfigError) as error:
        surrogates.refuse_model(
            cast(Any, config), cast(Any, model), cast(Any, None), cast(Any, None)
        )
    message = str(error.value)
    assert "plugin 'controls_like' cannot serve history matching: inputs: state (not taken" in (
        message
    )
    assert "use a surrogate that takes the state" in message


def test_the_listing_shows_each_surrogates_capabilities_and_workflows() -> None:
    surrogate_models.register("test_capabilities_early")(_Early)
    try:
        info = plugin_info("surrogate_models", "test_capabilities_early")
    finally:
        surrogate_models.remove("test_capabilities_early")
    assert info.capabilities == capabilities_of(_Early).as_dict()
    assert info.capabilities["inputs"] == ["controls", "early_responses", "state"]
    assert info.serves == ["history_matching", "prediction"]
    for name in surrogate_models.names():
        listed = plugin_info("surrogate_models", name)
        assert listed.serves == serves(capabilities_of(surrogate_models.factory(name)))
    other = plugin_info("executors", "serial")
    assert other.capabilities is None
    assert other.serves is None


def test_the_cli_listing_prints_them(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from apex.cli.main import app

    surrogate_models.register("test_capabilities_early")(_Early)
    try:
        result = CliRunner().invoke(app, ["plugins"])
    finally:
        surrogate_models.remove("test_capabilities_early")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    at = next(i for i, line in enumerate(lines) if "test_capabilities_early" in line)
    following = "\n".join(lines[at + 1 : at + 4])
    assert "capabilities: inputs controls+early_responses+state; outputs responses" in following
    assert "serves: history_matching, prediction" in following
