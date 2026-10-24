"""A simulator's own field data through the forecast stack.

The generated linear case's simulator is wrapped so that it declares a field
(:class:`~apex.core.simulation.FieldResponses`); ``[geophysics]`` is a stand-in recording
what it is asked to check. Checked before any member runs: the declared field's indices,
the keys left to ``[geophysics]``, a name both claim, and a stack built without checks.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

import apex.services.composition as composition
from apex.core.errors import ConfigError
from apex.core.plugins import simulators
from apex.domain.observations import DataKey, ObservationSet
from apex.testing import CaseFactory

FIELD = "vertical_displacement"


class _Declaring:
    """The linear simulator, declaring a field of 3 values."""

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 (any simulator)
        self.inner = inner

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 (delegates)
        return getattr(self.inner, name)

    def field_datatypes(self) -> Mapping[str, int]:
        return {FIELD: 3}


class _Geophysics:
    """A [geophysics] stand-in: owns ``avo_10`` and records what it checks."""

    state_variables: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.checked: list[DataKey] = []

    def owns(self, key: DataKey) -> bool:
        return key.index is not None and key.datatype in self.names

    names = frozenset({"avo_10"})

    def check_observations(self, keys: Sequence[DataKey]) -> None:
        self.checked = list(keys)

    def warnings(self) -> list[str]:
        return []


def _observations(*fields: tuple[str, int]) -> ObservationSet:
    keys = [DataKey("t0", "y")]
    for datatype, size in fields:
        keys += [DataKey("2020-01-31", datatype, i) for i in range(size)]
    return ObservationSet(keys, np.zeros(len(keys)), np.ones(len(keys)))


@pytest.fixture
def declaring(monkeypatch: pytest.MonkeyPatch) -> None:
    create = simulators.create
    monkeypatch.setattr(simulators, "create", lambda name, **kw: _Declaring(create(name, **kw)))


def _with_geophysics(monkeypatch: pytest.MonkeyPatch, model: _Geophysics) -> None:
    monkeypatch.setattr(composition, "build_geophysics", lambda config: model)
    monkeypatch.setattr(composition, "GeophysicsSimulator", lambda simulator, model: simulator)
    monkeypatch.setattr(composition, "geophysics_restart_warnings", lambda config: [])
    monkeypatch.setattr(composition, "geophysics_deck_warnings", lambda config, model: [])


@pytest.mark.usefixtures("declaring")
def test_a_declared_field_observed_in_part_is_refused_before_any_run(
    linear_case: CaseFactory,
) -> None:
    config = composition.load_config(linear_case().config)
    composition.forecast_stack(config, observations=_observations((FIELD, 3)))  # whole: fine
    with pytest.raises(ConfigError, match=f"{FIELD}@2020-01-31 is observed at 2 indices"):
        composition.forecast_stack(config, observations=_observations((FIELD, 2)))


@pytest.mark.usefixtures("declaring")
def test_geophysics_checks_only_the_fields_the_simulator_leaves(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Geophysics()
    _with_geophysics(monkeypatch, model)
    config = composition.load_config(linear_case().config)
    composition.forecast_stack(
        config, observations=_observations((FIELD, 3), ("avo_10", 2), ("other", 1))
    )

    assert {k.datatype for k in model.checked} == {"avo_10", "other"}  # "other": refused there


@pytest.mark.usefixtures("declaring")
def test_a_field_both_claim_is_refused(
    linear_case: CaseFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Geophysics()
    model.names = frozenset({FIELD})
    _with_geophysics(monkeypatch, model)
    config = composition.load_config(linear_case().config)
    with pytest.raises(ConfigError, match=f"{FIELD} are computed by the simulator and named by"):
        composition.forecast_stack(config, observations=_observations())


@pytest.mark.usefixtures("declaring")
def test_without_checks_the_stack_needs_no_observations(linear_case: CaseFactory) -> None:
    config = composition.load_config(linear_case().config)
    stack = composition.forecast_stack(config, checks=False)
    assert stack.simulator.field_datatypes() == {FIELD: 3}  # type: ignore[attr-defined]
