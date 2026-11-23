"""Surrogate capabilities and the workflows' requirements.

The requirement table reproduces what each workflow accepted before capabilities were
declared: the reference is the structural checks the workflows made then (an early-start
model refused by optimizations, a coarse-corrected one by predictions, any model taken by a
history match), restated here class by class; and two stricter rows switched on by the
user on 2026-11-21: a history match needs a model of the state, an optimization one of the
controls; and a third the same day, a prediction with control scenarios needs one of the
controls (the row ``prediction_scenarios``).
"""

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.core.surrogates import (
    SURROGATE_REQUIREMENTS,
    SurrogateCapabilities,
    SurrogateInputs,
    SurrogateRequirement,
    capabilities_of,
    missing_capabilities,
    refusal,
    serves,
)
from apex.domain.observations import DataKey


class _Plain:
    """A plugin declaring nothing, with nothing optional (a third party's, written before)."""

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.zeros((len(keys), 1))


class _Uncertain(_Plain):
    def predict_std(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        return np.zeros((len(keys), 1))


class _Fields(_Plain):
    def predict_fields(
        self, inputs: SurrogateInputs, names: Sequence[str], times: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        return {}


class _Early(_Plain):
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
    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return coarse, np.zeros(coarse.shape[1])

    def predict_corrected(
        self, inputs: SurrogateInputs, coarse: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        return coarse


class _Declared(_Plain):
    capabilities = SurrogateCapabilities(
        inputs=frozenset({"controls"}), outputs=frozenset({"responses"}), uncertainty=True
    )


class _StateOnly(_Plain):
    capabilities = SurrogateCapabilities(inputs=frozenset({"state"}))


class _ControlsOnly(_Plain):
    capabilities = SurrogateCapabilities(inputs=frozenset({"controls"}))


def test_the_vocabularies_are_checked() -> None:
    with pytest.raises(ValueError, match="unknown surrogate inputs"):
        SurrogateCapabilities(inputs=frozenset({"weather"}))
    with pytest.raises(ValueError, match="unknown surrogate outputs"):
        SurrogateCapabilities(inputs=frozenset({"state"}), outputs=frozenset({"maps"}))
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        SurrogateCapabilities(inputs=frozenset())
    with pytest.raises(ValueError, match="inputs cannot be empty"):
        SurrogateCapabilities(inputs=frozenset({"controls"})).without_inputs("controls")
    caps = SurrogateCapabilities(inputs=frozenset({"state", "controls"}))
    assert caps.without_inputs("controls").inputs == {"state"}
    assert caps.as_dict() == {
        "inputs": ["controls", "state"], "outputs": ["responses"], "extrapolates": False,
        "uncertainty": False, "gradients": False,
    }  # fmt: skip


def test_the_requirement_table_reproduces_what_each_workflow_accepted() -> None:
    """History matching takes any model of the state; predictions refuse a coarse-corrected
    one; optimizations need one of the controls and refuse an early-start one; none needs
    extrapolation, uncertainty or gradients. A prediction with control scenarios refuses what
    a prediction refuses and needs a model of the controls."""
    assert list(SURROGATE_REQUIREMENTS) == [
        "history_matching", "prediction", "prediction_scenarios", "optimization"
    ]  # fmt: skip
    for name, requirement in SURROGATE_REQUIREMENTS.items():
        assert requirement.workflow == name
        assert requirement.outputs == {"responses"}
        assert not requirement.extrapolates
        assert not requirement.uncertainty
        assert not requirement.gradients
    assert SURROGATE_REQUIREMENTS["history_matching"].refused_inputs == ()
    assert [i for i, _ in SURROGATE_REQUIREMENTS["history_matching"].required_inputs] == ["state"]
    assert SURROGATE_REQUIREMENTS["prediction"].required_inputs == ()
    assert [i for i, _ in SURROGATE_REQUIREMENTS["optimization"].required_inputs] == ["controls"]
    assert [i for i, _ in SURROGATE_REQUIREMENTS["prediction"].refused_inputs] == [
        "coarse_responses"
    ]
    assert [i for i, _ in SURROGATE_REQUIREMENTS["optimization"].refused_inputs] == [
        "early_responses"
    ]
    scenarios = SURROGATE_REQUIREMENTS["prediction_scenarios"]
    assert [i for i, _ in scenarios.required_inputs] == ["controls"]
    assert scenarios.refused_inputs == SURROGATE_REQUIREMENTS["prediction"].refused_inputs
    every = ["history_matching", "prediction", "prediction_scenarios", "optimization"]
    accepted = {
        _Plain: every,
        _Fields: every,
        _Uncertain: every,
        _Early: ["history_matching", "prediction", "prediction_scenarios"],
        _Coarse: ["history_matching", "optimization"],
        _StateOnly: ["history_matching", "prediction"],
        _ControlsOnly: ["prediction", "prediction_scenarios", "optimization"],
    }
    for kind, workflows in accepted.items():
        assert serves(capabilities_of(kind)) == workflows
        assert serves(capabilities_of(kind())) == workflows


def test_missing_capabilities_names_each_gap() -> None:
    caps = SurrogateCapabilities(
        inputs=frozenset({"early_responses"}), outputs=frozenset({"fields"})
    )
    requirement = SurrogateRequirement(
        "risk_assessment",
        outputs=frozenset({"responses", "fields"}),
        refused_inputs=(("early_responses", "no early runs"), ("controls", "no controls")),
        required_inputs=(("state", "use one of the state"),),
        extrapolates=True,
        gradients=True,
    )
    assert missing_capabilities(caps, requirement) == [
        "outputs: responses (not given)",
        "inputs: state (not taken: use one of the state)",
        "inputs: early_responses (no early runs)",
        "extrapolates (not supplied)",
        "gradients (not supplied)",
    ]
    full = SurrogateCapabilities(
        inputs=frozenset({"state"}), outputs=frozenset({"responses", "fields"}),
        extrapolates=True, gradients=True,
    )  # fmt: skip
    assert missing_capabilities(full, requirement) == []


def test_a_plugin_declaring_nothing_gets_capabilities_from_its_methods() -> None:
    plain = capabilities_of(_Plain)
    assert plain == SurrogateCapabilities(inputs=frozenset({"state", "controls"}))
    assert capabilities_of(_Uncertain()).uncertainty
    assert capabilities_of(_Fields).outputs == {"responses", "fields"}
    assert capabilities_of(_Early()).inputs == {"state", "controls", "early_responses"}
    assert capabilities_of(_Coarse).inputs == {"state", "controls", "coarse_responses"}
    assert not plain.extrapolates
    assert not plain.gradients
    assert not plain.uncertainty


def test_an_instance_cannot_claim_or_hide_the_methods_the_workflows_call() -> None:
    """A class's declaration is what it can do with some options; an instance's structural
    capabilities come from the instance (e.g. ``predict_std`` set by its options)."""
    assert capabilities_of(_Declared).uncertainty  # the class: as declared
    model = _Declared()
    assert not capabilities_of(model).uncertainty  # no predict_std on this instance
    model.predict_std = _Uncertain().predict_std  # type: ignore[attr-defined]
    assert capabilities_of(model).uncertainty
    assert capabilities_of(model).inputs == {"controls"}


def test_refusals_name_the_plugin_the_workflow_and_the_capability() -> None:
    assert refusal("p", capabilities_of(_Plain()), "history_matching") is None
    message = refusal("early_like", capabilities_of(_Early()), "optimization")
    assert message == (
        "the surrogate plugin 'early_like' cannot serve optimization: inputs: early_responses "
        "(each evaluation would need its own early simulation first)"
    )
    message = refusal("coarse_like", capabilities_of(_Coarse()), "prediction")
    assert message is not None
    assert message.startswith(
        "the surrogate plugin 'coarse_like' cannot serve prediction: inputs: coarse_responses"
    )
    message = refusal("state_like", capabilities_of(_StateOnly()), "optimization")
    assert message is not None
    assert message.startswith(
        "the surrogate plugin 'state_like' cannot serve optimization: inputs: controls (not taken: "
    )
    assert "use a surrogate that takes controls" in message
    message = refusal("controls_like", capabilities_of(_ControlsOnly()), "history_matching")
    assert message is not None
    assert message.startswith(
        "the surrogate plugin 'controls_like' cannot serve history matching: inputs: state "
        "(not taken: "
    )
    assert "use a surrogate that takes the state" in message
    message = refusal("state_like", capabilities_of(_StateOnly()), "prediction_scenarios")
    assert message is not None
    assert message.startswith(
        "the surrogate plugin 'state_like' cannot serve a prediction with control scenarios: "
        "inputs: controls (not taken: "
    )
    assert "use a surrogate that takes controls" in message
    assert "drop [controls]" in message
    assert refusal("state_like", capabilities_of(_StateOnly()), "prediction") is None
    assert refusal("controls_like", capabilities_of(_ControlsOnly()), "prediction_scenarios") is (
        None
    )
    with pytest.raises(KeyError):
        refusal("p", capabilities_of(_Plain()), "no_such_workflow")


class _ClaimsTooMuch(_Plain):
    capabilities = SurrogateCapabilities(
        inputs=frozenset({"early_responses", "coarse_responses"}),
        outputs=frozenset({"fields"}),
        uncertainty=True,
    )


def test_an_instances_inconsistent_declaration_is_read_from_its_structure() -> None:
    """A third party's declaration claiming protocols it lacks, or hiding ``responses``, is
    read from its structure: a model of the state alone, which optimizations refuse (they
    need the controls, the user's rule of 2026-11-21)."""
    caps = capabilities_of(_ClaimsTooMuch())
    assert caps.inputs == {"state"}
    assert caps.outputs == {"responses"}
    assert not caps.uncertainty
    assert serves(caps) == ["history_matching", "prediction"]
