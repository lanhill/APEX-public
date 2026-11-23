"""Surrogate models' interface: fitted models that stand in for a simulator stack.

A surrogate is a plugin of ``core.plugins.surrogate_models`` (entry-point group
``apex.surrogate_models``). It maps what a simulator receives, a member's state ``{variable:
(size, n_members)}`` (after any parameterization), to the simulator's answers at
:class:`~apex.domain.observations.DataKey` s, ``(n_keys, n_members)``, the ensemble axis last.

What a plugin does, and what core does instead:

- the plugin fits (:meth:`SurrogateModel.fit`, the only place it may draw random numbers,
  from the generator it is given), gives its input coordinates and projection residual
  (:meth:`SurrogateModel.coordinates`), predicts deterministically
  (:meth:`SurrogateModel.predict`), and turns itself into plain arrays and back
  (:meth:`SurrogateModel.to_arrays`, ``from_arrays``: no pickle, nothing executed on load);
- core builds the validated domain from those coordinates and checks members against it
  (``algorithms/surrogates/domain.py``), gates the model on held-out full runs
  (``algorithms/surrogates/gate.py``) and writes and reads its files
  (``storage/surrogate_files.py``). A plugin cannot widen its own domain or loosen its gate.

Optional capabilities, read with ``getattr`` as the simulators' are:

- ``predict_fields`` (:class:`FieldSurrogate`): state fields at given times;
- ``predict_std`` (:class:`UncertainSurrogate`): a predictive standard deviation, e.g. a
  Gaussian process's;
- ``predict_corrected`` (:class:`CoarseCorrectedSurrogate`): a correction of a coarse
  simulation's forecast, from the member's inputs and that forecast;
- ``trains_on_coarse``, a ``bool`` (``False`` by default): the plugin learns from a
  ``[coarse]`` stack's runs in training only (``SnapshotSet.coarse_outputs`` and the
  coarse-only members decision 13; e.g. pretraining on them) and predicts without
  a coarse forecast, so history matching runs no coarse stack for it;
- ``uses_calibration_runs``, a ``bool`` of the instance (``False`` by default): the plugin
  calibrates its spread on runs of their own (``SnapshotSet.calibration_runs()``, ``[design]
  calibration``, drawn automatically when unset), e.g. a Laplace posterior's two scales, and
  on its validation share without them;
- ``uses_gradients``, a ``bool`` (``False`` by default): the plugin learns from the runs'
  gradients (``SnapshotSet.gradients``, ``(n_keys, n_inputs, n_runs)``), so its build runs
  every member with its sensitivities (``SensitivitySimulator.run_with_sensitivities``,
  through :class:`apex.simulation.sensitivity.SensitivityRun`); a build is refused where the
  stack cannot give them (no sensitivities, parameterizations, state fields, a coarse stack,
  a plan other than ``prior``, ``lhs`` and ``file``);
- ``unstructured_fields``, a ``bool`` (``False`` by default): the plugin takes state fields
  (``SnapshotSet.fields``) on an unstructured mesh, each field over the mesh's cells in its
  case-state reader's order (decision 5); without it a build with ``[case] fields``
  needs a reader that always gives a structured grid (``structured_mesh``), on whose whole
  grid the fields are, in Eclipse (F) order with inactive cells 0. A plugin with
  ``unstructured_fields`` on a reader that gives a structured grid still gets that whole
  grid, padded with zeros on the inactive cells;
- ``uses_connectivity``, a ``bool`` (``False`` by default): the plugin takes the runs' cells
  and the faces between them through a method
  ``use_connectivity(mesh: CaseMesh, graph: CellGraph)``, which a build (with or without
  ``[case] fields``) calls before ``fit`` when its simulator's case-state reader has the
  capability ``connectivity`` (the graph of the first run read; the cells the reader gives,
  ``apex.domain.cell_graph``); without the capability it is not called and the plugin keeps
  to its own geometry. An instance may refine the class's word (``False`` for options that
  take no graph): a build whose plugin instance does not take it records none. A refit in
  history matching hands the new model the one the active model was fitted on, when the
  plugin gives it (``fitted_connectivity()``);
- ``requires``, a ``tuple[str, ...]`` of simulator capabilities the plugin needs to be
  trained (the intrusive reduced models); none by default;
- ``version``, a ``str`` recorded in the manifest;
- ``capabilities``, a :class:`SurrogateCapabilities`: what the plugin
  takes and gives, which each workflow's :class:`SurrogateRequirement` is checked against.
  The class's value is what the plugin can do with some options; an instance may narrow it
  for its own options (``self.capabilities = ...``). Read with :func:`capabilities_of`,
  which derives one for a plugin that declares none.

A plugin's options declare what their strings name (``core.options``), as every
plugin's.

Design plans are plugins of ``core.plugins.surrogate_designs`` (entry-point
group ``apex.surrogate_designs``): core's own (:class:`BuiltinDesign`) and adaptive ones
(:class:`AdaptiveDesign`) that choose runs from a fitted surrogate.
"""

import functools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import DatedStates
from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet

logger = logging.getLogger(__name__)

#: What a model is given: ``{variable: (size, n_members)}``.
SurrogateInputs = Mapping[str, NDArray[np.float64]]

#: What :meth:`SurrogateModel.to_arrays` returns: the plugin's own manifest (JSON values only:
#: strings, finite numbers, booleans, ``None``, lists and string-keyed mappings of them) and
#: its arrays (numbers or text, no objects).
SurrogateParts = tuple[Mapping[str, object], Mapping[str, NDArray[np.generic]]]


@runtime_checkable
class SurrogateModel(Protocol):
    """A fitted model that stands in for one simulator stack on one case.

    Attributes
    ----------
    keys
        The keys it answers, once fitted; any other is outside its domain.
    """

    keys: tuple[DataKey, ...]

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        """Fit to the training runs; all randomness comes from ``rng``.

        The held-out runs are never given to it.

        Raises
        ------
        ValueError
            If the snapshots do not suit the model (too few runs, no variation).
        """
        ...

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the inputs' coordinates and their projection residual.

        Coordinates are ``(n_coordinates, n_members)``: what the model's regression sees
        (the inputs themselves, or their reduced coordinates). The residual is
        ``(n_members,)``, how far each input lies off the reduced subspace (zeros without a
        reduction). Core bounds both to the validated domain.
        """
        ...

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the values at ``keys``, ``(len(keys), n_members)``.

        Deterministic: no generator, nothing drawn; equal inputs give equal outputs.
        """
        ...

    def to_arrays(self) -> SurrogateParts:
        """Return the fitted model as its own manifest and arrays (pure: no I/O)."""
        ...

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        """Return the fitted model :meth:`to_arrays` described (pure: no I/O)."""
        ...


@runtime_checkable
class FieldSurrogate(Protocol):
    """A surrogate that also predicts state fields (optional capability)."""

    def predict_fields(
        self, inputs: SurrogateInputs, names: Sequence[str], times: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{name: (n_cells, len(times), n_members)}``; deterministic."""
        ...


@runtime_checkable
class BlockStateSurrogate(Protocol):
    """A surrogate that gives one member's block states (optional).

    What a model-error model reads of a run (``core.simulation.BlockStates``), from the
    surrogate's own prediction: the static arrays, and the pressures, saturations,
    densities and wells' bottom-hole pressures at report dates. With it a run may combine
    ``[surrogate]`` and ``[model_error]``: the model-error residuals of a member the
    surrogate forecasts are computed from these states. ``predict_block_states`` must agree
    with ``predict`` for the same member (one prediction, one source), and give the values
    the deck's files would hold, in their units and cell order (the contract of
    ``core.simulation.DatedStates``); a plugin's tests check it on a deck's run with
    ``apex.testing.assert_block_states_conform``.
    """

    def predict_block_states(self, inputs: SurrogateInputs, dates: Sequence[str]) -> DatedStates:
        """Return one member's states (``inputs`` of one column) at the ISO ``dates``."""
        ...


@runtime_checkable
class UncertainSurrogate(Protocol):
    """A surrogate that gives a predictive standard deviation (optional capability)."""

    def predict_std(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the standard deviation at ``keys``, ``(len(keys), n_members)``, >= 0."""
        ...


class BuiltinDesign:
    """Core's own design plans (``prior``, ``lhs``, ``file``, ``run``).

    Registered in ``surrogate_designs`` under their names so that every plan is found in one
    place; the build draws them in one go (``services.surrogates.draw_design``).
    """

    adaptive = False


@runtime_checkable
class AdaptiveDesign(Protocol):
    """A design plan that chooses training runs where a fitted surrogate is uncertain.

    The build draws the held-out runs and :meth:`initial_runs` training runs
    from the prior, then a pool of :meth:`candidate_runs` prior draws; it runs the members,
    fits the surrogate on the training runs so far and asks :meth:`choose` for the next batch
    of :meth:`batch_runs` candidates, until ``[design] training`` runs are made. Choosing is
    deterministic (no generator): equal inputs choose alike.

    An optional ``uses``, a ``tuple[str, ...]`` of the surrogate's capabilities the plan
    calls (e.g. ``("predict_std",)``; not ``requires``, which names a plugin's modules), lets
    a build refuse an unsuitable surrogate before any run (read with ``getattr``).
    """

    adaptive: bool

    def initial_runs(self, training: int) -> int:
        """Return how many of the ``training`` runs are drawn before any choice (>= 2)."""
        ...

    def batch_runs(self) -> int:
        """Return how many runs each choice adds (the last batch may be smaller)."""
        ...

    def candidate_runs(self) -> int:
        """Return the size of the pool of prior draws the runs are chosen from."""
        ...

    def choose(
        self,
        model: SurrogateModel,
        chosen: Mapping[str, NDArray[np.float64]],
        candidates: Mapping[str, NDArray[np.float64]],
        count: int,
        *,
        keys: Sequence[DataKey],
        variances: NDArray[np.float64] | None,
    ) -> list[int]:
        """Return ``count`` distinct columns of ``candidates`` to run next.

        Parameters
        ----------
        model
            The surrogate fitted on the training runs so far.
        chosen, candidates
            The training runs so far and the candidates not chosen yet, as the surrogate's
            inputs ``{variable: (size, n)}``.
        keys
            The model's keys.
        variances
            ``(len(keys),)``, the observation errors' variances (``None``: unknown).
        """
        ...


@runtime_checkable
class CoarseCorrectedSurrogate(Protocol):
    """A surrogate that corrects a coarse simulation's forecast.

    Its training runs carry a coarse stack's outputs at the same keys
    (:attr:`~apex.domain.surrogate.SnapshotSet.coarse_outputs`); it predicts from a member's
    inputs and its coarse forecast. Its :meth:`SurrogateModel.predict` refuses, naming the
    coarse forecast it lacks. Core bounds its domain in both: the inputs' coordinates and
    :meth:`coarse_coordinates`, the latter's residual as one more coordinate.
    """

    def coarse_coordinates(
        self, coarse: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the coarse forecast's coordinates ``(d_c, n)`` and residual ``(n,)``.

        ``coarse`` is ``(n_keys, n)`` at the model's keys.
        """
        ...

    def predict_corrected(
        self,
        inputs: Mapping[str, NDArray[np.float64]],
        coarse: NDArray[np.float64],
        keys: Sequence[DataKey],
    ) -> NDArray[np.float64]:
        """Return the corrected values at ``keys``, ``(len(keys), n)``; deterministic.

        ``coarse`` ``(n_model_keys, n)``: the members' coarse forecasts at the model's keys.
        """
        ...


@runtime_checkable
class EarlyStartSurrogate(Protocol):
    """A surrogate that continues a member's own early data.

    Each member is simulated up to :attr:`early_until` (a report date: ``MemberInput.stop_at``,
    on a simulator with ``stops_at_report_dates``), and the model predicts the rest of its
    keys from that member's values at :meth:`early_keys` (its keys up to the date), returned
    as they are. A build needs no extra runs: a full run's early data is its values at those
    keys. Its :meth:`SurrogateModel.predict` refuses, naming the early data it lacks. Core
    bounds its domain in the inputs' coordinates and :meth:`early_coordinates`, the latter's
    residual as one more coordinate (as a coarse-corrected model's coarse forecast).
    """

    #: The report date (ISO) members are simulated up to.
    early_until: str

    def early_keys(self) -> tuple[DataKey, ...]:
        """Return the model's keys up to :attr:`early_until`, in its keys' order."""
        ...

    def early_coordinates(
        self, early: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the early data's coordinates ``(d_e, n)`` and residual ``(n,)``.

        ``early`` is ``(len(early_keys()), n)``.
        """
        ...

    def predict_from_early(
        self,
        inputs: Mapping[str, NDArray[np.float64]],
        early: NDArray[np.float64],
        keys: Sequence[DataKey],
    ) -> NDArray[np.float64]:
        """Return the values at ``keys``, ``(len(keys), n)``; deterministic.

        ``early`` ``(len(early_keys()), n)``: the members' simulated values up to the date,
        which a key among the early keys gets back unchanged.
        """
        ...


# ============================================== capabilities

#: What a surrogate may take: the member's ``state`` (after its parameterizations), the
#: wells' ``controls`` (an optimization's, or a prediction's scenario), the member's own
#: ``early_responses`` (its simulated responses up to a date, :class:`EarlyStartSurrogate`)
#: and a ``coarse_responses`` forecast (:class:`CoarseCorrectedSurrogate`).
SURROGATE_INPUTS = frozenset({"state", "controls", "early_responses", "coarse_responses"})

#: What a surrogate may give: ``responses`` at data keys (every model's ``predict``), and
#: state ``fields`` over time (:class:`FieldSurrogate`).
SURROGATE_OUTPUTS = frozenset({"responses", "fields"})


@dataclass(frozen=True)
class SurrogateCapabilities:
    """What a surrogate plugin takes and gives.

    Attributes
    ----------
    inputs
        A non-empty subset of :data:`SURROGATE_INPUTS`.
    outputs
        A non-empty subset of :data:`SURROGATE_OUTPUTS`.
    extrapolates
        Whether it predicts beyond its training runs' horizon in time.
    uncertainty
        Whether it gives a predictive standard deviation (``predict_std``).
    gradients
        Whether it supplies the derivatives of its outputs with respect to its inputs.
    """

    inputs: frozenset[str]
    outputs: frozenset[str] = frozenset({"responses"})
    extrapolates: bool = False
    uncertainty: bool = False
    gradients: bool = False

    def __post_init__(self) -> None:
        """Check the words against the vocabularies.

        Raises
        ------
        ValueError
            If ``inputs`` or ``outputs`` is empty or names an unknown word.
        """
        for name, given, known in (
            ("inputs", self.inputs, SURROGATE_INPUTS),
            ("outputs", self.outputs, SURROGATE_OUTPUTS),
        ):
            words = frozenset(given)
            object.__setattr__(self, name, words)  # a list or set given: kept as frozen
            if not words:
                raise ValueError(f"a surrogate's {name} cannot be empty")
            unknown = sorted(words - known)
            if unknown:
                raise ValueError(f"unknown surrogate {name} {unknown}: known are {sorted(known)}")

    def without_inputs(self, *names: str) -> "SurrogateCapabilities":
        """Return a copy without the inputs ``names``: an instance narrowing its class's.

        Raises
        ------
        ValueError
            If no input would be left.
        """
        return replace(self, inputs=self.inputs - frozenset(names))

    def as_dict(self) -> dict[str, object]:
        """Return the capabilities as JSON values (sorted lists for the sets)."""
        return {
            "inputs": sorted(self.inputs),
            "outputs": sorted(self.outputs),
            "extrapolates": self.extrapolates,
            "uncertainty": self.uncertainty,
            "gradients": self.gradients,
        }


@dataclass(frozen=True)
class SurrogateRequirement:
    """What a workflow needs of a surrogate.

    Attributes
    ----------
    workflow
        The use's name: a workflow's (``history_matching``, ``prediction``,
        ``optimization``, ...), or a workflow's under a condition that needs more of a model
        (``prediction_scenarios``: a prediction with control scenarios).
    outputs
        The outputs it needs, each.
    refused_inputs
        ``(input, why)`` pairs: inputs it cannot give a model, with the reason.
    required_inputs
        ``(input, why)`` pairs: inputs a model must take, with the reason and what to use
        instead.
    extrapolates, uncertainty, gradients
        The capabilities it needs (none by default).
    label
        How a refusal names it (default: ``workflow`` with spaces for underscores).
    """

    workflow: str
    outputs: frozenset[str] = frozenset({"responses"})
    refused_inputs: tuple[tuple[str, str], ...] = field(default=())
    required_inputs: tuple[tuple[str, str], ...] = field(default=())
    extrapolates: bool = False
    uncertainty: bool = False
    gradients: bool = False
    label: str = ""

    def name(self) -> str:
        """Return how a refusal names it: its ``label``, else its workflow's name."""
        return self.label or self.workflow.replace("_", " ")


#: Each workflow's requirement. They reproduce what each workflow
#: accepted before capabilities were declared, with two stricter rows (the user,
#: 2026-11-21): a history match needs a model of the member's ``state`` (the variables it
#: updates) and takes any such model; a prediction refuses a coarse-corrected one (its
#: coarse stack would have to run to the horizon; times beyond a model's keys are refused
#: by the keys themselves, so it does not need ``extrapolates``); an optimization needs a
#: model of the wells' ``controls`` (the variables it changes) and refuses an early-start
#: one. A prediction with control scenarios (``[controls]``) has a row of its own,
#: ``prediction_scenarios``, which also needs a model of the ``controls`` (the user,
#: 2026-11-21): a model that does not take them would forecast as if the controls were its
#: build's. A workflow added later (e.g. risk assessment) adds its row.
SURROGATE_REQUIREMENTS: Mapping[str, SurrogateRequirement] = MappingProxyType(
    {
        "history_matching": SurrogateRequirement(
            "history_matching",
            required_inputs=(
                (
                    "state",
                    "a history match updates each member's state, and a model that does not "
                    "take it does not answer from the state it updates; use a surrogate that "
                    "takes the state",
                ),
            ),
        ),
        "prediction": SurrogateRequirement(
            "prediction",
            refused_inputs=(
                (
                    "coarse_responses",
                    "a prediction would have to run its coarse stack to the horizon",
                ),
            ),
        ),
        "prediction_scenarios": SurrogateRequirement(
            "prediction_scenarios",
            label="a prediction with control scenarios",
            required_inputs=(
                (
                    "controls",
                    "a prediction with control scenarios gives every member the scenario's "
                    "controls, and a model that does not take them would forecast as if the "
                    "controls were its build's; use a surrogate that takes controls, built "
                    "over the scenario's controls, or drop [controls], or forecast with the "
                    "simulator (no [surrogate])",
                ),
            ),
            refused_inputs=(
                (
                    "coarse_responses",
                    "a prediction would have to run its coarse stack to the horizon",
                ),
            ),
        ),
        "optimization": SurrogateRequirement(
            "optimization",
            required_inputs=(
                (
                    "controls",
                    "an optimization changes the wells' controls, and a model that does not "
                    "take them does not answer from the controls it changes; use a surrogate "
                    "that takes controls, built over the optimization's controls",
                ),
            ),
            refused_inputs=(
                (
                    "early_responses",
                    "each evaluation would need its own early simulation first",
                ),
            ),
        ),
    }
)


def missing_capabilities(
    capabilities: SurrogateCapabilities, requirement: SurrogateRequirement
) -> list[str]:
    """Return what ``capabilities`` lack for ``requirement``, one phrase each (empty: none).

    An output the workflow needs that the model does not give, an input the workflow needs
    that the model does not take (with why and what to use instead), an input the model
    takes that the workflow cannot give (with why), and each capability the workflow needs.
    """
    found = [
        f"outputs: {name} (not given)"
        for name in sorted(requirement.outputs - capabilities.outputs)
    ]
    found += [
        f"inputs: {name} (not taken: {why})"
        for name, why in requirement.required_inputs
        if name not in capabilities.inputs
    ]
    found += [
        f"inputs: {name} ({why})"
        for name, why in requirement.refused_inputs
        if name in capabilities.inputs
    ]
    found += [
        f"{name} (not supplied)"
        for name in ("extrapolates", "uncertainty", "gradients")
        if getattr(requirement, name) and not getattr(capabilities, name)
    ]
    return found


def refusal(plugin: str, capabilities: SurrogateCapabilities, workflow: str) -> str | None:
    """Return why ``workflow`` refuses the plugin ``plugin``, or ``None`` if it serves.

    ``workflow`` is a key of :data:`SURROGATE_REQUIREMENTS`: a workflow, or a workflow under
    a condition (``prediction_scenarios``).

    Raises
    ------
    KeyError
        If ``workflow`` has no requirement (:data:`SURROGATE_REQUIREMENTS`).
    """
    requirement = SURROGATE_REQUIREMENTS[workflow]
    missing = missing_capabilities(capabilities, requirement)
    if not missing:
        return None
    return f"the surrogate plugin {plugin!r} cannot serve {requirement.name()}: " + "; ".join(
        missing
    )


def serves(capabilities: SurrogateCapabilities) -> list[str]:
    """Return the uses whose requirement ``capabilities`` meet, in the table's order.

    The uses are the table's keys: workflows, and a workflow under a condition that needs
    more (``prediction_scenarios``).
    """
    return [
        name
        for name, requirement in SURROGATE_REQUIREMENTS.items()
        if not missing_capabilities(capabilities, requirement)
    ]


def _has(model: object, *names: str) -> bool:
    """Whether ``model`` (a class or an instance) has a callable of every name."""
    return all(callable(getattr(model, name, None)) for name in names)


def _early_start(model: object) -> bool:
    """Whether ``model`` continues each member's early run (an instance by its protocol)."""
    if isinstance(model, type):
        return _has(model, "early_keys", "early_coordinates", "predict_from_early")
    return isinstance(model, EarlyStartSurrogate)


def _coarse_corrected(model: object) -> bool:
    """Whether ``model`` corrects a coarse forecast (an instance by its protocol)."""
    if isinstance(model, type):
        return _has(model, "coarse_coordinates", "predict_corrected")
    return isinstance(model, CoarseCorrectedSurrogate)


@functools.cache
def _derived_note(plugin: str) -> None:
    """Log once per plugin type that its capabilities were derived (it declares none)."""
    logger.debug("surrogate plugin %s declares no capabilities: derived from its methods", plugin)


def capabilities_of(model: object) -> SurrogateCapabilities:
    """Return the capabilities of a surrogate plugin's class or of a model (an instance).

    A declared ``capabilities`` is taken as given, except that an instance's structural
    capabilities are read from the instance itself: ``uncertainty`` is whether it has a
    ``predict_std``, ``fields`` whether it has a ``predict_fields`` (``responses`` always:
    every model predicts at keys), and ``early_responses``/``coarse_responses`` whether it
    is an :class:`EarlyStartSurrogate` or a :class:`CoarseCorrectedSurrogate` (an instance
    whose declared inputs would then be empty takes the ``state``). The workflows call
    those methods, so a declaration can neither claim nor hide them (several plugins set
    ``predict_std`` per instance, from their options).

    A plugin that declares none (one written before declarations: the interface stays
    stable) gets one derived from its methods: ``inputs`` ``{state, controls}``, plus
    ``early_responses`` or ``coarse_responses`` as above; ``outputs`` ``responses``, plus
    ``fields`` with ``predict_fields``; ``uncertainty`` with ``predict_std``; no
    ``extrapolates`` or ``gradients``.
    """
    declared = getattr(model, "capabilities", None)
    early, coarse = _early_start(model), _coarse_corrected(model)
    if not isinstance(declared, SurrogateCapabilities):
        kind = model if isinstance(model, type) else type(model)
        _derived_note(f"{kind.__module__}.{kind.__qualname__}")
        declared = SurrogateCapabilities(
            inputs=frozenset({"state", "controls"}),
            outputs=frozenset({"responses"}),
            uncertainty=_has(model, "predict_std"),
        )
        if _has(model, "predict_fields"):
            declared = replace(declared, outputs=declared.outputs | {"fields"})
    elif isinstance(model, type):
        return declared
    extra = {"early_responses"} if early else set()
    extra |= {"coarse_responses"} if coarse else set()
    if isinstance(model, type):
        return replace(declared, inputs=declared.inputs | extra) if extra else declared
    # An instance: every structural word from the instance itself (every model predicts at
    # keys, so it gives responses), the declaration adding only what structure cannot tell.
    inputs = (declared.inputs - {"early_responses", "coarse_responses"}) | extra
    outputs = {"responses"} | ({"fields"} if _has(model, "predict_fields") else set())
    return replace(
        declared,
        inputs=frozenset(inputs or {"state"}),
        outputs=frozenset(outputs),
        uncertainty=_has(model, "predict_std"),
    )
