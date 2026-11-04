"""Core's early-start surrogate path, with a test-only model.

The model continues each member's early series with its last early value (persistence), so
every answer is known by hand: the build's gate on held-out runs whose series are constant
after the early date passes exactly; the domain bounds the early data, not the inputs;
the surrogate simulator predicts a member only from the early data it was handed; and the
early data are the runs' own values (no extra run).
"""

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from apex.algorithms.surrogates.domain import model_coordinates, refuse_outside
from apex.core.errors import ConfigError, SurrogateDomainError, WorkflowError
from apex.core.simulation import MemberInput
from apex.core.surrogates import EarlyStartSurrogate, SurrogateInputs, SurrogateParts
from apex.domain.observations import DataKey
from apex.domain.state import StateLayout, StateVariable
from apex.domain.surrogate import GateTolerances, SnapshotProvenance, SnapshotSet, TrainingDomain
from apex.services import surrogates
from apex.simulation.surrogate import SurrogateSimulator
from apex.workflows.surrogate import early_data, fit_and_gate, gated_rows

TIMES = ("2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01")
EARLY = "2021-02-01"


class _Persist:
    """Each key after the early date: the member's last early value of its datatype."""

    early_until = EARLY
    version = "1"

    def __init__(self) -> None:
        self.keys: tuple[DataKey, ...] = ()

    def fit(self, snapshots: SnapshotSet, rng: np.random.Generator) -> None:
        self.keys = snapshots.keys

    def early_keys(self) -> tuple[DataKey, ...]:
        return tuple(k for k in self.keys if k.time <= EARLY)

    def early_coordinates(
        self, early: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.ascontiguousarray(early), np.zeros(early.shape[1])

    def coordinates(
        self, inputs: SurrogateInputs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = next(iter(inputs.values())).shape[-1]
        return np.zeros((0, n)), np.zeros(n)

    def predict(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        raise ValueError("needs early data")

    def predict_from_early(
        self, inputs: SurrogateInputs, early: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        index = {k: row for row, k in enumerate(self.early_keys())}
        last = {k.datatype: row for row, k in enumerate(self.early_keys()) if k.time == EARLY}
        rows = [index.get(k, last[k.datatype]) for k in keys]
        return np.ascontiguousarray(early[rows])

    def to_arrays(self) -> SurrogateParts:
        return {"keys": [[k.time, k.datatype, k.index] for k in self.keys]}, {}

    @classmethod
    def from_arrays(
        cls, manifest: Mapping[str, object], arrays: Mapping[str, NDArray[np.generic]]
    ) -> Self:
        model = cls()
        model.keys = tuple(DataKey(t, d, i) for t, d, i in manifest["keys"])  # type: ignore[attr-defined]
        return model


def _snapshots(levels: NDArray[np.float64]) -> SnapshotSet:
    """Runs rising until the early date, then constant at their level ``(n,)``."""
    keys = tuple(DataKey(t, d) for d in ("WOPR:P1", "WBHP:P1") for t in TIMES)
    n = levels.size
    outputs = np.empty((len(keys), n))
    for row, key in enumerate(keys):
        scale = 1.0 if key.datatype == "WOPR:P1" else 10.0
        outputs[row] = scale * levels * (0.5 if key.time < EARLY else 1.0)
    return SnapshotSet(
        layout=StateLayout([StateVariable("perm", 1)]),
        inputs={"perm": levels[None, :]},
        keys=keys,
        outputs=outputs,
        run_ids=tuple(range(n)),
        provenance=SnapshotProvenance(simulator="analytic"),
    )


def test_the_build_gates_on_the_held_out_runs_own_early_data() -> None:
    model = _Persist()
    assert isinstance(model, EarlyStartSurrogate)
    train = _snapshots(np.linspace(10.0, 50.0, 6))
    held = _snapshots(np.array([15.0, 33.0, 47.0]))
    outcome = fit_and_gate(
        model, train, held, np.random.default_rng(0), tolerances=GateTolerances(), variances=None
    )
    assert outcome.gate.passed, outcome.gate.failures()
    assert np.max(np.abs(outcome.predicted - held.outputs)) == 0.0
    # The domain bounds the early data (2 datatypes x 2 early dates) and its residual.
    assert outcome.domain.n_coordinates == 4 + 1
    early = early_data(model, held)
    np.testing.assert_array_equal(early, held.outputs[[0, 1, 4, 5]])


def test_the_domain_and_the_simulator_use_the_members_early_data() -> None:
    model = _Persist()
    train, held = _snapshots(np.linspace(10.0, 50.0, 6)), _snapshots(np.array([20.0]))
    outcome = fit_and_gate(
        model, train, held, np.random.default_rng(0), tolerances=GateTolerances(), variances=None
    )
    early = early_data(model, held)
    with pytest.raises(ValueError, match="early data"):
        model_coordinates(model, held.inputs)
    simulator = SurrogateSimulator(
        model, outcome.domain, identity="x", loader=lambda: (model, outcome.domain),
        guard=refuse_outside,
    )  # fmt: skip
    member = MemberInput(3, {"perm": np.array([20.0])})
    with pytest.raises(ValueError, match="needs the member's early data"):
        simulator.run(member, held.keys)
    staged = simulator.with_early({3: early[:, 0]})
    np.testing.assert_array_equal(staged.run(member, held.keys), held.outputs[:, 0])
    # Early data beyond the trained range lie outside the domain: refused, never dropped.
    far = simulator.with_early({3: early[:, 0] * 100.0})
    with pytest.raises(SurrogateDomainError):
        far.run(member, held.keys)


def test_the_gate_scores_only_the_keys_after_the_early_date() -> None:
    """The early keys' error is zero by construction: counted, they would dilute the later
    error by sqrt(1 - share) and let a continuation outside the tolerance pass (numerics
    review). Here the later keys are off by 0.6 sigma (sigma_ratio 0.5): over all four
    dates the RMS would be 0.42 and pass; over the two later ones it is 0.6 and fails."""
    model = _Persist()
    train = _snapshots(np.linspace(10.0, 50.0, 6))
    held = _snapshots(np.array([15.0, 33.0, 47.0]))
    late = [row for row, key in enumerate(held.keys) if key.time > EARLY]
    outputs = held.outputs.copy()
    outputs[late] += 0.6
    shifted = SnapshotSet(
        layout=held.layout, inputs=held.inputs, keys=held.keys, outputs=outputs,
        run_ids=held.run_ids, provenance=held.provenance,
    )  # fmt: skip
    outcome = fit_and_gate(
        model, train, shifted, np.random.default_rng(0), tolerances=GateTolerances(),
        variances=np.ones(len(held.keys)),
    )  # fmt: skip
    assert gated_rows(model, held.keys) == late
    assert not outcome.gate.passed
    # The prediction itself keeps every key (the early ones unchanged).
    assert outcome.predicted.shape == held.outputs.shape


def test_the_simulator_keeps_its_early_data_when_pickled() -> None:
    model = _Persist()
    train, held = _snapshots(np.linspace(10.0, 50.0, 6)), _snapshots(np.array([20.0]))
    outcome = fit_and_gate(
        model, train, held, np.random.default_rng(0), tolerances=GateTolerances(), variances=None
    )
    staged = SurrogateSimulator(
        model, outcome.domain, identity="x", loader=_load_persist, guard=refuse_outside,
    ).with_early({3: early_data(model, held)[:, 0]})  # fmt: skip
    copy = pickle.loads(pickle.dumps(staged))
    member = MemberInput(3, {"perm": np.array([20.0])})
    np.testing.assert_array_equal(copy.run(member, held.keys), staged.run(member, held.keys))


def _load_persist() -> tuple[_Persist, TrainingDomain]:
    """A picklable loader (a module-level function) for the pickling test."""
    model = _Persist()
    train, held = _snapshots(np.linspace(10.0, 50.0, 6)), _snapshots(np.array([20.0]))
    outcome = fit_and_gate(
        model, train, held, np.random.default_rng(0), tolerances=GateTolerances(), variances=None
    )
    return model, outcome.domain


def test_history_matching_and_optimizations_refuse_an_early_start_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(surrogate=SimpleNamespace(on_check_failure="stop"))
    model = SimpleNamespace(model=_Persist(), manifest=None, path=Path("model.json"))
    with pytest.raises(ConfigError, match="history matching with it is not built yet"):
        surrogates.refuse_model(cast(Any, config), cast(Any, model), cast(Any, None),
                                cast(Any, None))  # fmt: skip
    monkeypatch.setattr(
        surrogates, "load_surrogate", lambda path: SimpleNamespace(model=_Persist(), manifest=None)
    )
    optimization = SimpleNamespace(
        surrogate=SimpleNamespace(model_file="model.json"), simulator=object()
    )
    with pytest.raises(ConfigError, match="optimizations cannot use it"):
        surrogates.optimization_surrogate(cast(Any, optimization), cast(Any, None))


def test_a_model_with_no_key_after_its_early_date_is_not_gated() -> None:
    """Every key the members' own: a gate of no rows would pass on nothing (coder's review)."""
    model = _Persist()
    model.fit(_snapshots(np.linspace(10.0, 50.0, 6)), np.random.default_rng(0))
    early_only = model.early_keys()
    with pytest.raises(WorkflowError, match="answers no key after its early date"):
        gated_rows(model, early_only)
