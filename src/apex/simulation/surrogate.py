"""A fitted surrogate presented as a simulator.

:class:`SurrogateSimulator` answers a member's keys with a fitted surrogate model
(``core.surrogates.SurrogateModel``), as ``ModelErrorSimulator`` presents a simulator plus a
learned residual: the executors run it as any simulator, one member at a time.

- **Inside its domain only.** Every member is checked against the model's validated
  ``TrainingDomain`` (its coordinates and projection residual, the keys asked for) before
  it is predicted. Outside, :class:`~apex.core.errors.SurrogateDomainError` is raised: an
  ``ApexError``, *not* a ``SimulationError``, so no executor turns it into a dropped member
  (dropping the members a surrogate cannot predict would bias the posterior silently). The
  check itself is core's (``algorithms/surrogates/domain.py``), given to the simulator as a
  function, since this layer imports only ``core`` and ``domain``.
- **No files.** ``needs_work_dir`` is false; there is no ``accepts_runner``,
  ``max_parallel_runs`` or ``run_with_sensitivities``.
- **CPU.** Each prediction holds ``threads`` of the machine-wide ``cpu`` tokens
  (``resources``), as a simulator's run does, so surrogate forecasts count against the cores
  with every other run.
- **Picklable as a reference.** It pickles as its model file's identity and a loader (a
  module-level function, by reference), and loads the model in the worker that unpickles it,
  refusing files of another identity: process executors ship it cheaply, and a changed file
  is never used.

- **A coarse-corrected model** (``core.surrogates.CoarseCorrectedSurrogate``)
  predicts from each member's coarse forecast too. The run makes those first, as a forecast
  of its own on its executor (the user's choice, 2026-11-02: the domain check needs them, and
  the run's executor gives the coarse runs its pools, licences, retries and pruning), and
  hands them over with :meth:`SurrogateSimulator.with_coarse`; the simulator itself still
  writes no files.

- **An early-start model** (``core.surrogates.EarlyStartSurrogate``)
  continues each member's own early data: the run simulates each member up to the model's
  early date first (``MemberInput.stop_at``) and hands the values over with
  :meth:`SurrogateSimulator.with_early`, as the coarse forecasts above.

``predict`` is deterministic (the protocol's rule), so serial and process executors give
equal forecasts.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import DatedStates, MemberInput
from apex.core.surrogates import (
    BlockStateSurrogate,
    CoarseCorrectedSurrogate,
    EarlyStartSurrogate,
    SurrogateInputs,
    SurrogateModel,
)
from apex.domain.observations import DataKey
from apex.domain.surrogate import TrainingDomain

#: Loads a fitted model and its validated domain (in a worker), refusing another identity.
SurrogateLoader = Callable[[], tuple[SurrogateModel, TrainingDomain]]

#: Raises ``SurrogateDomainError`` for members, keys or times outside a model's domain:
#: ``(domain, model, inputs {variable: (size, n)}, member_ids, keys, coarse) -> None``, with
#: ``coarse`` the members' coarse forecasts ``(n_model_keys, n)`` for a coarse-corrected model
#: (``None`` otherwise).
DomainGuard = Callable[
    [
        TrainingDomain,
        SurrogateModel,
        SurrogateInputs,
        Sequence[int],
        Sequence[DataKey],
        NDArray[np.float64] | None,
    ],
    None,
]


class SurrogateSimulator:
    """A fitted surrogate as a :class:`~apex.core.simulation.Simulator`.

    Parameters
    ----------
    model, domain
        The fitted model and its validated domain (``None`` to load them with ``loader``).
    identity
        The model files' identity (the SHA-256 of manifest and arrays).
    loader
        Loads ``(model, domain)`` from the files of ``identity``; used after unpickling.
    guard
        Core's domain check (raises ``SurrogateDomainError`` outside it).
    threads
        ``cpu`` tokens each prediction holds.
    """

    #: A surrogate writes no files.
    needs_work_dir = False

    def __init__(
        self,
        model: SurrogateModel | None,
        domain: TrainingDomain | None,
        *,
        identity: str,
        loader: SurrogateLoader,
        guard: DomainGuard,
        threads: int = 1,
    ) -> None:
        if threads < 1:
            raise ValueError(f"threads must be at least 1, got {threads}")
        self.identity = identity
        self.loader = loader
        self.guard = guard
        self.threads = threads
        self._model = model
        self._domain = domain
        #: A coarse-corrected model's coarse forecasts by member, ``(n_model_keys,)`` each.
        self.coarse: dict[int, NDArray[np.float64]] = {}
        #: An early-start model's early data by member, ``(n_early_keys,)`` each.
        self.early: dict[int, NDArray[np.float64]] = {}

    def with_early(self, early: Mapping[int, NDArray[np.float64]]) -> "SurrogateSimulator":
        """Return a copy that predicts from the members' early data.

        ``early``: ``{member_id: (n_early_keys,)}``, each member's simulated values at the
        model's early keys. The copy shares the loaded model and pickles the data with its
        reference.
        """
        copy = SurrogateSimulator(
            self._model, self._domain, identity=self.identity, loader=self.loader,
            guard=self.guard, threads=self.threads,
        )  # fmt: skip
        copy.coarse = dict(self.coarse)
        copy.early = {
            int(m): np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))
            for m, v in early.items()
        }
        return copy

    def with_coarse(self, coarse: Mapping[int, NDArray[np.float64]]) -> "SurrogateSimulator":
        """Return a copy that predicts with the members' coarse forecasts.

        ``coarse``: ``{member_id: (n_model_keys,)}``, the coarse stack's runs at the model's
        keys. The copy shares the loaded model and pickles the forecasts with its reference.
        """
        copy = SurrogateSimulator(
            self._model, self._domain, identity=self.identity, loader=self.loader,
            guard=self.guard, threads=self.threads,
        )  # fmt: skip
        copy.coarse = {
            int(m): np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))
            for m, v in coarse.items()
        }
        copy.early = dict(self.early)
        return copy

    @property
    def resources(self) -> Mapping[str, int]:
        """The machine-wide tokens a prediction holds: ``threads`` cores."""
        return {"cpu": self.threads}

    def __getstate__(self) -> dict[str, Any]:
        """Pickle as a reference: the identity and the loader, never the model's arrays."""
        return {
            "identity": self.identity,
            "loader": self.loader,
            "guard": self.guard,
            "threads": self.threads,
            "coarse": self.coarse,
            "early": self.early,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore the reference; the model is loaded (and its identity checked) when used."""
        self.identity = str(state["identity"])
        self.loader = state["loader"]
        self.guard = state["guard"]
        self.threads = int(state["threads"])
        self.coarse = dict(state.get("coarse", {}))
        self.early = dict(state.get("early", {}))
        self._model = None
        self._domain = None

    def fitted(self) -> tuple[SurrogateModel, TrainingDomain]:
        """Return the model and its domain, loading them on first use."""
        if self._model is None or self._domain is None:
            self._model, self._domain = self.loader()
        return self._model, self._domain

    def gives_block_states(self) -> bool:
        """Whether the model gives a member's block states.

        A model that predicts from companion runs (coarse-corrected, early-start) does
        not: its states would not be those of one prediction from the member alone.
        """
        model, _ = self.fitted()
        return (
            isinstance(model, BlockStateSurrogate)
            and not isinstance(model, CoarseCorrectedSurrogate)
            and not isinstance(model, EarlyStartSurrogate)
        )

    def block_states(self, member: MemberInput, dates: Sequence[str]) -> DatedStates:
        """Return the member's states at the ISO ``dates`` (a model-error model's inputs).

        Raises
        ------
        SurrogateDomainError
            If the member lies outside the model's validated domain.
        ValueError
            If the model gives no block states.
        """
        model, domain = self.fitted()
        if not self.gives_block_states():
            raise ValueError(f"the surrogate {self.identity[:12]} gives no block states")
        assert isinstance(model, BlockStateSurrogate)
        inputs = {
            name: np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1, 1))
            for name, value in member.state.items()
        }
        self.guard(domain, model, inputs, (member.member_id,), (), None)
        return model.predict_block_states(inputs, list(dates))

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Predict one member at ``keys``: ``(len(keys),)``.

        Raises
        ------
        SurrogateDomainError
            If the member, a key or a key's time lies outside the model's validated domain.
        ValueError
            If a coarse-corrected model was given no coarse forecast of the member.
        """
        model, domain = self.fitted()
        inputs = {
            name: np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1, 1))
            for name, value in member.state.items()
        }
        if isinstance(model, EarlyStartSurrogate):
            if member.member_id not in self.early:
                raise ValueError(
                    f"member {member.member_id}: an early-start surrogate needs the member's "
                    "early data (with_early), which the run makes first"
                )
            early = np.ascontiguousarray(self.early[member.member_id].reshape(-1, 1))
            self.guard(domain, model, inputs, (member.member_id,), keys, early)
            predicted = np.asarray(model.predict_from_early(inputs, early, keys), np.float64)
            return np.ascontiguousarray(predicted.reshape(-1))
        if not isinstance(model, CoarseCorrectedSurrogate):
            self.guard(domain, model, inputs, (member.member_id,), keys, None)
            predicted = np.asarray(model.predict(inputs, keys), dtype=np.float64)
            return np.ascontiguousarray(predicted.reshape(-1))
        if member.member_id not in self.coarse:
            raise ValueError(
                f"member {member.member_id}: a coarse-corrected surrogate needs the member's "
                "coarse forecast (with_coarse), which the run makes first"
            )
        coarse = np.ascontiguousarray(self.coarse[member.member_id].reshape(-1, 1))
        self.guard(domain, model, inputs, (member.member_id,), keys, coarse)
        predicted = np.asarray(model.predict_corrected(inputs, coarse, keys), dtype=np.float64)
        return np.ascontiguousarray(predicted.reshape(-1))
