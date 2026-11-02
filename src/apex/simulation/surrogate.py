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

``predict`` is deterministic (the protocol's rule), so serial and process executors give
equal forecasts.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from apex.core.simulation import MemberInput
from apex.core.surrogates import SurrogateInputs, SurrogateModel
from apex.domain.observations import DataKey
from apex.domain.surrogate import TrainingDomain

#: Loads a fitted model and its validated domain (in a worker), refusing another identity.
SurrogateLoader = Callable[[], tuple[SurrogateModel, TrainingDomain]]

#: Raises ``SurrogateDomainError`` for members, keys or times outside a model's domain:
#: ``(domain, model, inputs {variable: (size, n)}, member_ids, keys) -> None``.
DomainGuard = Callable[
    [TrainingDomain, SurrogateModel, SurrogateInputs, Sequence[int], Sequence[DataKey]], None
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
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore the reference; the model is loaded (and its identity checked) when used."""
        self.identity = str(state["identity"])
        self.loader = state["loader"]
        self.guard = state["guard"]
        self.threads = int(state["threads"])
        self._model = None
        self._domain = None

    def fitted(self) -> tuple[SurrogateModel, TrainingDomain]:
        """Return the model and its domain, loading them on first use."""
        if self._model is None or self._domain is None:
            self._model, self._domain = self.loader()
        return self._model, self._domain

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Predict one member at ``keys``: ``(len(keys),)``.

        Raises
        ------
        SurrogateDomainError
            If the member, a key or a key's time lies outside the model's validated domain.
        """
        model, domain = self.fitted()
        inputs = {
            name: np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1, 1))
            for name, value in member.state.items()
        }
        self.guard(domain, model, inputs, (member.member_id,), keys)
        predicted = np.asarray(model.predict(inputs, keys), dtype=np.float64)
        return np.ascontiguousarray(predicted.reshape(-1))
