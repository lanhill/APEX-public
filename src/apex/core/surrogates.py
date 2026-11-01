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
- ``requires``, a ``tuple[str, ...]`` of simulator capabilities the plugin needs to be
  trained (the intrusive reduced models); none by default;
- ``version``, a ``str`` recorded in the manifest.

A plugin's options declare what their strings name (``core.options``), as every
plugin's.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from apex.domain.observations import DataKey
from apex.domain.surrogate import SnapshotSet

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
class UncertainSurrogate(Protocol):
    """A surrogate that gives a predictive standard deviation (optional capability)."""

    def predict_std(self, inputs: SurrogateInputs, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the standard deviation at ``keys``, ``(len(keys), n_members)``, >= 0."""
        ...
