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
- ``requires``, a ``tuple[str, ...]`` of simulator capabilities the plugin needs to be
  trained (the intrusive reduced models); none by default;
- ``version``, a ``str`` recorded in the manifest.

A plugin's options declare what their strings name (``core.options``), as every
plugin's.

Design plans are plugins of ``core.plugins.surrogate_designs`` (entry-point
group ``apex.surrogate_designs``): core's own (:class:`BuiltinDesign`) and adaptive ones
(:class:`AdaptiveDesign`) that choose runs from a fitted surrogate.
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
