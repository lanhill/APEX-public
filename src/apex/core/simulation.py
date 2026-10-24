"""Interfaces between workflows, forward simulators, prior generators and data transforms.

0006 (priors), 0007 (data transforms), 0012
(time series for predictions), 0037 (joint priors, parameterizations), 0058 (model-error
models) and 0061 (geophysical field data).

A :class:`Simulator` runs one ensemble member; an :class:`Executor` decides where and how
many members run at once. Both must be picklable so that executors can ship them to
worker processes.
"""

import fnmatch
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from apex.domain.observations import DataKey, ObservationSet
from apex.domain.state import StateVariable


@dataclass(frozen=True, slots=True)
class MemberInput:
    """Everything a simulator needs to run one member.

    Attributes
    ----------
    member_id
        Stable identifier of the member across iterations (``-1`` for the mean model).
    state
        The member's value of each state variable, ``{name: (size,)}``.
    work_dir
        A private directory for this run's files, or ``None`` for in-memory simulators.
    runner
        Where the simulator's program runs, set by an executor that runs members
        elsewhere (``ssh``); ``None`` runs it here, as always. Only simulators with the
        capability ``accepts_runner`` honour it, and wrappers that rebuild a member keep it
        (``dataclasses.replace``).
    """

    member_id: int
    state: Mapping[str, NDArray[np.float64]]
    work_dir: Path | None = None
    runner: "CommandRunner | None" = None


class CommandRunner(Protocol):
    """Runs a simulator's program for one member, here or on another host.

    The member's work directory is the program's working directory: what the simulator
    wrote there is staged in before :meth:`run`, and what the program wrote is back there
    when it returns. Paths in the command are relative to it.
    """

    def static_folder(self, local: Path, files: Sequence[str] | None = None) -> Path:
        """Return where the program sees the local folder ``local`` (a deck's includes).

        A remote runner copies the folder (or only ``files``, paths relative to it; a
        folder among them whole) to the host once and returns its path there; a local one
        returns ``local``.
        """
        ...

    def run(
        self,
        command: Sequence[str],
        work_dir: Path,
        *,
        env: Mapping[str, str],
        timeout_s: float,
        log: Path,
    ) -> int | None:
        """Run ``command`` in ``work_dir``; return its exit code, or ``None`` if it timed out.

        ``env`` is added to the program's environment; its standard output and error go to
        ``log`` (in ``work_dir``).

        Raises
        ------
        apex.core.errors.TransientSimulationError
            If the run could not be carried out for a reason that may not recur (the
            connection, the copying); the executor retries it, maybe elsewhere.
        """
        ...


@dataclass(frozen=True)
class RemoteCode:
    """The copy of APEX remote members run.

    Attributes
    ----------
    commit
        The APEX commit (``live`` only in tests: the running code itself).
    local_python
        The interpreter of this machine's copy, for the ``local`` slots.
    host_python
        The interpreter of each host's copy: under the home (``~/...``) or absolute.
    env
        The pinned environment the workers run in (``PINNED_ENV``).
    hosts
        The hosts whose copy was verified identical to this machine's; ``None``: every
        host (the running code, in tests). Others are left out of the run.
    verdicts
        The hosts verified now, by any process (the copies' record), read again before
        each forecast: a host verified after the run prepared its copies joins it, and one
        found to differ since leaves it. ``None``: ``hosts`` for the whole run.
    """

    commit: str
    local_python: str
    host_python: str
    env: Mapping[str, str]
    hosts: frozenset[str] | None = None
    verdicts: Callable[[], frozenset[str]] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class RemoteSpec:
    """How an executor rebuilds a run's simulator on other hosts (option A).

    A host cannot run a Python object of this process, so it rebuilds the simulator from
    the run's resolved configuration, in a copy of APEX at the run's commit. Composition
    makes the spec for the stack it builds and gives it to the executor; the executor runs
    members remotely only for that very stack (``simulator``, by identity), and refuses any
    other (a sensitivity run, a parameterization's wrapper, an optimization's objective).

    Attributes
    ----------
    simulator
        The stack the spec describes.
    config
        The resolved configuration (``config.json``'s document), with this machine's paths.
    includes
        The paths the deck template names under ``${case_dir}`` (``None``: all of it).
    code
        Prepares the copies on the given hosts and says which to use; called once, before
        the first forecast (a run built only to be validated or read deploys nothing).
    stopped
        Whether the run was asked to stop, for waits outside its steps: its folder's stop
        request (``apex stop``, or a job's stop while its worker holds the run); ``None``:
        never asked.
    """

    simulator: object
    config: Mapping[str, object]
    includes: tuple[str, ...] | None
    code: Callable[[Sequence[str]], RemoteCode]
    stopped: Callable[[], bool] | None = None


@dataclass(frozen=True, slots=True)
class MemberFailure:
    """A member whose forward run failed.

    ``transient`` says whether its last failure was a transient one (retried until the
    executor's retries ran out), and ``attempts`` how many runs it took.
    """

    member_id: int
    message: str
    transient: bool = False
    attempts: int = 1


class Simulator(Protocol):
    """Runs one forward simulation.

    Optional capabilities, read with ``getattr`` by the workflow and executors:
    ``needs_work_dir`` (the run writes files and gets its own directory) and
    ``max_parallel_runs`` (an ``int``: at most this many runs at once, e.g. the licences
    of a commercial simulator; parallel executors honour it).
    """

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the predicted value of each key, shape ``(len(keys),)``.

        Raises
        ------
        apex.core.errors.SimulationError
            If the run fails or does not produce every requested key.
        """
        ...


@dataclass(frozen=True, slots=True)
class SeriesRequest:
    """Time series one prediction run should return.

    Attributes
    ----------
    vectors
        Vector names or shell-style patterns, e.g. ``("FOPT", "WOPR:*")``.
    times
        ISO report dates to return; ``None`` for every report time of the run.
    """

    vectors: tuple[str, ...]
    times: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class MemberSeries:
    """The time series of one member's run.

    Attributes
    ----------
    times
        Report times, ``(n_times,)``, as ISO dates in increasing order.
    vectors
        Resolved vector names, ``(n_vectors,)``, sorted.
    values
        ``(n_vectors, n_times)``.
    """

    times: tuple[str, ...]
    vectors: tuple[str, ...]
    values: NDArray[np.float64]


class SeriesSimulator(Protocol):
    """A simulator that can return whole time series (optional capability).

    Without it, a simulator predicts only at explicit keys, through :meth:`Simulator.run`.
    """

    def run_series(self, member: MemberInput, request: SeriesRequest) -> MemberSeries:
        """Run one member and return the requested series.

        Raises
        ------
        apex.core.errors.SimulationError
            If the run fails, a pattern matches no vector, or a requested time is not a
            report time.
        """
        ...


@runtime_checkable
class SeriesRun(Protocol):
    """A prediction's view of its simulator, run by an executor.

    An executor that runs members whole elsewhere (the ssh executor's remote mode) cannot
    call it there: it runs ``simulator``'s series at ``request`` on the host and hands each
    member's series back here through :meth:`save_series` (``request.times`` ``None``: a
    pilot member, whose series fixes the axis).
    """

    simulator: "Simulator"
    request: SeriesRequest
    #: Whether the member resolves the request (patterns, report times): a pilot.
    resolves_axis: bool

    def save_series(self, member_id: int, series: MemberSeries) -> None:
        """Keep one member's series (checked against the request when it is resolved).

        Raises
        ------
        apex.core.errors.SimulationError
            If the series does not fit the request.
        """
        ...


def match_vectors(patterns: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    """Resolve vector names and patterns against the ``available`` ones, sorted.

    Raises
    ------
    ValueError
        If a pattern (or name) matches no available vector.
    """
    matched: set[str] = set()
    for pattern in patterns:
        # Case-sensitive on every platform (fnmatch.filter follows the OS's case rules).
        found = [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        if not found:
            raise ValueError(f"no vector matches {pattern!r}")
        matched.update(found)
    return tuple(sorted(matched))


#: Called by an executor as each member finishes: ``(member_id, ok)``.
MemberCallback = Callable[[int, bool], None]


class SensitivitySimulator(Protocol):
    """A simulator that also returns its sensitivities (optional capability).

    Used by strategies that take the simulator's gradients in place of the ensemble's
    estimate (``ApproxUpdate(sensitivities=True)``, PET's adjoint branch).
    """

    def run_with_sensitivities(
        self, member: MemberInput, keys: Sequence[DataKey]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run one member and return its values and their sensitivity to its state.

        The values are at ``keys`` ``(n_keys,)``; the sensitivity ``d values / d state`` is
        ``(n_keys, n_state)``, the state stacked in the layout's order.

        Raises
        ------
        apex.core.errors.SimulationError
            If the run fails.
        """
        ...


class Executor(Protocol):
    """Runs many members, sequentially or in parallel.

    Optional class-level hooks an executor may define, found by composition:

    - ``config_defaults``, a static or class method ``(HistoryMatchConfig) ->
      HistoryMatchConfig``: defaults the executor sets in a run's configuration before any
      identity is taken of it (the ssh executor's remote mode);
    - ``remote_code_provider``, a ``"module:name"`` string naming the callable that
      prepares the copies of APEX remote members run, for an executor with
      ``attach_remote``.
    """

    def run_all(
        self,
        simulator: Simulator,
        members: Sequence[MemberInput],
        keys: Sequence[DataKey],
        *,
        on_result: MemberCallback | None = None,
    ) -> list[NDArray[np.float64] | MemberFailure]:
        """Run every member and return its prediction or failure, in input order.

        Only :class:`~apex.core.errors.SimulationError` is converted to a failure; any
        other exception is a bug and propagates.

        ``on_result``, if given, is called in the calling process once per member as it
        finishes (in completion order, not input order), with ``ok = False`` for a
        failure. It is for progress reporting only and must not change results.
        The workflow always passes it, so every executor must accept the keyword. A member
        that raised (a bug) is not reported.
        """
        ...


@runtime_checkable
class CaseFileSimulator(Protocol):
    """A simulator whose runs leave files that others may read before they are pruned.

    The deck simulators (``opm_flow``, ``eclipse``) have it; a model-error simulator wraps
    one: ``simulate``, then ``read`` and its own reading, then ``clean``.
    Optional: ``release(case)``, called instead of ``clean`` when a reading fails, forgets
    what the simulator holds for the case and leaves its files (a failed member's are kept
    for diagnosis, ``keep_simulations``); it must not raise, as it runs while the
    reading's error propagates.
    """

    def simulate(self, member: MemberInput) -> Path:
        """Run the member and return its case path (without extension)."""
        ...

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the simulated values of ``case`` at ``keys``, ``(len(keys),)``."""
        ...

    def clean(self, case: Path) -> None:
        """Prune the case's files as the simulator's options say."""
        ...


@runtime_checkable
class FieldResponses(Protocol):
    """A simulator that computes field data itself (shared with).

    A coupled flow-geomechanics simulator answers field keys ``DataKey(time, datatype, i)``
    of the datatypes it declares from its own ``read`` (and ``run``), alongside the summary's
    scalar keys: its ``read(case, keys)`` and ``run(member, keys)`` must accept them.
    Composition checks the observed field keys' indices against the declaration before any
    member runs; their times must be dates the simulator reports (a time it does not report
    fails when read, in every member, as a ``ConfigError``). The observations are
    ``[[observations.fields]]``. A ``[geophysics]`` field may not share a declared name.

    Datatypes shared by the geomechanical simulators:

    - ``vertical_displacement``: of the top surface (or seafloor), metres, **up positive**
      (subsidence is negative), since the simulation's start;
    - ``compaction``: of each reservoir cell, metres, **positive when compacting**, since
      the simulation's start.
    """

    def field_datatypes(self) -> Mapping[str, int]:
        """Return each field datatype the simulator computes and its size (indices 0..size-1)."""
        ...


class ModelErrorModel(Protocol):
    """A model-error (MEC) model r(m, theta), added to a simulator's output.

    Its parameters theta are the state variable ``state_variable``, of ``n_params`` values
    once :meth:`fit` has seen the observations. ``residual_keys`` are the extra keys the
    effective simulator answers with the residuals themselves (datatype ``MEC:<type>``).
    """

    state_variable: str

    @property
    def n_params(self) -> int:
        """Number of parameters (the state variable's size)."""
        ...

    def fit(self, observations: ObservationSet) -> None:
        """Set what depends on the observations (e.g. output scales)."""
        ...

    def residual_keys(self, keys: Sequence[DataKey]) -> list[DataKey]:
        """Return the residual key of each key the model corrects, in order."""
        ...

    def residuals(
        self, case: Path, theta: NDArray[np.float64], keys: Sequence[DataKey]
    ) -> NDArray[np.float64]:
        """Return r at ``keys`` (0 where it does not apply) for one simulated member."""
        ...


class CaseFieldModel(Protocol):
    """Field data computed from a simulated case's files: geophysics.

    A geophysics simulator wraps a deck simulator: it gives the keys this model owns to
    :meth:`forecast` and the others to the deck simulator's own ``read``.

    A model with uncertain parameters is also a :class:`StatefulFieldModel`.
    """

    def owns(self, key: DataKey) -> bool:
        """Return whether ``key`` is one of the model's field data."""
        ...

    def forecast(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the values at ``keys`` (all owned) from the case's files, ``(len(keys),)``.

        Raises
        ------
        SimulationError
            If the member's own model is outside the forward model's validity.
        ConfigError
            If the case lacks what every member would lack (a restart at a vintage, the
            densities), so that the run stops instead of failing every member.
        """
        ...


@runtime_checkable
class StatefulFieldModel(CaseFieldModel, Protocol):
    """A :class:`CaseFieldModel` whose parameters come from state variables.

    The variables are the model's alone: the deck never sees them.
    """

    @property
    def state_variables(self) -> tuple[str, ...]:
        """The state variables the model takes (kept from the deck)."""
        ...

    def for_state(self, state: Mapping[str, ArrayLike]) -> CaseFieldModel:
        """Return the model for one member's values of :attr:`state_variables`."""
        ...


@runtime_checkable
class PriorGenerator(Protocol):
    """Draws the initial ensemble of one state variable."""

    def generate(
        self, variable: StateVariable, n_ensemble: int | None, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return ``(variable.size, n_ensemble)`` realisations.

        ``n_ensemble`` may be ``None`` only for generators whose source fixes the ensemble
        size (e.g. a stored ensemble); others raise ``ValueError``. State bounds are
        applied by the caller, not here, unless the generator sets a class attribute
        ``clip = False`` (used as is, e.g. a stored ensemble).
        """
        ...


class DataTransform(Protocol):
    """Maps simulated data into the space the analysis works.

    Example: wavelet compression of a seismic field into its leading coefficients. The
    workflow requests the *raw* observation keys from the simulator, then passes every
    forecast through the fitted transforms; methods, localization and data mismatches see only
    the transformed data. Transforms must be deterministic functions of the observations
    and their options, so that a resumed run refits them identically.
    """

    def fit(self, observations: ObservationSet) -> ObservationSet:
        """Fit to the observed data and return them transformed, with their variances.

        Must be called once before :meth:`apply`.

        Raises
        ------
        ValueError
            If the observations do not contain what the transform is configured for.
        """
        ...

    def apply(self, predicted: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform predictions ``(n_in, n_ensemble)`` at the fitted observations' keys.

        Returns ``(n_out, n_ensemble)``, row-aligned with the set :meth:`fit` returned.
        """
        ...


@runtime_checkable
class PortableTransform(DataTransform, Protocol):
    """A data transform whose fitted state can be applied elsewhere.

    An executor that runs members whole on other hosts may apply the run's transforms
    there, member by member, and send back only the transformed forecast. It may do so
    only when the transform says ``member_wise``: applied to one member (a column), it
    gives, bit for bit, that member's column of the transform applied to the ensemble.
    The host rebuilds the transform from the configuration and loads the state fitted here
    (never refits: the fit is this machine's).
    """

    member_wise: bool

    def state(self) -> dict[str, NDArray[np.generic]]:
        """Return the fitted state as named arrays (numbers or text, no objects)."""
        ...

    def load_state(self, state: Mapping[str, NDArray[np.generic]]) -> None:
        """Take a fitted state :meth:`state` returned, instead of :meth:`fit`.

        Raises
        ------
        ValueError
            If the state does not fit the transform's options.
        """
        ...


class JointPriorGenerator(Protocol):
    """Draws several state variables together, e.g. the latent fields of a PGS.

    Configured in ``[prior.groups.<name>]`` with the variables it draws; per-variable
    generators (:class:`PriorGenerator`) draw the others. Randomness comes from the run's
    ``prior`` stream; the variables are not clipped unless the generator sets ``clip``.
    """

    def generate_joint(
        self,
        variables: Sequence[StateVariable],
        n_ensemble: int | None,
        rng: np.random.Generator,
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{name: (variable.size, n_ensemble)}`` for every variable given.

        Raises
        ------
        ValueError
            If the variables do not fit the generator (sizes, grids) or ``n_ensemble`` is
            missing.
        """
        ...


class Parameterization(Protocol):
    """Maps a member's state to the properties its forward model reads.

    Example: plurigaussian truncation, from latent Gaussian fields to facies and to the
    porosity and permeability of each facies. The methods update the state; the simulator,
    the performance metrics and the figures see the properties. It must be a deterministic
    function of the state and its options. State variables it does not consume pass
    through unchanged.
    """

    #: The state variables it consumes.
    variables: tuple[str, ...]
    #: The properties it produces. Those that are codes, not quantities (``facies``), are
    #: named in an optional ``categorical`` tuple: metrics of quantities leave them out.
    outputs: tuple[str, ...]

    def properties(
        self, state: Mapping[str, NDArray[np.float64]]
    ) -> dict[str, NDArray[np.float64]]:
        """Return ``{output: (n_cells,)}`` from ``{variable: (size,)}`` of one member.

        Raises
        ------
        ValueError
            If a consumed variable is missing or has the wrong size.
        """
        ...
