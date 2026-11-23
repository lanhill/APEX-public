"""Reading a simulated case's state from any simulator.

A :class:`CaseStateReader` opens one finished member's case and gives its mesh, static
arrays, the states it wrote (:class:`~apex.domain.case_state.StateTime`) and their arrays,
each in the unit its simulator wrote and stated by :meth:`CaseStateReader.unit`.
Readers are plugins (``core.plugins.case_state_readers``, the entry-point group
``apex.case_state_readers``) keyed by the simulator's name; the services choose one by the
simulator a configuration names and hand it to the consumers (field snapshots, geophysics,
model error), which never open a case's files themselves.

Capabilities beyond the protocol are declared by a reader class's ``capabilities``:
``corner_geometry`` is a protocol of its own, :class:`CornerGeometryReader` (a corner-point
grid's geometry, a deck's unit system and phases: what geophysics and model error need);
``structured_mesh`` says that :meth:`CaseStateReader.mesh` always gives a structured grid
(``CaseMesh.dims``), what the grid-based field surrogates need, known before any run;
``connectivity`` is a protocol of its own, :class:`ConnectivityReader` (the faces between the
state's cells, what a graph network on the mesh needs); ``wells`` is
:class:`WellsReader` (the wells' blocks, bottom-hole pressures and report dates, what
model error needs); ``well_controls`` is :class:`WellControlsReader` (the
wells' summary vectors and control modes, revision 3's W3: what a forecasting model-error
model needs). A consumer
checks the capability it needs and refuses a reader without it, with the reason.

Names: the canonical names consumers share (``PRESSURE``, ``SWAT``, ``SGAS``, ``SOIL``,
``TEMP``, ``RS``, ``WAT_DEN``, ``GAS_DEN``, ``OIL_DEN``, ``PORO``, ``PERMX``, ...) are each
reader's to map; any other name a simulator wrote passes through as written. Behaviour on
the arrays (settling saturations, deriving a missing phase) stays with the consumers.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from apex.core.errors import ConfigError, SimulationError
from apex.domain.case_state import (
    CaseMesh,
    CornerGrid,
    StateAt,
    StateOutput,
    StateTime,
    UnitSystem,
)
from apex.domain.cell_graph import CellGraph
from apex.domain.wells import WellSchedule

__all__ = [
    "CaseStateReader",
    "CaseStateReaderFactory",
    "ConnectivityReader",
    "CornerGeometryReader",
    "MissingState",
    "StateWriter",
    "WellControlsReader",
    "WellLayout",
    "WellLimits",
    "WellsReader",
    "check_region",
]


class MissingState(SimulationError):
    """The run wrote no state at the time asked for (a consumer may tell its cause)."""


@runtime_checkable
class CaseStateReader(Protocol):
    """One finished member's case: its mesh, static arrays and written states.

    ``region`` names one of :meth:`regions` (a multi-region case: OpenFOAM's gas and solid,
    a coupled model's two); ``None`` is the only region of a single-region case, and an
    error on a case of several (:func:`check_region`).
    """

    @property
    def case(self) -> Path:
        """The case the reader opened (for messages)."""

    def regions(self) -> tuple[str, ...]:
        """Return the case's regions' names; empty for a single-region case."""

    def mesh(self, region: str | None = None) -> CaseMesh:
        """Return the region's cells."""

    def static_names(self, region: str | None = None) -> frozenset[str]:
        """Return the names of the static arrays the case wrote."""

    def static(self, name: str, region: str | None = None) -> NDArray[np.float64]:
        """Return a static cell array as written, ``(cells,)``.

        Raises
        ------
        SimulationError
            If the case has no such array (naming those it has).
        """

    def times(self, region: str | None = None) -> tuple[StateTime, ...]:
        """Return the states the run wrote, in its own order (two on one day: two entries)."""

    def names(self, time: StateTime, region: str | None = None) -> frozenset[str]:
        """Return the names of the arrays written at ``time``."""

    def at_step(self, step: int) -> StateTime:
        """Return the state with index ``step``, found without reading any other state.

        Its ``seconds`` and ``date`` may be ``None``: :meth:`state` finds a state by its
        index. For a consumer that reads by output number (model error, by report step).

        Raises
        ------
        MissingState
            If the run wrote no state with that index.
        """

    def state(
        self,
        time: StateTime,
        names: Sequence[str],
        region: str | None = None,
        *,
        skip_missing: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        """Return the arrays ``names`` at ``time`` as written, each ``(cells,)``.

        With ``skip_missing``, only those written at ``time``, in the order of ``names``:
        for a consumer that tells a missing array's cause itself.

        The state is found by ``time.index``.

        Raises
        ------
        MissingState
            If the run wrote no state with ``time``'s index. Nothing is interpolated.
        SimulationError
            If an array is not at ``time`` (unless ``skip_missing``).
        """

    def unit(self, name: str) -> str | None:
        """Return an array's unit (a key of :data:`~apex.domain.case_state.SI_UNITS`).

        ``None`` when the reader cannot state it: the array is as its simulator wrote it,
        and :func:`~apex.domain.case_state.to_si` refuses it.
        """

    def reached(self) -> StateTime | None:
        """Return how far the run went by its own record (a summary), or ``None``.

        Tells a run that ended early (this member's failure) from a state its input never
        asks for (every member's).
        """


@runtime_checkable
class CornerGeometryReader(CaseStateReader, Protocol):
    """A reader of a corner-point grid with a deck's unit system and phases.

    What geophysics and model error need; only the Eclipse-format reader has
    it for now.
    """

    @property
    def unit_system(self) -> UnitSystem:
        """The deck's unit system."""

    @property
    def phases(self) -> tuple[str, ...]:
        """The deck's phases (``"oil"``, ``"water"``, ``"gas"``), in that order."""

    def corner_geometry(self, region: str | None = None) -> CornerGrid:
        """Return the grid's geometry in SI units."""


@runtime_checkable
class ConnectivityReader(CaseStateReader, Protocol):
    """A reader that gives the faces between its state's cells.

    The capability ``connectivity``. The graph is over the cells :meth:`state` returns, in
    its order (``CaseMesh.cells``: a structured grid's active cells in Eclipse (F) order, an
    unstructured mesh's cells in the simulator's own); a consumer that lays fields over a
    whole grid maps them through ``CaseMesh.active``.
    """

    def connectivity(self, region: str | None = None) -> CellGraph:
        """Return the region's cell graph: each face once, with its area and distance.

        Raises
        ------
        ConfigError
            If the case's mesh cannot give one (every member's would fail the same way),
            naming why.
        SimulationError
            If the mesh's files cannot be read (this member).
        """


@runtime_checkable
class WellsReader(CaseStateReader, Protocol):
    """A reader that gives what model error reads of the wells.

    The capability ``wells``. A well block is an Eclipse-format grid's 1-based ``(i, j, k)``
    or a mesh's 0-based cell index in :meth:`state`'s order; the wells' layout (their blocks
    and open steps) is known before any run, from the deck's text or the simulator's input
    (:class:`WellLayout`), not read here.
    """

    def block_cell(self, block: tuple[int, int, int] | int) -> int:
        """Return the index into :meth:`state`'s arrays of a well block, ``-1`` if inactive."""

    def bottom_hole_pressure(self, well: str, time: StateTime) -> float:
        """Return a well's bottom-hole pressure at a state, in :meth:`unit` ``WBHP``.

        ``time`` is one of the reader's states (:meth:`CaseStateReader.at_step` or
        :meth:`CaseStateReader.times`). Also for a shut well, as the run reports it.

        Raises
        ------
        SimulationError
            If the run reported no bottom-hole pressure for the well at that state.
        """

    def report_dates(self, steps: Sequence[int]) -> list[str]:
        """Return the ISO dates of 1-based report steps, if the run numbers them.

        A simulator whose states are not numbered by report step (GEOS's cycles) refuses:
        its consumer gives the steps' dates itself (model error's ``report_dates``).

        Raises
        ------
        SimulationError
            If the run did not report one of the steps, or numbers no report steps.
        """


@runtime_checkable
class WellControlsReader(CaseStateReader, Protocol):
    """A reader that gives its wells' summary vectors and control modes.

    The capability ``well_controls``: what a model-error model that forecasts under changed
    controls reads beyond the block states (the lab's ``mec_forecast_lstm``): each well's
    control mode with Eclipse's ``WMCTL`` codes (0 shut, 1 oil rate, 2 water rate, 3 gas
    rate, 4 liquid rate, 5 reservoir volume rate, 6 THP, 7 BHP; negative: a group's target)
    and its rates and pressures (``WOPR``, ``WWPR``, ``WGPR``, ``WWIR``, ``WGIR``, ``WBHP``,
    ...) at a state, in the unit :meth:`~CaseStateReader.unit` states for the vector's name
    (an Eclipse-format run's deck units; SI on the others, as their bottom-hole pressures).
    Consumers convert (``apex.simulation.model_error.ReaderDatedStates.well_vector``).
    """

    def has_well_vector(self, well: str, name: str) -> bool:
        """Return whether the run reports the vector ``name`` (``WMCTL`` included) of ``well``."""

    def well_vector(self, well: str, name: str, time: StateTime) -> float:
        """Return a well's vector at one of the reader's states (``WMCTL``: its code).

        Raises
        ------
        SimulationError
            If the run does not report the vector, or not at that state.
        """


@runtime_checkable
class WellLimits(Protocol):
    """A simulator that states its wells' rate and BHP targets before any run (W3c).

    What a model-error model that scales each control mode's target
    reads where a deck's schedule would give it (the lab's ``mec_forecast_lstm``): per well
    its largest rate target and BHP target or limit, in METRIC deck units (sm3/day, bar; 0:
    none), from the simulator's input.
    """

    def well_limits(
        self, wells: Sequence[tuple[str, str, Sequence[str]]]
    ) -> dict[str, tuple[float, float]]:
        """Return ``{well: (rate, bhp)}`` for ``(name, kind, outputs)`` wells.

        Raises
        ------
        ConfigError
            If a well's targets cannot be read, or depend on the input's values.
        """


@runtime_checkable
class WellLayout(Protocol):
    """A simulator that states its wells' blocks before any run.

    For model error on a simulator whose input is not a deck's schedule: the blocks are
    0-based cells of its reader's :meth:`~CaseStateReader.state` order.
    """

    def well_layout(
        self, wells: Sequence[tuple[str, str, Sequence[str]]], dates: Sequence[str]
    ) -> list["WellSchedule"]:
        """Return each well's blocks and status at the ISO report ``dates``.

        ``wells`` are ``(name, kind, outputs)``.

        Raises
        ------
        ConfigError
            If a well is not in the input, its blocks cannot be placed on the mesh, or its
            position depends on a template's values (it would differ by member).
        """


@runtime_checkable
class StateWriter(Protocol):
    """A simulator that states, before any run, what its runs write for its reader.

    A consumer asks for states (names at dates), and the simulator checks that
    its input writes them, refusing a configuration whose input cannot, with what to add.
    The services ask it in place of reading a deck's template (an Eclipse-format deck's
    grid and restart schedule); a simulator without it is read as a deck.
    """

    def state_output(
        self, names: Sequence[str], times: Sequence[StateAt], *, region: str | None = None
    ) -> StateOutput:
        """Return what the runs write for the states ``names`` at ``times``.

        ``times`` are all dates or all seconds since the run's start
        (:func:`~apex.domain.case_state.state_window` gives the seconds each asks for).
        ``region`` names one region of a multi-region case (``[case] field_region``); the
        services pass it only when it is given, so a writer of single-region cases may
        leave the keyword out, and one that takes it refuses a region its case lacks.

        Raises
        ------
        ConfigError
            If the simulator's input cannot write them (every member would fail the same
            way), naming what is missing and what to add.
        """


#: A reader plugin: a class whose constructor opens one case (the run's path, without a
#: suffix for a deck run). The class form is required: the services read what a reader
#: gives beyond the state from the class's ``capabilities`` (a ``ClassVar[frozenset[str]]``,
#: e.g. ``{"corner_geometry"}``) before any case exists; a function, which has none, is
#: taken as a reader of the state alone.
CaseStateReaderFactory = Callable[[Path], CaseStateReader]


def check_region(case: Path, regions: Sequence[str], region: str | None) -> None:
    """Refuse a region the case does not have, or none on a case of several.

    Raises
    ------
    ConfigError
        Naming the case's regions: every member would fail the same way.
    """
    if not regions:
        if region is not None:
            raise ConfigError(f"{case.name} has one region, not {region!r}: name none")
        return
    if region is None:
        raise ConfigError(f"{case.name} has regions {', '.join(regions)}: name one")
    if region not in regions:
        raise ConfigError(f"{case.name} has no region {region!r} (it has {', '.join(regions)})")
