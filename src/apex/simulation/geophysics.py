"""The geophysics simulator and its reading of a deck run's files.

:class:`GeophysicsSimulator` wraps a deck simulator (``opm_flow``, ``eclipse``), as
:class:`~apex.simulation.model_error.ModelErrorSimulator` does, and is itself a
:class:`~apex.core.simulation.CaseFileSimulator`: its ``read`` gives the summary keys to
the deck simulator and the field keys to a :class:`~apex.core.simulation.CaseFieldModel`
(the geophysical forward model of :mod:`apex.workflows.geophysics`), which reads the run's files
through :class:`CaseState`.

:class:`CaseState` reads a case's grid (EGRID), INIT and restarts in SI units, through a
case-state reader (the files are read by
:class:`~apex.simulation.eclipse_files.EclipseFiles`): restarts are found by their own date
(INTEHEAD), not by step index, so a deck's report steps never need counting. Unified
(``.UNRST``) and separate (``.Xnnnn``) restarts are both read.

Ported from SimulatorWrap
-------------------------
Module     : src/subsurface/multphaseflow/flow_rock.py (``flow_rock.calc_pem``, L364-491:
             the flow state at a vintage; ``flow_avo.call_sim``, L1002-1019: run, then
             read); and PET's src/misc/ecl.py (``EclipseCase``) for the restart reading
Commit     : d29dc57 (SimulatorWrap, NORCE, GPL-3.0); PET e65fb10 (GPL-3.0)
Differences: ``pet_parity.md`` ("Not reproduced, the simulator"): unified or separate
             restarts, units and phases from INTEHEAD, saturations clipped only within
             round-off, NTG not read.

Shapes: grid arrays are ``(n_x, n_y, n_z)`` in Eclipse (F) order, i fastest; cell arrays
over the active cells ``(n_active,)``, in the simulator's active order.
"""

import datetime as dt
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from apex.core.case_state import CaseStateReader, CornerGeometryReader
from apex.core.errors import ConfigError, SimulationError
from apex.core.simulation import (
    CaseFieldModel,
    CaseFileSimulator,
    MemberInput,
    Simulator,
    StatefulFieldModel,
)
from apex.domain.case_state import CornerGrid, StateTime, UnitSystem
from apex.domain.observations import DataKey
from apex.simulation.eclipse_files import (
    UNIT_SYSTEMS,
    EclipseFiles,
    Grid,
    column_centres,
    grid_unit_stated,
)

__all__ = [
    "UNIT_SYSTEMS",
    "CaseState",
    "GeophysicsSimulator",
    "Grid",
    "UnitSystem",
    "column_centres",
    "grid_unit_stated",
]

logger = logging.getLogger(__name__)

_SATURATION = {"oil": "SOIL", "water": "SWAT", "gas": "SGAS"}
_DENSITY = {"oil": "OIL_DEN", "water": "WAT_DEN", "gas": "GAS_DEN"}


class CaseState:
    """Geophysics' view of one simulated case: its grid, INIT and restarts by date.

    Reads through a :class:`~apex.core.case_state.CornerGeometryReader`: the
    Eclipse-format reader (:class:`~apex.simulation.eclipse_files.EclipseFiles`) for a
    case's path, or the reader the services chose for the simulator. Its conversions to SI
    and its rules (the later of two restarts on one day, a missing restart told apart as
    the member's failure or the deck's) are geophysics' own.

    Parameters
    ----------
    case
        The run's path without a suffix (``str`` or path), or an opened reader.

    Raises
    ------
    SimulationError
        If the grid or INIT cannot be read (the run did not finish as it should).
    ConfigError
        If ``case`` is neither, or a reader lacks the corner-point capability geophysics
        needs.
    """

    def __init__(self, case: "str | os.PathLike[str] | CaseStateReader") -> None:
        if isinstance(case, str | os.PathLike):
            reader: CaseStateReader = EclipseFiles(Path(case))
        elif isinstance(case, CaseStateReader):
            reader = case
        else:
            raise ConfigError(
                f"CaseState takes a case's path or a case-state reader, not a {type(case).__name__}"
            )
        if not isinstance(reader, CornerGeometryReader):
            raise ConfigError(
                f"{reader.case.name}: geophysics needs a corner-point grid with a deck's "
                f"unit system and phases, which the {type(reader).__name__} reader does not "
                "give"
            )
        self.reader = reader
        self.case = reader.case
        self.units = reader.unit_system
        self.phases = reader.phases

    @cached_property
    def init(self) -> frozenset[str]:
        """The names of the INIT's arrays (``"EQLNUM" in state.init``)."""
        return self.reader.static_names()

    @cached_property
    def grid(self) -> CornerGrid:
        """The case's grid (EGRID)."""
        return self.reader.corner_geometry()

    def static(self, name: str) -> NDArray[np.float64]:
        """Return an INIT array over the active cells, as written (deck units)."""
        return self.reader.static(name)

    @cached_property
    def depth(self) -> NDArray[np.float64]:
        """Each active cell's centre depth (m), INIT's DEPTH."""
        return self.static("DEPTH") * self.units.length

    @cached_property
    def run_end(self) -> dt.date | None:
        """The date the run's summary ends at, or ``None`` if it has no readable summary."""
        reached = self.reader.reached()
        return None if reached is None else reached.date

    @cached_property
    def _restarts(self) -> dict[dt.date, StateTime]:
        """Each restart's date and time.

        Two restarts on one date (two report steps on a day) give the later one, as the
        summary's values are its report steps' last (``deck.report_values``).
        """
        found: dict[dt.date, StateTime] = {}
        for time in self.reader.times():
            if time.date is not None:
                found[time.date] = time
        return found

    @property
    def restart_dates(self) -> list[dt.date]:
        """The dates the case wrote restarts at, in order."""
        return sorted(self._restarts)

    def restart(self, date: dt.date, names: Sequence[str]) -> dict[str, NDArray[np.float64]]:
        """Return the restart arrays ``names`` at ``date`` (deck units), those it has.

        Raises
        ------
        ConfigError
            If the case wrote no restart at ``date`` although its summary reaches it: the
            deck's RPTRST does not ask for one, so every member would fail the same way.
        SimulationError
            If the run's summary ends before ``date`` (this member's run ended early), or
            has no summary to tell (this member's files are broken).
        """
        if date not in self._restarts:
            dates = self.restart_dates
            written = ", ".join(d.isoformat() for d in dates) or "none"
            message = f"{self.case.name} has no restart at {date.isoformat()} (written: {written})"
            # By the summary's end, not the last restart: a deck that writes restarts
            # yearly ends its restarts before its last report time in every member.
            end = self.run_end
            if end is None:
                raise SimulationError(
                    f"{message}, and no summary tells how far it ran (if every member fails "
                    "so, check the deck's RPTRST)"
                )
            if end < date:
                raise SimulationError(f"{message}; the run ended on {end.isoformat()}")
            raise ConfigError(
                f"{message}; the deck's RPTRST must write one at every vintage, e.g. RPTRST "
                "BASIC=2 (every report step) with a report step at each vintage's date"
            )
        return self.reader.state(self._restarts[date], names, skip_missing=True)

    def pore_volume(self, date: dt.date) -> NDArray[np.float64]:
        """Return each active cell's pore volume at reservoir conditions (m³), RPORV.

        Raises
        ------
        ConfigError
            If the restart lacks RPORV: the deck's RPTRST must ask for it, so every
            member would.
        """
        arrays = self.restart(date, ["RPORV"])
        if "RPORV" not in arrays:
            raise ConfigError(
                f"{self.case.name} at {date.isoformat()}: the restart has no RPORV; add it to "
                "the deck's RPTRST (e.g. RPTRST BASIC=2 RPORV /)"
            )
        out: NDArray[np.float64] = arrays["RPORV"] * self.units.volume
        return out

    def flow_state(
        self, date: dt.date, densities: bool
    ) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """Return the pressure (Pa), saturations and densities (kg/m³) at ``date``.

        Saturations and densities are keyed by phase (``"oil"``, ``"water"``, ``"gas"``),
        over the deck's phases; the one phase whose saturation the restart lacks (the
        simulators do not write oil's) is 1 minus the others. A saturation within
        :data:`SATURATION_SLACK` outside [0, 1] is clipped to it (the solver's overshoot or
        round-off; the reference clips every one), and then a sum within it of 1 is
        normalized.
        Densities are empty unless ``densities``.

        Raises
        ------
        ConfigError
            If the restart lacks a density asked for: the deck's RPTRST must ask for them
            (``DEN``), so every member would.
        SimulationError
            If it lacks the pressure or two saturations, which every restart holds: this
            member's files are broken.
        """
        names = ["PRESSURE", *(_SATURATION[p] for p in self.phases)]
        if densities:
            names += [_DENSITY[p] for p in self.phases]
        arrays = self.restart(date, names)
        when = f"{self.case.name} at {date.isoformat()}"
        if "PRESSURE" not in arrays:
            raise SimulationError(f"{when}: the restart has no PRESSURE")
        written = [p for p in self.phases if _SATURATION[p] in arrays]
        missing = [p for p in self.phases if p not in written]
        if len(missing) > 1:
            raise SimulationError(
                f"{when}: the restart has no saturation of {', '.join(missing)}; at most one "
                "phase's may be left out (it is 1 minus the others)"
            )
        raw = {p: arrays[_SATURATION[p]] for p in written}
        for p in missing:
            raw[p] = 1.0 - sum(raw.values(), np.zeros_like(arrays["PRESSURE"]))
        saturations = {p: _settle(s) for p, s in raw.items()}
        # Clipping one phase (e.g. a derived oil saturation of -2e-4) moves the sum off 1 by
        # as much; within the slack it is normalized, as the solver's round-off.
        total = sum(saturations.values(), np.zeros_like(arrays["PRESSURE"]))
        close = (np.abs(total - 1.0) <= SATURATION_SLACK) & (total > 0)
        saturations = {
            p: np.divide(s, total, out=s.copy(), where=close) for p, s in saturations.items()
        }
        moved = max(float(np.max(np.abs(saturations[p] - raw[p]))) for p in raw)
        if moved > 0:
            changed = int(sum((saturations[p] != raw[p]) for p in raw).astype(bool).sum())
            (logger.warning if moved > _SATURATION_DRIFT else logger.debug)(
                "geophysics_saturations_settled",
                extra={"case": self.case.name, "date": date.isoformat(), "cells": changed,
                       "largest_change": moved},
            )  # fmt: skip
        # In the deck's phase order (oil, water, gas), whichever was derived.
        saturations = {p: saturations[p] for p in self.phases}
        density: dict[str, NDArray[np.float64]] = {}
        if densities:
            lacking = [_DENSITY[p] for p in self.phases if _DENSITY[p] not in arrays]
            if lacking:
                raise ConfigError(
                    f"{when}: the restart has no {', '.join(lacking)}; the data need the "
                    "phases' densities (a petro-elastic model's, gravity), so add DEN to "
                    "the deck's RPTRST (e.g. RPTRST BASIC=2 DEN /)"
                )
            density = {p: arrays[_DENSITY[p]] * self.units.density for p in self.phases}
        return arrays["PRESSURE"] * self.units.pressure, saturations, density


#: How far outside [0, 1] a restart's saturation may be and still be the solver's overshoot
#: (OPM Flow wrote -4.8e-5 in a 3dBox member, and GIES's members on 3dBox an oil saturation
#: up to 1.009, in one cell each, which lost 9 of 100 members at 1e-3; with porosity
#: uncertain too, 1.017 and 1.021: 3e-2, the user's choice of 2026-10-20): such values are
#: clipped, as the reference clips every one, with a warning above the drift below. Beyond
#: it they are left for the petro-elastic model to refuse, as a fault of the member's run.
SATURATION_SLACK = 3e-2
#: A settled saturation that moved more than this is logged as a warning (a systematic
#: drift, e.g. a units or phase mix-up, should not hide in the slack); less, at debug.
_SATURATION_DRIFT = 1e-4


def _settle(saturation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clip the saturations within :data:`SATURATION_SLACK` of [0, 1] to it, ``(n_active,)``."""
    near = (saturation >= -SATURATION_SLACK) & (saturation <= 1.0 + SATURATION_SLACK)
    out: NDArray[np.float64] = np.where(near, np.clip(saturation, 0.0, 1.0), saturation)
    return out


class GeophysicsSimulator:
    """A deck simulator whose runs also give geophysical field data.

    ``read`` splits the keys: those ``model`` owns are computed from the run's files, the
    others are the deck simulator's (the summary). Everything about where and how the run
    happens is the deck simulator's: its runner, resources, program and pruning.
    """

    needs_work_dir = True

    def __init__(self, inner: Simulator, model: CaseFieldModel) -> None:
        if getattr(model, "state_variables", ()) and not isinstance(model, StatefulFieldModel):
            raise ValueError(
                f"{type(model).__name__} takes state variables but cannot give a member's "
                "model (for_state): the deck would see them"
            )
        if not isinstance(inner, CaseFileSimulator):
            raise ValueError(
                f"geophysics needs a simulator whose runs leave files to read "
                f"(opm_flow, eclipse); {type(inner).__name__} has none"
            )
        self.inner: CaseFileSimulator = inner
        self.model = model
        # The model each case's member set (its uncertain petro-elastic parameters), until
        # the case is cleaned: keyed by case, as threads may run members at once.
        self._models: dict[Path, CaseFieldModel] = {}

    def state_names_read(self) -> frozenset[str] | None:
        """Return the wrapped simulator's (its members' state reaches it unchanged)."""
        read = getattr(self.inner, "state_names_read", None)
        return read() if callable(read) else None

    @property
    def resources(self) -> Mapping[str, int]:
        """The inner simulator's machine-wide tokens."""
        return dict(getattr(self.inner, "resources", {}))

    @property
    def max_parallel_runs(self) -> int | None:
        """The inner simulator's cap on parallel runs, if any."""
        return getattr(self.inner, "max_parallel_runs", None)

    @property
    def accepts_runner(self) -> bool:
        """Whether the inner simulator can run its program elsewhere."""
        return bool(getattr(self.inner, "accepts_runner", False))

    @property
    def executable(self) -> str | None:
        """The inner simulator's program, which an executor checks on its hosts."""
        return getattr(self.inner, "executable", None)

    @property
    def version_command(self) -> Callable[[str], str] | None:
        """The inner simulator's shell command for its version on a host, if it has one.

        An executor probing its hosts runs it in place of ``<program> --version`` (ADR
        0060; PFLOTRAN has no ``--version``).
        """
        found = getattr(self.inner, "version_command", None)
        return found if callable(found) else None

    @property
    def required_programs(self) -> tuple[str, ...]:
        """The inner simulator's other programs (an MPI launcher), which hosts must have."""
        return tuple(getattr(self.inner, "required_programs", ()))

    def version(self) -> str | None:
        """Return the inner simulator's version, for provenance."""
        version = getattr(self.inner, "version", None)
        return version() if callable(version) else None

    def simulate(self, member: MemberInput) -> Path:
        """Run the deck simulator; its files stay until :meth:`clean`.

        A model with uncertain parameters (``state_variables``, ``for_state``) takes
        them from the member's state, which the deck never sees.
        """
        if not isinstance(self.model, StatefulFieldModel) or not self.model.state_variables:
            return self.inner.simulate(member)
        names = set(self.model.state_variables)
        model = self.model.for_state(member.state)
        # replace(), not a new MemberInput: the member keeps where it runs.
        physical = replace(member, state={k: v for k, v in member.state.items() if k not in names})
        case = self.inner.simulate(physical)
        self._models[case] = model
        return case

    def read(self, case: Path, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Return the values at ``keys``: field data from the model, the rest the deck's."""
        model = self._models.get(case, self.model)
        owned = [model.owns(k) for k in keys]
        values = np.empty(len(keys))
        mine = [i for i, o in enumerate(owned) if o]
        theirs = [i for i, o in enumerate(owned) if not o]
        if theirs:
            values[theirs] = self.inner.read(case, [keys[i] for i in theirs])
        if mine:
            values[mine] = model.forecast(case, [keys[i] for i in mine])
        return values

    def clean(self, case: Path) -> None:
        """Prune the case's files as the deck simulator's options say."""
        self._models.pop(case, None)
        self.inner.clean(case)

    def release(self, case: Path) -> None:
        """Forget the case's member model after a failed reading; its files stay."""
        self._models.pop(case, None)
        release = getattr(self.inner, "release", None)
        if callable(release):
            release(case)

    def run(self, member: MemberInput, keys: Sequence[DataKey]) -> NDArray[np.float64]:
        """Simulate, read and clean, as the deck simulators' own ``run`` does."""
        case = self.simulate(member)
        try:
            values = self.read(case, keys)
        except BaseException:
            self.release(case)  # a failed member's files stay, for diagnosis
            raise
        self.clean(case)
        return values
